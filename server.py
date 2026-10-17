"""Paper Edit -- local API + UI. Runs on the desktop, used from her laptop.

    .venv/Scripts/python.exe server.py      then open http://<desktop>:8100

Port 8100 keeps out of the way of the ports local AI tooling tends to claim
(8080, 11434, 11435).
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from paperedit import auth, clips, store
from paperedit.audio import detect_silences, silence_gaps, snap_points
from paperedit.edl import Word, derive_cuts, is_filler
from paperedit.captions import PRESETS as CAPTION_PRESETS, words_on_output_timeline, write_ass
from paperedit.enhance import PRESETS as SOUND_PRESETS, build_chain, recommend_preset
from paperedit.render import escape_filter_path
from paperedit.render import make_proxy, media_info, render

PORT = 8100
app = FastAPI(title="Paper Edit")

# Export presets: each platform gets the shape it actually wants, so she is not
# guessing at resolutions after the edit is done.
PRESETS = {
    "source":  {"label": "Original quality",     "vf": "", "video": True},
    "youtube": {"label": "YouTube 1080p 16:9",
                "vf": "scale=-2:1080,fps=30", "video": True},
    "reels":   {"label": "Reels / Shorts 9:16",
                "vf": "scale=1080:1920:force_original_aspect_ratio=increase,"
                      "crop=1080:1920,fps=30", "video": True},
    "square":  {"label": "Feed 1:1",
                "vf": "scale=1080:1080:force_original_aspect_ratio=increase,"
                      "crop=1080:1080,fps=30", "video": True},
    "audio":   {"label": "Audio only (podcast)", "vf": "", "video": False},
}


def _project_or_404(pid: str) -> dict:
    p = store.get_project(pid)
    if not p:
        raise HTTPException(404, "no such project")
    return p


def _words(pid: str, raw: list[dict] | None = None) -> list[Word]:
    return [Word(w["text"], w["start"], w["end"], speaker=w["speaker"],
                 deleted=bool(w["deleted"]), probability=w["probability"])
            for w in (store.get_words(pid) if raw is None else raw)]


def _clip_words(pid: str, first_idx: int, last_idx: int) -> list[dict]:
    """The project's words with everything outside one clip deleted.

    A short is an export of a clip, so it is built the same way as every other
    export -- from the words that survive -- rather than by trimming a render.
    Deletions the user made inside the clip still apply, and the captions,
    silence removal and Studio Sound all come along unchanged. Nothing is
    written back: the project's own edit is untouched.
    """
    out = []
    for w in store.get_words(pid):
        w = dict(w)
        if not first_idx <= int(w["idx"]) <= last_idx:
            w["deleted"] = 1
        out.append(w)
    return out


def _silences(pid: str, src: str) -> list[tuple[float, float]]:
    """Detected silences for a project, cached -- it is a full pass over the
    audio and the answer never changes for a given source file."""
    cache = store.project_dir(pid) / "silences.json"
    if cache.exists():
        return [tuple(x) for x in json.loads(cache.read_text())]
    found = detect_silences(src, min_dur=0.20)      # threshold measured per file
    cache.write_text(json.dumps(found))
    return found


def _plan(pid: str, raw: list[dict] | None = None):
    """Cuts derived from the surviving words, snapped to silence, with dead air
    removed if the project asks for it. `raw` replaces the stored words (a
    clip's, from _clip_words)."""
    p = _project_or_404(pid)
    src = p["source"]
    sil: list[tuple[float, float]] = []
    if src and Path(src).exists():
        sil = _silences(pid, src)
    removed = []
    if p["silence_on"]:
        removed = silence_gaps(sil, keep=p["silence_keep"], min_remove=p["silence_min"])
    return derive_cuts(_words(pid, raw), snap_points=snap_points(sil), snap_tolerance=0.15,
                       removed_ranges=removed, duration=p["duration"] or None,
                       fps=p["fps"] or None)


def _sound_chain(p: dict) -> tuple[str, str]:
    """(preset_used, filter_chain) for a project. 'auto' asks the recording."""
    choice = p["sound_preset"] or "auto"
    if choice == "auto":
        try:
            choice = recommend_preset(p["source"])[0]
        except Exception:
            choice = "normalize"
    if choice not in SOUND_PRESETS:
        choice = "off"
    return choice, build_chain(choice)


def _output_size(p: dict, preset: str) -> tuple[int, int]:
    """The frame size this export will really have.

    Captions are authored for these dimensions, not the source's: an .ass
    written for a 1920x1080 frame and burned into a 1080x1920 one gets
    stretched by libass.
    """
    w, h = p["width"] or 1920, p["height"] or 1080
    if preset == "reels":
        return 1080, 1920
    if preset == "square":
        return 1080, 1080
    if preset == "youtube":                     # height fixed, width follows
        return max(2, int(round(w / h * 1080)) // 2 * 2), 1080
    return w, h


def _caption_filter(pid: str, p: dict, plan, width: int, height: int, *,
                    raw: list[dict] | None = None, style: str | None = None,
                    name: str = "captions.ass") -> str:
    """Write the .ass for this edit and return the filter that burns it in.

    Regenerated per export, deliberately: the caption timings depend on the cut
    list, so a stale file would drift out of sync the moment anything is edited.
    `raw`/`style`/`name` let a short use its clip's words, its own style and its
    own file, so two exports running at once never share a caption file.
    """
    style = style or p["caption_style"] or "off"
    if style == "off" or style not in CAPTION_PRESETS or not p["has_video"]:
        return ""
    words = words_on_output_timeline(store.get_words(pid) if raw is None else raw, plan)
    if not words:
        return ""
    path = store.project_dir(pid) / name
    write_ass(words, path, width=width, height=height, style=style)
    # A BARE filename, with ffmpeg run from the project directory. Escaping the
    # drive colon inside a filter argument does not survive ffmpeg's option
    # splitting on Windows; removing the colon entirely does.
    return "subtitles=" + path.name


def _ingest(pid: str, jid: str) -> None:
    """Probe -> proxy -> transcribe. The one long job in the app."""
    try:
        p = store.get_project(pid)
        src = p["source"]
        store.update_job(jid, status="running", progress=0.02, message="reading media")
        info = media_info(src)
        store.update_project(pid, duration=info["duration"], fps=info["fps"],
                             width=info["width"], height=info["height"],
                             has_video=int(info["has_video"]), status="ingesting")

        if info["has_video"]:
            store.update_job(jid, progress=0.06, message="building preview copy")
            proxy = str(store.project_dir(pid) / "proxy.mp4")
            make_proxy(src, proxy)
            store.update_project(pid, proxy=proxy)

        def prog(frac: float, msg: str) -> None:
            store.update_job(jid, progress=0.1 + 0.85 * frac,
                             message="transcribing " + msg)

        from paperedit.transcribe import transcribe
        words = transcribe(src, progress=prog)
        store.set_words(pid, words)

        try:
            preset, snr, _ = recommend_preset(src)
            store.update_project(pid, sound_snr=round(snr, 2))
        except Exception:
            pass                       # a measurement failure must not fail ingest
        store.update_project(pid, status="ready")
        store.update_job(jid, status="done", progress=1.0,
                         message=str(len(words)) + " words")
    except Exception as e:
        store.update_project(pid, status="failed")
        store.update_job(jid, status="failed",
                         message=(type(e).__name__ + ": " + str(e))[:500])


def _export(pid: str, jid: str, preset: str, clip: tuple[int, int] | None = None) -> None:
    try:
        store.update_job(jid, status="running", progress=0.05, message="planning cuts")
        p = store.get_project(pid)
        raw = _clip_words(pid, *clip) if clip else None
        plan = _plan(pid, raw)
        if not plan.cuts:
            raise ValueError("everything is deleted -- nothing left to export"
                             + (" in that clip" if clip else ""))
        ext = "m4a" if preset == "audio" else "mp4"
        stamp = int(time.time())
        out = store.project_dir(pid) / (f"short-{clip[0]}-{clip[1]}-{preset}-{stamp}.{ext}" if clip
                                        else f"export-{preset}-{stamp}.{ext}")
        store.update_job(jid, progress=0.15,
                         message=f"rendering {len(plan.cuts)} cuts, {plan.duration/60:.1f} min")
        spec = PRESETS[preset]
        caption = ""
        if spec["video"] and clip:
            # A short is watched with the sound off more often than not, so it
            # always gets captions: the project's style, or "highlight" if the
            # full-length edit has them switched off.
            style = p["caption_style"] if p["caption_style"] in CAPTION_PRESETS else "highlight"
            caption = _caption_filter(pid, p, plan, *_output_size(p, preset), raw=raw,
                                      style=style, name=f"captions-{jid}.ass")
        elif spec["video"]:
            caption = _caption_filter(pid, p, plan, *_output_size(p, preset))
        # Reframe first, then burn the captions on the reframed picture --
        # the other order would crop the text off the sides of a 9:16 export.
        vfilter = ",".join(x for x in (spec["vf"], caption) if x)
        used, chain = _sound_chain(p)
        if chain:
            store.update_job(jid, message=f"rendering with Studio Sound ({used})")
        render(p["source"], plan, out, video=spec["video"],
               audio_filters=chain, video_filters=vfilter,
               cwd=store.project_dir(pid) if caption else None)
        # Report the real duration, not the arithmetic one: ffmpeg's video trim
        # includes the partial frame at each boundary, so the file runs ~0.15%
        # long. Measured in spikes/drift.py -- audio and video stay in sync.
        actual = media_info(out)["duration"]
        store.update_job(jid, status="done", progress=1.0, result=out.name,
                         message=(f"{actual:.0f} s short" if clip
                                  else f"{actual/60:.1f} min from {len(plan.cuts)} cuts"))
    except Exception as e:
        store.update_job(jid, status="failed",
                         message=(type(e).__name__ + ": " + str(e))[:500])
    finally:
        if clip:
            (store.project_dir(pid) / f"captions-{jid}.ass").unlink(missing_ok=True)


def _find_clips(pid: str, jid: str, n: int) -> None:
    """Ask the local LLM for short-worthy moments. Tens of seconds on a 12B
    model, so it is a job like export rather than a request that hangs."""
    try:
        store.update_job(jid, status="running", progress=0.05, message="reading the transcript")

        def prog(frac: float, msg: str) -> None:
            store.update_job(jid, progress=0.1 + 0.8 * frac, message=msg)

        out = clips.find_clips(store.get_words(pid), n=n, progress=prog)
        found = len(out["clips"])
        msg = f"{found} clip{'s' if found != 1 else ''} found"
        if out.get("note"):
            msg += " -- " + out["note"]
        store.update_job(jid, status="done", progress=1.0, message=msg,
                         result=json.dumps(out))
    except clips.ClipsError as e:
        store.update_job(jid, status="failed", message=str(e)[:500])
    except Exception as e:
        store.update_job(jid, status="failed",
                         message=(type(e).__name__ + ": " + str(e))[:500])


# Paths reachable without signing in: the sign-in page itself and the calls it
# needs. Everything else -- the editor, every API route, the proxy and the
# finished exports -- is behind the cookie.
PUBLIC_PATHS = {"/login", "/login.html", "/api/session", "/favicon.ico"}


@app.middleware("http")
async def require_sign_in(request: Request, call_next):
    path = request.url.path
    if path in PUBLIC_PATHS or auth.see_session(request.cookies.get(auth.COOKIE)):
        return await call_next(request)
    # An API caller gets a status it can act on; a person gets the sign-in page.
    if path.startswith("/api/"):
        return JSONResponse({"error": "not signed in"}, status_code=401)
    return RedirectResponse("/login", status_code=303)


@app.get("/login")
def login_page():
    return FileResponse(STATIC / "login.html")


@app.get("/api/session")
def api_session(request: Request):
    """What the sign-in page needs to know before it draws itself: whether a
    password exists yet, and whether this browser is already signed in."""
    return {"signed_in": auth.valid_session(request.cookies.get(auth.COOKIE)),
            "needs_setup": not auth.has_password()}


@app.post("/api/session")
def api_sign_in(request: Request, body: dict = Body(...)):
    """Sign in -- or, on a machine where no password has been set yet, set one.

    First-run setup is deliberately open to whoever reaches it first on the home
    network. The alternative is a password only one of them knows how to
    create, and an editor the other person simply cannot get into.
    """
    who = request.client.host if request.client else "?"
    wait = auth.locked_out(who)
    if wait:
        raise HTTPException(429, f"too many tries -- wait {int(wait) + 1}s")

    password = str(body.get("password", ""))
    if not auth.has_password():
        if len(password) < 4:
            raise HTTPException(400, "pick a password of at least 4 characters")
        auth.set_password(password)
    elif not auth.check_password(password):
        auth.note_failure(who)
        raise HTTPException(401, "that is not the password")

    auth.clear_failures(who)
    # Label the session by what signed in, so a list of devices is something a
    # person can actually read -- and so an automated check is visibly a script
    # rather than looking like somebody using the editor.
    token = auth.new_session(auth.device_label(request.headers.get("user-agent")), who)
    r = JSONResponse({"ok": True})
    # Not Secure: this is plain http on a home LAN, and a Secure cookie would
    # simply never be sent. Lax keeps it off cross-site requests.
    r.set_cookie(auth.COOKIE, token, max_age=auth.SESSION_DAYS * 86400,
                 httponly=True, samesite="lax", path="/")
    return r


@app.delete("/api/session")
def api_sign_out(request: Request):
    auth.end_session(request.cookies.get(auth.COOKIE))
    r = JSONResponse({"ok": True})
    r.delete_cookie(auth.COOKIE, path="/")
    return r


@app.get("/api/health")
def health():
    return {"ok": True, "presets": {k: v["label"] for k, v in PRESETS.items()}}


@app.get("/api/projects")
def api_list():
    out = []
    for p in store.list_projects():
        p["job"] = store.latest_job(p["id"])
        out.append(p)
    return out


@app.post("/api/projects")
def api_create(body: dict = Body(...)):
    name = (body.get("name") or "Untitled").strip()[:120]
    return {"id": store.create_project(name)}


@app.delete("/api/projects/{pid}")
def api_delete(pid: str):
    _project_or_404(pid)
    shutil.rmtree(store.project_dir(pid), ignore_errors=True)
    store.delete_project(pid)
    return {"ok": True}


@app.get("/api/projects/{pid}")
def api_get(pid: str):
    p = _project_or_404(pid)
    p["job"] = store.latest_job(pid)
    p["word_count"] = len(store.get_words(pid))
    return p


@app.get("/api/projects/{pid}/upload")
def api_upload_status(pid: str, name: str = "source.bin"):
    """Bytes already received -- lets the browser resume a failed upload."""
    _project_or_404(pid)
    dst = store.project_dir(pid) / Path(name).name
    return {"received": dst.stat().st_size if dst.exists() else 0}


@app.put("/api/projects/{pid}/upload")
async def api_upload(pid: str, request: Request, name: str = "source.bin",
                     offset: int = 0, final: int = 0):
    """Append one chunk at `offset`. Resumable, because a dropped Wi-Fi
    connection 90% through a multi-GB file must not mean starting over."""
    _project_or_404(pid)
    dst = store.project_dir(pid) / Path(name).name
    have = dst.stat().st_size if dst.exists() else 0
    if offset > have:
        raise HTTPException(409, f"gap: server has {have} bytes, chunk starts at {offset}")
    with open(dst, "r+b" if dst.exists() else "wb") as fh:
        fh.seek(offset)
        async for chunk in request.stream():
            fh.write(chunk)
        size = fh.tell()
    if final:
        store.update_project(pid, source=str(dst), status="uploaded")
    return {"received": size}


@app.post("/api/projects/{pid}/ingest")
def api_ingest(pid: str):
    p = _project_or_404(pid)
    if not p["source"] or not Path(p["source"]).exists():
        raise HTTPException(400, "upload a file first")
    jid = store.create_job(pid, "ingest")
    store.run_in_background(_ingest, pid, jid)
    return {"job": jid}


@app.get("/api/projects/{pid}/words")
def api_words(pid: str):
    _project_or_404(pid)
    return store.get_words(pid)


@app.post("/api/projects/{pid}/words")
def api_set_deleted(pid: str, body: dict = Body(...)):
    """Mark words deleted or restored. Returns the new edit length so the UI can
    show the running duration without re-fetching the whole transcript."""
    _project_or_404(pid)
    n = store.set_deleted(pid, body.get("indices", []), bool(body.get("deleted", True)))
    plan = _plan(pid)
    return {"updated": n, "duration": round(plan.duration, 3), "cuts": len(plan.cuts)}


@app.post("/api/projects/{pid}/silence")
def api_silence(pid: str, body: dict = Body(...)):
    """Turn dead-air removal on or off, and tune how much of each pause to keep.

    Non-destructive like everything else: it only changes how the cut list is
    derived, so switching it off puts every pause straight back.
    """
    p = _project_or_404(pid)
    on = bool(body.get("enabled", True))
    keep = max(0.0, min(2.0, float(body.get("keep", p["silence_keep"]))))
    min_remove = max(0.15, min(5.0, float(body.get("min_remove", p["silence_min"]))))
    store.update_project(pid, silence_on=int(on), silence_keep=keep,
                         silence_min=min_remove)

    src = p["source"]
    sil = _silences(pid, src) if src and Path(src).exists() else []
    gaps = silence_gaps(sil, keep=keep, min_remove=min_remove)
    plan = _plan(pid)
    return {"enabled": on, "keep": keep, "min_remove": min_remove,
            "pauses_found": len(gaps),
            "seconds_removable": round(sum(e - s for s, e in gaps), 2),
            "duration": round(plan.duration, 3), "cuts": len(plan.cuts)}


@app.post("/api/projects/{pid}/fillers")
def api_fillers(pid: str, body: dict = Body(default={})):
    """Delete -- or put back -- the filler words the transcript spells out.

    Deliberately the cheap half of the job. Whisper transcribes only 7-17% of
    the fillers a person actually hears (FINDINGS.md), so this catches the ones
    it wrote down and dead-air removal does the heavier lifting on the rest.

    Returns the indices it touched, so the browser can undo it exactly like a
    hand-made edit rather than treating it as a special case.
    """
    _project_or_404(pid)
    deleted = bool(body.get("deleted", True))
    idx = [w["idx"] for w in store.get_words(pid) if is_filler(w["text"])]
    store.set_deleted(pid, idx, deleted)
    plan = _plan(pid)
    return {"marked": len(idx), "indices": idx, "deleted": deleted,
            "duration": round(plan.duration, 3), "cuts": len(plan.cuts)}


@app.post("/api/projects/{pid}/sound")
def api_sound(pid: str, body: dict = Body(...)):
    """Choose the Studio Sound preset, or leave it on 'auto' to be measured.

    Applied at export only -- it never touches the source file.
    """
    p = _project_or_404(pid)
    choice = str(body.get("preset", "auto"))
    if choice != "auto" and choice not in SOUND_PRESETS:
        raise HTTPException(400, "unknown sound preset " + choice)
    store.update_project(pid, sound_preset=choice)
    p = _project_or_404(pid)
    used, chain = _sound_chain(p)
    why = ""
    if choice == "auto" and p["source"]:
        try:
            _, _, why = recommend_preset(p["source"])
        except Exception:
            why = ""
    snr = p["sound_snr"]
    if not snr and p["source"]:
        try:                       # projects ingested before SNR was measured
            from paperedit.enhance import measure_snr
            snr = round(measure_snr(p["source"]), 2)
            store.update_project(pid, sound_snr=snr)
        except Exception:
            snr = 0
    return {"preset": choice, "using": used, "snr": snr,
            "reason": why, "chain": chain}


@app.post("/api/projects/{pid}/captions")
def api_captions(pid: str, body: dict = Body(...)):
    """Pick a caption style, or 'off'. Burned in at export."""
    _project_or_404(pid)
    style = str(body.get("style", "off"))
    if style != "off" and style not in CAPTION_PRESETS:
        raise HTTPException(400, "unknown caption style " + style)
    store.update_project(pid, caption_style=style)
    plan = _plan(pid)
    words = words_on_output_timeline(store.get_words(pid), plan)
    return {"style": style, "words": len(words),
            "styles": sorted(CAPTION_PRESETS)}


@app.get("/api/projects/{pid}/plan")
def api_plan(pid: str):
    """The cut list. The player uses this to seek over deleted ranges, which is
    what makes preview instant -- nothing is re-encoded until export."""
    plan = _plan(pid)
    return {"duration": round(plan.duration, 3),
            "cuts": [{"start": c.start, "end": c.end} for c in plan.cuts]}


@app.post("/api/projects/{pid}/export")
def api_export(pid: str, body: dict = Body(default={})):
    """Export the edit -- or, with "clip": {"first_idx", "last_idx"}, just that
    stretch of it as a short (9:16 unless a preset is named)."""
    p = _project_or_404(pid)
    clip = None
    if body.get("clip") is not None:
        c = body["clip"]
        try:
            clip = (int(c["first_idx"]), int(c["last_idx"]))
        except (TypeError, KeyError, ValueError):
            raise HTTPException(400, "clip needs first_idx and last_idx")
        idx = [int(w["idx"]) for w in store.get_words(pid)]
        if not idx or clip[0] > clip[1] or clip[0] < min(idx) or clip[1] > max(idx):
            raise HTTPException(400, "that clip is not in this transcript")
        if not p["has_video"]:
            raise HTTPException(400, "shorts need a video; this project is audio only")
    preset = body.get("preset", "reels" if clip else "source")
    if preset not in PRESETS:
        raise HTTPException(400, "unknown preset " + str(preset))
    if clip and not PRESETS[preset]["video"]:
        raise HTTPException(400, "a short has to be a video preset")
    jid = store.create_job(pid, "export")
    store.run_in_background(_export, pid, jid, preset, clip)
    return {"job": jid}


@app.post("/api/projects/{pid}/clips")
def api_find_clips(pid: str, body: dict = Body(default={})):
    """Start looking for clips. Poll /api/jobs/{job}, then GET this path."""
    p = _project_or_404(pid)
    if p["status"] != "ready":
        raise HTTPException(400, "the transcript is not ready yet")
    try:
        n = max(1, min(10, int(body.get("count", 3))))
    except (TypeError, ValueError):
        raise HTTPException(400, "count must be a number")
    jid = store.create_job(pid, "clips")
    store.run_in_background(_find_clips, pid, jid, n)
    return {"job": jid}


@app.get("/api/projects/{pid}/clips")
def api_clips(pid: str):
    """The latest clip suggestions, so they survive closing the project.

    They were found against the transcript as it was then; an edit made since
    does not move them.
    """
    _project_or_404(pid)
    j = store.latest_job(pid, "clips")
    out = {"job": j, "clips": [], "dropped": []}
    if j and j["status"] == "done" and j["result"]:
        found = json.loads(j["result"])
        out.update(clips=found.get("clips", []), dropped=found.get("dropped", []),
                   model=found.get("model"), note=found.get("note", ""))
    return out


@app.get("/api/jobs/{jid}")
def api_job(jid: str):
    j = store.get_job(jid)
    if not j:
        raise HTTPException(404, "no such job")
    return j


@app.get("/api/projects/{pid}/proxy")
def api_proxy(pid: str):
    """The 720p preview. FileResponse serves Range requests, so scrubbing works."""
    p = _project_or_404(pid)
    path = p["proxy"] or p["source"]
    if not path or not Path(path).exists():
        raise HTTPException(404, "no preview yet")
    return FileResponse(path)


@app.get("/api/projects/{pid}/file/{name}")
def api_file(pid: str, name: str):
    """Download a render. Opening this on her phone saves it to the camera roll,
    which is the fastest route to Instagram and Facebook."""
    _project_or_404(pid)
    f = store.project_dir(pid) / Path(name).name
    if not f.exists():
        raise HTTPException(404, "no such file")
    return FileResponse(f, filename=f.name, media_type="application/octet-stream")


@app.exception_handler(HTTPException)
def _http_err(_req, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


STATIC = ROOT / "static"
STATIC.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


def local_addresses() -> list[str]:
    """Addresses this machine can be reached on, so the console tells you where
    to point a laptop or phone instead of making you go and look it up."""
    import socket
    out = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in out:
                out.append(ip)
    except OSError:
        pass
    return out


if __name__ == "__main__":
    import uvicorn
    store.init()
    lines = ["", "  Paper Edit is running. Open it at:",
             f"    on this computer     http://localhost:{PORT}"]
    lines += [f"    on your network      http://{ip}:{PORT}" for ip in local_addresses()]
    # flush explicitly: stdout is block-buffered when redirected to a file, and
    # these addresses are the whole reason the window is worth looking at.
    for line in lines:
        print(line, flush=True)
    print(flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
