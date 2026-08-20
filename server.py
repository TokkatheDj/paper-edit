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
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from paperedit import store
from paperedit.audio import detect_silences, silence_gaps, snap_points
from paperedit.edl import Word, derive_cuts
from paperedit.render import make_proxy, media_info, render

PORT = 8100
app = FastAPI(title="Paper Edit")

# Export presets: each platform gets the shape it actually wants, so she is not
# guessing at resolutions after the edit is done.
PRESETS = {
    "source":  {"label": "Original quality", "args": []},
    "youtube": {"label": "YouTube 1080p 16:9",
                "args": ["-vf", "scale=-2:1080", "-r", "30"]},
    "reels":   {"label": "Reels / Shorts 9:16",
                "args": ["-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
                         "-r", "30"]},
    "square":  {"label": "Feed 1:1",
                "args": ["-vf", "scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080",
                         "-r", "30"]},
    "audio":   {"label": "Audio only (podcast)", "args": ["-vn"]},
}


def _project_or_404(pid: str) -> dict:
    p = store.get_project(pid)
    if not p:
        raise HTTPException(404, "no such project")
    return p


def _words(pid: str) -> list[Word]:
    return [Word(w["text"], w["start"], w["end"], speaker=w["speaker"],
                 deleted=bool(w["deleted"]), probability=w["probability"])
            for w in store.get_words(pid)]


def _silences(pid: str, src: str) -> list[tuple[float, float]]:
    """Detected silences for a project, cached -- it is a full pass over the
    audio and the answer never changes for a given source file."""
    cache = store.project_dir(pid) / "silences.json"
    if cache.exists():
        return [tuple(x) for x in json.loads(cache.read_text())]
    found = detect_silences(src, min_dur=0.20)      # threshold measured per file
    cache.write_text(json.dumps(found))
    return found


def _plan(pid: str):
    """Cuts derived from the surviving words, snapped to silence, with dead air
    removed if the project asks for it."""
    p = _project_or_404(pid)
    src = p["source"]
    sil: list[tuple[float, float]] = []
    if src and Path(src).exists():
        sil = _silences(pid, src)
    removed = []
    if p["silence_on"]:
        removed = silence_gaps(sil, keep=p["silence_keep"], min_remove=p["silence_min"])
    return derive_cuts(_words(pid), snap_points=snap_points(sil), snap_tolerance=0.15,
                       removed_ranges=removed, duration=p["duration"] or None,
                       fps=p["fps"] or None)


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

        store.update_project(pid, status="ready")
        store.update_job(jid, status="done", progress=1.0,
                         message=str(len(words)) + " words")
    except Exception as e:
        store.update_project(pid, status="failed")
        store.update_job(jid, status="failed",
                         message=(type(e).__name__ + ": " + str(e))[:500])


def _export(pid: str, jid: str, preset: str) -> None:
    try:
        store.update_job(jid, status="running", progress=0.05, message="planning cuts")
        p = store.get_project(pid)
        plan = _plan(pid)
        if not plan.cuts:
            raise ValueError("everything is deleted -- nothing left to export")
        ext = "m4a" if preset == "audio" else "mp4"
        out = store.project_dir(pid) / f"export-{preset}-{int(time.time())}.{ext}"
        store.update_job(jid, progress=0.15,
                         message=f"rendering {len(plan.cuts)} cuts, {plan.duration/60:.1f} min")
        render(p["source"], plan, out, extra=PRESETS[preset]["args"])
        # Report the real duration, not the arithmetic one: ffmpeg's video trim
        # includes the partial frame at each boundary, so the file runs ~0.15%
        # long. Measured in spikes/drift.py -- audio and video stay in sync.
        actual = media_info(out)["duration"]
        store.update_job(jid, status="done", progress=1.0, result=out.name,
                         message=f"{actual/60:.1f} min from {len(plan.cuts)} cuts")
    except Exception as e:
        store.update_job(jid, status="failed",
                         message=(type(e).__name__ + ": " + str(e))[:500])


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


@app.get("/api/projects/{pid}/plan")
def api_plan(pid: str):
    """The cut list. The player uses this to seek over deleted ranges, which is
    what makes preview instant -- nothing is re-encoded until export."""
    plan = _plan(pid)
    return {"duration": round(plan.duration, 3),
            "cuts": [{"start": c.start, "end": c.end} for c in plan.cuts]}


@app.post("/api/projects/{pid}/export")
def api_export(pid: str, body: dict = Body(default={})):
    _project_or_404(pid)
    preset = body.get("preset", "source")
    if preset not in PRESETS:
        raise HTTPException(400, "unknown preset " + str(preset))
    jid = store.create_job(pid, "export")
    store.run_in_background(_export, pid, jid, preset)
    return {"job": jid}


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
