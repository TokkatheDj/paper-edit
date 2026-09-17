"""Turn an EditPlan into a file, with ffmpeg.

Uses trim/concat in a filter graph rather than the concat demuxer: the demuxer
can only cut on keyframes, which drifts cuts by up to a GOP (~2s). Frame-accurate
cutting requires re-encoding, so we do that, on NVENC where available.

The filter graph is written to a SCRIPT FILE, not the command line -- a two-hour
podcast with filler removal can reach thousands of cuts and blow the Windows
32k command-line limit.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from .edl import EditPlan

BS = chr(92)
SEP_CHAIN = chr(59) + chr(10)   # ";" newline -- ffmpeg filterchain separator
AUDIO_XFADE = 0.02  # 20 ms; hides the sample discontinuity at a join
# 100 ms; hides the JUMP in the picture at a join. Audio was already smoothed
# and the cuts are already snapped to silence, so what was left to fix was
# purely visual: the speaker's head and hands are in a different position either side of
# a cut, and the picture snapped between them.
#
# From the first real edit, 7 Sep 2026: "there is an unnatural/inorganic flow that breaks the
# video in a way that does not look right. It needs to blend the parts that are
# left to connect after words or sentences were removed."
VIDEO_XFADE = 0.10


def ffprobe(path: str | Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def media_info(path: str | Path) -> dict:
    d = ffprobe(path)
    v = next((s for s in d["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in d["streams"] if s["codec_type"] == "audio"), None)
    fps = 0.0
    if v and v.get("r_frame_rate", "0/1") != "0/0":
        num, _, den = v["r_frame_rate"].partition("/")
        fps = float(num) / float(den or 1)
    return {
        "duration": float(d["format"].get("duration", 0.0)),
        "has_video": v is not None,
        "has_audio": a is not None,
        "width": int(v["width"]) if v else 0,
        "height": int(v["height"]) if v else 0,
        "fps": fps,
    }


def has_nvenc() -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True).stdout
        return "h264_nvenc" in out
    except FileNotFoundError:
        return False


def escape_filter_path(path: str | Path) -> str:
    r"""Make a Windows path safe inside an ffmpeg filter argument.

    NOTE: escaping a drive letter here is unreliable -- ffmpeg still split
    'D\:/dir/x.ass' on the colon and tried to read the tail as another option.
    Prefer running ffmpeg with cwd set to the file's directory and passing a
    bare filename; this helper remains for paths that have no drive letter.
    """
    return str(path).replace(BS, '/').replace(':', BS + ':')


def video_dissolves(plan: EditPlan, vxfade: float = VIDEO_XFADE) -> list[float]:
    """How long to dissolve at each join -- one entry per gap between cuts.

    THE DISSOLVE IS TAKEN FROM THE MATERIAL BEING THROWN AWAY. Segment i's
    picture is extended past its cut point INTO the deleted range, and that
    extension is what segment i+1 fades up through. Neither segment's kept
    content is shortened, so the output is exactly as long as a hard cut would
    have been.

    That matters more than it looks. A plain xfade overlaps two clips and so
    eats its own duration at every join -- 28 deletions would have made the
    export 2.8 seconds shorter than the plan says. Caption timings are computed
    against the EDITED timeline, so every subtitle after the first cut would
    have drifted, and the drift would have grown with each join.

    Clamped to the gap (we cannot borrow more than was deleted) and to half of
    each neighbouring segment (a dissolve cannot be longer than the shot).
    """
    cuts = plan.cuts
    out = []
    for i in range(len(cuts) - 1):
        gap = cuts[i + 1].start - cuts[i].end
        out.append(max(0.0, min(vxfade, gap,
                                cuts[i].duration / 2, cuts[i + 1].duration / 2)))
    return out


def build_filtergraph(plan: EditPlan, *, video: bool, xfade: float = AUDIO_XFADE,
                      audio_filters: str = "", video_filters: str = "",
                      vxfade: float = VIDEO_XFADE) -> str:
    """trim each surviving range, then concat. Audio gets a short fade at each
    join so a cut through a waveform doesn't click, and the picture gets a
    short dissolve so it doesn't jump."""
    parts, vlabels, alabels = [], [], []
    n = len(plan.cuts)
    dissolves = (video_dissolves(plan, vxfade)
                 if video and n > 1 and vxfade > 0 else [])
    blending = any(d > 0 for d in dissolves)
    for i, c in enumerate(plan.cuts):
        dur = c.duration
        fade = min(xfade, dur / 4) if dur > 0 else 0
        if video:
            # Reaches PAST the cut, into the deleted material, so the next
            # segment has something to dissolve through.
            v_end = c.end + (dissolves[i] if i < len(dissolves) else 0.0)
            parts.append(
                f"[0:v]trim=start={c.start:.4f}:end={v_end:.4f},"
                f"setpts=PTS-STARTPTS[v{i}];")
            vlabels.append(f"[v{i}]")
        parts.append(
            f"[0:a]atrim=start={c.start:.4f}:end={c.end:.4f},"
            f"asetpts=PTS-STARTPTS,"
            f"afade=t=in:st=0:d={fade:.4f},"
            f"afade=t=out:st={max(0.0, dur - fade):.4f}:d={fade:.4f}[a{i}];")
        alabels.append(f"[a{i}]")

    # Studio Sound runs AFTER the concat, on the finished edit: loudness must
    # be measured across what the listener actually hears, not per fragment.
    tail = "acat" if audio_filters else "aout"
    vtail = "vcat" if video_filters else "vout"
    # ffmpeg separates filterchains with ';' -- a newline is only whitespace.
    # A separator is needed whenever ANY chain follows the concat, not just
    # an audio one -- captions alone would otherwise produce a broken graph.
    sep = ";" if (audio_filters or (video and video_filters)) else ""
    if blending:
        # Video and audio are assembled separately here: the picture is a chain
        # of xfades, the sound stays a plain concat of the exact trims. Both
        # come out the same length -- see video_dissolves -- so they stay in
        # sync without the audio ever being stretched or overlapped twice.
        run = plan.cuts[0].duration + dissolves[0]
        cur = vlabels[0]
        for i in range(1, n):
            d = dissolves[i - 1]
            out = f"[vx{i}]" if i < n - 1 else f"[{vtail}]"
            parts.append(f"{cur}{vlabels[i]}xfade=transition=fade:"
                         f"duration={d:.4f}:offset={run - d:.4f}{out};")
            run += plan.cuts[i].duration + (dissolves[i] if i < len(dissolves)
                                            else 0.0) - d
            cur = out
        parts.append("".join(alabels) + f"concat=n={n}:v=0:a=1[{tail}]{sep}")
    elif video:
        parts.append("".join(f"{v}{a}" for v, a in zip(vlabels, alabels))
                     + f"concat=n={n}:v=1:a=1[{vtail}][{tail}]{sep}")
    else:
        parts.append("".join(alabels) + f"concat=n={n}:v=0:a=1[{tail}]{sep}")
    chains = []
    if video and video_filters:
        chains.append(f"[vcat]{video_filters}[vout]")
    if audio_filters:
        chains.append(f"[acat]{audio_filters}[aout]")
    parts.append((SEP_CHAIN.join(chains)) if chains else "")
    parts = [x for x in parts if x]
    return "\n".join(parts)


def render(source: str | Path, plan: EditPlan, out_path: str | Path, *,
           gpu: bool | None = None, crf: int = 20, extra: list[str] | None = None,
           audio_filters: str = "", video_filters: str = "",
           video: bool | None = None, cwd: str | Path | None = None) -> Path:
    """Render the plan to out_path.

    Reframing and scaling belong in `video_filters`, NOT in `extra` as a -vf:
    the cut list is already a complex filtergraph, and ffmpeg refuses to apply
    simple and complex filtering to the same stream.

    `video=False` drops the video stream entirely (an audio-only export) --
    again not via `-vn` in `extra`, which would contradict the -map that puts
    the filtered video on the output.
    """
    if not plan.cuts:
        raise ValueError("EditPlan has no surviving cuts -- nothing to render")
    info = media_info(source)
    want_video = info["has_video"] and video is not False
    use_gpu = has_nvenc() if gpu is None else gpu
    graph = build_filtergraph(plan, video=want_video,
                              audio_filters=audio_filters,
                              video_filters=video_filters)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(graph)
        script = fh.name

    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(source),
           "-filter_complex_script", script]
    if want_video:
        cmd += ["-map", "[vout]"]
        cmd += (["-c:v", "h264_nvenc", "-preset", "p5", "-cq", str(crf)]
                if use_gpu else ["-c:v", "libx264", "-preset", "veryfast",
                                 "-crf", str(crf)])
    cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    cmd += (extra or []) + [str(out_path)]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True,
                       cwd=str(cwd) if cwd else None)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed:\n{e.stderr[-2000:]}") from e
    finally:
        Path(script).unlink(missing_ok=True)
    return Path(out_path)


def make_proxy(source: str | Path, out_path: str | Path, *, height: int = 720,
               gpu: bool | None = None) -> Path:
    """Small, seekable, keyframe-dense copy for instant scrubbing in the editor."""
    use_gpu = has_nvenc() if gpu is None else gpu
    vcodec = (["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "28"] if use_gpu
              else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "28"])
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(source),
           "-vf", f"scale=-2:{height}", *vcodec,
           "-g", "30", "-c:a", "aac", "-b:a", "128k",
           "-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return Path(out_path)
