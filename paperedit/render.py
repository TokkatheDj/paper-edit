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

AUDIO_XFADE = 0.02  # 20 ms; hides the sample discontinuity at a join


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


def build_filtergraph(plan: EditPlan, *, video: bool, xfade: float = AUDIO_XFADE) -> str:
    """trim each surviving range, then concat. Audio gets a short fade at each
    join so a cut through a waveform doesn't click."""
    parts, vlabels, alabels = [], [], []
    for i, c in enumerate(plan.cuts):
        dur = c.duration
        fade = min(xfade, dur / 4) if dur > 0 else 0
        if video:
            parts.append(
                f"[0:v]trim=start={c.start:.4f}:end={c.end:.4f},"
                f"setpts=PTS-STARTPTS[v{i}];")
            vlabels.append(f"[v{i}]")
        parts.append(
            f"[0:a]atrim=start={c.start:.4f}:end={c.end:.4f},"
            f"asetpts=PTS-STARTPTS,"
            f"afade=t=in:st=0:d={fade:.4f},"
            f"afade=t=out:st={max(0.0, dur - fade):.4f}:d={fade:.4f}[a{i}];")
        alabels.append(f"[a{i}]")

    n = len(plan.cuts)
    if video:
        parts.append("".join(f"{v}{a}" for v, a in zip(vlabels, alabels))
                     + f"concat=n={n}:v=1:a=1[vout][aout]")
    else:
        parts.append("".join(alabels) + f"concat=n={n}:v=0:a=1[aout]")
    return "\n".join(parts)


def render(source: str | Path, plan: EditPlan, out_path: str | Path, *,
           gpu: bool | None = None, crf: int = 20, extra: list[str] | None = None) -> Path:
    if not plan.cuts:
        raise ValueError("EditPlan has no surviving cuts -- nothing to render")
    info = media_info(source)
    use_gpu = has_nvenc() if gpu is None else gpu
    graph = build_filtergraph(plan, video=info["has_video"])

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(graph)
        script = fh.name

    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(source),
           "-filter_complex_script", script]
    if info["has_video"]:
        cmd += ["-map", "[vout]"]
        cmd += (["-c:v", "h264_nvenc", "-preset", "p5", "-cq", str(crf)]
                if use_gpu else ["-c:v", "libx264", "-preset", "veryfast",
                                 "-crf", str(crf)])
    cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    cmd += (extra or []) + [str(out_path)]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
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
