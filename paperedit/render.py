"""Turn an EditPlan into a file, with ffmpeg.

Uses trim/concat in a filter graph rather than the concat demuxer: the demuxer
can only cut on keyframes, which drifts cuts by up to a GOP (~2s). Frame-accurate
cutting requires re-encoding, so we do that, on NVENC where available.

The filter graph is written to a SCRIPT FILE, not the command line -- a two-hour
podcast with filler removal can reach thousands of cuts and blow the Windows
32k command-line limit.

EACH CUT READS ITS OWN SEEKED INPUT when the command line allows it. With one
input, every cut's trim branch is fed the source from the very beginning up to
where that cut ends, and throws the early frames away -- so the work is
(number of cuts x length of the video). A real 97-cut, 9-minute edit took about
half an hour that way (measured 30 Sep 2026: 40 cuts over one minute, 72.5 s
with one input vs 6.8 s seeked). An input opened with -ss decodes only its own
span. Past the command-line limit it falls back to the single input.

Every seeked input is a decoder of its own, all open at once, and they decode
ahead into memory: about 87 MB per cut even with one decoder thread each (24
cuts 2.2 GB, 48 cuts 4.2 GB, 97 cuts 8.4 GB). With ffmpeg's default threads
per decoder the 97-cut edit ran out of memory. So each input gets -threads 1,
and past MAX_SEEKED_INPUTS the export takes the single-input path (about
3.4 GB, slower) instead.
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
# Below this, the whole export falls back to hard cuts. See build_filtergraph.
# One frame at 24 fps, so it is at least a frame at every common recording rate.
MIN_VIDEO_DISSOLVE = 1 / 24


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


def join_dissolves(plan: EditPlan, *, video: bool,
                   vxfade: float = VIDEO_XFADE) -> list[float]:
    """The dissolves the export will actually use: video_dissolves, or none at
    all when any join is too short to blend (see build_filtergraph)."""
    n = len(plan.cuts)
    dissolves = (video_dissolves(plan, vxfade)
                 if video and n > 1 and vxfade > 0 else [])
    if dissolves and min(dissolves) < MIN_VIDEO_DISSOLVE:
        return []
    return dissolves


def build_filtergraph(plan: EditPlan, *, video: bool, xfade: float = AUDIO_XFADE,
                      audio_filters: str = "", video_filters: str = "",
                      vxfade: float = VIDEO_XFADE, seeked: bool = False) -> str:
    """trim each surviving range, then concat. Audio gets a short fade at each
    join so a cut through a waveform doesn't click, and the picture gets a
    short dissolve so it doesn't jump.

    `seeked=True` expects one input per cut, opened at that cut's start (see
    seek_inputs), so each trim is relative to its own input and starts at 0."""
    parts, vlabels, alabels = [], [], []
    n = len(plan.cuts)
    dissolves = join_dissolves(plan, video=video, vxfade=vxfade)
    # ONE SHORT JOIN AND THE WHOLE PICTURE GOES HARD-CUT. An xfade narrower
    # than a frame does not fail -- it quietly ends the video stream, and
    # ffmpeg exits 0 with the picture stopping up to 23 seconds before the
    # sound. Measured 16 Sep 2026 against a 6 ms kept sliver and a 0.03 ms gap
    # (which prints as duration=0.0000); the hard-cut path handled both within
    # a frame. Silence removal makes both shapes reachable: it is subtracted
    # AFTER the cuts are put on the frame grid, so it can leave a fragment
    # shorter than a frame.
    #
    # Falling back for the whole export rather than per join keeps this one
    # branch instead of a mixed concat/xfade graph. A correct video without
    # fades beats a broken one with them, and real edits have not come
    # near it: a measured 97-cut edit's shortest dissolve was 75 ms.
    blending = bool(dissolves)
    for i, c in enumerate(plan.cuts):
        dur = c.duration
        fade = min(xfade, dur / 4) if dur > 0 else 0
        if video:
            # Reaches PAST the cut, into the deleted material, so the next
            # segment has something to dissolve through.
            v_end = c.end + (dissolves[i] if i < len(dissolves) else 0.0)
            v_in, v_from, v_to = ((f"[{i}:v]", 0.0, v_end - c.start) if seeked
                                  else ("[0:v]", c.start, v_end))
            parts.append(
                f"{v_in}trim=start={v_from:.4f}:end={v_to:.4f},"
                f"setpts=PTS-STARTPTS[v{i}];")
            vlabels.append(f"[v{i}]")
        a_in, a_from, a_to = ((f"[{i}:a]", 0.0, dur) if seeked
                              else ("[0:a]", c.start, c.end))
        parts.append(
            f"{a_in}atrim=start={a_from:.4f}:end={a_to:.4f},"
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


# Windows refuses command lines over 32,767 characters. Leave room for the
# encoder options and the output path that follow the inputs.
MAX_INPUT_ARGS_CHARS = 30_000
# About 87 MB each (see the module docstring): 150 inputs is roughly 13 GB.
MAX_SEEKED_INPUTS = 150


def seek_inputs(source: str | Path, plan: EditPlan,
                dissolves: list[float]) -> list[str]:
    """One -ss/-t input per cut: it starts at the cut and runs to the end of
    what that cut needs (its dissolve tail included), plus a little slack --
    the trims in the graph set the exact ends."""
    args: list[str] = []
    for i, c in enumerate(plan.cuts):
        end = c.end + (dissolves[i] if i < len(dissolves) else 0.0)
        args += ["-threads", "1",
                 "-ss", f"{c.start:.4f}", "-t", f"{end - c.start + 0.1:.4f}",
                 "-i", str(source)]
    return args


def choose_inputs(source: str | Path, plan: EditPlan, *,
                  video: bool) -> tuple[list[str], bool]:
    """(ffmpeg input args, seeked?) -- one seeked input per cut when there are
    few enough to hold in memory and to fit on the command line, otherwise the
    whole file once."""
    inputs = seek_inputs(source, plan, join_dissolves(plan, video=video))
    if (1 < len(plan.cuts) <= MAX_SEEKED_INPUTS
            and len(" ".join(inputs)) <= MAX_INPUT_ARGS_CHARS):
        return inputs, True
    return ["-i", str(source)], False


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
    inputs, seeked = choose_inputs(source, plan, video=want_video)
    graph = build_filtergraph(plan, video=want_video,
                              audio_filters=audio_filters,
                              video_filters=video_filters, seeked=seeked)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(graph)
        script = fh.name

    cmd = ["ffmpeg", "-y", "-v", "error", *inputs,
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
