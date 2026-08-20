"""Spike 2 -- does the cut engine produce clean, in-sync joins?

Builds a real video from the AMI speech, deletes words, renders, and then checks
the three things that would sink the core feature:
  1. output duration == sum of surviving ranges (an off-by-one here silently
     desyncs a two-hour podcast)
  2. video and audio stream durations agree (A/V sync at joins)
  3. joins do not click -- measured, not eyeballed
"""
import subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

import json
import numpy as np

from paperedit.audio import detect_silences, snap_points
from paperedit.edl import Word, derive_cuts, mark_fillers
from paperedit.render import media_info, render

MEDIA = HERE / "media"
OUT = HERE / "out"
CLIP = 300.0


def build_test_video() -> Path:
    """testsrc2 video + the real meeting audio, so joins are tested on speech.

    No drawtext burn-in: fontconfig is unavailable on this Windows ffmpeg build and
    crashes it. The automated checks below do not need one.
    """
    dst = MEDIA / "test-300s.mp4"
    if dst.exists():
        return dst
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=30:duration={CLIP}",
        "-i", str(MEDIA / "ami-ES2002a.wav"), "-t", str(CLIP),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", str(dst)], check=True)
    return dst


def load_words() -> list[Word]:
    d = json.loads((OUT / "lv3_gpu_300.json").read_text(encoding="utf-8"))
    return [Word(w["t"], w["s"], w["e"], probability=w["p"]) for w in d["words"]]


def click_score(path: Path, joins: list[float]) -> tuple[float, float]:
    """Max sample-to-sample jump near a join vs. elsewhere. A click is a jump."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1",
         "-ar", "16000", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    d = np.abs(np.diff(x))
    sr, win = 16000, int(0.01 * 16000)
    mask = np.zeros(len(d), bool)
    for j in joins:
        i = int(j * sr)
        mask[max(0, i - win):min(len(d), i + win)] = True
    at = d[mask].max() if mask.any() else 0.0
    away = d[~mask].max() if (~mask).any() else 0.0
    return float(at), float(away)


def check(label: str, src: Path, plan, gpu: bool):
    dst = OUT / f"cut_{label}.mp4"
    t0 = time.perf_counter()
    render(src, plan, dst, gpu=gpu)
    dt = time.perf_counter() - t0
    info = media_info(dst)
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(dst)],
        capture_output=True, text=True, check=True).stdout)
    durs = {s["codec_type"]: float(s.get("duration", 0)) for s in probe["streams"]}

    joins, acc = [], 0.0
    for c in plan.cuts[:-1]:
        acc += c.duration
        joins.append(acc)
    at, away = click_score(dst, joins)

    print(f"\n{label}  ({len(plan.cuts)} cuts, {'GPU/NVENC' if gpu else 'CPU/x264'})")
    print(f"  render          {dt:.1f}s for {plan.duration:.1f}s out "
          f"({plan.duration/dt:.1f}x realtime)")
    print(f"  expected dur    {plan.duration:.3f}s")
    print(f"  actual dur      {info['duration']:.3f}s   "
          f"drift {abs(info['duration']-plan.duration)*1000:+.0f} ms")
    print(f"  video/audio     {durs.get('video',0):.3f} / {durs.get('audio',0):.3f}  "
          f"skew {abs(durs.get('video',0)-durs.get('audio',0))*1000:.0f} ms")
    print(f"  peak jump       at joins {at:.3f}   elsewhere {away:.3f}   "
          f"{'CLICK RISK' if at > away * 1.2 else 'clean'}")
    return dst


def main():
    src = build_test_video()
    print(f"test video: {media_info(src)}")

    sil = detect_silences(src)
    pts = snap_points(sil)
    print(f"silences detected: {len(sil)}  (snap points for cut boundaries)")

    words = load_words()
    n_fill = mark_fillers(words)
    # Also delete a whole sentence in the middle, the way a real edit would.
    for w in words:
        if 150.0 <= w.start <= 165.0:
            w.deleted = True
    n_del = sum(1 for w in words if w.deleted)
    print(f"deleted {n_del} words ({n_fill} fillers + one 15s passage)")

    raw = derive_cuts(words, snap_points=(), duration=CLIP)
    snapped = derive_cuts(words, snap_points=pts, snap_tolerance=0.15, duration=CLIP)
    print(f"cuts: {len(raw.cuts)} unsnapped / {len(snapped.cuts)} snapped-to-silence")

    check("nosnap_gpu", src, raw, gpu=True)
    check("snapped_gpu", src, snapped, gpu=True)


if __name__ == "__main__":
    sys.exit(main())
