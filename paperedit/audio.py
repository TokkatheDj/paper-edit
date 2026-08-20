"""Silence analysis -- the snap points that make cuts sound intentional.

Whisper word boundaries carry ~50-130 ms of median error, so cutting exactly on a
word boundary clips consonants and leaves half-breaths. Cutting at the quietest
nearby moment instead hides the seam. This module finds those moments, and is
also the basis of Phase 2 silence removal.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


def detect_silences(path: str | Path, *, noise_db: float = -32.0,
                    min_dur: float = 0.12) -> list[tuple[float, float]]:
    """Return (start, end) of every silent stretch, via ffmpeg silencedetect."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
           "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
           "-f", "null", "-"]
    err = subprocess.run(cmd, capture_output=True, text=True).stderr
    starts = [float(m) for m in _START.findall(err)]
    ends = [float(m) for m in _END.findall(err)]
    out, pending = [], None
    for line in err.splitlines():
        if (m := _START.search(line)):
            pending = float(m.group(1))
        elif (m := _END.search(line)) and pending is not None:
            out.append((max(0.0, pending), float(m.group(1))))
            pending = None
    return out


def snap_points(silences: list[tuple[float, float]]) -> list[float]:
    """Midpoint of each silence -- the safest instant to cut."""
    return [round((s + e) / 2, 4) for s, e in silences]


def silence_gaps(silences: list[tuple[float, float]], *, keep: float = 0.30,
                 min_remove: float = 0.60) -> list[tuple[float, float]]:
    """Ranges to DELETE so no pause runs longer than `keep` seconds.

    Only touches silences longer than min_remove, so natural beats survive and
    only genuinely dead air is compressed.
    """
    out = []
    for s, e in silences:
        if e - s >= min_remove:
            out.append((round(s + keep / 2, 4), round(e - keep / 2, 4)))
    return [(s, e) for s, e in out if e > s]
