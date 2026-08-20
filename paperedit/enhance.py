"""Studio Sound -- make a room recording sound like it was meant to be heard.

Built entirely from ffmpeg's own filters. That is a deliberate choice: the
alternative (DeepFilterNet) means torch plus ~2.5 GB of CUDA wheels, and this
project's whole premise is that it runs on your machine without a stack. The
chain below is measured in spikes/studio_sound.py rather than assumed good; if
the numbers ever stop justifying it, the door to a neural denoiser is still open.

The order matters:
  highpass     kill rumble, HVAC and desk thumps before anything tries to model
               the noise floor
  afftdn       spectral subtraction with noise tracking -- the actual denoise
  deesser      tame sibilance that denoising tends to leave standing proud
  acompressor  even out the distance-from-mic swings that make casual recordings
               feel amateur
  loudnorm     EBU R128 to -16 LUFS, the podcast/streaming target, with true peak
               held at -1.5 dB so nothing clips on playback
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

# -16 LUFS integrated / -1.5 dBTP is the usual target for spoken-word streaming.
TARGET_LUFS = -16.0
TARGET_TP = -1.5

PRESETS = {
    "off": None,
    # Loudness only, no denoise. Exists as the honest BASELINE for measurement:
    # loudnorm lifts a quiet recording by several dB, which lifts its hiss too,
    # so comparing the full chain against the raw file makes denoising look like
    # it made things worse. Compare against this instead.
    "normalize": {"nr": 0, "comp": None, "deess": 0},
    # Denoise WITHOUT the compressor. Measured: the compressor lifts quiet
    # passages, and residual hiss with them, partly undoing the denoise.
    "denoise": {"nr": 14, "comp": None, "deess": 0.4},
    "light": {"nr": 8, "comp": "threshold=-20dB:ratio=2:attack=20:release=300",
              "deess": 0.3},
    "studio": {"nr": 14, "comp": "threshold=-18dB:ratio=3:attack=15:release=250",
               "deess": 0.4},
    "strong": {"nr": 22, "comp": "threshold=-16dB:ratio=4:attack=10:release=200",
               "deess": 0.5},
}


def build_chain(preset: str = "studio", *, normalize: bool = True) -> str:
    """The audio filter chain for a preset, as an ffmpeg filter string."""
    cfg = PRESETS.get(preset)
    if cfg is None:
        return ""
    parts = ["highpass=f=80:poles=2"]
    if cfg["nr"]:
        parts.append(f"afftdn=nr={cfg['nr']}:nf=-30:tn=1")
    if cfg["deess"]:
        parts.append(f"deesser=i={cfg['deess']}")
    if cfg["comp"]:
        parts.append(f"acompressor={cfg['comp']}")
    if normalize:
        parts.append(f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA=11")
    return ",".join(parts)


_LUFS = re.compile(r"I:\s*(-?[\d.]+) LUFS")
_TP = re.compile(r"Peak:\s*(-?[\d.]+) dBFS")


def measure_loudness(path: str | Path, *, filters: str = "") -> dict:
    """Integrated loudness and true peak, via ffmpeg's ebur128."""
    chain = (filters + "," if filters else "") + "ebur128=peak=true"
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", chain, "-f", "null", "-"],
        capture_output=True, text=True).stderr
    lufs = _LUFS.findall(out)
    peak = _TP.findall(out)
    return {"lufs": float(lufs[-1]) if lufs else None,
            "true_peak": float(peak[-1]) if peak else None}


def noise_floor_rms(path: str | Path, quiet_ranges, *, filters: str = "",
                    limit: int = 12) -> float | None:
    """Mean RMS inside known-quiet stretches -- i.e. how loud the hiss is.

    This is the number that says whether denoising worked: speech should stay
    put while the gaps between words get quieter.
    """
    import numpy as np
    vals = []
    for start, end in list(quiet_ranges)[:limit]:
        if end - start < 0.25:
            continue
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end-start:.3f}",
               "-i", str(path)]
        if filters:
            cmd += ["-af", filters]
        cmd += ["-f", "f32le", "-ac", "1", "-ar", "16000", "-"]
        raw = subprocess.run(cmd, capture_output=True).stdout
        if not raw:
            continue
        x = np.frombuffer(raw, dtype=np.float32)
        if x.size:
            vals.append(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))))
    if not vals:
        return None
    import numpy as np
    return float(np.mean(vals))


def render_audio(src: str | Path, dst: str | Path, *, preset: str = "studio",
                 start: float | None = None, dur: float | None = None) -> Path:
    """Write an enhanced copy -- used for A/B samples and for tests."""
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-i", str(src)]
    chain = build_chain(preset)
    if chain:
        cmd += ["-af", chain]
    cmd += ["-vn", "-c:a", "aac", "-b:a", "192k", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True)
    return Path(dst)


def measure_snr(path: str | Path, *, window: float = 0.05) -> float:
    """Rough speech-to-noise ratio in dB, without needing silence detection.

    Short-term RMS over the whole file: the 10th percentile is essentially the
    noise floor, the median is essentially speech. Their ratio says how clean the
    recording is. This works on a noisy file, where silence detection finds no
    silences at all precisely BECAUSE it is noisy.
    """
    import numpy as np
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1",
         "-ar", "16000", "-"], capture_output=True).stdout
    x = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    if x.size == 0:
        return 0.0
    n = int(window * 16000)
    usable = (x.size // n) * n
    if usable < n:
        return 0.0
    frames = x[:usable].reshape(-1, n)
    rms = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-12
    floor, speech = np.percentile(rms, 10), np.percentile(rms, 50)
    import math
    return float(20 * math.log10(speech / floor))


# Thresholds set from measurement, not taste: her quiet room recording and the
# same clip with room noise mixed in sit either side of these.
def recommend_preset(path: str | Path) -> tuple[str, float, str]:
    """(preset, snr_db, human reason) -- what this recording actually needs.

    Denoising a clean recording measurably HURTS it: on a quiet-room take, the
    denoise presets lost words and dropped transcription confidence, while plain
    loudness normalisation improved both. On a noisy take the reverse held. So
    the app picks by measurement instead of applying one chain to everything.
    """
    snr = measure_snr(path)
    # Boundaries chosen from measured outcomes, not taste. A quiet-room take
    # measured 21.4 dB and was harmed by every denoise preset; the same clip with
    # room noise mixed in measured 13.1 dB and was clearly helped by them.
    if snr >= 20:
        return "normalize", snr, "already clean - levelling only, denoising would hurt it"
    if snr >= 15:
        return "light", snr, "a little background noise"
    if snr >= 11:
        return "studio", snr, "noticeable room noise"
    return "strong", snr, "heavy background noise"
