"""Spike 3 -- does the Studio Sound chain actually help, and does it hurt speech?

Three measurements on a real recording:
  1. loudness      does it land on the -16 LUFS broadcast target?
  2. noise floor   do the gaps between words get quieter? (that IS denoising)
  3. speech        does transcription confidence survive? A denoiser that eats
                   consonants will show up here as a confidence drop -- this is
                   the check that stops us shipping something that "sounds clean"
                   because it has smoothed the voice into mush.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from paperedit.audio import detect_silences, noise_floor
from paperedit.enhance import (PRESETS, build_chain, measure_loudness,
                               noise_floor_rms, render_audio)

EXCERPT = 90.0
START = 60.0


def db(x):
    import math
    return 20 * math.log10(x) if x and x > 0 else float("-inf")


def transcribe_confidence(path):
    from paperedit.transcribe import transcribe
    words = transcribe(path)
    if not words:
        return 0, 0.0, 0.0
    probs = [w["probability"] for w in words]
    return len(words), sum(probs) / len(probs), sum(1 for p in probs if p < 0.5) / len(probs)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else None
    if not src or not Path(src).exists():
        print("usage: studio_sound.py <media file>")
        return 1

    work = HERE / "out"
    work.mkdir(exist_ok=True)
    raw = work / "ss_raw.m4a"
    render_audio(src, raw, preset="off", start=START, dur=EXCERPT)
    quiet = [s for s in detect_silences(raw) if s[1] - s[0] >= 0.3]
    print(f"excerpt: {EXCERPT:.0f}s from {START:.0f}s, {len(quiet)} quiet stretches to sample\n")

    base_loud = measure_loudness(raw)
    base_noise = noise_floor_rms(raw, quiet)
    n0, p0, low0 = transcribe_confidence(raw)
    print(f"{'preset':<9} {'LUFS':>7} {'peak':>7} {'noise floor':>12} {'vs raw':>8} "
          f"{'words':>6} {'conf':>6} {'low-conf':>9}")
    print("-" * 76)
    print(f"{'raw':<9} {base_loud['lufs']:>7.1f} {base_loud['true_peak']:>7.1f} "
          f"{db(base_noise):>11.1f}dB {'--':>8} {n0:>6} {p0:>6.3f} {low0*100:>8.1f}%")

    # "normalize" is the fair baseline: same loudness, no denoising. Comparing
    # the full chain against the RAW file instead makes denoise look harmful,
    # because loudnorm lifts the hiss along with the voice.
    for preset in ("normalize", "light", "studio", "strong"):
        out = work / f"ss_{preset}.m4a"
        render_audio(src, out, preset=preset, start=START, dur=EXCERPT)
        loud = measure_loudness(out)
        noise = noise_floor_rms(out, quiet)
        n, p, low = transcribe_confidence(out)
        if preset == "normalize":
            ref_noise = noise                     # everything after is judged vs this
        delta = db(noise) - db(ref_noise if preset != "normalize" else base_noise)
        print(f"{preset:<9} {loud['lufs']:>7.1f} {loud['true_peak']:>7.1f} "
              f"{db(noise):>11.1f}dB {delta:>+7.1f}dB {n:>6} {p:>6.3f} {low*100:>8.1f}%")

    print("\nwhat to look for:")
    print("  LUFS should land near -16, peak at or under -1.5")
    print("  noise floor: the normalize row is vs raw and SHOULD rise -- that is")
    print("    loudnorm lifting everything. Later rows are vs normalize, and")
    print("    THOSE should be negative. That is the denoise actually working.")
    print("  words/confidence should hold steady; a fall means speech is damaged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
