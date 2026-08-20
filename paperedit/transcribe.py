"""Transcription with word-level timestamps.

Model choice is measured, not guessed (see FINDINGS.md):
  * large-v3 on GPU  -- 30x realtime, best quality; used when the 3080 is free
  * small on CPU     -- 8x realtime; beats `base` on BOTH timestamp accuracy and
                        filler recall, and 8x realtime is still ~14 min for a
                        two-hour podcast, so the extra quality is nearly free
The GPU may be shared with other local AI tools, so we check free VRAM before
claiming it rather than crashing halfway through someone's podcast.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Iterator

from cuda_setup import enable_cuda

GPU_MODEL = "large-v3"
CPU_MODEL = "small"
GPU_VRAM_NEEDED_MB = 4000


def free_vram_mb() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return int(out[0]) if out else 0
    except Exception:
        return 0


def pick_device() -> tuple[str, str, str]:
    """(device, compute_type, model). Falls back to CPU whenever the GPU is busy."""
    enable_cuda()
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0 and free_vram_mb() >= GPU_VRAM_NEEDED_MB:
            return "cuda", "float16", GPU_MODEL
    except Exception:
        pass
    return "cpu", "int8", CPU_MODEL


def transcribe(path: str | Path, *, language: str = "en",
               progress: Callable[[float, str], None] | None = None) -> list[dict]:
    """Return word dicts: text, start, end, probability."""
    device, compute, model_name = pick_device()
    if progress:
        progress(0.0, f"loading {model_name} on {device}")

    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device=device, compute_type=compute,
                         cpu_threads=32)

    segments, info = model.transcribe(
        str(path), language=language, beam_size=5, word_timestamps=True,
        condition_on_previous_text=False,
    )
    total = info.duration or 1.0
    words: list[dict] = []
    for seg in segments:
        for w in (seg.words or []):
            text = w.word
            if not text.strip():
                continue
            words.append({"text": text, "start": round(w.start, 3),
                          "end": round(max(w.end, w.start + 0.01), 3),
                          "probability": round(w.probability, 3)})
        if progress:
            progress(min(0.99, seg.end / total), f"{seg.end/60:.1f} / {total/60:.1f} min")
    if progress:
        progress(1.0, f"{len(words)} words")
    return words
