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

import re
import subprocess
from pathlib import Path
from typing import Callable, Iterator

from cuda_setup import enable_cuda

GPU_MODEL = "large-v3"
CPU_MODEL = "small"
GPU_VRAM_NEEDED_MB = 4000

# VERBATIM (22 Oct 2026, FINDINGS.md section 5): CrisperWhisper writes down the "um"s and
# "uh"s Whisper tidies away (0 of 76 -> 53 of 76 on AMI; 2 -> 11 on a real 10-minute
# talk), plus the repeats real speech has ("I I think"). Used when it has been downloaded and the
# GPU is free - it is a large-v3-sized model, too slow on CPU. Settings are the repeatable
# ones: beam 5, silence skipped (VAD), NO temperature fallback - the fallback samples, so
# the same video gave different cuts on different runs - and a no-repeat guard, because
# without the fallback it loops (see below).
VERBATIM_MODEL = "nyralabs/faster_CrisperWhisper"


def verbatim_available() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(VERBATIM_MODEL, "model.bin"), str)
    except Exception:
        return False


def clean_word(text: str) -> str:
    """CrisperWhisper marks fillers as [UM] / [UH] - sometimes with the previous word's
    comma glued on (",[UM]", seen on AMI 22 Oct); the editor shows them as um / uh."""
    m = re.fullmatch(r"\W*\[([A-Za-z]+)\]\W*", text.strip())
    if m:
        return " " + m.group(1).lower()
    return text


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
    verbatim = device == "cuda" and verbatim_available()
    if verbatim:
        model_name = VERBATIM_MODEL
    if progress:
        progress(0.0, f"loading {model_name} on {device}")

    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device=device, compute_type=compute,
                         cpu_threads=32)

    # no_repeat_ngram_size=4: without it, no-fallback decoding LOOPS - "So, we're. So,
    # we're..." in 10 ms "words"; 324 of 545 words of the test video were loop garbage
    # (spikes/loop_check.py, 23 Oct). Forbidding any 4-word run from repeating took that
    # to 0 AND lifted AMI filler recall from 38/76 to 53/76, still repeatable. 3 cost
    # fillers (27/76); a repetition penalty left 71 loops.
    extra = ({"vad_filter": True, "temperature": 0.0, "no_repeat_ngram_size": 4}
             if verbatim else {})
    segments, info = model.transcribe(
        str(path), language=language, beam_size=5, word_timestamps=True,
        condition_on_previous_text=False, **extra,
    )
    total = info.duration or 1.0
    words: list[dict] = []
    for seg in segments:
        for w in (seg.words or []):
            text = clean_word(w.word)
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
