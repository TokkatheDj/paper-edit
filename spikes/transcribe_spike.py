"""Spike 1 — transcription speed, word timestamps, and filler-word emission.

Answers the Phase 0 questions:
  * How fast is faster-whisper on this machine (CPU vs GPU, model size)?
  * Do we get usable word-level timestamps?
  * Does Whisper emit "um"/"uh" at all -- and does a filler-primed prompt help?

Usage:
  python transcribe_spike.py MEDIA [--model base] [--device cpu] [--threads 32]
                                   [--limit 300] [--prompt-fillers] [--out NAME]
"""
import argparse, json, sys, time
from pathlib import Path

# Words we would want to strip in Phase 2. Kept deliberately narrow: these are
# the ones that are always disfluency, never content.
FILLERS = {"um", "uh", "erm", "hmm", "mm", "uhh", "umm", "er", "ah", "eh"}
# Primes the decoder to transcribe verbatim rather than tidy the speech up.
FILLER_PROMPT = "Um, uh, so like, you know, I mean, er, hmm -- verbatim transcript with all filler words included."


def norm(word: str) -> str:
    return word.strip().strip(".,!?;:-—\"'").lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("media")
    ap.add_argument("--model", default="base")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--compute-type", default=None)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--limit", type=float, default=None, help="only transcribe first N seconds")
    ap.add_argument("--prompt-fillers", action="store_true")
    ap.add_argument("--vad", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cuda_setup import enable_cuda
    if args.device in ('cuda', 'auto'):
        print('cuda dll dirs:', enable_cuda())
    from faster_whisper import WhisperModel

    compute = args.compute_type or ("float16" if args.device == "cuda" else "int8")
    print(f"model={args.model} device={args.device} compute={compute} "
          f"threads={args.threads} vad={args.vad} filler_prompt={args.prompt_fillers}", flush=True)

    t0 = time.perf_counter()
    model = WhisperModel(args.model, device=args.device, compute_type=compute,
                         cpu_threads=args.threads)
    load_s = time.perf_counter() - t0
    print(f"model loaded in {load_s:.1f}s", flush=True)

    t0 = time.perf_counter()
    segments, info = model.transcribe(
        args.media,
        language="en",
        beam_size=5,
        word_timestamps=True,
        vad_filter=args.vad,
        condition_on_previous_text=False,
        initial_prompt=FILLER_PROMPT if args.prompt_fillers else None,
    )

    words, filler_hits = [], {}
    last_end = 0.0
    for seg in segments:                       # generator -- work happens here
        if args.limit and seg.start > args.limit:
            break
        for w in (seg.words or []):
            words.append({"t": w.word, "s": round(w.start, 3), "e": round(w.end, 3),
                          "p": round(w.probability, 3)})
            n = norm(w.word)
            if n in FILLERS:
                filler_hits[n] = filler_hits.get(n, 0) + 1
        last_end = seg.end
    elapsed = time.perf_counter() - t0

    audio_s = min(last_end, args.limit) if args.limit else info.duration
    rtf = audio_s / elapsed if elapsed else 0

    print(f"\n--- RESULT ---")
    print(f"audio          {audio_s/60:.1f} min")
    print(f"transcribe     {elapsed:.1f}s  ({rtf:.1f}x realtime)")
    print(f"2-hour podcast would take ~{(7200/rtf)/60:.0f} min at this rate")
    print(f"words          {len(words)}")
    print(f"fillers found  {sum(filler_hits.values())}  {filler_hits or '(NONE)'}")

    # Timestamp sanity: monotonic, non-overlapping, plausible durations.
    bad_order = sum(1 for a, b in zip(words, words[1:]) if b["s"] < a["s"] - 1e-6)
    zero_len = sum(1 for w in words if w["e"] - w["s"] <= 0)
    print(f"out-of-order   {bad_order}")
    print(f"zero-length    {zero_len}")
    print(f"\nfirst 15 words: {' '.join(w['t'].strip() for w in words[:15])}")

    if args.out:
        out = Path(__file__).parent / "out" / f"{args.out}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({
            "config": {"model": args.model, "device": args.device, "compute": compute,
                       "threads": args.threads, "vad": args.vad,
                       "filler_prompt": args.prompt_fillers},
            "timing": {"load_s": round(load_s, 2), "transcribe_s": round(elapsed, 2),
                       "audio_s": round(audio_s, 2), "realtime_factor": round(rtf, 2)},
            "fillers": filler_hits, "word_count": len(words), "words": words,
        }, indent=1), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
