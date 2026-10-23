# Phase 0 findings

Measured on the desktop (RTX 3080 10 GB, 32-thread CPU, ffmpeg 8.1.2), 19 Aug 2026.
Test material is the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) ES2002a
(CC-BY): 21 minutes of unscripted 4-person meeting audio with human word-level
transcripts. Spontaneous speech with real disfluencies, and a **worst case** — overlapping
crosstalk is harder than the solo podcast this app is actually for.

Reproduce: `python spikes/<name>.py`. Raw output lands in `spikes/out/`.

---

## 1. Python 3.14 is fine — the planned 3.12 install is not needed

The plan assumed the desktop's Python 3.14 was too new for the ML stack. It isn't:
`faster-whisper 1.2.1`, `ctranslate2 4.8.1` and `onnxruntime 1.29.0` all ship 3.14 wheels
and installed in 28 seconds. **No second Python install, no interpreter juggling.**

CUDA needs one extra step: ctranslate2 loads `cublas64_12.dll` / `cudnn64_9.dll` by name,
and pip puts them somewhere Windows does not search. `device="cuda"` *constructs* fine and
then fails at first inference. `cuda_setup.enable_cuda()` fixes it by registering the DLL
directories — call it before importing `faster_whisper`.

## 2. Transcription is fast enough that the GPU is optional

| model | device | speed | a 2-hour podcast takes |
|---|---|---|---|
| base | CPU, 32 threads | 19.4x realtime | ~6 min |
| small | CPU, 32 threads | 8.4x realtime | ~14 min |
| large-v3 | GPU (float16) | **30.2x realtime** | **~4 min** |

**This resolves the GPU-contention worry in the plan.** Even on CPU, transcription is far
faster than realtime, so it never has to compete with other local AI tools for the GPU.
Use the GPU when it is free, fall back to CPU with no drama.

## 3. Whisper hides filler words — the headline Phase 2 feature needs rethinking

Humans annotated **30 fillers** in the first 5 minutes (6.0/min). Whisper found:

| run | fillers found | recall |
|---|---|---|
| base, CPU | 2 | 7% |
| base + filler-primed prompt | 1 | 3% |
| small + filler-primed prompt | 5 | 17% |
| large-v3, GPU | 4 | 13% |
| large-v3 + filler-primed prompt | 4 | 13% |

**Whisper catches 7-17% of fillers.** Not a model-size problem, and not fixable by priming
the prompt with fillers — Whisper is trained to produce clean readable text and tidies
disfluencies away.

Checking where the missing ones went (`filler_gaps.py`): 43% leave a hole in the word
timeline (recoverable acoustically), 57% are absorbed inside a neighbouring word's span
(invisible to any transcript-based method).

**Consequence:** "remove all the ums" cannot be built from the transcript alone. Options
for Phase 2, in order of preference:

1. **Silence/pause removal does most of the perceived work** — reliable, already
   implemented in `paperedit/audio.py`, and needs no cooperation from the ASR.
2. Acoustic filler detection over the gaps Whisper leaves (reaches ~43% of the rest).
3. A verbatim ASR (wav2vec2 / Parakeet CTC) as a second pass purely for disfluencies.

`mark_fillers()` handles the 7-17% Whisper *does* spell out, and its docstring says plainly
that it is a supplement, not the whole feature.

## 4. Word timestamps are good enough — if cuts snap to silence

Aligned against human word boundaries:

| model | median start error | within 250 ms |
|---|---|---|
| small, CPU | 50 ms | 80% |
| base, CPU | 60 ms | 79% |
| large-v3, GPU | 130 ms | 72% |

Median 50-130 ms is small but audible if you cut exactly on a word boundary — it clips
consonants. **This makes the plan's "snap cuts to the nearest silence" mandatory rather
than a refinement.** Implemented in `audio.snap_points()` + `edl.derive_cuts(snap_points=)`.

*Caveat:* the p90 and max figures from this script are inflated by the text-alignment step
matching repeated words to the wrong occurrence. Treat the medians as sound and the tail as
unmeasured.

## 5. The cut engine works, and it is fast

30 cuts on a 300 s clip, rendered on NVENC:

- **56-59x realtime** — a 2-hour edit exports in about 2 minutes.
- **No clicks.** Peak sample-to-sample jump at the joins measured *lower* than elsewhere in
  the file (0.011 vs 0.045). The 20 ms crossfade in `render.build_filtergraph` works.
- **No A/V drift.** Skew between video and audio streams stayed at **0-4 ms across 10, 50,
  120 and 250 cuts** — well under one frame, and it does not grow with cut count.

One real but cosmetic quirk: total output runs ~0.15% longer than the arithmetic sum of the
kept ranges (~6 ms per cut), because ffmpeg's video `trim` includes the partial frame at
each boundary. Frame-quantising the boundaries did **not** remove it (tested — the first
hypothesis was wrong). Since audio and video stay locked to each other, this is a
length-*prediction* offset, not a sync defect: report export duration from `ffprobe` rather
than trusting the arithmetic.

Caveat found later, while adding filler removal: the 0-4 ms figure above is from
**synthetic, evenly spaced** cuts. With real transcript-derived cuts the video/audio
duration difference is usually a few ms but has been measured at **67 ms** (~2 frames at
30 fps) on some cut sets, which trips the `< 0.05` assertion in
`tests/test_api.py::test_export_matches_the_plan`. The test is therefore **intermittent
when Whisper runs on CPU**: multi-threaded float reduction makes the transcript vary
slightly between runs, which moves the cut boundaries. Observed failing twice and passing
twice on identical code. It is a difference in total stream length, not progressive
desync -- `spikes/drift.py` shows the offset is a constant tail, not per-cut -- so it is
not audible or visible in the export. Left as-is rather than widening the threshold,
because the threshold is the only thing watching for real desync.

## 6. Decisions this locks in

- Skip the Python 3.12 install. Use `.venv` on 3.14.
- Transcribe with **large-v3 on GPU when free, small on CPU otherwise**. `small` beats
  `base` on CPU for both timestamps and filler recall, at acceptable speed.
- Always snap cuts to silence. Always crossfade audio at joins.
- Rebuild Phase 2's filler feature around silence removal first, acoustic detection second,
  and set expectations accordingly. *(Superseded Oct 2026: a verbatim model does better
  than acoustic gap detection - see section 5.)*
- Keep the EDL derivation under test — the frame-quantisation ordering bug caught by
  `tests/test_edl.py::test_frame_quantisation_lands_on_the_grid` was invisible to
  inspection.

## 5. A verbatim model finds fillers Whisper hides (22 Oct 2026)

`spikes/filler_score.py` scores filler detection by TIME against AMI's hand-timed words
(a detection is a hit if it overlaps a human-marked filler or starts within 0.25 s; every
other detection is a false alarm - a real word cut). mm-hmm / uh-huh are answers, not
fillers, and are excluded. Windows: 0-300 s (25 fillers) and a held-out 300-600 s (51).

| detector (faster-whisper, GPU) | 0-300 s recall / precision | 300-600 s | repeatable? |
|---|---|---|---|
| Whisper large-v3 | 0% / - | 0% / - | yes |
| CrisperWhisper, default retries | 60-68% / 55-67% | 41-57% / 65-73% | **no** |
| CrisperWhisper, beam 5, VAD, retries | 72-76% / 86% | 41-51% / 65-81% | **no** |
| CrisperWhisper, no retries, no VAD | 56-64% / 11-39% | 20-24% / 16-62% | yes (loops: "um um um") |
| **CrisperWhisper, beam 5, VAD, no retries** | **64% / 67%** | **43% / 96%** | **yes** |

Model: `nyralabs/faster_CrisperWhisper` (CTranslate2 conversion, 3.09 GB, CC BY-NC 4.0 -
personal use). It writes fillers as [UH] / [UM].

- faster-whisper's temperature fallback SAMPLES; `ctranslate2.set_random_seed` did not make
  runs repeat on the GPU. An editor must give the same cuts twice, so no retries - and VAD
  (silence skipped) is what stops the no-retry loops.
- Chosen: beam 5 + VAD + no retries. Over both windows: 38 of 76 fillers found (50%),
  38 of 47 detections real (81%).
- **Caveats:** CrisperWhisper was trained partly on AMI, so this benchmark may flatter it.
  And this is the HARD case - a four-person meeting on one mix, with overlapping talk
  (most 300-600 s misses are during overlaps). One person on a good mic - what Paper Edit is for - is
  the real test, and it must be a recording the model never trained on.
- Timing: the faster version's word timings are looser than the original's (the model
  card says so). Allowing 1.6 s of slack only lifts recall 53% -> 57%, so timing is not
  the main loss - but cuts must still snap to silence (section 4).

### 5b. ...and without a guard it LOOPS - fixed with no_repeat_ngram_size=4 (23 Oct 2026)

Found by test_api failing after the switch: around 100.5 s of the test video the verbatim
model wrote "So, we're. So, we're. ..." dozens of times inside half a second, each "word"
10 ms long. `spikes/loop_check.py` (a loop = a word under 30 ms repeating one of the
previous four):

| setting (beam 5, VAD, no fallback) | test video loops | AMI 0-300 found / real | AMI 300-600 | repeatable |
|---|---|---|---|---|
| no guard | **324 of 545 words** | 16/25, 67% | 22/51, 96% | yes |
| **no_repeat_ngram_size=4** | **0** | **20/25, 83%** | **33/51, 75%** | **yes** |
| no_repeat_ngram_size=3 | 0 | 10/25, 67% | 17/51, 74% | yes |
| repetition_penalty=1.1 | 71 | 17/25, 81% | 16/51, 64% | yes |

The guard removes the loops AND finds more fillers - 53 of 76 (70%), up from 38 - because
the loops were crowding out real words. A real 10-minute single-speaker talk: 1563 words -> 1424 (about 140
of the "extra" words were loop garbage, not real repeats) and 11 fillers, 5 of them in the
last second of the recording (to be judged by ear: spikes/listen_sheet.py).

The three test_api export failures were this, not an old problem: the AMI test video is
silent until ~40 s (large-v3 had invented 2 words at 10-25 s that the tests relied on;
they now use 45-60 s), and the short-clip test's "deletion" was four loop words 10 ms long,
so it made no cut. 112/112 tests pass.
