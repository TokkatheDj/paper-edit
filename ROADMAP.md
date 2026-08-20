# Roadmap

What this project is trying to be, what is built, and what is deliberately not.

## The idea

Edit long-form video and podcasts by editing the transcript. Delete a sentence in the
text and the video loses exactly that sentence. Tools like Descript popularised this;
the goal here is a version you own outright, that runs on your own hardware, with no
subscription and no upload to anyone else's servers.

The core of it is smaller than it looks: word-level timestamps from Whisper, an edit
decision list derived from which words survive, and ffmpeg. Most of the rest is plumbing
around those three.

## Feature status

### Built (Phase 1)

| Feature | How |
|---|---|
| **Transcript editing** | Whisper word-level timestamps, delete words, derive cuts, render. The heart of it. |
| **Resumable upload** | Chunked upload that survives a dropped connection. Source files are often several GB. |
| **Instant preview** | The player seeks over deleted ranges instead of re-encoding. Nothing is rendered until export. |
| **Export + platform presets** | NVENC-accelerated render at source resolution, plus 16:9 / 9:16 / 1:1 / audio-only presets. |

### Planned

| Feature | Approach | Phase |
|---|---|---|
| **Silence removal** | VAD or ffmpeg `silencedetect`; trim pauses over a threshold. Reliable, already half-built in `paperedit/audio.py`. | 2 |
| **Filler word removal** | See the honest caveat below — this cannot be done from the transcript alone. | 2 |
| **Studio-quality audio** | DeepFilterNet (denoise + dereverb) then a mastering chain: highpass, de-ess, compress, `loudnorm` to -16 LUFS. | 2 |
| **Animated captions** | Word timestamps to generated `.ass` subtitles with karaoke highlighting, burned in with ffmpeg. | 3 |
| **Speaker labels** | pyannote is the strong option but needs a HuggingFace token and licence acceptance; a simpler turn-detector may be enough. Spike first. | 3 |
| **Clip detection** | Feed the timestamped transcript to a local LLM and ask for the most engaging segments. Free if you already run one locally. | 4 |
| **Shorts export** | Auto-reframe to 9:16 / 1:1 with captions burned in. | 4 |
| **Multitrack timeline** | Non-destructive timeline over the same edit decision list, with volume keyframes and crossfades. | 5 |
| **Stock and B-roll** | Pexels and Pixabay free APIs. | 5 |
| **Green screen** | RobustVideoMatting (ONNX) runs comfortably on a mid-range GPU. | 6 |
| **Voice cloning / overdub** | Local F5-TTS or Chatterbox. Should be gated on recorded consent, and limited to the speaker's own voice. | 6 |

### Deliberately not built

| Feature | Why | What to do instead |
|---|---|---|
| **AI eye contact** | The only good implementation is NVIDIA's Maxine model. The Windows SDK is a C++/TensorRT build with licensing friction; the hosted version is paid. Not a practical local post-process. | NVIDIA Broadcast does it live while recording, free, on any RTX card. Fix it at record time. |
| **Real-time collaborative editing** | CRDT/OT editing over video is a multi-month subsystem plus hosting cost, which contradicts the no-subscription goal. | A review page with timestamped comments that come back into the editor as markers. Covers the actual need — feedback — without the machinery. |
| **Direct publish to Instagram / Facebook** | Requires a Business or Creator account, a linked Page, app review, and a publicly reachable URL for the file. | Export with the right preset and open the download on your phone. It lands in the camera roll; post natively. This is also the fastest route in practice. |
| **Direct publish to YouTube** | Possible via the Data API, but until an app passes Google verification its uploads are locked to private. | Same handoff, or build it later accepting that videos arrive as private drafts. |

## The two pieces that are actually load-bearing

Everything else is plumbing around ffmpeg. These two decide whether the tool feels good:

**1. The transcript *is* the edit.** One structure is the source of truth: an ordered list
of words, each with a source time span and a `deleted` flag. The cut list is *derived* from
the words that survive, never stored separately. Silence removal, filler removal and the
future timeline are all just different writers against that one structure. Get it right and
every later feature is a view over it; get it wrong and they fight each other.

Cuts happen where text was **deleted**, never where the speaker merely paused. An earlier
version split on any time gap between surviving words, which silently stripped every natural
pause before the user had edited anything — a 5:00 recording came back as 2:56 untouched.
Removing silence is a feature someone asks for, not something an editor does behind their back.

**2. Preview without rendering.** The player never re-encodes. It plays a 720p proxy and the
browser seeks *over* deleted ranges: on reaching the end of a kept segment it jumps to the
start of the next. Only export runs ffmpeg. This is the difference between editing that
feels instant and editing that makes you wait forty seconds to hear a cut.

## Honest caveat: filler words

"Remove all the ums with one click" is the feature people most want, and it **cannot be
built from the transcript alone**. Measured against human-annotated ground truth, Whisper
transcribes only **7-17%** of the fillers a person actually hears — it is trained to produce
clean readable text and tidies disfluencies away. Neither a larger model nor a
filler-primed prompt fixes it. Of the ones it drops, roughly 43% leave a hole in the word
timeline (recoverable acoustically) and 57% are absorbed into a neighbouring word's span
(invisible to any transcript-based method).

So filler removal has to be built as: silence and pause removal first, acoustic detection
over the gaps second, and the transcript only as a supplement. `FINDINGS.md` has the
numbers and the method.

## Constraints worth knowing

- **Python 3.14 runs the whole stack.** faster-whisper, ctranslate2 and onnxruntime all ship
  3.14 wheels. No older interpreter needed.
- **CUDA on Windows needs a nudge.** ctranslate2 loads `cublas64_12.dll` and `cudnn64_9.dll`
  by name, and pip installs them somewhere Windows does not search, so `device="cuda"`
  constructs fine and then fails at first inference. `cuda_setup.enable_cuda()` handles it.
- **The GPU may be shared.** If you run a local LLM, it can hold most of your VRAM. The
  transcriber checks free VRAM and falls back to CPU rather than crashing — and CPU
  transcription is still ~19x realtime, so the fallback is not painful.
- **Long jobs must survive a hard power-off.** Queue state lives in SQLite and an interrupted
  job is reported as interrupted on the next start.

## Verification

- Spikes under `spikes/` each print their own measurements; `FINDINGS.md` records them.
- `tests/test_edl.py` covers the cut derivation — the function where an off-by-one silently
  desyncs a two-hour podcast.
- `tests/test_api.py` runs real media through upload, transcription, editing and export, and
  asserts on the result with `ffprobe`, including that audio and video stay in sync.
- The test that matters most is not automated: edit something you actually intend to
  publish, and see what breaks.
