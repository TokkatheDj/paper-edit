# Paper Edit

A local, text-based video editor — edit long-form video and podcasts by editing the
transcript. Everything runs on your own machine: no cloud, no subscription, no account.

![The Paper Edit editor: transcript on the left with deleted words struck through, preview and export on the right](docs/screenshot.png)

*Struck-through words are gone from the video. The preview skips them immediately — nothing
is re-encoded until you export. (Sample audio: [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/), CC BY 4.0.)*

**Status: working end to end** — upload, transcribe, edit the transcript, remove dead
air and filler words, clean up the sound, burn in captions, export for YouTube, Reels or
a square feed.
See `FINDINGS.md` for what was measured, and `ROADMAP.md` for what is built,
what is planned, and what is deliberately left out.

## Using it

On Windows, double-click **`Start Paper Edit.cmd`** (it sets itself up on first run).
Anywhere else, run `python server.py` in the venv. Either way it prints the addresses it
is reachable on:

```
  on this computer     http://localhost:8100
  on your network      http://192.168.x.x:8100
```

Open the network address from a laptop, phone or tablet and the editing happens on the
machine with the GPU while you drive it from anywhere in the house.

Pick a video or audio file and press **Upload & transcribe**. Uploads resume by themselves
if the Wi-Fi drops. When the transcript appears, click a word to jump there, drag to select
a phrase, and press Delete — the video loses exactly that piece. Preview is instant because
nothing is re-encoded until you press Export.

**The host machine has to be awake.** The editor is only there while the server is running.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest tests/ -q
```

## What exists so far

| file | what it does |
|---|---|
| `paperedit/edl.py` | The edit decision list. The transcript *is* the edit — words carry `deleted`, cuts are derived from what survives. |
| `paperedit/audio.py` | Silence detection: snap points for clean cuts, and dead-air removal. Measures the noise floor per file rather than assuming a fixed threshold. |
| `paperedit/enhance.py` | Studio Sound: denoise, de-ess, compress and loudness-normalise to -16 LUFS. Measures SNR and picks a preset, because denoising a clean recording measurably damages it. |
| `paperedit/captions.py` | Animated word-highlight captions as ASS subtitles, burned in at export. |
| `paperedit/render.py` | ffmpeg export (NVENC), proxy generation, filter-graph builder with audio crossfades at joins. |
| `cuda_setup.py` | Registers the venv's bundled CUDA DLLs so `device="cuda"` actually works on Windows. |
| `paperedit/auth.py` | The shared password (scrypt) and server-side sessions. Sign-out revokes; changing the password drops every session. |
| `paperedit/store.py` | SQLite project, word and job state. On disk, so a nightly shutdown mid-transcription shows as interrupted instead of spinning forever. |
| `paperedit/transcribe.py` | Word-level transcription; large-v3 on the GPU when it is free, small on CPU otherwise. |
| `server.py` | FastAPI on :8100 — resumable upload, ingest, edit, export. |
| `static/` | The editor UI. Plain HTML and JS, no build step. |
| `spikes/` | The Phase 0 measurements. Each script prints its own numbers. |
| `tests/` | 42 tests: EDL invariants, sign-in and session revocation, the filler-word endpoint against seeded words, and a full upload-to-export run against real media. |

## Next

Phases 1-3 are built and tested: transcript editing, resumable upload, dead air, filler
words, Studio Sound, animated captions, and per-platform export.

A caveat worth stating plainly, measured in `FINDINGS.md`: **Whisper transcribes only
7-17% of the filler words a person actually hears.** It is trained to produce clean
readable text and tidies most of them away, and no model size or filler-primed prompt
fixes it. So the filler button removes the ones the transcript spells out and says so;
the pauses the rest leave behind are what dead-air removal is for.

Still open, roughly in order: clip and shorts detection, real YouTube/Instagram API
upload, a multitrack timeline, green screen, and local overdub. `ROADMAP.md` has the
detail, including what is deliberately left out and why.

**Sign-in.** One shared password for the household, hashed with scrypt, with server-side
sessions that last 30 days per device and can actually be revoked. The first person to
open it sets the password; `Set Password.cmd` changes it later and signs every device out.

That is sized for its real threat model -- keeping a guest phone or a kid's tablet on the
same wifi out of someone's unreleased videos. It is **not** hardening for the open
internet: there are no accounts, no roles, and it speaks plain http. Run it on a home
network, not on a forwarded port.

## License

MIT — see `LICENSE`. Use it, change it, ship it.
