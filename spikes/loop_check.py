"""Spike 8 -- does the verbatim model loop, and does a no-repeat guard stop it without
costing fillers? (23 Oct 2026)

Found while fixing test_api: CrisperWhisper (beam 5, VAD, no temperature fallback) wrote
"So, we're. So, we're. ..." dozens of times inside half a second at ~100.5 s of the test
video, each "word" 10 ms long. A LOOP here = a word under 30 ms that repeats one of the
previous 4 words; real stutters ("I I think") last longer than that.

    .venv\\Scripts\\python.exe spikes\\loop_check.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "spikes"))
import cuda_setup                                            # noqa: E402
cuda_setup.enable_cuda()
from faster_whisper import WhisperModel                      # noqa: E402
from faster_whisper.audio import decode_audio                # noqa: E402
import filler_score as fs                                    # noqa: E402

MODEL = WhisperModel("nyralabs/faster_CrisperWhisper", device="cuda", compute_type="float16")
AMI = decode_audio(str(fs.MEDIA / "ami-ES2002a.wav"), sampling_rate=16000)
TEST = decode_audio(str(fs.MEDIA / "test-300s.mp4"), sampling_rate=16000)


def run(audio, offset, **kw):
    segs, _ = MODEL.transcribe(audio, language="en", beam_size=5, word_timestamps=True, vad_filter=True,
                               temperature=0.0, condition_on_previous_text=False, **kw)
    return [(w.start + offset, w.end + offset, w.word) for s in segs for w in (s.words or [])]


def loops(words):
    n = 0
    for i, (s, e, t) in enumerate(words):
        prev = {fs.norm(x[2]) for x in words[max(0, i - 4):i]}
        if e - s < 0.03 and fs.norm(t) in prev:
            n += 1
    return n


SETTINGS = {"no guard": {}, "no_repeat_ngram 3": {"no_repeat_ngram_size": 3},
            "no_repeat_ngram 4": {"no_repeat_ngram_size": 4},
            "repetition_penalty 1.1": {"repetition_penalty": 1.1}}
for name, kw in SETTINGS.items():
    tw = run(TEST, 0.0, **kw)
    line = ["%-24s test video: %4d words, %3d looped" % (name, len(tw), loops(tw))]
    for w in ((0, 300), (300, 600)):
        aw = run(AMI[int(w[0] * 16000):int(w[1] * 16000)], w[0], **kw)
        found = [(s, e, fs.norm(t)) for s, e, t in aw if fs.norm(t) in fs.FILLERS]
        r = fs.score(found, fs.truth(w))
        line.append("AMI %d-%d: %d/%d found, %.0f%% real, %d looped" % (
            w[0], w[1], r["hits"], r["true"], r["precision"] * 100, loops(aw)))
    again = run(TEST, 0.0, **kw)
    line.append("repeatable: %s" % (again == tw))
    print(" | ".join(line), flush=True)
