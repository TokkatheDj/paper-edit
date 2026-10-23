"""Spike 5 -- filler detection scored by TIME against AMI's human word timings (22 Oct 2026).

filler_truth.py counted fillers (how many did Whisper spell out?). That can't tell a
real "um" from a false one. This matches by time: a detected filler counts as FOUND if
a human-annotated filler overlaps it or starts within MATCH_S of it; every other
detection is a FALSE ALARM - and a false alarm is a real word cut out of someone's
video, so precision matters as much as recall.

    .venv\\Scripts\\python.exe spikes\\filler_score.py whisper-large-v3 crisper
    .venv\\Scripts\\python.exe spikes\\filler_score.py crisper --window 300 600

Windows: 0-300 s is the window FINDINGS.md section 3 used. 300-600 s is held out, so a
setting tuned on the first window is checked on audio it never saw.
"""
import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
MEDIA = HERE / "media"
FILLERS = {"um", "uh", "erm", "hmm", "mm", "uhh", "umm", "er", "ah", "eh", "hm"}
# mm-hmm / uh-huh are answers ("yes"), not fillers: an editor must not cut them
MATCH_S = 0.25


def norm(w):
    return re.sub(r"[^a-z\-]", "", (w or "").lower())


def truth(window, meeting="ES2002a"):
    """Human-annotated fillers [(start, end, word)] in the window, all speakers."""
    z = zipfile.ZipFile(MEDIA / "ami-annot.zip")
    out = []
    for name in z.namelist():
        if f"words/{meeting}." not in name or not name.endswith(".words.xml"):
            continue
        for w in ET.fromstring(z.read(name)).iter():
            if not w.tag.endswith("w") or w.get("starttime") is None:
                continue
            s, e = float(w.get("starttime")), float(w.get("endtime") or w.get("starttime"))
            if window[0] <= s < window[1] and norm(w.text) in FILLERS:
                out.append((s, e, norm(w.text)))
    return sorted(out)


def score(found, true):
    """found, true: [(start, end, word)] -> dict with recall, precision and the misses."""
    used = set()
    hits = 0
    false = []
    for fs, fe, fw in found:
        best = None
        for i, (ts, te, tw) in enumerate(true):
            if i in used:
                continue
            overlap = min(fe, te) - max(fs, ts)
            near = abs(fs - ts) <= MATCH_S
            if overlap > 0 or near:
                best = i
                break
        if best is None:
            false.append((fs, fe, fw))
        else:
            used.add(best)
            hits += 1
    missed = [t for i, t in enumerate(true) if i not in used]
    return {"true": len(true), "found": len(found), "hits": hits,
            "recall": hits / len(true) if true else 0.0,
            "precision": hits / len(found) if found else 0.0,
            "false": false, "missed": missed}


def audio(window):
    from faster_whisper.audio import decode_audio
    wav = decode_audio(str(MEDIA / "ami-ES2002a.wav"), sampling_rate=16000)
    return wav[int(window[0] * 16000):int(window[1] * 16000)]


_MODELS = {}


def detect(name, window, beam=1, vad=False, fallback=True):
    """Run one detector over the window -> [(start, end, word)] in meeting time.
    faster-whisper falls back to SAMPLING when unsure, so the same audio gave 68% then 72%
    recall on two runs (22 Oct). Turning the fallback off made it repeatable but it then
    looped ("um um um", precision 11%). So: keep the fallback, fix the random seed."""
    from faster_whisper import WhisperModel
    import ctranslate2
    import cuda_setup
    cuda_setup.enable_cuda()
    ctranslate2.set_random_seed(1234)
    models = {"whisper-large-v3": "large-v3", "crisper": "nyralabs/faster_CrisperWhisper"}
    if name not in _MODELS:
        _MODELS[name] = WhisperModel(models[name], device="cuda", compute_type="float16")
    model = _MODELS[name]
    t0 = time.time()
    segs, _ = model.transcribe(audio(window), language="en", word_timestamps=True,
                               beam_size=beam, vad_filter=vad,
                               temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if fallback else 0.0,
                               condition_on_previous_text=False)
    out = []
    for seg in segs:
        for w in seg.words or []:
            if norm(w.word) in FILLERS:
                out.append((w.start + window[0], w.end + window[0], norm(w.word)))
    return out, time.time() - t0


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("detectors", nargs="+")
    ap.add_argument("--window", nargs=2, type=float, default=[0, 300])
    a = ap.parse_args(argv)
    window = tuple(a.window)
    true = truth(window)
    print(f"HUMAN FILLERS {window[0]:.0f}-{window[1]:.0f}s of ES2002a: {len(true)}\n")
    print(f"{'detector':<18} {'found':>6} {'hits':>5} {'recall':>7} {'precision':>10} {'secs':>6}")
    results = {}
    for name in a.detectors:
        found, secs = detect(name, window)
        r = score(found, true)
        results[name] = r
        print(f"{name:<18} {r['found']:>6} {r['hits']:>5} {r['recall']*100:>6.0f}% "
              f"{r['precision']*100:>9.0f}% {secs:>6.1f}")
    for name, r in results.items():
        print(f"\n{name}: false alarms {[f'{s:.1f}s {w}' for s, e, w in r['false']]}")
        print(f"{name}: missed       {[f'{s:.1f}s {w}' for s, e, w in r['missed']]}")
    out = HERE / "out" / ("filler_score_%d_%d.json" % window)
    out.write_text(json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
                              indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
