"""Spike 1b -- how many fillers does Whisper actually keep, vs human ground truth?

AMI ships manual word-level transcripts. Counting fillers in the human transcript
for the same 0-300s window tells us Whisper's filler RECALL, which decides whether
Phase 2's "remove ums and ahs" feature can work off the transcript at all.
"""
import json, re, sys, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).parent
FILLERS = {"um", "uh", "erm", "hmm", "mm", "uhh", "umm", "er", "ah", "eh",
           "mm-hmm", "uh-huh", "hm"}
WINDOW = 300.0


def norm(w):
    return w.strip().strip(".,!?;:-—\"'").lower()


def truth_counts(zip_path, meeting="ES2002a", window=WINDOW):
    z = zipfile.ZipFile(zip_path)
    counts, total, timeline = {}, 0, []
    for name in z.namelist():
        if f"words/{meeting}." not in name or not name.endswith(".words.xml"):
            continue
        root = ET.fromstring(z.read(name))
        for w in root.iter():
            if not w.tag.endswith("w"):
                continue
            start = w.get("starttime")
            if start is None or float(start) > window:
                continue
            text = (w.text or "").strip()
            if not text:
                continue
            total += 1
            n = norm(text)
            if n in FILLERS:
                counts[n] = counts.get(n, 0) + 1
                timeline.append((float(start), n))
    return counts, total, sorted(timeline)


def main():
    zip_path = HERE / "media" / "ami-annot.zip"
    truth, truth_words, timeline = truth_counts(zip_path)
    truth_n = sum(truth.values())

    print(f"HUMAN GROUND TRUTH, first {WINDOW:.0f}s of ES2002a")
    print(f"  words       {truth_words}")
    print(f"  fillers     {truth_n}   {truth}")
    print(f"  rate        {truth_n/(WINDOW/60):.1f} fillers/min\n")

    print(f"{'run':<28} {'words':>6} {'fillers':>8} {'recall':>8}")
    print("-" * 54)
    for out in sorted((HERE / "out").glob("*.json")):
        d = json.loads(out.read_text(encoding="utf-8"))
        got = sum(d["fillers"].values())
        print(f"{out.stem:<28} {d['word_count']:>6} {got:>8} "
              f"{got/truth_n*100 if truth_n else 0:>7.0f}%")

    print(f"\nfirst 12 fillers a human heard (time, word):")
    for t, w in timeline[:12]:
        print(f"  {t:7.2f}s  {w}")


if __name__ == "__main__":
    sys.exit(main())
