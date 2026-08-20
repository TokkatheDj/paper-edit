"""Spike 1c -- are the fillers Whisper drops recoverable as timeline gaps?

Whisper suppresses "um"/"uh", but the audio is still there. If a dropped filler
leaves a HOLE in the word timeline (speech energy, no transcribed word), we can
find it acoustically and cut it -- without needing Whisper to spell it.

If instead the filler is absorbed inside a neighbouring word's span, it is
invisible and Phase 2 must fall back to silence removal only.
"""
import json, sys, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).parent
FILLERS = {"um", "uh", "erm", "hmm", "mm", "uhh", "umm", "er", "ah", "eh",
           "mm-hmm", "uh-huh", "hm"}
WINDOW = 300.0


def truth_spans(zip_path, meeting="ES2002a", window=WINDOW):
    """(start, end, word) for every filler a human annotated."""
    z = zipfile.ZipFile(zip_path)
    out = []
    for name in z.namelist():
        if f"words/{meeting}." not in name or not name.endswith(".words.xml"):
            continue
        for w in ET.fromstring(z.read(name)).iter():
            if not w.tag.endswith("w"):
                continue
            s, e = w.get("starttime"), w.get("endtime")
            if s is None or e is None or float(s) > window:
                continue
            t = (w.text or "").strip().lower()
            if t in FILLERS:
                out.append((float(s), float(e), t))
    return sorted(out)


def main():
    truth = truth_spans(HERE / "media" / "ami-annot.zip")
    run = json.loads((HERE / "out" / "lv3_gpu_300.json").read_text(encoding="utf-8"))
    words = run["words"]

    covered = gap = 0
    rows = []
    for s, e, t in truth:
        mid = (s + e) / 2
        # A word whose span contains the filler's midpoint = absorbed/invisible.
        hit = next((w for w in words if w["s"] <= mid <= w["e"]), None)
        if hit:
            covered += 1
            rows.append((s, t, "ABSORBED into", hit["t"].strip()))
        else:
            gap += 1
            # How much room is there between the neighbouring words?
            prev = max((w["e"] for w in words if w["e"] <= s), default=0.0)
            nxt = min((w["s"] for w in words if w["s"] >= e), default=WINDOW)
            rows.append((s, t, "GAP", f"{nxt - prev:.2f}s hole"))

    n = len(truth)
    print(f"Fillers Whisper DROPPED, checked against its own word timeline (n={n})")
    print(f"  recoverable as a timeline gap : {gap:>3}  ({gap/n*100:.0f}%)")
    print(f"  absorbed inside another word  : {covered:>3}  ({covered/n*100:.0f}%)\n")
    print(f"{'time':>8}  {'filler':<8} {'verdict':<14} detail")
    print("-" * 60)
    for s, t, verdict, detail in rows[:20]:
        print(f"{s:8.2f}  {t:<8} {verdict:<14} {detail}")


if __name__ == "__main__":
    sys.exit(main())
