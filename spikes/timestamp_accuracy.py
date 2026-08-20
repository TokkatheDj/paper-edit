"""Spike 1d -- how accurate are Whisper's word timestamps, really?

This is the load-bearing measurement for the whole app. "Delete a word, the video
cuts there" is only as good as the word's start/end time. AMI gives human word
boundaries for the same audio, so we can align and measure the error directly.

NOTE this is a WORST CASE: 4-person meeting audio with overlapping speech and
crosstalk. A solo podcast should do better.
"""
import json, statistics, sys, zipfile
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).parent
WINDOW = 300.0


def norm(w):
    return "".join(c for c in w.lower() if c.isalnum() or c == "'")


def truth_words(zip_path, meeting="ES2002a", window=WINDOW):
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
            t = norm(w.text or "")
            if t:
                out.append({"t": t, "s": float(s), "e": float(e)})
    out.sort(key=lambda d: d["s"])
    return out


def report(label, path, truth):
    words = [w for w in json.loads(Path(path).read_text(encoding="utf-8"))["words"]]
    hyp = [{"t": norm(w["t"]), "s": w["s"], "e": w["e"]} for w in words if norm(w["t"])]

    sm = SequenceMatcher(None, [w["t"] for w in truth], [w["t"] for w in hyp],
                         autojunk=False)
    start_err, end_err, dur_ratio = [], [], []
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            t, h = truth[a + k], hyp[b + k]
            start_err.append(abs(h["s"] - t["s"]))
            end_err.append(abs(h["e"] - t["e"]))
            td = t["e"] - t["s"]
            if td > 0.05:
                dur_ratio.append((h["e"] - h["s"]) / td)

    if not start_err:
        print(f"{label}: no aligned words"); return
    n = len(start_err)
    se = sorted(start_err)
    print(f"\n{label}  ({n} words aligned of {len(truth)} truth / {len(hyp)} hyp)")
    print(f"  start error   median {statistics.median(se)*1000:6.0f} ms   "
          f"p90 {se[int(n*0.9)-1]*1000:6.0f} ms   max {se[-1]*1000:6.0f} ms")
    print(f"  end error     median {statistics.median(end_err)*1000:6.0f} ms")
    print(f"  within 100ms  {sum(1 for x in se if x <= 0.10)/n*100:5.0f}%"
          f"      within 250ms {sum(1 for x in se if x <= 0.25)/n*100:5.0f}%")
    if dur_ratio:
        print(f"  span width    median {statistics.median(dur_ratio):.2f}x human "
              f"(>1 means Whisper's word spans are stretched)")


def main():
    truth = truth_words(HERE / "media" / "ami-annot.zip")
    print(f"human truth: {len(truth)} words in first {WINDOW:.0f}s")
    for label, f in [("large-v3 GPU", "lv3_gpu_300.json"),
                     ("small CPU  ", "small_cpu_300_prompt.json"),
                     ("base CPU   ", "base_cpu_300.json")]:
        p = HERE / "out" / f
        if p.exists():
            report(label, p, truth)


if __name__ == "__main__":
    sys.exit(main())
