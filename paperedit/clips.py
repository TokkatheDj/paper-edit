"""Clip suggestions: ask a local LLM for the moments that would stand alone as a short.

The model only proposes. It reads the surviving transcript as timestamped lines
and names a start, an end, a title and a reason; everything after that is done
here, in code, because models are sloppy with times. They cite a line's start as
the clip's end, run 20 s or 3 min when asked for 30-60, overlap each other, and
occasionally point past the end of the talk. repair_clips() treats every time as
a hint: it snaps to word boundaries, re-fits the length, and drops what cannot
be fixed, tagging each change so it can be counted.

Boundaries come in tiers, best first:

    sentence   the previous word ends in . ? !   (or this one does, for an end)
    clause     , ; :
    pause      at least PAUSE seconds of silence between the words
    word       any word boundary -- the fallback

Measured on four real ~10-minute talks, Whisper wrote only 24-38 sentence ends
per ~1,400 words and some "sentences" ran over two minutes. So a clip cannot
always start and end on a full stop; each clip says which tier it landed on
rather than pretending.

Deleted words: a clip's ends are always kept words, and its length is the KEPT
time -- what the player and an export would actually run -- not the wall-clock
span. Deleted words inside a clip are fine (the edit skips them); a model clip
that sits mostly in deleted text is dropped.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Sequence

OLLAMA_URL = os.environ.get("PAPEREDIT_OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("PAPEREDIT_CLIP_MODEL", "gemma3:12b")
TIMEOUT = float(os.environ.get("PAPEREDIT_OLLAMA_TIMEOUT", "600"))

MIN_S, MAX_S = 30.0, 60.0
PAUSE = 0.5                  # a gap this long reads as a breath, i.e. a boundary
SEGMENT_WORDS = 25           # force a timestamp at least this often for the model

# Context. Ollama's default window is small and it drops the FRONT of an
# over-long prompt without telling the client, so this is always set explicitly
# and the transcript is split into parts well before it would overflow.
NUM_CTX = 16384
NUM_PREDICT = 1500
PROMPT_TOKEN_BUDGET = 9000   # transcript tokens per request; the rest is headroom

TIERS = ("sentence", "clause", "pause", "word")
_SENT = ".?!"
_CLAUSE = ",;:"


class ClipsError(Exception):
    """A failure worth showing the person as-is: Ollama down, model missing."""


# --------------------------------------------------------------------------- words

@dataclass(slots=True)
class _W:
    idx: int
    text: str
    start: float
    end: float


def _kept(words: Sequence[dict]) -> list[_W]:
    return [_W(int(w["idx"]), str(w["text"]).strip(), float(w["start"]), float(w["end"]))
            for w in words if not w.get("deleted")]


def _tail(text: str) -> str:
    """The punctuation a word ends on, ignoring closing quotes and brackets."""
    t = text.rstrip("\"')]”’")
    return t[-1:] if t else ""


def fmt_time(t: float) -> str:
    t = max(0, int(t))
    h, m, s = t // 3600, t // 60 % 60, t % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parse_time(v: Any) -> float | None:
    """Seconds from whatever the model wrote: 192, 192.5, "192", "3:12",
    "03:12.4", "0:03:12", "[03:12]", or "03:12-03:45" (the first one wins)."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if v == v and v >= 0 else None
    m = re.search(r"\d+(?::\d+){0,2}(?:\.\d+)?", str(v))
    if not m:
        return None
    parts = m.group(0).split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    secs = 0.0
    for n in nums:
        secs = secs * 60 + n
    return secs


# --------------------------------------------------------------------------- prompt

def segments(words: Sequence[dict]) -> list[tuple[float, str]]:
    """Group the kept words into short timestamped lines for the model.

    A line ends at a sentence end, at a pause, or after SEGMENT_WORDS words
    (preferring a comma past the halfway mark) -- a two-minute run-on with one
    timestamp would leave the model nothing to cite.
    """
    kept = _kept(words)
    out: list[tuple[float, str]] = []
    cur: list[_W] = []
    for i, w in enumerate(kept):
        cur.append(w)
        nxt = kept[i + 1] if i + 1 < len(kept) else None
        gap = (nxt.start - w.end) if nxt else 0.0
        end_here = (nxt is None or _tail(w.text) in _SENT or gap >= PAUSE
                    or len(cur) >= SEGMENT_WORDS
                    or (len(cur) >= SEGMENT_WORDS // 2 and _tail(w.text) in _CLAUSE))
        if end_here:
            out.append((cur[0].start, " ".join(x.text for x in cur)))
            cur = []
    return out


def _lines(segs: Sequence[tuple[float, str]]) -> str:
    return "\n".join(f"[{fmt_time(t)}] {s}" for t, s in segs)


def estimate_tokens(text: str) -> int:
    return len(text) // 3 + 1          # deliberately pessimistic for English


def chunk_segments(segs: Sequence[tuple[float, str]], budget: int = PROMPT_TOKEN_BUDGET,
                   overlap_s: float = 60.0) -> list[list[tuple[float, str]]]:
    """Split a long transcript into parts that each fit the context.

    Parts overlap by about a minute so a moment that straddles a seam is whole
    in at least one of them; the duplicates this creates are removed later as
    overlaps.
    """
    if estimate_tokens(_lines(segs)) <= budget:
        return [list(segs)]
    parts: list[list[tuple[float, str]]] = []
    i = 0
    while i < len(segs):
        part: list[tuple[float, str]] = []
        used = 0
        j = i
        while j < len(segs):
            cost = estimate_tokens(_lines([segs[j]])) + 1
            if part and used + cost > budget:
                break
            part.append(segs[j])
            used += cost
            j += 1
        parts.append(part)
        if j >= len(segs):
            break
        # Step back about overlap_s for the next part, but always move forward.
        back = j
        while back - 1 > i + 1 and part[-1][0] - segs[back - 1][0] < overlap_s:
            back -= 1
        i = max(back, i + 1)
    return parts


SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "title": {"type": "string"},
                    "reason": {"type": "string"},
                    "score": {"type": "integer"},
                },
                "required": ["start", "end", "title", "reason", "score"],
            },
        },
    },
    "required": ["clips"],
}


def build_prompt(segs: Sequence[tuple[float, str]], n: int = 3,
                 part: tuple[int, int] | None = None) -> str:
    where = ""
    if part and part[1] > 1:
        where = (f"This is part {part[0]} of {part[1]} of a longer recording; "
                 "choose only from this part.\n\n")
    return (
        "Below is the transcript of a talk. Each line starts with the time it is "
        "spoken, as [mm:ss].\n\n"
        f"{where}"
        f"Pick the {n} best moments to post on their own as a short vertical video. "
        "Each one must:\n"
        f"- last between {int(MIN_S)} and {int(MAX_S)} seconds\n"
        "- open with a strong line that makes sense without anything before it\n"
        "- make one idea, and end where that idea naturally finishes\n"
        "- not overlap any other moment you pick\n\n"
        "Answer in JSON only: {\"clips\": [{\"start\", \"end\", \"title\", \"reason\", "
        "\"score\"}]}.\n"
        "- start: the [mm:ss] time of the line the moment begins on, copied exactly\n"
        "- end: the [mm:ss] time of the line AFTER the moment's last line\n"
        "- title: a short title, at most 8 words\n"
        "- reason: one sentence on why it works as a short\n"
        "- score: 1-10, how strong it is\n\n"
        "Transcript:\n" + _lines(segs) + "\n")


# --------------------------------------------------------------------------- model

def ask_model(prompt: str, model: str | None = None, url: str | None = None,
              timeout: float | None = None) -> tuple[str, dict]:
    """One non-streamed chat call to Ollama with structured output.

    Returns (the reply text, the timing/token counts Ollama reports). Raises
    ClipsError with something a person can act on when Ollama is not there.
    """
    model = model or MODEL
    url = (url or OLLAMA_URL).rstrip("/")
    body = {
        "model": model, "stream": False, "format": SCHEMA,
        "think": False,          # qwen3 would otherwise spend the budget thinking
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.2, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
    }
    req = urllib.request.Request(url + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout or TIMEOUT) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:200]
        if e.code == 404:
            raise ClipsError(f"the model {model} is not installed in Ollama -- "
                             f"run: ollama pull {model}") from None
        raise ClipsError(f"Ollama refused the request ({e.code}): {detail}") from None
    except (urllib.error.URLError, ConnectionError) as e:
        raise ClipsError(f"Ollama is not running at {url} -- start it and try again "
                         f"({getattr(e, 'reason', e)})") from None
    except TimeoutError:
        raise ClipsError(f"Ollama took longer than {int(timeout or TIMEOUT)} s") from None
    except json.JSONDecodeError:
        raise ClipsError("Ollama answered with something that is not JSON") from None
    meta = {k: data.get(k) for k in ("total_duration", "load_duration", "prompt_eval_count",
                                     "eval_count", "done_reason")}
    return (data.get("message") or {}).get("content", ""), meta


def parse_reply(text: Any) -> list[dict]:
    """The model's clips, as a list of dicts, from whatever came back.

    Structured output makes garbage rare, not impossible: a reply cut off at
    num_predict, a bare list, a code fence. Anything unreadable is [] -- the
    caller reports "no usable suggestions" rather than crashing.
    """
    if isinstance(text, (list, dict)):
        data = text
    else:
        s = str(text or "").strip()
        s = re.sub(r"^```(?:json)?|```$", "", s).strip()
        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}|\[.*\]", s, re.S)
            try:
                data = json.loads(m.group(0)) if m else None
            except json.JSONDecodeError:
                data = None
    if isinstance(data, dict):
        data = data.get("clips", data.get("moments", []))
    if not isinstance(data, list):
        return []
    return [c for c in data if isinstance(c, dict)]


# --------------------------------------------------------------------------- repair

class _Index:
    """Precomputed boundary facts over the kept words, so the search is cheap."""

    def __init__(self, words: Sequence[dict]):
        allw = sorted(({"idx": int(w["idx"]), "start": float(w["start"]),
                        "end": float(w["end"]), "deleted": bool(w.get("deleted"))}
                       for w in words), key=lambda w: w["idx"])
        self.kept = _kept(words)
        self.kept.sort(key=lambda w: w.idx)
        pos = {w["idx"]: i for i, w in enumerate(allw)}
        k = self.kept
        n = len(k)
        # Lead-in and tail: a little of the silence either side of the clip, so
        # the first consonant is not clipped (Whisper's word times are off by
        # 50-130 ms) and the last word gets its natural decay.
        self.lead = []
        self.tail = []
        for w in k:
            p = pos[w.idx]
            before = allw[p - 1]["end"] if p > 0 else 0.0
            after = allw[p + 1]["start"] if p + 1 < len(allw) else w.end + 0.4
            self.lead.append(min(0.15, max(0.0, w.start - before) / 2))
            self.tail.append(min(0.4, max(0.0, after - w.end)))
        # Time removed by deletions between consecutive kept words, as a prefix
        # sum, so a clip's KEPT duration is O(1).
        self.cut = [0.0] * n
        for i in range(1, n):
            skipped = k[i].idx != k[i - 1].idx + 1
            self.cut[i] = self.cut[i - 1] + (max(0.0, k[i].start - k[i - 1].end) if skipped else 0.0)
        # Tier of starting AT word i, and of ending AT word i (0 best .. 3 word).
        self.start_tier = [0] * n
        self.end_tier = [0] * n
        for i in range(n):
            self.start_tier[i] = 0 if i == 0 else self._tier(k[i - 1], k[i])
            self.end_tier[i] = 0 if i == n - 1 else self._tier(k[i], k[i + 1])

    @staticmethod
    def _tier(a: _W, b: _W) -> int:
        """How good a cut between word a and word b is."""
        p = _tail(a.text)
        if p in _SENT:
            return 0
        if p in _CLAUSE:
            return 1
        if b.start - a.end >= PAUSE or b.idx != a.idx + 1:
            return 2                     # a pause, or a cut the edit already makes
        return 3

    def t_start(self, i: int) -> float:
        return max(0.0, self.kept[i].start - self.lead[i])

    def t_end(self, j: int) -> float:
        return self.kept[j].end + self.tail[j]

    def duration(self, i: int, j: int) -> float:
        return self.t_end(j) - self.t_start(i) - (self.cut[j] - self.cut[i])

    def word_at(self, t: float) -> int:
        """The kept word nearest to time t."""
        k = self.kept
        lo, hi = 0, len(k) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if k[mid].end < t:
                lo = mid + 1
            else:
                hi = mid
        if lo > 0 and abs(k[lo - 1].end - t) < abs(k[lo].start - t):
            return lo - 1
        return lo


def _deleted_share(words: Sequence[dict], a: float, b: float) -> float:
    inside = [w for w in words if float(w["end"]) > a and float(w["start"]) < b]
    if not inside:
        return 1.0
    return sum(1 for w in inside if w.get("deleted")) / len(inside)


def _fit(ix: _Index, s: float, e: float, taken: list[tuple[float, float]],
         min_s: float, max_s: float) -> tuple[int, int] | None:
    """The best (first kept word, last kept word) for a model's (s, e).

    Searches start words within ~20 s before / 10 s after the model's start and
    every end that makes the clip min_s..max_s long. The cost prefers better
    boundary tiers, then staying close to what the model said -- moving the
    start LATER costs double, because that is what cuts off an opening line.
    """
    k = ix.kept
    i0 = ix.word_at(s)
    best, best_cost = None, float("inf")
    i = i0
    while i > 0 and k[i - 1].start >= s - 20:
        i -= 1
    while i < len(k) and k[i].start <= s + 10:
        ts = ix.t_start(i)
        ds = ts - s
        cost_s = 10 * ix.start_tier[i] + (2 * ds if ds > 0 else -ds)
        j = i
        while j < len(k):
            d = ix.duration(i, j)
            if d > max_s:
                break
            if d >= min_s:
                te = ix.t_end(j)
                if not any(te > a and ts < b for a, b in taken):
                    cost = cost_s + 10 * ix.end_tier[j] + 0.5 * abs(te - e)
                    if cost < best_cost:
                        best, best_cost = (i, j), cost
            j += 1
        i += 1
    return best


def repair_clips(words: Sequence[dict], raw: Any, n: int = 3,
                 min_s: float = MIN_S, max_s: float = MAX_S) -> dict:
    """Turn the model's suggestions into clips that are safe to play and cut.

    Deterministic, and the one place model output is trusted -- or not. Takes
    the words exactly as store.get_words returns them and the model's reply in
    any shape parse_reply accepts. Returns {"clips": [...], "dropped": [...]}:
    each clip has seconds, word indices, an excerpt, its boundary tiers and the
    list of repairs it needed; each dropped entry says why.
    """
    items = raw if isinstance(raw, list) else parse_reply(raw)
    ix = _Index(words)
    clips: list[dict] = []
    dropped: list[dict] = []
    if not ix.kept:
        return {"clips": [], "dropped": [{"title": "", "why": "nothing_kept"} for _ in items]}
    first, last = ix.kept[0].start, ix.kept[-1].end
    taken: list[tuple[float, float]] = []

    for rank, c in enumerate(items):
        if len(clips) >= n:
            break
        title = " ".join(str(c.get("title") or "").split())[:80] or "Untitled moment"
        reason = " ".join(str(c.get("reason") or "").split())[:300]
        repairs: list[str] = []
        s, e = parse_time(c.get("start")), parse_time(c.get("end"))
        if s is None:
            dropped.append({"title": title, "why": "dropped_bad_time"})
            continue
        if e is None:
            e = s + (min_s + max_s) / 2
            repairs.append("end_missing")
        if e < s:
            s, e = e, s
            repairs.append("swapped")
        if s >= last:
            dropped.append({"title": title, "why": "dropped_past_end"})
            continue
        if e > last + 1:
            e = last
            repairs.append("end_past_end")
        if _deleted_share(words, s, max(e, s + 1)) > 0.5:
            dropped.append({"title": title, "why": "dropped_deleted"})
            continue
        if ix.duration(0, len(ix.kept) - 1) < min_s:
            dropped.append({"title": title, "why": "dropped_too_short"})
            continue

        fit = _fit(ix, max(s, first), e, taken, min_s, max_s)
        if fit is None:
            # Nothing fits without overlapping a clip already chosen -- it is
            # the same moment again, or squeezed between two.
            ok_alone = _fit(ix, max(s, first), e, [], min_s, max_s)
            dropped.append({"title": title,
                            "why": "dropped_overlap" if ok_alone else "dropped_no_fit"})
            continue
        i, j = fit
        ts, te = ix.t_start(i), ix.t_end(j)
        if any(e > a and s < b for a, b in taken):
            repairs.append("moved_off_overlap")
        model_len = e - s
        if model_len < min_s:
            repairs.append("extended")
        elif model_len > max_s:
            repairs.append("trimmed")
        if abs(ts - s) > 1.0:
            repairs.append("start_snapped")
        if abs(te - e) > 1.0 and min_s <= model_len <= max_s:
            repairs.append("end_snapped")
        taken.append((ts, te))
        text = " ".join(w.text for w in ix.kept[i:j + 1])
        clips.append({
            "rank": rank + 1, "title": title, "reason": reason,
            "start": round(ts, 3), "end": round(te, 3),
            "duration": round(ix.duration(i, j), 2),
            "first_idx": ix.kept[i].idx, "last_idx": ix.kept[j].idx,
            "start_boundary": TIERS[ix.start_tier[i]],
            "end_boundary": TIERS[ix.end_tier[j]],
            "excerpt": text if len(text) <= 220 else text[:217].rsplit(" ", 1)[0] + "...",
            "model_start": round(s, 2), "model_end": round(e, 2),
            "repairs": repairs,
        })
    return {"clips": clips, "dropped": dropped}


# --------------------------------------------------------------------------- the job

def find_clips(words: Sequence[dict], n: int = 3, model: str | None = None,
               url: str | None = None,
               progress: Callable[[float, str], None] | None = None,
               ask: Callable[..., tuple[str, dict]] | None = None) -> dict:
    """Prompt -> model -> repair, for one project's words.

    Long recordings are asked about in parts; every part's suggestions go into
    one pool, strongest first, and the repair step keeps the best n that do not
    overlap.
    """
    ask = ask or ask_model
    model = model or MODEL
    segs = segments(words)
    if not segs:
        return {"clips": [], "dropped": [], "model": model, "parts": 0, "calls": [],
                "note": "nothing left in the transcript to choose from"}
    parts = chunk_segments(segs)
    pool: list[dict] = []
    calls: list[dict] = []
    for k, part in enumerate(parts):
        if progress:
            progress(k / len(parts), f"asking {model}"
                     + (f" (part {k + 1} of {len(parts)})" if len(parts) > 1 else ""))
        text, meta = ask(build_prompt(part, n, (k + 1, len(parts))), model=model, url=url)
        got = parse_reply(text)
        meta = dict(meta or {}, returned=len(got), reply_chars=len(text or ""))
        calls.append(meta)
        pool.extend(got)
    # Strongest first; the model's own order breaks ties (sorted() is stable).
    def score(c: dict) -> float:
        try:
            return -float(c.get("score", 0))
        except (TypeError, ValueError):
            return 0.0
    if len(parts) > 1:
        pool.sort(key=score)
    out = repair_clips(words, pool, n=n)
    out.update(model=model, parts=len(parts), calls=calls, returned=len(pool))
    if len(parts) > 1:
        out["note"] = f"long recording: asked in {len(parts)} parts"
    elif not pool:
        out["note"] = "the model gave no usable suggestions"
    return out
