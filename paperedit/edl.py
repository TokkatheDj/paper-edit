"""The edit decision list -- the one structure the whole app writes against.

The transcript IS the edit. Every word carries a source time span and a `deleted`
flag; the EDL is DERIVED from the surviving words, never stored independently.
Filler removal, silence removal, and the timeline are all just different writers
setting `deleted` on words (or inserting gap-cuts). Keeping one source of truth is
what stops those features from fighting each other.

Boundary snapping matters: measured Whisper word timestamps carry ~50-130 ms of
median error (spikes/timestamp_accuracy.py), so cutting exactly on a word boundary
clips consonants. We snap each cut to the quietest nearby moment instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence


@dataclass(slots=True)
class Word:
    text: str
    start: float
    end: float
    speaker: str | None = None
    deleted: bool = False
    probability: float = 1.0


@dataclass(slots=True, frozen=True)
class Cut:
    """A half-open source range [start, end) that survives into the output."""
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(slots=True)
class EditPlan:
    cuts: list[Cut] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(c.duration for c in self.cuts)

    def source_to_output(self, t: float) -> float | None:
        """Map a source timestamp onto the edited timeline (None if cut out)."""
        acc = 0.0
        for c in self.cuts:
            if c.start <= t < c.end:
                return acc + (t - c.start)
            acc += c.duration
        return None

    def output_to_source(self, t: float) -> float:
        """Map an edited-timeline timestamp back to the source. Drives the player."""
        acc = 0.0
        for c in self.cuts:
            if t < acc + c.duration:
                return c.start + (t - acc)
            acc += c.duration
        return self.cuts[-1].end if self.cuts else 0.0


def _snap(t: float, points: Sequence[float], tolerance: float) -> float:
    """Move t to the nearest silence point within tolerance, else leave it."""
    if not points:
        return t
    best = min(points, key=lambda p: abs(p - t))
    return best if abs(best - t) <= tolerance else t


def derive_cuts(
    words: Iterable[Word],
    *,
    pad: float = 0.04,
    snap_points: Sequence[float] = (),
    snap_tolerance: float = 0.15,
    duration: float | None = None,
    fps: float | None = None,
) -> EditPlan:
    """Merge surviving words into contiguous source ranges.

    Cuts happen where TEXT WAS DELETED -- never where the speaker merely paused.
    An earlier version split on any time gap between surviving words, which
    quietly stripped every natural pause before the user had edited anything: a
    5:00 recording came back as 2:56 with nothing deleted. Silence removal is a
    Phase 2 feature she asks for, not something the editor does behind her back.

    pad             breathing room kept either side of a surviving run
    snap_points     candidate silence midpoints to align boundaries to; cutting
                    exactly on a word boundary clips consonants, because Whisper
                    timestamps carry ~50-130 ms of error
    fps             quantise boundaries onto the video frame grid. NOTE this does
                    NOT remove the small export-length overshoot -- that was
                    measured and disproved in spikes/drift.py. Kept because
                    frame-aligned boundaries make cuts land predictably.
    """
    all_words = list(words)
    if not any(not w.deleted for w in all_words):
        return EditPlan([])

    # Close the current run only when a deleted word interrupts it.
    runs: list[list[float]] = []
    open_run: list[float] | None = None
    for w in all_words:
        if w.deleted:
            if open_run is not None:
                runs.append(open_run)
                open_run = None
        elif open_run is None:
            open_run = [w.start, w.end]
        else:
            open_run[1] = max(open_run[1], w.end)
    if open_run is not None:
        runs.append(open_run)

    # Nothing was deleted before the first surviving word or after the last, so
    # whatever is out there (music, an intro, room tone) is content, not a gap.
    if not all_words[0].deleted:
        runs[0][0] = 0.0
    if not all_words[-1].deleted and duration is not None:
        runs[-1][1] = duration

    cuts: list[Cut] = []
    for start, end in runs:
        s = _snap(max(0.0, start - pad), snap_points, snap_tolerance)
        e = _snap(end + pad, snap_points, snap_tolerance)
        if duration is not None:
            e = min(e, duration)
        # Quantise LAST: rounding after this would knock the boundary back off
        # the frame grid (1/30 s is not representable at 4 decimal places).
        s, e = (round(s, 4), round(e, 4)) if not fps else (
            round(s * fps) / fps, round(e * fps) / fps)
        if e > s:
            cuts.append(Cut(s, e))

    # Snapping can push neighbours into overlap; fold those together.
    merged: list[Cut] = []
    for c in cuts:
        if merged and c.start <= merged[-1].end:
            merged[-1] = Cut(merged[-1].start, max(merged[-1].end, c.end))
        else:
            merged.append(c)
    return EditPlan(merged)


FILLERS = {"um", "uh", "erm", "hmm", "mm", "uhh", "umm", "er", "eh",
           "mm-hmm", "uh-huh", "hm"}


def _norm(text: str) -> str:
    return text.strip().strip(".,!?;:-\u2014\"'").lower()


def mark_fillers(words: Sequence[Word], vocabulary: set[str] = FILLERS) -> int:
    """Delete filler words Whisper actually spelled out.

    Measured recall is only 7-17% (spikes/filler_truth.py) -- Whisper silently
    tidies most fillers away -- so this is a supplement to acoustic gap removal,
    never the whole feature.
    """
    n = 0
    for w in words:
        if not w.deleted and _norm(w.text) in vocabulary:
            w.deleted = True
            n += 1
    return n
