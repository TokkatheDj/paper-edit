"""Animated captions -- word-level subtitles burned into the video.

Two things make this harder than it looks:

1. Captions belong on the EDITED timeline, not the source one. A word at 04:12
   in the original might be at 03:48 after cuts, and words that were deleted must
   not appear at all. Callers map times through EditPlan.source_to_output first;
   `words_on_output_timeline` does it for them.

2. "Animated" here means the current word is highlighted inside its phrase --
   the style people recognise from short-form video. That is one subtitle event
   PER WORD, each showing the whole phrase with a different word emphasised, not
   one event per phrase.

ASS colours are &HBBGGRR -- blue and red swapped relative to hex you would write
for the web. Getting that backwards is the classic way to ship blue captions you
meant to be orange.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


C = chr(92) + "c"      # the ASS colour-override tag, kept out of f-strings


def ass_colour(hex_rgb: str) -> str:
    """'FFD166' -> '&H0066D1FF' (ASS wants alpha + BGR)."""
    h = hex_rgb.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b}{g}{r}".upper()


@dataclass(slots=True)
class Style:
    name: str
    font: str = "Arial"
    size_ratio: float = 0.052      # of video height, so it scales with the frame
    primary: str = "FFFFFF"        # the phrase
    active: str = "FFD166"         # the word being spoken
    outline: str = "000000"
    outline_w: int = 3
    shadow: int = 1
    bold: bool = True
    uppercase: bool = False
    align: int = 2                 # 2 = bottom centre in ASS numbering
    margin_v_ratio: float = 0.10   # of video height
    max_words: int = 4
    max_chars: int = 30


PRESETS: dict[str, Style] = {
    "clean":     Style("clean"),
    "highlight": Style("highlight", active="7DFF9B", size_ratio=0.058,
                       outline_w=4, max_words=3),
    "bold":      Style("bold", active="FF5C7C", size_ratio=0.065, uppercase=True,
                       outline_w=5, max_words=3, max_chars=22),
    "subtle":    Style("subtle", size_ratio=0.040, active="FFFFFF",
                       outline_w=2, bold=False, max_words=6, max_chars=42),
}


def words_on_output_timeline(words: Sequence[dict], plan) -> list[dict]:
    """Re-time words onto the edited timeline, dropping anything cut out."""
    out = []
    for w in words:
        if w.get("deleted"):
            continue
        start = plan.source_to_output(w["start"])
        end = plan.source_to_output(max(w["end"] - 0.001, w["start"]))
        if start is None or end is None or end <= start:
            continue               # the word fell inside a removed range
        out.append({"text": w["text"].strip(), "start": start, "end": end})
    out.sort(key=lambda w: w["start"])
    return out


def group_words(words: Sequence[dict], style: Style,
                max_gap: float = 0.7) -> list[list[dict]]:
    """Break the stream into short phrases that fit on screen.

    A phrase ends on word count, character count, or a pause -- a pause is a
    natural caption break, and honouring it keeps captions in step with speech
    instead of running across a beat.
    """
    phrases: list[list[dict]] = []
    cur: list[dict] = []
    chars = 0
    for w in words:
        gap = w["start"] - cur[-1]["end"] if cur else 0.0
        if cur and (len(cur) >= style.max_words
                    or chars + len(w["text"]) + 1 > style.max_chars
                    or gap > max_gap):
            phrases.append(cur)
            cur, chars = [], 0
        cur.append(w)
        chars += len(w["text"]) + 1
    if cur:
        phrases.append(cur)
    return phrases


def _ts(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _escape(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")")


def build_ass(words: Sequence[dict], *, width: int, height: int,
              style: Style | str = "clean") -> str:
    """Full .ass document with one event per word."""
    if isinstance(style, str):
        style = PRESETS.get(style, PRESETS["clean"])
    size = max(12, int(height * style.size_ratio))
    margin_v = int(height * style.margin_v_ratio)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{style.font},{size},{ass_colour(style.primary)},{ass_colour(style.active)},{ass_colour(style.outline)},&H80000000,{-1 if style.bold else 0},0,0,0,100,100,0,0,1,{style.outline_w},{style.shadow},{style.align},{int(width*0.06)},{int(width*0.06)},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    active = ass_colour(style.active)
    primary = ass_colour(style.primary)
    lines = []
    for phrase in group_words(words, style):
        for i, w in enumerate(phrase):
            end = w["end"]
            if i + 1 < len(phrase):
                # Hold the highlight until the next word starts, so the caption
                # never blinks out during the gap between two words.
                end = max(end, phrase[i + 1]["start"])
            parts = []
            for j, other in enumerate(phrase):
                t = _escape(other["text"])
                if style.uppercase:
                    t = t.upper()
                parts.append(f"{{{C}{active}}}{t}{{{C}{primary}}}"
                             if j == i else t)
            lines.append(f"Dialogue: 0,{_ts(w['start'])},{_ts(end)},Cap,,0,0,0,,"
                         + " ".join(parts))
    return header + "\n".join(lines) + "\n"


def write_ass(words: Sequence[dict], path: str | Path, *, width: int, height: int,
              style: Style | str = "clean") -> Path:
    p = Path(path)
    p.write_text(build_ass(words, width=width, height=height, style=style),
                 encoding="utf-8")
    return p
