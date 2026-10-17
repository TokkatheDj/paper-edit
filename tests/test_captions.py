"""Caption layout for the frame it is burned into.

Found making the first real short (17 Oct 2026): text sized by height alone
was 111 px on a 1080 px-wide vertical frame and ran edge to edge, with
wrapping off; and it sat in the bottom 10%, under the Reels/TikTok overlay.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperedit.captions import PRESETS, build_ass  # noqa: E402

WORDS = [{"text": "one", "start": 0.0, "end": 0.3},
         {"text": "longer", "start": 0.3, "end": 0.7},
         {"text": "sentence", "start": 0.7, "end": 1.2}]


def _style_line(ass: str) -> list[str]:
    line = next(l for l in ass.splitlines() if l.startswith("Style: Cap,"))
    return line.split(",")


def _layout(ass: str) -> dict:
    f = _style_line(ass)
    return {"size": int(f[2]), "margin_l": int(f[19]), "margin_r": int(f[20]), "margin_v": int(f[21])}


def test_landscape_layout_is_unchanged():
    lay = _layout(build_ass(WORDS, width=1920, height=1080, style="highlight"))
    assert lay["size"] == int(1080 * PRESETS["highlight"].size_ratio)
    assert lay["margin_v"] == int(1080 * 0.10)
    assert lay["margin_l"] == lay["margin_r"] == int(1920 * 0.06)


def test_vertical_text_is_capped_by_width_for_every_style():
    for name in PRESETS:
        lay = _layout(build_ass(WORDS, width=1080, height=1920, style=name))
        assert lay["size"] <= int(1080 * 0.075), name


def test_vertical_captions_sit_above_the_app_overlay():
    lay = _layout(build_ass(WORDS, width=1080, height=1920, style="highlight"))
    assert lay["margin_v"] >= int(1920 * 0.25)       # out of the bottom quarter
    assert lay["margin_l"] == lay["margin_r"] == int(1080 * 0.08)


def test_long_phrases_wrap_rather_than_run_off_screen():
    assert "WrapStyle: 0" in build_ass(WORDS, width=1080, height=1920)
