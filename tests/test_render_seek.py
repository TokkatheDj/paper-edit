"""Each cut reads its own seeked input, so export time no longer grows with
(cuts x video length). These check that the fast path builds the SAME edit as
the single-input graph, and that it steps aside when the command line is full.

Measured 30 Sep 2026 on a real 1080p60 recording: 40 cuts over a minute rendered in 20 s
instead of 104 s; the two outputs had identical frame counts and durations,
PSNR >= 49 dB on every frame, and audio within 1-2 samples.
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paperedit.render as R
from paperedit.edl import Cut, EditPlan

PLAN = EditPlan([Cut(1.0, 4.0), Cut(4.5, 8.25), Cut(9.0, 12.0)])
MEDIA = ROOT / "spikes" / "media" / "test-300s.mp4"


def trims(graph: str) -> list[tuple[str, float]]:
    """(stream, trimmed length) for every trim/atrim, in order."""
    return [(kind, float(end) - float(start)) for kind, start, end in
            re.findall(r"(a?trim)=start=([\d.]+):end=([\d.]+)", graph)]


@pytest.mark.parametrize("video", [True, False])
def test_seeked_graph_is_the_same_edit(video):
    one = R.build_filtergraph(PLAN, video=video)
    per_cut = R.build_filtergraph(PLAN, video=video, seeked=True)
    assert [k for k, _ in trims(per_cut)] == [k for k, _ in trims(one)]
    assert [n for _, n in trims(per_cut)] == pytest.approx([n for _, n in trims(one)], abs=1e-4)
    assert "[0:v]trim=start=1" not in per_cut
    assert "[2:a]atrim=start=0.0000:end=3.0000" in per_cut
    # everything after the trims -- fades, dissolve offsets, concat -- identical
    tail = lambda g: [l.split("]", 1)[1] if l.startswith("[") and "trim=" in l else l
                      for l in g.splitlines()]
    assert [re.sub(r"trim=start=[\d.]+:end=[\d.]+", "", l) for l in tail(per_cut)] == \
           [re.sub(r"trim=start=[\d.]+:end=[\d.]+", "", l) for l in tail(one)]


def test_each_input_starts_at_its_cut_and_covers_its_dissolve():
    args, seeked = R.choose_inputs("clip.mov", PLAN, video=True)
    assert seeked and args.count("-i") == 3
    starts = [float(args[i + 1]) for i, a in enumerate(args) if a == "-ss"]
    lengths = [float(args[i + 1]) for i, a in enumerate(args) if a == "-t"]
    assert starts == [1.0, 4.5, 9.0]
    dissolves = R.join_dissolves(PLAN, video=True) + [0.0]
    for c, d, t in zip(PLAN.cuts, dissolves, lengths):
        assert t >= c.duration + d                              # nothing is starved


def test_single_cut_and_oversized_plans_use_one_input():
    assert R.choose_inputs("clip.mov", EditPlan([Cut(0, 5)]), video=True) == (["-i", "clip.mov"], False)
    # 400 cuts of a long path cannot fit on a Windows command line
    long_path = "D:/" + "x" * 120 + "/source.MOV"
    many = EditPlan([Cut(i * 2.0, i * 2.0 + 1.5) for i in range(400)])
    args, seeked = R.choose_inputs(long_path, many, video=True)
    assert not seeked and args == ["-i", long_path]
    assert len(" ".join(R.seek_inputs(long_path, many, []))) > R.MAX_INPUT_ARGS_CHARS


def test_too_many_cuts_to_hold_in_memory_use_one_input():
    # a short path, so only the input count stops it
    n = R.MAX_SEEKED_INPUTS + 1
    many = EditPlan([Cut(i * 2.0, i * 2.0 + 1.5) for i in range(n)])
    assert len(" ".join(R.seek_inputs("a.mov", many, []))) <= R.MAX_INPUT_ARGS_CHARS
    assert R.choose_inputs("a.mov", many, video=True)[1] is False
    fewer = EditPlan(many.cuts[:R.MAX_SEEKED_INPUTS])
    assert R.choose_inputs("a.mov", fewer, video=True)[1] is True


def test_each_decoder_gets_one_thread():
    args, _ = R.choose_inputs("clip.mov", PLAN, video=True)
    assert args.count("-threads") == args.count("-i") == 3


@pytest.mark.skipif(not MEDIA.exists(), reason="run spikes/cut_quality.py first to build the test media")
def test_fallback_and_fast_path_render_the_same_length(tmp_path, monkeypatch):
    plan = EditPlan([Cut(2.0, 5.0), Cut(6.0, 9.5), Cut(11.0, 13.0), Cut(14.2, 17.0)])
    fast = R.render(MEDIA, plan, tmp_path / "fast.mp4")
    monkeypatch.setattr(R, "MAX_INPUT_ARGS_CHARS", 0)          # force the fallback
    slow = R.render(MEDIA, plan, tmp_path / "slow.mp4")
    a, b = R.media_info(fast), R.media_info(slow)
    assert a["duration"] == pytest.approx(b["duration"], abs=1 / 30)
    assert a["duration"] == pytest.approx(plan.duration, abs=0.1)
