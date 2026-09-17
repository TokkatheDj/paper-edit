"""The video dissolve must never produce a graph that truncates the picture.

These check the graph, not a render, so they run in milliseconds. The render
behaviour behind them was measured separately: a sub-frame xfade makes ffmpeg
end the video stream early and still exit 0.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperedit.edl import Cut, EditPlan
from paperedit.render import MIN_VIDEO_DISSOLVE, build_filtergraph, video_dissolves


def test_ordinary_joins_dissolve():
    plan = EditPlan([Cut(0.0, 4.0), Cut(4.5, 8.0), Cut(8.4, 12.0)])
    g = build_filtergraph(plan, video=True)
    assert g.count("xfade=") == 2


def test_dissolve_borrows_from_the_deleted_gap_not_the_kept_shot():
    plan = EditPlan([Cut(0.0, 4.0), Cut(4.5, 8.0)])
    g = build_filtergraph(plan, video=True)
    assert "trim=start=0.0000:end=4.1000" in g     # reaches into the gap
    assert "trim=start=4.5000:end=8.0000" in g     # last shot is untouched


def test_a_sub_frame_kept_sliver_falls_back_to_hard_cuts():
    plan = EditPlan([Cut(0.0, 10.0), Cut(10.2, 10.206), Cut(10.5, 20.0)])
    assert min(video_dissolves(plan)) < MIN_VIDEO_DISSOLVE
    g = build_filtergraph(plan, video=True)
    assert "xfade=" not in g
    assert "concat=n=3:v=1:a=1" in g
    # and the trims are NOT extended into the gap when not blending
    assert "trim=start=0.0000:end=10.0000" in g


def test_a_near_zero_gap_falls_back_to_hard_cuts():
    plan = EditPlan([Cut(10.0, 20.3456), Cut(20.34563, 30.0), Cut(31.1234, 40.0)])
    g = build_filtergraph(plan, video=True)
    assert "xfade=" not in g
    assert "duration=0.0000" not in g


def test_audio_only_is_unaffected():
    plan = EditPlan([Cut(0.0, 4.0), Cut(4.5, 8.0)])
    g = build_filtergraph(plan, video=False)
    assert "xfade=" not in g and "[0:v]" not in g
