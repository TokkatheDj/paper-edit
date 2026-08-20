"""The EDL is where an off-by-one silently desyncs a two-hour podcast."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from paperedit.edl import Cut, EditPlan, Word, derive_cuts, mark_fillers


def w(text, s, e, deleted=False):
    return Word(text, s, e, deleted=deleted)


def test_no_deletions_yields_one_continuous_cut():
    plan = derive_cuts([w("a", 0, 1), w("b", 1, 2), w("c", 2, 3)], pad=0)
    assert len(plan.cuts) == 1
    assert plan.cuts[0] == Cut(0.0, 3.0)


def test_deleting_middle_word_splits_into_two_cuts():
    words = [w("keep", 0, 1), w("drop", 1, 2, deleted=True), w("keep", 5, 6)]
    plan = derive_cuts(words, pad=0)
    assert [(c.start, c.end) for c in plan.cuts] == [(0.0, 1.0), (5.0, 6.0)]
    assert plan.duration == pytest.approx(2.0)


def test_everything_deleted_gives_empty_plan():
    plan = derive_cuts([w("x", 0, 1, deleted=True)])
    assert plan.cuts == [] and plan.duration == 0


def test_a_natural_pause_is_not_a_cut():
    """The bug this guards: splitting on time gaps silently removed every pause
    before the user had deleted anything."""
    words = [w("a", 0, 1), w("b", 4, 5), w("c", 9, 10)]   # long pauses, nothing deleted
    plan = derive_cuts(words, pad=0, duration=12.0)
    assert len(plan.cuts) == 1
    assert plan.cuts[0] == Cut(0.0, 12.0)


def test_deleting_a_short_word_still_cuts():
    """Deleting a 0.1s filler must actually remove it -- that IS the feature."""
    words = [w("a", 0, 1), w("um", 1, 1.1, deleted=True), w("b", 1.1, 2)]
    plan = derive_cuts(words, pad=0)
    assert len(plan.cuts) == 2
    assert plan.duration < 2.0


def test_pad_does_not_run_past_media_duration():
    plan = derive_cuts([w("end", 9.9, 10.0)], pad=0.5, duration=10.0)
    assert plan.cuts[-1].end <= 10.0


def test_snapping_moves_boundary_to_nearby_silence():
    words = [w("a", 0, 1), w("d", 1, 2, deleted=True), w("b", 2, 3)]
    plan = derive_cuts(words, pad=0, snap_points=[1.08], snap_tolerance=0.15)
    assert plan.cuts[0].end == pytest.approx(1.08)


def test_snapping_beyond_tolerance_is_ignored():
    words = [w("a", 0, 1), w("d", 1, 2, deleted=True), w("b", 2, 3)]
    plan = derive_cuts(words, pad=0, snap_points=[1.9], snap_tolerance=0.15)
    assert plan.cuts[0].end == pytest.approx(1.0)


def test_snap_induced_overlap_is_folded_not_duplicated():
    """Snapping can push two boundaries past each other; that must not produce
    an inverted or overlapping cut."""
    words = [w("a", 0, 1), w("d", 1, 1.2, deleted=True), w("b", 1.2, 2)]
    plan = derive_cuts(words, pad=0.3, snap_points=[1.1], snap_tolerance=0.5)
    for a, b in zip(plan.cuts, plan.cuts[1:]):
        assert a.end <= b.start
    assert all(c.end > c.start for c in plan.cuts)


def test_frame_quantisation_lands_on_the_grid():
    plan = derive_cuts([w("a", 0.017, 1.031)], pad=0, fps=30.0)
    for c in plan.cuts:
        assert c.start * 30 == pytest.approx(round(c.start * 30), abs=1e-6)
        assert c.end * 30 == pytest.approx(round(c.end * 30), abs=1e-6)


def test_duration_equals_sum_of_cuts():
    words = [w("a", 0, 1), w("x", 1, 2, deleted=True), w("b", 2, 3),
             w("y", 3, 4, deleted=True), w("c", 4, 5)]
    plan = derive_cuts(words, pad=0)
    assert plan.duration == pytest.approx(sum(c.duration for c in plan.cuts))


def test_output_and_source_time_round_trip():
    plan = EditPlan([Cut(0, 2), Cut(10, 12)])
    assert plan.output_to_source(0.5) == pytest.approx(0.5)
    assert plan.output_to_source(2.5) == pytest.approx(10.5)   # jumps the hole
    assert plan.source_to_output(10.5) == pytest.approx(2.5)
    assert plan.source_to_output(5.0) is None                  # inside a deletion


def test_player_seek_never_lands_in_deleted_audio():
    """The whole preview-without-rendering trick depends on this."""
    plan = EditPlan([Cut(0, 2), Cut(10, 12), Cut(20, 21)])
    t = 0.0
    while t < plan.duration:
        src = plan.output_to_source(t)
        assert any(c.start <= src < c.end for c in plan.cuts), f"{t} -> {src}"
        t += 0.05


def test_mark_fillers_only_hits_standalone_fillers():
    words = [w("Um", 0, 1), w("umbrella", 1, 2), w("uh,", 2, 3), w("hello", 3, 4)]
    assert mark_fillers(words) == 2
    assert [x.deleted for x in words] == [True, False, True, False]


# --------------------------------------------------------------- silence removal

def test_removing_a_pause_splits_the_cut():
    plan = derive_cuts([w("a", 0, 1), w("b", 9, 10)], pad=0, duration=10.0,
                       removed_ranges=[(2.0, 8.0)])
    assert [(c.start, c.end) for c in plan.cuts] == [(0.0, 2.0), (8.0, 10.0)]
    assert plan.duration == pytest.approx(4.0)


def test_removed_range_outside_any_cut_changes_nothing():
    words = [w("a", 0, 1), w("x", 1, 2, deleted=True), w("b", 5, 6)]
    base = derive_cuts(words, pad=0)
    same = derive_cuts(words, pad=0, removed_ranges=[(2.5, 4.5)])
    assert [(c.start, c.end) for c in same.cuts] == [(c.start, c.end) for c in base.cuts]


def test_removed_range_covering_a_whole_cut_drops_it():
    plan = derive_cuts([w("a", 0, 1), w("b", 9, 10)], pad=0, duration=10.0,
                       removed_ranges=[(0.0, 3.0)])
    assert all(c.start >= 3.0 for c in plan.cuts)


def test_overlapping_removed_ranges_do_not_corrupt_the_plan():
    plan = derive_cuts([w("a", 0, 20)], pad=0, duration=20.0,
                       removed_ranges=[(5, 10), (8, 12), (11, 13)])
    for a, b in zip(plan.cuts, plan.cuts[1:]):
        assert a.end <= b.start
    assert all(c.end > c.start for c in plan.cuts)
    assert plan.duration == pytest.approx(12.0)      # 20 minus 5..13


def test_silence_removal_never_lands_the_player_in_removed_audio():
    """The preview seeks over cuts; a bad subtraction would strand the playhead."""
    plan = derive_cuts([w("a", 0, 30)], pad=0, duration=30.0,
                       removed_ranges=[(4, 6), (11, 14), (22, 25)])
    t = 0.0
    while t < plan.duration:
        src = plan.output_to_source(t)
        assert not any(a <= src < b for a, b in [(4, 6), (11, 14), (22, 25)]), src
        t += 0.05


def test_deletions_and_silence_removal_compose():
    """Both features write to the same cut list and must not fight."""
    words = [w("keep", 0, 2), w("drop", 3, 4, deleted=True), w("keep2", 10, 12)]
    plan = derive_cuts(words, pad=0, duration=12.0, removed_ranges=[(5.0, 9.0)])
    for a, b in zip(plan.cuts, plan.cuts[1:]):
        assert a.end <= b.start
    assert not any(c.start < 3.5 < c.end for c in plan.cuts)     # deleted word gone
    assert not any(c.start < 7.0 < c.end for c in plan.cuts)     # pause gone


def test_silence_gaps_leaves_a_natural_beat():
    from paperedit.audio import silence_gaps
    # a 2s pause, keeping 0.3s of it
    gaps = silence_gaps([(10.0, 12.0)], keep=0.3, min_remove=0.6)
    assert len(gaps) == 1
    s, e = gaps[0]
    assert (e - s) == pytest.approx(2.0 - 0.3)
    assert s > 10.0 and e < 12.0          # trimmed from the middle, not the edges


def test_silence_gaps_ignores_short_pauses():
    from paperedit.audio import silence_gaps
    assert silence_gaps([(1.0, 1.4)], keep=0.3, min_remove=0.6) == []
