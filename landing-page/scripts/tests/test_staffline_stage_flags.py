"""Regression test for SF-8 (ALPHA_TRANSITION_PLAN.md / issue #213 batch):
stored JSOMR records were dropping component-filter and assignment/grouping
flags (only fit-level flags survived to persist_staffline_detection), and
this test also confirms the *other* half of SF-8's ticket -- that per-record
ids are already stable through the (stave_id, within_stave_index) sort, not
derived from list position.

staffline_stage.py imports `job_store`, which imports `auth_api`, which
connects to a live Postgres at module import time (see
test_resolve_hints_staleness.py's docstring for the same constraint) -- this
file stubs `job_store` in sys.modules before importing staffline_stage so the
import never touches a real DB.

_assemble_jsomr_records is exercised directly with hand-built fake
FitResult/StaveAssignment/GroupingResult stand-ins (plain
types.SimpleNamespace, matching only the attributes that function actually
reads) rather than running the real component_filter/fit_centerline/
group_staves pipeline -- this test is about the assembly function's own
flag-merging and id-stability logic, not the algorithmic modules feeding it
(those already have their own coverage in staff-finding/scripts/script_tests/).

No DB, no Celery, no cv2/scipy/scikit-image.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "job_store" not in sys.modules:
    job_store_stub = types.ModuleType("job_store")
    job_store_stub.check_cancelled = lambda job_id: None

    class _JobCancelled(Exception):
        pass

    job_store_stub.JobCancelled = _JobCancelled
    sys.modules["job_store"] = job_store_stub

from staffline_stage import _assemble_jsomr_records  # noqa: E402


def _fake_fit(x_start, x_end, y_values, flags):
    return types.SimpleNamespace(
        x_start=x_start, x_end=x_end, y_values=list(y_values),
        coefficients=[0.0, 0.0, y_values[0] if y_values else 0.0],
        residual_mean=0.1, residual_max=0.2,
        n_pixels_used=len(y_values), n_pixels_total=len(y_values),
        x_page_offset=0.0, y_page_offset=0.0,
        flags=list(flags),
    )


def _fake_assignment(fit_index, stave_id, within_stave_index, flags):
    return types.SimpleNamespace(
        fit_index=fit_index, stave_id=stave_id,
        within_stave_index=within_stave_index, flags=list(flags),
    )


def _fake_grouping_result(assignments, mode_lines_per_stave=4, rhythm_anomalies=None):
    return types.SimpleNamespace(
        assignments=assignments,
        interpolated_lines=[],
        mode_lines_per_stave=mode_lines_per_stave,
        rhythm_anomalies=rhythm_anomalies or {},
        flags=[],  # page-level flags -- read by run_staffline_detection, not this function
    )


def test_quality_flags_merge_all_three_tiers():
    """fit.flags (already merged with component-filter flags upstream, per
    _fit_and_group's own SF-8 fix) + asg.flags (grouping-level) both survive
    into the persisted record's quality.flags -- previously only fit.flags did."""
    # Simulates a fit whose flags already carry a component-filter flag
    # merged in by _fit_and_group ("multiple_components_kept") alongside its
    # own fit-level flag ("line_following_no_improvement").
    fit = _fake_fit(0, 10, [5.0] * 11, flags=["multiple_components_kept", "line_following_no_improvement"])
    asg = _fake_assignment(fit_index=0, stave_id=0, within_stave_index=0, flags=["gap_near_threshold"])
    grouping_result = _fake_grouping_result(assignments=[asg])

    records = _assemble_jsomr_records("test_img", [fit], [(0, 0, 10, 10)], grouping_result, scale_unit=10.0)

    assert len(records) == 1
    flags = records[0]["quality"]["flags"]
    assert "multiple_components_kept" in flags  # component-filter tier
    assert "line_following_no_improvement" in flags  # fit tier
    assert "gap_near_threshold" in flags  # assignment/grouping tier


def test_quality_flags_when_no_assignment():
    """A fit with no matching assignment (asg is None) must not crash on
    asg.flags -- quality.flags degrades to just the fit's own flags."""
    fit = _fake_fit(0, 5, [1.0] * 6, flags=["no_components_survived"])
    grouping_result = _fake_grouping_result(assignments=[])  # no assignment for fit_index 0

    records = _assemble_jsomr_records("test_img", [fit], [(0, 0, 5, 5)], grouping_result, scale_unit=10.0)

    assert records[0]["quality"]["flags"] == ["no_components_survived"]
    assert records[0]["stave_id"] is None


def test_ids_stable_and_unique_after_sort():
    """Ids are assigned from the pre-sort enumerate() index and are already
    content-derived (image_name + pre-sort index), not list position -- the
    sort at the end of _assemble_jsomr_records must not disturb them. Build
    fit_results in an order that the (stave_id, within_stave_index) sort key
    will reorder, and confirm every id is still unique and traceable."""
    fits = [
        _fake_fit(0, 5, [10.0] * 6, flags=[]),   # idx 0 -> ends up stave 1, index 0
        _fake_fit(0, 5, [50.0] * 6, flags=[]),   # idx 1 -> ends up stave 0, index 1
        _fake_fit(0, 5, [30.0] * 6, flags=[]),   # idx 2 -> ends up stave 0, index 0
    ]
    boxes = [(0, 10, 5, 15), (0, 50, 5, 55), (0, 30, 5, 35)]
    assignments = [
        _fake_assignment(fit_index=0, stave_id=1, within_stave_index=0, flags=[]),
        _fake_assignment(fit_index=1, stave_id=0, within_stave_index=1, flags=[]),
        _fake_assignment(fit_index=2, stave_id=0, within_stave_index=0, flags=[]),
    ]
    grouping_result = _fake_grouping_result(assignments=assignments)

    records = _assemble_jsomr_records("test_img", fits, boxes, grouping_result, scale_unit=10.0)

    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), "ids must stay unique after the (stave_id, within_stave_index) sort"
    assert ids == ["test_img_line0002", "test_img_line0001", "test_img_line0000"], (
        "sort reorders records by (stave_id, within_stave_index), but each id must still be "
        "traceable back to its original pre-sort enumerate() index, not reassigned from new list position"
    )
    # Confirm the sort actually reordered by stave/within-stave-index, not left in input order.
    assert [r["stave_id"] for r in records] == [0, 0, 1]
    assert [r["within_stave_index"] for r in records] == [0, 1, 0]
