"""Unit tests for staffline_adapter.py against hand-built JSOMR fixtures.

No DB, no Celery, no cv2/scipy/scikit-image -- staffline_adapter.py is a pure
JSON-in/dataclass-out module, so these run without the staff-finding package
or its dependencies installed at all.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from staffline_adapter import staves_from_jsomr, y_values_at, _dedupe_line_ys  # noqa: E402


def _line(stave_id, within_stave_index, x_start, x_end, y_values,
          ulx=None, uly=None, lrx=None, lry=None, source="detected"):
    """Build one hand-built JSOMR record. bounding_box is omitted (None)
    when ulx/uly/lrx/lry aren't given, matching an interpolated line."""
    bounding_box = None
    if ulx is not None:
        bounding_box = {"ulx": ulx, "uly": uly, "lrx": lrx, "lry": lry}
    centerline_page = {"x_start": x_start, "x_end": x_end, "y_values": list(y_values)}
    return {
        "id": f"test_s{stave_id}_l{within_stave_index}",
        "source": source,
        "bounding_box": bounding_box,
        "centerline": centerline_page,
        "centerline_page": centerline_page,
        "fit": {"method": "quadratic_huber", "coefficients": [0, 0, y_values[0]],
                "residual_mean": 0.1, "residual_max": 0.2,
                "n_pixels_used": len(y_values), "n_pixels_total": len(y_values)},
        "quality": {"confidence": None, "flags": []},
        "scale_unit": 10.0,
        "column_id": None,
        "stave_id": stave_id,
        "lines_detected": None,
        "lines_interpolated": None,
        "lines_expected": None,
        "rhythm_status": None,
        "within_stave_index": within_stave_index,
    }


def test_detected_only_two_staves():
    records = []
    for i, y in enumerate([100, 120, 140, 160]):
        records.append(_line(0, i, 0, 700, [y] * 701, ulx=50, uly=y - 5, lrx=750, lry=y + 5))
    for i, y in enumerate([300, 320, 340]):
        records.append(_line(1, i, 0, 700, [y] * 701, ulx=50, uly=y - 5, lrx=750, lry=y + 5))

    staves = staves_from_jsomr(records)

    assert len(staves) == 2
    assert staves[0].id == "jsomr-stave-0"
    assert staves[1].id == "jsomr-stave-1"
    assert staves[0].line_ys == [100.0, 120.0, 140.0, 160.0]
    assert staves[1].line_ys == [300.0, 320.0, 340.0]
    # Bounding box is the union across the stave's own member lines.
    assert (staves[0].ulx, staves[0].uly, staves[0].lrx, staves[0].lry) == (50, 95, 750, 165)


def test_stave_id_none_records_excluded():
    records = [
        _line(0, 0, 0, 700, [100] * 701, ulx=50, uly=95, lrx=750, lry=105),
        {**_line(None, None, 0, 700, [500] * 701), "stave_id": None, "within_stave_index": None},
    ]
    staves = staves_from_jsomr(records)
    assert len(staves) == 1
    assert staves[0].line_ys == [100.0]


def test_mixed_detected_and_interpolated_lines_in_one_stave():
    records = [
        _line(0, 0, 0, 700, [100] * 701, ulx=50, uly=95, lrx=750, lry=105),
        # Interpolated line: no bounding_box, but a real centerline -- must
        # still be grouped into the stave and contribute to line_ys.
        _line(0, 1, 0, 700, [120] * 701, source="interpolated"),
        _line(0, 2, 0, 700, [140] * 701, ulx=50, uly=135, lrx=750, lry=145),
    ]
    staves = staves_from_jsomr(records)
    assert len(staves) == 1
    # bbox comes only from the two boxed (detected) lines, not the interpolated one.
    assert (staves[0].ulx, staves[0].uly, staves[0].lrx, staves[0].lry) == (50, 95, 750, 145)
    assert staves[0].line_ys == [100.0, 120.0, 140.0]


def test_all_interpolated_stave_falls_back_to_centerline_extent():
    records = [
        _line(0, 0, 10, 90, [200] * 81, source="interpolated"),
        _line(0, 1, 10, 90, [220] * 81, source="interpolated"),
    ]
    staves = staves_from_jsomr(records)
    assert len(staves) == 1
    s = staves[0]
    assert s.ulx == 10 and s.lrx == 90
    assert s.uly == 200 and s.lry == 220
    assert s.line_ys == [200.0, 220.0]


def test_y_values_at_clamps_outside_line_extent():
    # A stave whose lines only cover x in [100, 200] -- sampling far outside
    # that range must clamp to the nearest endpoint, not raise or extrapolate.
    records = [
        _line(0, 0, 100, 200, [float(y) for y in range(50, 151)], ulx=100, uly=45, lrx=200, lry=55),
    ]
    ys_inside = y_values_at(records, 150)   # x=150 -> offset 50 -> y_values[50] = 100
    ys_below = y_values_at(records, 0)      # clamped to x_start=100 -> y_values[0] = 50
    ys_above = y_values_at(records, 99999)  # clamped to x_end=200 -> y_values[-1] = 150

    assert ys_inside == [100.0]
    assert ys_below == [50.0]
    assert ys_above == [150.0]


def test_within_stave_index_none_does_not_crash_sort():
    records = [
        _line(0, None, 0, 700, [130] * 701, ulx=50, uly=125, lrx=750, lry=135),
        _line(0, 0, 0, 700, [100] * 701, ulx=50, uly=95, lrx=750, lry=105),
        _line(0, 1, 0, 700, [120] * 701, ulx=50, uly=115, lrx=750, lry=125),
    ]
    staves = staves_from_jsomr(records)
    assert len(staves) == 1
    # Indexed lines sort first (0, 1), the None-indexed one goes last.
    assert staves[0].line_ys == [100.0, 120.0, 130.0]


def test_empty_input_produces_no_staves():
    assert staves_from_jsomr([]) == []


# --- _dedupe_line_ys / the real-page duplicate-fragment regression --------

def test_dedupe_line_ys_merges_close_pairs_keeps_real_gaps():
    """Reproduces the actual reported bug directly against real numbers
    pulled from a live page: one stave's 4 real ruled lines were each
    split into a left-half/right-half JSOMR fragment pair by the
    detector, sampled a couple px apart (see y_values_at's docstring for
    why), inflating line_ys to 7 entries -- which then badly corrupted
    encode_to_mei.py's clef-line lookup and rendered every note on that
    stave at a wrong pitch."""
    raw = [841.2, 842.6, 869.5, 871.7, 899.7, 901.9, 931.6]
    deduped = _dedupe_line_ys(raw)
    assert len(deduped) == 4
    # Merged pairs collapse to their mean; the unpaired last line passes through.
    assert deduped[0] == (841.2 + 842.6) / 2
    assert deduped[-1] == 931.6

def test_dedupe_line_ys_leaves_genuinely_spaced_lines_alone():
    assert _dedupe_line_ys([100.0, 120.0, 140.0, 160.0]) == [100.0, 120.0, 140.0, 160.0]

def test_dedupe_line_ys_merges_exactly_coincident_samples():
    """Two fragments of one ruled line can both clamp to the identical
    fitted y (a 0.0 gap), not just a couple px apart. If that zero gap also
    happens to be the smaller side at the biggest jump, small_side <= 0
    used to bail out of the merge entirely (0 is not "clearly bimodal" by
    the old guard's own math), leaving the duplicate line in line_ys -- the
    same inflated-line-count bug this function exists to fix. A zero small
    side is a valid (indeed the clearest possible) duplicate cluster, not
    a reason to give up."""
    raw = [100.0, 100.0, 130.0, 160.0]
    deduped = _dedupe_line_ys(raw)
    assert deduped == [100.0, 130.0, 160.0]

def test_dedupe_line_ys_short_lists_pass_through_unchanged():
    assert _dedupe_line_ys([]) == []
    assert _dedupe_line_ys([100.0]) == [100.0]
    assert _dedupe_line_ys([100.0, 105.0]) == [100.0, 105.0]  # only 2 points -- nothing to compare

def test_staves_from_jsomr_dedupes_left_right_fragment_pairs_end_to_end():
    """Same scenario as above, but through the real staves_from_jsomr()
    entry point -- a stave whose staff-line detector emitted separate
    left-half and right-half records for each of 4 real lines."""
    records = []
    real_line_ys = [130.0, 160.0, 190.0, 220.0]
    for i, y in enumerate(real_line_ys):
        # Right half genuinely covers the stave's x_mid; left half doesn't,
        # so it gets clamped to its own edge a couple px off from y.
        records.append(_line(0, i * 2, 400, 900, [y] * 501, ulx=400, uly=int(y) - 5, lrx=900, lry=int(y) + 5))
        records.append(_line(0, i * 2 + 1, 50, 350, [y + 1.5] * 301, ulx=50, uly=int(y) - 4, lrx=350, lry=int(y) + 6))
    staves = staves_from_jsomr(records)
    assert len(staves) == 1
    assert len(staves[0].line_ys) == 4


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
