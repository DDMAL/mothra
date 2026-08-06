"""Backs up the [trace] job-log instrumentation added alongside these tests
with real assertions, not just log lines someone has to read -- "bbox
information is carried securely to Neon without being overridden."

Starts from a hand-built JSOMR record set (same fixture style as
test_staffline_adapter.py) rather than a raw YOLO stave-boxes .txt: the
actual detection stage (staffline_stage.run_staffline_detection) imports
job_store -> auth_api, which connects to Postgres at import time (see
auth_api.py's module-level init_db() call) -- exactly the constraint
test_staffline_adapter.py's own docstring sidesteps by never importing
staffline_stage either. JSOMR is the next stage down (it's literally what
staffline_detections.jsomr_json stores), so this still covers the real
handoff this session's tracing cares about: stored detection data ->
staffline_adapter.staves_from_jsomr() -> encode_to_mei.build_mei() -> the
MEI zones Neon actually loads.

No DB, no Celery, no cv2/scipy/scikit-image.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from staffline_adapter import staves_from_jsomr  # noqa: E402
from encode_to_mei import assign_glyphs_to_staves, build_mei, trace_stave_zone_parity  # noqa: E402
from medieval_models import STAVE_FILENAME, STAVE_CLASS_MAP  # noqa: E402


def _line(stave_id, within_stave_index, x_start, x_end, y_values, ulx, uly, lrx, lry):
    """Same shape as test_staffline_adapter.py's _line(), trimmed to what
    this file needs (always a real detected box -- no interpolated-line
    cases here, that's already covered there)."""
    centerline_page = {"x_start": x_start, "x_end": x_end, "y_values": list(y_values)}
    return {
        "id": f"test_s{stave_id}_l{within_stave_index}",
        "source": "detected",
        "bounding_box": {"ulx": ulx, "uly": uly, "lrx": lrx, "lry": lry},
        "centerline": centerline_page,
        "centerline_page": centerline_page,
        "fit": {"method": "quadratic_huber", "coefficients": [0, 0, y_values[0]],
                "residual_mean": 0.1, "residual_max": 0.2,
                "n_pixels_used": len(y_values), "n_pixels_total": len(y_values)},
        "quality": {"confidence": None, "flags": []},
        "scale_unit": 10.0,
        "column_id": None,
        "stave_id": stave_id,
        "lines_detected": None, "lines_interpolated": None, "lines_expected": None,
        "rhythm_status": None,
        "within_stave_index": within_stave_index,
    }


def test_jsomr_to_mei_zones_survive_unchanged():
    """Two staves' worth of JSOMR records -> staves_from_jsomr() ->
    build_mei() -> parse the MEI zones back out and confirm they're
    byte-for-byte the same coordinates staves_from_jsomr() produced -- the
    exact integrity trace_stave_zone_parity() checks on every real encode."""
    records = []
    for i, y in enumerate([100, 120, 140, 160]):
        records.append(_line(0, i, 0, 700, [y] * 701, 50, y - 5, 750, y + 5))
    for i, y in enumerate([300, 320, 340]):
        records.append(_line(1, i, 0, 700, [y] * 701, 50, y - 5, 750, y + 5))

    staves = staves_from_jsomr(records)
    assert len(staves) == 2

    glyphs_by_stave, staves = assign_glyphs_to_staves([], staves, page_w=900, page_h=1300)
    mei_bytes = build_mei(glyphs_by_stave, staves, Path("page.jpg"), 900, 1300, "test")

    trace = trace_stave_zone_parity(staves, mei_bytes)
    assert len(trace) == 1
    assert "verified identical" in trace[0]
    assert "[warn]" not in trace[0]


def test_trace_flags_a_genuine_divergence():
    """Sanity check on trace_stave_zone_parity() itself: if the StaveBbox
    list handed to it doesn't match what's actually in the MEI (simulating
    something silently mutating stave geometry between resolution and
    encoding), it must say so loudly, not pass silently."""
    records = [_line(0, 0, 0, 700, [100] * 701, 50, 95, 750, 105)]
    staves = staves_from_jsomr(records)
    glyphs_by_stave, staves = assign_glyphs_to_staves([], staves, page_w=900, page_h=1300)
    mei_bytes = build_mei(glyphs_by_stave, staves, Path("page.jpg"), 900, 1300, "test")

    tampered = [s for s in staves]
    tampered[0].ulx = 99999  # pretend something moved this stave after encoding

    trace = trace_stave_zone_parity(tampered, mei_bytes)
    assert len(trace) == 1
    assert "[warn]" in trace[0]
    assert "diverged" in trace[0]


def test_medieval_stave_model_still_points_at_expected_checkpoint():
    """Catches a future accidental repoint of the bundled stave detector (or
    its merged-class-space slot) silently -- see CLAUDE.md's "Updating the
    bundled medieval models" section for how/why these would ever change."""
    assert STAVE_FILENAME == "stave_detector_fulldata.pt"
    assert STAVE_CLASS_MAP == {0: 2}  # single class -> merged slot 2 ("staves")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
