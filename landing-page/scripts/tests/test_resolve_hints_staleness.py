"""Regression test for tasks_encode.py's _resolve_hints() tier-1 staleness
guard (documentation_allons-y/STAFFLINE_INTEGRATION_FOLLOWUPS.md's
"stale staffline_detections row" bug).

_resolve_hints imports `tasks_encode`, which imports `auth_api`/`job_store` --
both of which connect to a live Postgres at *module import time*
(auth_api.py calls init_db() unconditionally on import, per CLAUDE.md's
Backend section). test_bbox_pipeline_integrity.py's own docstring
deliberately avoids importing tasks_encode/staffline_stage for exactly this
reason, staying one level below at staffline_adapter/encode_to_mei instead.

_resolve_hints' new staleness-guard logic lives inside tasks_encode.py
itself, so there's no lower-level module to test it through -- this file
stubs out `auth_api`/`job_store`/`celery_app` in sys.modules *before*
importing tasks_encode, so the import never touches a real DB or broker.
A hand-rolled fake cursor then answers each SQL query by inspecting which
table it targets, tracking exactly what _resolve_hints asked for.

No DB, no Celery, no cv2/scipy/scikit-image.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _install_db_free_stubs():
    """Installs minimal auth_api/job_store/celery_app stand-ins so importing
    tasks_encode doesn't try to reach a real Postgres/Redis. Idempotent --
    safe to call more than once in the same test session."""
    if "auth_api" not in sys.modules:
        auth_api_stub = types.ModuleType("auth_api")
        auth_api_stub.get_db_conn = lambda: None
        auth_api_stub.release_db_conn = lambda con: None

        def _stub_get_latest_text_alignment(cur, project_id, image_name, image_id=None):
            # Faithful-enough stand-in for auth_api.get_latest_text_alignment:
            # every test in this file passes text_alignment_row=None (this
            # file only exercises the staffline-detection staleness guard,
            # not text-alignment data), so this only ever needs to survive
            # FakeCursor routing the query by table name and returning
            # nothing -- but it's written to also handle a real row, in case
            # a future test in this file exercises that path.
            import json
            if image_id:
                cur.execute(
                    "SELECT alignment_json FROM text_alignments WHERE image_id=%s AND project_id=%s"
                    " ORDER BY created_at DESC LIMIT 1",
                    (image_id, project_id),
                )
            else:
                cur.execute(
                    "SELECT alignment_json FROM text_alignments WHERE image_name=%s AND project_id=%s"
                    " ORDER BY created_at DESC LIMIT 1",
                    (image_name, project_id),
                )
            row = cur.fetchone()
            if not row or not row[0]:
                return None
            try:
                parsed = json.loads(row[0])
            except (TypeError, ValueError):
                return None
            return parsed if isinstance(parsed, dict) else None

        auth_api_stub.get_latest_text_alignment = _stub_get_latest_text_alignment
        sys.modules["auth_api"] = auth_api_stub

    if "job_store" not in sys.modules:
        job_store_stub = types.ModuleType("job_store")
        job_store_stub.publish_event = lambda *a, **k: None
        job_store_stub.fetch_upload = lambda *a, **k: None
        job_store_stub.drop_upload = lambda *a, **k: None
        job_store_stub.session_put = lambda *a, **k: None
        job_store_stub.check_cancelled = lambda *a, **k: None

        class _JobCancelled(Exception):
            pass

        job_store_stub.JobCancelled = _JobCancelled
        sys.modules["job_store"] = job_store_stub

    if "celery_app" not in sys.modules:
        celery_app_stub = types.ModuleType("celery_app")

        class _FakeCeleryApp:
            def task(self, *a, **k):
                def _decorator(fn):
                    return fn
                return _decorator

        celery_app_stub.celery_app = _FakeCeleryApp()
        sys.modules["celery_app"] = celery_app_stub


_install_db_free_stubs()

import tasks_encode  # noqa: E402


class FakeCursor:
    """Answers _resolve_hints' three SELECTs by table name, and records every
    query it was asked (SQL substring + params) for assertions."""

    def __init__(self, annotations_row=None, staffline_row_by_annotation_id=None,
                 text_alignment_row=None):
        self.annotations_row = annotations_row
        self.staffline_row_by_annotation_id = staffline_row_by_annotation_id or {}
        self.text_alignment_row = text_alignment_row
        self.queries = []
        self._pending = None

    def execute(self, sql, params=()):
        self.queries.append((sql, params))
        if "FROM text_alignments" in sql:
            self._pending = self.text_alignment_row
        elif "FROM annotations" in sql:
            self._pending = self.annotations_row
        elif "FROM staffline_detections" in sql:
            # params = (image_name, project_id, annotation_id)
            annotation_id = params[2]
            self._pending = self.staffline_row_by_annotation_id.get(annotation_id)
        else:
            self._pending = None

    def fetchone(self):
        return self._pending

    def close(self):
        pass


class FakeConnection:
    pass


def _patch_db_conn(monkeypatch, cursor):
    monkeypatch.setattr(tasks_encode, "get_db_conn", lambda: FakeConnection())
    monkeypatch.setattr(tasks_encode, "release_db_conn", lambda con: None)
    monkeypatch.setattr(FakeConnection, "cursor", lambda self: cursor, raising=False)


def test_stale_staffline_detection_is_ignored_in_favor_of_current_annotation(monkeypatch):
    """Two staffline_detections rows exist for this image (accumulate-forever
    design): an OLDER one, newest by created_at, tied to an annotation_id
    that is no longer current (the image was re-annotated since). Tier 1
    must NOT return that stale row's geometry -- it should fall through to
    tier 2 (parse_yolo_stave_hints on the CURRENT annotation's own yolo_txt)
    instead of silently encoding against superseded stave data."""
    current_annotation_id = "ann-current"
    stale_annotation_id = "ann-old"

    stale_jsomr = [{"stave_id": 0, "bounding_box": {"ulx": 0, "uly": 0, "lrx": 10, "lry": 10},
                    "centerline_page": {"x_start": 0, "x_end": 10, "y_values": [5] * 11}}]

    cursor = FakeCursor(
        annotations_row=(current_annotation_id, "2 0.5 0.3 0.9 0.02\n2 0.5 0.35 0.9 0.02", "img-1"),
        staffline_row_by_annotation_id={stale_annotation_id: (stale_jsomr, {})},
        text_alignment_row=None,
    )
    monkeypatch.setattr(tasks_encode, "get_db_conn", lambda: FakeConnection())
    monkeypatch.setattr(tasks_encode, "release_db_conn", lambda con: None)
    monkeypatch.setattr(FakeConnection, "cursor", lambda self: cursor, raising=False)

    _text_alignment, _yolo_stave_hints, stave_source, _jsomr = tasks_encode._resolve_hints(
        project_id=1, image_name="page.jpg", page_w=1000, page_h=1000,
    )

    # Tier 1 must have been asked to match the CURRENT annotation_id, not
    # just ordered by created_at with no annotation filter at all.
    staffline_queries = [q for q in cursor.queries if "FROM staffline_detections" in q[0]]
    assert len(staffline_queries) == 1
    assert "annotation_id" in staffline_queries[0][0]
    assert staffline_queries[0][1][2] == current_annotation_id

    # The stale row (keyed to ann-old) must not have been used.
    assert stave_source != "staffline_detection"
    # Falls through to tier 2 instead (real yolo_txt was supplied above).
    assert stave_source == "yolo_annotation"


def test_current_staffline_detection_still_wins_when_it_matches():
    """Sanity check the fix doesn't over-correct: when the newest
    staffline_detections row DOES match the current annotation, tier 1
    should still win over tier 2, same as before this fix."""
    import pytest
    monkeypatch = pytest.MonkeyPatch()
    try:
        current_annotation_id = "ann-current"
        fresh_jsomr = [
            {"stave_id": 0, "within_stave_index": 0,
             "bounding_box": {"ulx": 0, "uly": 0, "lrx": 700, "lry": 10},
             "centerline_page": {"x_start": 0, "x_end": 700, "y_values": [5] * 701}},
        ]
        cursor = FakeCursor(
            annotations_row=(current_annotation_id, "2 0.5 0.3 0.9 0.02\n2 0.5 0.35 0.9 0.02", "img-1"),
            staffline_row_by_annotation_id={current_annotation_id: (fresh_jsomr, {})},
            text_alignment_row=None,
        )
        monkeypatch.setattr(tasks_encode, "get_db_conn", lambda: FakeConnection())
        monkeypatch.setattr(tasks_encode, "release_db_conn", lambda con: None)
        monkeypatch.setattr(FakeConnection, "cursor", lambda self: cursor, raising=False)

        _text_alignment, _yolo_stave_hints, stave_source, _jsomr = tasks_encode._resolve_hints(
            project_id=1, image_name="page.jpg", page_w=1000, page_h=1000,
        )

        assert stave_source == "staffline_detection"
        assert len(_yolo_stave_hints) == 1
    finally:
        monkeypatch.undo()


def test_frame_matching_dimensions_pass_through_unchanged(monkeypatch, capsys):
    """SF-7: predict-time and encode-time dimensions match exactly -- tier 1
    is used as-is, no scaling, no log noise."""
    current_annotation_id = "ann-current"
    jsomr = [{"stave_id": 0, "within_stave_index": 0,
              "bounding_box": {"ulx": 0, "uly": 0, "lrx": 700, "lry": 10},
              "centerline_page": {"x_start": 0, "x_end": 700, "y_values": [5.0] * 701}}]
    cursor = FakeCursor(
        annotations_row=(current_annotation_id, "2 0.5 0.3 0.9 0.02", "img-1"),
        staffline_row_by_annotation_id={
            current_annotation_id: (jsomr, {"predict_image_width": 1000, "predict_image_height": 1000}),
        },
        text_alignment_row=None,
    )
    _patch_db_conn(monkeypatch, cursor)

    _text_alignment, yolo_stave_hints, stave_source, _jsomr = tasks_encode._resolve_hints(
        project_id=1, image_name="page.jpg", page_w=1000, page_h=1000,
    )

    assert stave_source == "staffline_detection"
    assert yolo_stave_hints[0].ulx == 0 and yolo_stave_hints[0].lrx == 700  # unscaled
    assert "frame mismatch" not in capsys.readouterr().err


def test_frame_resize_is_scaled_not_rejected(monkeypatch, capsys):
    """SF-7: predict-time image was half the encode-time image's resolution
    (same aspect ratio, e.g. a client-side resize) -- coordinates are scaled
    up by 2x rather than either used unscaled (silently misplacing every
    stave) or rejected (unnecessarily falling through to tier 2)."""
    current_annotation_id = "ann-current"
    jsomr = [{"stave_id": 0, "within_stave_index": 0,
              "bounding_box": {"ulx": 0, "uly": 0, "lrx": 350, "lry": 5},
              "centerline_page": {"x_start": 0, "x_end": 350, "y_values": [2.5] * 351}}]
    cursor = FakeCursor(
        annotations_row=(current_annotation_id, "2 0.5 0.3 0.9 0.02", "img-1"),
        staffline_row_by_annotation_id={
            current_annotation_id: (jsomr, {"predict_image_width": 500, "predict_image_height": 500}),
        },
        text_alignment_row=None,
    )
    _patch_db_conn(monkeypatch, cursor)

    _text_alignment, yolo_stave_hints, stave_source, _jsomr = tasks_encode._resolve_hints(
        project_id=1, image_name="page.jpg", page_w=1000, page_h=1000,
    )

    assert stave_source == "staffline_detection"
    assert yolo_stave_hints[0].lrx == 700  # 350 * (1000/500) scale_x
    stderr = capsys.readouterr().err
    assert "scaled by" in stderr
    assert "frame mismatch" not in stderr


def test_frame_genuine_mismatch_is_rejected_and_logged(monkeypatch, capsys):
    """SF-7: predict-time and encode-time dimensions have a genuinely
    different aspect ratio (not just a resize) -- tier 1 is rejected and
    logged, falling through to tier 2, instead of silently misplacing every
    stave with a same-page-different-resolution scale factor that doesn't
    actually apply here."""
    current_annotation_id = "ann-current"
    jsomr = [{"stave_id": 0, "within_stave_index": 0,
              "bounding_box": {"ulx": 0, "uly": 0, "lrx": 700, "lry": 10},
              "centerline_page": {"x_start": 0, "x_end": 700, "y_values": [5.0] * 701}}]
    cursor = FakeCursor(
        annotations_row=(current_annotation_id, "2 0.5 0.3 0.9 0.02\n2 0.5 0.35 0.9 0.02", "img-1"),
        staffline_row_by_annotation_id={
            # 1000x1000 (square) predict-time vs 2000x500 (4:1) encode-time --
            # not a plain resize, aspect ratio is genuinely different.
            current_annotation_id: (jsomr, {"predict_image_width": 1000, "predict_image_height": 1000}),
        },
        text_alignment_row=None,
    )
    _patch_db_conn(monkeypatch, cursor)

    _text_alignment, yolo_stave_hints, stave_source, _jsomr = tasks_encode._resolve_hints(
        project_id=1, image_name="page.jpg", page_w=2000, page_h=500,
    )

    assert stave_source == "yolo_annotation"  # fell through to tier 2
    assert "frame mismatch" in capsys.readouterr().err


def test_tier1_failure_is_logged_not_silently_swallowed(monkeypatch, capsys):
    """SF-7: the tier-1 except used to be a bare `except: pass`. A malformed
    JSOMR row (missing required keys) must now be logged before falling
    through to tier 2, matching this function's other two lookups."""
    current_annotation_id = "ann-current"
    malformed_jsomr = [{"stave_id": 0}]  # missing bounding_box/centerline_page entirely
    cursor = FakeCursor(
        annotations_row=(current_annotation_id, "2 0.5 0.3 0.9 0.02\n2 0.5 0.35 0.9 0.02", "img-1"),
        staffline_row_by_annotation_id={current_annotation_id: (malformed_jsomr, {})},
        text_alignment_row=None,
    )
    _patch_db_conn(monkeypatch, cursor)

    _text_alignment, yolo_stave_hints, stave_source, _jsomr = tasks_encode._resolve_hints(
        project_id=1, image_name="page.jpg", page_w=1000, page_h=1000,
    )

    assert stave_source != "staffline_detection"
    stderr = capsys.readouterr().err
    assert "staffline-detection lookup failed" in stderr


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
