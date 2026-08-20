"""Regression test for mothra#260: "resized images result in offset in text
boxes (again?)".

SF-2 (tasks_predict.py) made text-finding/staffline-detection run against
project_images.original_data (full resolution) whenever it's present, rather
than the resized working copy (project_images.data) every viewer's imageSrc
used to point at unconditionally. syl_boxes/JSOMR bounding boxes are
absolute-pixel, so a viewer that loads the working copy but scales stored
boxes computed against the (larger) original drifts every box further off
the further it sits from the top-left corner -- exactly the symptom in the
issue's screenshot.

The fix: track which project_images column produced a given text_alignments/
staffline_detections row (storage_variant, "original"/"working_copy") and
have projects_api._image_src build a URL that matches -- /original when the
row says "original", the plain working-copy route otherwise.

This only tests the pure URL-selection logic in projects_api.py (_image_src,
_map_text_alignment_row, _map_staffline_row), not the SQL/DB wiring around it.
projects_api imports auth_api/job_store at module level, both of which touch
a live Postgres or require MOTHRA_SECRET at import time -- stubbed here as
bare module stand-ins, mirroring test_resolve_hints_staleness.py's and
test_tasks_text_batch_logs.py's pattern for tasks_encode/tasks_text_batch.

No DB, no FastAPI app, no real auth.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _install_stubs():
    """Installs auth_api/job_store stand-ins, or patches in whatever
    projects_api.py needs from them onto an already-installed stub. Several
    test modules in this directory install their own partial auth_api/
    job_store stub the same way (test_resolve_hints_staleness.py,
    test_tasks_text_batch_logs.py) -- since sys.modules is process-global,
    whichever test file's stub installs first "wins" for the rest of the
    pytest session, so every attribute added here is add-if-missing, never
    an outright replacement that would strip another test's own stub of
    something it needs."""
    class _JobCancelled(Exception):
        pass

    if "auth_api" not in sys.modules:
        sys.modules["auth_api"] = types.ModuleType("auth_api")
    auth_api_stub = sys.modules["auth_api"]
    # Superset of every attribute any auth_api-stubbing test file in this
    # directory needs (test_resolve_hints_staleness.py,
    # test_tasks_text_batch_logs.py) -- add-if-missing so this file can run
    # in either collection order without starving whichever test's own
    # _install_stubs() only fires "if auth_api not in sys.modules" and
    # therefore wouldn't otherwise get a chance to add what it needs.
    # projects_api.py imports _log_activity by name at module level, so
    # something must exist here regardless of stub-install order -- given
    # the same real-INSERT implementation test_tasks_text_batch_logs.py's own
    # hasattr-guarded install uses, so whichever of the two wins the race is
    # functionally identical (that test's FakeCursor only ever sees a plain
    # `cur.execute(sql, params)` call either way).
    def _log_activity(cur, project_id, action_type, detail=""):
        cur.execute(
            "INSERT INTO activity_log (project_id, action_type, detail) VALUES (%s, %s, %s)",
            (project_id, action_type, detail),
        )

    for name, val in {
        "get_current_user": lambda: None,
        "db_cursor": lambda: None,
        "require_project_owner": lambda *a, **k: None,
        "_log_activity": _log_activity,
        "MODELS_DIR": Path("/tmp"),
        "get_db_conn": lambda: None,
        "release_db_conn": lambda con: None,
        "get_latest_text_alignment": lambda *a, **k: None,
    }.items():
        if not hasattr(auth_api_stub, name):
            setattr(auth_api_stub, name, val)

    if "job_store" not in sys.modules:
        sys.modules["job_store"] = types.ModuleType("job_store")
    job_store_stub = sys.modules["job_store"]
    # Same superset reasoning as auth_api above, covering
    # test_resolve_hints_staleness.py / test_staffline_stage_flags.py /
    # test_tasks_text_batch_logs.py's own job_store stubs.
    for name, val in {
        "get_active_job_for_project": lambda *a, **k: None,
        "publish_event": lambda *a, **k: None,
        "fetch_upload": lambda *a, **k: None,
        "drop_upload": lambda *a, **k: None,
        "session_put": lambda *a, **k: None,
        "check_cancelled": lambda *a, **k: None,
        "JobCancelled": _JobCancelled,
    }.items():
        if not hasattr(job_store_stub, name):
            setattr(job_store_stub, name, val)


_install_stubs()

from projects_api import _image_src, _map_text_alignment_row, _map_staffline_row  # noqa: E402


def test_image_src_none_when_no_image_id():
    assert _image_src(None, "original") is None
    assert _image_src(None, "working_copy") is None


def test_image_src_original_variant_uses_original_route():
    assert _image_src("img-1", "original") == "/api/images/img-1/original"


def test_image_src_working_copy_variant_uses_plain_route():
    assert _image_src("img-1", "working_copy") == "/api/images/img-1"


def test_image_src_defaults_to_working_copy_for_legacy_none():
    # Rows written before the storage_variant column existed (or before
    # staffline_stage.py's storage_variant param was added) come back with
    # a NULL/None settings_json key -- must not be mistaken for "original".
    assert _image_src("img-1", None) == "/api/images/img-1"


def test_map_text_alignment_row_picks_url_matching_storage_variant():
    original_row = _map_text_alignment_row(
        "tid-1", "img-1", "folio1.png", 42.0, 5, "original",
    )
    assert original_row["imageSrc"] == "/api/images/img-1/original"

    working_copy_row = _map_text_alignment_row(
        "tid-2", "img-2", "folio2.png", 42.0, 5, "working_copy",
    )
    assert working_copy_row["imageSrc"] == "/api/images/img-2"

    # Default param (no explicit storage_variant passed) matches every
    # pre-existing call site that predates this column.
    legacy_row = _map_text_alignment_row("tid-3", "img-3", "folio3.png", 42.0, 5)
    assert legacy_row["imageSrc"] == "/api/images/img-3"


def test_map_staffline_row_picks_url_matching_storage_variant():
    original_row = _map_staffline_row(
        "did-1", "img-1", "folio1.png", 4, 5, "succeeded",
        storage_variant="original",
    )
    assert original_row["imageSrc"] == "/api/images/img-1/original"

    working_copy_row = _map_staffline_row(
        "did-2", "img-2", "folio2.png", 4, 5, "succeeded",
        storage_variant="working_copy",
    )
    assert working_copy_row["imageSrc"] == "/api/images/img-2"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
