"""Regression test for tasks_text_batch.py's per-folio log_text capture.

run_text_batch_task used to hard-code log_text="" on every text_alignments
row it inserted (the batch/Cantus-folio path behind POST
/api/projects/{id}/text-batch/run), unlike text_api.py's stream_text_finding
(the single-folio /predict path), which already accumulates every
{"type": "log"} SSE event into log_text. Since the batch path is this app's
default workflow, "view logs" in the Detected text viewer always showed "no
logs recorded for this run".

The fix buffers every published log line into `pending_logs` and snapshots +
resets that buffer at each folio_result boundary, so each folio's log_text
contains only lines emitted since the previous folio_result (batch-global
preamble -- model loading, resolution messages, stage announcements, and the
per-image YOLO/staffline logs that all run up front before any folio_result
comes back -- lands on the first folio only).

tasks_text_batch imports auth_api/job_store/celery_app/models_api/
yolo_inference/staffline_stage/text_api, several of which touch a live
Postgres/Redis or heavy ML deps (torch/ultralytics, cv2/scipy) at import
time. This stubs all of them as bare module stand-ins in sys.modules
*before* importing tasks_text_batch, mirroring
test_resolve_hints_staleness.py's pattern for tasks_encode.

No DB, no Celery, no real YOLO/staffline/text-service calls.
"""
import io
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image


def _install_stubs():
    """Installs minimal stand-ins for every module tasks_text_batch imports
    that would otherwise touch a live DB/broker or pull in heavy ML deps.
    Idempotent -- safe to call more than once in the same test session."""
    if "auth_api" not in sys.modules:
        auth_api_stub = types.ModuleType("auth_api")
        auth_api_stub.get_db_conn = lambda: None
        auth_api_stub.release_db_conn = lambda con: None
        sys.modules["auth_api"] = auth_api_stub

    if "job_store" not in sys.modules:
        job_store_stub = types.ModuleType("job_store")
        job_store_stub.publish_event = lambda *a, **k: None
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

    if "models_api" not in sys.modules:
        models_api_stub = types.ModuleType("models_api")
        models_api_stub.get_model_file_path = lambda *a, **k: None
        sys.modules["models_api"] = models_api_stub

    if "yolo_inference" not in sys.modules:
        yolo_inference_stub = types.ModuleType("yolo_inference")
        yolo_inference_stub.resolve_yolo_models = lambda *a, **k: (None, [])
        yolo_inference_stub.write_annotation = lambda *a, **k: "ann-id"
        sys.modules["yolo_inference"] = yolo_inference_stub

    if "staffline_stage" not in sys.modules:
        staffline_stage_stub = types.ModuleType("staffline_stage")
        staffline_stage_stub.run_staffline_detection = lambda *a, **k: iter([])
        staffline_stage_stub.has_class = lambda *a, **k: False
        staffline_stage_stub.STAFFLINE_CLASS_ID = 2
        sys.modules["staffline_stage"] = staffline_stage_stub

    if "text_api" not in sys.modules:
        text_api_stub = types.ModuleType("text_api")
        text_api_stub.TEXT_API_URL = "http://fake-text-service"
        text_api_stub._stream_multipart = lambda *a, **k: iter([])
        text_api_stub._music_boxes_for_image = lambda *a, **k: []
        text_api_stub._mask_json_for_image = lambda *a, **k: None
        sys.modules["text_api"] = text_api_stub


_install_stubs()

import tasks_text_batch  # noqa: E402


class FakeCursor:
    """Answers the project_images SELECT with canned rows and records every
    text_alignments INSERT's params for assertions."""

    def __init__(self, images_by_id):
        self.images_by_id = images_by_id
        self.inserted_alignments = []
        self._pending = None

    def execute(self, sql, params=()):
        if "FROM project_images" in sql:
            self._pending = self.images_by_id[params[0]]
        elif "INSERT INTO text_alignments" in sql:
            self.inserted_alignments.append(params)
            self._pending = None
        else:
            self._pending = None

    def fetchone(self):
        return self._pending

    def close(self):
        pass


class FakeConnection:
    def commit(self):
        pass

    def rollback(self):
        pass


class FakeYoloModels:
    stored_model_id = "model-1"
    model_label = "fake-model"
    model_hash = "hash-123"

    def infer(self, img_arr):
        return ""


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="PNG")
    return buf.getvalue()


def _fake_stream_multipart(url, fields=None, files=None, timeout=None):
    events = [
        {"type": "log", "message": "folio0-specific-log"},
        {"type": "folio_result", "image_index": 0,
         "text_alignment": {"median_line_spacing": 1.0, "syl_boxes": [{}, {}]}},
        {"type": "log", "message": "folio1-specific-log"},
        {"type": "folio_result", "image_index": 1,
         "text_alignment": {"median_line_spacing": 1.0, "syl_boxes": [{}]}},
        {"type": "result", "batchId": "b1", "fileCount": 2},
        {"type": "done"},
    ]
    for ev in events:
        yield "data: " + json.dumps(ev) + "\n"


def test_batch_run_scopes_log_text_per_folio(monkeypatch):
    png = _png_bytes()
    cursor = FakeCursor(images_by_id={
        "id-a": ("folioA.png", png, "image/png"),
        "id-b": ("folioB.png", png, "image/png"),
    })

    monkeypatch.setattr(tasks_text_batch, "get_db_conn", lambda: FakeConnection())
    monkeypatch.setattr(tasks_text_batch, "release_db_conn", lambda con: None)
    monkeypatch.setattr(FakeConnection, "cursor", lambda self: cursor, raising=False)
    monkeypatch.setattr(tasks_text_batch, "resolve_yolo_models",
                         lambda *a, **k: (FakeYoloModels(), []))
    monkeypatch.setattr(tasks_text_batch, "_stream_multipart", _fake_stream_multipart)

    body = {
        "image_ids": ["id-a", "id-b"],
        "folios": ["1r", "1v"],
        "model_preset": "medieval",
        "yolo_confidence_threshold": 0.5,
        "yolo_device": "cpu",
        "masking_enabled": False,
        "source_id": 42,
        "device": "cpu",
        "column_bimodal_threshold": 0.5,
        "mask_padding": 15,
    }

    tasks_text_batch.run_text_batch_task("job-1", 1, body)

    assert len(cursor.inserted_alignments) == 2
    folio0_log = cursor.inserted_alignments[0][-1]
    folio1_log = cursor.inserted_alignments[1][-1]

    # Folio 0 gets the batch-global preamble (per-image YOLO logs, the
    # Kraken/HTR stage announcement) plus its own text-service log line and
    # its own "aligned" summary.
    assert "layer separation done" in folio0_log
    assert "running Kraken segmentation" in folio0_log
    assert "folio0-specific-log" in folio0_log
    assert "syllable(s) aligned" in folio0_log
    assert "folio1-specific-log" not in folio0_log

    # Folio 1 gets ONLY its own lines -- no preamble, no folio 0 content.
    assert "folio1-specific-log" in folio1_log
    assert "syllable(s) aligned" in folio1_log
    assert "layer separation done" not in folio1_log
    assert "running Kraken segmentation" not in folio1_log
    assert "folio0-specific-log" not in folio1_log


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
