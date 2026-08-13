"""Regression test for yolo_inference.py's conf= passthrough.

infer()/infer_text_music()/infer_staves() used to call the Ultralytics model
bare (no conf=), so Ultralytics' own internal default (0.25) governed
candidate generation regardless of what a caller configured; _append_boxes()
only discarded boxes below the configured threshold afterward, in Python.
That meant the effective confidence floor could never go below 0.25 no matter
what was configured. The fix passes conf= directly into the model call,
matching infer_staves_raw_boxes()'s already-correct pattern.

yolo_inference.py imports models_api at module scope, which imports auth_api
(DB side effects at import). This stubs models_api as a bare stand-in in
sys.modules before importing yolo_inference, mirroring
test_tasks_text_batch_logs.py's stubbing pattern for the same reason.

test_tasks_text_batch_logs.py installs its own bare "yolo_inference" stub
(no YoloModelSet/_append_boxes) in sys.modules for its own purposes, and
pytest's module cache persists across test files within one session. If that
file is collected first, "yolo_inference" already points at the stub by the
time this file imports it, so the real module never gets loaded here. Popping
any pre-existing entry first forces a fresh import of the real module,
regardless of collection order; this doesn't disturb tasks_text_batch's
already-imported reference to the stub's functions.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "models_api" not in sys.modules:
    models_api_stub = types.ModuleType("models_api")
    models_api_stub.get_model_file_path = lambda *a, **k: None
    sys.modules["models_api"] = models_api_stub

sys.modules.pop("yolo_inference", None)
from yolo_inference import YoloModelSet, _append_boxes  # noqa: E402


class _TensorLike(list):
    """Minimal stand-in for a torch tensor slice: real Box.xywhn[0] supports
    .tolist(), a plain list doesn't."""

    def tolist(self):
        return list(self)


class FakeBox:
    def __init__(self, cls, conf, xywhn):
        self.cls = [cls]
        self.conf = [conf]
        self.xywhn = [_TensorLike(xywhn)]


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYoloModel:
    """Records the conf= it was called with; returns a fixed empty result."""

    def __init__(self):
        self.last_conf = None
        self.call_count = 0

    def __call__(self, img_arr, conf=None, device=None, verbose=False):
        self.last_conf = conf
        self.call_count += 1
        return [FakeResult(boxes=[])]


def _medieval_model_set(tm_threshold, st_threshold):
    tm_model, st_model = FakeYoloModel(), FakeYoloModel()
    model_set = YoloModelSet(
        medieval_models=(tm_model, st_model),
        class_maps=({0: 0, 1: 1}, {0: 2}),
        single_model=None, custom_cls_map=None,
        tm_threshold=tm_threshold, tm_device="cpu",
        st_threshold=st_threshold, st_device="cpu",
        confidence_threshold=0.5, device="cpu",
        stored_model_id="medieval", model_label="medieval", model_hash=None,
    )
    return model_set, tm_model, st_model


def test_infer_passes_conf_to_both_medieval_models():
    model_set, tm_model, st_model = _medieval_model_set(
        tm_threshold=0.10, st_threshold=0.20,
    )
    model_set.infer(img_arr=object())
    assert tm_model.last_conf == 0.10
    assert st_model.last_conf == 0.20


def test_infer_text_music_passes_conf_to_text_music_model_only():
    model_set, tm_model, st_model = _medieval_model_set(
        tm_threshold=0.10, st_threshold=0.20,
    )
    model_set.infer_text_music(img_arr=object())
    assert tm_model.last_conf == 0.10
    assert st_model.call_count == 0


def test_infer_staves_passes_conf_to_stave_model_only():
    model_set, tm_model, st_model = _medieval_model_set(
        tm_threshold=0.10, st_threshold=0.20,
    )
    model_set.infer_staves(img_arr=object())
    assert st_model.last_conf == 0.20
    assert tm_model.call_count == 0


def test_infer_below_quarter_threshold_reaches_the_model_call():
    """The whole point of the fix: a threshold below Ultralytics' own
    internal 0.25 default must reach the actual model call, not get
    silently clamped."""
    model_set, tm_model, st_model = _medieval_model_set(
        tm_threshold=0.05, st_threshold=0.05,
    )
    model_set.infer(img_arr=object())
    assert tm_model.last_conf == 0.05
    assert st_model.last_conf == 0.05


def test_infer_custom_model_passes_confidence_threshold():
    single_model = FakeYoloModel()
    model_set = YoloModelSet(
        medieval_models=None, class_maps=None,
        single_model=single_model, custom_cls_map={0: 0},
        tm_threshold=0.5, tm_device="cpu",
        st_threshold=0.5, st_device="cpu",
        confidence_threshold=0.15, device="cpu",
        stored_model_id="model-1", model_label="custom", model_hash="hash",
    )
    model_set.infer(img_arr=object())
    assert single_model.last_conf == 0.15


def test_append_boxes_maps_class_and_formats_line():
    lines = []
    boxes = [
        FakeBox(cls=0, conf=0.9, xywhn=[0.1, 0.2, 0.3, 0.4]),
        FakeBox(cls=1, conf=0.8, xywhn=[0.5, 0.6, 0.7, 0.8]),
    ]
    _append_boxes(lines, FakeResult(boxes=boxes), {0: 2})
    assert lines == ["2 0.100000 0.200000 0.300000 0.400000"]


def test_append_boxes_skips_when_no_boxes():
    lines = []
    _append_boxes(lines, FakeResult(boxes=[]), {0: 0})
    assert lines == []
    _append_boxes(lines, FakeResult(boxes=None), {0: 0})
    assert lines == []
