"""Sanity checks for the post-refactor pieces.

After the refactor:
  - yolo_io.py owns YoloDetection, parse_yolo_txt, filter_to_class
  - run_page.py owns crop_with_padding, compute_page_scale_unit
  - bgr_adapter.py owns the BGR helpers (not tested here; needs the model)
"""

import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _apply_stubs() -> pytest.MonkeyPatch:
    """Stub inference_simple/bgr_adapter/torch/ultralytics in sys.modules,
    then import yolo_io/run_page while the stubs are active, so both come
    from a real import cached in sys.modules for any later `import yolo_io`
    (cheap dict lookup, no re-execution) to retrieve.

    Deliberately NOT called at module top level: that runs during pytest's
    COLLECTION phase, which imports every test file's top-level code across
    the whole run *before* executing any test -- so a module-level stub can
    still leak into another file's own collection-time imports if that file
    is collected later in the same run, even with fixture-based teardown,
    since a fixture only fires once execution of *this* file's own tests
    begins, well after collection of every file has already finished.
    Called from a fixture (for teardown) or directly from __main__ (no
    teardown needed -- the process exits right after).
    """
    mp = pytest.MonkeyPatch()

    fake_inference = types.ModuleType("inference_simple")
    fake_inference.load_model = lambda *a, **kw: None
    fake_inference.sliding_window_inference = lambda *a, **kw: None
    fake_inference.post_process_ink = lambda *a, **kw: None
    fake_inference.separate_layers = lambda *a, **kw: (None, None)
    mp.setitem(sys.modules, "inference_simple", fake_inference)

    # bgr_adapter.py itself also needs stubbing, not just inference_simple: it
    # does its own os.path.isfile() check across a few hardcoded developer-machine
    # paths and raises ModuleNotFoundError directly if none exist, before ever
    # reaching its own "from inference_simple import ..." line -- so the
    # inference_simple stub above never even gets consulted on a machine (e.g. a
    # CI runner) that doesn't have one of those exact paths. Stubbing bgr_adapter
    # itself sidesteps that check entirely, the same way the inference_simple
    # stub sidesteps the module it fakes.
    fake_bgr_adapter = types.ModuleType("bgr_adapter")
    fake_bgr_adapter.load_bgr_model = lambda *a, **kw: None
    fake_bgr_adapter.run_bgr_inference = lambda *a, **kw: None
    fake_bgr_adapter.DEFAULT_BGR_WINDOW_SIZE = 512
    fake_bgr_adapter.DEFAULT_BGR_STRIDE = 256
    fake_bgr_adapter.DEFAULT_BGR_CONFIDENCE = 0.5
    mp.setitem(sys.modules, "bgr_adapter", fake_bgr_adapter)

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    mp.setitem(sys.modules, "torch", fake_torch)

    fake_ultralytics = types.ModuleType("ultralytics")

    class _FakeYOLO:
        def __init__(self, *a, **kw):
            pass

    fake_ultralytics.YOLO = _FakeYOLO
    mp.setitem(sys.modules, "ultralytics", fake_ultralytics)

    import yolo_io  # noqa: F401  (cached in sys.modules for callers to import by name)
    import run_page  # noqa: F401

    return mp


@pytest.fixture(scope="module", autouse=True)
def _stubbed_modules():
    mp = _apply_stubs()
    yield
    mp.undo()


def test_parse_yolo_txt():
    import yolo_io

    sample = """\
1 0.550555 0.350755 0.023019 0.029412
2 0.512238 0.351451 0.019814 0.016097
2 0.488636 0.342905 0.025059 0.032393
1 0.456438 0.333168 0.023602 0.030405

# stray comment that's actually not legal — should be skipped
malformed line
2 0.431671 0.333764 0.016026 0.032393
"""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(sample)
        path = Path(f.name)
    detections = yolo_io.parse_yolo_txt(path)
    print(f"parsed {len(detections)} detections from sample")
    assert len(detections) == 5, f"expected 5 parseable lines, got {len(detections)}"

    class_2 = yolo_io.filter_to_class(detections, 2)
    print(f"  filtered to class 2: {len(class_2)}")
    assert len(class_2) == 3


def test_yolo_to_pixel_box():
    import yolo_io

    d = yolo_io.YoloDetection(
        class_id=2,
        x_center_norm=0.5,
        y_center_norm=0.5,
        width_norm=0.4,
        height_norm=0.1,
    )
    box = d.to_pixel_box(image_width=1000, image_height=2000)
    # Center 500,1000; width 400, height 200. Box should be (300, 900, 700, 1100).
    print(f"box for centered 0.4x0.1 on 1000x2000: {box}")
    assert box == (300, 900, 700, 1100)


def test_compute_scale_unit():
    import run_page
    import yolo_io

    # Three detections with heights 20, 30, 40 in pixel space at 1000x1000.
    detections = [
        yolo_io.YoloDetection(2, 0.5, 0.1, 0.5, 0.02),  # h = 20
        yolo_io.YoloDetection(2, 0.5, 0.3, 0.5, 0.03),  # h = 30
        yolo_io.YoloDetection(2, 0.5, 0.5, 0.5, 0.04),  # h = 40
    ]
    scale = run_page.compute_page_scale_unit(detections, 1000, 1000)
    print(f"scale unit (median of 20,30,40): {scale}")
    assert scale == 30.0


def test_crop_with_padding_clamping():
    import run_page

    img = np.zeros((100, 200, 3), dtype=np.uint8)
    # Box near top-left corner; padding should clamp.
    box = (5, 5, 50, 50)
    crop, actual = run_page.crop_with_padding(img, box, padding=10)
    print(
        f"crop near corner with padding=10: actual box={actual}, crop shape={crop.shape}"
    )
    assert actual == (0, 0, 60, 60)
    assert crop.shape == (60, 60, 3)

    # Box near bottom-right; padding should clamp.
    box = (180, 90, 199, 99)
    crop, actual = run_page.crop_with_padding(img, box, padding=10)
    print(
        f"crop near far corner with padding=10: actual box={actual}, crop shape={crop.shape}"
    )
    assert actual == (170, 80, 200, 100)
    assert crop.shape == (20, 30, 3)


if __name__ == "__main__":
    _apply_stubs()
    test_parse_yolo_txt()
    test_yolo_to_pixel_box()
    test_compute_scale_unit()
    test_crop_with_padding_clamping()
    print("\nAll driver sanity checks passed.")
