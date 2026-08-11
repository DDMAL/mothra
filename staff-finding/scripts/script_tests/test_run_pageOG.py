"""Sanity check for the parts of run_page.py that don't need the BGR model."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Import only the pieces that don't trigger the inference_simple import.
# We do this by importing from the module file directly, bypassing the
# sys.path injection at module top.
import importlib.util
import types

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent


def _load_stubbed_run_page():
    """Stub inference_simple/torch, then load run_pageOG.py under the
    isolated name "run_page_partial" (bypassing run_page.py's own
    bgr_adapter/inference_simple import chain -- see module docstring).
    Returns (module, MonkeyPatch) so a caller can undo the stubs after use.

    Deliberately NOT done at module top level: that runs during pytest's
    COLLECTION phase, which imports every test file's top-level code across
    the whole run *before* executing any test -- so a module-level stub can
    still leak into another file's own collection-time imports if that file
    is collected later in the same run, even with fixture-based teardown,
    since a fixture only fires once execution of *this* file's own tests
    begins, well after collection of every file has already finished.
    Called from the run_page fixture below (for teardown) or directly from
    __main__ (no teardown needed -- the process exits right after). Returns
    the module rather than binding it at module level, since it's loaded
    under a synthetic name ("run_page_partial") that a plain `import
    run_page` elsewhere couldn't retrieve anyway.
    """
    sys.path.insert(0, str(_SCRIPTS_DIR))  # so run_pageOG.py's sibling imports resolve

    mp = pytest.MonkeyPatch()

    fake_inference = types.ModuleType("inference_simple")
    fake_inference.load_model = lambda *a, **kw: None
    fake_inference.sliding_window_inference = lambda *a, **kw: None
    fake_inference.post_process_ink = lambda *a, **kw: None
    fake_inference.separate_layers = lambda *a, **kw: (None, None)
    mp.setitem(sys.modules, "inference_simple", fake_inference)

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    mp.setitem(sys.modules, "torch", fake_torch)

    spec = importlib.util.spec_from_file_location(
        "run_page_partial", str(_SCRIPTS_DIR / "run_pageOG.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, mp


@pytest.fixture(scope="module")
def run_page():
    module, mp = _load_stubbed_run_page()
    yield module
    mp.undo()


def test_parse_yolo_txt(run_page):
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
    detections = run_page.parse_yolo_txt(path)
    print(f"parsed {len(detections)} detections from sample")
    assert len(detections) == 5, f"expected 5 parseable lines, got {len(detections)}"

    class_2 = run_page.filter_to_class(detections, 2)
    print(f"  filtered to class 2: {len(class_2)}")
    assert len(class_2) == 3


def test_yolo_to_pixel_box(run_page):
    d = run_page.YoloDetection(
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


def test_compute_scale_unit(run_page):
    # Three detections with heights 20, 30, 40 in pixel space at 1000x1000.
    detections = [
        run_page.YoloDetection(2, 0.5, 0.1, 0.5, 0.02),  # h = 20
        run_page.YoloDetection(2, 0.5, 0.3, 0.5, 0.03),  # h = 30
        run_page.YoloDetection(2, 0.5, 0.5, 0.5, 0.04),  # h = 40
    ]
    scale = run_page.compute_page_scale_unit(detections, 1000, 1000)
    print(f"scale unit (median of 20,30,40): {scale}")
    assert scale == 30.0


def test_crop_with_padding_clamping(run_page):
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
    _run_page, _mp = _load_stubbed_run_page()
    test_parse_yolo_txt(_run_page)
    test_yolo_to_pixel_box(_run_page)
    test_compute_scale_unit(_run_page)
    test_crop_with_padding_clamping(_run_page)
    print("\nAll driver sanity checks passed.")
