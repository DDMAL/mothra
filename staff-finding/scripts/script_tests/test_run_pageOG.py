"""Sanity check for the parts of run_page.py that don't need the BGR model."""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/claude")
# Import only the pieces that don't trigger the inference_simple import.
# We do this by importing from the module file directly, bypassing the
# sys.path injection at module top.
import importlib.util

spec = importlib.util.spec_from_file_location(
    "run_page_partial", "/home/claude/run_page.py"
)
# Stub out imports the test environment doesn't have.
import types

fake_inference = types.ModuleType("inference_simple")
fake_inference.load_model = lambda *a, **kw: None
fake_inference.sliding_window_inference = lambda *a, **kw: None
fake_inference.post_process_ink = lambda *a, **kw: None
fake_inference.separate_layers = lambda *a, **kw: (None, None)
sys.modules["inference_simple"] = fake_inference

fake_torch = types.ModuleType("torch")
fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules["torch"] = fake_torch

run_page = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_page)


def test_parse_yolo_txt():
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


def test_yolo_to_pixel_box():
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


def test_compute_scale_unit():
    # Three detections with heights 20, 30, 40 in pixel space at 1000x1000.
    detections = [
        run_page.YoloDetection(2, 0.5, 0.1, 0.5, 0.02),  # h = 20
        run_page.YoloDetection(2, 0.5, 0.3, 0.5, 0.03),  # h = 30
        run_page.YoloDetection(2, 0.5, 0.5, 0.5, 0.04),  # h = 40
    ]
    scale = run_page.compute_page_scale_unit(detections, 1000, 1000)
    print(f"scale unit (median of 20,30,40): {scale}")
    assert scale == 30.0


def test_crop_with_padding_clamping():
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
    test_parse_yolo_txt()
    test_yolo_to_pixel_box()
    test_compute_scale_unit()
    test_crop_with_padding_clamping()
    print("\nAll driver sanity checks passed.")
