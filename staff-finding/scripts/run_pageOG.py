"""
Page-level driver for the Stage 1 component filter.

Given a page image, its YOLO detection output, and (optionally) the BGR
(background removal) model checkpoint, this script:

  1. Loads the page (RGB).
  2. Optionally runs BGR inference once on the full page, producing an
     ink-on-white image. With --no-bgr, the original page is used directly.
  3. Parses the YOLO .txt detections and filters to the staffline class.
  4. For each staffline detection: crops the (BGR-processed or raw) page, runs
     the component filter, saves a diagnostic PNG.
  5. Writes a per-page summary CSV.

Usage:
    python run_page.py \\
        --page /path/to/page.jpg \\
        --yolo /path/to/page.txt \\
        --bgr-model /path/to/best_model.pth \\
        --output /path/to/output_dir \\
        [--staffline-class 2] \\
        [--crop-padding 2] \\
        [--device cpu] \\
        [--no-bgr]

When --no-bgr is set, --bgr-model is not required and BGR is skipped entirely.
Output goes to <output>/<page_name>_no_bgr/ so it doesn't clobber a BGR run.

See ADR-001 for component-filter design decisions.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from component_filter import filter_components

# Adapter pulls helpers from the existing inference script. The path below is a
# placeholder — adjust to wherever inference_simple.py lives in your environment,
# or pip-install it as a package and import normally.
import sys

INFERENCE_SCRIPT_DIR = "/Users/kyriebouressa/Documents/muscrat/layer_sep/scripts"
# mini: /Users/kyriebouressa/Documents/muscrat/layer_sep/scripts
# macbook: "/Users/ekaterina/Documents/Documents - angantyr/muscrat/layer_sep/scripts/"
sys.path.insert(0, INFERENCE_SCRIPT_DIR)
from inference_simple import (  # noqa: E402  (sys.path insertion above)
    load_model,
    sliding_window_inference,
    post_process_ink,
    separate_layers,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_STAFFLINE_CLASS = 2
DEFAULT_CROP_PADDING_PX = 2  # small margin around YOLO box; see driver plan
DEFAULT_BGR_WINDOW_SIZE = 512
DEFAULT_BGR_STRIDE = 256
DEFAULT_BGR_CONFIDENCE = 0.5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class YoloDetection:
    """One detection parsed from a YOLO-format .txt line."""

    class_id: int
    x_center_norm: float
    y_center_norm: float
    width_norm: float
    height_norm: float

    def to_pixel_box(
        self, image_width: int, image_height: int
    ) -> tuple[int, int, int, int]:
        """Convert normalized coords to pixel (ulx, uly, lrx, lry)."""
        cx = self.x_center_norm * image_width
        cy = self.y_center_norm * image_height
        w = self.width_norm * image_width
        h = self.height_norm * image_height
        ulx = int(round(cx - w / 2))
        uly = int(round(cy - h / 2))
        lrx = int(round(cx + w / 2))
        lry = int(round(cy + h / 2))
        return ulx, uly, lrx, lry


# ---------------------------------------------------------------------------
# YOLO parsing
# ---------------------------------------------------------------------------


def parse_yolo_txt(yolo_path: Path) -> list[YoloDetection]:
    """Parse a YOLO .txt file. One detection per line: class cx cy w h."""
    detections = []
    with yolo_path.open("r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"  Skipping malformed line {line_num} in {yolo_path}: {line!r}")
                continue
            try:
                detections.append(
                    YoloDetection(
                        class_id=int(parts[0]),
                        x_center_norm=float(parts[1]),
                        y_center_norm=float(parts[2]),
                        width_norm=float(parts[3]),
                        height_norm=float(parts[4]),
                    )
                )
            except ValueError:
                print(
                    f"  Skipping unparseable line {line_num} in {yolo_path}: {line!r}"
                )
    return detections


def filter_to_class(
    detections: list[YoloDetection],
    class_id: int,
) -> list[YoloDetection]:
    return [d for d in detections if d.class_id == class_id]


# ---------------------------------------------------------------------------
# BGR adapter
# ---------------------------------------------------------------------------


def run_bgr_inference(
    model,
    image_rgb: np.ndarray,
    window_size: int = DEFAULT_BGR_WINDOW_SIZE,
    stride: int = DEFAULT_BGR_STRIDE,
    confidence: float = DEFAULT_BGR_CONFIDENCE,
    device: str = "cpu",
) -> np.ndarray:
    """Run the BGR model on a full RGB page and return the ink-on-white layer.

    This is the in-memory equivalent of inference_simple.process_image, without
    the disk writes or the parchment/comparison outputs. We want the ink layer
    only.
    """
    probability_map = sliding_window_inference(
        model,
        image_rgb,
        window_size=window_size,
        stride=stride,
        device=device,
    )
    ink_mask = post_process_ink(probability_map, confidence_threshold=confidence)
    ink_layer, _parchment_layer = separate_layers(image_rgb, ink_mask)
    return ink_layer


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------


def crop_with_padding(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    padding: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop image to box plus padding, clamped to image bounds.

    Returns the cropped array and the actual pixel box used (after clamping).
    """
    h, w = image.shape[:2]
    ulx, uly, lrx, lry = box
    ulx_p = max(0, ulx - padding)
    uly_p = max(0, uly - padding)
    lrx_p = min(w, lrx + padding)
    lry_p = min(h, lry + padding)
    crop = image[uly_p:lry_p, ulx_p:lrx_p]
    return crop, (ulx_p, uly_p, lrx_p, lry_p)


# ---------------------------------------------------------------------------
# Scale unit derivation
# ---------------------------------------------------------------------------


def compute_page_scale_unit(
    detections: list[YoloDetection],
    image_width: int,
    image_height: int,
) -> float:
    """Page-level scale unit h: median pixel height of staffline boxes.

    A tight detector box's height is roughly the line thickness plus a small
    slack. Median across the page is more robust than mean for the occasional
    skewed/tall box.
    """
    heights = []
    for d in detections:
        _, uly, _, lry = d.to_pixel_box(image_width, image_height)
        heights.append(lry - uly)
    if not heights:
        return 0.0
    return float(np.median(heights))


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def process_page(
    page_path: Path,
    yolo_path: Path,
    bgr_model_path: Optional[Path],
    output_dir: Path,
    staffline_class: int = DEFAULT_STAFFLINE_CLASS,
    crop_padding: int = DEFAULT_CROP_PADDING_PX,
    device: str = "cpu",
    bgr_window_size: int = DEFAULT_BGR_WINDOW_SIZE,
    bgr_stride: int = DEFAULT_BGR_STRIDE,
    bgr_confidence: float = DEFAULT_BGR_CONFIDENCE,
    use_bgr: bool = True,
) -> None:
    """Run the full page pipeline. See module docstring for sequence."""
    page_name = page_path.stem
    # Tag output dir with the BGR/no-BGR variant so the two coexist.
    variant_suffix = "" if use_bgr else "_no_bgr"
    page_output_dir = output_dir / f"{page_name}{variant_suffix}"
    page_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load page ---
    print(f"Loading page: {page_path}")
    bgr_loaded = cv2.imread(str(page_path))
    if bgr_loaded is None:
        raise FileNotFoundError(f"Could not read page image: {page_path}")
    page_rgb = cv2.cvtColor(bgr_loaded, cv2.COLOR_BGR2RGB)
    h, w = page_rgb.shape[:2]
    print(f"  Page size: {w} x {h}")

    # --- Parse YOLO ---
    print(f"Parsing YOLO detections: {yolo_path}")
    all_detections = parse_yolo_txt(yolo_path)
    print(f"  Total detections: {len(all_detections)}")
    stafflines = filter_to_class(all_detections, staffline_class)
    print(f"  Stafflines (class {staffline_class}): {len(stafflines)}")
    if not stafflines:
        print("  No stafflines on this page; nothing to do.")
        return

    # --- Compute scale unit ---
    h_scale = compute_page_scale_unit(stafflines, w, h)
    print(f"  Page scale unit h (median staffline box height): {h_scale:.1f} px")

    # --- BGR or skip ---
    if use_bgr:
        if bgr_model_path is None:
            raise ValueError("BGR is enabled but no --bgr-model was provided.")
        print(f"Loading BGR model: {bgr_model_path}")
        bgr_model = load_model(str(bgr_model_path), device)
        print("Running BGR inference (this can take a minute on CPU)...")
        page_for_crops = run_bgr_inference(
            bgr_model,
            page_rgb,
            window_size=bgr_window_size,
            stride=bgr_stride,
            confidence=bgr_confidence,
            device=device,
        )
        # Save the BGR-processed page for inspection.
        processed_save_path = page_output_dir / f"{page_name}_bgr.png"
        cv2.imwrite(
            str(processed_save_path), cv2.cvtColor(page_for_crops, cv2.COLOR_RGB2BGR)
        )
        print(f"  Saved BGR-processed page to {processed_save_path}")
    else:
        print("BGR disabled (--no-bgr): cropping directly from original page.")
        page_for_crops = page_rgb

    # --- Per-box processing ---
    summary_rows = []
    print(f"Processing {len(stafflines)} staffline boxes...")
    for idx, detection in enumerate(stafflines):
        box = detection.to_pixel_box(w, h)
        crop, actual_box = crop_with_padding(page_for_crops, box, crop_padding)

        if crop.size == 0:
            print(f"  Box {idx}: degenerate crop (empty), skipping.")
            continue

        diag_path = page_output_dir / f"box_{idx:04d}.png"
        result = filter_components(
            crop=crop,
            scale_unit=h_scale,
            save_path=diag_path,
        )

        summary_rows.append(
            {
                "box_index": idx,
                "ulx": actual_box[0],
                "uly": actual_box[1],
                "lrx": actual_box[2],
                "lry": actual_box[3],
                "n_kept_pixels": len(result.coords),
                "flags": ";".join(result.flags),
                "n_discarded": len(result.discarded),
                "top_score": _top_score_of(result),
                "diagnostic_path": str(diag_path.relative_to(output_dir)),
            }
        )

    # --- Write summary CSV ---
    summary_path = page_output_dir / "summary.csv"
    fieldnames = [
        "box_index",
        "ulx",
        "uly",
        "lrx",
        "lry",
        "n_kept_pixels",
        "flags",
        "n_discarded",
        "top_score",
        "diagnostic_path",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote per-page summary: {summary_path}")
    print(f"Outputs under: {page_output_dir}")


def _top_score_of(result) -> Optional[float]:
    """Pull the kept component's total score from a ComponentFilterResult."""
    for entry in result.score_breakdown.values():
        if entry.get("kept"):
            return float(entry["total"])
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run Stage 1 component filter on every staffline detection of a page."
    )
    parser.add_argument("--page", type=Path, required=True, help="Page image path")
    parser.add_argument("--yolo", type=Path, required=True, help="YOLO .txt path")
    parser.add_argument(
        "--bgr-model",
        type=Path,
        required=False,
        help="BGR model checkpoint (required unless --no-bgr is set)",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output directory root"
    )
    parser.add_argument(
        "--staffline-class",
        type=int,
        default=DEFAULT_STAFFLINE_CLASS,
        help=f"YOLO class id for stafflines (default: {DEFAULT_STAFFLINE_CLASS})",
    )
    parser.add_argument(
        "--crop-padding",
        type=int,
        default=DEFAULT_CROP_PADDING_PX,
        help=f"Pixels of padding around YOLO box (default: {DEFAULT_CROP_PADDING_PX})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for BGR inference (cuda/cpu)",
    )
    parser.add_argument("--bgr-window-size", type=int, default=DEFAULT_BGR_WINDOW_SIZE)
    parser.add_argument("--bgr-stride", type=int, default=DEFAULT_BGR_STRIDE)
    parser.add_argument("--bgr-confidence", type=float, default=DEFAULT_BGR_CONFIDENCE)
    parser.add_argument(
        "--no-bgr",
        action="store_true",
        help="Skip BGR preprocessing entirely; crop directly from original page.",
    )
    args = parser.parse_args()

    use_bgr = not args.no_bgr
    if use_bgr and args.bgr_model is None:
        parser.error("--bgr-model is required unless --no-bgr is set.")

    process_page(
        page_path=args.page,
        yolo_path=args.yolo,
        bgr_model_path=args.bgr_model,
        output_dir=args.output,
        staffline_class=args.staffline_class,
        crop_padding=args.crop_padding,
        device=args.device,
        bgr_window_size=args.bgr_window_size,
        bgr_stride=args.bgr_stride,
        bgr_confidence=args.bgr_confidence,
        use_bgr=use_bgr,
    )


if __name__ == "__main__":
    main()
