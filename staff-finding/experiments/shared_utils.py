"""
Shared utilities for staffline detection experiments.

Provides YOLO parsing, scale-unit computation, and JSOMR JSON writing so each
experiment runner doesn't have to reimplement them or import the full run_page
dependency chain.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# -- Reuse the existing YOLO I/O rather than duplicating it -----------------
_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
from yolo_io import YoloDetection, filter_to_class, parse_yolo_txt  # noqa: E402
from group_staves import (
    group_staves,
    _save_grouping_diagnostic,
    InterpolatedLine,
)  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal FitResult for experiment runners
# ---------------------------------------------------------------------------


@dataclass
class ExperimentFitResult:
    """Subset of FitResult needed to produce JSOMR output and run stave grouping.

    Coordinates follow the same convention as FitResult in fit_centerline.py:
    x_start/x_end and y_values are crop-local; page-absolute values are obtained
    by adding x_page_offset and y_page_offset respectively.
    """

    x_start: int = 0
    x_end: int = 0
    y_values: list[float] = field(default_factory=list)
    x_page_offset: float = 0.0
    y_page_offset: float = 0.0
    method: str = "unknown"
    meta: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers shared across runners
# ---------------------------------------------------------------------------


def load_page_gray(page_path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Load page image.  Returns (bgr, gray, width, height)."""
    bgr = cv2.imread(str(page_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read page image: {page_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = bgr.shape[:2]
    return bgr, gray, w, h


def compute_scale_unit(
    detections: list[YoloDetection],
    image_width: int,
    image_height: int,
) -> float:
    """Median pixel height of staffline boxes — same logic as run_page.py."""
    heights = []
    for d in detections:
        _, uly, _, lry = d.to_pixel_box(image_width, image_height)
        heights.append(lry - uly)
    return float(np.median(heights)) if heights else 0.0


def write_jsomr(
    page_name: str,
    fit_results: list[ExperimentFitResult],
    boxes: list[tuple[int, int, int, int]],
    grouping_result,
    scale_unit: float,
    save_path: Path,
) -> None:
    """Write JSOMR JSON matching the design-doc §5.8 schema."""
    asg_by_fit = {a.fit_index: a for a in grouping_result.assignments}

    # Pre-compute per-stave counts so every record can carry them at a glance.
    stave_detected: dict[int, int] = {}
    stave_interpolated: dict[int, int] = {}
    for asg in grouping_result.assignments:
        if asg.stave_id is not None:
            stave_detected[asg.stave_id] = stave_detected.get(asg.stave_id, 0) + 1
    for il in grouping_result.interpolated_lines:
        stave_interpolated[il.stave_id] = stave_interpolated.get(il.stave_id, 0) + 1

    records = []
    for idx, (fit, box) in enumerate(zip(fit_results, boxes)):
        ulx, uly, lrx, lry = box
        asg = asg_by_fit.get(idx)

        record = {
            "id": f"{page_name}_line{idx:04d}",
            "source": "detected",
            "bounding_box": {"ulx": ulx, "uly": uly, "lrx": lrx, "lry": lry},
            "centerline": {
                "x_start": fit.x_start,
                "x_end": fit.x_end,
                "y_values": [round(float(y), 1) for y in fit.y_values],
            },
            "fit": {
                "method": fit.method,
                **{
                    k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in fit.meta.items()
                },
            },
            "quality": {
                "confidence": None,
                "flags": fit.flags,
            },
            "scale_unit": scale_unit,
            "column_id": None,
            "stave_id": asg.stave_id if asg else None,
            "lines_detected": (
                stave_detected.get(asg.stave_id, 0)
                if asg and asg.stave_id is not None
                else None
            ),
            "lines_interpolated": (
                stave_interpolated.get(asg.stave_id, 0)
                if asg and asg.stave_id is not None
                else None
            ),
            "lines_expected": grouping_result.mode_lines_per_stave,
            "rhythm_status": (
                grouping_result.rhythm_anomalies.get(asg.stave_id, {}).get("status")
                if asg and asg.stave_id is not None
                else None
            ),
            "within_stave_index": asg.within_stave_index if asg else None,
        }
        records.append(record)

    # Append synthesized lines produced by the interpolation pass.
    for interp in grouping_result.interpolated_lines:
        record = {
            "id": f"{page_name}_interp_s{interp.stave_id:02d}_l{interp.within_stave_index}",
            "source": "interpolated",
            "bounding_box": None,
            "centerline": {
                "x_start": interp.x_start,
                "x_end": interp.x_end,
                "y_values": [round(float(y), 1) for y in interp.y_values],
            },
            "fit": {
                "method": "interpolated",
                "neighbor_fit_indices": list(interp.neighbor_fit_indices),
            },
            "quality": {
                "confidence": None,
                "flags": ["interpolated"],
            },
            "scale_unit": scale_unit,
            "column_id": None,
            "stave_id": interp.stave_id,
            "lines_detected": stave_detected.get(interp.stave_id, 0),
            "lines_interpolated": stave_interpolated.get(interp.stave_id, 0),
            "lines_expected": grouping_result.mode_lines_per_stave,
            "rhythm_status": grouping_result.rhythm_anomalies.get(
                interp.stave_id, {}
            ).get("status"),
            "within_stave_index": interp.within_stave_index,
        }
        records.append(record)

    # Sort by stave then by position within stave so each stave's lines are
    # contiguous in the file.  Unassigned lines (stave_id=None) go at the end.
    records.sort(
        key=lambda r: (
            r["stave_id"] if r["stave_id"] is not None else float("inf"),
            (
                r["within_stave_index"]
                if r["within_stave_index"] is not None
                else float("inf")
            ),
        )
    )

    with save_path.open("w") as f:
        json.dump(records, f, indent=2)


def run_grouping_and_save(
    page_name: str,
    fit_results: list[ExperimentFitResult],
    boxes: list[tuple[int, int, int, int]],
    scale_unit: float,
    page_bgr: np.ndarray,
    output_dir: Path,
    use_valley_threshold: bool = False,
    interpolation_max_gap: float | None = None,
) -> None:
    """Run stave grouping, save JSOMR + HQ diagnostic.  Mirrors run_page.py.

    use_valley_threshold: pass True to use valley-finding gap detection instead
    of the default median-based threshold.  See group_staves._find_valley_threshold
    for rationale.  Controlled per-runner rather than globally so each approach
    can be evaluated with and without it.
    """
    from fit_centerline import FitResult  # use real FitResult for group_staves

    # group_staves expects real FitResult objects — bridge the gap
    real_fits = []
    for ef in fit_results:
        rf = FitResult(
            x_start=ef.x_start,
            x_end=ef.x_end,
            y_values=ef.y_values,
            x_page_offset=ef.x_page_offset,
            y_page_offset=ef.y_page_offset,
            flags=ef.flags,
        )
        real_fits.append(rf)

    w, h = page_bgr.shape[1], page_bgr.shape[0]
    grouping_result = group_staves(
        fits=real_fits,
        scale_unit=scale_unit,
        interpolate_missing=True,
        interpolation_max_gap=interpolation_max_gap,
        page_size=(w, h),
        page_image=page_bgr,
        save_path=output_dir / f"{page_name}_stave_grouping.png",
        use_valley_threshold=use_valley_threshold,
    )

    # Print summary
    dist = grouping_result.line_count_distribution
    total_lines = sum(cnt * n for cnt, n in dist.items())
    total_staves = sum(dist.values())
    avg_lines = total_lines / total_staves if total_staves else 0.0
    print(
        f"  Stave grouping: {total_staves} staves  |  "
        f"mode={grouping_result.mode_lines_per_stave} lines/stave  |  "
        f"avg={avg_lines:.1f} lines/stave  |  "
        f"distribution={dict(sorted(dist.items()))}"
    )

    jsomr_path = output_dir / f"{page_name}_stafflines.json"
    write_jsomr(page_name, fit_results, boxes, grouping_result, scale_unit, jsomr_path)
    print(f"Wrote JSOMR: {jsomr_path}")
    print(f"Outputs under: {output_dir}")
