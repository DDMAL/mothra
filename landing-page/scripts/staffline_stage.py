"""
Staffline detection: connected-component filtering, centerline fitting, and
stave grouping for the stave-class ("staves", merged slot 2) YOLO boxes a
predict job already produced earlier in the same request.

Mirrors text_api.stream_text_finding's shape: a generator yielding raw event
dicts and persisting its own result (to staffline_detections) on completion.
Called from tasks_predict.py's per-image loop, right after write_annotation().

Deliberately does not import staff-finding/scripts/run_page.py or
bgr_adapter.py -- both unconditionally import an external, unvendored
ink-separation ("BGR") dependency at module load time, which isn't available
outside a handful of developer machines. This stage runs Stage 1/2 directly
on raw page crops; ink-separation is a deferred, pluggable future stage (see
CLAUDE.md's "Staffline detection" section).

Heavy imports (component_filter/fit_centerline/group_staves, which pull in
cv2/scipy/scikit-image via the mothra-staff-finding package) are deferred into
run_staffline_detection()'s body, not this module's top level -- so merely
importing staffline_stage (which happens transitively at backend startup via
inference_api -> tasks_predict) can never fail on a machine that hasn't yet
`pip install -e staff-finding/`; the failure only surfaces the first time
run_staffline_detection() actually runs, inside the try/except below, exactly
like a stream_text_finding failure would.
"""

import json
import uuid as _uuid
from typing import Iterator, Optional

import numpy as np

from job_store import check_cancelled, JobCancelled

# Merged 0-indexed class space slot for "staves" -- matches
# medieval_models.STAVE_CLASS_MAP / yolo_inference.CATEGORY_TO_SLOT["staves"]
# and staff-finding/scripts/run_page.py's own DEFAULT_STAFFLINE_CLASS.
STAFFLINE_CLASS_ID = 2

# Matches staff-finding/scripts/run_page.py's DEFAULT_CROP_PADDING_PX.
CROP_PADDING_PX = 2


def has_class(yolo_txt: str, class_id: int) -> bool:
    """True if any line of a merged YOLO-txt string is the given class.

    Used by tasks_predict.py to skip entering run_staffline_detection()'s
    per-image loop entirely when a page has no stave-class boxes at all.
    """
    target = str(class_id)
    for line in yolo_txt.splitlines():
        line = line.strip()
        if line and line.split(None, 1)[0] == target:
            return True
    return False


def _package_version() -> Optional[str]:
    try:
        from importlib.metadata import version
        return version("mothra-staff-finding")
    except Exception:
        return None


def _compute_page_scale_unit(detections, image_width: int, image_height: int) -> float:
    """Median pixel height of staffline boxes -- ported verbatim from
    staff-finding/scripts/run_page.py's compute_page_scale_unit (that module
    can't be imported directly; see this file's module docstring)."""
    heights = []
    for d in detections:
        _, uly, _, lry = d.to_pixel_box(image_width, image_height)
        heights.append(lry - uly)
    return float(np.median(heights)) if heights else 0.0


def _crop_with_padding(image: np.ndarray, box, padding: int):
    """Crop image to box plus padding, clamped to image bounds. Ported
    verbatim from staff-finding/scripts/run_page.py's crop_with_padding."""
    h, w = image.shape[:2]
    ulx, uly, lrx, lry = box
    ulx_p = max(0, ulx - padding)
    uly_p = max(0, uly - padding)
    lrx_p = min(w, lrx + padding)
    lry_p = min(h, lry + padding)
    crop = image[uly_p:lry_p, ulx_p:lrx_p]
    return crop, (ulx_p, uly_p, lrx_p, lry_p)


def _assemble_jsomr_records(image_name, fit_results, boxes, grouping_result, scale_unit):
    """Build per-line JSOMR-shaped records for both detected and (if enabled)
    interpolated lines. Modeled on staff-finding/experiments/shared_utils.py's
    write_jsomr -- unlike run_page.py's own _write_jsomr_json, that function
    correctly emits interpolated lines and QA fields (lines_detected/
    lines_interpolated/lines_expected/rhythm_status), which matters here since
    interpolate_missing is a caller-controlled parameter, not hardcoded off.
    The "fit" sub-dict shape matches run_page.py's real-FitResult version
    (method/coefficients/residual_mean/residual_max/n_pixels_used/
    n_pixels_total), not shared_utils's own ExperimentFitResult-shaped one.
    """
    asg_by_fit = {a.fit_index: a for a in grouping_result.assignments}

    stave_detected: dict = {}
    for asg in grouping_result.assignments:
        if asg.stave_id is not None:
            stave_detected[asg.stave_id] = stave_detected.get(asg.stave_id, 0) + 1
    stave_interpolated: dict = {}
    for il in grouping_result.interpolated_lines:
        stave_interpolated[il.stave_id] = stave_interpolated.get(il.stave_id, 0) + 1

    records = []
    for idx, (fit, box) in enumerate(zip(fit_results, boxes)):
        ulx, uly, lrx, lry = box
        asg = asg_by_fit.get(idx)
        stave_id = asg.stave_id if asg else None
        centerline_page = {
            "x_start": int(fit.x_start + fit.x_page_offset),
            "x_end": int(fit.x_end + fit.x_page_offset),
            "y_values": [round(float(y) + fit.y_page_offset, 1) for y in fit.y_values],
        }
        records.append({
            "id": f"{image_name}_line{idx:04d}",
            "source": "fallback_redetected" if "fallback_redetected" in fit.flags else "detected",
            "bounding_box": {"ulx": ulx, "uly": uly, "lrx": lrx, "lry": lry},
            "centerline": {"x_start": fit.x_start, "x_end": fit.x_end, "y_values": fit.y_values},
            "centerline_page": centerline_page,
            "fit": {
                "method": "quadratic_huber",
                "coefficients": fit.coefficients,
                "residual_mean": round(fit.residual_mean, 3),
                "residual_max": round(fit.residual_max, 3),
                "n_pixels_used": fit.n_pixels_used,
                "n_pixels_total": fit.n_pixels_total,
            },
            "quality": {"confidence": None, "flags": list(fit.flags)},
            "scale_unit": scale_unit,
            "column_id": None,
            "stave_id": stave_id,
            "lines_detected": stave_detected.get(stave_id) if stave_id is not None else None,
            "lines_interpolated": stave_interpolated.get(stave_id, 0) if stave_id is not None else None,
            "lines_expected": grouping_result.mode_lines_per_stave,
            "rhythm_status": (
                grouping_result.rhythm_anomalies.get(stave_id, {}).get("status")
                if stave_id is not None else None
            ),
            "within_stave_index": asg.within_stave_index if asg else None,
        })

    # Interpolated lines carry page-absolute x_start/x_end/y_values already
    # (there's no crop to be local to) -- see interpolate_staves.InterpolatedLine's
    # own docstring. centerline and centerline_page get the identical dict so a
    # reader doesn't need to branch on `source` to know which field is populated.
    for interp in grouping_result.interpolated_lines:
        centerline_page = {
            "x_start": interp.x_start,
            "x_end": interp.x_end,
            "y_values": [round(float(y), 1) for y in interp.y_values],
        }
        records.append({
            "id": f"{image_name}_interp_s{interp.stave_id:02d}_l{interp.within_stave_index}",
            "source": "interpolated",
            "bounding_box": None,
            "centerline": centerline_page,
            "centerline_page": centerline_page,
            "fit": {"method": "interpolated", "neighbor_fit_indices": list(interp.neighbor_fit_indices)},
            "quality": {"confidence": None, "flags": ["interpolated"]},
            "scale_unit": scale_unit,
            "column_id": None,
            "stave_id": interp.stave_id,
            "lines_detected": stave_detected.get(interp.stave_id, 0),
            "lines_interpolated": stave_interpolated.get(interp.stave_id, 0),
            "lines_expected": grouping_result.mode_lines_per_stave,
            "rhythm_status": grouping_result.rhythm_anomalies.get(interp.stave_id, {}).get("status"),
            "within_stave_index": interp.within_stave_index,
        })

    records.sort(key=lambda r: (
        r["stave_id"] if r["stave_id"] is not None else float("inf"),
        r["within_stave_index"] if r["within_stave_index"] is not None else float("inf"),
    ))
    return records


def run_staffline_detection(
    job_id: str, cur, con,
    project_id: int, image_id: str, image_name: str, annotation_id: str,
    image_arr: np.ndarray, yolo_txt: str,
    interpolate_missing: bool = False,
) -> Iterator[dict]:
    """Run staffline detection for one image, yielding {"type": "log"|"error",
    "message": ...} event dicts and persisting the result to
    staffline_detections on completion (or a status='failed' row -- see
    below -- when boxes existed but detection itself errored).

    interpolate_missing is plumbed through, default off, matching
    staff-finding/dox/STATUS.md's "not yet validated across the corpus"
    caveat -- flipping it later is a default change, not a re-plumbing job.

    tasks_predict.py's own per-image check_cancelled(job_id) call only runs
    once per image; a page with many stave boxes can spend real time in the
    per-box loop below with no cancellation observed in between, so this
    checks again on every box. JobCancelled is deliberately re-raised, not
    swallowed by the broad except below -- a cancelled job must actually
    stop, not be recorded as one failed staffline-detection attempt among
    many while the outer per-image loop keeps going.
    """
    from yolo_io import parse_yolo_lines, filter_to_class
    from component_filter import filter_components
    from fit_centerline import fit_centerline
    from group_staves import group_staves

    detections = filter_to_class(
        parse_yolo_lines(yolo_txt.splitlines(), source=f"{image_name} annotation"),
        STAFFLINE_CLASS_ID,
    )
    if not detections:
        return

    h, w = image_arr.shape[:2]
    scale_unit = _compute_page_scale_unit(detections, w, h)

    try:
        fit_results = []
        boxes = []
        for det in detections:
            check_cancelled(job_id)
            box = det.to_pixel_box(w, h)
            crop, actual_box = _crop_with_padding(image_arr, box, CROP_PADDING_PX)
            if crop.size == 0:
                continue
            filter_result = filter_components(crop, scale_unit=scale_unit)
            fit_result = fit_centerline(filter_result, scale_unit=scale_unit)
            fit_result.x_page_offset = float(actual_box[0])
            fit_result.y_page_offset = float(actual_box[1])
            fit_results.append(fit_result)
            boxes.append(actual_box)

        if not fit_results:
            yield {
                "type": "log",
                "message": f"{image_name}: {len(detections)} stave box(es) found but all crops were degenerate; skipping staffline detection",
            }
            return

        grouping_result = group_staves(
            fit_results, scale_unit=scale_unit, interpolate_missing=interpolate_missing,
        )
        records = _assemble_jsomr_records(image_name, fit_results, boxes, grouping_result, scale_unit)
        stave_ids = {r["stave_id"] for r in records if r["stave_id"] is not None}
        settings = {
            "interpolate_missing": interpolate_missing,
            "binarization": "sauvola",
            "bgr_enabled": False,  # ink-separation deferred this pass, see module docstring
            "package_version": _package_version(),
        }

        cur.execute(
            "INSERT INTO staffline_detections"
            " (id, project_id, image_id, image_name, annotation_id, jsomr_json,"
            "  scale_unit, stave_count, mode_lines_per_stave, settings_json, status)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'succeeded')",
            (
                _uuid.uuid4().hex, project_id, image_id, image_name, annotation_id,
                json.dumps(records), scale_unit, len(stave_ids),
                grouping_result.mode_lines_per_stave, json.dumps(settings),
            ),
        )
        con.commit()

        message = (
            f"{image_name}: staffline detection found {len(records)} line(s) across "
            f"{len(stave_ids)} stave(s) (mode {grouping_result.mode_lines_per_stave} lines/stave)"
        )
        if grouping_result.rhythm_anomalies:
            message += f", {len(grouping_result.rhythm_anomalies)} stave(s) flagged for review"
        yield {"type": "log", "message": message}
    except JobCancelled:
        con.rollback()
        raise
    except Exception as e:
        con.rollback()
        try:
            cur.execute(
                "INSERT INTO staffline_detections"
                " (id, project_id, image_id, image_name, annotation_id, jsomr_json,"
                "  scale_unit, stave_count, mode_lines_per_stave, settings_json, status)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'failed')",
                (
                    _uuid.uuid4().hex, project_id, image_id, image_name, annotation_id,
                    json.dumps([]), scale_unit, 0, None, json.dumps({"error": str(e)}),
                ),
            )
            con.commit()
        except Exception:
            con.rollback()
        yield {"type": "error", "message": str(e)}
