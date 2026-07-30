"""
Adapts staffline_stage.py's rich, per-line JSOMR records into
encode_to_mei.py's StaveBbox shape, so tasks_encode.py can prefer this data
source over the coarser parse_yolo_stave_hints()/estimate_staves_from_glyphs()
fallback chain when it's available.

Deliberately separate from staffline_stage.py: this module only needs to read
a JSON blob back and do arithmetic on plain dicts, so it stays free of
cv2/scipy/scikit-image -- tasks_encode.py never needs the heavier staff-finding
dependencies just to build MEI from an already-computed detection.
"""

from encode_to_mei import StaveBbox


def y_values_at(stave_records: list[dict], x_page: float) -> list[float]:
    """Per-stave, per-column y lookup: for each line record (in the order
    given), the y-value of its fitted centerline at x_page (clamped to the
    line's own x_start/x_end if x_page falls outside its detected extent).

    centerline_page.y_values is one sample per integer pixel column starting
    at x_start (see staff-finding/scripts/fit_centerline.py / run_page.py's
    own docstring), so the lookup is a direct offset, not a fractional index
    -- matching staff-finding/scripts/interpolate_staves.py's own y_at_x
    helper exactly.

    Exported standalone (not inlined into staves_from_jsomr) since this is
    structurally the same per-column sampler a future pitch-finding stage
    would need (staff-finding/dox/PITCH_FINDING_NOTES.md) -- reuse, don't
    re-derive.
    """
    ys = []
    for rec in stave_records:
        cl = rec.get("centerline_page") or rec.get("centerline")
        if not cl or not cl.get("y_values"):
            continue
        x_start = cl["x_start"]
        y_values = cl["y_values"]
        x_clamped = max(x_start, min(x_page, cl["x_end"]))
        idx = max(0, min(int(round(x_clamped - x_start)), len(y_values) - 1))
        ys.append(float(y_values[idx]))
    return ys


def staves_from_jsomr(jsomr_records: list[dict]) -> list[StaveBbox]:
    """Group JSOMR per-line records by stave_id into StaveBbox groups.

    Records with stave_id=None are unassigned/reconciled-away outliers (per
    group_staves.StaveAssignment semantics), not a stave of their own, and
    are excluded. line_ys is sampled at each stave's own horizontal midpoint
    via y_values_at() -- a real per-line curve-fit value, replacing
    parse_yolo_stave_hints()'s naive linear interpolation between stave
    top/bottom (encode_to_mei._staves_from_staff_lines).
    """
    by_stave: dict[int, list[dict]] = {}
    for r in jsomr_records:
        if r.get("stave_id") is not None:
            by_stave.setdefault(r["stave_id"], []).append(r)

    staves = []
    for stave_id in sorted(by_stave):
        lines = sorted(
            by_stave[stave_id],
            key=lambda r: (r.get("within_stave_index") is None, r.get("within_stave_index")),
        )

        boxed = [r for r in lines if r.get("bounding_box")]
        if boxed:
            ulx = min(r["bounding_box"]["ulx"] for r in boxed)
            uly = min(r["bounding_box"]["uly"] for r in boxed)
            lrx = max(r["bounding_box"]["lrx"] for r in boxed)
            lry = max(r["bounding_box"]["lry"] for r in boxed)
        else:
            # All-interpolated stave (can't happen while interpolate_missing
            # stays off, but stay correct once it's turned on) -- fall back
            # to the centerline extent since there's no crop-based box.
            xs = [x for r in lines for x in (r["centerline_page"]["x_start"], r["centerline_page"]["x_end"])]
            ys = [y for r in lines for y in r["centerline_page"]["y_values"]]
            ulx, lrx = min(xs), max(xs)
            uly, lry = min(ys), max(ys)

        x_mid = (ulx + lrx) / 2
        line_ys = sorted(y_values_at(lines, x_mid))

        staves.append(StaveBbox(
            id=f"jsomr-stave-{stave_id}",
            ulx=int(ulx), uly=int(uly), lrx=int(lrx), lry=int(lry),
            line_ys=line_ys,
        ))
    return staves
