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

# SF-7: aspect-ratio tolerance for treating a predict-time/encode-time
# dimension mismatch as "the same page at a different resolution" (safe to
# scale) rather than "a genuinely different image" (reject, fall through to
# tier 2). A resize (e.g. imageResize.ts's client-side downscale, or SF-2's
# original_data vs. working-copy pixel-dimension difference) preserves aspect
# ratio; a wrong/stale image generally won't.
ASPECT_RATIO_TOLERANCE = 0.02


def scale_jsomr_records(jsomr_records: list[dict], scale_x: float, scale_y: float) -> list[dict]:
    """Returns a new list of JSOMR records with every absolute-pixel field
    (bounding_box, centerline_page) scaled by (scale_x, scale_y) -- used when
    the predict-time image these coordinates were computed against differs
    in resolution from the encode-time image tasks_encode.py is actually
    building MEI zones against (see SF-7 in ALPHA_TRANSITION_PLAN.md).

    Deliberately does not touch `centerline` (crop-local coordinates, unused
    by staves_from_jsomr/y_values_at) or any non-geometric field -- only the
    two page-absolute-pixel fields those functions actually read.
    """
    scaled = []
    for r in jsomr_records:
        r = dict(r)
        bbox = r.get("bounding_box")
        if bbox:
            r["bounding_box"] = {
                "ulx": bbox["ulx"] * scale_x, "uly": bbox["uly"] * scale_y,
                "lrx": bbox["lrx"] * scale_x, "lry": bbox["lry"] * scale_y,
            }
        cl_page = r.get("centerline_page")
        if cl_page:
            r["centerline_page"] = {
                "x_start": cl_page["x_start"] * scale_x,
                "x_end": cl_page["x_end"] * scale_x,
                "y_values": [y * scale_y for y in cl_page["y_values"]],
            }
        scaled.append(r)
    return scaled


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


def _dedupe_line_ys(line_ys: list[float]) -> list[float]:
    """Merge near-duplicate y-samples that are really the SAME real ruled
    line sampled twice, not two distinct lines.

    y_values_at() samples every line record at the stave's shared x_mid,
    clamping to that record's own detected x-extent when x_mid falls
    outside it (see that function's docstring). When the staffline
    detector splits one real, continuous ruled line into separate
    left-half/right-half fragment records (e.g. because a decorative
    initial or damage breaks the line mid-page), each fragment is a
    DIFFERENT JSOMR record with its own within_stave_index, so
    staves_from_jsomr has no way to know they're the same physical line
    -- and whichever fragment doesn't actually cover x_mid gets clamped
    to its own edge instead, landing a few pixels away from the other
    fragment's genuine x_mid sample.

    Confirmed on a real page where one stave's 4 real lines were split
    this way into 7 JSOMR records, inflating line_ys to 7 entries -- which
    badly corrupts encode_to_mei.py's clef-line lookup (_step_from_y),
    since it assumes len(line_ys) reflects the real line count. Every
    note on that stave rendered at a systematically wrong pitch as a
    result (confirmed: notes rendered a full line-spacing or more away
    from where they visually belong).

    Finds the split point via the biggest jump in the SORTED gap
    distribution, rather than a fixed fraction of the overall median gap
    -- a fixed-fraction-of-median threshold is sensitive to how many
    duplicate-pairs vs. real gaps are present (a page with more duplicate
    pairs than real gaps can drag the median itself down into the
    duplicate cluster, defeating a fixed-fraction cut entirely). The
    split is only trusted when the two sides are clearly bimodal (the
    larger side at least 3x the smaller side at the split) -- guards
    against merging real, uniformly-spaced lines on a normal stave, where
    the "biggest jump" between very-similar real gaps is just noise, not
    a genuine duplicate-vs-real distinction."""
    if len(line_ys) < 3:
        return line_ys
    gaps = [line_ys[i + 1] - line_ys[i] for i in range(len(line_ys) - 1)]
    sorted_gaps = sorted(gaps)
    if len(sorted_gaps) < 2:
        return line_ys
    jumps = [(sorted_gaps[i + 1] - sorted_gaps[i], i) for i in range(len(sorted_gaps) - 1)]
    biggest_jump, split_i = max(jumps, key=lambda t: t[0])
    small_side, large_side = sorted_gaps[split_i], sorted_gaps[split_i + 1]
    if biggest_jump <= 0 or small_side < 0 or (small_side > 0 and large_side < small_side * 3):
        return line_ys  # no clear bimodal split -- treat every gap as real
    threshold = (small_side + large_side) / 2

    merged: list[float] = [line_ys[0]]
    counts = [1]
    for y in line_ys[1:]:
        if y - merged[-1] < threshold:
            merged[-1] = (merged[-1] * counts[-1] + y) / (counts[-1] + 1)
            counts[-1] += 1
        else:
            merged.append(y)
            counts.append(1)
    return merged


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
        line_ys = _dedupe_line_ys(sorted(y_values_at(lines, x_mid)))

        staves.append(StaveBbox(
            id=f"jsomr-stave-{stave_id}",
            ulx=int(ulx), uly=int(uly), lrx=int(lrx), lry=int(lry),
            line_ys=line_ys,
        ))
    return staves
