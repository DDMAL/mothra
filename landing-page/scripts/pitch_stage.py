"""
pitch_stage.py -- real pitch finding for the MEI encoding step.

Runs **algorithm #1** of the `pitch-finding/` submodule
(DDMAL/Standalone-Pitch-Finder, `scripts/pitch_finder.py`) over one page and
hands `encode_to_mei.build_mei()` two lookup tables:

  * `pitches_by_glyph`  -- glyph id -> [(pname, oct), ...], one entry per note
    component of that glyph, in `<nc>` order;
  * `clef_lines_by_glyph` -- clef glyph id -> MEI `<clef @line>`, derived from
    where the clef's own ink centroid actually sits on its stave.

Both are advisory: build_mei falls back, *per glyph*, to its pre-existing
geometric placeholder (`_step_from_y`/`_component_pitches`) for anything this
stage couldn't resolve, so a page with no staff-line data, no detected clef, or
no `pitch-finding/` checkout encodes exactly as it did before.

What algorithm #1 adds over that placeholder, in one sentence each:

  * the anchor is an **ink centroid of a per-class crop** of the glyph (a
    virga's notehead, not its stem; a podatus's bottom-left head, not the
    midpoint of both heads), instead of the glyph bbox's geometric center;
  * a multi-note neume is **decomposed** from that one anchor plus the
    cheatsheet's interval list, with the anchor bound to the note the crop
    actually isolated -- so a torculus's three notes are placed relative to the
    head that was measured, not relative to the whole shape's center;
  * the clef is the page's **own detected clef glyph**, resolved per stave and
    per reading position, instead of an assumed shape on an assumed line.

Why the tables are keyed by glyph id rather than replacing build_mei's own
component list: mothra's `<nc>` structure (count, `@tilt`/`@ligated` attrs,
`<liquescent/>`, per-component zone splitting) comes from
`assets/mei_encoding/{square,hufnagel}.csv` via neume_mapping.py and is what
Neon corrects against. This stage only supplies *pitch*; it never changes what
elements are emitted. The two happen to agree on component counts for
CSV-backed classes because `neume_shapes.load_neume_shapes()` is pointed at
that very same CSV (see `_shape_table`) -- when they don't agree (a
repeated-note neume that resolves to one note, an unknown class that falls back
to one), build_mei chains the remaining components off this stage's first
pitch by `@intm` instead of dropping to the placeholder wholesale.

Staff-line input, in preference order:
  1. the `staffline_detections` JSOMR records tasks_encode.py already resolves
     as its tier-1 stave source -- real per-line curve fits, so a note's step
     is read against the fitted line *at that note's x*;
  2. failing that, flat lines synthesized from each `StaveBbox.line_ys` (tier
     2/3 staves, and the rows `assign_glyphs_to_staves()` recovers), which is
     the same data the placeholder itself uses.
Either way the entries go through the submodule's own `staff_regroup`/`staff_io`
loaders, so its two-column regrouping and fragment-collapsing apply here too.

Nothing in this module is imported at backend/worker startup cost: the
submodule's modules (which pull in cv2 via `glyph_pixels`) are imported inside
`run_pitch_finding()`, so a checkout without `git submodule update --init` --
or a dev venv without opencv -- degrades to the placeholder with a log line
rather than breaking the import chain, exactly as staffline_stage.py does for
the staff-finding package.
"""

from __future__ import annotations

import io
import json
import statistics
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import MEI_ENCODING_DIR, PITCH_FINDING_DIR, PITCH_FINDING_ENABLED
from encode_to_mei import STAFF_LINES

# staffline_adapter.staves_from_jsomr() stamps this onto every StaveBbox it
# builds; it is how _staff_entries tells a tier-1 stave (whose real per-line
# JSOMR records are in hand) from a tier-2/3 or recovered one (whose only line
# data is the StaveBbox's own line_ys).
_JSOMR_STAVE_ID_PREFIX = "jsomr-stave-"

# Flags the algorithm raises that are worth counting in the job log rather than
# leaving to a debug JSON nobody reads. Every one of them marks a result that
# is still a pitch, just a weaker one -- see the submodule's "never silently
# degrade" note.
_REPORTED_FLAGS = (
    "pixel_anchor_unavailable",
    "anchor_x_fell_back_to_center",
    "approximate_unknown_shape",
    "shape_from_class_name",
    "sparse_stave_lines",
    "clef_after_glyph",
    "clef_octave_unconfigured",
)

_DEFAULT_NOTATION = "square"


@dataclass
class PitchFindingResult:
    """What build_mei needs, plus what the job log should say about it."""
    pitches_by_glyph: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    clef_lines_by_glyph: dict[str, int] = field(default_factory=dict)
    log_lines: list[str] = field(default_factory=list)
    # "pixel_centroid" (an image was available) | "bbox_span" (geometry only)
    # | None (the stage did not run at all)
    source: Optional[str] = None
    resolved: int = 0      # glyphs this stage pitched
    considered: int = 0    # glyphs it was asked about (clefs and text included)


def _ensure_import_path() -> None:
    """Put the submodule's flat module directory on sys.path.

    APPENDED, not inserted at the front: those modules import each other by
    bare name (`import clef_rules`, `from staff_io import Stave`), the same
    flat layout staff-finding uses, and none of those names exist in
    landing-page/scripts -- but appending means a future collision resolves in
    favour of mothra's own module rather than silently shadowing it.
    """
    d = str(PITCH_FINDING_DIR)
    if d not in sys.path:
        sys.path.append(d)


def _ic_glyphs(glyphs: list) -> tuple[list, list[str]]:
    """(ic_io.Glyph list, parallel list of mothra glyph ids).

    encode_to_mei.Glyph and ic_io.Glyph carry the same seven GameraXML fields
    off the same `<glyph>` elements; they differ only in their identity field
    (a uuid string vs. the XML position). The returned id list is what maps a
    GlyphResult back to the glyph build_mei is about to emit.
    """
    from ic_io import Glyph as IcGlyph

    ic, ids = [], []
    for i, g in enumerate(glyphs):
        ic.append(IcGlyph(
            index=i, ulx=g.ulx, uly=g.uly, nrows=g.nrows, ncols=g.ncols,
            class_name=g.class_name, confidence=g.confidence, state=g.state,
        ))
        ids.append(g.id)
    return ic, ids


def _median_spacing(ys: list[float]) -> float:
    gaps = [b - a for a, b in zip(ys, ys[1:]) if b > a]
    return statistics.median(gaps) if gaps else 0.0


def _staff_entries(staves: list, jsomr_records: Optional[list[dict]]) -> list[dict]:
    """Staff-finding-shaped entries for staff_io/staff_regroup to load.

    Tier-1 JSOMR records pass through with their `centerline_page` x bounds
    coerced back to int: staff_regroup.split_columns() builds a per-pixel
    coverage array with `range(x0, x1)`, and staffline_adapter.scale_jsomr_records()
    (the SF-7 resolution-mismatch path) leaves them as floats.

    Any stave *without* JSOMR records behind it -- a tier-2/3 stave, or a row
    `assign_glyphs_to_staves()` recovered and appended -- contributes one
    synthesized flat line per entry in its `line_ys`. `y_values` is deliberately
    two samples rather than one-per-column: `StaffLine.y_at_x()` clamps its index
    into the list, so for a horizontal line two entries answer every x in the
    span identically, without allocating a page-width array per line.

    `scale_unit` on a synthesized line is the stave's measured line-to-line
    spacing, not (as in real JSOMR) the median staffline *thickness* -- the only
    consumer is `Stave.continuous_step_at_y`'s one-line-at-this-x fallback,
    which reads it as pixels-per-line-gap.
    """
    entries: list[dict] = []
    for r in (jsomr_records or []):
        cl = r.get("centerline_page")
        if not cl or not cl.get("y_values"):
            continue
        entries.append({**r, "centerline_page": {
            "x_start": int(round(cl["x_start"])),
            "x_end": int(round(cl["x_end"])),
            "y_values": [float(y) for y in cl["y_values"]],
        }})
    have_jsomr = bool(entries)

    for idx, sb in enumerate(staves):
        if have_jsomr and str(sb.id).startswith(_JSOMR_STAVE_ID_PREFIX):
            continue
        ys = sorted(float(y) for y in (sb.line_ys or []))
        if len(ys) < 2 or int(sb.lrx) <= int(sb.ulx):
            continue
        spacing = _median_spacing(ys)
        for i, y in enumerate(ys):
            entries.append({
                "id": f"synth-s{idx:02d}-l{i:02d}",
                "centerline_page": {"x_start": int(sb.ulx), "x_end": int(sb.lrx),
                                     "y_values": [y, y]},
                "scale_unit": spacing,
                "column_id": None,
                "stave_id": idx,
                "within_stave_index": i,
            })
    return entries


def _shape_table(notation_type: Optional[str]):
    """Interval table, read from mothra's OWN bundled cheatsheet CSV.

    Deliberately not the submodule's `neumes-cheatsheet/` copy: the CSV in
    `assets/mei_encoding/` is the one neume_mapping.py builds build_mei's `<nc>`
    components from, so sourcing intervals from the same file is what keeps the
    two in step (same classes, same `@intm` values, same component counts).
    Both files descend from ic/'s csv-*_neume_level_newest.csv; mothra's carries
    extra rows (clef.f2, divisio.maior/finalis) and, per notation preset, a
    hufnagel variant the submodule has no copy of at all.
    """
    from neume_shapes import load_neume_shapes

    csv_path = MEI_ENCODING_DIR / f"{notation_type or _DEFAULT_NOTATION}.csv"
    if not csv_path.exists():
        csv_path = MEI_ENCODING_DIR / f"{_DEFAULT_NOTATION}.csv"
    return load_neume_shapes(csv_path)


def _page_gray(image_bytes: Optional[bytes]):
    """The page as a 2-D uint8 array, or None (which selects bbox-span mode).

    Decoded with PIL rather than `cv2.imdecode` -- PIL is already a hard
    dependency here and `.convert("L")` sidesteps any question of whether the
    buffer decodes as BGR or RGB. glyph_pixels only ever greyscales a 3-channel
    crop anyway, so a 2-D array is the shape it would arrive at regardless.
    """
    if not image_bytes:
        return None
    try:
        import numpy as np
        from PIL import Image
        return np.array(Image.open(io.BytesIO(image_bytes)).convert("L"))
    except Exception:  # noqa: BLE001 - an undecodable page is not a reason to
        # fail the encode; it only costs the pixel anchor (see run_pitch_finding's
        # own log line), and geometry still places every note.
        return None


def _clef_line_from_step(stave_step: float, staff_lines: int) -> tuple[int, bool]:
    """MEI `<clef @line>` for a clef measured at `stave_step`, and whether the
    value had to be clamped into range.

    staff_io's step 0 is the bottom-most *detected* line and one line is two
    steps, which is exactly MEI's @line counting-from-the-bottom convention
    offset by one: step 0 -> line 1.

    Emitting this at all is what keeps the MEI internally consistent. Verovio
    positions an `<nc>` from its pname/oct *against the declared clef line*, so
    a clef read off the page at line 1 but declared at line 3 renders every
    note of that stave two lines from where the ink is -- the same class of
    silent, page-wide offset encode_to_mei._stave_zone_bounds documents for
    zone height vs. line spacing.
    """
    line = round(stave_step / 2) + 1
    clamped = min(max(line, 1), staff_lines)
    return clamped, clamped != line


def run_pitch_finding(
    glyphs: list,
    staves: list,
    jsomr_records: Optional[list[dict]],
    image_bytes: Optional[bytes],
    notation_type: Optional[str] = None,
    tmp_dir: Optional[Path] = None,
    staff_lines: int = STAFF_LINES,
) -> PitchFindingResult:
    """Pitch every glyph on one page with algorithm #1.

    `staves` must be the list build_mei will actually use (i.e. the one
    `assign_glyphs_to_staves()` returned, which can be longer than the detected
    set), so a recovered row's synthesized lines are in play here too.

    Never raises: every failure mode -- missing submodule, missing opencv, no
    staff-line data, an error inside the algorithm -- comes back as an empty
    result plus a log line saying which one it was, and build_mei then encodes
    the page with its existing placeholder.
    """
    result = PitchFindingResult()
    if not PITCH_FINDING_ENABLED:
        result.log_lines.append(
            " pitch finding disabled (MOTHRA_PITCH_FINDING) -- using placeholder geometric pitch")
        return result
    if not glyphs:
        return result

    _ensure_import_path()
    try:
        from pitch_finder import find_pitches
        from staff_io import load_staves_with_report
    except Exception as e:  # noqa: BLE001 - no submodule checkout, or no cv2
        # (glyph_pixels imports it at module level). Both are "this deployment
        # can't run the real algorithm", not "this page failed".
        result.log_lines.append(
            f" [warn] pitch finding unavailable ({e}) -- falling back to placeholder geometric pitch."
            f" Check `git submodule update --init pitch-finding` and that opencv is installed.")
        return result

    try:
        entries = _staff_entries(staves, jsomr_records)
        if not entries:
            result.log_lines.append(
                " [warn] no staff-line geometry usable for pitch finding -- falling back to"
                " placeholder geometric pitch")
            return result

        shapes = _shape_table(notation_type)
        ic, ids = _ic_glyphs(glyphs)

        # staff_io only exposes a path-based loader, and going through it (rather
        # than rebuilding Stave objects here) is what keeps its regrouping,
        # fragment collapsing and step convention as the single implementation.
        # tmp_dir is _encode_one's own per-item scratch dir, already removed in
        # its finally; the mkdtemp fallback is for callers that don't have one
        # (never the cwd, which can be read-only in a container).
        with tempfile.TemporaryDirectory() as fallback_dir:
            scratch = Path(tmp_dir) if tmp_dir else Path(fallback_dir)
            entries_path = scratch / "pitch_staff_entries.json"
            entries_path.write_text(json.dumps(entries))
            pf_staves, report = load_staves_with_report(entries_path, regroup=True)
        if not pf_staves:
            result.log_lines.append(
                " [warn] pitch finding: staff-line regrouping produced no staves -- falling back to"
                " placeholder geometric pitch")
            return result
        if report is not None:
            # summary() is multi-line when a stave is missing a detected line;
            # flattened because each log line is one SSE frame in the job log.
            result.log_lines.append(
                " pitch finding staves: " + " ".join(report.summary().split()))

        image = _page_gray(image_bytes)
        result.source = "pixel_centroid" if image is not None else "bbox_span"
        if image is None:
            result.log_lines.append(
                " [warn] pitch finding has no decodable page image -- anchoring on bbox geometry"
                " instead of ink centroids")

        results = find_pitches(ic, pf_staves, shapes, image)
    except Exception as e:  # noqa: BLE001 - an encode job must still produce MEI
        result.log_lines.append(
            f" [warn] pitch finding failed ({e!r}) -- falling back to placeholder geometric pitch")
        return result

    reasons: dict[str, int] = {}
    flags: dict[str, int] = {}
    clef_lines_clamped = 0
    result.considered = len(results)

    for r in results:
        for flag in r.flags + r.stave_assignment_flags:
            if flag in _REPORTED_FLAGS:
                flags[flag] = flags.get(flag, 0) + 1
        if r.reason:
            reasons[r.reason] = reasons.get(r.reason, 0) + 1
            continue
        components = r.note_components
        if not components or any(nc.pitch is None for nc in components):
            reasons["no_pitch_resolved"] = reasons.get("no_pitch_resolved", 0) + 1
            continue

        gid = ids[r.glyph_index]
        is_clef = shapes.is_clef(r.ic["class_name"])
        if is_clef:
            # A clef's own <clef> element is emitted by build_mei from the
            # neume_mapping CSV; what this stage contributes is the LINE it
            # sits on, measured rather than assumed.
            step = components[0].stave_step
            if step is not None:
                line, clamped = _clef_line_from_step(step, staff_lines)
                result.clef_lines_by_glyph[gid] = line
                clef_lines_clamped += int(clamped)
            # Its pitch goes in the map too, and is inert for every clef class
            # mothra's CSV maps to <clef> (that element has no @pname, and
            # build_mei never consults pitch_map for it). It matters for the
            # ones the CSV does NOT list -- clef.g in real IC output -- which
            # build_mei falls back to emitting as a plain <neume><nc>: the
            # clef's own letter is a far better pitch for that <nc> than the
            # placeholder's read of an assumed clef.

        result.pitches_by_glyph[gid] = [
            (str(nc.pitch["pname"]).lower(), str(nc.pitch["oct"])) for nc in components
        ]
        # `resolved` counts NOTES pitched, so a clef -- whose entry above is a
        # fallback for a class mothra's CSV doesn't map -- isn't tallied twice
        # against the clef-line count in the same log line.
        result.resolved += 0 if is_clef else 1

    result.log_lines.append(
        f" pitch finding ({result.source}): pitched {result.resolved} glyph(s),"
        f" {len(result.clef_lines_by_glyph)} clef line(s) measured")
    if reasons:
        # "pitchless_symbol" is the expected majority (text bboxes, divisions,
        # Gamera junk) -- reported anyway so the ratio is visible rather than
        # inferred.
        result.log_lines.append(
            " pitch finding, glyphs left to the placeholder: "
            + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())))
    if flags:
        result.log_lines.append(
            " pitch finding weakened-result flags: "
            + ", ".join(f"{k}={v}" for k, v in sorted(flags.items())))
    if clef_lines_clamped:
        result.log_lines.append(
            f" [warn] pitch finding: {clef_lines_clamped} measured clef line(s) fell outside"
            f" 1..{staff_lines} and were clamped -- likely an undetected bottom staff line")
    return result
