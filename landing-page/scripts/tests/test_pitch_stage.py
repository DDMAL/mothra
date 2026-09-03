"""Tests for pitch_stage.py (real pitch finding, algorithm #1 of the
pitch-finding/ submodule) and for encode_to_mei.py's consumption of its two
output tables.

Split in two halves:

  * everything that is mothra's own plumbing -- staff-entry building, the
    step-to-MEI-clef-line conversion, and build_mei's per-glyph choice between
    a supplied pitch, an @intm chain off a supplied first pitch, and the older
    geometric placeholder -- runs unconditionally, with no submodule and no
    opencv needed;
  * one end-to-end test drives the real algorithm over a synthetic page and is
    skipped when `pitch-finding/` isn't checked out (or opencv isn't installed,
    which its glyph_pixels imports at module level). That page is built so the
    expected pitches are derivable by hand: three noteheads two diatonic steps
    apart, and a C clef whose own ink sits on the middle one.

Follows test_encode_to_mei.py's convention: sys.path insert, bare module
import, plain pytest functions, no fixtures/conftest. No DB, no Celery.
"""
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import encode_to_mei as mei  # noqa: E402
import pitch_stage  # noqa: E402
from neume_mapping import NcTemplate  # noqa: E402

MEI_NS = "{http://www.music-encoding.org/ns/mei}"


def _stave(sid, uly, lry, ulx=0, lrx=500, line_ys=None):
    return mei.StaveBbox(id=sid, ulx=ulx, uly=uly, lrx=lrx, lry=lry, line_ys=line_ys or [])


def _glyph(gid, ulx, uly=110, ncols=20, nrows=20, class_name="neume.punctum"):
    return mei.Glyph(id=gid, ulx=ulx, uly=uly, ncols=ncols, nrows=nrows,
                      class_name=class_name, confidence=1.0, state="AUTOMATIC")


def _jsomr_line(stave_id, index, y, x_start=0, x_end=400):
    return {
        "id": f"line-{stave_id}-{index}",
        "centerline_page": {"x_start": x_start, "x_end": x_end, "y_values": [y, y]},
        "scale_unit": 5.0,
        "stave_id": stave_id,
        "within_stave_index": index,
    }


# --- _staff_entries: what the submodule's staff loader gets handed ---

def test_staff_entries_synthesizes_flat_lines_from_stave_line_ys():
    """A stave with no JSOMR behind it (tier 2/3, or a row recovered by
    assign_glyphs_to_staves) still contributes usable lines: one entry per
    line_ys value, flat, spanning the stave's own x range."""
    staves = [_stave("auto-0", 100, 160, ulx=50, lrx=450, line_ys=[100, 120, 140, 160])]
    entries = pitch_stage._staff_entries(staves, [])

    assert len(entries) == 4
    assert [e["within_stave_index"] for e in entries] == [0, 1, 2, 3]
    assert all(e["stave_id"] == 0 for e in entries)
    first = entries[0]["centerline_page"]
    assert (first["x_start"], first["x_end"]) == (50, 450)
    # Two samples answer every x for a horizontal line -- StaffLine.y_at_x
    # clamps its index into the list. See _staff_entries' docstring.
    assert first["y_values"] == [100.0, 100.0]
    # scale_unit on a synthesized line is the measured line GAP, which is what
    # Stave.continuous_step_at_y's single-line fallback reads it as.
    assert entries[0]["scale_unit"] == 20


def test_staff_entries_skips_staves_with_too_little_line_data():
    staves = [
        _stave("one-line", 100, 120, line_ys=[110]),
        _stave("no-lines", 200, 220, line_ys=[]),
        _stave("degenerate-x", 300, 360, ulx=90, lrx=90, line_ys=[300, 320]),
    ]
    assert pitch_stage._staff_entries(staves, []) == []


def test_staff_entries_prefers_jsomr_and_skips_the_staves_built_from_it():
    """staves_from_jsomr() flattens each line to one y; the records carry the
    whole fitted centerline. When both are available for the same stave, only
    the records go through -- otherwise that stave's lines would be counted
    twice."""
    records = [_jsomr_line(0, i, y) for i, y in enumerate([100, 120, 140, 160])]
    staves = [
        _stave("jsomr-stave-0", 100, 160, line_ys=[100, 120, 140, 160]),
        _stave("row-0", 300, 360, line_ys=[300, 320, 340, 360]),  # recovered row
    ]
    entries = pitch_stage._staff_entries(staves, records)

    assert len(entries) == 8
    assert [e["id"] for e in entries[:4]] == [r["id"] for r in records]
    # The recovered row is NOT jsomr-backed, so it still gets synthesized lines.
    assert all(e["id"].startswith("synth-s01-") for e in entries[4:])


def test_staff_entries_coerces_scaled_jsomr_x_bounds_back_to_int():
    """staffline_adapter.scale_jsomr_records() (the SF-7 predict-vs-encode
    resolution path) leaves x_start/x_end as floats, and
    staff_regroup.split_columns() indexes a coverage array with them."""
    record = _jsomr_line(0, 0, 100.5, x_start=10.4, x_end=399.6)
    [entry] = pitch_stage._staff_entries([], [record])

    assert entry["centerline_page"]["x_start"] == 10
    assert entry["centerline_page"]["x_end"] == 400
    assert isinstance(entry["centerline_page"]["x_start"], int)
    assert isinstance(entry["centerline_page"]["x_end"], int)


def test_staff_entries_ignores_records_without_a_centerline():
    """Interpolated-line records can carry bounding_box=None, and a malformed
    row could carry no centerline at all -- neither is a line to read a step
    from."""
    records = [{"id": "x", "stave_id": 0, "within_stave_index": 0},
               {"id": "y", "stave_id": 0, "within_stave_index": 1,
                "centerline_page": {"x_start": 0, "x_end": 10, "y_values": []}}]
    assert pitch_stage._staff_entries([], records) == []


# --- step -> MEI <clef @line> ---

def test_clef_line_from_step_maps_bottom_line_to_line_one():
    # staff_io: step 0 = bottom-most detected line, one line = two steps.
    assert pitch_stage._clef_line_from_step(0.0, 4) == (1, False)
    assert pitch_stage._clef_line_from_step(2.0, 4) == (2, False)
    assert pitch_stage._clef_line_from_step(6.0, 4) == (4, False)
    # Real measurements land near, not on, the integer -- see the clef steps
    # this reproduces on McGill_MS234-064 (5.78-6.02 all read as line 4).
    assert pitch_stage._clef_line_from_step(5.78, 4) == (4, False)
    assert pitch_stage._clef_line_from_step(-0.27, 4) == (1, False)


def test_clef_line_from_step_reports_clamping():
    """A clef above the top detected line (or below the bottom one) means the
    stave's real line count isn't what was detected -- clamp, but say so."""
    assert pitch_stage._clef_line_from_step(8.0, 4) == (4, True)
    assert pitch_stage._clef_line_from_step(-3.0, 4) == (1, True)


# --- _chain_from_pitch: absolute first pitch + @intm for the rest ---

def test_chain_from_pitch_ascends_on_positive_intm():
    components = [NcTemplate(), NcTemplate(intm=1), NcTemplate(intm=-1)]
    assert mei._chain_from_pitch(("c", "4"), components) == [
        ("c", "4"), ("d", "4"), ("c", "4")]


def test_chain_from_pitch_crosses_an_octave_boundary():
    components = [NcTemplate(), NcTemplate(intm=1)]
    assert mei._chain_from_pitch(("b", "3"), components) == [("b", "3"), ("c", "4")]
    components = [NcTemplate(), NcTemplate(intm=-1)]
    assert mei._chain_from_pitch(("c", "4"), components) == [("c", "4"), ("b", "3")]


def test_chain_from_pitch_survives_an_unparseable_pitch():
    components = [NcTemplate(), NcTemplate(intm=1)]
    assert mei._chain_from_pitch(("?", "x"), components) == [("?", "x"), ("?", "x")]


# --- build_mei's per-glyph pitch source ---

def _pitched_ncs(xml_bytes):
    root = ET.fromstring(xml_bytes)
    out = {}
    for neume in root.iter(f"{MEI_NS}neume"):
        gid = neume.get(mei.XML_ID).removeprefix("neume-")
        out[gid] = [(nc.get("pname"), nc.get("oct")) for nc in neume.findall(f"{MEI_NS}nc")]
    return out


def _build(pitch_map=None, clef_line_map=None, class_name="neume.punctum"):
    staves = [_stave("auto-0", 100, 160, line_ys=[100, 120, 140, 160])]
    glyphs_by_stave = {0: [_glyph("g1", 100, uly=130, class_name=class_name)]}
    return mei.build_mei(
        glyphs_by_stave, staves, image_path=Path("page.jpg"), image_w=500, image_h=400,
        manuscript_name="test", pitch_map=pitch_map, clef_line_map=clef_line_map,
    )


def test_build_mei_uses_the_supplied_pitch_verbatim():
    ncs = _pitched_ncs(_build(pitch_map={"g1": [("e", "4")]}))
    assert ncs["g1"] == [("e", "4")]


def test_build_mei_falls_back_to_the_placeholder_for_an_unmapped_glyph():
    """A glyph pitch finding couldn't resolve (missing_staff/missing_clef/...)
    still gets today's geometric pitch, not a hole in the MEI."""
    mapped = _pitched_ncs(_build(pitch_map={"g1": [("e", "4")]}))["g1"]
    unmapped = _pitched_ncs(_build(pitch_map={"other": [("e", "4")]}))["g1"]
    assert unmapped != mapped
    assert all(p and o for p, o in unmapped)


def test_build_mei_chains_when_the_note_count_disagrees():
    """neume.clivis2's CSV row is two <nc>s a step apart. Handing in ONE pitch
    (what the interval table yields for a class it only knows approximately,
    or a repeated-note neume) must keep that measured first pitch and place the
    second by @intm -- not discard the measurement for the whole glyph."""
    ncs = _pitched_ncs(_build(pitch_map={"g1": [("c", "4")]}, class_name="neume.clivis2"))
    assert len(ncs["g1"]) == 2
    assert ncs["g1"][0] == ("c", "4")
    # clivis descends: <nc intm="-1S"/>
    assert ncs["g1"][1] == ("b", "3")


def test_build_mei_takes_a_full_pitch_list_over_chaining():
    """When the counts DO agree, every component uses its own measured pitch --
    the decomposition, not a chain off note 1."""
    ncs = _pitched_ncs(_build(pitch_map={"g1": [("c", "4"), ("g", "3")]},
                              class_name="neume.clivis2"))
    assert ncs["g1"] == [("c", "4"), ("g", "3")]


def _clef_lines(xml_bytes):
    root = ET.fromstring(xml_bytes)
    return [c.get("line") for c in root.iter(f"{MEI_NS}clef")]


def test_clef_line_map_overrides_the_assumed_clef_line():
    """The measured line has to reach the emitted <clef>: Verovio positions an
    <nc> from pname/oct against the DECLARED line, so a clef read at line 1 and
    declared at line 3 renders the whole stave two lines off."""
    staves = [_stave("auto-0", 100, 160, line_ys=[100, 120, 140, 160])]
    glyphs_by_stave = {0: [
        _glyph("clef1", 10, uly=130, class_name="clef.c"),
        _glyph("g1", 100, uly=130),
    ]}
    kwargs = dict(image_path=Path("page.jpg"), image_w=500, image_h=400,
                  manuscript_name="test")

    assert _clef_lines(mei.build_mei(glyphs_by_stave, staves, **kwargs)) == ["3"]
    assert _clef_lines(mei.build_mei(glyphs_by_stave, staves,
                                     clef_line_map={"clef1": 1}, **kwargs)) == ["1"]


def test_clef_line_map_also_moves_the_placeholder_reference():
    """The fallback placeholder measures its step against the clef line too, so
    an overridden line must move both or the two disagree on the same stave."""
    staves = [_stave("auto-0", 100, 160, line_ys=[100, 120, 140, 160])]
    glyphs_by_stave = {0: [
        _glyph("clef1", 10, uly=130, class_name="clef.c"),
        _glyph("g1", 100, uly=130),
    ]}
    kwargs = dict(image_path=Path("page.jpg"), image_w=500, image_h=400,
                  manuscript_name="test")

    default = _pitched_ncs(mei.build_mei(glyphs_by_stave, staves, **kwargs))["g1"]
    moved = _pitched_ncs(mei.build_mei(glyphs_by_stave, staves,
                                       clef_line_map={"clef1": 1}, **kwargs))["g1"]
    # Declaring the clef two lines lower puts the same glyph four
    # diatonic steps above it instead of two below.
    assert default == [("a", "3")]
    assert moved == [("e", "4")]


# --- run_pitch_finding's own guard rails (no submodule needed) ---

def test_run_pitch_finding_off_switch(monkeypatch):
    monkeypatch.setattr(pitch_stage, "PITCH_FINDING_ENABLED", False)
    result = pitch_stage.run_pitch_finding([_glyph("g1", 100)], [], [], None)
    assert result.pitches_by_glyph == {}
    assert result.source is None
    assert any("disabled" in line for line in result.log_lines)


def test_run_pitch_finding_reports_missing_staff_geometry():
    """A page whose staves carry no line data at all can't be pitched -- and
    must say so rather than looking like a page with no notes."""
    result = pitch_stage.run_pitch_finding(
        [_glyph("g1", 100)], [_stave("auto-0", 100, 160, line_ys=[])], [], None)
    assert result.pitches_by_glyph == {}
    assert any("no staff-line geometry" in line for line in result.log_lines)


def test_run_pitch_finding_reports_an_unavailable_submodule(monkeypatch, tmp_path):
    """The failure mode on a checkout without `git submodule update --init`
    (or a venv without opencv): placeholder pitch plus a log line naming it,
    never an exception out of an encode job."""
    monkeypatch.setattr(pitch_stage, "PITCH_FINDING_DIR", tmp_path / "not-checked-out")
    monkeypatch.delitem(sys.modules, "pitch_finder", raising=False)
    monkeypatch.delitem(sys.modules, "staff_io", raising=False)
    monkeypatch.setattr(sys, "path", [p for p in sys.path
                                      if "pitch-finding" not in p])
    result = pitch_stage.run_pitch_finding(
        [_glyph("g1", 100)], [_stave("auto-0", 100, 160, line_ys=[100, 120, 140, 160])],
        [], None)
    assert result.pitches_by_glyph == {}
    assert any("pitch finding unavailable" in line for line in result.log_lines)


# --- end to end, against the real algorithm ---

def _submodule_available():
    pitch_stage._ensure_import_path()
    try:
        import pitch_finder  # noqa: F401
        import staff_io  # noqa: F401
    except Exception:
        return False
    return True


def _synthetic_page(tmp_path, glyphs):
    """A white page carrying one horizontal ink band per glyph bbox, centered
    in it. Deliberately no drawn staff lines: the stave geometry comes from
    line_ys, and painted lines would only pull each ink centroid around.

    A full-width band makes the expected centroid independent of which x
    sub-band the per-class crop rule picks -- every column has the same row
    profile -- so the expected pitches below are hand-derivable.
    """
    import numpy as np
    from PIL import Image

    page = np.full((400, 500), 255, dtype=np.uint8)
    for g in glyphs:
        page[g.uly + 5:g.uly + 15, g.ulx:g.ulx + g.ncols] = 0
    path = tmp_path / "page.png"
    Image.fromarray(page).save(path)
    return path.read_bytes()


@pytest.mark.skipif(not _submodule_available(),
                    reason="pitch-finding/ submodule or opencv not available")
def test_run_pitch_finding_end_to_end_on_a_synthetic_page(tmp_path):
    """Three noteheads, each two diatonic steps from the next, read against a
    C clef whose own ink sits on the middle one.

    Lines at y=100/120/140/160 with the bottom line as step 0, so one line is
    two steps and 10px is one step. Each glyph's ink band centroids to its own
    vertical center, which puts the clef and the middle punctum on the same
    step -> that punctum IS the clef pitch (c4), the one a line higher is two
    steps up (e4), the one a line lower two steps down (a3).
    That clef line, y=140, is the second one up from the bottom.
    """
    glyphs = [
        _glyph("clef1", 60, uly=130, class_name="clef.c"),
        _glyph("high", 200, uly=110),
        _glyph("mid", 260, uly=130),
        _glyph("low", 320, uly=150),
    ]
    staves = [_stave("auto-0", 100, 160, ulx=50, lrx=450, line_ys=[100, 120, 140, 160])]

    result = pitch_stage.run_pitch_finding(
        glyphs, staves, [], _synthetic_page(tmp_path, glyphs),
        notation_type="square", tmp_dir=tmp_path,
    )

    assert result.source == "pixel_centroid"
    assert result.pitches_by_glyph["mid"] == [("c", "4")]
    assert result.pitches_by_glyph["high"] == [("e", "4")]
    assert result.pitches_by_glyph["low"] == [("a", "3")]
    # MEI @line counts from the bottom of the staff, and the clef's own ink
    # sits on y=140 -- the second line up from the bottom one at y=160.
    assert result.clef_lines_by_glyph["clef1"] == 2
    assert result.resolved == 3


@pytest.mark.skipif(not _submodule_available(),
                    reason="pitch-finding/ submodule or opencv not available")
def test_pitch_finding_reaches_the_mei_end_to_end(tmp_path):
    """The same page through build_mei: the measured pitches, and the measured
    clef line, both land in the document."""
    glyphs = [
        _glyph("clef1", 60, uly=130, class_name="clef.c"),
        _glyph("high", 200, uly=110),
        _glyph("mid", 260, uly=130),
    ]
    staves = [_stave("auto-0", 100, 160, ulx=50, lrx=450, line_ys=[100, 120, 140, 160])]
    result = pitch_stage.run_pitch_finding(
        glyphs, staves, [], _synthetic_page(tmp_path, glyphs),
        notation_type="square", tmp_dir=tmp_path,
    )

    xml_bytes = mei.build_mei(
        {0: glyphs}, staves, image_path=Path("page.png"), image_w=500, image_h=400,
        manuscript_name="test",
        pitch_map=result.pitches_by_glyph, clef_line_map=result.clef_lines_by_glyph,
    )
    ncs = _pitched_ncs(xml_bytes)
    assert ncs["mid"] == [("c", "4")]
    assert ncs["high"] == [("e", "4")]
    assert _clef_lines(xml_bytes) == ["2"]
