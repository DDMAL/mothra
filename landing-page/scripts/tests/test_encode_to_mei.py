"""Unit/end-to-end tests for encode_to_mei.py's row-fragmentation-aware
syllable matching (_group_staves_by_row / _pool_group_syllable_data /
_stave_share_of_group).

Regression coverage for: assign_glyphs_to_staves's Y-gap glyph clustering
can split one physical manuscript row's glyphs across two (or more)
stave_idx buckets. Before this fix, _assign_boxes_to_staves's pure
Y-distance bucketing sent that row's real syl_boxes to whichever stave_idx
got most of the row's glyphs (the "main" bucket), leaving a "fragment"
bucket's boxes_by_stave entry empty -- so build_mei's per-stave loop fell
straight to the "-" no-text-alignment fallback for the fragment's glyphs,
even though its own row-mate had the real word right there. See the plan
doc for the full root-cause trace.

Follows test_staffline_adapter.py's convention: sys.path insert, bare
module import, plain pytest functions, no fixtures/conftest.
"""
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import encode_to_mei as mei  # noqa: E402

MEI_NS = "{http://www.music-encoding.org/ns/mei}"


def _stave(sid, uly, lry, ulx=0, lrx=500):
    return mei.StaveBbox(id=sid, ulx=ulx, uly=uly, lrx=lrx, lry=lry)


def _glyph(gid, ulx, uly=110, ncols=20, nrows=20, class_name="punctum"):
    return mei.Glyph(id=gid, ulx=ulx, uly=uly, ncols=ncols, nrows=nrows,
                      class_name=class_name, confidence=1.0, state="AUTOMATIC")


def _box(syl, ulx, lrx, uly, lry):
    return {"syl": syl, "ul": [ulx, uly], "lr": [lrx, lry]}


# --- _group_staves_by_row --------------------------------------------------

def test_group_staves_by_row_transitive_chain_and_isolated_control():
    staves = [
        _stave("s0", 100, 160),   # main
        _stave("s1", 155, 170),   # adjacent to s0 and s2 -> chains them together
        _stave("s2", 168, 200),   # only adjacent to s1 directly
        _stave("s3", 500, 560),   # genuinely separate -- far below, own row
    ]
    groups = mei._group_staves_by_row(staves)
    assert {frozenset(g) for g in groups} == {frozenset({0, 1, 2}), frozenset({3})}


def test_group_staves_by_row_no_staves_close_together_stays_singletons():
    staves = [_stave("s0", 100, 160), _stave("s1", 400, 460)]
    groups = mei._group_staves_by_row(staves)
    assert {frozenset(g) for g in groups} == {frozenset({0}), frozenset({1})}


# --- end-to-end: fragmented row shares its row-mate's real syllables ------

def test_fragmented_row_shares_syllables_with_its_row_mate():
    """The actual bug scenario: a 'main' stave and a 'fragment' stave with
    Y-adjacent ranges (simulating assign_glyphs_to_staves splitting one
    physical row across two stave_idx values), plus a genuinely separate
    'control' stave. Both real syl_boxes sit at the main/fragment row's
    shared Y-band, so _assign_boxes_to_staves (unchanged) buckets both onto
    the main stave only -- reproducing the bug precondition that a naive
    per-stave_idx lookup would leave the fragment's boxes_by_stave entry
    empty."""
    staves = [
        _stave("main", 100, 160, lrx=1000),
        _stave("fragment", 95, 115, lrx=1000),
        _stave("control", 500, 560, lrx=1000),
    ]
    glyphs_by_stave = {
        0: [_glyph("m1", 10), _glyph("m2", 40), _glyph("m3", 70)],
        1: [_glyph("f1", 100), _glyph("f2", 130), _glyph("f3", 160)],
        2: [_glyph("c1", 10, uly=520), _glyph("c2", 40, uly=520)],
    }
    text_alignment = {
        "syl_boxes": [
            _box("Al-", 5, 95, 128, 148),
            _box("-le-lu-ia", 98, 185, 128, 148),
        ]
    }

    xml_bytes = mei.build_mei(
        glyphs_by_stave, staves,
        image_path=Path("page.jpg"), image_w=2000, image_h=2000,
        manuscript_name="test", text_alignment=text_alignment,
    )
    root = ET.fromstring(xml_bytes)

    # ElementTree has no getparent(), so walk <syllable>s directly and
    # record which glyph ids each one's <neume> children cover.
    syllables = root.findall(f".//{MEI_NS}syllable")
    glyph_to_syl_text = {}
    for syllable in syllables:
        syl_el = syllable.find(f"{MEI_NS}syl")
        syl_text = syl_el.text if syl_el is not None else None
        for neume in syllable.findall(f"{MEI_NS}neume"):
            gid = neume.get(f"{{http://www.w3.org/XML/1998/namespace}}id").removeprefix("neume-")
            glyph_to_syl_text[gid] = syl_text

    # Main's glyphs (leftward, x 10-90) get the first real word.
    assert glyph_to_syl_text["m1"] == "Al-"
    assert glyph_to_syl_text["m2"] == "Al-"
    assert glyph_to_syl_text["m3"] == "Al-"
    # Fragment's glyphs (rightward, x 100-180) get the second real word --
    # THIS is the assertion that fails without the fix (would be "-").
    assert glyph_to_syl_text["f1"] == "-le-lu-ia"
    assert glyph_to_syl_text["f2"] == "-le-lu-ia"
    assert glyph_to_syl_text["f3"] == "-le-lu-ia"
    # Control stave is genuinely isolated (no adjacent sibling, no boxes of
    # its own) -- must be unaffected, still falls to the "-" fallback.
    assert glyph_to_syl_text["c1"] == "-"
    assert glyph_to_syl_text["c2"] == "-"

    # Exactly one <syllable> carries each real word -- no duplicate empty
    # copy of a box that does have glyphs, on the box's Y-native stave.
    al_syllables = [s for s in syllables if (s.find(f"{MEI_NS}syl") is not None
                                              and s.find(f"{MEI_NS}syl").text == "Al-")]
    lulia_syllables = [s for s in syllables if (s.find(f"{MEI_NS}syl") is not None
                                                 and s.find(f"{MEI_NS}syl").text == "-le-lu-ia")]
    assert len(al_syllables) == 1
    assert len(lulia_syllables) == 1
    assert len(al_syllables[0].findall(f"{MEI_NS}neume")) == 3
    assert len(lulia_syllables[0].findall(f"{MEI_NS}neume")) == 3


def test_isolated_stave_with_no_boxes_still_falls_back_to_dash():
    """A stave with no adjacent sibling and no syl_boxes of its own must
    behave exactly as it did before this fix -- '-' fallback, unaffected by
    the row-grouping machinery entirely."""
    staves = [_stave("only", 100, 160, lrx=1000)]
    glyphs_by_stave = {0: [_glyph("g1", 10), _glyph("g2", 40)]}

    xml_bytes = mei.build_mei(
        glyphs_by_stave, staves,
        image_path=Path("page.jpg"), image_w=2000, image_h=2000,
        manuscript_name="test", text_alignment=None,
    )
    root = ET.fromstring(xml_bytes)
    syl_texts = [
        el.text for el in root.findall(f".//{MEI_NS}syllable/{MEI_NS}syl")
    ]
    assert syl_texts and all(t == "-" for t in syl_texts)
