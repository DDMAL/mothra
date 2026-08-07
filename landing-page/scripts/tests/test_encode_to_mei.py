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
    """s0 is the only originally-detected stave (n_detected_staves=1); s1-s3
    are all synthetic (assign_glyphs_to_staves-recovered) indices. A chain
    of synthetic-involving adjacent pairs (s0-s1, s1-s2) merges
    transitively even though s0-s2 aren't directly adjacent; s3 has no
    adjacent neighbor at all and stays isolated."""
    staves = [
        _stave("s0", 100, 160),   # the only originally-detected stave
        _stave("s1", 155, 170),   # synthetic; adjacent to s0 and s2 -> chains them together
        _stave("s2", 168, 200),   # synthetic; only adjacent to s1 directly
        _stave("s3", 500, 560),   # synthetic but genuinely separate -- far below, own row
    ]
    groups = mei._group_staves_by_row(staves, n_detected_staves=1)
    assert {frozenset(g) for g in groups} == {frozenset({0, 1, 2}), frozenset({3})}


def test_group_staves_by_row_no_staves_close_together_stays_singletons():
    staves = [_stave("s0", 100, 160), _stave("s1", 400, 460)]
    groups = mei._group_staves_by_row(staves, n_detected_staves=1)
    assert {frozenset(g) for g in groups} == {frozenset({0}), frozenset({1})}


def test_group_staves_by_row_never_merges_two_originally_detected_staves():
    """Regression guard for the actual reported bug: two originally
    DETECTED, independent staves (both indices < n_detected_staves) must
    NEVER merge, no matter how close together or similar in size. A prior
    version of this function tried to infer "fragment-ness" purely from
    Y-adjacency plus bbox-height lopsidedness; that was empirically
    insufficient -- on a real page, two genuinely distinct, independently
    detected rows were both Y-adjacent AND similar/lopsided enough in
    height to satisfy that old check, and merging them pooled their real
    syl_boxes together and reassigned them by raw X-position with no Y
    discrimination -- the resulting syllable sequence alternated between
    the two unrelated rows one-for-one (e.g. 'mi cum nus dum sal pro...'
    was really 'mi'[row A] 'cum'[row B] 'nus'[row A] 'dum'[row B] ...
    zippered together by the shared X-sort). Gating strictly on
    assign_glyphs_to_staves's own synthetic-stave signal instead --
    neither index here is >= n_detected_staves -- prevents this
    unconditionally, regardless of proximity or height."""
    staves = [
        _stave("row1", 100, 160),   # originally detected, full-height
        _stave("row2", 165, 225),   # also originally detected, only 5px away
    ]
    groups = mei._group_staves_by_row(staves, n_detected_staves=2)
    assert {frozenset(g) for g in groups} == {frozenset({0}), frozenset({1})}


# --- _assign_boxes_to_staves ------------------------------------------------

def test_assign_boxes_to_staves_prefers_stave_above_over_raw_nearest():
    """A syl_box sitting between two staves must go to the stave ABOVE it
    (whose own text it visually is) even when it happens to sit
    geometrically closer to the stave below -- e.g. because the scribe left
    more space above the text than below it. Regression coverage for
    syllables getting attached to the neumes in the wrong (next) staff."""
    staves = [_stave("above", 100, 160), _stave("below", 250, 310)]
    # Box center at y=220: 60px below 'above' (lry=160), only 30px above
    # 'below' (uly=250) -- deliberately CLOSER to 'below' in raw distance
    # (the exact "scribe left more space above the text than below it"
    # case), to prove this isn't just accidentally correct by proximity.
    boxes = [_box("word", 10, 90, 210, 230)]
    result = mei._assign_boxes_to_staves(staves, boxes)
    assert result[0] == boxes
    assert result[1] == []


def test_assign_boxes_to_staves_box_above_everything_falls_back_to_nearest():
    """A box sitting above every stave (e.g. a rubric before the first
    system) has no stave above it to prefer -- falls back to nearest by
    raw distance, same as the pre-fix behavior for this edge case."""
    staves = [_stave("only", 300, 360)]
    boxes = [{"syl": "Incipit", "ul": [10, 10], "lr": [90, 30]}]
    result = mei._assign_boxes_to_staves(staves, boxes)
    assert result[0] == boxes


# --- end-to-end: fragmented row shares its row-mate's real syllables ------

def test_fragmented_row_shares_syllables_with_its_row_mate():
    """The actual bug scenario: a 'main' stave and a 'fragment' stave with
    Y-adjacent ranges (simulating assign_glyphs_to_staves splitting one
    physical row across two stave_idx values), plus a genuinely separate
    'control' stave. Both real syl_boxes sit at the main/fragment row's
    shared Y-band, so _assign_boxes_to_staves (unchanged) buckets both onto
    the main stave only -- reproducing the bug precondition that a naive
    per-stave_idx lookup would leave the fragment's boxes_by_stave entry
    empty.

    The fragment is placed LAST (index 2) and n_detected_staves=2 passed,
    matching real usage: assign_glyphs_to_staves always appends any
    synthesized/recovered stave after every originally-detected one, and
    only a stave at or beyond that boundary is ever eligible to merge (see
    _group_staves_by_row)."""
    staves = [
        _stave("main", 100, 160, lrx=1000),
        _stave("control", 500, 560, lrx=1000),
        _stave("fragment", 95, 115, lrx=1000),
    ]
    glyphs_by_stave = {
        0: [_glyph("m1", 10), _glyph("m2", 40), _glyph("m3", 70)],
        1: [_glyph("c1", 10, uly=520), _glyph("c2", 40, uly=520)],
        2: [_glyph("f1", 100), _glyph("f2", 130), _glyph("f3", 160)],
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
        n_detected_staves=2,
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


def test_two_detected_adjacent_rows_never_cross_assign_syllables():
    """End-to-end regression test for the actual reported bug: two
    originally-detected, Y-adjacent, height-LOPSIDED rows (row B is much
    shorter than row A -- exactly the profile that would have satisfied
    the old, geometry-only merge heuristic) each get their own real
    syl_boxes at the SAME x-range on purpose, so that if they were ever
    incorrectly pooled together (as the old heuristic would have done
    here), _assign_glyphs_to_boxes's x-tie-breaking would deterministically
    misroute row B's glyph onto row A's word. With n_detected_staves=2
    (both rows are originally detected, neither synthetic), they must
    never merge, so each row's glyph independently and correctly finds
    only its own row's word."""
    staves = [
        _stave("rowA", 100, 160, lrx=1000),   # tall, originally detected
        _stave("rowB", 165, 190, lrx=1000),   # short, originally detected, 5px away
    ]
    glyphs_by_stave = {
        0: [_glyph("a1", 20, uly=130)],
        1: [_glyph("b1", 25, uly=175)],
    }
    text_alignment = {
        "syl_boxes": [
            _box("WordA", 10, 90, 128, 148),   # sits in rowA's own band
            _box("WordB", 10, 90, 170, 185),   # sits in rowB's own band, SAME x-range
        ]
    }

    xml_bytes = mei.build_mei(
        glyphs_by_stave, staves,
        image_path=Path("page.jpg"), image_w=2000, image_h=2000,
        manuscript_name="test", text_alignment=text_alignment,
        n_detected_staves=2,
    )
    root = ET.fromstring(xml_bytes)
    syllables = root.findall(f".//{MEI_NS}syllable")
    glyph_to_syl_text = {}
    for syllable in syllables:
        syl_el = syllable.find(f"{MEI_NS}syl")
        syl_text = syl_el.text if syl_el is not None else None
        for neume in syllable.findall(f"{MEI_NS}neume"):
            gid = neume.get(f"{{http://www.w3.org/XML/1998/namespace}}id").removeprefix("neume-")
            glyph_to_syl_text[gid] = syl_text

    assert glyph_to_syl_text["a1"] == "WordA"
    assert glyph_to_syl_text["b1"] == "WordB"


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
