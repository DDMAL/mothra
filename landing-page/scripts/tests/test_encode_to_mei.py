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


def _stave(sid, uly, lry, ulx=0, lrx=500, line_ys=None):
    return mei.StaveBbox(id=sid, ulx=ulx, uly=uly, lrx=lrx, lry=lry, line_ys=line_ys or [])


def _glyph(gid, ulx, uly=110, ncols=20, nrows=20, class_name="punctum"):
    return mei.Glyph(id=gid, ulx=ulx, uly=uly, ncols=ncols, nrows=nrows,
                      class_name=class_name, confidence=1.0, state="AUTOMATIC")


def _box(syl, ulx, lrx, uly, lry):
    return {"syl": syl, "ul": [ulx, uly], "lr": [lrx, lry]}


# --- _typical_line_spacing / _cluster_glyphs_into_staves's reliable-spacing path ---

def test_typical_line_spacing_ignores_staves_with_fewer_than_two_lines():
    staves = [
        _stave("real1", 100, 160, line_ys=[100, 120, 140, 160]),   # spacing 20
        _stave("real2", 300, 372, line_ys=[300, 324, 348, 372]),   # spacing 24
        _stave("crude", 500, 520, line_ys=[510]),                  # single point -- excluded
        _stave("empty", 600, 620, line_ys=[]),                     # no data -- excluded
    ]
    # median of the two real per-stave spacings (20 and 24) -> 22
    assert mei._typical_line_spacing(staves) == 22

def test_typical_line_spacing_none_when_nothing_qualifies():
    staves = [_stave("crude", 100, 120, line_ys=[110]), _stave("empty", 200, 220, line_ys=[])]
    assert mei._typical_line_spacing(staves) is None

def test_cluster_glyphs_into_staves_uses_reliable_spacing_over_bbox_guess():
    """Regression test for the reported bug: a row with one unusually high
    outlier note would badly inflate the OLD bbox-quartered line_ys guess
    (skewing every note's computed pitch, since Neon/Verovio renders from
    pitch, not raw pixel position -- see _step_from_y). Given the page's
    own reliable typical_line_spacing, the estimated lines must use THAT
    spacing, anchored on the row's median note height, not the row's own
    skewed bounding box."""
    # A smooth ascending run, each consecutive note within the row-gap
    # threshold (avg_h * 1.5 = 30px here) of its neighbor -- so it stays
    # ONE row-cluster -- but the row's total bbox span (cy 100 to 225) is
    # still much taller than a real 20px-spaced staff, exactly like a
    # melismatic run with genuine pitch range would produce.
    row_glyphs = [
        _glyph("g1", 10, uly=90),
        _glyph("g2", 40, uly=115),
        _glyph("g3", 70, uly=140),
        _glyph("g4", 100, uly=165),
        _glyph("g5", 130, uly=190),
        _glyph("g6", 160, uly=215),
    ]
    groups_with_hint = mei._cluster_glyphs_into_staves(
        row_glyphs, page_w=1000, page_h=1000, typical_line_spacing=20.0,
    )
    assert len(groups_with_hint) == 1
    stave, _ = groups_with_hint[0]
    spacings = [stave.line_ys[i + 1] - stave.line_ys[i] for i in range(len(stave.line_ys) - 1)]
    assert all(abs(s - 20.0) < 1e-9 for s in spacings)

    groups_without_hint = mei._cluster_glyphs_into_staves(row_glyphs, page_w=1000, page_h=1000)
    assert len(groups_without_hint) == 1
    stave_old, _ = groups_without_hint[0]
    old_spacings = [stave_old.line_ys[i + 1] - stave_old.line_ys[i] for i in range(len(stave_old.line_ys) - 1)]
    # Without the hint, spacing is still derived from the (outlier-inflated)
    # bbox span, so it's nowhere near the real 20px spacing -- confirms the
    # fix is actually doing something, not a no-op.
    assert all(s > 40.0 for s in old_spacings)


# --- _step_from_y ------------------------------------------------------------

def test_step_from_y_matches_normal_indexing_for_a_full_four_line_stave():
    """Sanity check: for the common case (exactly 4 real, evenly-spaced
    lines), the new bottom-anchored/extrapolated formula must still land
    on the same clef_y (and therefore the same step) the old direct-index
    formula did -- this is a robustness fix for abnormal line counts, not
    a change in behavior for the normal case."""
    line_ys = [100.0, 120.0, 140.0, 160.0]
    # Old formula: clef_idx = len(line_ys) - 3 = 1 -> clef_y = line_ys[1] = 120.
    assert mei._step_from_y(120.0, line_ys, clef_line=3) == 0
    assert mei._step_from_y(140.0, line_ys, clef_line=3) == 2  # 20px below clef, /10 half-spacing

def test_step_from_y_real_page_regression_stave_no_longer_shifted():
    """Reproduces the actual reported bug with real numbers from a live
    page. Before this fix, the affected stave's raw (undeduplicated) 7
    line_ys made clef_idx = 7 - 3 = 4 index into the wrong real line,
    landing a note-height's worth away from the real clef position, and
    the old mean-based spacing was itself dragged down by the near-zero
    duplicate gaps -- compounding the error. After staffline_adapter's
    dedup, this stave's 4 real line_ys must place a mid-staff point at
    essentially the same diatonic step as a NORMAL sibling stave's
    mid-staff point on the same page, instead of multiple steps higher."""
    normal_stave_line_ys = [487.6, 513.9, 543.5, 572.6]     # jsomr-stave-2
    fixed_stave_line_ys = [841.9, 870.6, 900.8, 931.6]      # jsomr-stave-4, post-dedup
    normal_center = (487 + 588) / 2
    fixed_center = (826 + 945) / 2
    normal_step = mei._step_from_y(normal_center, normal_stave_line_ys, clef_line=3)
    fixed_step = mei._step_from_y(fixed_center, fixed_stave_line_ys, clef_line=3)
    assert abs(normal_step - fixed_step) <= 1

def test_step_from_y_under_detected_stave_no_longer_wraps_negative_index():
    """A stave with fewer real lines detected than clef_line (e.g. only 2
    of a nominal 4) used to compute clef_idx = 2 - 3 = -1, silently
    wrapping to line_ys[-1] via Python's negative indexing -- an
    accident, not a deliberate choice, and wrong for clef_line=3 (which
    should sit 2 spacings above the bottom line, not AT it). The
    extrapolated formula must place clef_y strictly above the bottom
    line by 2 spacings instead of exactly on it."""
    line_ys = [302.6, 330.8]  # only 2 real lines detected, spacing ~28.2
    spacing = line_ys[1] - line_ys[0]
    expected_clef_y = line_ys[-1] - spacing * 2
    # A point exactly at the expected clef_y must be step 0.
    assert mei._step_from_y(expected_clef_y, line_ys, clef_line=3) == 0
    # It must NOT be step 0 at the bottom line itself (the old wraparound
    # bug's effective clef_y) -- confirms this isn't a no-op.
    assert mei._step_from_y(line_ys[-1], line_ys, clef_line=3) != 0

def test_step_from_y_none_for_too_few_or_degenerate_lines():
    assert mei._step_from_y(100.0, []) is None
    assert mei._step_from_y(100.0, [100.0]) is None
    assert mei._step_from_y(100.0, [100.0, 100.0]) is None  # zero spacing -- can't quantize


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
            gid = neume.get(mei.XML_ID).removeprefix("neume-")
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
            gid = neume.get(mei.XML_ID).removeprefix("neume-")
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


# --- end-to-end: recovered/missed stave gets reliable, not bbox-skewed, line_ys ---

def test_recovered_stave_uses_page_spacing_not_its_own_outlier_skewed_bbox():
    """Reproduces the actual reported bug: a real detected stave elsewhere
    on the page carries genuine line_ys; a second physical row's glyphs
    don't overlap ANY detected stave (the detector missed that system
    entirely) and include one outlier-high note, so
    assign_glyphs_to_staves has to synthesize a stave for it via
    _cluster_glyphs_into_staves. Before this fix, that synthesized stave's
    line_ys came purely from the row's own (outlier-inflated) bounding
    box -- wildly wrong spacing, which Neon/Verovio would render pitches
    against (see _step_from_y), floating notes far from their real staff.
    After this fix, the recovered stave's line_ys uses the page's own
    real spacing instead."""
    detected = _stave("real", 100, 160, line_ys=[100, 120, 140, 160])  # real, 20px spacing
    # A smooth ascending run (see the _cluster_glyphs_into_staves test
    # above for why it's shaped this way): stays one row-cluster, but its
    # own bbox span is much taller than the real 20px staff spacing, and
    # it's far below (500+) the detected stave so it doesn't overlap --
    # exactly the "detector missed this whole system" scenario.
    missed_row_glyphs = [
        _glyph("m1", 10, uly=490),
        _glyph("m2", 40, uly=515),
        _glyph("m3", 70, uly=540),
        _glyph("m4", 100, uly=565),
        _glyph("m5", 130, uly=590),
        _glyph("m6", 160, uly=615),
    ]

    _glyphs_by_stave, staves = mei.assign_glyphs_to_staves(
        missed_row_glyphs, [detected], page_w=1000, page_h=1000,
    )

    assert len(staves) == 2  # the real one, plus one recovered for the missed row
    recovered = staves[1]
    spacings = [recovered.line_ys[i + 1] - recovered.line_ys[i] for i in range(len(recovered.line_ys) - 1)]
    # Must match the real stave's own 20px spacing, not an outlier-skewed guess.
    assert all(abs(s - 20.0) < 1e-9 for s in spacings)


# --- _stave_zone_bounds --------------------------------------------------

def test_stave_zone_bounds_height_matches_measured_line_spacing():
    stave = _stave("s", uly=100, lry=200, line_ys=[100.0, 120.0, 140.0, 160.0])
    uly, lry = mei._stave_zone_bounds(stave)
    spacing = mei._line_spacing(stave.line_ys)
    assert lry == round(stave.line_ys[-1])
    assert lry - uly == round(spacing * (mei.STAFF_LINES - 1))


def test_stave_zone_bounds_falls_back_to_raw_bbox_with_fewer_than_two_lines():
    # A single sampled line -- not enough points for _line_spacing -- must
    # fall back to the stave's own raw bbox rather than divide-by-nothing.
    one_line = _stave("s1", uly=100, lry=200, line_ys=[130.0])
    assert mei._stave_zone_bounds(one_line) == (100, 200)
    # No line_ys at all (e.g. a stave detected purely from its bbox).
    no_lines = _stave("s2", uly=50, lry=90)
    assert mei._stave_zone_bounds(no_lines) == (50, 90)


# --- verify_and_correct_syllables -----------------------------------------

def test_verify_and_correct_syllables_is_a_noop_when_already_matching():
    """Re-verifying an MEI against the exact text_alignment it was built
    from must change nothing -- the base case every other
    verify_and_correct_syllables test builds on."""
    staves = [_stave("main", 100, 160, lrx=1000, line_ys=[100.0, 120.0, 140.0, 160.0])]
    glyphs_by_stave = {0: [_glyph("g1", 10), _glyph("g2", 40), _glyph("g3", 70)]}
    text_alignment = {"syl_boxes": [_box("Alleluia", 5, 95, 128, 148)]}

    xml_bytes = mei.build_mei(
        glyphs_by_stave, staves,
        image_path=Path("page.jpg"), image_w=2000, image_h=2000,
        manuscript_name="test", text_alignment=text_alignment,
        n_detected_staves=1,
    )
    corrected_bytes, logs = mei.verify_and_correct_syllables(xml_bytes, text_alignment, 2000, 2000)

    assert logs == []
    assert corrected_bytes == xml_bytes


def test_verify_and_correct_syllables_never_erases_real_text_with_dash():
    """Regression test for the bug this fix closes: verify_and_correct_syllables
    deliberately does NOT redo _group_staves_by_row's cross-stave pooling
    (see its own docstring's scope boundary), so re-running
    _assign_boxes_to_staves/_build_syllable_units against the SAME
    text_alignment a fragmented row was originally built from recomputes
    the fragment stave's own syl_boxes as empty -- exactly the precondition
    the pooling fix exists to route around. Before this fix, that empty
    box list fell straight to the "-" fallback and clobbered the fragment's
    real, already-correct syllable text. The fix must recognize that the
    fragment's glyph grouping hasn't actually changed and keep its
    existing real text instead."""
    staves = [
        _stave("main", 100, 160, lrx=1000, line_ys=[100.0, 120.0, 140.0, 160.0]),
        _stave("fragment", 95, 115, lrx=1000, line_ys=[95.0, 115.0]),
    ]
    glyphs_by_stave = {
        0: [_glyph("m1", 10), _glyph("m2", 40), _glyph("m3", 70)],
        1: [_glyph("f1", 100), _glyph("f2", 130), _glyph("f3", 160)],
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
        n_detected_staves=1,  # "fragment" (index 1) must be >= n_detected_staves
                               # to be eligible for _group_staves_by_row's merge
    )
    # Sanity check on the precondition: build_mei's row-pooling really did
    # give the fragment stave the real second word, not "-".
    root_before = ET.fromstring(xml_bytes)
    syllables_before = root_before.findall(f".//{MEI_NS}syllable")
    fragment_text_before = next(
        s.find(f"{MEI_NS}syl").text for s in syllables_before
        if any(n.get(mei.XML_ID) == "neume-f1"
               for n in s.findall(f"{MEI_NS}neume"))
    )
    assert fragment_text_before == "-le-lu-ia"

    corrected_bytes, logs = mei.verify_and_correct_syllables(xml_bytes, text_alignment, 2000, 2000)

    # Nothing about mothra-text's data changed -- this must be a no-op,
    # not a "correction" that erases the fragment's real text to "-".
    assert logs == []
    assert corrected_bytes == xml_bytes


def test_verify_and_correct_syllables_applies_a_genuine_text_disagreement():
    """When mothra-text's data genuinely changed (a different word for the
    same glyphs), that's a real correction and must be applied -- the
    text-preservation guard above must not make this function inert."""
    staves = [_stave("main", 100, 160, lrx=1000, line_ys=[100.0, 120.0, 140.0, 160.0])]
    glyphs_by_stave = {0: [_glyph("g1", 10), _glyph("g2", 40), _glyph("g3", 70)]}
    original_alignment = {"syl_boxes": [_box("Alleluia", 5, 95, 128, 148)]}

    xml_bytes = mei.build_mei(
        glyphs_by_stave, staves,
        image_path=Path("page.jpg"), image_w=2000, image_h=2000,
        manuscript_name="test", text_alignment=original_alignment,
        n_detected_staves=1,
    )

    corrected_alignment = {"syl_boxes": [_box("Kyrie", 5, 95, 128, 148)]}
    corrected_bytes, logs = mei.verify_and_correct_syllables(xml_bytes, corrected_alignment, 2000, 2000)

    assert logs != []
    root = ET.fromstring(corrected_bytes)
    syl_texts = [s.find(f"{MEI_NS}syl").text for s in root.findall(f".//{MEI_NS}syllable")]
    assert syl_texts == ["Kyrie"]

def test_verify_and_correct_syllables_keeps_a_distinct_repeated_occurrence():
    """Regression test for #209: a repeated word ("Alleluia" said twice on
    the page, common in chant) must not be conflated with the
    fragment-pooling duplicate case above just because the text matches.
    Stave 2's own "Alleluia" here is a real, legitimate "text with nothing
    under it" occurrence -- distinct from, and spatially far from, stave
    1's real Alleluia-with-glyphs. Before this fix, the old text-only
    guard treated stave 2's box as a duplicate of stave 1's (same text,
    anywhere in the document) and silently deleted it on every re-verify --
    this is the exact "syllables mysteriously missing" bug."""
    staves = [
        _stave("s1", 100, 160, lrx=1000, line_ys=[100.0, 120.0, 140.0, 160.0]),
        _stave("s2", 300, 360, lrx=1000, line_ys=[300.0, 320.0, 340.0, 360.0]),
    ]
    glyphs_by_stave = {
        0: [_glyph("g1", 10), _glyph("g2", 40), _glyph("g3", 70)],
        1: [],  # s2 has no notation at all under its own "Alleluia"
    }
    text_alignment = {
        "syl_boxes": [
            _box("Alleluia", 5, 95, 128, 148),   # stave 1's real occurrence, has glyphs
            _box("Alleluia", 5, 95, 328, 348),   # stave 2's real, distinct, glyph-less occurrence
        ]
    }

    xml_bytes = mei.build_mei(
        glyphs_by_stave, staves,
        image_path=Path("page.jpg"), image_w=2000, image_h=2000,
        manuscript_name="test", text_alignment=text_alignment,
        n_detected_staves=2,  # both real/detected -- never eligible for row-pooling merge
    )
    corrected_bytes, logs = mei.verify_and_correct_syllables(xml_bytes, text_alignment, 2000, 2000)

    # Re-verifying against the SAME alignment it was built from must be a
    # no-op -- stave 2's genuinely distinct "Alleluia" must survive.
    assert logs == []
    assert corrected_bytes == xml_bytes
    root = ET.fromstring(corrected_bytes)
    syl_texts = [s.find(f"{MEI_NS}syl").text for s in root.findall(f".//{MEI_NS}syllable")]
    assert syl_texts == ["Alleluia", "Alleluia"]

def test_verify_and_correct_syllables_keeps_syllable_on_its_own_stave_despite_zone_distortion():
    """Regression test for #208: _stave_zone_bounds's Verovio-spacing-
    matched written zone can diverge sharply from the RAW stave bbox
    build_mei actually ran _assign_boxes_to_staves against, whenever a
    stave has fewer real detected lines than the nominal STAFF_LINES=4 (a
    real, observed condition -- see _step_from_y's docstring, 2-7 lines
    detected per stave on a real page). "s2" here has only 2 real lines
    with a wide 34px gap, so its written zone extrapolates up to y=198 --
    far above its own raw bbox's uly=245 -- reaching up into the gap where
    "s1"'s own text sits. Before this fix, _reconstruct_mei_state had no
    way to recover s2's raw bbox, so verify_and_correct_syllables's re-run
    of _assign_boxes_to_staves used the distorted written-zone band
    instead, reassigning s1's "Alleluia" to s2's <sb> section and matching
    it against s2's own (unrelated) neumes: "syllables incorrectly
    aligning with the neumes in the stave below it." s1 was then left with
    no syl_boxes of its own, falling to the generic "-" gap-cluster
    fallback across its entire line -- the linked "blank syllables that
    take up neumes on an entire line" symptom."""
    staves = [
        _stave("s1", 100, 160, lrx=1000, line_ys=[100.0, 120.0, 140.0, 160.0]),
        _stave("s2", 245, 300, lrx=1000, line_ys=[266.0, 300.0]),
    ]
    glyphs_by_stave = {
        0: [_glyph("g1", 10), _glyph("g2", 40), _glyph("g3", 70)],
        1: [_glyph("g4", 10, uly=260), _glyph("g5", 40, uly=260), _glyph("g6", 70, uly=260)],
    }
    text_alignment = {
        "syl_boxes": [
            _box("Alleluia", 5, 95, 190, 210),   # genuinely s1's own text
            _box("Kyrie", 5, 95, 305, 325),        # genuinely s2's own text
        ]
    }

    xml_bytes = mei.build_mei(
        glyphs_by_stave, staves,
        image_path=Path("page.jpg"), image_w=2000, image_h=2000,
        manuscript_name="test", text_alignment=text_alignment,
        n_detected_staves=2,
    )
    root_before = ET.fromstring(xml_bytes)
    syllables_before = root_before.findall(f".//{MEI_NS}syllable")
    assert [s.find(f"{MEI_NS}syl").text for s in syllables_before] == ["Alleluia", "Kyrie"]

    corrected_bytes, logs = mei.verify_and_correct_syllables(xml_bytes, text_alignment, 2000, 2000)

    # Must be a no-op: re-verifying against the SAME alignment must not
    # migrate "Alleluia" onto s2's section just because s2's written zone
    # (correctly, for Verovio's sake) doesn't match its raw detection bbox.
    assert logs == []
    assert corrected_bytes == xml_bytes


# --- special-element (clef/custos/divLine) encoding -- mothra#257 ---

def test_custos_and_divline_interleaved_in_reading_order_between_syllables():
    """build_mei must emit <custos>/<divLine> as real elements in their own
    X-position order relative to <syllable>s -- not wrapped inside one (the
    old punctum-fallback bug), and not all shoved before or after every
    syllable."""
    stave = _stave("s1", 100, 160, lrx=1000, line_ys=[100.0, 120.0, 140.0, 160.0])
    glyphs_by_stave = {
        0: [
            _glyph("g1", 10, class_name="neume.punctum"),
            _glyph("gdiv", 40, class_name="divisio.maior"),
            _glyph("g2", 70, class_name="neume.punctum"),
            _glyph("gcustos", 100, class_name="custos"),
        ]
    }
    xml_bytes = mei.build_mei(
        glyphs_by_stave, [stave],
        image_path=Path("page.jpg"), image_w=1000, image_h=1000,
        manuscript_name="test", notation_type="hufnagel",
    )
    root = ET.fromstring(xml_bytes)
    layer = root.find(f".//{MEI_NS}layer")
    tags = [child.tag.rsplit("}", 1)[-1] for child in layer]
    # pb, sb, opening clef, syllable(g1), divLine, syllable(g2), custos
    assert tags == ["pb", "sb", "clef", "syllable", "divLine", "syllable", "custos"]

    div_el = layer.find(f"{MEI_NS}divLine")
    assert div_el.get("form") == "maior"
    custos_el = layer.find(f"{MEI_NS}custos")
    assert custos_el.get("pname") and custos_el.get("oct")
    # Neither special element was wrapped in a <syllable>/<neume>.
    assert layer.find(f".//{MEI_NS}syllable/{MEI_NS}neume/{MEI_NS}divLine") is None
    assert not any(n.get("facs", "").endswith("gcustos") for n in layer.iter(f"{MEI_NS}nc"))


def test_divisio_maior_excluded_from_neume_glyph_filter():
    """Regression test for the substring-matching bug this fix replaced:
    _SKIP_CLASS_FRAGMENTS = {"custos", "divline", "division"} matched
    "custos" but never matched "divisio.maior"/"divisio.maxima" (no "n"
    before the "." in "divisio"), so a divisio glyph fell through into
    _filter_neume_glyphs's kept list instead of being excluded. Checking
    special_mapping membership instead must exclude it regardless of the
    class name's literal substrings."""
    special_mapping = mei.resolve_special_mapping("hufnagel")
    glyphs = [_glyph("gdiv", 40, class_name="divisio.maior")]
    kept = mei._filter_neume_glyphs(glyphs, stave_idx=0, special_mapping=special_mapping)
    assert kept == []


def test_verify_and_correct_syllables_preserves_special_element_position():
    """A divLine sitting between two syllables must stay between them (by
    X position) after verify_and_correct_syllables rebuilds the stave's
    syllables to match new text-finding output -- not get shoved to the
    end. Exercises the remove-and-reinsert path where element identity and
    ordering both matter."""
    stave = _stave("s1", 100, 160, lrx=1000, line_ys=[100.0, 120.0, 140.0, 160.0])
    glyphs_by_stave = {
        0: [
            _glyph("g1", 10, class_name="neume.punctum"),
            _glyph("gdiv", 55, class_name="divisio.maior"),
            _glyph("g2", 100, class_name="neume.punctum"),
        ]
    }
    text_alignment = {
        "syl_boxes": [
            _box("old1", 5, 45, 90, 110),
            _box("old2", 95, 135, 90, 110),
        ]
    }
    xml_bytes = mei.build_mei(
        glyphs_by_stave, [stave],
        image_path=Path("page.jpg"), image_w=1000, image_h=1000,
        manuscript_name="test", notation_type="hufnagel", text_alignment=text_alignment,
    )

    new_alignment = {
        "syl_boxes": [
            _box("new1", 5, 45, 90, 110),
            _box("new2", 95, 135, 90, 110),
        ]
    }
    corrected_bytes, logs = mei.verify_and_correct_syllables(xml_bytes, new_alignment, 1000, 1000)
    assert logs != []

    root = ET.fromstring(corrected_bytes)
    layer = root.find(f".//{MEI_NS}layer")
    tags = [child.tag.rsplit("}", 1)[-1] for child in layer]
    assert tags == ["pb", "sb", "clef", "syllable", "divLine", "syllable"]
    # Exactly one divLine -- no duplication from the remove/reinsert pass.
    assert len(layer.findall(f"{MEI_NS}divLine")) == 1
    syl_texts = [s.find(f"{MEI_NS}syl").text for s in layer.findall(f"{MEI_NS}syllable")]
    assert syl_texts == ["new1", "new2"]