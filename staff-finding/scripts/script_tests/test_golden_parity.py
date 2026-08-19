"""Golden-page parity test (ALPHA_TRANSITION_PLAN.md Phase 1, task 1.10 --
the phase-exit gate for the "Alpha 1 -- Staff-finding parity" fix batch,
issue #213).

Re-runs the real Stage 1/2 pipeline (component filter -> centerline fit ->
stave grouping) against checked-in golden-page fixtures via
parity_harness.py's own `run_variant()`/`compare()` -- reused directly, not
reimplemented, so this test and the harness's own `--sweep` report can never
silently drift apart on what "matches" means (MATCH_THRESHOLD_SCALE).

Two golden pages were specified for this gate: MS234_64 and Gent right.

**Gent right** has a real, checked-in raw source image
(`image-sets/gent/right/GentAnt1475_0017_AC_rightcrop.jpg`) and a checked-in
YOLO detection file (`e2e_tests/29may/Gent15_17_right/Gent15_17_right_corrected.txt`),
so this test re-runs the real pipeline against them and pins whatever the
pipeline PRODUCES TODAY as the regression baseline -- per an explicit
decision made when this test was added: Gent right is documented
(STAFFLINE_INTEGRATION_FOLLOWUPS.md's "Three checked-in e2e baselines show
genuinely fragmented grouping" section) as still exhibiting a real,
unresolved `group_staves.py` dense-stave-reconciliation limitation on this
page. Pinning today's numbers is NOT a claim that this grouping is correct
-- it is a claim that it must not get WORSE without someone deliberately
updating this test. A future fix to that reconciliation gap is expected to
require updating GENT_RIGHT_EXPECTED below; that's a feature of this test,
not a bug in it.

**MS234_64 could NOT be added as a live-reproducing test.** Its raw source
image (`/Volumes/Expansion/script_sorter_mss/McGill_MS234/McGill_MS234-064.jpg`
per ALPHA_TRANSITION_PLAN.md's own citation) lives only on the machine that
ran the original 2026-08-10 parity sweep -- it was never checked into this
repository, under any path, in any form (confirmed by a repo-wide search for
"MS234-064"/"ms234_64" image files; only diagnostic overlay PNGs and JSON
outputs exist). Without the source image, `run_variant()` cannot be
re-invoked against it here, so this test cannot catch a future regression on
that specific page. What IS checked in is `e2e_tests/10aug_parity/ms234_64/
baseline.json` -- a snapshot of what `run_variant()` produced on 2026-08-10.
`test_ms234_64_historical_baseline_is_unchanged` below only guards that
snapshot's own recorded numbers against silent corruption/regeneration (e.g.
someone "fixing" the file by hand) -- it is NOT a regression test against
current code, and must not be mistaken for one. Getting a real MS234_64
live-reproducing test needs the source image checked in (or Git-LFS'd)
first -- flagged as a follow-up, not fixed here (see this session's
running-notes; out of scope for the nine SF-* tickets this batch closes).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parity_harness import _load_original_rgb, run_variant, compare  # noqa: E402
from yolo_io import parse_yolo_txt, filter_to_class  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "e2e_tests"
IMAGE_SETS_DIR = Path(__file__).resolve().parent.parent.parent / "image-sets"

GENT_RIGHT_IMAGE = IMAGE_SETS_DIR / "gent" / "right" / "GentAnt1475_0017_AC_rightcrop.jpg"
GENT_RIGHT_YOLO = FIXTURES_DIR / "29may" / "Gent15_17_right" / "Gent15_17_right_corrected.txt"
MS234_64_BASELINE_JSON = FIXTURES_DIR / "10aug_parity" / "ms234_64" / "baseline.json"

# Pinned 2026-08-17 (post SF-1/4/5/6/7/8/9 fixes in this batch) against the
# checked-in Gent-right fixture pair above, via run_variant(pass_crop=True)
# -- staffline-class 0 (the bundled single-class model's real output class,
# per this batch's SF-9 fix). Update deliberately, not silently, if
# group_staves.py's dense-stave reconciliation gap is ever actually fixed.
GENT_RIGHT_EXPECTED = {
    "n_fits": 88,
    "stave_count": 20,
    "mode_lines_per_stave": 1,
    "cut_threshold_px": 15.1,
}
# MATCH_THRESHOLD_SCALE-relative tolerances (see parity_harness.py) -- loose
# enough to absorb float rounding across machines/library versions, tight
# enough to catch a real behavioral regression (e.g. SF-1's conf mismatch,
# which changed stave_count by -7 out of 8 on MS234_64).
STAVE_COUNT_TOLERANCE = 1
CUT_THRESHOLD_TOLERANCE_PX = 1.0


def _load_gent_right_detections():
    return filter_to_class(parse_yolo_txt(GENT_RIGHT_YOLO), 0)


def test_gent_right_parity_pinned_baseline():
    """Live re-run against the checked-in Gent-right fixture pair -- catches
    any future regression in the shared component_filter/fit_centerline/
    group_staves pipeline (e.g. a reintroduced SF-1 confidence mismatch, an
    SF-4 channel-order regression, an SF-5 missing-crop regression), without
    claiming this page's current grouping is already correct (it isn't --
    see module docstring)."""
    assert GENT_RIGHT_IMAGE.exists(), f"missing Gent-right fixture image: {GENT_RIGHT_IMAGE}"
    assert GENT_RIGHT_YOLO.exists(), f"missing Gent-right fixture YOLO txt: {GENT_RIGHT_YOLO}"

    image = _load_original_rgb(GENT_RIGHT_IMAGE)
    detections = _load_gent_right_detections()
    result = run_variant(image, detections, pass_crop=True)

    assert result["n_fits"] == GENT_RIGHT_EXPECTED["n_fits"], (
        f"box/fit survival count changed: {result['n_fits']} vs pinned "
        f"{GENT_RIGHT_EXPECTED['n_fits']} -- check for a component-filter or "
        f"crop-padding regression"
    )
    assert abs(result["stave_count"] - GENT_RIGHT_EXPECTED["stave_count"]) <= STAVE_COUNT_TOLERANCE, (
        f"stave_count drifted: {result['stave_count']} vs pinned "
        f"{GENT_RIGHT_EXPECTED['stave_count']} (tolerance {STAVE_COUNT_TOLERANCE})"
    )
    assert result["mode_lines_per_stave"] == GENT_RIGHT_EXPECTED["mode_lines_per_stave"], (
        f"mode_lines_per_stave changed: {result['mode_lines_per_stave']} vs pinned "
        f"{GENT_RIGHT_EXPECTED['mode_lines_per_stave']}"
    )
    assert abs(result["cut_threshold_px"] - GENT_RIGHT_EXPECTED["cut_threshold_px"]) <= CUT_THRESHOLD_TOLERANCE_PX, (
        f"cut_threshold_px drifted: {result['cut_threshold_px']} vs pinned "
        f"{GENT_RIGHT_EXPECTED['cut_threshold_px']} (tolerance {CUT_THRESHOLD_TOLERANCE_PX}px)"
    )


def test_gent_right_self_comparison_is_a_perfect_match():
    """Sanity check on compare() itself (reused, not reimplemented, from
    parity_harness.py): a page compared against its own freshly-computed
    result must report zero drift -- if this ever fails, compare()'s
    matching logic itself has a bug, independent of any pipeline change."""
    image = _load_original_rgb(GENT_RIGHT_IMAGE)
    detections = _load_gent_right_detections()
    result = run_variant(image, detections, pass_crop=True)

    cmp_ = compare(result, result)
    assert cmp_["stave_count_delta"] == 0
    assert cmp_["mode_delta"] == 0
    assert cmp_["unmatched_baseline"] == 0
    assert cmp_["unmatched_variant"] == 0
    assert cmp_["y_mae_px"] in (None, 0.0)


def test_ms234_64_historical_baseline_is_unchanged():
    """NOT a live regression test -- see module docstring. MS234_64's raw
    source image isn't in this repository, so this only guards the
    already-checked-in 2026-08-10 baseline.json snapshot against silent
    hand-editing/bad regeneration, pinning it to the exact numbers
    ALPHA_TRANSITION_PLAN.md's findings register cites (8 staves, mode 4,
    cut 62.0px)."""
    assert MS234_64_BASELINE_JSON.exists(), f"missing historical fixture: {MS234_64_BASELINE_JSON}"
    data = json.loads(MS234_64_BASELINE_JSON.read_text())

    assert data["image_size"] == [2154, 2750]
    assert data["stave_count"] == 8
    assert data["mode_lines_per_stave"] == 4
    assert data["line_count_distribution"] == {"4": 7, "5": 1}
    assert data["cut_threshold_px"] == 62.0
