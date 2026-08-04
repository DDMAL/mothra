"""
Synthetic test for the merge step.

Constructs a fragmented line (multiple components, all at the same y, with small
x-gaps between them) and confirms:

1. Without merging: only one fragment is kept (the longest/most-central one),
   others are discarded as not_top_scoring.
2. With merging: all fragments cluster together, the merged cluster is the
   would-be winner, and the diagnostic shows them in green together.

Also constructs a "real" neighboring-line case (a second line at a clearly
different y) and confirms the merge step does NOT bridge across y.

Regression coverage for the x-overlap merge bug (found via box_0060 in the
28July e2e corpus, page layer_1_3801 x 5013): `_compute_merge_groups` clustered
two components using only y-center distance and an x-gap upper bound, with no
check that the x-ranges were actually disjoint. Two fully-overlapping-in-x,
physically distinct parallel stafflines made the x-gap negative (trivially
under the upper bound), so the only thing standing between them and a false
merge was the y-center threshold (1.0 * scale_unit) -- which on that page
(scale_unit=71px) exceeded the real measured inter-line spacing (~44-53px).
`test_two_close_overlapping_lines_do_not_merge` reproduces that precondition
directly; `test_wide_gap_fragments_still_merge_at_realistic_scale` confirms the
fix (excluding negative gaps) doesn't disturb legitimate same-line fragment
bridging at the same scale.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from component_filter import filter_components


def make_fragmented_line_crop(
    width=400, height=80, line_y=40, thickness=4, n_fragments=5, gap_width=10
):
    """White background with a horizontal line broken into N evenly-spaced fragments."""
    crop = np.full((height, width, 3), 255, dtype=np.uint8)
    half = thickness // 2

    fragment_pitch = width // n_fragments
    fragment_len = fragment_pitch - gap_width
    for i in range(n_fragments):
        x_start = i * fragment_pitch
        x_end = x_start + fragment_len
        crop[line_y - half : line_y + half + 1, x_start:x_end] = 30

    return crop


def make_two_lines_crop(width=400, height=120, y_top=30, y_bottom=90, thickness=4):
    """White background with two horizontal lines at different y."""
    crop = np.full((height, width, 3), 255, dtype=np.uint8)
    half = thickness // 2
    crop[y_top - half : y_top + half + 1, :] = 30
    crop[y_bottom - half : y_bottom + half + 1, :] = 30
    return crop


def test_fragmented_line_merges():
    n_fragments = 5  # matches make_fragmented_line_crop's default
    crop = make_fragmented_line_crop(n_fragments=n_fragments)
    save_path = Path("/tmp/merge_test_fragmented.png")
    if save_path.exists():
        save_path.unlink()

    # Default mode: merge_components=True. The active coords/mask should
    # reflect the merged cluster (all fragments together), while no_merge_*
    # should still reflect the single-fragment winner.
    result = filter_components(crop, scale_unit=4.0, save_path=save_path)
    assert "no_components_survived" not in result.flags
    n_active = len(result.coords)
    n_no_merge = len(result.no_merge_coords)
    print(
        f"fragmented line (merge=ON, default): "
        f"active coords={n_active}, no_merge coords={n_no_merge}"
    )
    # Active should be much larger than no-merge (all fragments vs. one).
    assert (
        n_active > n_no_merge * 2
    ), f"merge mode should keep substantially more pixels: active={n_active}, no_merge={n_no_merge}"
    assert len(result.merged_cluster_labels) >= 2

    # No-merge mode: explicit merge_components=False. Companion retention
    # (see COMPANION_SCORE_FLOOR in component_filter.py) runs in the no-merge
    # path too, so the active coords/mask are the winner PLUS any qualifying
    # companions -- not just the bare winner. no_merge_coords, by contrast,
    # always holds only the winner's own pixels. Here every other fragment is
    # equal-sized, shares the winner's y-center, and doesn't overlap it in x,
    # so all of them qualify as companions of the single no-merge winner.
    result_no_merge = filter_components(
        crop, scale_unit=4.0, save_path=None, merge_components=False
    )
    assert len(result_no_merge.companion_labels) == n_fragments - 1, (
        "expected every other fragment to qualify as a companion of the "
        f"no-merge winner; got companion_labels={result_no_merge.companion_labels}"
    )
    assert len(result_no_merge.coords) == len(result_no_merge.no_merge_coords) * n_fragments, (
        "no-merge active coords should be the winner plus all qualifying "
        f"companions: active={len(result_no_merge.coords)}, "
        f"no_merge (winner only)={len(result_no_merge.no_merge_coords)}, "
        f"n_fragments={n_fragments}"
    )
    n_not_top = sum(
        1 for d in result_no_merge.discarded if d.get("reason") == "not_top_scoring"
    )
    print(
        f"fragmented line (merge=OFF): kept components = 1 winner + "
        f"{len(result_no_merge.companion_labels)} companion(s), "
        f"discarded as 'not_top_scoring' = {n_not_top}"
    )
    assert (
        n_not_top >= 1
    ), "expected fragments to compete for the kept spot when merge is off"

    print(f"  diagnostic saved to {save_path}, size={save_path.stat().st_size} bytes")


def test_two_lines_do_not_merge():
    crop = make_two_lines_crop()
    save_path = Path("/tmp/merge_test_two_lines.png")
    if save_path.exists():
        save_path.unlink()
    result = filter_components(crop, scale_unit=4.0, save_path=save_path)
    print(
        f"two lines: kept pixels={len(result.coords)}, "
        f"discarded count={len(result.discarded)}, flags={result.flags}"
    )
    # 60px apart in y, far beyond y_threshold (1.0*4.0=4px) -- baseline
    # sanity check. Does not exercise the x-overlap bug itself; see
    # test_two_close_overlapping_lines_do_not_merge for that.
    assert len(result.merged_cluster_labels) == 1, (
        f"lines 60px apart in y should never merge; got "
        f"merged_cluster_labels={result.merged_cluster_labels}"
    )
    assert len(result.coords) == len(result.no_merge_coords), (
        "active (merged) coords should match the single no-merge winner"
    )
    print(f"  diagnostic saved to {save_path}, size={save_path.stat().st_size} bytes")


def test_two_close_overlapping_lines_do_not_merge():
    """Regression test for the x-overlap merge bug in
    component_filter._compute_merge_groups.

    Reproduces the real failure precondition (see box_0060 in the e2e test
    corpus): two full-width, fully x-overlapping stafflines, close enough in
    y that the y-center check alone does not exclude them. Uses the exact
    scale_unit measured on the diagnosed page (71.0) and a y-separation
    (48px) inside the page's own measured inter-line spacing (44-53px), so
    spacing is comfortably under 1.0 * scale_unit = 71px and only the
    x-overlap guard stands between a correct result and the bug.

    Pre-fix: gap = right.x_start - left.x_end goes negative for these
    x-overlapping lines, and only `gap > x_gap_threshold` was checked -- a
    negative gap trivially passes, so the two distinct lines get unioned.
    Post-fix: `gap < 0` excludes them.
    """
    scale_unit = 71.0
    crop = make_two_lines_crop(width=600, height=160, y_top=45, y_bottom=93, thickness=4)
    save_path = Path("/tmp/merge_test_two_close_overlapping_lines.png")
    if save_path.exists():
        save_path.unlink()
    result = filter_components(crop, scale_unit=scale_unit, save_path=save_path)
    print(
        f"two close overlapping lines: kept={len(result.coords)}, "
        f"no_merge={len(result.no_merge_coords)}, "
        f"merged_cluster_labels={result.merged_cluster_labels}, "
        f"flags={result.flags}"
    )
    assert len(result.merged_cluster_labels) == 1, (
        "two x-overlapping, y-close but physically distinct stafflines must "
        f"NOT be unioned; got merged_cluster_labels={result.merged_cluster_labels}"
    )
    assert len(result.coords) == len(result.no_merge_coords), (
        "with the two lines correctly kept separate, active coords should "
        "equal the single no-merge winner's coords"
    )
    print(f"  diagnostic saved to {save_path}, size={save_path.stat().st_size} bytes")


def test_wide_gap_fragments_still_merge_at_realistic_scale():
    """Non-regression check for the x-overlap fix: two non-overlapping
    fragments of the SAME line, separated by a wide horizontal gap (e.g. a
    decorated initial), must still merge. The fix only excludes negative
    (overlapping) gaps, not large positive ones.

    Same scale_unit as test_two_close_overlapping_lines_do_not_merge, for
    direct comparison. Gap=500px, under
    MERGE_X_GAP_MULTIPLIER * scale_unit = 10.0 * 71.0 = 710px.
    """
    scale_unit = 71.0
    # n_fragments=2, gap_width=500, width=1400: fragment 0 spans x=[0,200),
    # fragment 1 spans x=[700,900) -- gap = 700-200 = 500px, positive and
    # well under the 710px threshold.
    crop = make_fragmented_line_crop(
        width=1400, height=80, line_y=40, thickness=4, n_fragments=2, gap_width=500
    )
    save_path = Path("/tmp/merge_test_wide_gap_same_line.png")
    if save_path.exists():
        save_path.unlink()
    result = filter_components(crop, scale_unit=scale_unit, save_path=save_path)
    print(
        f"wide-gap same-line fragments: kept={len(result.coords)}, "
        f"no_merge={len(result.no_merge_coords)}, "
        f"merged_cluster_labels={result.merged_cluster_labels}, "
        f"flags={result.flags}"
    )
    assert len(result.merged_cluster_labels) == 2, (
        "two non-overlapping fragments of the same line, 500px apart "
        f"(< 710px threshold), should still merge; got "
        f"merged_cluster_labels={result.merged_cluster_labels}"
    )
    assert len(result.coords) > len(result.no_merge_coords) * 1.5, (
        "merged coords should cover both fragments, not just one winner"
    )
    print(f"  diagnostic saved to {save_path}, size={save_path.stat().st_size} bytes")


if __name__ == "__main__":
    test_fragmented_line_merges()
    test_two_lines_do_not_merge()
    test_two_close_overlapping_lines_do_not_merge()
    test_wide_gap_fragments_still_merge_at_realistic_scale()
    print("\nMerge-step synthetic tests done.")
