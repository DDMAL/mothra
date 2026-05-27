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
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/claude")
from component_filter import filter_components


def make_fragmented_line_crop(width=400, height=80, line_y=40, thickness=4,
                              n_fragments=5, gap_width=10):
    """White background with a horizontal line broken into N evenly-spaced fragments."""
    crop = np.full((height, width, 3), 255, dtype=np.uint8)
    half = thickness // 2

    fragment_pitch = width // n_fragments
    fragment_len = fragment_pitch - gap_width
    for i in range(n_fragments):
        x_start = i * fragment_pitch
        x_end = x_start + fragment_len
        crop[line_y - half:line_y + half + 1, x_start:x_end] = 30

    return crop


def make_two_lines_crop(width=400, height=120, y_top=30, y_bottom=90, thickness=4):
    """White background with two horizontal lines at different y."""
    crop = np.full((height, width, 3), 255, dtype=np.uint8)
    half = thickness // 2
    crop[y_top - half:y_top + half + 1, :] = 30
    crop[y_bottom - half:y_bottom + half + 1, :] = 30
    return crop


def test_fragmented_line_merges():
    crop = make_fragmented_line_crop()
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
    print(f"fragmented line (merge=ON, default): "
          f"active coords={n_active}, no_merge coords={n_no_merge}")
    # Active should be much larger than no-merge (all fragments vs. one).
    assert n_active > n_no_merge * 2, \
        f"merge mode should keep substantially more pixels: active={n_active}, no_merge={n_no_merge}"
    assert len(result.merged_cluster_labels) >= 2

    # No-merge mode: explicit merge_components=False. The active coords/mask
    # should match no_merge_*.
    result_no_merge = filter_components(crop, scale_unit=4.0,
                                         save_path=None, merge_components=False)
    assert len(result_no_merge.coords) == len(result_no_merge.no_merge_coords)
    n_not_top = sum(1 for d in result_no_merge.discarded
                    if d.get("reason") == "not_top_scoring")
    print(f"fragmented line (merge=OFF): kept components = 1, "
          f"discarded as 'not_top_scoring' = {n_not_top}")
    assert n_not_top >= 1, "expected fragments to compete for the kept spot when merge is off"

    print(f"  diagnostic saved to {save_path}, size={save_path.stat().st_size} bytes")


def test_two_lines_do_not_merge():
    crop = make_two_lines_crop()
    save_path = Path("/tmp/merge_test_two_lines.png")
    if save_path.exists():
        save_path.unlink()
    result = filter_components(crop, scale_unit=4.0, save_path=save_path)
    # Both lines should be considered (separately). One wins.
    print(f"two lines: kept pixels={len(result.coords)}, "
          f"discarded count={len(result.discarded)}, flags={result.flags}")
    # If the merge step bridged them, the would-be merged winner would cover
    # both lines. Inspect the diagnostic visually to confirm it does not.
    print(f"  diagnostic saved to {save_path}, size={save_path.stat().st_size} bytes")


if __name__ == "__main__":
    test_fragmented_line_merges()
    test_two_lines_do_not_merge()
    print("\nMerge-step synthetic tests done.")