"""
Shortest-path staffline tracer using dynamic programming.

Given a grayscale page image and a rough y-hint (from a YOLO box), traces the
staffline by finding the minimum-cost path through the image column by column.
Dark pixels (ink) carry low cost; bright pixels (parchment) carry high cost.
A sliding-window minimum enforces a smoothness constraint: the path y-position
can shift by at most MAX_STEP_PX between adjacent columns.

This replaces the binarize-then-fit-components pipeline entirely.  No model,
no binarization threshold, no connected-component analysis — just the raw
grayscale image and a DP sweep.

Coordinate convention
---------------------
All inputs and outputs are in page-absolute pixel coordinates.
"""

import numpy as np
from scipy.ndimage import minimum_filter1d, gaussian_filter1d

# ---------------------------------------------------------------------------
# Tunable defaults
# ---------------------------------------------------------------------------

# Half-width of the y-search band around the YOLO box y-center, expressed as
# a multiple of the scale unit h.  1.5h gives one full line thickness of slack
# on each side — enough to follow moderate parchment warp without jumping to an
# adjacent staffline.
BAND_HALF_MULTIPLIER = 1.5

# Maximum y-step allowed between adjacent columns (pixels).  Controls path
# smoothness; 3px per column gives ~2.7° maximum local slope.
MAX_STEP_PX = 3

# Sigma for Gaussian smoothing of the grayscale image before computing data
# costs.  Reduces sensitivity to isolated noise pixels.  Expressed as a
# multiple of h; 0.2h is gentle — just enough to blur single-pixel speckle.
BLUR_SIGMA_MULTIPLIER = 0.2

# Weight of the distance-from-y_hint penalty added to the per-column data
# cost.  Without this, the cost only rewards dark pixels, so when the search
# band (±band_half_multiplier*h) overlaps a neighbouring staffline, the DP
# can converge onto that darker neighbour instead of following its own
# YOLO-box hint. The penalty is normalised to [0, 1] across the band before
# weighting, so it scales consistently regardless of band size.
#
# 0.7 was chosen empirically, not just theoretically: on a synthetic band
# with a faint true line (grey 120) and a darker, thicker distractor
# (grey 20) 12px from the hint in a 15px half-band, weights below ~0.6 still
# let the distractor win (confirming the bug is real, not just theoretical);
# 0.6 is the observed tipping point, and 0.7 keeps a small margin above it.
# Real manuscript contrast/spacing will differ -- tune alongside real
# dp_tracing runs before relying on this for anything beyond comparison
# against the other experiment runners.
HINT_DISTANCE_PENALTY_WEIGHT = 0.7


# ---------------------------------------------------------------------------
# Core DP
# ---------------------------------------------------------------------------


def dp_trace(
    gray: np.ndarray,
    y_hint: float,
    x_start: int,
    x_end: int,
    scale_unit: float,
    band_half_multiplier: float = BAND_HALF_MULTIPLIER,
    max_step_px: int = MAX_STEP_PX,
    blur_sigma_multiplier: float = BLUR_SIGMA_MULTIPLIER,
    hint_distance_penalty_weight: float = HINT_DISTANCE_PENALTY_WEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace one staffline by DP.

    Args:
        gray:                 Full-page grayscale image (H, W), dtype uint8.
        y_hint:               Approximate y-center of the staffline (page-abs).
        x_start:              First column to trace (page-absolute, inclusive).
        x_end:                Last column to trace (page-absolute, inclusive).
        scale_unit:           Page-level h (median staffline box height, px).
        band_half_multiplier: Search band = ± band_half_multiplier × h.
        max_step_px:          Max y-shift allowed between adjacent columns.
        blur_sigma_multiplier: Gaussian blur sigma = multiplier × h.
        hint_distance_penalty_weight: Weight of the distance-from-y_hint
            penalty, keeping the trace anchored to its own YOLO hint instead
            of drifting onto a darker neighbouring staffline within the same
            search band. 0 reproduces the original darkness-only cost.

    Returns:
        (xs, ys): page-absolute x and y arrays for the traced path.
        xs has length (x_end - x_start + 1).  ys[i] is the traced y at xs[i].
    """
    page_h, page_w = gray.shape[:2]

    # Clamp column range to image bounds (both endpoints), and reject an
    # inverted/fully out-of-bounds range rather than let n_cols go negative.
    x_start = max(0, min(x_start, page_w - 1))
    x_end = max(0, min(x_end, page_w - 1))
    if x_end < x_start:
        xs = np.array([x_start])
        ys = np.array([y_hint])
        return xs, ys
    n_cols = x_end - x_start + 1

    # Search band (clamped to image).
    band_half = band_half_multiplier * scale_unit
    y_min = max(0, int(np.floor(y_hint - band_half)))
    y_max = min(page_h - 1, int(np.ceil(y_hint + band_half)))
    n_band = y_max - y_min + 1

    if n_band < 1 or n_cols < 1:
        xs = np.arange(x_start, x_end + 1)
        ys = np.full(len(xs), y_hint)
        return xs, ys

    # Light Gaussian blur to reduce pixel-level noise. 0.5px cutoff below is
    # a "not worth the full-page filter1d call" threshold, not a
    # correctness one -- also means very low-resolution scans (small
    # scale_unit) silently get no pre-blur at all.
    sigma = blur_sigma_multiplier * scale_unit
    if sigma > 0.5:
        smoothed = gaussian_filter1d(gray.astype(np.float32), sigma=sigma, axis=0)
    else:
        smoothed = gray.astype(np.float32)

    # Data cost: [0=dark/ink, 1=bright/parchment]
    # shape: (n_band, n_cols)
    band_img = smoothed[y_min : y_max + 1, x_start : x_end + 1] / 255.0

    # Distance-from-y_hint penalty, normalised to [0, 1] across the band and
    # added at every column so the DP stays anchored to its own YOLO hint
    # instead of drifting onto a darker neighbouring staffline that also
    # falls within the search band. Same value at every column (the hint
    # doesn't move column to column), so computed once here.
    band_rows = np.arange(y_min, y_max + 1, dtype=np.float32)
    hint_penalty = np.clip(np.abs(band_rows - y_hint) / max(band_half, 1e-6), 0.0, 1.0)
    hint_penalty *= hint_distance_penalty_weight

    # DP forward pass.
    # dp[y_idx] = minimum cumulative cost to reach y_idx at the current column.
    dp_prev = band_img[:, 0].copy() + hint_penalty
    dp_table = np.empty((n_cols, n_band), dtype=np.float32)
    dp_table[0] = dp_prev

    window = 2 * max_step_px + 1
    for col in range(1, n_cols):
        reachable = minimum_filter1d(dp_prev, size=window, mode="nearest")
        dp_col = band_img[:, col] + hint_penalty + reachable
        dp_table[col] = dp_col
        dp_prev = dp_col

    # Backtrack.
    y_idx = int(np.argmin(dp_table[-1]))
    path_indices = [y_idx]
    for col in range(n_cols - 1, 0, -1):
        lo = max(0, path_indices[-1] - max_step_px)
        hi = min(n_band, path_indices[-1] + max_step_px + 1)
        best_prev = int(lo + np.argmin(dp_table[col - 1, lo:hi]))
        path_indices.append(best_prev)
    path_indices.reverse()

    xs = np.arange(x_start, x_end + 1)
    ys = np.array(path_indices, dtype=float) + y_min

    return xs, ys
