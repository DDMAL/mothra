"""
Periodicity-aware staffline tracer using a comb-filter cost and dynamic programming.

Background
----------
Plain DP tracing (dp_tracer.py) minimises pixel brightness column-by-column: dark
pixels are cheap; bright pixels are expensive.  This works well for isolated lines
but can drift when ink is faint or when two nearby stafflines have similar density.

This module adds a **comb-filter cost**: before running DP, the data cost at every
candidate row y is augmented with the darkness values at y ± h, y ± 2h, … (where h
is the estimated inter-staffline period).  If the traced staffline truly sits at y,
then the adjacent stafflines should create matching dark signals at y ± h — so the
comb rewards self-consistent hypotheses and penalises drift into inter-line gaps.

With n_teeth=1, the comb degenerates to plain DP (no additional teeth).
With n_teeth=3, the comb samples y-h, y, y+h — one neighbour on each side.
With n_teeth=5, the comb samples y-2h, y-h, y, y+h, y+2h — two on each side.

The period h is estimated from the vertical autocorrelation of the crop's mean
intensity profile.  This makes the approach almost parameter-free on new material:
only MAX_STEP_PX and BAND_HALF_MULTIPLIER need tuning, and both are shared with the
plain DP tracer.

Coordinate convention
---------------------
All inputs and outputs are in page-absolute pixel coordinates.
"""

import numpy as np
from scipy.ndimage import minimum_filter1d, gaussian_filter1d

# ---------------------------------------------------------------------------
# Tunable defaults
# ---------------------------------------------------------------------------

# Half-width of the y-search band around the YOLO box y-center, as a multiple
# of the scale unit h.  1.5h mirrors the plain DP default.
BAND_HALF_MULTIPLIER = 1.5

# Maximum y-step allowed between adjacent columns (pixels).  Matches dp_tracer.
MAX_STEP_PX = 3

# Sigma for Gaussian pre-blur, as a multiple of h.  0.2h is gentle — just enough
# to suppress isolated noise pixels without smearing staffline structure.
BLUR_SIGMA_MULTIPLIER = 0.2

# Number of comb teeth.  Must be odd (1, 3, 5, …).  1 = plain DP; 3 = one
# neighbour on each side.  Beyond 5 teeth rarely helps and risks pulling toward
# distant structures.
#
# Unlike dp_tracer.py's HINT_DISTANCE_PENALTY_WEIGHT (same role: how much an
# auxiliary signal should outweigh the primary darkness cost), N_TEETH and
# TEETH_WEIGHT below never got an equivalent synthetic-benchmark derivation.
# NOTES.md's own run log found the underlying autocorrelation period
# estimate unreliable per-YOLO-box on the one manuscript tested (falling
# back to h_est=scale_unit for most/all lines) -- these defaults are
# initial guesses awaiting the same kind of empirical tuning pass, not
# validated values.
N_TEETH = 3

# Weight given to each off-centre tooth relative to the centre tooth (weight=1).
# 0.4 means neighbours contribute 40 % each; 0.0 collapses to plain DP regardless
# of N_TEETH.
TEETH_WEIGHT = 0.4


# ---------------------------------------------------------------------------
# Period estimation
# ---------------------------------------------------------------------------


def estimate_period(
    gray_crop: np.ndarray,
    scale_unit: float,
) -> tuple[float, float]:
    """Estimate the inter-staffline period from a grayscale crop.

    Uses the vertical autocorrelation of the column-mean intensity profile.
    Dark pixels produce a high signal; the autocorrelation peaks at the dominant
    period.

    Args:
        gray_crop:   Grayscale sub-image (H_crop, W_crop), dtype uint8.
                     Should span several stafflines so the period is visible.
        scale_unit:  Page-level scale unit h (median staffline box height, px).
                     Used to define the lag search window.

    Returns:
        (h_est, confidence) where:
            h_est      — estimated period in pixels.  Falls back to scale_unit
                         if the autocorrelation is flat or the crop is too small.
            confidence — normalised peak height in [0, 1].  Values above ~0.3
                         usually indicate a reliable estimate.
    """
    fallback = (float(scale_unit), 0.0)

    if gray_crop.size == 0 or scale_unit <= 0:
        return fallback

    # Mean intensity along each row -> 1-D vertical profile.
    profile = gray_crop.mean(axis=1).astype(np.float64)  # shape: (H_crop,)

    if len(profile) < 4:
        return fallback

    # Invert so dark (ink) regions become high-amplitude signal.
    signal = 255.0 - profile

    # Zero-centre and normalise to unit variance so the autocorrelation
    # coefficient is dimensionless.
    signal -= signal.mean()
    std = signal.std()
    if std < 1e-6:
        return fallback
    signal /= std

    # Full autocorrelation via numpy; length = 2*N - 1.
    full_corr = np.correlate(signal, signal, mode="full")
    N = len(signal)
    # Positive-lag half (lag 0 at index 0 after slicing mid-point).
    acorr = full_corr[N - 1 :]  # acorr[0] = zero-lag (maximum)
    acorr = acorr / (acorr[0] + 1e-12)  # normalise so zero-lag = 1

    # Define the lag search window: [lag_lo, lag_hi]. Deliberately wider
    # than run_periodicity_page.py's H_RATIO_LO/HI=(0.7, 1.5) acceptance
    # gate around scale_unit -- search broadly here for any plausible
    # autocorrelation peak, then let the caller reject it if it lands
    # outside the physically-plausible range. Two-stage by design: a narrow
    # search window here would risk missing the true period on pages where
    # it deviates further from scale_unit than the acceptance gate allows.
    lag_lo = max(1, int(np.floor(0.4 * scale_unit)))
    lag_hi = min(len(acorr) - 1, int(np.ceil(2.5 * scale_unit)))

    if lag_lo >= lag_hi:
        return fallback

    search = acorr[lag_lo : lag_hi + 1]
    peak_idx = int(np.argmax(search))
    peak_lag = lag_lo + peak_idx
    confidence = float(np.clip(search[peak_idx], 0.0, 1.0))

    return (float(peak_lag), confidence)


# ---------------------------------------------------------------------------
# Core comb-filter DP
# ---------------------------------------------------------------------------


def compute_dark_field(
    gray: np.ndarray, scale_unit: float, blur_sigma_multiplier: float = BLUR_SIGMA_MULTIPLIER
) -> np.ndarray:
    """Precompute the page-wide darkness field once per page.

    smoothed/dark depend only on gray and scale_unit (via
    blur_sigma_multiplier), both invariant across a whole page -- hoisted out
    of periodicity_trace() so a per-staffline loop doesn't redo this full-page
    Gaussian blur and normalisation once per detection.
    """
    sigma = blur_sigma_multiplier * scale_unit
    # Below 0.5px a Gaussian kernel barely touches neighbouring pixels --
    # visually negligible, but a full-page gaussian_filter1d call isn't
    # free, so this is a "not worth paying for" cutoff rather than a
    # correctness one. Since sigma scales with scale_unit, it also means
    # very low-resolution scans get no pre-blur at all -- the noise-
    # suppression benefit silently disappears there, not just shrinks.
    if sigma > 0.5:
        smoothed = gaussian_filter1d(gray.astype(np.float32), sigma=sigma, axis=0)
    else:
        smoothed = gray.astype(np.float32)
    return (255.0 - smoothed) / 255.0


def periodicity_trace(
    gray: np.ndarray,
    y_hint: float,
    x_start: int,
    x_end: int,
    scale_unit: float,
    h_est: float | None = None,
    band_half_multiplier: float = BAND_HALF_MULTIPLIER,
    max_step_px: int = MAX_STEP_PX,
    blur_sigma_multiplier: float = BLUR_SIGMA_MULTIPLIER,
    n_teeth: int = N_TEETH,
    teeth_weight: float = TEETH_WEIGHT,
    dark: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace one staffline using DP with a comb-filter data cost.

    The comb augments the standard pixel-darkness cost at row y with additional
    darkness samples at y ± k*h for k = 1, 2, … up to (n_teeth-1)//2.  When the
    traced row is genuinely a staffline, the comb teeth land on adjacent stafflines
    and reinforce the cost signal; when it is between lines the teeth land on bright
    gaps and increase the cost, discouraging drift.

    All coordinates are page-absolute.

    Args:
        gray:                  Full-page grayscale image (H, W), dtype uint8.
        y_hint:                Approximate y-centre of the staffline (page-abs).
        x_start:               First column to trace (page-abs, inclusive).
        x_end:                 Last column to trace (page-abs, inclusive).
        scale_unit:            Page-level h (median staffline box height, px).
        h_est:                 Inter-staffline period in pixels.  Defaults to
                               scale_unit when None.
        band_half_multiplier:  Search band = ± band_half_multiplier × h.
        max_step_px:           Max y-shift allowed between adjacent columns.
        blur_sigma_multiplier: Gaussian pre-blur sigma = multiplier × h.
        n_teeth:               Number of comb teeth (must be odd; 1 = plain DP).
        teeth_weight:          Weight per off-centre tooth (0 = plain DP).
        dark:                  Precomputed page-wide darkness field from
                               compute_dark_field(), reused as-is if given.
                               Computed here (and thrown away) if None --
                               pass it explicitly from a per-page loop to
                               avoid redoing the full-page blur per call.

    Returns:
        (xs, ys): page-absolute x and y arrays for the traced path.
        xs has length (x_end - x_start + 1).  ys[i] is the traced y at xs[i].
    """
    if h_est is None:
        h_est = float(scale_unit)

    page_h, page_w = gray.shape[:2]

    # Clamp column range to image bounds.
    x_start = max(0, x_start)
    x_end = min(page_w - 1, x_end)
    n_cols = x_end - x_start + 1

    # Search band (clamped to image).
    band_half = band_half_multiplier * scale_unit
    y_lo = max(0, int(np.floor(y_hint - band_half)))
    y_hi = min(page_h - 1, int(np.ceil(y_hint + band_half)))
    n_band = y_hi - y_lo + 1

    if n_band < 1 or n_cols < 1:
        xs = np.arange(x_start, x_end + 1)
        ys = np.full(len(xs), y_hint)
        return xs, ys

    # dark[y, x] = (255 - pixel) / 255 for the full page, so dark ink = high
    # value. We keep the full page here so that comb teeth can reach rows
    # outside the band. Reuse the caller's precomputed field when given
    # (see the dark parameter above); only compute it here as a fallback.
    if dark is None:
        dark = compute_dark_field(gray, scale_unit, blur_sigma_multiplier)

    # -----------------------------------------------------------------
    # Build comb cost for the search band.
    # cost[r, c] = darkness(y_lo+r, x_start+c)
    #            + teeth_weight * darkness(y_lo+r - h_est, x_start+c)  [if in bounds]
    #            + teeth_weight * darkness(y_lo+r + h_est, x_start+c)  [if in bounds]
    #            + … for further teeth
    # Then normalised to [0, 1] and inverted so low cost = dark = good.
    # -----------------------------------------------------------------

    # Centre slice: band rows only, columns x_start..x_end.
    # shape: (n_band, n_cols)
    band_dark = dark[y_lo : y_hi + 1, x_start : x_end + 1]

    comb_cost = band_dark.copy()

    n_extra = (n_teeth - 1) // 2  # number of teeth on each side
    for k in range(1, n_extra + 1):
        offset_px = int(round(k * h_est))

        # Which band rows have a valid upper tooth?
        # Band row r maps to page row y_lo + r.
        # Upper tooth page row: y_lo + r - offset_px.
        # Valid when 0 <= y_lo + r - offset_px <= page_h - 1
        #       i.e. offset_px <= r + y_lo  and  r + y_lo - offset_px <= page_h - 1
        r_up_lo = max(0, offset_px - y_lo)  # first band row with valid upper tooth
        r_up_hi = min(n_band - 1, page_h - 1 - y_lo + offset_px)

        if r_up_lo <= r_up_hi and r_up_lo < n_band:
            tooth_up_rows = slice(
                max(0, y_lo + r_up_lo - offset_px),
                min(page_h, y_lo + r_up_hi - offset_px + 1),
            )
            band_rows_up = slice(r_up_lo, r_up_hi + 1)
            comb_cost[band_rows_up, :] += (
                teeth_weight * dark[tooth_up_rows, x_start : x_end + 1]
            )

        # Lower tooth: y_lo+r + offset_px
        r_dn_lo = max(0, 0)
        r_dn_hi = min(n_band - 1, page_h - 1 - y_lo - offset_px)

        if r_dn_lo <= r_dn_hi and r_dn_hi >= 0:
            tooth_dn_rows = slice(
                y_lo + r_dn_lo + offset_px,
                y_lo + r_dn_hi + offset_px + 1,
            )
            band_rows_dn = slice(r_dn_lo, r_dn_hi + 1)
            comb_cost[band_rows_dn, :] += (
                teeth_weight * dark[tooth_dn_rows, x_start : x_end + 1]
            )

    # Normalise comb_cost to [0, 1] then invert so low cost = good (dark).
    # Global min/max over the whole band x columns array, not per-column:
    # a per-column rescale would flatten every column to the same [0, 1]
    # range regardless of how strongly periodic its signal actually is,
    # erasing exactly the comb-teeth signal this cost is built to preserve.
    # Also unlike dp_tracer.py's plain darkness cost (bare /255.0, no
    # rescale needed there), comb_cost isn't naturally bounded to [0, 1] --
    # teeth_weight contributions can push it above 1 (e.g. n_teeth=3,
    # teeth_weight=0.4 → worst case ~1.8) and fewer teeth land in-bounds
    # near page edges, so its raw scale isn't uniform across rows either.
    # When the field is perfectly flat (c_max == c_min, e.g. a blank/uniform
    # crop), there's no signal at all to trace: minimum_filter1d only bounds
    # how far the path can step between columns, it doesn't penalise moving,
    # so a uniform data_cost leaves np.argmin's tie-break to pick the first
    # band row (y_lo) below -- a flat field would silently return a path
    # pinned to the top of the search band instead of near y_hint. Bail out
    # the same way the n_band/n_cols guard above does: a stable path at
    # y_hint is preferable to running DP over a cost surface with nothing in
    # it to optimise.
    c_min = comb_cost.min()
    c_max = comb_cost.max()
    if c_max <= c_min:
        xs = np.arange(x_start, x_end + 1)
        ys = np.full(len(xs), float(np.clip(y_hint, y_lo, y_hi)))
        return xs, ys
    comb_cost = (comb_cost - c_min) / (c_max - c_min)
    data_cost = 1.0 - comb_cost  # low = good signal = ink

    # -----------------------------------------------------------------
    # Forward DP — same vectorised pattern as dp_tracer.py.
    # dp_prev[r] = minimum cumulative cost to be at band row r at the
    # current column.  minimum_filter1d slides a window of size
    # 2*max_step_px+1 and replaces each entry with the min over its
    # neighbourhood, effectively evaluating all reachable predecessors
    # without a Python inner loop.
    # -----------------------------------------------------------------
    dp_prev = data_cost[:, 0].copy()
    dp_table = np.empty((n_cols, n_band), dtype=np.float32)
    dp_table[0] = dp_prev

    window = 2 * max_step_px + 1
    for col in range(1, n_cols):
        reachable = minimum_filter1d(dp_prev, size=window, mode="nearest")
        dp_col = data_cost[:, col] + reachable
        dp_table[col] = dp_col
        dp_prev = dp_col

    # -----------------------------------------------------------------
    # Backtrack.
    # -----------------------------------------------------------------
    y_idx = int(np.argmin(dp_table[-1]))
    path_indices = [y_idx]
    for col in range(n_cols - 1, 0, -1):
        lo = max(0, path_indices[-1] - max_step_px)
        hi = min(n_band, path_indices[-1] + max_step_px + 1)
        best_prev = int(lo + np.argmin(dp_table[col - 1, lo:hi]))
        path_indices.append(best_prev)
    path_indices.reverse()

    xs = np.arange(x_start, x_end + 1)
    ys = np.array(path_indices, dtype=float) + y_lo

    return xs, ys
