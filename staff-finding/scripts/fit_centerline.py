"""
Centerline fitting for Stage 1 of the staff detection pipeline.

Given a ComponentFilterResult (from component_filter.filter_components), fits
a robust quadratic curve to the kept pixels and samples y at every integer x
in the kept-component's horizontal extent. Output is the per-line centerline
representation consumed by Stage 2 and downstream pitch finding.

See ADR-001 (component filter design) and the Stage 1 design doc §5.8 for the
output schema this module's results contribute to.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from component_filter import ComponentFilterResult

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# Polynomial degree for the fit. Quadratic per the design doc — handles
# straight and mildly-curved lines well, low overfitting risk.
POLY_DEGREE = 2

# Huber's f_scale: the transition distance (in pixels) where the loss switches
# from quadratic to linear. Expressed as a multiple of the scale unit h, so it
# adapts to manuscript scale. 0.5 * h means pixels more than half a line
# thickness off the fit get down-weighted.
HUBER_F_SCALE_MULTIPLIER = 0.5

# Maximum iterations for the robust fit. Default in scipy is ample; we cap it
# so a pathological case doesn't hang.
MAX_FIT_ITERATIONS = 100

# Decimals to round sampled y-values to. One decimal per ADR-001 §7 / design
# doc §5.4 — sufficient for square notation; revisit for thinner scripts.
Y_SAMPLE_DECIMALS = 1

# Line-following: triggered when the initial Huber fit's mean residual exceeds
# this multiple of h. At 1.0 * h the fit is already off by a full line
# thickness on average — a strong signal that the box contains more than one
# staffline.
REFIT_TRIGGER_MULTIPLIER = 1.0

# Half-width of the sliding band that follows the current estimated y (pixels,
# expressed as a multiple of h). 1.5 * h gives one full line thickness of
# tolerance on each side, letting the trace follow typical curvature while
# refusing to jump to a neighbouring staffline.
LINE_FOLLOW_BAND_MULTIPLIER = 1.5

# Sliding window width for the trace = scale_unit / this divisor (min 5 px).
# Divisor 2 → roughly half a line-thickness per window; fine-grained enough to
# track gentle curves without being so narrow that sparse pixels dominate.
LINE_FOLLOW_WINDOW_DIVISOR = 2

# Minimum number of window medians before a polynomial refit is attempted.
# Below this the polynomial is ill-conditioned.
LINE_FOLLOW_MIN_TRACE_POINTS = 5

# Polynomial degree for the line-following refit — separate from the global
# POLY_DEGREE=2 for the initial Huber fit.  Cubic (3) can represent both
# C-shaped arches (page curvature) and S-shaped waves (parchment warping in
# two spots), without material overfitting risk when gated on enough trace
# points. Falls back to POLY_DEGREE if fewer than 2*(degree+1) = 8 trace
# points are available.
LINE_FOLLOW_POLY_DEGREE = 3


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    """Output of the centerline fit.

    Coordinate convention
    ---------------------
    x_start, x_end, and y_values are all in **crop-local** coordinates — the
    origin is the top-left corner of the YOLO bounding box that was cropped
    before fitting. To convert to page-absolute coordinates add x_page_offset
    (for x) or y_page_offset (for y). These offsets are populated by run_page.py
    from the YOLO bounding box (ulx, uly) and default to 0 for synthetic data
    used in unit tests.

    Attributes:
        x_start: First x of the sampled centerline (crop-local integer pixel).
        x_end: Last x of the sampled centerline (crop-local integer pixel; inclusive).
        y_values: One y per integer x in [x_start, x_end], crop-local. Empty
            when no fit was produced.
        coefficients: Polynomial coefficients for the fitted curve. Empty when
            no fit was produced.
        residual_mean: Mean absolute deviation of fitted curve from kept pixels.
        residual_max: Max absolute deviation.
        n_pixels_used: Number of kept-coord pixels fed to the fit.
        n_pixels_total: Total pixels in the kept component (same as n_pixels_used
            while Huber down-weights rather than drops; field exists for schema
            compliance and future RANSAC / hard-rejection modes).
        x_page_offset: Page-absolute x of the crop's left edge (ulx). Add to
            x_start/x_end to get page coordinates.
        y_page_offset: Page-absolute y of the crop's top edge (uly). Add to
            y_values to get page coordinates.
        flags: Notable conditions, e.g. 'no_fit_attempted', 'fit_did_not_converge',
            'line_following_applied:deg3'.
    """

    x_start: int = 0
    x_end: int = 0
    y_values: list[float] = field(default_factory=list)
    coefficients: list[float] = field(default_factory=list)
    residual_mean: float = 0.0
    residual_max: float = 0.0
    n_pixels_used: int = 0
    n_pixels_total: int = 0
    x_page_offset: float = 0.0
    y_page_offset: float = 0.0
    flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def fit_centerline(
    filter_result: ComponentFilterResult,
    scale_unit: float,
    crop: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
) -> FitResult:
    """Fit a robust quadratic centerline to a component filter's kept pixels.

    Args:
        filter_result: Output of component_filter.filter_components.
        scale_unit: Page-level scale unit h (median staffline box height).
            Used to size the Huber transition distance.
        crop: The original (BGR-preprocessed) crop the filter ran on. Only
            required if save_path is provided; used for the diagnostic overlay.
        save_path: If provided, save a single-panel diagnostic (crop with the
            fitted centerline overlaid). If None, no visualization is produced.

    Returns:
        FitResult. If no kept pixels are available or the fit fails to
        converge, the result carries an appropriate flag and empty y_values.
    """
    # --- Handle empty input ---
    if not filter_result.coords:
        result = FitResult(flags=["no_fit_attempted"])
        if save_path is not None and crop is not None:
            _save_fit_diagnostic(crop, result, save_path)
        return result

    # --- Extract coords as numpy arrays ---
    coords_arr = np.asarray(filter_result.coords, dtype=np.float64)
    xs = coords_arr[:, 0]
    ys = coords_arr[:, 1]

    x_start = int(xs.min())
    x_end = int(xs.max())

    # --- Set up the robust fit ---
    # Initial guess: ordinary least-squares polynomial fit. Huber refines from
    # there, with the f_scale parameter controlling outlier down-weighting.
    initial_coefficients = np.polyfit(xs, ys, deg=POLY_DEGREE)

    f_scale = HUBER_F_SCALE_MULTIPLIER * scale_unit
    if f_scale <= 0:
        # Defensive: scale_unit should be positive; if not, fall back to a
        # sensible constant rather than crashing.
        f_scale = 1.0

    def residuals(coeffs: np.ndarray) -> np.ndarray:
        return np.polyval(coeffs, xs) - ys

    try:
        ls_result = least_squares(
            residuals,
            initial_coefficients,
            loss="huber",
            f_scale=f_scale,
            max_nfev=MAX_FIT_ITERATIONS,
        )
    except Exception as e:
        # Catch any numerical issue and surface it as a flag rather than crashing.
        result = FitResult(
            x_start=x_start,
            x_end=x_end,
            n_pixels_used=len(filter_result.coords),
            n_pixels_total=len(filter_result.coords),
            flags=["fit_did_not_converge", f"reason:{type(e).__name__}"],
        )
        if save_path is not None and crop is not None:
            _save_fit_diagnostic(crop, result, save_path)
        return result

    fitted_coeffs = ls_result.x
    final_residuals = np.polyval(fitted_coeffs, xs) - ys
    abs_residuals = np.abs(final_residuals)

    # --- Line-following refinement (high-residual case) ---
    # When the initial Huber fit's mean residual exceeds REFIT_TRIGGER_MULTIPLIER * h
    # the box likely contains multiple arching stafflines (e.g. page-curvature makes
    # two adjacent lines bow together). Trace a single line left-to-right from the
    # box vertical center, then refit using only those trace medians. The trace
    # itself follows any shape (arch, S-wave, etc.); the polynomial degree controls
    # how smoothly the final curve is expressed.
    line_follow_flags: list[str] = []
    trace_xs_out: np.ndarray = np.array([], dtype=np.float64)
    trace_ys_out: np.ndarray = np.array([], dtype=np.float64)

    if float(abs_residuals.mean()) > REFIT_TRIGGER_MULTIPLIER * scale_unit:
        seed_y = (
            (float(crop.shape[0]) / 2.0)
            if crop is not None
            else float((ys.min() + ys.max()) / 2.0)
        )
        trace_xs_arr, trace_ys_arr = _trace_line(
            xs, ys, x_start, x_end, scale_unit, seed_y
        )
        if len(trace_xs_arr) >= LINE_FOLLOW_MIN_TRACE_POINTS:
            try:
                # Use cubic when we have enough points; otherwise quadratic.
                # Rule of thumb: need at least 2*(degree+1) points for a stable fit.
                refit_degree = (
                    LINE_FOLLOW_POLY_DEGREE
                    if len(trace_xs_arr) >= 2 * (LINE_FOLLOW_POLY_DEGREE + 1)
                    else POLY_DEGREE
                )
                refined_coeffs = np.polyfit(
                    trace_xs_arr, trace_ys_arr, deg=refit_degree
                )
                refined_residuals = np.abs(
                    np.polyval(refined_coeffs, trace_xs_arr) - trace_ys_arr
                )
                if refined_residuals.mean() < abs_residuals.mean():
                    fitted_coeffs = refined_coeffs
                    # Report residuals over the trace points (the set we optimised for).
                    abs_residuals = refined_residuals
                    line_follow_flags.append(
                        f"line_following_applied:deg{refit_degree}"
                    )
                    trace_xs_out = trace_xs_arr
                    trace_ys_out = trace_ys_arr
                else:
                    line_follow_flags.append("line_following_no_improvement")
            except np.linalg.LinAlgError:
                line_follow_flags.append("line_following_refit_failed")
        else:
            line_follow_flags.append("line_following_insufficient_trace")

    # --- Sample at integer x ---
    sample_xs = np.arange(x_start, x_end + 1)
    sample_ys = np.polyval(fitted_coeffs, sample_xs)
    y_values_list = [round(float(y), Y_SAMPLE_DECIMALS) for y in sample_ys]

    result = FitResult(
        x_start=x_start,
        x_end=x_end,
        y_values=y_values_list,
        coefficients=[float(c) for c in fitted_coeffs],
        residual_mean=float(abs_residuals.mean()),
        residual_max=float(abs_residuals.max()),
        n_pixels_used=len(filter_result.coords),
        n_pixels_total=len(filter_result.coords),
        flags=([] if ls_result.success else ["fit_did_not_converge"])
        + line_follow_flags,
    )

    if save_path is not None and crop is not None:
        _save_fit_diagnostic(
            crop, result, save_path, trace_xs=trace_xs_out, trace_ys=trace_ys_out
        )

    return result


# ---------------------------------------------------------------------------
# Line-following helper
# ---------------------------------------------------------------------------


def _trace_line(
    xs: np.ndarray,
    ys: np.ndarray,
    x_start: int,
    x_end: int,
    scale_unit: float,
    seed_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk left-to-right collecting one median y per sliding window.

    Keeps only pixels within LINE_FOLLOW_BAND_MULTIPLIER * h of the current
    estimated y. Each window's median becomes the next window's center, so the
    trace naturally follows the staffline's curve — arch, S-wave, or straight —
    while refusing to hop to an adjacent line.

    Args:
        xs: x-coordinates of all kept pixels (from ComponentFilterResult).
        ys: Corresponding y-coordinates.
        x_start: Left edge of the horizontal extent.
        x_end: Right edge (inclusive).
        scale_unit: Page-level scale unit h.
        seed_y: Starting y estimate (box vertical center recommended).

    Returns:
        (trace_xs, trace_ys): Representative x/y pairs, one per window that
        contained at least 2 in-band pixels. Both arrays are empty if no
        windows qualified.
    """
    band_half = LINE_FOLLOW_BAND_MULTIPLIER * scale_unit
    window_width = max(5, int(scale_unit / LINE_FOLLOW_WINDOW_DIVISOR))
    trace_xs: list[float] = []
    trace_ys: list[float] = []
    current_y = seed_y

    for win_start in range(x_start, x_end + 1, window_width):
        win_end = min(win_start + window_width, x_end + 1)
        in_window = (xs >= win_start) & (xs < win_end)
        in_band = in_window & (np.abs(ys - current_y) <= band_half)
        if in_band.sum() >= 2:
            window_y = float(np.median(ys[in_band]))
            trace_xs.append((win_start + win_end) / 2.0)
            trace_ys.append(window_y)
            current_y = window_y
        # If no pixels land in the band this window, current_y holds —
        # the band stays centred where we last found the line.

    return np.array(trace_xs, dtype=np.float64), np.array(trace_ys, dtype=np.float64)


# ---------------------------------------------------------------------------
# Diagnostic visualization
# ---------------------------------------------------------------------------


def _save_fit_diagnostic(
    crop: np.ndarray,
    fit_result: FitResult,
    save_path: Path,
    trace_xs: Optional[np.ndarray] = None,
    trace_ys: Optional[np.ndarray] = None,
) -> None:
    """Save a single-panel figure: the original crop with the fitted centerline
    overlaid as a bright line, plus markers at x_start and x_end.

    If line-following was applied, cyan scatter dots show the sliding-window
    trace points used for the refit.

    If the fit produced no y_values (empty input or failure), the panel still
    renders the crop and a "no fit" annotation.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(16.0, 3.0))
    if crop.ndim == 3:
        ax.imshow(crop)
    else:
        ax.imshow(crop, cmap="gray")

    has_legend = False

    if fit_result.y_values:
        xs = np.arange(fit_result.x_start, fit_result.x_end + 1)
        ys = np.asarray(fit_result.y_values)
        # Bright magenta-ish line; high contrast against most parchment/ink.
        ax.plot(xs, ys, color=(1.0, 0.2, 0.7), linewidth=1.5, label="Fitted centerline")
        # Endpoint markers.
        ax.plot(
            [fit_result.x_start, fit_result.x_end],
            [ys[0], ys[-1]],
            marker="o",
            linestyle="none",
            color=(1.0, 0.2, 0.7),
            markersize=6,
        )
        has_legend = True

        # Line-following trace points — cyan scatter.
        if trace_xs is not None and len(trace_xs) > 0:
            ax.scatter(
                trace_xs,
                trace_ys,
                color=(0.2, 0.8, 1.0),
                s=25,
                zorder=5,
                label="Trace points (line-following)",
            )

        title = (
            f"Fitted centerline | "
            f"residual mean={fit_result.residual_mean:.2f}, "
            f"max={fit_result.residual_max:.2f} | "
            f"pixels used={fit_result.n_pixels_used}"
        )
        if fit_result.flags:
            title += f" | flags={','.join(fit_result.flags)}"
    else:
        title = "No fit produced"
        if fit_result.flags:
            title += f" ({','.join(fit_result.flags)})"

    ax.set_title(title, fontsize=8)
    ax.axis("off")
    if has_legend:
        ax.legend(loc="upper right", fontsize=7)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
