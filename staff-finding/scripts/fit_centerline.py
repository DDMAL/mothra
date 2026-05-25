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


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Output of the centerline fit.

    Attributes:
        x_start: First x of the sampled centerline (integer pixel).
        x_end: Last x of the sampled centerline (integer pixel; inclusive).
        y_values: One y per integer x in [x_start, x_end]. Empty when no fit
            was produced.
        coefficients: Polynomial coefficients [a, b, c] for y = a*x^2 + b*x + c.
            Empty when no fit was produced.
        residual_mean: Mean absolute deviation of fitted curve from kept pixels.
        residual_max: Max absolute deviation.
        n_pixels_used: Number of kept-coord pixels fed to the fit.
        flags: Notable conditions, e.g. 'no_fit_attempted', 'fit_did_not_converge'.
    """
    x_start: int = 0
    x_end: int = 0
    y_values: list[float] = field(default_factory=list)
    coefficients: list[float] = field(default_factory=list)
    residual_mean: float = 0.0
    residual_max: float = 0.0
    n_pixels_used: int = 0
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
            flags=["fit_did_not_converge", f"reason:{type(e).__name__}"],
        )
        if save_path is not None and crop is not None:
            _save_fit_diagnostic(crop, result, save_path)
        return result

    fitted_coeffs = ls_result.x
    final_residuals = np.polyval(fitted_coeffs, xs) - ys
    abs_residuals = np.abs(final_residuals)

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
        flags=[] if ls_result.success else ["fit_did_not_converge"],
    )

    if save_path is not None and crop is not None:
        _save_fit_diagnostic(crop, result, save_path)

    return result


# ---------------------------------------------------------------------------
# Diagnostic visualization
# ---------------------------------------------------------------------------

def _save_fit_diagnostic(
    crop: np.ndarray,
    fit_result: FitResult,
    save_path: Path,
) -> None:
    """Save a single-panel figure: the original crop with the fitted centerline
    overlaid as a bright line, plus markers at x_start and x_end.

    If the fit produced no y_values (empty input or failure), the panel still
    renders the crop and a "no fit" annotation.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(16.0, 3.0))
    if crop.ndim == 3:
        ax.imshow(crop)
    else:
        ax.imshow(crop, cmap="gray")

    if fit_result.y_values:
        xs = np.arange(fit_result.x_start, fit_result.x_end + 1)
        ys = np.asarray(fit_result.y_values)
        # Bright magenta-ish line; high contrast against most parchment/ink.
        ax.plot(xs, ys, color=(1.0, 0.2, 0.7), linewidth=1.5,
                label="Fitted centerline")
        # Endpoint markers.
        ax.plot([fit_result.x_start, fit_result.x_end],
                [ys[0], ys[-1]],
                marker="o", linestyle="none", color=(1.0, 0.2, 0.7),
                markersize=6)
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

    ax.set_title(title)
    ax.axis("off")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)