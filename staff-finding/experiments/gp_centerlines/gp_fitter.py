"""
Gaussian Process centerline fitter.

Replaces the quadratic Huber fit in fit_centerline.py with a GP regression
using a Matérn-5/2 kernel.  The GP gives a smooth, flexible curve that can
follow arbitrary parchment warp and returns per-column uncertainty (std) as a
free by-product.

Input: the filtered ink pixel coordinates from component_filter.filter_components
— the same set the Huber fit uses, so this is a drop-in replacement for
fit_centerline.fit_centerline.

Computational strategy
----------------------
sklearn's GaussianProcessRegressor is O(n³) in number of observations.  To
keep inference fast we reduce the observation set to one representative point
per x-column: the median y of all filtered pixels in that column.  For a
typical staffline crop (~900 px wide, a few hundred kept pixels) this yields
≤ 900 observations — comfortably within sklearn GP limits.

Coordinate convention
---------------------
Input coords are crop-local; output y_values are also crop-local (matching
fit_centerline.FitResult).  The caller adds y_page_offset for page-absolute
coordinates, exactly as run_page.py does for the Huber fit.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Matérn smoothness parameter.  ν=2.5 gives twice-differentiable curves —
# appropriate for stafflines which are smooth but not infinitely so.
MATERN_NU = 2.5

# Initial length scale (pixels).  Controls how rapidly the curve can change.
# 100 px ≈ a tenth of a typical page width; allows gentle warp while refusing
# to track high-frequency noise.
LENGTH_SCALE_INIT = 100.0

# Bounds for the kernel length-scale hyperparameter optimisation.
LENGTH_SCALE_BOUNDS = (5.0, 800.0)

# Initial noise level for the WhiteKernel (observation noise in pixels²).
NOISE_LEVEL_INIT = 2.0
NOISE_LEVEL_BOUNDS = (0.1, 50.0)

# Number of random restarts for kernel hyperparameter optimisation.
# More restarts → better kernel fit but slower; 3 is usually sufficient for
# a smooth 1-D regression.
N_RESTARTS = 3


# ---------------------------------------------------------------------------
# Core fitter
# ---------------------------------------------------------------------------

def gp_fit(
    coords: list[tuple[float, float]],
    x_query_start: int,
    x_query_end: int,
    length_scale_init: float = LENGTH_SCALE_INIT,
    length_scale_bounds: tuple[float, float] = LENGTH_SCALE_BOUNDS,
    noise_level_init: float = NOISE_LEVEL_INIT,
    noise_level_bounds: tuple[float, float] = NOISE_LEVEL_BOUNDS,
    n_restarts: int = N_RESTARTS,
) -> tuple[list[float], list[float], dict]:
    """Fit a GP to filtered ink pixel coordinates.

    Args:
        coords:              List of (x, y) crop-local coordinates (from
                             ComponentFilterResult.coords).
        x_query_start:       First x to predict (crop-local, inclusive).
        x_query_end:         Last x to predict (crop-local, inclusive).
        length_scale_init:   Initial Matérn length scale (pixels).
        length_scale_bounds: (min, max) bounds for length-scale optimisation.
        noise_level_init:    Initial noise variance (pixels²).
        noise_level_bounds:  (min, max) for noise optimisation.
        n_restarts:          Kernel hyperparameter optimisation restarts.

    Returns:
        (y_pred, y_std, meta):
            y_pred  — predicted y for each integer x in [x_query_start, x_query_end]
            y_std   — posterior standard deviation at each predicted x
            meta    — dict with fitted kernel params and observation count
    """
    if not coords:
        n = x_query_end - x_query_start + 1
        return [0.0] * n, [float("inf")] * n, {"error": "no_coords"}

    arr = np.asarray(coords, dtype=np.float64)
    xs_raw = arr[:, 0]
    ys_raw = arr[:, 1]

    # Aggregate to one median-y per integer x-column.
    x_int = xs_raw.astype(int)
    unique_xs = np.unique(x_int)
    obs_x = []
    obs_y = []
    for xi in unique_xs:
        mask = x_int == xi
        obs_x.append(float(xi))
        obs_y.append(float(np.median(ys_raw[mask])))

    obs_x = np.array(obs_x).reshape(-1, 1)
    obs_y = np.array(obs_y)

    kernel = (
        Matern(
            length_scale=length_scale_init,
            length_scale_bounds=length_scale_bounds,
            nu=MATERN_NU,
        )
        + WhiteKernel(
            noise_level=noise_level_init,
            noise_level_bounds=noise_level_bounds,
        )
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        normalize_y=True,
    )

    try:
        gpr.fit(obs_x, obs_y)
    except Exception as exc:
        n = x_query_end - x_query_start + 1
        return (
            list(np.full(n, float(np.mean(obs_y)))),
            list(np.full(n, float("inf"))),
            {"error": str(exc)},
        )

    x_pred = np.arange(x_query_start, x_query_end + 1, dtype=float).reshape(-1, 1)
    y_pred, y_std = gpr.predict(x_pred, return_std=True)

    # Extract fitted kernel params for the meta dict.
    fitted_kernel = gpr.kernel_
    y_arr = y_pred.astype(float)
    steps = np.abs(np.diff(y_arr))
    meta = {
        "n_obs_cols": int(len(unique_xs)),
        "n_obs_pixels": int(len(coords)),
        "fitted_length_scale": round(float(fitted_kernel.k1.length_scale), 2),
        "fitted_noise_level":  round(float(fitted_kernel.k2.noise_level),  2),
        "log_marginal_likelihood": round(float(gpr.log_marginal_likelihood(gpr.kernel_.theta)), 3),
        # Shape metrics: how much does the fitted curve move across the crop?
        # y_range_px: total vertical excursion; high values flag lines tracking warp or noise.
        # mean_step_px: average per-column y-movement; high values flag oscillating fits.
        "y_range_px": round(float(y_arr.max() - y_arr.min()), 2),
        "mean_step_px": round(float(steps.mean()) if len(steps) else 0.0, 3),
    }

    return list(y_pred.astype(float)), list(y_std.astype(float)), meta
