"""
Implicit neural representation staffline fitter.

Instead of extracting ink-pixel coordinates and fitting a curve to them, this
approach trains a tiny MLP f(x) → y directly on the grayscale page image at
test time.  The network is optimised via gradient descent so that its predicted
path samples dark (ink) pixels.  No training data is required — the only
supervision is the pixel intensities themselves.

Relationship to GP fitter
-------------------------
The GP fitter operates globally on a pre-filtered set of ink-pixel coordinates:
it receives detected-component (x, y) pairs and regresses a curve through them,
returning per-column posterior uncertainty as a free by-product.  The implicit
neural approach instead treats the image itself as the signal: the MLP is
differentiably connected to the grayscale tensor via bilinear grid_sample, so
gradients from pixel brightness flow directly into the network weights.  There
is no explicit uncertainty estimate; stability is controlled by a band-clamping
constraint and a smoothness regulariser.

What positional encoding buys
------------------------------
A plain MLP with a scalar x input would need many hidden units to represent a
staffline that warps by even a few pixels over the page width — the network has
to learn high-frequency spatial variation through its weights alone.  Random
Fourier / sinusoidal positional encoding maps x to a fixed bank of sin/cos
features at geometrically-spaced frequencies before the first linear layer.
This lets a very small network (hidden=32) capture sharp local deviations
because the frequency content is already explicit in the input representation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

N_FREQS = 8  # number of sinusoidal frequency bands in positional encoding
HIDDEN = 32  # MLP hidden layer width
LR = 1e-3  # Adam learning rate
N_STEPS = 150  # gradient steps per staffline
LAMBDA_SMOOTH = 0.01  # weight on finite-difference smoothness regulariser
BAND_HALF_MULTIPLIER = 1.5  # clamp band = ± multiplier × scale_unit around y_hint


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class StafflineMLP(nn.Module):
    """Tiny MLP that maps a normalised x coordinate to a predicted y (pixels).

    Architecture:
        x (scalar)  →  PositionalEncoding (2·n_freqs features)
                     →  Linear(2·n_freqs, hidden)  →  ReLU
                     →  Linear(hidden, hidden)      →  ReLU
                     →  Linear(hidden, 1)
                     →  y_pred (page-absolute pixel, float)

    The network is designed to be optimised per-line at test time — it is never
    pretrained.
    """

    def __init__(self, n_freqs: int = N_FREQS, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.n_freqs = n_freqs
        in_dim = 2 * n_freqs  # sin and cos for each frequency band

        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def _encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Sinusoidal positional encoding.

        Args:
            x_norm: shape (N,), values in [-1, 1].

        Returns:
            Tensor of shape (N, 2·n_freqs).
        """
        freqs = 2.0 ** torch.arange(
            self.n_freqs, dtype=x_norm.dtype, device=x_norm.device
        )
        # x_norm: (N,) → (N, 1); freqs: (n_freqs,) → (1, n_freqs) → broadcast (N, n_freqs)
        angles = torch.pi * x_norm.unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat(
            [torch.sin(angles), torch.cos(angles)], dim=1
        )  # (N, 2·n_freqs)

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Predict y values for a batch of x coordinates.

        Args:
            x_norm: shape (N,), values in [-1, 1].

        Returns:
            y_pred: shape (N,), page-absolute y values.
        """
        enc = self._encode(x_norm)  # (N, 2·n_freqs)
        h = F.relu(self.fc1(enc))  # (N, hidden)
        h = F.relu(self.fc2(h))  # (N, hidden)
        y = self.fc3(h).squeeze(-1)  # (N,)
        return y


# ---------------------------------------------------------------------------
# Warm-start helper
# ---------------------------------------------------------------------------


def _warm_start(mlp: StafflineMLP, y_hint: float) -> None:
    """Initialise the MLP to predict a flat horizontal line at y_hint.

    Zeroes the final layer's weight matrix and sets its bias to y_hint.
    With all weights in the last layer zero the network output is entirely
    determined by the bias regardless of the hidden activations, giving a
    horizontal line at the right height from step 0.
    """
    with torch.no_grad():
        mlp.fc3.weight.zero_()
        mlp.fc3.bias.fill_(y_hint)


# ---------------------------------------------------------------------------
# Public fitter
# ---------------------------------------------------------------------------


def implicit_neural_fit(
    gray: np.ndarray,
    y_hint: float,
    x_start: int,
    x_end: int,
    scale_unit: float,
    n_freqs: int = N_FREQS,
    hidden: int = HIDDEN,
    lr: float = LR,
    n_steps: int = N_STEPS,
    lambda_smooth: float = LAMBDA_SMOOTH,
    band_half_multiplier: float = BAND_HALF_MULTIPLIER,
) -> tuple[list[float], dict]:
    """Fit a staffline path by test-time MLP optimisation on the page image.

    The MLP f(x) → y is trained to predict y positions that land on dark
    (ink) pixels.  The data loss is the mean normalised brightness along the
    predicted path (low = dark = good).  A finite-difference smoothness
    regulariser prevents the path from oscillating.

    A band-clamping constraint keeps the predicted path within
    ± band_half_multiplier × scale_unit pixels of y_hint so the network
    cannot drift to unrelated structure elsewhere on the page.

    Args:
        gray:                Full-page grayscale image, shape (H, W), dtype uint8.
        y_hint:              YOLO box y-centre in page-absolute pixels.
        x_start:             First column to predict (page-absolute, inclusive).
        x_end:               Last column to predict (page-absolute, inclusive).
        scale_unit:          Median staffline-box height in pixels; sets band width.
        n_freqs:             Number of positional encoding frequency bands.
        hidden:              MLP hidden layer width.
        lr:                  Adam learning rate.
        n_steps:             Number of gradient steps.
        lambda_smooth:       Weight on the smoothness regulariser.
        band_half_multiplier: Band half-width = multiplier × scale_unit.

    Returns:
        (y_values, meta):
            y_values — page-absolute predicted y for each integer x in
                        [x_start, x_end], length = x_end - x_start + 1.
            meta     — dict with keys: n_steps, final_loss, final_data_loss,
                        final_smooth_loss, n_freqs, hidden, lambda_smooth.
    """
    H, W = gray.shape

    # Build the page tensor once.  Shape: (1, 1, H, W), float32 in [0, 1].
    gray_tensor = (
        torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    )  # (1, 1, H, W), no grad

    # Column indices for this staffline (page-absolute integers).
    xs_abs = torch.arange(x_start, x_end + 1, dtype=torch.float32)  # (N,)
    N = xs_abs.shape[0]

    # Normalise x to [-1, 1] over the page width for positional encoding.
    x_norm = (xs_abs / max(W - 1, 1)) * 2.0 - 1.0  # (N,)

    # Also precompute normalised x for grid_sample's grid coordinate (same values).
    grid_x = x_norm.clone()  # (N,), already in [-1, 1]

    # Band limits for clamping.
    band_half = band_half_multiplier * scale_unit
    y_lo = y_hint - band_half
    y_hi = y_hint + band_half

    # Build and warm-start the MLP.
    mlp = StafflineMLP(n_freqs=n_freqs, hidden=hidden)
    _warm_start(mlp, y_hint)

    optimiser = torch.optim.Adam(mlp.parameters(), lr=lr)

    final_loss = 0.0
    final_data_loss = 0.0
    final_smooth_loss = 0.0

    for _ in range(n_steps):
        optimiser.zero_grad()

        y_pred = mlp(x_norm)  # (N,), page-absolute y (unclamped from network)

        # Clamp to the search band so the path stays near the staffline.
        y_pred_clamped = torch.clamp(y_pred, y_lo, y_hi)

        # Normalise y to [-1, 1] for grid_sample.
        y_grid = (y_pred_clamped / max(H - 1, 1)) * 2.0 - 1.0  # (N,)

        # Build sampling grid: shape (1, N, 1, 2) = (batch, height, width, xy).
        # grid_sample with input (1,1,H,W) and grid (1,N,1,2) samples N points.
        # grid[..., 0] = x (width), grid[..., 1] = y (height).
        grid = torch.stack([grid_x, y_grid], dim=-1)  # (N, 2)
        grid = grid.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)

        sampled = F.grid_sample(
            gray_tensor,
            grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="border",
        )  # (1, 1, N, 1)
        sampled = sampled.squeeze()  # (N,)

        data_loss = sampled.mean()  # low = dark = ink = good

        # Finite-difference second derivative (shape N-2).
        smooth_loss = (
            (y_pred_clamped[:-2] - 2 * y_pred_clamped[1:-1] + y_pred_clamped[2:])
            .abs()
            .mean()
            if N > 2
            else torch.tensor(0.0)
        )

        loss = data_loss + lambda_smooth * smooth_loss
        loss.backward()
        optimiser.step()

        final_loss = float(loss.detach())
        final_data_loss = float(data_loss.detach())
        final_smooth_loss = float(smooth_loss.detach())

    # Final prediction pass — no grad, use clamped values.
    with torch.no_grad():
        y_final = mlp(x_norm)
        y_final = torch.clamp(y_final, y_lo, y_hi)

    y_values = y_final.tolist()

    meta = {
        "n_steps": n_steps,
        "final_loss": round(final_loss, 5),
        "final_data_loss": round(final_data_loss, 5),
        "final_smooth_loss": round(final_smooth_loss, 5),
        "n_freqs": n_freqs,
        "hidden": hidden,
        "lambda_smooth": lambda_smooth,
    }

    return y_values, meta
