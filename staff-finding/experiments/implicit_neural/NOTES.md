# Implicit Neural Representation — Staffline Detection

## Concept

Instead of fitting a curve to observed pixels, we fit a tiny neural network
that *is* the curve.  For each staffline on a given page, we optimise a small
MLP f(x) → y that maps x-column to predicted y-position.  The network is
initialised from the YOLO y-hint and refined by gradient descent directly on
the page image: the loss rewards the predicted path for landing on dark pixels
and penalises it for landing on bright ones.

Each page gets its own freshly-initialised network.  There is no training
phase and no generalisation across pages — it is pure per-page test-time
optimisation.  The benefit is that the learned curve can be arbitrarily smooth
(controlled by the network depth/width and the frequency encoding) and requires
no binarization or component analysis.

## Architecture sketch

```
Input:  x  (normalised to [-1, 1] over the page width)
       → PositionalEncoding(x, n_freqs)     # sinusoidal, à la NeRF
       → Linear(2*n_freqs, hidden)
       → ReLU
       → Linear(hidden, hidden)
       → ReLU
       → Linear(hidden, 1)
Output: y  (predicted page-absolute y-coordinate)
```

Small network: hidden=32, n_freqs=8 is sufficient — stafflines are smooth, not
high-frequency.  Warm-started by zeroing the final layer's weight matrix and
setting its bias to `y_hint`, so step 0 predicts a flat horizontal line at the
correct height rather than a random curve.

## Loss function

For each predicted (x, y) along the staffline:

```
L = mean(gray_bilinear(x, y) / 255)   # low = dark = ink = good
  + λ_smooth * mean(|d²y/dx²|)        # finite-difference second derivative
```

The image is differentiable via bilinear `grid_sample`, so gradients flow
back through the pixel lookup into the network weights.  A band clamp
(`torch.clamp(y_pred, y_hint ± band_half)`) prevents drifting to unrelated
dark structure elsewhere on the page.

## Default hyperparameters

| Parameter              | Default | Effect                                    |
|------------------------|---------|-------------------------------------------|
| `N_FREQS`              | 8       | sin/cos frequency bands; lower = smoother |
| `HIDDEN`               | 32      | Model capacity                            |
| `LR`                   | 1e-3    | Adam learning rate                        |
| `N_STEPS`              | 150     | Gradient steps per staffline              |
| `LAMBDA_SMOOTH`        | 0.01    | Smoothness regulariser weight             |
| `BAND_HALF_MULTIPLIER` | 1.5     | Search band = ± 1.5 × scale_unit          |

## Implementation notes

- `extend_to_page=False` by default: traces only within the YOLO box x-range.
  When given a long span with no ink the MLP can overfit to image texture, so
  per-box is safer than extending to full page width.  `--extend-to-page` flag
  available for comparison.

- Grid sample shape convention: `grid` has shape `(1, N, 1, 2)` where the last
  dimension is `(x_norm, y_norm)` — x first, y second, consistent with
  `align_corners=True` semantics.

## Expected output format

Same ExperimentFitResult / JSOMR JSON as other experiments.

---

## First run results — Gent right page (2026-06-01)

**Settings:** all defaults (N_STEPS=150, HIDDEN=32, N_FREQS=8, LAMBDA_SMOOTH=0.01,
BAND_HALF_MULTIPLIER=1.5, extend_to_page=False).

**Output:** `staff-finding/e2e_tests/29may/Gent15_17_right/run_page/implicit_neural/`

**Stave grouping:** mode=8 lines/stave — matches GP (target: 8).  **Best result
of all methods run to date.**

### Overall quality

The MLP paths track staffline curvature closely, are smooth, and stay on ink
where ink is present.  The per-box x-range default prevents wandering in blank
regions.  Losses were consistently low (final_data_loss ≈ 0.20–0.35 across 86
lines), indicating the network found dark paths on almost every line.

### Issues identified

**Dropped / partial lines — all before decorative initials or rubrics:**

| Stave | Issue |
|-------|-------|
| 0     | 2 lines missing |
| 7     | partial missing line |
| 10    | 2 lines missing in leftmost section (before initial) |
| 14    | lines missing before rubric / initial |

**Stave split:**
- Output staves 5 and 6 are a single physical stave that was split by the grouper.

### Root cause

Both failure modes share the same cause: **locally high noise-to-signal ratio
near decorative initials and rubrics**.  The stafflines in these regions are the
same red ink as elsewhere on the page (where detection is correct) — the ink
itself is not the problem.  What differs is the density of *competing* dark
elements: heavy Gothic text, black square noteheads, and large decorative
letterforms cluster tightly around the stave entry point.  The brightness-
minimisation loss treats all dark pixels equally, so the MLP path is pulled
toward the densest dark mass rather than staying on the structurally correct
(but locally lower-contrast) staffline.

This is not specific to the implicit neural approach: DP and GP also struggle
at these sections for the same reason.  The staffline signal is present but is
locally dominated by non-staffline ink.

The stave 5/6 split is also grouper-side: one apparent gap between the two
groups fell just above the inter-stave threshold, possibly because a missing line
shifted the gap distribution.

### Path forward

**Interpolation pass** (design doc §6.3, currently stubbed in `group_staves.py`):
After grouping, use the known stave structure (N lines per stave, expected spacing
≈ scale_unit) to detect and fill in missing lines by interpolation from their
neighbours.  This would:

1. Reconstruct dropped lines at expected positions within a stave.
2. Correct the stave 5/6 split by recognising that the apparent two-group
   structure is one stave with a missing line at the boundary.

This fix is method-agnostic — it applies equally to GP, DP, periodicity, and
implicit neural outputs — so it is earmarked for implementation after all five
experiment runners are complete.

### v2 run — green channel (2026-06-01)

**Settings:** `--channel green`, all other defaults unchanged.

**Output:** `staff-finding/e2e_tests/29may/Gent15_17_right/run_page/implicit_neural_v2_green/`

**Stave grouping:** mode=8 (same as v1).

**Result:** marginally worse than v1.  Predicted lines are a few pixels higher
(further from the true staffline) across the page compared to the standard
grayscale run.  The green channel makes the cream/yellow parchment appear
brighter overall, raising the baseline loss (~0.39–0.78 vs ~0.20–0.35 in v1),
which slightly shifts where the optimum lands.  The colored-ink hypothesis was
not confirmed — the problem regions did not improve.

**Conclusion:** standard luminance grayscale (v1) remains the better input
representation for this manuscript.  Channel separation is not the lever for
the initial/rubric failure mode.

### Comparison to other methods

| Method           | Mode | Notes                                                 |
|------------------|------|-------------------------------------------------------|
| GP centerlines   | 8    | Good curves; O(n³) per line; requires coord extraction |
| Periodicity comb | 1–2  | Autocorrelation fails per-YOLO-box on this MS         |
| Implicit neural  | 8    | Best overall; issues only at ink-absent regions       |

---

## Potential improvements

- **Text/initial masking:** the failure regions are specifically where large
  decorative initials and dense text blocks crowd the stave entry.  A pre-pass
  that masks non-staffline ink (e.g. using a text detector or the YOLO non-
  staffline classes) before running the MLP would suppress the competing
  gradient signal without needing color information.

- **Extend-to-page with gap masking:** run `--extend-to-page` but zero out the
  data loss in columns with no dark pixels, so the smoothness regulariser holds
  the path flat across blank stretches instead of gradient noise pulling it away.

- **Increase N_STEPS near stave edges:** adaptive step count based on detected ink
  density — more steps where ink is present, fewer (or frozen) where absent.

- **Multi-resolution frequency bands:** add lower frequencies (k < 0) to give the
  network a gentler prior on global curvature.

- **Stave-level bbox input:** if stave-level annotations are available, running one
  MLP per stave (5 outputs) rather than per individual line would allow the network
  to exploit inter-line consistency — the periodicity idea applied to the neural
  approach.
