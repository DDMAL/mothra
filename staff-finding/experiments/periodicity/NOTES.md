# Periodicity Self-Supervision — Staffline Detection

## Concept

Stafflines are periodic: they repeat at a known spatial frequency h in the
vertical axis.  This periodicity is a strong structural prior that can be
exploited as a *self-supervision signal* — no staffline labels required.

The core idea has two phases:

**Phase 1 — Detect the period.**
For each x-column, compute the autocorrelation (or power spectrum) of the
vertical intensity profile.  The dominant peak gives the local staffline
spacing h(x).  This already gives a rough h-map across the page without any
annotation.

**Phase 2 — Track the phase.**
Given h(x), find the vertical offset (phase) at each x-column that best aligns
a periodic comb of width h with the dark pixels.  Tracking how this phase
drifts left-to-right recovers the staffline curves.

No model needed for the classical version.  For a learned version, a small CNN
can predict (h, phase) per column and be trained by maximising the alignment
between the predicted comb and the observed column intensity — all without any
labels.

## Classical (no-ML) sketch

```python
for x in range(page_width):
    col = gray_page[:, x]                     # vertical intensity profile
    ac  = np.correlate(col, col, mode='full') # autocorrelation
    ac  = ac[len(ac)//2:]                     # positive lags only
    h_x = first_significant_peak(ac)          # staffline spacing at this column

    # Phase: shift a comb of period h_x to maximise dark-pixel coverage
    best_phase = argmax over φ of sum(col[φ + k*h_x] for k in range(n_lines))
    # → yields y-positions of all stafflines at column x simultaneously
```

The staffline curves are then the connected phase-consistent paths across columns.

## Key challenges

1. **Multiple staves on one page.**  The autocorrelation will show multiple
   periodicities (h and 2h and 3h...) if several staves are visible in the
   column.  Need to handle the ambiguity or restrict to per-stave strips.

2. **Damaged sections.**  Missing ink breaks the periodicity signal.  The phase
   tracker needs to bridge gaps, similar to the existing line-following logic in
   fit_centerline.py.

3. **Non-uniform staffline density.**  Pages with mixed text and music regions
   have columns where the periodicity signal is absent.  A confidence measure
   (autocorrelation peak height) can flag these columns.

4. **Variable number of stafflines.**  Unlike the fixed-cardinality approach,
   periodicity naturally handles 1, 2, 3, 4, 5+ line staves — the comb just
   has a different number of teeth.

## Learned extension

Train a lightweight CNN head on top of the autocorrelation features to predict
(h, phase, confidence) per column.  The loss is the negative alignment score
(dark-pixel coverage under the predicted comb) — no labels needed.  This
adapts the period estimate to handle damaged parchment and partial staves.

## What's needed to implement

- [ ] Classical phase 1 (autocorrelation peak detection): self-contained,
      implement first as a diagnostic to understand h-variation across pages
- [ ] Phase tracker (DP over columns, tracking phase consistency): reuse the
      DP infrastructure from dp_tracing/
- [ ] Evaluation: compare detected h values against scale_unit derived from
      YOLO boxes — built-in quantitative check without any staffline GT
- [ ] Learned version (optional later): small CNN, PyTorch, self-supervised loss

## Potential paper angle

Self-supervised period detection as a *pre-training task* or *auxiliary loss*
for the heatmap regression model.  The periodicity signal requires zero
annotation and can be computed on any manuscript page — a large unlabelled
corpus (e.g. IIIF collections) could be used for pre-training, with fine-tuning
on the small labelled set.

## Expected output format

Same ExperimentFitResult / JSOMR JSON as other experiments.

---

## Implementation status

**Implemented (2026-06-01):** `periodicity_detector.py` + `run_periodicity_page.py`.

Approach taken: per-YOLO-box comb-filter DP, mirroring `run_dp_page.py` structurally.
For each detection the runner (1) crops a vertical strip ±2·scale_unit around
the YOLO y-centre and estimates h via autocorrelation, then (2) runs DP across
the full page width using a comb cost: darkness at y augmented by darkness at
y ± h (and y ± 2h for n_teeth=5).  JSOMR `fit` block stores `h_est_px`,
`autocorr_confidence`, `n_teeth`, and `teeth_weight`.

---

## First run results — Gent right page (2026-06-01)

**Settings:** n_teeth=3, teeth_weight=0.4, band_half_multiplier=1.5,
max_step_px=3, extend_to_page=True (default).

**Output:** `staff-finding/e2e_tests/29may/Gent15_17_right/run_page/periodicity/`

**Stave grouping:** mode=2 lines/stave (target: ~8, matching sauvola/GP).

### Period estimation

Most lines returned h_est=6 px (below scale_unit=15 px) with low–moderate
confidence (0.1–0.5).  A handful returned h_est≈37–38 px (~2.5× scale_unit)
with slightly higher confidence.

**Root cause:** the autocorrelation crop spans the full page width but only
±2·scale_unit (~30 px) in height.  Averaging across the full width mixes stave
ink with text and neume columns, masking the true inter-line period.  The 6 px
peak is an artefact of the staffline's own ink density (sub-pixel autocorrelation
of the line width), not the spacing between lines.  The 37–38 px hits appear
where the crop straddles a stave boundary and inadvertently picks up the
inter-stave period instead.

### Comb filter behaviour

Because h_est ≈ 6 px for most lines, the comb teeth land within the same
staffline's ink region rather than on the neighbouring stafflines.  This
eliminates the periodicity prior entirely — the algorithm degenerates to plain
DP for those lines.  Mode=2 is consistent with plain DP behaviour on this page
(DP also groups poorly without the horizontal-emphasis fix).

### v2 run (2026-06-01) — fixes 1+2+3 applied, mode=1 (worse)

Output: `periodicity_v2/`

Applied all three proposed fixes simultaneously:
- Vertical crop widened to ±4·scale_unit (±60 px)
- X-range restricted to YOLO box [ulx, lrx]
- Confidence floor: h_est → scale_unit if autocorr_confidence < 0.3

**Result:** h_est=15px now appeared for roughly half of lines (correct), but
h_est=6px persisted for the other half with confidence ≥ 0.3, and h_est≈37px
(inter-stave period) appeared for a handful.  Mode dropped to 1 (worse than v1).

**New finding:** h_est≈37px is actively harmful.  With n_teeth=3 and h=37px,
comb teeth at y±37 reach neighbouring *staves*, pulling those DP traces off
their correct lines and corrupting the gap distribution.

### v3 run (2026-06-01) — period range gate added, mode=1 (unchanged)

Output: `periodicity_v3/`

Added range gate: h_est accepted only if 0.7·h ≤ h_est ≤ 1.5·h; otherwise
fall back to scale_unit.  Result: **all 86 lines now use h_est=15px**.

**Result:** mode=1 unchanged.  The range gate is working defensively (no more
spurious 6px or 37px periods), but that means the autocorrelation is providing
zero signal — every line falls back to scale_unit.  The comb is running at
h=15px with no genuine autocorrelation input.

**Diagnosis:** The mode=1 stave grouping is the same failure mode as plain DP
tracing: traces land on text and neumes (denser dark ink than stafflines in this
manuscript), so the y-distribution has no clear bimodal structure.  The comb
teeth at ±15px do not selectively suppress text — they reward *any* periodic
dark pattern at that spacing.  With h_est=scale_unit for all lines, the
periodicity runner is functionally equivalent to plain DP.

### Fundamental limitation identified

The autocorrelation period estimation does not work reliably per-YOLO-box on
this manuscript:
- Intra-stave crops are too narrow (15 px box height) to show periodicity.
- Widened crops mix stave ink with text/neumes, generating false peaks at 6 px
  (ink stroke width) and 37–38 px (inter-stave spacing).
- Even the correct 15 px estimate doesn't help because the DP cost is dominated
  by raw pixel darkness, and text is often darker than stafflines.

### Path forward

**Option A — Column-wise autocorrelation (full-height strips).**
Rather than cropping around YOLO boxes, compute the autocorrelation of each
*x-column* across the full page height.  The full column contains many stave
lines and gives a much stronger periodic signal.  This requires restructuring
the runner away from the per-box interface.

**Option B — Restrict to stave x-range, use YOLO stave groups.**
Run grouping first (using GP or sauvola output) to identify stave regions, then
apply column-wise phase detection within each stave strip.  Uses the existing
infrastructure as a pre-filter.

**Option C — Use the comb as a post-processing refinement** on top of GP
centerlines rather than as a standalone tracer.  Apply a local comb vote around
each GP-predicted y to snap the estimate to the nearest true staffline.

Option A is the most faithful to the original periodicity concept and is worth
implementing as a separate mode (`--full-column` flag).  Options B and C are
more pragmatic and build on existing working outputs.

**Annotation-level note:** a stave-level bounding box (one box enclosing all N
lines of a stave) would solve the vertical-extent problem naturally.  The crop
height would be ~N·h by construction, giving the autocorrelation the multi-line
view it needs without any tuning.  The phase sweep across that single box would
then recover all N line positions simultaneously.  This is effectively Option A
but with the stave box doing the horizontal isolation that the per-line box
cannot provide.  If stave-level annotations are prepared for other purposes,
this experiment should be revisited with that input format first.
