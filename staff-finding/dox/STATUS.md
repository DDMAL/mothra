# Staff-Finding — Status Note (June 2026)

Hi! This note is for anyone picking up this work while I'm away.  It covers
what's done, what's in progress, what to run, and where the sharp edges are.

---

## Where we are

The two-stage pipeline is **complete and passing tests**:

**Stage 1 — per-box** (Component filter → Centerline fit)
- `scripts/component_filter.py` — binarizes each YOLO crop, runs connected
  components, scores by horizontal extent and vertical proximity, merges
  fragments.  Sauvola binarization is now the default (ADR-002).
- `scripts/fit_centerline.py` — fits a Huber-robust quadratic (or cubic) to
  the surviving pixels; outputs y-values sampled at every integer x-column.

**Stage 2 — per-page** (Stave grouping → Interpolation)
- `scripts/group_staves.py` — fully implemented, all three former stubs
  resolved.  See `IMPLEMENTATION_AUDIT.md` for what changed.
- `scripts/interpolate_staves.py` — the **placeholder is in place but
  interpolation is not yet called** by default.  This is the main thing left
  to implement (see below).

**Evaluation framework** — complete
- `scripts/eval_page.py` — single page, GT vs. predicted, outputs one CSV row
  (precision, recall, F1, split ratio, y-MAE).
- `scripts/eval_batch.py` — reads a manifest CSV, aggregates metrics per
  variant.
- All five experiment runners emit the same JSOMR JSON, so they can be
  compared head-to-head via the eval scripts.

**Experiment runners** — all five implemented
- `dp_tracing/` — DP minimum-cost path through raw grayscale (no binarization)
- `gp_centerlines/` — reuses component filter; replaces poly fit with a GP
- `implicit_neural/` — per-page per-line MLP trained by gradient descent on
  the image; **best result to date** (mode=8 lines/stave on Gent right page)
- `periodicity/` — autocorrelation → comb-filter DP
- `heatmap_regression/` — **design doc only**, not yet implemented

Results are in `staff-finding/e2e_tests/29may/Gent15_17_right/run_page`
and `staff-finding/e2e_tests/29may/Gent15_17_left/run_page`. 

---

## What to work on next

### 1. Interpolation pass (highest priority)

After grouping, use the known stave structure (N lines per stave, expected
spacing ≈ `scale_unit`) to detect and fill missing lines.  The stub lives in
`scripts/interpolate_staves.py`; the design is in `dox/` §6.3.  This fix is
method-agnostic — it will improve every experiment runner.

Trigger: a stave with fewer lines than the modal count, where a missing line
can be placed at a predicted position from its neighbours.

This would also fix the "stave 5/6 split" seen in the implicit neural run
(see `experiments/implicit_neural/NOTES.md`) — the apparent two-group
structure is one stave with a dropped line at the boundary.

As a general note, I have been proceding with gp_centerlines' approach
as my "to the end" run, given its performance over my initial baseline
and the other results shown in `dp_tracing` and `periodicity`.

### 2. Batch evaluation across methods and pages

All five runners produce JSOMR JSON.  We have 84 annotated pages in
`addtl-gt/` and the two Gent crops.  Running `eval_batch.py` over a
multi-page manifest would give us the first real method-level comparison.
A manifest CSV just needs columns: `page_name, image, gt_txt, pred_json,
variant, gt_source`.

### 3. Heatmap regression

The only experiment runner not yet implemented.  `heatmap_regression/NOTES.md`
has the design.  Lower priority than (1) and (2) above.

Given the stupendous performance of gp_centerlines, putting this on 
permanent hold until we cycle back from proof of concept phase is fine. 

---

## Running things

```bash
# Single page (main pipeline)
python scripts/run_page.py \
    --image image-sets/gent/right/GentAnt1475_0017_AC_rightcrop.jpg \
    --yolo  image-sets/gent/right/inference/corrected/...txt \
    --output /tmp/out/

# Experiment runners (same flags, same output format)
python experiments/implicit_neural/run_implicit_neural_page.py --image ... --yolo ...
python experiments/dp_tracing/run_dp_page.py --image ... --yolo ...
python experiments/gp_centerlines/run_gp_page.py --image ... --yolo ...
python experiments/periodicity/run_periodicity_page.py --image ... --yolo ...

# Evaluate one page
python scripts/eval_page.py --image ... --gt-txt ... --pred-json ...

# Evaluate a batch
python scripts/eval_batch.py --manifest manifest.csv --output results.csv
```

---

## Known issues / things to watch out for

| Issue | Where | Notes |
|-------|-------|-------|
| Implicit neural drops lines near decorative initials / rubrics | `experiments/implicit_neural/NOTES.md` | Root cause: competing dark mass; fix = interpolation pass (see §1 above) |
| GP centerlines need `--valley-threshold` flag | `experiments/gp_centerlines/run_gp_page.py` | Intra-stave gap clusters near scale_unit; median-ratio threshold fails |
| Periodicity autocorrelation per-YOLO-box can lock on wrong frequency | `experiments/periodicity/NOTES.md` | Range gate [0.7h, 1.5h] mitigates; per-page crop is more reliable |
| Interpolation stub is a no-op | `scripts/interpolate_staves.py` | Not yet wired up; `group_staves.py` flags missing lines but does not fill them |
| Binarization on low-contrast faint ink | ADR-002 | Sauvola is in place; DeepOtsu is the planned escalation path if needed |

---

## Test coverage

```bash
# Unit tests (group_staves)
python -m pytest scripts/test_group_staves.py

# Integration tests
python -m pytest scripts/script_tests/
```

All tests were passing as of the last commit.  If you add or change
grouping/interpolation logic, please add a corresponding test case.

---

## Key files at a glance

| File | What it is |
|------|-----------|
| `scripts/run_page.py` | End-to-end driver (Stage 1 + 2) |
| `scripts/group_staves.py` | Stave grouping (Stage 2, fully implemented) |
| `scripts/interpolate_staves.py` | Interpolation stub — the main TODO |
| `scripts/eval_page.py` | Evaluation (single page) |
| `scripts/eval_batch.py` | Evaluation (batch) |
| `experiments/*/NOTES.md` | Per-method run results and known issues |
| `dox/` | Architecture Decision Records |
| `addtl-gt/` | 84 annotated pages for evaluation |
| `image-sets/gent/right/` | Primary test image + corrected GT |

Questions? The code is pretty well commented at decision points; `dox/` has the
reasoning behind the big design choices.  Good luck!
