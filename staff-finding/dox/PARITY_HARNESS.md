# Parity harness (`scripts/parity_harness.py`)

**Purpose:** measure — not guess — how much each known landing-vs-standalone
divergence contributes to the staffline-result mismatch confirmed on McGill
MS234_64 (commit `597c3ad`). Companion to
`documentation_allons-y/ALPHA_TRANSITION_PLAN.md` §2.1 (findings SF-1…SF-6);
Phase 1 fixes must attach a before/after harness run.

## What it does

Runs the shared Stage 1/2 pipeline (component filter → centerline fit → stave
grouping) under toggles that each isolate one divergence:

| toggle | finding | landing behavior it reproduces |
|---|---|---|
| `--no-pass-crop` | SF-5 (D1) | `staffline_stage.py:218` omits `crop`, changing the line-following seed |
| `--channel-order bgr` | SF-4 (D5c) | BGR array into `component_filter`'s hardcoded `COLOR_RGB2GRAY` |
| `--image working-copy` | SF-2 (D5a) | client-side resize/re-encode (`imageResize.ts` semantics, >5 MB trigger) |
| `--image paco-layer` | SF-3 (D5b) | stafflines-only layer from paco-classifier-service (needs :8003 up) |
| `--conf 0.5` (with `--weights`) | SF-1 (D3) | landing's default stave confidence vs the validated 0.25 |

Baseline (all toggles at standalone settings) reproduces
`run_page.py --no-bgr` exactly.

## Usage

Always in the `mothra` conda env. Full attribution sweep (one flip at a time
plus a combined landing-exact run):

```bash
python staff-finding/scripts/parity_harness.py \
  --page /path/to/page.jpg \
  --yolo staff-finding/e2e_tests/10aug/ms234_64/yolo_txt/McGill_MS234-064.txt \
  --staffline-class 0 \
  --weights staff-finding/models/stave_detector_fulldata.pt \
  --output staff-finding/e2e_tests/<date>_parity/ms234_64/ \
  --sweep
```

Variants needing an unavailable input (paco service down, no `--weights`) are
skipped with a note, not fatal. Output: one JSON per variant (summary +
per-line records + `comparison_vs_baseline`) and a `report.md` diff table
(stave count, mode, distribution, cut threshold, matched-line y-MAE,
lost/new lines).

`--staffline-class` matters: `detect_stafflines.py` output uses class **0**;
landing-produced merged annotations use class **2** (see SF-9).

## Reading the report

- `Δstaves` / `Δmode` — grouping-level disagreement (the user-visible symptom).
- `yMAE` — per-line centerline drift among lines both runs found.
- `lost / new` — lines only one side produced (box-set or fit-survival changes).
- A variant matching baseline within noise ⇒ that divergence is *not* a
  contributor on this page; record the null result in the findings register
  all the same.

## Committing results

Commit sweep outputs under `staff-finding/e2e_tests/<date>_parity/<page>/`
(JSONs + `report.md`; no large PNGs — the harness deliberately writes none).
Update ALPHA_TRANSITION_PLAN.md §2.1's Status column to **measured** with the
numbers.
