# ADR-002: Binarization Method — Known Issue and Proposed Escalation

**Status:** ACCEPTED (see _Outcome_)
**Date:** 2026-05-25
**Scope:** Stage 1 binarization step, prior to connected-components analysis.
**Related:** ADR-001 §3 (initial binarization decision); Staff Detection Pipeline Design Document.

---

## Context

ADR-001 §3 committed to Otsu thresholding on the grayscale-converted BGR output, on the basis that BGR's RGB-on-white output gives a near-bimodal histogram and Otsu handles that strongly. Initial proof-of-concept runs confirm Otsu works well on most boxes. However, eyeballing across several pages has surfaced a recurring failure pattern on populated lines with originally-faint staffline ink: Otsu's global threshold misses the line entirely in regions where local contrast is low, even though the line ink is visibly present in the original image.

The diagnostic signature is distinctive. In the binarized panel, the staffline disappears across long stretches; what survives is mostly neume ink. The component filter then has no line ink to work with in those stretches, and even the merge step cannot recover the line because there is no ink-based evidence to merge.

This is a binarization-layer problem, not a component-filter or merge-step problem. Faint-but-present ink is a known weakness of global thresholding methods, and the standard solution in historical-document binarization is local adaptive thresholding.

## Decisions

### 1. Recognize the issue

The faint-line failure mode is acknowledged as a known limitation of the current Otsu binarization, not a flaw in downstream stages. Component filtering and merging cannot be expected to recover information that binarization discards.

### 2. Defer the fix until the fit step is in place

The fit step (downstream of the component filter) uses robust regression to recover a centerline from whatever line evidence exists. A robust fit may handle sparse-but-correctly-placed line pixels well enough that binarization improvements show smaller wins than expected. Conversely, if the fit struggles on faint-line cases, the case for switching binarization strengthens.

The evaluation criterion for any binarization change should therefore be "does the fitted centerline improve?" rather than "do the components look better in isolation." Switching binarization without the fit in place would be evaluating against the wrong metric.

### 3. Sauvola as the first-line escalation

When the fit is in place and binarization improvements are evaluated, Sauvola thresholding is the proposed first move. Reasons:

- Designed specifically for historical-document binarization. Established and well-understood.
- Local adaptive: each pixel's threshold is computed from a local window's mean and standard deviation, handling uneven contrast cleanly.
- No training data or model dependency. Available in `skimage.filters.threshold_sauvola`.
- Drop-in replacement for the current Otsu line: one or two extra lines of code, plus two tunable parameters (window size and `k` constant).
- Recommended starting parameters: window size scaled to staffline thickness `h` (e.g., `2 * h + 1`, kept odd), `k = 0.2` per typical defaults. Both tunable empirically.

### 4. DeepOtsu as the considered second escalation

If Sauvola also fails on the messiest pages — e.g., when local contrast is so low that even adaptive thresholding misses the line — DeepOtsu (a learned, iterative deep-network binarizer for degraded documents) is the proposed next move. Reasons it is deferred to second:

- Model dependency, with weights to source or train.
- Non-trivial inference cost per page.
- More moving parts to maintain than a classical method.
- Sauvola is expected to handle the common case, leaving DeepOtsu for the genuinely hard residue.

### 5. Evaluation requires a controlled comparison

Whichever binarization is being evaluated, the comparison must follow the same pattern as the BGR-on-vs-off and merge-on-vs-off experiments: add a flag, output to a separate directory, eyeball side by side on the same pages. Switching binarization globally without this comparison risks regressing currently-working cases (Sauvola can occasionally over-threshold, picking up parchment noise that Otsu correctly ignored).

## Rationale Summary

Otsu is sufficient for clean, high-contrast inputs and is the right starting point for a proof of concept. Faint stafflines on populated lines surface its limits, but the right time to address those limits is after the fit step exists — both because the fit may absorb some of the failures, and because the fit gives us the right evaluation metric for binarization changes.

## Consequences

**Accepted in the short term:**

- Pages with faint stafflines will show degraded component-filter output, surfacing as sparse kept masks even with merging applied. These cases are visibly identifiable in diagnostic panels.
- Downstream fit and grouping stages will operate on impoverished input for these cases. Whether this matters depends on the fit's robustness, which is the test ADR-002 §2 defers.

**Cost when implemented:**

- Sauvola adds two tunable parameters to the pipeline. Both expressible in scale-relative units, both tunable empirically.
- Switching binarization is a global change affecting every box on every page. A controlled comparison is required (ADR-002 §5) to confirm net improvement before adoption.

## Escalation Triggers

The following observations would prompt moving from "deferred" to "implemented" on this ADR's decisions:

- **Trigger for Sauvola adoption:** Fit step in place; fitted centerlines on faint-line cases show systematic drift or fail to recover where line ink is visibly present in the original. Confirmed by side-by-side Otsu-vs-Sauvola comparison showing Sauvola improves fits without regressing clean cases.
- **Trigger for DeepOtsu consideration:** Sauvola in place and tuned; faint-line failures persist on a non-trivial fraction of pages.

## Deferred Decisions

- Exact Sauvola parameters (window size multiplier, `k` value). Empirical tuning during the comparison experiment.
- Per-manuscript binarization profiles (some manuscript classes may need different parameters). Not for proof of concept; revisit if a single parameter set proves insufficient across the corpus.
- Whether binarization should be parameterized per-box rather than per-page. Likely overkill; deferred unless evidence demands it.

## Notes

- This ADR documents a known issue and a deferred fix; it is not a commitment to switch binarization. The trigger conditions in §Escalation must be observed before action.
- The faint-line failure mode is partially upstream — BGR retraining could also reduce it, by preserving thin-horizontal-structure ink more aggressively. That is a separate concern owned by the BGR layer, tracked elsewhere.
- The decision to defer is explicitly time-bounded: it holds until the fit step is in place and evaluated. After that, this ADR should be revisited.
# Outcome

## Update — Sauvola promoted to default

**Date:** 2026-05-25
### 1. Status change
Status moves from **Proposed** to **Accepted**. Sauvola is now the default binarization for Stage 1; Otsu is retained as an opt-in fallback for comparison.

### 2. Trigger condition met

Per §Escalation, adoption required: "Fit step in place; fitted centerlines on faint-line cases show systematic drift or fail to recover where line ink is visibly present in the original. Confirmed by side-by-side Otsu-vs-Sauvola comparison showing Sauvola improves fits without regressing clean cases."

Both clauses were observed:

- With the fit step in place, faint-line failure cases produced fits that covered only the small portions of the line that survived Otsu's threshold, with the rest of the line unrecovered despite being visibly present in the original crops.
- Side-by-side comparison on the same pages (Otsu in `<page>/`, Sauvola in `<page>_sauvola/`) showed Sauvola recovering substantially more line ink on faint cases. A brief tuning pass (`k = 0.15` rather than the literature default of 0.2) further improved recovery without introducing noticeable parchment-noise regressions on clean cases.

### 3. Final parameters

Settled during the comparison experiment, all tunable constants at the top of `component_filter.py`:

- `SAUVOLA_WINDOW_MULTIPLIER = 8 (window size = 15 × h, rounded to nearest odd integer)
- `SAUVOLA_K = 0.1` (lowered from the literature 0.2 to recover more faint ink)
- `SAUVOLA_MIN_WINDOW = 5` (floor, in pixels, for very small scale units)
These are severe, and may need lightening for future general use. 
### 4. Interface changes

- `filter_components(binarization=...)` parameter default flipped from `"otsu"` to `"sauvola"`.
- `_binarize` and `_save_diagnostic` defaults likewise.
- `run_page.py` CLI flag inverted: `--sauvola` (opt-in to Sauvola) became `--otsu` (opt-in to Otsu).
- Output directory naming convention inverted: defaults now produce an untagged `<page>/` directory; Otsu runs produce `<page>_otsu/`.

### 5. Open follow-ups

- DeepOtsu remains a considered second escalation (§4), unchanged. Trigger condition: Sauvola in place and tuned; faint-line failures persist on a non-trivial fraction of pages.
- The `k = 0.15` tuning was empirical on a small set of pages. If broader corpus testing surfaces manuscripts where this value misbehaves, per-manuscript binarization profiles (§Deferred Decisions) become live.

### 6. Supersedes

This update supersedes ADR-001 §3 (Otsu as the initial binarization decision). ADR-001 §3 is retained in its document for historical context but is no longer current; readers should follow §3 here for the active state.