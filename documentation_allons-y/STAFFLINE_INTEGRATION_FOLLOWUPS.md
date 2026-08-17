# Staffline integration — follow-ups

Written after wiring staff-finding's staffline detection into the
landing-page predict/encode pipeline (see `CLAUDE.md`'s **Staffline
detection** section for how it works today). Nothing below is started;
this is a punch list, not a plan.

## Deferred by design (not started)

- **Ink-separation ("BGR")** — `staff-finding/scripts/bgr_adapter.py` wraps
  an external, unvendored `muscrat/layer_sep` repo reachable only via
  hardcoded paths on specific dev machines. No tests, no Docker/LFS story,
  and its checkpoint (`best_model.pth`) doesn't match the repo's `*.pt`
  Git-LFS rule. Staffline detection runs on raw page crops instead; Sauvola
  binarization is the interim mitigation. Real next step: give
  `muscrat/layer_sep` a stable, team-owned location (vendor it or add it as
  a submodule) and at least one test, before wiring it into
  `staffline_stage.py`.
- **`interpolate_staves.py`** — implemented and unit-tested, but off by
  default (`interpolate_missing=False`) per `staff-finding/dox/STATUS.md`'s
  "not yet validated across the corpus" caveat. It's a plumbed-through
  parameter already, so turning it on is a default flip, not new code —
  but do the corpus validation first.
- **`fallback_redetect.py`** — not wired into `staffline_stage.py`. Needs an
  already-loaded stave-detector YOLO model instance; the medieval preset has
  one resident, but custom uploaded models don't have a dedicated one to
  re-probe with.
- **Frontend UI** — staffline QA data (rhythm anomalies, low-confidence
  fits, reconciliation flags) only surfaces in `job_events` log lines today.
  No frontend references staffline/stave concepts at all yet. Natural home
  for a future overlay: `AnnotationsTab.tsx`.
- ~~**`batch_api.py`'s text-batch-run path**~~ — **stale, superseded.** This
  bullet said the text-batch path never called `staffline_stage.py`; that
  changed in `916f465` (see "Diagnosed this session" → Root cause 1 below),
  and confirmed again during the 2026-08-17 Alpha-1 fix batch:
  `tasks_text_batch.py` does call `run_staffline_detection()` today. Left in
  the file (struck through) rather than silently deleted, since it was cited
  as a real gap once. What IS still true, and newly documented as part of
  that same batch's SF-2 work: this path builds its own `img_arr` from
  `project_images.data` only, never `original_data` — see SF-2's "Collision
  noted, explicitly NOT fixed here" note for why that wasn't backported.

## Found along the way, not fixed

- **`encode_to_mei.py`'s `extract_ms_id` import** already silently falls
  back to `Path(filename).stem` in the deployed Docker image (the file it
  wants lives at the repo root, outside `landing-page/`'s build context —
  same class of gap as the one just fixed for `staff-finding/`, just never
  addressed for this one). Needs the same treatment: either add it as
  another named build context, or move/vendor it.
- **Two `.pyc` files are tracked in git**:
  `staff-finding/scripts/__pycache__/{bgr_adapter,yolo_io}.cpython-312.pyc`.
  They show as "modified" on essentially every local session that imports
  those modules. Worth a `git rm --cached` + a `__pycache__/` gitignore rule
  scoped to `staff-finding/` (the existing repo-wide `.gitignore` doesn't
  seem to cover this path).
- **`test_run_page.py`/`test_run_pageOG.py`'s fake-module stubbing has no
  teardown** — they replace `sys.modules["torch"]`/`["ultralytics"]`/
  `["inference_simple"]` at module level and never restore them. CI now
  works around this by running them as isolated `pytest` steps (see
  `.github/workflows/tests.yml`'s comments), but the underlying fragility
  in those two files is still there for anyone running them ad hoc alongside
  other test files. Proper fix: move the stubbing into a fixture with
  teardown (`monkeypatch.setitem`, scoped to the test module).
  **Update**: `test_run_page.py` specifically also stubbed the wrong
  module — `sys.modules["inference_simple"]` never took effect because
  `bgr_adapter.py` raises its own `ModuleNotFoundError` before ever reaching
  that import (an unconditional `os.path.isfile()` check across hardcoded
  developer paths). This only surfaced in real CI, not local runs, since
  the local dev machine happened to have the real dependency at one of
  those paths. Fixed by also stubbing `sys.modules["bgr_adapter"]` itself —
  the no-teardown concern above still applies to both files, unfixed.
- **`auth_api.py` had a duplicate `ALTER TABLE project_images ADD COLUMN
  folio TEXT` migration** (around two different line numbers in
  `_migrate_db()`), most likely from the same large `main` merge that also
  broke `init_db()` (see below). Harmless — `DuplicateColumn` makes the
  second one a no-op — but worth deleting the redundant block.
- **`staffline_detections` has no retention/archival plan** (CodeRabbit,
  PR #53) — it accumulates forever by design (see the schema comment), which
  is deliberate for the interpolate_staves before/after comparison use case,
  but storage still grows unbounded per re-run. Same category of gap as
  `job_uploads`/`job_sessions` (CLAUDE.md's own "things that don't exist
  yet" list) — plan a cleanup job alongside those, not in isolation.
  **Explicit deferral** (CodeRabbit followed up asking for either a bound or
  an explicit deferral statement): treat this as blocking, not a nice-to-have
  — do not point staffline detection at a production database until a
  retention bound (time- or count-based, scoped so it doesn't discard rows
  `interpolate_staves`' before/after comparison still needs) ships alongside
  the `job_uploads`/`job_sessions` cleanup job.
- **Three checked-in e2e `stave_grouping_report.txt` baselines show genuinely
  fragmented grouping — confirmed still-reproducible on current code, not
  stale fixtures** (CodeRabbit, PR #53: Gent right,
  `F-Pn-Latin-15181_107r`, `F-Pn-Latin-15181_221`). Re-ran all three through
  the current pipeline rather than assuming the checked-in numbers just
  predated a later fix:
  - `F-Pn-Latin-15181_107r`/`_221` (`test_model.sh --fallback-redetect`)
    reproduce their checked-in `reconciled_duplicate_fits` /
    `staves_with_unexpected_count` numbers almost exactly (gap values match
    to ~0.1px), even after the fallback-redetect existing-line-distance fix
    (`ecaa401`) — that fix had nothing to reject on either page
    (`fallback_redetect_report.txt`: 0 candidates / "no under-populated
    staves found"), so it isn't the relevant bottleneck here. These two are
    an accurate record of a real, current `group_staves.py`
    duplicate/companion-reconciliation limitation on dense, tightly-spaced
    staves (35-45% of fits end up marked duplicate/companion; mode line
    count collapses to 1 or 4 instead of this manuscript's real stave size).
  - Gent right didn't reproduce cleanly enough to even compare: two
    plausible source YOLO `.txt` files exist
    (`image-sets/gent/right/inference/corrected/...` vs
    `e2e_tests/29may/Gent15_17_right/Gent15_17_right_corrected.txt` — they
    differ in content and line count), and re-running `run_page.py` against
    either one produces box_index 0 at different coordinates than the
    checked-in fixture, so neither is confirmed as the original input.
    Both attempts still show heavy fragmentation (mode 5 with 8/17 staves
    unexpected, or mode 1 with 14/20 unexpected) — same symptom as the
    other two pages, just via an input this session couldn't pin down
    precisely.
  - Not attempting a fix here: this is a real algorithmic gap in
    `group_staves.py`'s reconciliation logic on dense staves, not a
    one-line bug, and forcing a low-confidence change (or regenerating the
    fixtures as-is) to make these three pages look clean risks either
    overfitting to them or just checking in a different flavor of the same
    fragmentation. CodeRabbit's ask — regenerate, then assert acceptable
    per-stave counts — is the right ask; it's blocked on the reconciliation
    fix landing first.
- **`component_filter.py`'s `discarded`/`score_breakdown[...]["kept"]`
  bookkeeping doesn't account for companion retention** (CodeRabbit, PR #53,
  `component_filter.py:280-299`, Minor) — companions folded into the active
  output via the retention pass (lines 337-401) still show up in `discarded`
  (reason `"not_top_scoring"`) and never get `"kept": True`, since the two
  passes don't talk to each other. Affects diagnostic/summary-CSV accuracy
  (`run_page.py`'s `n_discarded` column), not the actual detection/fit
  output. Deferred rather than fixed in the same pass as the other findings
  since the correct fix needs to reconcile against whichever companion-label
  set matches the caller's actual `merge_components` mode, and this file's
  companion-selection logic (811 lines total) deserved more careful reading
  than the time available for a Minor-severity bookkeeping issue.

## Resolved since this was written

- **Rebase-onto-main evaluated, decided against**: checked what rebasing
  `kyrie/staff-finding` onto `main` would take. `origin/main` hasn't moved
  since the 2026-07-30 merge (`15061d1`) — this branch already fully
  contains it (`git merge-base --is-ancestor origin/main kyrie/staff-finding`
  is true) — so a rebase would only replay this branch's own 54 commits
  against the merge commit's absence, re-hitting the same `auth_api.py`/
  `CLAUDE.md` conflicts already resolved once, and force-pushing over PR
  #53's live review thread for no content gain. Decided: no rebase, no
  force-push; `main` gets a clean single commit via squash-merge when PR #53
  lands instead (repo already has `allow_squash_merge: true`).
- **`CLAUDE.md`'s Deployment section repaired**: the 2026-07-30 merge kept
  only this branch's own `### Deployment (Docker)` section and silently
  dropped both of `main`'s — `### Deployment (Kubernetes, CI/CD via GitHub
  Actions)` (the real production path — auto-deploy to a k8s cluster on
  push to `main`) and `### Local/manual container runs`, which explicitly
  scopes `docker-compose.yml` as local-only ("the k8s manifests were
  modeled on it, not the other way around"). This branch's text had claimed
  Compose "is now the only deployment path," which was wrong — the actual
  `k8s/` manifests and CI/CD deploy job were untouched throughout (confirmed
  byte-identical to `main`), only the docs were stale. Restored both
  sections, folding this branch's genuinely new local-testing content
  (staff-finding's Docker build-context wiring, the buildx/OOM/Tridis/
  redeploy-together notes) into the restored `Local/manual` section. Also
  fixed a second, smaller staleness bug found in the same paragraph: it
  still said `pip install -e` for staff-finding's install, which an earlier
  commit this same session had already changed to non-editable.
  Also refreshed three `## Key files` table rows (`job_store.py`,
  `jobs_api.py`, `tasks_predict.py`/`tasks_encode.py`) that had fallen
  behind `main`'s versions — missing mentions of job cancel/retry and
  `tasks_text_batch.py` — table only; the full prose elsewhere in the file
  already covered this correctly. And added the `staffline_detections` row
  to the Database schema table itself — a previous session summary had
  claimed this row already existed; it didn't, this is the actual fix.
- **CI job added**: `.github/workflows/tests.yml` runs the staff-finding
  suite and the new `landing-page/scripts/tests/` on every push.
- **`main` merged in and briefly broke `auth_api.py`**: a large merge from
  `main` (bringing in real, unrelated feature work — job cancellation,
  image resizing, a new `tasks_text_batch.py`, ~45 files) collided with the
  `staffline_detections` addition to `init_db()`, leaving a duplicated
  `annotations` table block and code dedented outside its `try:` block —
  a genuine syntax error, not just a style issue. Repaired by comparing
  directly against both merge parents and reconstructing `init_db()` with
  the other side's new error-handling (`except DuplicateTable/
  DuplicateObject/UniqueViolation`) preserved. Worth noting for the eventual
  main-rejoin: `main` is moving fast, re-diff before assuming anything about
  its current state.
- **`tasks_predict.py` scoping bug**: the same merge's independent
  `has_annotation`/`has_text_alignment` skip logic left `yolo_txt`/
  `img_arr`/`ann_id` unassigned when an image already had an annotation but
  not yet a text-alignment — exactly the case that logic exists to support.
  Fixed by decoding the image unconditionally and fetching the existing
  annotation when YOLO is skipped.

## CodeRabbit review pass (PR #53), 2026-07-30

Went through every open CodeRabbit comment on
[PR #53](https://github.com/DDMAL/mothra/pull/53) (`kyrie/staff-finding` →
`main`) individually, including several that a GitHub posting failure left
only in the review-body text, never as inline comments. Fixed the ones
agreed with; two items above (`component_filter.py` bookkeeping,
`staffline_detections` retention) were agreed with but deferred rather than
rushed. Declined one on purpose: the `detect_stafflines.py`/`test_model.sh`/
`scripts/run_inference.py` same-stem-filename-collision finding is real, but
`test_model.sh`'s own help text already documents it as a known, accepted
limitation across all three files — not an oversight to fix without that
being a deliberate decision.

Fixed, each independently verified (a real test proving the specific
behavior, not just a passing compile):
- `staffline_stage.py`: the docstring promised a `status='failed'` row on
  error; the code never wrote one. Implemented it.
- `staffline_stage.py` / `tasks_predict.py`: cooperative cancellation was
  only checked once per image, not inside `run_staffline_detection`'s own
  per-box loop — exactly the gap CLAUDE.md warns new long-running stages
  need to avoid. Added a `check_cancelled` call per box, with `JobCancelled`
  explicitly re-raised past the stage's own broad `except Exception`.
- `tasks_predict.py`: the `has_text_alignment` skip's `continue` sat before
  the staffline-detection block, so an image with fresh YOLO boxes but an
  already-existing text alignment never got a `staffline_detections` row at
  all. Reordered so staffline detection is gated only on `has_class`.
- `run_page.py` / `bgr_adapter.py`: `run_page.py` imported `bgr_adapter` at
  module top, so `--no-bgr` (and even `--help`) still needed the external
  `inference_simple` dependency to resolve. Made the import lazy, reached
  only inside the `use_bgr` branch.
- `run_page.py`'s fallback-redetect path: `box_index` was computed before
  any candidate was accepted, so multiple candidates in one region collided
  on the same diagnostic filename (each overwriting the last), and the
  summary row's recorded path didn't match what was actually written
  anyway. Gave each candidate its own per-region, per-candidate path,
  carried through to its summary row via the candidate's own identity.
- `fallback_redetect.py`: candidates that just re-detected an existing line
  could win a `max_new_lines` slot before `group_staves`' own duplicate
  reconciliation ever got a chance to run, displacing a genuinely missing
  line. Added an existing-line-distance rejection before ranking.
- `eval_page.py`: crashed on interpolated JSOMR records missing
  `centerline_page` (no `bounding_box` to fall back to). Now degrades to an
  empty result instead of crashing. CodeRabbit followed up asking for the
  fixture-regeneration side too (the double-page-offset risk for stale
  pre-`centerline_page` fixtures) — regenerated all four tracked
  `implicit_neural*` Gent-right fixtures via `run_implicit_neural_page.py`;
  every record now carries `centerline_page`, verified against
  `eval_page.py`'s own `_pred_page_y`/`_pred_page_x` helpers.
- `dp_tracer.py` (experiment): the DP cost only rewarded dark pixels, so a
  trace could drift onto a darker neighbouring staffline within the same
  search band. Added a distance-from-hint penalty, weight chosen by actually
  reproducing the drift on a synthetic adversarial case and finding the
  tipping point, not guessed.
- `periodicity_detector.py` / `run_periodicity_page.py` (experiment): the
  full-page Gaussian blur was recomputed on every staffline call instead of
  once per page. Hoisted into `compute_dark_field()`; confirmed
  byte-identical output before/after.
- `scripts/run_inference.py`: failed images were silently skipped with no
  structural record and no non-zero exit code, so a partial batch run looked
  identical to a complete one to any automation checking the exit code.
  Now writes `failed_images.json` and exits 1 on partial failure;
  `all_predictions.json`'s shape is unchanged for annotator compatibility.
- `group_staves.py`: docstring claimed `interpolation_max_gap` defaults to
  `cut_threshold * INTERPOLATION_GAP_MULTIPLIER`, but the adaptive
  gap-distribution path is what actually resolves it in the normal case.
- `.gitignore`: `!models/**/*.pt` only rescues the repo-root `models/`
  directory, not `landing-page/scripts/assets/models/medieval/` (that
  path's `.pt` files are tracked today purely because gitignore doesn't
  apply retroactively — a future accidental untrack would silently need
  `-f` to re-add them). Added a path-specific negation. Left
  `staff-finding/models/` alone — confirmed those files are genuinely,
  intentionally untracked, not a gap.
- `staff-finding/pyproject.toml`: `gp_fitter.py` needs `scikit-learn` and
  `implicit_neural_fitter.py` needs `torch`, neither declared anywhere.
  Added both as a new `experiments` extra.
- `landing-page/Dockerfile`: switched the staff-finding install from
  editable to normal, since the whole source tree is already baked into
  the image regardless.

## Diagnosed this session (kyrie/staffline-landing-audit)

Kyrie reported the same manuscript page (`Plimpton041_167v`) producing
visibly worse staffline-detection results through mothra-landing (the
deployed k8s instance) than through a local `staff-finding` pipeline run,
with a direct local-vs-deployed comparison as evidence: the local run
grouped 5 staves cleanly at mode 4 lines/stave (correct for this chant
manuscript); the deployed run's `staffline_detections` JSOMR showed
`lines_expected: 2` (a computed statistic — `group_staves.py`'s own
`mode_lines_per_stave`, not a config knob) — i.e. a real quality gap, not a
different reading of the same data.

- **Root cause 1 — git version drift, not an algorithm bug.**
  `kyrie/staff-finding` was 50 commits behind `origin/main` (what's actually
  deployed). `git diff --stat` confirmed `staff-finding/`'s own algorithmic
  package hadn't drifted at all — the divergence was entirely in
  landing-page's integration layer, which had picked up several relevant
  fixes since the last merge: `916f465` (staffline detection now also runs
  in the text-batch path), `1b2ea4d` (interpolate preview/confirm flow),
  `7c80858` (flips tier-3's synthetic-line fabrication to off by default,
  tags every MEI's `stave_source`). Bundled `.pt` model weights were
  byte-identical between local and deployed, ruling out a model-retrain
  explanation. Branched fresh from `origin/main`
  (`kyrie/staffline-landing-audit`) to re-diagnose on current code rather
  than chasing an already-fixed gap.
- **Root cause 2 — real, independent staleness bug, fixed this session.**
  `tasks_encode.py`'s `_resolve_hints()` tier-1 query picked the newest
  `succeeded` `staffline_detections` row by `created_at`, with no check that
  it matched the image's *current* annotation. Since `staffline_detections`
  accumulates forever (see the retention note above) while `annotations` is
  delete-then-insert on re-predict, a project re-annotated/re-predicted
  since its last staffline detection would still match on `created_at DESC`
  and silently encode against geometrically stale stave data — exactly the
  scenario a fresh local DB never hits, but a long-running deployed DB
  eventually does. `inference_api.py`'s interpolate-preview/confirm routes
  already guarded against this exact class of staleness (resolving the
  current annotation via `image_id` before trusting a detection row); the
  same guard was missing from `_resolve_hints` itself. Fixed: `_resolve_hints`
  now looks up the image's current annotation first, then requires tier 1's
  `staffline_detections` row to match that annotation's id — falling
  through to tier 2 (using the current annotation's own `yolo_txt`) when no
  detection matches, instead of trusting the newest timestamp alone. Test:
  `landing-page/scripts/tests/test_resolve_hints_staleness.py` (hand-rolled
  fake cursor + `sys.modules` stubs for `auth_api`/`job_store`/`celery_app`,
  since importing `tasks_encode` for real would need a live Postgres/Redis —
  same DB-at-import-time constraint `test_bbox_pipeline_integrity.py`'s own
  docstring already documents).
- **Live end-to-end re-verification against the actual reported page could
  not be completed in this session** — the sandbox this diagnosis ran in
  has neither `ic/api/.venv` nor `text-service/.venv` set up, and no Redis
  broker reachable, so `./dev.sh`'s full 5-service stack (needed for a real
  predict-job re-run) couldn't be started. Whoever picks this up next should
  re-run `Plimpton041_167v` through a real `POST /projects/{id}/predict` on
  `kyrie/staffline-landing-audit` (or `main`) and confirm the resulting
  `staffline_detections.mode_lines_per_stave` now matches the local e2e
  baseline (mode 4) instead of the originally-reported `lines_expected: 2`.
  If it still doesn't match after root causes 1 and 2 above are accounted
  for, the next suspects (in order, per this session's process of
  elimination — model weights and `staff-finding/` code already ruled out):
  1. Fewer/worse stave-class (YOLO class id 2) boxes in the deployed
     project's actual `annotations.yolo_txt` for this image than the local
     e2e fixture's `yolo_txt/Plimpton041_167v copy.txt` — check which model
     that specific project actually uses (medieval preset vs. a custom
     uploaded model row in `project_models`).
  2. `landing-page/src/utils/imageResize.ts`'s client-side resize (uploads
     over 5MB) landing the deployed project's source image at different
     pixel dimensions than the local e2e fixture's raw file — this changes
     `scale_unit` (median stave-box pixel height) and therefore the Sauvola
     binarization window size and downstream fit/grouping thresholds even
     with identical code and models.
  3. Only if both check out clean: this may be a real instance of the
     already-documented, unfixed `group_staves.py` dense-stave
     reconciliation gap (see "Three checked-in e2e baselines show genuinely
     fragmented grouping" above) rather than an integration bug — a real
     algorithmic limitation, not a quick fix.

## Watching for calvo-integration merge

`gianna/calvo-integration` (not yet merged; reviewed via `gh api` this
session, not a local checkout) is an **ink-separation front-end**, not a
bounding-box detector in its own right — exactly the "ink-separation (BGR)"
deferral already flagged in `staff-finding/dox/STATUS.md` and above, just
implemented via DDMAL's own `Paco_classifier` submodule instead of the
unvendored `muscrat/layer_sep` repo. `paco_api.py`/`paco-classifier-service`
wrap `Paco_classifier.recognition_engine.process_image_msae()` (a TensorFlow
pixel classifier) to split a page into background/stafflines RGBA layers;
`tasks_predict.py`'s `_run_medieval_inference` on that branch then runs the
*existing* stave YOLO model against the classified "stafflines" layer
instead of the raw page crop, still producing ordinary YOLO-txt boxes at
`STAFFLINE_CLASS_ID = 2` (confirmed unchanged: `CATEGORY_TO_SLOT["staves"] =
2` in `yolo_inference.py` on both branches).

No format-level blocker for `staff-finding/` when this merges. Two things
worth validating at merge time rather than assuming they hold:
- `component_filter.py`'s Sauvola binarization is tuned against raw
  grayscale crops; once ink-separation actually lands, it'll instead see a
  pre-segmented layer with background forced to pure white (per
  `_layer_to_rgba_png`'s masking). Worth a validation pass against a few
  e2e fixtures — the tuning that was an "interim mitigation for faint ink"
  may behave differently (better or worse) against an already-cleaner
  input than it was tuned for.
- `run_staffline_detection` on that branch crops `staffline_source_arr` (the
  same classified image the stave YOLO model scored), not the raw page.
  `process_image_msae` restores full input resolution internally, so
  `scale_unit` (median stave-box pixel height) should still resolve in the
  correct coordinate frame — but this is a "should," not yet an assertion;
  add an explicit test for it once this merges rather than assuming it
  holds.

Also noted for later, out of scope here: the eventual pitch-finding consumer
of this staffline geometry is
[DDMAL/Standalone-Pitch-Finder](https://github.com/DDMAL/Standalone-Pitch-Finder)
(in progress, not yet integrated) — its JSOMR/stave-geometry output contract
needs to satisfy that project too, not just today's MEI-encoding consumer
(`staffline_adapter.py`/`encode_to_mei.py`).

## Alpha 1 fix batch (kyrie/alpha1-staffline-parity), 2026-08-17

Closed all nine "Alpha 1 — Staff-finding parity" tickets in one PR, closing
[#213](https://github.com/DDMAL/mothra/issues/213). Full before/after detail
lives in `ALPHA_TRANSITION_PLAN.md`'s findings register (§2.1, SF-1 through
SF-9 rows, each updated in the same PR); summary here for anyone landing on
this file first:

- **SF-1** (primary cause of #213) — stave-class confidence default was 0.5
  in landing vs. staff-finding's own proven 0.25; decoupled into its own
  `yolo_inference.DEFAULT_STAVE_CONFIDENCE` constant rather than falling
  back to the shared text/music default. Resolves DL-14.
- **SF-9** — `DEFAULT_STAFFLINE_CLASS`/`STAFFLINE_CLASS_DEFAULT` fixed 2→0
  across `run_page.py`/`run_pageOG.py`/`eval_page.py`/`eval_batch.py` to
  match the bundled single-class stave model's real output class; without
  this, every real box was silently matching nothing.
- **SF-4** — paco-classifier layer converted BGR→RGB immediately at decode
  (`_decode_paco_layer`), fixing a channel-order split that fed
  `component_filter.py`'s `COLOR_RGB2GRAY` step transposed R/B luminance on
  the classifier-success path.
- **SF-5** — `crop=crop` now threaded through both `fit_centerline()` call
  sites in `staffline_stage.py`.
- **SF-8** — `quality.flags` now merges component-filter + fit + assignment
  flags (previously only fit flags survived); new `settings_json.page_flags`
  carries page-level grouping flags. Per-record ids were already stable
  through the sort (confirmed, not a bug).
- **SF-7** — predict-time image dimensions now recorded in `settings_json`;
  `tasks_encode.py`'s `_resolve_hints()` scales or rejects tier-1 JSOMR when
  the encode-time image's dimensions don't match, instead of trusting stale
  pixel coordinates blind. The tier-1 bare `except: pass` now logs, matching
  its sibling lookups in the same function.
- **SF-2** — predict now prefers `project_images.original_data` over the
  resized working copy (mirrors `get_original_image()`'s existing fallback).
  Depends on SF-7 landing first, since JSOMR's absolute-pixel coordinates
  are resolution-sensitive in a way YOLO's own normalized boxes aren't.
- **SF-6** — `_decode_paco_layer()` helper removes literal duplication
  between the fresh and reused-annotation classify/decode paths; every
  `staffline_detections` row now records a `source_label` provenance field
  in `settings_json`.
- **Golden-page parity test** — `staff-finding/scripts/script_tests/test_golden_parity.py`,
  picked up automatically by the existing CI glob. Gent right is a real,
  live-reproducing regression test pinned to today's (still fragmented, see
  "Three checked-in e2e baselines" above) output. **MS234_64 could not get a
  live test** — its raw source image was never checked into this repo (only
  ever lived at an external path on the machine that ran the original
  2026-08-10 parity sweep); the test instead pins the checked-in
  `baseline.json` snapshot against silent corruption, explicitly documented
  as not a regression test. Checking that source image into the repo (or
  Git-LFS'ing it) is an open follow-up.

Also fixed in passing: the stale "`batch_api.py`'s text-batch-run path"
bullet above (struck through, was already superseded by `916f465`).

## Needs a manual step, not a code change

- **Branch protection**: `.github/workflows/tests.yml` runs on every push,
  but nothing requires it to pass before merging yet. Turn that on under
  the repo's Settings → Branches → branch protection rule for `main` →
  "Require status checks to pass before merging" → select `tests / test`.
