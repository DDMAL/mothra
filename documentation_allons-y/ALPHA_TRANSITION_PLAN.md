# Mothra PoC → Alpha Transition Plan

**Status:** active · **Created:** 2026-08-10 · **Owner:** kyrie (program), owner slots per phase below
**Tracking:** milestone-per-phase on GitHub + the `mothra` project kanban (joint scrum-kanban: phases = sprint goals, day-to-day flow on the board)

---

## 1. Why this document exists

Mothra is moving from proof-of-concept to an alpha phase with real users. Three forcing functions:

1. **A confirmed landing-vs-standalone result mismatch.** Commit `597c3ad` (2026-08-10) reran McGill MS234_64 through the standalone staff-finding CLI with basic settings and got clean results (8 staves, mode 4 lines/stave, cut threshold 62 px) — while the same page inside mothra-landing produces degraded results. A code audit (2026-08-10, three parallel deep-reads of the staff-finding integration, the text pipeline, and the scaffolding surface) found the mechanisms; they are catalogued in the findings register (§2) and attacked in Phase 1–2.
2. **Accumulated PoC scaffolding.** Placeholders, mocks, dev bypasses, unauthenticated routes, and temporary workarounds exist throughout landing-page, text-service, and the deployment story. §2.3 inventories all of them with a disposition each.
3. **No end-to-end procedural review has ever happened.** Every integration seam (YOLO → staffline stage → adapter → encode; YOLO → mask → text-service → alignments; landing → paco layer) was built incrementally. This plan schedules a full review-and-test pass, seam by seam.

**End-state goals** (in priority order):

- A **functional alpha**: results inside the landing app match validated standalone results, failures are loud, and nothing fabricated reaches a user.
- **ONE central configuration layer**: every tunable, default, path, and cross-service constant declared exactly once (§ Phase 4).
- An **organized, professional repo**: clean to navigate, learnable by a new contributor from the root README in one hop (§ Phase 6).

**Rules of engagement:**

- Nothing in the findings register gets "fixed" before Phase 0's parity harness can measure the fix. Measured attribution first, then change.
- Every DECIDE item goes through the decision log (§4), not a drive-by commit.
- Each phase exits by demo against its acceptance criteria, not by calendar. Week numbers below are planning guides.

---

## 2. Findings register

Status legend: **confirmed** = re-verified directly in source at HEAD (`74c2243`) · **audited** = from the 2026-08-10 deep-read, not independently re-verified · **measured** = quantified by the parity harness (Phase 0 updates this column).

### 2.1 Staff-finding: landing vs standalone divergences

The standalone 10aug run was: `detect_stafflines.py --conf 0.25` (imgsz 640, iou 0.7) on the raw source JPEG → `run_page.py --staffline-class 0 --no-bgr` (sauvola, merge on, no interpolation, no fallback-redetect). The landing path differs as follows, ranked by expected impact:

| ID | Finding | Where | Status | Phase |
|---|---|---|---|---|
| SF-1 (D3) | Stave-box confidence 0.5 (landing default) vs 0.25 (standalone run). Different box set → different `scale_unit` → shifts every scale-relative constant downstream (Sauvola window, min component size, merge/dedup tolerances, cut threshold). **Measured (2026-08-10 parity sweep, MS234_64): catastrophic — at conf 0.5 only 9/50 boxes survive; page collapses from 8 staves (mode 4) to 1 stave (mode 9), cut threshold 62 → 309.5 px. This alone reproduces the landing symptom.** | `landing-page/scripts/inference_api.py:21`, `landing-page/src/hooks/useInferenceSettings.ts:20` vs `staff-finding/test_model.sh:50`; `staff-finding/e2e_tests/10aug_parity/ms234_64/` | **fixed 2026-08-17** — stave-class default now 0.25 via `yolo_inference.DEFAULT_STAVE_CONFIDENCE`, decoupled from the shared text/music default; resolves DL-14 | 1 |
| SF-2 (D5a) | Landing runs on the client-side downscaled, re-encoded JPEG working copy (`project_images.data`, resize triggered >5 MB, target 2 MB, JPEG q 0.9→0.5); standalone ran the raw source file. The untouched original sits unused in `project_images.original_data`. **Measured null on MS234_64** — its source file is 1.4 MB, under the 5 MB trigger, so no resize occurred (yMAE 0.0). Still live risk for larger pages; re-measure on a >5 MB golden page. | `landing-page/src/utils/imageResize.ts:1-47`, `landing-page/scripts/images_api.py:59-74`, `tasks_predict.py:195` | **fixed 2026-08-17** — `tasks_predict.py`'s validating-stage fetch now prefers `original_data`/`original_mime_type`, falling back to `data`/`mime_type` (mirrors `get_original_image()`); YOLO boxes are unaffected (xywhn is normalized), JSOMR is resolution-sensitive but SF-7's frame-parity guard now covers it. Re-measurement against a real >5MB page still needed (tracked as a running-note, not blocking) | 1 |
| SF-3 (D5b) | Landing crops the **paco-classifier stafflines layer**; standalone crops the raw page. Sauvola constants (window ×9, k 0.1) were tuned on raw parchment crops; masked-out pixels are forced pure white (flat background, σ≈0), and ink the classifier missed is erased entirely (fragmentation risk). Flagged unvalidated in `STAFFLINE_INTEGRATION_FOLLOWUPS.md:363-376`. **Not yet measured — paco service unavailable on the dev machine during the 2026-08-10 sweep (variant skipped); run when the service stack is up.** | `tasks_predict.py:64,69,286`, `paco-classifier-service/main.py:119-140`, `staff-finding/scripts/component_filter.py:86-99` | confirmed, unmeasured — **not in this batch's scope** (not one of the nine Alpha-1 tickets) | 1 |
| SF-4 (D5c) | **BGR fed to a hardcoded RGB→gray conversion.** The paco layer is decoded `cv2.IMREAD_COLOR` (BGR) and passed to `component_filter._binarize`, which does `COLOR_RGB2GRAY` — R/B luminance weights transposed on every ink pixel. Inverted on the classifier-unreachable fallback path: raw RGB goes to ultralytics, which expects BGR ndarrays. **Measured (MS234_64): minor real contributor — yMAE 0.48 px (max 5.1), grouping distribution shifts 4:7,5:1 → 4:6,5:2, one new line. (Measured on the raw page; interaction with the paco layer still unmeasured, see SF-3.)** | `tasks_predict.py:64` vs `staff-finding/scripts/component_filter.py:470-471`; fallback `tasks_predict.py:121-122` | **fixed 2026-08-17** — paco-layer array converted BGR→RGB immediately after decode (`_decode_paco_layer`, both the fresh and reused-annotation call sites), so every array reaching `infer_staves()`/`component_filter` is consistently RGB | 1 |
| SF-5 (D1) | `fit_centerline` called **without `crop`** in landing → different line-following `seed_y` (crop vertical centre standalone vs kept-pixel y-midpoint landing). 48/50 boxes on MS234_64 evaluated the seed (43 accepted the refit). **Measured null on MS234_64** — identical output with and without crop (yMAE 0.0); the two seed formulas evidently landed in the same trace band on every box here. Fix is still correct hygiene; re-check on a page with companion fragments. | `landing-page/scripts/staffline_stage.py:218` vs `staff-finding/scripts/run_page.py:448-453`; mechanism `staff-finding/scripts/fit_centerline.py:257-262` | **fixed 2026-08-17** — `crop=crop` now threaded through both `staffline_stage.py` call sites; `fit_centerline.py`'s own docstring corrected (was "BGR-preprocessed", now correctly RGB) | 1 |
| SF-6 (D7/D6/D8) | **Five image variants** feed staffline detection depending on path: paco layer (fresh medieval predict); raw RGB page (re-predict when `has_annotation`, `tasks_predict.py:272`); raw RGB page (interpolate-preview/confirm, `inference_api.py:163`); raw RGB page (text-batch, `tasks_text_batch.py:125`); raw RGB on the classifier-fallback path. Results are non-reproducible against themselves, and `settings_json` does not record which image was used. | as cited | **fixed 2026-08-17** — shared `_decode_paco_layer()` helper removes the literal duplication between the fresh and reused-annotation classify/decode paths; every `run_staffline_detection()`/`compute_staffline_interpolation()` call now records a `source_label` (`paco_layer`/`raw_page`/`raw_page_fallback`) in `settings_json` for provenance | 1 |
| SF-7 (§4) | Tier-1 JSOMR stave hints are absolute pixels in the *predict-time working-copy* frame, but `tasks_encode.py` derives `page_w/h` from a *separate encode-time upload*. No frame-parity assertion; tier-1 failures swallowed by bare `except Exception: pass` → silent degradation to tier 2 with no log. | `landing-page/scripts/tasks_encode.py:117-139,169,228-236` | **fixed 2026-08-17** — predict-time image dimensions now recorded in `settings_json`; `_resolve_hints()` compares them against the encode-time image and scales (aspect-ratio-preserving resize) or rejects (genuine mismatch) tier-1 accordingly; bare `except: pass` now logs via the same `[resolve-hints]` pattern as its sibling lookups | 1 |
| SF-8 (D4) | Stored JSOMR loses component-filter + grouping flags (only fit flags kept); records re-sorted by (stave_id, index) while ids stay in fit order — hampers landing-vs-standalone artifact diffing. | `staffline_stage.py:122,135,177-180` vs `run_page.py:728,735-739` | **fixed 2026-08-17** — ids confirmed already stable/content-derived through the sort (not a bug); `quality.flags` now merges all three tiers (component-filter + fit + assignment); new page-level `settings_json.page_flags` carries `grouping_result.flags` | 1 |
| SF-9 | `run_page.py`'s `DEFAULT_STAFFLINE_CLASS = 2` matches neither real invocation (detect_stafflines emits class 0; landing emits class 2 but never calls run_page). Silent zero-box run if invoked without the flag. `staffline_stage.py:37`'s comment claiming parity is misleading. | `staff-finding/scripts/run_page.py:57`, `test_model.sh:126-136`, `staffline_stage.py:37-38` | **fixed 2026-08-17** — default changed 2→0 in `run_page.py`/`run_pageOG.py`/`eval_page.py`/`eval_batch.py` (module constant + CSV-manifest fallback + usage docstrings); `eval_page.py`'s misleading comment corrected to cite `train_staffline_detector.py`'s actual remap | 1 |

### 2.2 Text pipeline: overrides & silent failure modes

| ID | Finding | Where | Status | Phase |
|---|---|---|---|---|
| TX-1 | **Silent stub mode.** `RECOGNITION_MODEL` resolved once at import; stub warning logged only in OCR-only mode — never Cantus-aligned, never on batch; batch persists `log_text=""`. Stub output is a valid-looking row with `syllable_count=0`, indistinguishable from a genuine no-text page. This dev machine currently has no Tridis model. `dev.sh` does not preflight it. | `text-service/main.py:47,221-226,401`, `tasks_text_batch.py:176`, `dev.sh:107-109` | confirmed (local stub state verified) | 2 |
| TX-2 | Mask/music-box lookups keyed by `image_name` while everything else keys `image_id`; names have no uniqueness constraint — same-named images can pull each other's masks. | `text_api.py:51-56,97-103`, `auth_api.py:786-793` | audited | 2 |
| TX-3 | **Failure asymmetry.** Application errors from text-service → swallowed into a log line, predict job *succeeds* with no row and no persisted per-image trace. Mid-stream socket `TimeoutError` → caught by neither handler → *whole job fails*, including already-completed images. No heartbeat around text-finding vs the 90 s stale-job killer (paco has one; text doesn't). | `tasks_predict.py:316-317,329-331`, `text_api.py:133,158-159,245-253`, `jobs_api.py:17,46-54` | audited | 2 |
| TX-4 | **No supported re-run.** `text_alignments` accumulates forever; `has_text_alignment` permanently skips predict/batch; only the synchronous endpoint appends more rows. No delete/supersede path. | `tasks_predict.py:200-211,293-295`, `text_api.py:256` | audited | 2 |
| TX-5 | **Divergent annotation writers.** `tasks_text_batch.py` runs plain raw-page YOLO and overwrites classifier-derived annotations unconditionally; predict then reuses them forever via `has_annotation`. Newest `text_alignments` row can be built against a *previous* annotation's boxes with no currency check (contrast `_resolve_hints`' `annotation_id` guard). | `tasks_text_batch.py:103-104`, `tasks_predict.py:228-229`, `auth_api.py:779-834` | audited | 2/4 |
| TX-6 | Dead code + misreporting: `filter_lines_over_music`/`_bbox_overlap_ratio` never called (filter moved into `run()`); debug payload hardcodes `"threshold": 0.30`; `lines_pre_filter` is actually post-filter; stale docstrings. | `text-service/main.py:71-95,189,287,317,359,478,506` | audited | 2 |
| TX-7 | Defaults duplicated across 4 layers (device cpu, bimodal 0.5, mask padding 15, masking on, overlap filter on) with no source of truth. Device asymmetry: YOLO defaults `auto` (GPU), text hardcodes `cpu`. | `inference_api.py:30-34`, `batch_api.py:54-59`, `text_api.py:171-175`, `text-service/main.py:175-180` | audited | 4 |
| TX-8 | Zero text boxes → mask is `None` → **silently degrades to unmasked full-page OCR** (by design, to avoid blacking out the page — but unlogged in any persisted place). | `text_api.py:127-128`, `text-service/main.py:240-241` | audited | 2 |
| TX-9 | Two box conventions in one request payload: `music_boxes` = xmin/ymin/xmax/ymax; `mask_json` bbox = x,y,w,h. | `text_api.py:75-78,123-129` | audited | 4 |
| TX-10 | Batch failure-index bug: `folio_list[result_holder["completed"]]` IndexErrors when failure lands after the last folio increments. | `text-service/main.py:522-524` | audited | 2 |
| TX-11 | Custom seg/rec model paths are backend-container filesystem paths sent as strings; `stored_models` is not mounted on text-service — custom text models cannot resolve in containers. | `text_api.py:209-212`, `docker-compose.yml` | audited | 2 |
| TX-12 | `mothra-text/` submodule uninitialized on the primary dev checkout — internals (run_pipeline defaults, internal YOLO usage given music_boxes) unreviewed. | `git submodule status` | confirmed | 0 |
| TX-13 | k8s memory limit for text-service is 2 Gi — the exact figure documented as OOM-killing Kraken. | `k8s/text-service.yaml`, CLAUDE.md resource notes | audited | 5 |

### 2.3 Scaffolding inventory (dispositions)

**SAFE — remove, no replacement needed** (Phase 3, one PR):

| Item | Where |
|---|---|
| Dead unauthenticated `POST /api/encode` + entire `MOCK_DATA_DIR` chain (mock dir is gitignored and absent → route 500s everywhere; frontend never calls it) | `encode_api.py:9,20-27`, `config.py:19`, `config.yaml:9` |
| `"neon-test"` View union member (zero references) | `landing-page/src/types.ts:123` |
| Dormant fake-progress timer block (dead in all current call paths; only runs when no `streamRequest`) | `ProcessingPage.tsx:146-186` |
| "mock XML data will be used" string (describes an unreachable path) | `ICCompletionTestPage.tsx:139` |
| Committed `staff-finding/config.yaml` (its own header says never commit it) | `staff-finding/config.yaml` |
| Foreign-developer absolute path | `scripts/convert2greyscale.py:11` |
| `z_quicktest/` scratch dir (untracked; duplicate `.pt` copy) | repo root |

**REPLACE — must have a substitute before alpha** (Phase 3):

| Item | Where | Replacement |
|---|---|---|
| No auth on any `encode_api.py` route; caller-supplied `project_id`, no ownership check | `encode_api.py` (all 6 routes) | `Depends(get_current_user)` + `require_project_owner`, matching every other router |
| `MOTHRA_SECRET` silent random fallback (JWTs invalidated per restart; backend/worker diverge) | `auth_api.py:51` | fail-fast like `DATABASE_URL` (`auth_api.py:31`) |
| CORS `*` defaults: backend env default, text-service unconditional | `landing-page/scripts/main.py:32`, `text-service/main.py:42`, `docker-compose.yml:74` | explicit origin allowlist, env-configured, no `*` default |
| Cleanup functions run only at process start | `main.py:53-54`, `job_store.py:287,306` (note typo `cleanup_stale_uplaods`) | Celery beat schedule (or equivalent periodic invocation) |
| No `/healthz` on backend/text-service (TCP probes only; paco already has real `/health`+`/ready`) | `k8s/backend.yaml:36-45`, `k8s/text-service.yaml:29,34` | real health endpoints checking DB/broker reachability, wired to probes |
| `BATCH_DIR` + `NEON_MANIFESTS_DIR` node-local disk state | `text-service/main.py:26`, `config.py:18`, `mei_api.py:212-213` | Postgres-backed (like `job_sessions`) or shared storage |
| `init_db()`/`_migrate_db()` at import → replicas pinned to 1 | `auth_api.py:253,624`, `k8s/backend.yaml:9` | one-shot migration Job; remove import-time side effect |
| 800 ms sleep hoping Neon's save lands before `PATCH .../corrected` | `NeonBatchEditor.tsx:108` | real completion signal from Neon's `updateDatabase()` |
| Triplicated inline 86400 s cleanup sweeps at import/request time | `auth_api.py:617-623`, `mei_api.py:115-120`, `text-service/main.py:29-34` | fold into the periodic cleanup above |

**DECIDE — decision log §4** (recommendation included there): SKIP_PREDICT/SKIP_YOLO/placeholder-bbox-grid triad · `ICCompletionTestPage` + hardcoded "ta-da!" copy · paco-classifier personal-branch pin · checked-in artifact corpus (HF-first policy, §5.8) · eslint disabled in CI · tests not a required check · `run_pageOG.py` · unauth'd `GET /jobs/{id}/stream` · `bgr_adapter.py` machine paths · ProcessingPage cosmetic pacing (drip-feed + 4 s completion delay) · "printed text coming soon" pill · `OLD-annotator/`, root `scripts/`, `configs/` legacy dirs · null `confidence` placeholder in JSOMR output.

**Stale documentation to correct now:** CLAUDE.md claims fake `setTimeout` timers are gone (dormant, not gone) and that no `job_uploads`/`job_sessions` cleanup exists (exists, but start-only).

---

## 3. Phases

Each phase lists: goal, tasks (issue-granular), exit criteria. Owner slots to be filled at sprint planning.

### Phase 0 — Measure before touching (Week 0)

**Goal:** every proposed fix has a measurement in hand before it lands.

| # | Task | Acceptance |
|---|---|---|
| 0.1 | **Parity harness** `staff-finding/scripts/parity_harness.py`: runs the exact landing-path staffline invocation (staffline_stage semantics, no DB) and exact standalone invocation side by side, with independent toggles: `--conf {0.25,0.5}`, `--channel-order {rgb,bgr}`, `--pass-crop/--no-pass-crop`, `--image {original,working-copy,paco-layer}` (working copy simulated at imageResize.ts parameters; paco layer needs :8003 up). Emits stave count, mode, distribution, cut threshold, per-line y-MAE vs baseline, diff table. | Baseline toggle-set reproduces the 10aug MS234_64 result exactly (8 staves, mode 4, cut 62 px) |
| 0.2 | Run harness on MS234_64 + Gent right, flipping one toggle at a time; commit results to `staff-finding/e2e_tests/<date>_parity/`; update the findings register's Status column to **measured** with the attribution numbers. | Every SF-1…SF-5 row has a measured delta (or measured null result) |
| 0.3 | Provenance spec: extend `staffline_detections.settings_json` to record image-source variant, conf threshold, channel order, model hash. (Spec here; implementation in Phase 1.) | Spec reviewed |
| 0.4 | Local stub-mode audit: init `mothra-text` submodule, install Tridis via `htrmopo get` + rename, verify text-service leaves stub mode; document exact local setup in text-service README. | Local text run produces non-empty OCR |
| 0.5 | `mothra-text` internals review checklist: run_pipeline defaults vs the 4-layer landing defaults (TX-7), internal YOLO behavior when `music_boxes` provided, mask semantics. | Checklist doc committed; TX findings updated |

### Phase 1 — Staff-finding parity (Weeks 1–2)

**Goal:** landing staffline output ≈ standalone output on the golden pages, within a defined tolerance, enforced by CI.

| # | Task | Notes |
|---|---|---|
| 1.1 | Fix SF-4: convert paco layer BGR→RGB at decode (`tasks_predict.py:64`), and audit every array handoff for channel order (incl. the ultralytics-fallback inversion). Add a channel-order assertion or convention comment at each seam. | Measured first (0.2) |
| 1.2 | Fix SF-5: pass `crop` in `staffline_stage.py:218`; fix the misleading `fit_centerline` docstring ("only required if save_path…"). | |
| 1.3 | Resolve SF-1: decide the canonical stave conf (likely 0.25 to match evaluated settings); single declaration (Phase 4 config layer), UI reads the same number; document that eval and product must share it. | Decision-log entry if product wants a different default than eval |
| 1.4 | SF-2 experiment: run predict path against `original_data` vs `data` on the golden set; decide whether staffline/YOLO inference should read `original_data` (with coordinate mapping back to working-copy space for display) or whether resize params need loosening. | Decision + implementation |
| 1.5 | SF-3 A/B: paco layer vs raw page with everything else held equal (harness `--image` toggle); keep the layer only if it measurably wins; if kept, schedule a Sauvola re-tune on layer crops. | |
| 1.6 | Fix SF-6: single `resolve_staffline_source()` used by all five call paths (fresh predict, re-predict, interpolate-preview/confirm, text-batch, fallback), so one image-selection policy exists. Record the chosen source in provenance (0.3). | |
| 1.7 | Fix SF-7: assert JSOMR frame ↔ encode-time `page_w/h` parity (store predict-time dimensions in the detection row; scale or reject on mismatch); replace the bare `except` with logged tier-degradation. | |
| 1.8 | SF-8: persist full flag set (component-filter + fit + grouping) in JSOMR `quality.flags`; keep ids stable through sort. | |
| 1.9 | SF-9: fix `DEFAULT_STAFFLINE_CLASS` mismatch story (make run_page require the flag, or align defaults); correct `staffline_stage.py:37`'s comment. | |
| 1.10 | Golden-page CI test: harness baseline run on MS234_64 (+ Gent right) as a pytest, tolerances defined, added to `tests.yml`. | **Done 2026-08-17** (`staff-finding/scripts/script_tests/test_golden_parity.py`, picked up automatically by the existing `script_tests/` glob in `tests.yml`, no new workflow step). Gent right: live re-run against the checked-in fixture pair, pinned as today's (still-fragmented, per the known `group_staves.py` reconciliation gap) baseline. **MS234_64 gap found while implementing this: its raw source image was never checked into the repo** (only ever lived at `/Volumes/Expansion/script_sorter_mss/McGill_MS234/McGill_MS234-064.jpg` on the machine that ran the 2026-08-10 sweep) — so a live-reproducing test against it isn't possible yet. `test_golden_parity.py` instead pins the checked-in `baseline.json` snapshot's own recorded numbers against silent corruption, clearly documented as not a regression test. Getting the source image into the repo (or Git-LFS'd) is a follow-up, not done here. |

**Exit criteria:** golden-page test green in CI; findings register SF rows closed with measured before/after; provenance recorded on every new `staffline_detections` row.

### Phase 2 — Text pipeline correctness (Weeks 2–3)

**Goal:** text results are explainable, failures are loud, re-runs are possible.

| # | Task |
|---|---|
| 2.1 | Loud stub mode (TX-1): text-service `/health` reports recognition-model presence + path; stub check on **every** mode incl. batch; stub state persisted into `text_alignments.log_text` (batch stops writing `""`); `dev.sh` preflights the Tridis model and warns; UI surfaces "OCR ran in stub mode" on affected alignments. |
| 2.2 | Key mask/music lookups by `image_id` (TX-2); decide `image_name` uniqueness constraint as backstop (decision log). |
| 2.3 | Fix failure asymmetry (TX-3): catch `TimeoutError`/`OSError` in `text_api`'s stream loop; unify semantics — per-image text failure = logged, *persisted*, job continues, job result marked partial (not silently "succeeded"); add a heartbeat around text-finding like paco's 20 s one so the 90 s stale-killer can't fire on healthy jobs. |
| 2.4 | Supported re-run (TX-4): supersede-or-delete semantics for `text_alignments` (mirror the `annotation_id`-currency pattern), endpoint + UI affordance; `has_text_alignment` skip becomes "skip unless stale or forced". |
| 2.5 | Reconcile annotation writers (TX-5): text-batch either reuses existing annotations (`has_annotation` check) or upgrades to the same medieval path predict uses; a re-annotate invalidates dependent text alignments (staleness guard). |
| 2.6 | Remove dead filter code, fix hardcoded `0.30` debug threshold + `lines_pre_filter` naming + stale docstrings (TX-6). |
| 2.7 | Decide + document empty-mask degradation (TX-8): keep the fallback but log it persistently and surface it in the alignment record. |
| 2.8 | Fix batch failure-index bug (TX-10). |
| 2.9 | Custom-model path story (TX-11): mount `stored_models` on text-service, or ship bytes via `job_uploads`, or scope custom text models out of alpha (decision log). |
| 2.10 | Route tests for `text_api`/`batch_api` failure paths (error event, timeout, stub) — first tests these modules have. |

**Exit criteria:** a text-service outage or stub state is visible in the UI and in persisted rows; a killed text-service cannot fail a healthy job via staleness; re-running text on an image is a supported action.

### Phase 3 — Scaffolding removal (Weeks 3–4)

**Goal:** the SAFE list is gone, the REPLACE list is replaced, every DECIDE item has a logged decision.

- 3.1 Execute the SAFE list (§2.3) — one reviewed PR, plus `.gitignore`/docs touch-ups.
- 3.2–3.9 Implement the REPLACE list (§2.3), one PR per row, in this order: auth on encode routes → `MOTHRA_SECRET` fail-fast → CORS → periodic cleanup (absorbing the triplicated 86400 sweeps) → `/healthz` + probes → shared-storage/Postgres for `BATCH_DIR`/`NEON_MANIFESTS_DIR` → migration Job + import-time side-effect removal → Neon save-race fix.
- 3.10 Walk the decision log (§4) in a triage session; convert each decision to an issue (implement/keep-documented/delete).

**Exit criteria:** grep sweep for the SAFE items returns nothing; every REPLACE row demoed; decision log has no `open` rows.

### Phase 4 — Data integrity, consistency & **central configuration** (Weeks 4–5)

**Goal:** one declaration site per fact; storage semantics uniform and staleness-guarded.

**Central-config workstream (priority focus):**

- 4.1 Literal inventory: every constant existing in ≥2 places — the TX-7 quad-duplication; class-ID maps; conf thresholds (`inference_api.py` + `useInferenceSettings.ts`); timeouts (`text_api`/`batch_api`/`cantus_api`/paco); 86400 cleanup constant; image-resize thresholds (`imageResize.ts`); stale-job timeout; box conventions.
- 4.2 Hoist each into the config layer: extend `landing-page/scripts/config.py`/`config.yaml` as the authoritative store (env-overridable, as today). For frontend-shared values, add `GET /api/config` (or a generated constants module consumed by Vite) so the UI reads the numbers the backend enforces — no re-declaration.
- 4.3 Give `text-service` and `paco-classifier-service` the same `config.py` + YAML + env-override pattern; per-service README config table.
- 4.4 CI guard: lint/grep check that fails when a known-centralized constant reappears as a literal (simple denylist file next to config.yaml).

**Data-integrity workstream:**

- 4.5 Per-table semantics doc + enforcement: `annotations` replace / `staffline_detections` accumulate-with-currency-key / `text_alignments` accumulate→supersede (from 2.4); every consumer checks currency (`annotation_id` pattern) — no "latest row" reads without a guard.
- 4.6 `image_name` uniqueness or full `image_id` migration (from 2.2 decision).
- 4.7 Box-convention note at the `text_api` seam (TX-9) or unify to one convention.

**Exit criteria:** every parameter named in this register traces to exactly one declaration site (spot-check script); CI guard active; semantics doc merged.

### Phase 5 — Test & CI hardening (Weeks 5–6)

- 5.1 Golden-page parity test (from 1.10) required in CI.
- 5.2 FastAPI route tests: auth/ownership on every router first (currently zero route tests), then the failure paths from 2.10.
- 5.3 text-service + paco-classifier-service test suites (currently none).
- 5.4 Frontend: vitest runner + smoke tests for AppRouter view switching and `apiFetch` refresh flow (currently zero frontend tests; CI only type-checks).
- 5.5 Re-enable eslint in CI: burn down or baseline the ~197 pre-existing errors (decision log).
- 5.6 Make `tests.yml` a **required status check** on `main` (repo-admin action, GitHub UI).
- 5.7 k8s: raise text-service memory limit above the documented-insufficient 2 Gi; wire `/healthz` probes (from 3.x); revisit the four-invocation pytest workaround (`sys.modules` leaks) with proper teardown.

**Exit criteria:** a PR cannot merge to `main` with red tests; every service has at least a smoke suite.

### Phase 6 — Repo professionalization, docs, process, release readiness (Week 6+)

**Goal (explicit end-state):** an organized, professional repo, clean and clear to navigate and learn.

- 6.1 Top-level restructure proposal: product code (`landing-page/`, `staff-finding/`, `text-service/`, `paco-classifier-service/`, `k8s/`) clearly separated from research/experiment areas; legacy dirs (`OLD-annotator/`, root `scripts/`, `configs/`) archived to a labeled home or deleted per decision log; fix the malformed `"data ` directory entry.
- 6.2 Artifact register execution (policy §5.8): triage the ~570 checked-in artifacts (`data/`, `training_outputs/`, `inference_runs/`, `inference-outputs/`, `summer26_samples/`, `models/`) into **unneeded** (delete) or **upload-to-HF-then-delete-locally** (confirm on https://huggingface.co/DDMAL-lab, record repo/path, then delete from git in a dedicated PR).
- 6.3 Root `README.md` mapping the whole repo: what each directory is, product vs research, where config lives, how to run — one hop to anything.
- 6.4 Per-service READMEs on one template: purpose, run, config table (from 4.3), API surface.
- 6.5 One documented docs layout consolidating `documentation_allons-y/` + `staff-finding/dox/` conventions: ADRs, status docs, and plans each have a defined place.
- 6.6 Naming/style pass: superseded code is deleted, not suffix-renamed (`run_pageOG.py` goes away per its decision); no orphan drivers.
- 6.7 Correct CLAUDE.md's stale claims; keep it current as the contributor-facing architecture guide.
- 6.8 ADRs for every decided item; alpha release checklist (deploy runbook, secrets checklist, known-limitations page for testers).

**Exit criteria:** newcomer test — someone unfamiliar locates any config value, service, or doc from the root README in one hop; CLAUDE.md audit clean.

---

## 4. Decision log

Every DECIDE item. Status: `open` → `decided (YYYY-MM-DD)` → issue link.

| # | Item | Options | Recommendation | Decider | Status |
|---|---|---|---|---|---|
| DL-1 | `SKIP_PREDICT`/`SKIP_YOLO`/placeholder-bbox-grid triad (`AppRouter.tsx:32-44,287,301-304`, `config.py:30-35`, `ic_api.py:67-120`) — fabricated 6×8 neume grid reaches the real IC | keep as dev-only; gate the grid behind env; remove entirely | Keep the env flags for dev machines, but the fabricated grid must never render without an unmissable "SYNTHETIC — no prediction ran" banner in the IC; alternatively block IC entry without a real annotation row | kyrie | open |
| DL-2 | `ICCompletionTestPage` + hardcoded "ta-da!" copy + `ic-completion` view | rewrite as a real completion view; remove and re-route step 2 | Rewrite: the XML-upload affordance is genuinely useful; the unconditional celebration copy goes | kyrie | open |
| DL-3 | `paco-classifier` pinned to personal branch `gianna/calvo-training-script` (ships production weights) | merge to a stable branch/tag upstream; keep pin with policy | Get calvo work merged upstream to `main`/a tag, then re-pin; until then, freeze the pin (no `--remote` updates) and note it in CODEOWNERS | DDMAL + gianna | open |
| DL-4 | ~570 checked-in artifacts | see §5.8 HF-first policy | Triage per §5.8; nothing new to LFS | kyrie | open |
| DL-5 | eslint disabled in CI (~197 errors) | burn down; baseline + ratchet | Baseline now (fail on *new* errors), burn down as a rolling chore | team | open |
| DL-6 | `tests.yml` not a required check | make required | Make required as soon as the suite is stable (Phase 5.6) | repo admin | open |
| DL-7 | `run_pageOG.py` + its test + dedicated CI step | delete; keep | Delete once parity harness covers the comparison need; superseded code isn't renamed, it's removed | kyrie | open |
| DL-8 | Unauth'd `GET /jobs/{id}/stream` (EventSource header limitation; 32-bit ids) | signed short-lived stream token in query; keep with longer ids | Signed one-time token minted at kickoff, passed as query param | kyrie | open |
| DL-9 | `bgr_adapter.py` hardcoded machine paths (BGR fully deferred) | env-var + lazy import; delete | Env-var (`MUSCRAT_LAYER_SEP_DIR`) + lazy import so nothing load-bearing sits on developer paths | kyrie | open |
| DL-10 | ProcessingPage cosmetic pacing (log drip-feed, 4 s completion delay) | keep as UX; remove | Remove the 4 s delay; keep at most a short (<1 s) settle animation | kyrie | open |
| DL-11 | "printed text coming soon" disabled radio + backend 400 | keep as roadmap signal; hide | Keep — it's honest roadmap signaling; tidy the copy | kyrie | open |
| DL-12 | Legacy dirs `OLD-annotator/`, root `scripts/`, `configs/` | archive branch; separate research repo; delete | Move still-useful ML scripts to a `research/` area (Phase 6.1); delete the rest; OLD-annotator deleted (tagged release preserves history) | kyrie | open |
| DL-13 | Null `confidence` placeholder in JSOMR output (`run_page.py:770`) | compute it; drop the field | Compute from fit quality once defined; until then keep null but document | kyrie | open |
| DL-14 | Stave conf default: eval-matched 0.25 vs product 0.5 (SF-1/1.3) | align to 0.25; keep 0.5 and re-evaluate at 0.5 | Align product to whatever the golden-set eval validates; one number, one declaration | kyrie | **decided 2026-08-17** — aligned to 0.25 (`yolo_inference.DEFAULT_STAVE_CONFIDENCE`); see SF-1 |
| DL-15 | `image_name` uniqueness constraint (2.2/4.6) | constraint + migration; id-only keying everywhere | Both: key by id, add the constraint as a backstop | kyrie | open |
| DL-16 | Custom text models in containers (TX-11) | mount volume; ship bytes; out of alpha scope | Out of alpha scope unless a tester needs it; log clearly when a custom path fails to resolve | kyrie | open |

---

## 5. PM / senior-dev operating recommendations

1. **Single source of truth per fact — the #1 structural rule.** The mismatch happened because defaults live in 4 places and image identity in 5. Any constant, default, path, or URL crossing a file or service boundary is declared exactly once; frontend-shared values come from `GET /api/config`/generated constants; enforced by the Phase 4.4 CI guard.
2. **Provenance is non-negotiable at alpha.** Every stored detection row records model hash (staves already do — extend), image-source variant, conf threshold, package version. Debugging "unusual results" without provenance is what cost weeks here.
3. **Golden datasets are first-class assets.** MS234_64 + Gent right become named, owned fixtures with expected-output tolerances in CI; new manuscripts added deliberately; GT provenance tracked (`gt_source`) to avoid circular evaluation.
4. **Fail loud policy.** No bare `except: pass`; no silent degradation (stub mode, empty mask, tier fallthrough). Degrading is allowed; degrading *quietly* is not — every fallback emits a persisted, user-visible signal.
5. **Ownership.** CODEOWNERS mapping: landing frontend · landing backend/jobs · staff-finding package · text-service + mothra-text · paco-service + weights · k8s/CI. Submodule pin policy: pins move only via reviewed PR; no personal-branch pins for anything shipping weights (DL-3).
6. **Branch protection.** `tests.yml` required (DL-6); golden-parity test in it; PR template with "provenance impact" and "phase/issue" fields.
7. **Cadence, fitted to the joint scrum-kanban system.** Phases map to sprint goals (milestones); day-to-day flow stays on the `mothra` kanban board; weekly triage of the decision log + findings register doubles as backlog refinement; phases exit by demo, not calendar. WIP discipline: Phase 1 parity work finishes before Phase 3 removals churn the same files.
8. **Repo hygiene — HuggingFace is the artifact home.** DDMAL keeps datasets/weights/artifacts at https://huggingface.co/DDMAL-lab. Checked-in experiment artifacts get triaged into exactly two buckets: **unneeded** (delete from git, no upload) or **upload-to-HF-then-delete-locally** (verify uploaded, record HF repo/path in the artifact register, then remove from git in a dedicated PR — deletion only after upload is confirmed). Nothing new moves to Git LFS; LFS remains only for the two bundled medieval `.pt` checkpoints the app loads at runtime. `e2e_tests` baselines stay small, named, and in-repo since CI depends on them.
9. **Deploy discipline.** Redeploy the stack together (documented text-service/worker stale-connection incident); post-deploy smoke = one golden-page predict.

---

## 6. Verification, ongoing

- **Harness gate:** no SF fix merges without a before/after harness run attached to its PR.
- **Register audit:** at each phase exit, every claim in §2 re-checked against HEAD; rows closed with links.
- **Doc freshness:** CLAUDE.md and this plan corrected in the same PR that changes the behavior they describe.
- **Test commands** (as CI runs them): `pytest staff-finding/scripts/script_tests` per-file + `pytest landing-page/scripts/tests`; frontend `npx tsc -b` (+ vitest once 5.4 lands).
