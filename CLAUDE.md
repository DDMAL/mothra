# Mothra — Claude Code Guide

## What this project is

YOLO-based Optical Music Recognition (OMR) for medieval manuscripts, developed at DDMAL (McGill). It challenges the existing Rodan pipeline by replacing multi-stage pixel-level analysis with end-to-end YOLO object detection.

Distinct parts live in this repo:
- **`landing-page/`** — The primary active web application (React + FastAPI)
- **`staff-finding/`** — Staffline detection (component filtering, centerline fitting, stave grouping); packaged as a local pip distribution and consumed by `landing-page/`'s predict pipeline — see **Staffline detection** below
- **`configs/`, `data/`, `OLD-annotator/`** — ML pipeline experiments and legacy tooling (less active)

---

## Landing page architecture

### Frontend
- **React 19** + **TypeScript** + **Vite** (port 5173 in dev)
- **Tailwind CSS v4** via `@tailwindcss/vite` — no CSS modules, no inline styles
- **No React Router** — navigation is a `view` string switched in `AppRouter.tsx`
- **pdf.js** (`pdfjs-dist`) for PDF → PNG page conversion on upload

Theme colours: `#1D3335` (dark teal, primary bg/text), `#4AADAA` (accent), `#C8E6E3` (light teal)

### Backend
- **FastAPI** + **uvicorn** (port 8001)
- **PostgreSQL** via `psycopg2` — hosted on [Neon](https://neon.tech), connection string in `.env`
- **bcrypt** for password hashing, **python-jose** for JWT (HS256, 72h access token expiry)
- **Refresh tokens**: a separate, longer-lived (30-day) opaque token, hashed (SHA-256) and stored
  in the `refresh_tokens` table, issued alongside the access token at login/register. `POST
  /api/auth/refresh` (sent via `X-Refresh-Token` header, not `Authorization`) verifies it against
  the table and mints a new access token + a new refresh token, revoking the old one (rotation on
  every use). `POST /api/auth/logout` revokes the current refresh token. This replaces the old
  `/api/auth/refresh`, which depended on `get_current_user` and so could never actually help once
  the access token had expired — the exact case it existed to handle.
- Vite proxies `/api`, `/neon`, `/Neon-gh` → `localhost:8001` in dev

### Database schema (auto-created on startup in `auth_api.py:init_db`)
| Table | Purpose |
|---|---|
| `users` | id, username, email, first/last name, password_hash |
| `projects` | id, user_id, name, steps_unlocked, used_image_names, used_model_names, deleted_at, last_opened_at, is_pinned |
| `project_images` | stores image bytes as `BYTEA` |
| `project_models` | model name references only (no file stored) |
| `mei_files` | xml_content as TEXT, corrected flag, `created_at` (used to pick each image's latest MEI revision for the cantus-bundle export) |
| `activity_log`, `project_logs` | audit trail per project |
| `annotations` | YOLO detections per image (`yolo_txt`), written by the predict job |
| `text_alignments` | text-finding output per image, written by the predict job |
| `staffline_detections` | per-image JSOMR staffline detections (`jsomr_json`), written by the predict job — accumulate-forever, unlike `annotations`'s delete-then-insert; see **Staffline detection** below |
| `jobs` | one row per predict/encode-upload/encode-batch/text-batch job; `status` ∈ `pending/running/succeeded/failed/cancelled`; `params JSONB` stores the exact kickoff kwargs (needed for retry), `retry_of`/`attempt` track retry lineage — see **Job queue** below |
| `job_events` | ordered progress-event log per job (`job_id`, `payload JSONB`) — what `GET /api/jobs/{id}/stream` polls |
| `job_uploads` | raw XML/image bytes staged for a Celery task to pick up, keyed by a short-lived `upload_id` |
| `job_sessions` | encode-job output (`mei_bytes`, `stem`, `manifest`) — replaces the old in-memory `_sessions` dict + `MANIFEST_DIR` tempfiles |
| `refresh_tokens` | `user_id`, `token_hash` (SHA-256 of the raw token), `expires_at`, `revoked_at` — backs the real JWT refresh flow, see **Backend** above |

Schema is migrated forward via `_migrate_db()` in `auth_api.py`. New columns go in the
`_ADDED_COLUMNS` list — `(table, column, definition)` tuples replayed as
`ALTER TABLE ... ADD COLUMN IF NOT EXISTS` on a single pooled connection, with a
`DuplicateColumn` catch kept only as a race safety net (`IF NOT EXISTS` isn't atomic across
concurrent sessions, and backend + worker migrate independently at import).

**`IF NOT EXISTS` is load-bearing, not cosmetic.** These were once bare `ADD COLUMN`s using
`except DuplicateColumn` as control flow. Postgres logs a server-side `ERROR` for a failed
statement *before* the client's exception handler ever sees it, so every pod start wrote ~26
false `ERROR`s into the database log — ~1,900 accumulated lines in production, enough that a
real error would have gone unnoticed. Anything added here must be idempotent *without raising*.

That has a sharp edge for any migration that **backfills data**: `mei_files.created_at`'s
`ADD COLUMN` used to raise on every run after the first, and that raise — aborting the
transaction — was the only thing stopping its backfill from re-running. It now carries an
explicit `created_at IS NULL` guard. A backfill added without such a guard will silently
re-run on every backend and worker start and overwrite live data.

---

## Running locally

**Six processes run together** — the frontend, its backend, the Interactive Classifier service (the IC step iframes it), the text-finding service, the staffline/background layer-separation service, and a Celery worker (predict/encode jobs). Redis must also be running as the Celery broker:

| Port | Process | Role |
|---|---|---|
| `5173` | Vite dev server (landing-page) | Open this in the browser; proxies `/api`, `/neon`, `/Neon-gh` → `:8001` |
| `8001` | landing-page FastAPI (`uvicorn`) | `/api/*`; also reaches IC/text-service/paco-classifier-service server-to-server |
| `8000` | IC service (`ic/` submodule) | IC REST API **and** the built IC SPA (served single-origin) |
| `8002` | text-finding service (`text-service/`) | wraps the `mothra-text` pipeline; called from `:8001` |
| `8003` | staffline classifier service (`paco-classifier-service/`) | wraps the `paco-classifier/` submodule's TensorFlow layer-separation model; called from `:8001`'s worker — see **Staffline detection** below |
| — | Celery worker | runs `/predict`, `/encode-upload`, `/encode-batch`, `/text-batch/run` jobs; no port of its own |

Redis (default `redis://localhost:6379/0`, override via `CELERY_BROKER_URL`) is
**only the Celery task broker** — it holds no application state. All job
status/progress/results live in Postgres (`jobs`/`job_events`/`job_uploads`/`job_sessions`,
see schema above), so restarting Redis doesn't lose any job history.

### One command (recommended)

```bash
./dev.sh          # starts all six; Ctrl-C tears all six down cleanly
./dev.sh -b       # rebuild the IC frontend bundle first (do this after the
                  #   ic/ submodule's frontend changes — else the iframe
                  #   serves a stale build)
./dev.sh -f       # free the ports first if something is stuck on them
./dev.sh -h       # help
```

Ports are overridable (`WEB_PORT=3000 ./dev.sh`). The script assumes the venvs
already exist (`ic/api/.venv`, `landing-page/scripts/.venv`, `text-service/.venv`,
`paco-classifier-service/.venv`) and `landing-page/node_modules` is installed —
it runs them, it doesn't create them. It also soft-warns (doesn't block) if
Redis isn't reachable at `CELERY_BROKER_URL`.

`paco-classifier-service/.venv` uses the same `requirements.txt` as
Docker/CI — no separate macOS requirements file needed. It pins
`tensorflow==2.15.1` (not `2.13.1`, which is what `Paco_classifier`'s own
`newrequirements.txt` recommends): `2.13.1` hard-pins `typing-extensions<4.6.0`
on **every** platform (confirmed via PyPI metadata directly — base package,
`tensorflow-cpu-aws`, `tensorflow-intel` all carry the same bound), which no
`fastapi>=0.110.0` can satisfy, so it fails identically on macOS, Linux/amd64,
and Linux/arm64. `2.15.1`'s `typing-extensions` requirement has no upper
bound at all, and it has a real native `macosx_11_0_arm64` wheel directly
under the plain `tensorflow` package name — the separately-named
`tensorflow-macos` package is vestigial for versions this recent (an empty
dispatcher shim with no actual `tensorflow` module inside — confirmed by
actually installing it, not assumed). `2.15.1` still predates TF 2.16's
default switch to Keras 3, so `.h5` loading behavior should match `2.13.1`'s
(both Keras-2 line):
```bash
cd paco-classifier-service && python3.10 -m venv .venv \
  && source .venv/bin/activate && pip install -r requirements.txt
```

### Manual (one terminal each)

```bash
# Terminal 1 — IC service (:8000)
cd ic/api && HOST=127.0.0.1 PORT=8000 .venv/bin/ic-api

# Terminal 2 — text-finding service (:8002)
cd text-service && .venv/bin/uvicorn main:app --port 8002

# Terminal 3 — staffline classifier service (:8003)
cd paco-classifier-service && .venv/bin/uvicorn main:app --port 8003

# Terminal 4 — landing-page backend (:8001)
cd landing-page/scripts && source .venv/bin/activate
uvicorn main:app --reload --port 8001

# Terminal 5 — Celery worker (predict/encode jobs)
cd landing-page/scripts && source .venv/bin/activate
celery -A celery_app.celery_app worker --loglevel=info --pool=threads --concurrency=2

# Terminal 6 — landing-page frontend (:5173)
cd landing-page && npm run dev
```

Open `http://localhost:5173`. All `/api/*` calls proxy to the backend automatically.

**Environment** — `landing-page/scripts/.env` (secrets only; non-secret paths/URLs
live in `landing-page/scripts/config.yaml`, see **Configuration** below):
```
DATABASE_URL=postgresql://...   # Neon connection string
MOTHRA_SECRET=...               # JWT signing key
```
Optional env var overrides (all have working `config.yaml` defaults):
`IC_API_URL`, `IC_PUBLIC_URL`, `TEXT_API_URL`, `PACO_API_URL`, `CELERY_BROKER_URL`.

### Configuration (`config.py` / `config.yaml`)

`landing-page/scripts/config.yaml` centralizes non-secret paths (`MODELS_DIR`,
`NEON_MANIFESTS_DIR`, `MOCK_DATA_DIR`, `MEDIEVAL_MODELS_DIR`) and service URLs
(`IC_API_URL`, `IC_PUBLIC_URL`, `TEXT_API_URL`, Celery's `broker_url`) that used
to be scattered as inline `Path(__file__).parent / "..."` literals or
`os.environ.get(..., "http://localhost:PORT")` defaults across several files.
`config.py` loads the YAML and lets any matching env var override it — this is
what makes it possible to re-point services at container hostnames in Docker
(see **Deployment** below) without editing source. `DATABASE_URL`/`MOTHRA_SECRET`
are deliberately **not** in `config.yaml` — they're secrets and stay in `.env`/environment only.

### Job queue (Celery + Postgres)

`POST /api/projects/{id}/predict`, `POST /api/encode-upload`,
`POST /api/encode-batch`, and `POST /api/projects/{id}/text-batch/run` no
longer stream results directly — each returns `{"job_id": ...}` immediately
after enqueuing a Celery task (`tasks_predict.py` / `tasks_encode.py` /
`tasks_text_batch.py`), and the frontend connects separately to
`GET /api/jobs/{job_id}/stream` to watch progress. That endpoint
(`jobs_api.py`) polls the `job_events` table and re-emits the same
`data: {...}\n\n` SSE frames the old in-request generator used to `yield` —
`ProcessingPage.tsx` didn't need to change at all, only the two-step
kickoff-then-stream wiring in `AppRouter.tsx` (via `apiFetchJobStream()` in
`lib/apiFetch.ts`, which also reports the kickoff's `job_id` back via an
`onJobId` callback so the frontend can cancel/retry it later). Redis is purely
the Celery broker; Postgres (`jobs`/`job_events`) is the single source of
truth for job status, so there's no Celery result backend configured. Known
gaps: no periodic cleanup of `job_uploads`/`job_sessions` rows yet (see the
retry note below — this got slightly more important, not less), and a dead
worker is detected via a ~90s staleness timeout rather than immediately.

**Job cancellation** (`POST /api/jobs/{id}/cancel`, `jobs_api.py`) calls
`celery_app.control.revoke(job_id, terminate=True)` **and** flips a
cooperative flag other tasks check between iterations
(`job_store.check_cancelled()`, raises `JobCancelled`). Both matter: because
the worker runs `--pool=threads` (see below), `terminate=True` cannot actually
kill an in-flight task — there's no child OS process to signal, only a
thread. `revoke()` alone only stops a task that hasn't started yet (workers
check a broadcast revoked-id set before picking one up). For a task already
running, the cooperative check inside `tasks_predict.py`'s and
`tasks_text_batch.py`'s per-image loops (and `tasks_encode.py`'s per-item
batch loop) is what actually stops it. If you add a new long-running task,
it needs its own `check_cancelled()` call in its loop or cancellation will
silently no-op for it once it's running.

**Job retry** (`POST /api/jobs/{id}/retry`) replays a `failed` job's exact
original kickoff kwargs (persisted in `jobs.params` at kickoff time) as a new
job, linked via `jobs.retry_of`/`jobs.attempt`. This is deliberately separate
from `ProcessingPage.tsx`'s older `retryKey`-based "restart" button, which
just re-invokes `streamRequest` client-side with freshly-collected params —
"retry" (server-tracked, same params) and "restart" (client-side, re-collects
params) now coexist as distinct buttons. For `encode_upload`/`encode_batch`
retry to work at all, `tasks_encode.py` had to stop dropping staged
`job_uploads` rows in a blanket `finally` — they're now only dropped on the
success path, so a failed job's XML/image bytes survive long enough to be
retried. **This is the other half of the "known gap" above**: those rows now
leak indefinitely for jobs that fail and are never retried, since there's
still no TTL/cleanup job for `job_uploads`.

**The worker must run with `--pool=threads`, not Celery's default `prefork`.**
`prefork` works by `fork()`-ing a child process per worker slot; PyTorch (and
other native BLAS/OpenMP-using libraries pulled in by `ultralytics`) is not
fork-safe, and a `predict.run` task will segfault the forked child almost
immediately (`WorkerLostError: Worker exited prematurely: signal 11 (SIGSEGV)`)
the instant it touches a loaded YOLO model — confirmed by actually running a
predict job locally, not just by inspection. `--pool=threads` runs tasks in
threads within one process instead, avoiding the fork entirely; concurrency is
still real since PyTorch/numpy release the GIL during actual tensor ops.

### Staffline detection

`tasks_predict.py`'s per-image loop runs a staffline-detection stage right
after YOLO produces stave-class ("staves", merged slot 2) boxes for that
image: connected-component filtering → Huber-robust centerline fit →
duplicate/fragment reconciliation → stave grouping, via
`landing-page/scripts/staffline_stage.py`. This wraps the `staff-finding/`
module — a separate, actively-developed staffline-detection codebase in this
repo (see `staff-finding/dox/STATUS.md` for its own design notes and ADRs) —
packaged as a local pip distribution (`staff-finding/pyproject.toml`,
installed via `pip install -e staff-finding/`; only the six algorithmic
modules are packaged, not its CLI drivers).

Results are stored per-image in `staffline_detections` (schema above) as a
JSOMR-shaped JSON array — one record per detected line, with its bounding
box, fitted centerline, and stave assignment. `tasks_encode.py`'s
`_resolve_hints()` now resolves stave positions through a 3-tier fallback:
`staffline_detections` (via `staffline_adapter.py`'s `staves_from_jsomr()`,
richer per-line curve fits) → `annotations.yolo_txt` (`parse_yolo_stave_hints()`,
the older geometry-only heuristic) → glyph-position clustering
(`estimate_staves_from_glyphs()`). Any project/image without a
`staffline_detections` row (predates this feature, or detection failed/found
nothing) falls through unchanged to the pre-existing behavior.

**Medieval preset only:** before the stave `.pt` model runs, the page image
is also sent to `paco-classifier-service` (`landing-page/scripts/paco_api.py`'s
`classify_stafflines()`, a plain-`urllib` HTTP bridge — see **Running
locally** above for why this is its own service rather than an in-process
import), which wraps the `paco-classifier/` submodule
(`DDMAL/Paco_classifier`, branch `gianna/calvo-training-script`) — a
TensorFlow auto-encoder that splits the page into a background-only layer
and a stafflines-only layer (both transparent PNGs). The weights,
`paco-classifier/models_v4/{model_0.h5,model_1.h5}` (`model_0`=background,
`model_1`=stafflines), live **inside the submodule itself** — checked into
`DDMAL/Paco_classifier`, not bundled/Git-LFS-tracked separately in this
repo, unlike the medieval `.pt` checkpoints. Bumping to a retrained pair is
therefore a `git submodule update --remote` (or re-pointing the submodule
at a new commit), not a file swap in this repo; `main.py`'s
`_resolve_staffline_models_dir()` also honors a `STAFFLINE_MODELS_DIR` env
override for pointing at an unreleased pair without touching the submodule
pin at all. The stave model then runs on the **stafflines-only layer**, not
the raw page, so `staffline_stage.py`'s component-filtering/centerline-fitting
crops from that cleaner signal instead of raw parchment texture.
Implemented in `tasks_predict.py`'s `_run_medieval_inference()`: the
classifier-then-stave-model pass runs in a background `threading.Thread`
concurrent with the main thread's text/music YOLO pass (the classifier's
TF inference is the slow half of a predict request) — deliberately kept off
the task's own shared `cur`/`con` connection, since nothing about this
pattern has ever needed cross-thread DB access before. If
`paco-classifier-service` is unreachable or errors, it falls back to
running the stave model on the raw page image (today's pre-existing
behavior) rather than failing the job. **Known accepted gap:** an image
whose annotation row was first written by `tasks_text_batch.py` (unaffected
by this — it still calls `YoloModelSet.infer()` on the raw page) stays on
raw-page stave boxes permanently under `tasks_predict.py`'s
`has_annotation`-reuse skip, unless that row is deleted and predict is
re-run — same class of staleness already accepted for `job_uploads`/
`job_sessions` (see **Things that don't exist yet** below).

Deliberately deferred for now (each is a caller-controlled parameter or a
swappable stage, not a hardcoded limitation, so enabling any of them later
needs no re-plumbing):
- **Ink-separation ("BGR")** — `staff-finding/scripts/bgr_adapter.py` wraps an
  external, unvendored `muscrat/layer_sep` repo reachable only via hardcoded
  paths on specific developer machines (no tests, no Docker/LFS story yet).
  This is a *different* layer-separation model than `paco-classifier-service`
  above (muscrat/layer_sep vs. Paco_classifier) and remains fully deferred —
  `staffline_stage.py` itself still runs directly on whatever crop it's
  handed (raw page, or now the Paco-classifier's stafflines layer); Sauvola
  binarization (`component_filter.py`'s tuned default) is the interim
  mitigation for faint ink either way.
- **`interpolate_staves.py`** (gap-fill/edge-extrapolation for missing lines)
  stays off (`interpolate_missing=False`) per `staff-finding/dox/STATUS.md`'s
  "not yet validated across the corpus" caveat.
- **`fallback_redetect.py`** (re-probing under-populated staves with a second
  YOLO pass) isn't wired up — it needs an already-loaded stave-detector model
  instance, which custom (non-medieval-preset) models don't have one of.

### Deployment (Kubernetes, CI/CD via GitHub Actions)

**Merging to `main` deploys production automatically. Staging is deployed on
demand from any branch** — Actions → `ci-cd` → Run workflow → pick the branch →
leave `environment: auto`. Staging is deliberately *not* push-triggered: it's a
single shared environment behind a ~25-minute three-image build, so having ~20
active branches auto-deploy into it would only thrash both.
`.github/workflows/build-images.yml` (job name `ci-cd`) therefore runs on push to
`main` only, plus `workflow_dispatch` from any ref:
1. **build** — builds `backend` (`landing-page/Dockerfile`), `ic` (`ic/Dockerfile`),
   and `text-service` (`text-service/Dockerfile`, build context is the repo root
   since it needs the sibling `mothra-text/` submodule) and pushes each to
   `ghcr.io/ddmal/mothra-{backend,ic,text-service}`, tagged by short SHA
   (`sha-<short>`), by branch, and `latest` **from `main` only** (both environments
   share these image repos, so an ungated `latest` would be whichever one pushed
   last). `worker` reuses the `mothra-backend` image, so it isn't built separately.
   Checkout pulls submodules recursively and Git LFS (the bundled medieval `.pt`
   weights — without `lfs: true` they'd check out as pointer stubs and inference
   would fail at runtime).
2. **resolve** — maps the ref to an environment: `main` → production, any other ref
   → staging. Outputs `dir` (`k8s` or `k8s/staging`) and `suffix` (`""` or
   `-staging`). `workflow_dispatch` takes an `environment` input
   (`auto`/`staging`/`production`, default `auto`) so staging can be redeployed from
   `main`; dispatching `production` from a non-`main` ref is refused.
3. **deploy** (needs `build` + `resolve`) — using the `KUBECONFIG` repo secret, pins
   `$dir/backend.yaml`/`worker.yaml`/`ic.yaml`/`text-service.yaml` to this commit's
   `sha-<short>` tag (and now *fails* if that `sed` matched nothing, since `sed`
   exits 0 on no-match and would otherwise ship a stale tag), applies those plus
   `$dir/configmap.yaml`/`ingress.yaml`, then `kubectl rollout status` on
   `backend{suffix}`/`worker{suffix}`/`ic{suffix}`/`text-service{suffix}`. redis,
   postgres, secrets and the PV/PVC are excluded from CD. `concurrency` is keyed on
   the resolved environment, so production and staging deploys don't block each other.

A dispatched run executes the **selected branch's** copy of the workflow and of
`k8s/staging/`, not `main`'s. That's what makes it possible to test manifest edits
on the branch that makes them, but it also means a branch cut before the staging
commit (`654f18e`) still carries the pre-staging workflow — no `resolve` job, so it
would deploy the *production* manifests. Merge `main` into a branch before
dispatching it.

**Two environments share the `mothra` namespace**, separated only by naming:
production manifests are in `k8s/`, staging's are in `k8s/staging/` with *identical
filenames*, and every staging object suffixes `-staging` onto its `metadata.name`,
`app` label **and selectors** (see `k8s/README.md`). Dropping that suffix from an
`app` label makes production's Service select a staging pod; leaving a staging
manifest pointing at `mothra-secrets`/`mothra-config` makes staging boot green
against the production database and `_migrate_db()` ALTER production's tables — both
fail silently, so treat the suffix as the invariant when editing these files.
Committed staging manifests carry the placeholder image tag `sha-0000000`, which is
deliberately not a real tag so a missed rewrite fails loudly.

**Postgres is not part of this repo's deploy** — each environment has its own
deployment in the `postgres` namespace, reached cross-namespace via `DATABASE_URL`
(`mothra-postgres.postgres.svc.cluster.local:5432` for production,
`mothra-staging-postgres.postgres.svc.cluster.local:5432` for staging; **same
database name `mothra`, only the host differs**) — so a Mothra deploy never touches
the database deployment. Ingress is Traefik: `mothra.simssa.ca` → backend and
`mothra-ic.simssa.ca` → ic for production, `mothra.staging.simssa.ca` /
`mothra-ic.staging.simssa.ca` for staging, each with a `Middleware` in its
`ingress.yaml` adding a `frame-ancestors` CSP on the IC host so the campus proxy's
blanket `X-Frame-Options: SAMEORIGIN` doesn't block the IC iframe. Staging also runs
its own `redis-staging` broker — a shared broker would let the two environments'
Celery workers steal each other's tasks off the default `celery` queue, and the
worker's `celery inspect ping` probes would still pass while that happened.
`stored_models` (locally-uploaded custom YOLO checkpoints, written by
`models_api.py`) is **not baked into the image** — it's a static NFS
PersistentVolume (RWX, `stored-models-pv.yaml`/`-pvc.yaml`) mounted on both
`backend` and `worker` so uploads persist across rollouts and are visible to
whichever service reads them; each environment gets its own PV on its own NFS path
(`/srv/nfs/mothra/…` vs `/srv/nfs/mothra-staging/…`), since a shared one would let a
staging delete destroy a production checkpoint.

**Manual apply** (redis/postgres/secrets/PV excluded from CD; apply by hand when needed):
```
kubectl apply -f k8s/secret.yaml -f k8s/configmap.yaml
kubectl apply -f k8s/stored-models-pv.yaml -f k8s/stored-models-pvc.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/ic.yaml -f k8s/text-service.yaml -f k8s/paco-classifier-service.yaml -f k8s/backend.yaml -f k8s/worker.yaml
kubectl apply -f k8s/ingress.yaml
```
Same commands with `k8s/staging/` for staging. `kubectl apply -f k8s/` does not
recurse into `k8s/staging/` (that needs `-R`), so the two can't be mixed up by a
directory-wide apply. Staging's one-time prerequisites (its Postgres deployment, the
NFS export, DNS/proxy vhosts, its Secret, and the first-boot ordering) are in
`k8s/README.md` — none of them are created by this repo.

**Known follow-ups** (from `k8s/README.md`): no real `/healthz` yet, so probes
are TCP/exec only; `init_db()`/`_migrate_db()` run at import, so keep
`backend`/`worker` at **1 replica** until a one-shot migration Job replaces
that (and on a *fresh* database, backend and worker racing that import-time
migration can `CrashLoopBackOff` once before self-healing — bring backend up first);
`text-service`'s `/batch-download/{id}` uses local disk keyed by
`batch_id`, so it needs shared storage (or a single replica) if batch
downloads are used at scale. Both `worker` Deployments are pinned to
`k3s-gpu-node-1` and share one MIG instance with no scheduler arbitration, so
concurrent inference in both environments can raise `torch.OutOfMemoryError` in
either one (not visible as `OOMKilled` — the memory limit covers host RAM, not VRAM).

The old `render.yaml` Render Blueprint (single-service, no worker, no Redis)
is retired — it predated the job queue and would never have processed
predict/encode jobs correctly. If a Render service was ever created from it,
that needs to be deleted/disconnected on Render's side too — removing the
file doesn't un-hook an existing deploy.

### Local/manual container runs (`docker-compose.yml`)

A `docker-compose.yml` at the repo root mirrors the same stack (redis + ic +
text-service + paco-classifier-service + backend + Celery worker) for local
testing without a cluster — the k8s manifests were modeled on it, not the
other way around. `worker` reuses the `backend` image, just overrides the
`command:`. Before building: `git submodule update --init --recursive`
(brings in `paco-classifier/models_v4/`'s weights along with the rest of
that submodule — no LFS involved) and `git lfs pull` (see above — same LFS
caveat applies locally for the medieval `.pt` weights specifically:
`git-lfs` must be installed and registered, `brew install git-lfs && git
lfs install`, *before* pulling, or those files silently stay as pointer
text). `DATABASE_URL`/`MOTHRA_SECRET` come from a root-level `.env`
(gitignored, separate from `landing-page/scripts/.env`) that Compose
auto-loads.

`staff-finding/` (a sibling directory of `landing-page/`, holding the
staffline-detection module `staffline_stage.py` imports — see **Staffline
detection** above) sits outside `landing-page/`'s own Docker build context.
`docker-compose.yml`'s `backend`/`worker` services and
`.github/workflows/build-images.yml`'s backend image build both pass it in as
a named BuildKit "additional build context" (`staff_finding`), which
`landing-page/Dockerfile` then `COPY --from=staff_finding`s into the image
before `pip install`-ing it (non-editable — the source tree is already
baked into the image via that `COPY` regardless, so `-e` would only add an
unused editable-install pointer back at itself; `staff-finding/.dockerignore`
keeps its large test-fixture/experiment directories out of what actually
gets sent to the Docker daemon). Skipping this wiring doesn't silently ship a
broken image — it fails the build itself: `docker compose build`'s
`backend`/`worker` targets and the GitHub Actions backend build both error out
at `COPY --from=staff_finding` with a missing-named-context failure, before
any image is produced.

**Requires the `docker-buildx` plugin (BuildKit).** Without it, `docker
compose build` prints `Docker Compose requires buildx plugin to be
installed` and silently falls back to the legacy builder, which does **not**
apply `landing-page/.dockerignore` correctly — the build context balloons to
several GB (it'll include `node_modules`, the Python `.venv`, the `neon/`
submodule's own `node_modules`/demo assets, and `scripts/stored_models`) and
builds take much longer than they should. Install with
`brew install docker docker-compose docker-buildx colima` (or Docker
Desktop, which bundles buildx already) and confirm with `docker buildx
version` before assuming a slow/huge build is a real problem rather than a
missing plugin.

For local Compose runs, `scripts/stored_models` is likewise excluded from the
image (`landing-page/.dockerignore`) and instead backed by a named volume
(`stored_models:/app/scripts/stored_models`) mounted on both `backend` and
`worker`.

**Resource requirements: at least 8GB RAM, 4 CPUs for the Docker host/VM.**
Confirmed by actually running a real predict job through the containers —
with the default Colima allocation (2GB), `text-service`'s Kraken/HTR
segmentation step got SIGKILL'd by the VM's OOM killer mid-request
(`docker compose ps` shows `Exited (137)`, easy to mistake for an
application bug rather than an OOM kill). `worker` (YOLO inference) and
`text-service` (Kraken segmentation + HTR) are the two memory-heavy
containers; `backend`/`ic`/`redis` are comparatively light. `paco-classifier-service`
(TensorFlow sliding-window inference over a full page) is likely a third —
not yet confirmed the same way as the other two, but its `k8s/` resource
requests/limits are deliberately set higher than `backend`/`ic` as a
starting assumption pending real usage data. If you see a service
unexpectedly exit with code 137 mid-job, check available memory before
debugging application code.

`text-service`'s recognition model (Tridis, via `htrmopo`) is baked into
the image at build time (both Compose and the CI-built image share the same
`text-service/Dockerfile`), matching local dev's one-time manual
`htrmopo get 10.5281/zenodo.10788590` step — without it, text-finding
silently runs in stub mode (segmentation/YOLO still work, OCR returns no
syllables, with a `"STUB — no recognition model installed"` log line as the
only sign). The Zenodo record currently serves the file as
`Tridis_v2_Medieval_EarlyModern.mlmodel`, which doesn't match the
auto-discovery glob (`mothra-text`'s `run_pipeline.py`) expecting the exact
name `Tridis_Medieval_EarlyModern.mlmodel` — `text-service/Dockerfile`
renames the file post-download to work around this; if a future Zenodo
update changes the filename again, re-check that rename step still matches.

**Prefer redeploying the whole stack together over recreating one service at
a time.** Recreating just `text-service` (e.g.
`docker compose up -d --no-deps --build text-service`) while `worker` keeps
running left `worker` holding a stale connection to the now-gone old
`text-service` container — its next task sat idle instead of failing fast,
occupying a worker thread until `worker` was manually restarted. Root cause:
`text_api.py`'s `_stream_multipart()` passed `urlopen(..., timeout=600)` —
a 10-minute *per-read* socket timeout, not a hard deadline, so a peer that
goes unreachable mid-connection can tie up a thread for the full 600s before
Python ever raises. Fixed by lowering that default to 120s (real single-image
text-finding calls complete in well under a minute; `batch_api.py`'s
multi-file batch call already passes its own explicit, much larger timeout
and is unaffected). 120s is still slower than ideal for this failure mode —
restarting dependents together after any redeploy (Compose or `kubectl
rollout restart`) remains the safer habit.

---

## Key files

| Path | Role |
|---|---|
| `landing-page/scripts/main.py` | FastAPI app, CORS, mounts routers |
| `landing-page/scripts/auth_api.py` | Auth endpoints incl. refresh-token issuance/rotation/logout, DB init (incl. job-queue + `refresh_tokens` tables), project CRUD, image storage |
| `landing-page/scripts/account_api.py` | Profile update, password change, account delete |
| `landing-page/scripts/projects_api.py` | Project CRUD, export/duplicate, activity/log-download endpoints |
| `landing-page/scripts/images_api.py` | Project image upload/fetch/delete endpoints |
| `landing-page/scripts/ic_api.py` | Bridges to the Interactive Classifier service — `POST /projects/{id}/ic/start` and related IC-step endpoints |
| `landing-page/scripts/inference_api.py` | `POST /projects/{id}/predict` kickoff endpoint (enqueues `tasks_predict.py`), annotation CRUD |
| `landing-page/scripts/mei_api.py` | MEI file CRUD, Neon batch-editor edit-session bootstrap |
| `landing-page/scripts/cantus_api.py` | Proxies Cantus source lookups (incl. `siglum`) to the text-service |
| `landing-page/scripts/model_validation.py` | Validates uploaded YOLO checkpoints, derives text/music/staves class maps |
| `landing-page/scripts/config.py` / `config.yaml` | Centralized non-secret paths + service URLs, env-var overridable |
| `landing-page/scripts/celery_app.py` | Celery app instance/config; entrypoint for `celery -A celery_app.celery_app worker` |
| `landing-page/scripts/job_store.py` | Postgres-backed job state: create/status/events (incl. `params`/`retry_of`/`attempt`), `check_cancelled()`/`JobCancelled` cooperative-cancellation helper, staged uploads, encode session/manifest storage |
| `landing-page/scripts/jobs_api.py` | `GET /api/jobs/{id}/stream` (polls `job_events`, re-emits SSE frames), `POST /api/jobs/{id}/cancel`, `POST /api/jobs/{id}/retry` |
| `landing-page/scripts/tasks_predict.py` / `tasks_encode.py` / `tasks_text_batch.py` | Celery tasks: YOLO inference / MEI-building / batch text-finding work, run out-of-request |
| `landing-page/scripts/staffline_stage.py` | Staffline detection stage (component filter → centerline fit → stave grouping), wraps the `staff-finding/` package; called from `tasks_predict.py`, writes `staffline_detections` — see **Staffline detection** above |
| `landing-page/scripts/staffline_adapter.py` | Converts `staffline_detections`' JSOMR records into `encode_to_mei.py`'s `StaveBbox` shape; used by `tasks_encode.py` |
| `landing-page/scripts/paco_api.py` | Bridges to `paco-classifier-service` (`classify_stafflines()`) — medieval-preset staffline/background layer separation, called from `tasks_predict.py`'s `_run_medieval_inference()`; see **Staffline detection** above |
| `landing-page/scripts/yolo_inference.py` | YOLO model loading/inference (`resolve_yolo_models`, `YoloModelSet`, incl. the split `infer_text_music()`/`infer_staves()` medieval-only methods) shared by the predict task and `batch_api.py` |
| `landing-page/scripts/encode_api.py` | `POST /api/encode-upload` / `/encode-batch` — kickoff endpoints, enqueue Celery tasks |
| `landing-page/scripts/encode_to_mei.py` | Core encoding logic: parse XML, estimate staves, build MEI, validate |
| `landing-page/scripts/batch_api.py` | `POST /text-batch/run` job-queue kickoff, `GET /text-batch/{id}/download`, `GET /sources/{id}/export`, and `GET /sources/{id}/cantus-bundle` (corrected-MEI zip for manual hand-off to `production_mei_files`) |
| `landing-page/src/lib/apiFetch.ts` | `apiFetch` (also drives the silent JWT-refresh-on-401 flow via `X-Refresh-Token`) / `apiFetchJobStream` — auth-aware fetch wrapper + job kickoff-then-stream helper, reports `job_id` via an `onJobId` callback |
| `landing-page/src/types.ts` | All shared TypeScript types |
| `landing-page/src/components/AppRouter.tsx` | All view routing (switch on `view` string) |
| `landing-page/src/hooks/useEncodingFlow.ts` | Encoding state: pending files, logs, MEI content |
| `landing-page/src/hooks/useProjectMutations.ts` | Project CRUD mutations |
| `landing-page/src/hooks/useAssetSection.tsx` | Shared state for grid tabs (selection, pagination, modal, drag) |

---

## Workflow pipeline (step order)

```
1. Create project, upload images ("use" selected images)
2. Interactive Classifier  →  view: "ic"
3. IC completion / upload XML output  →  view: "ic-completion"
4. Encoding (GameraXML → MEI via encode_to_mei.py)  →  view: "encoding-processing"
5. Neon.js batch editor for human correction  →  view: "neon-editor"
6. Send to Cantus Ultimus  →  downloads a corrected-MEI zip bundle (no dedicated view)
```

`stepsUnlocked` on the project record gates which steps are accessible. It increments as each step completes and is persisted via `PUT /api/projects/{id}`.

Step 6 is **not** a live push to Cantus Ultimus — the DDMAL/cantus (Cantus
Ultimus) repo has no write API (its DRF views are all `ListAPIView`/
`RetrieveAPIView`, GET-only). The real workflow there is manual: a maintainer
commits correctly-named MEI files into the separate `DDMAL/production_mei_files`
repo and runs `index_manuscript_mei` by hand. So "send to Cantus Ultimus"
(`AppRouter.tsx`'s `handleSendToCantus`) downloads a zip from
`GET /api/projects/{id}/sources/{sourceId}/cantus-bundle` — corrected MEI
files renamed `{siglum}_{folio}.mei` plus a `README.txt` with the exact
manual hand-off steps — rather than routing through any `ProcessingPage`/job
queue (it's a fast synchronous zip build, no Celery task needed). The old
`"sending"` view/animation and its no-op fake progress bar are gone.

---

## Frontend patterns

- **View routing**: add a new string literal to the `View` union in `types.ts`, then add a `case` in `AppRouter.tsx`
- **Upload handlers**: defined inline in `AppRouter.tsx`, passed down as props — `onUploadImage` calls `POST /api/projects/{id}/images` with `FormData`
- **Auth headers**: always use `authHeaders()` from `useAuth.ts` — it adds `Authorization: Bearer {token}` from localStorage
- **Image display**: use `<AuthImage>` for any authenticated image URL (fetches with auth and converts to blob URL)
- **Modals**: use `Modal.tsx` as the base (`fixed inset-0 z-50 bg-black/60` overlay pattern)
- **Grid tabs**: use `useAssetSection` hook for selection, pagination, drag state, and upload modal visibility

---

## Backend patterns

- All routers share `get_db_conn()` and `get_current_user()` from `auth_api.py` — import from there
- Image bytes stored directly in `project_images.data BYTEA` — no file system or S3
- `/predict`, `/encode-upload`, `/encode-batch`, `/text-batch/run` all run as Celery tasks (`tasks_predict.py`/`tasks_encode.py`/`tasks_text_batch.py`); kickoff endpoints validate synchronously, create a `jobs` row (with `params=` set to the exact kickoff kwargs, for retry), enqueue, and return `{"job_id": ...}` — see **Job queue** above
- Encode job output (`mei_bytes`, `stem`, `manifest`) lives in the `job_sessions` Postgres table, read by `GET /mei/{id}` / `GET /manifest/{id}` — no more in-memory `_sessions` dict or `MANIFEST_DIR` tempfiles

---

## Build

```bash
cd landing-page
npm run build        # builds Neon submodule first, then tsc + vite build
npm run lint         # eslint
npm run format       # prettier --write src/
```

The build also compiles the embedded Neon.js editor from the `neon/` submodule — this step requires `yarn` and sets `NODE_OPTIONS=--openssl-legacy-provider`.

---

## Testing

`.github/workflows/tests.yml` runs on every push, to any branch — the
staff-finding algorithmic suite (`staff-finding/scripts/test_group_staves.py`
+ `script_tests/`, as several separate `pytest` invocations rather than one
combined run — see the workflow's own comments for why) and
`landing-page/scripts/tests/`. It is not yet configured as a required status
check in the repo's branch protection settings for `main` — that is a
separate, repo-admin-level step, done in GitHub's own UI, not this file.

---

## Things that don't exist yet (planned)

- *Periodic* cleanup of `job_uploads`/`job_sessions` rows — `job_store.py` has real cleanup
  functions (`cleanup_stale_uplaods(max_age_days=1)` — note the typo — and
  `cleanup_stale_sessions(max_age_days=14)`, invoked from `main.py`), but they run **once at
  process start only**; there is no scheduler, so a long-lived pod never sweeps again. Rows
  still accumulate between restarts (and matter slightly more now: failed-but-not-yet-retried
  encode jobs keep their staged `job_uploads` rows around on purpose, so retry can still fetch
  them — see **Job queue** above)
- Health/status page — no way to check backend/Postgres/Redis/Celery-worker/IC/text-service
  liveness from the app; not implemented
- IIIF manifest import — no way to bulk-import project images from a IIIF manifest URL; not
  implemented (single-file/PDF upload via `images_api.py`'s `POST /projects/{id}/images` is
  still the only ingestion path)

## Things that have been implemented (no longer placeholders)

- **Job queue** — `/predict`, `/encode-upload`, `/encode-batch`, `/text-batch/run` all run as
  Celery tasks with Postgres-backed status/progress (`jobs`/`job_events`); see **Job queue** above
- **Job cancellation** — `POST /api/jobs/{id}/cancel` combines `celery_app.control.revoke()` with
  a cooperative in-task check, since the worker's `--pool=threads` config means `terminate=True`
  can't actually kill an already-running task; see **Job queue** above
- **Job retry** — `POST /api/jobs/{id}/retry` replays a failed job's stored `params` as a new,
  lineage-tracked job; see **Job queue** above
- **Real JWT refresh** — a separate, rotating, revocable refresh token (`refresh_tokens` table)
  replaces the old `/api/auth/refresh`, which depended on the very access token it was meant to
  refresh and so never worked once that token actually expired; see **Backend** above
- **Cantus bundle export** — the old mocked "sending" animation is gone; "send to Cantus Ultimus"
  now downloads a real zip of corrected MEI files (`GET /sources/{id}/cantus-bundle`) for manual
  hand-off, since Cantus Ultimus has no write API; see **Workflow pipeline** above
- **Batch encoding** — `POST /api/encode-batch` (`batch_api.py`/`tasks_encode.py`) handles multiple XML+image pairs in one job
- **Batch text-finding logs/activity parity** — `tasks_text_batch.py` now captures and persists
  per-folio `log_text` on each `text_alignments` row (scoped between `folio_result` boundaries, with
  the batch-global preamble landing on the first folio), matching `text_api.py`'s single-folio
  `/predict` path — previously hard-coded to `""`, which is why the Detected text viewer's "view
  logs" always showed "no logs recorded for this run". It also now calls `_log_activity(...,
  "text_batch_run", ...)` once per batch run, mirroring `tasks_predict.py`'s `"predict_run"` entry —
  previously batch text-finding runs left no trace in a project's `activity_log`/exported
  `activity_log.txt` at all. Note: neither of these feeds `GET /projects/{id}/logs/download`'s
  `encoding_logs.txt` (that file is unrelated — sourced from `project_logs`'s encoding-step rows,
  see **Workflow pipeline** step 4); surfacing text-finding logs in that zip export would be a
  separate, not-yet-implemented feature.
- **YOLO inference** — `POST /api/predict` is live; `ModelTab` `.h5` uploads are wired up
- **Stave detection** — `estimate_staves_from_glyphs()` in `encode_to_mei.py` uses real staff-line glyph clustering (primary) with neume Y-gap clustering as fallback; `parse_staves()` / `parse_yolo_stave_hints()` handle YOLO-format stave detections
- **Staffline detection** — `POST /api/predict` runs connected-component filtering, centerline fitting, and stave grouping on stave-class YOLO boxes (`staffline_stage.py`, wrapping the `staff-finding/` package); results are `tasks_encode.py`'s preferred stave source ahead of `parse_yolo_stave_hints()` — see **Staffline detection** above
- **Staffline/background layer separation (medieval preset)** — the stave model now runs on a
  TensorFlow-classifier-isolated stafflines layer instead of the raw page, via the standalone
  `paco-classifier-service` (wraps the `paco-classifier/` submodule), called concurrently with the
  text/music YOLO pass from `tasks_predict.py`'s `_run_medieval_inference()`; falls back to
  raw-page stave detection if the service is unreachable — see **Staffline detection** above
- **SSE/streaming for encoding** — `ProcessingPage.tsx` streams real log lines when given a
  `streamRequest`, which every current `AppRouter` call site passes. The fake `setTimeout`
  timer path is **dormant, not gone**: `ProcessingPage.tsx` still contains the timer-driven
  fake-progress block (guarded by `if (streamRequest) return;`), which would reactivate for any
  future call site that omits `streamRequest` — slated for removal in the alpha transition
  (see `documentation_allons-y/ALPHA_TRANSITION_PLAN.md`)
- **Annotation overlay viewer** — `AnnotationsTab.tsx` renders YOLO bounding boxes on top of the source image
- **Project export (zip)** — `GET /api/projects/{id}/export` bundles MEI files + manifest into a ZIP; a second endpoint zips logs
- **Soft-delete + hard-delete** — `deleted_at` is the soft-delete flag; a separate hard-delete path does `DELETE FROM project_images` + `DELETE FROM projects` to purge BYTEA data

## Updating the bundled medieval models

`landing-page/scripts/assets/models/medieval/{text_music_detector_fulldata.pt,stave_detector_fulldata.pt}`
are committed to the repo via Git LFS so the "medieval manuscripts" preset
works out of the box, offline, with no HuggingFace token. When DDMAL retrains
these checkpoints:

1. Get the new `.pt` files (from HuggingFace, if you have access to the
   gated `DDMAL-lab/mothra-yolov11-checkpoints` repo, or wherever the retrain
   produced them).
2. Replace the two files in place at the path above and commit normally —
   `git add`/`git commit`/`git push` (Git LFS handles the upload
   transparently since `*.pt` is already tracked in `.gitattributes`).
3. No code changes needed — `landing-page/scripts/medieval_models.py`'s
   `resolve_medieval_model_paths()` always reads whatever is at that path.
4. If the new checkpoints use a different class ordering than
   `0=text,1=music` (text/music detector) or `0=staves` (stave detector),
   update `TEXT_MUSIC_CLASS_MAP`/`STAVE_CLASS_MAP` in `medieval_models.py` to
   match — see the merged 0/1/2 (text/music/staves) slot convention
   documented in that file.

For testing an unreleased checkpoint without committing it, set
`MOTHRA_MEDIEVAL_MODELS_DIR` to a local directory containing both filenames —
it takes priority over the bundled copies (see `resolve_medieval_model_paths()`).