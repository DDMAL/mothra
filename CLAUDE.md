# Mothra — Claude Code Guide

## What this project is

YOLO-based Optical Music Recognition (OMR) for medieval manuscripts, developed at DDMAL (McGill). It challenges the existing Rodan pipeline by replacing multi-stage pixel-level analysis with end-to-end YOLO object detection.

Two distinct parts live in this repo:
- **`landing-page/`** — The primary active web application (React + FastAPI)
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
- **bcrypt** for password hashing, **python-jose** for JWT (HS256, 72h expiry)
- Vite proxies `/api`, `/neon`, `/Neon-gh` → `localhost:8001` in dev

### Database schema (auto-created on startup in `auth_api.py:init_db`)
| Table | Purpose |
|---|---|
| `users` | id, username, email, first/last name, password_hash |
| `projects` | id, user_id, name, steps_unlocked, used_image_names, used_model_names, deleted_at, last_opened_at, is_pinned |
| `project_images` | stores image bytes as `BYTEA` |
| `project_models` | model name references only (no file stored) |
| `mei_files` | xml_content as TEXT, corrected flag |
| `activity_log`, `project_logs` | audit trail per project |
| `annotations` | YOLO detections per image (`yolo_txt`), written by the predict job |
| `text_alignments` | text-finding output per image, written by the predict job |
| `jobs` | one row per predict/encode-upload/encode-batch job; `status` ∈ `pending/running/succeeded/failed` — see **Job queue** below |
| `job_events` | ordered progress-event log per job (`job_id`, `payload JSONB`) — what `GET /api/jobs/{id}/stream` polls |
| `job_uploads` | raw XML/image bytes staged for a Celery task to pick up, keyed by a short-lived `upload_id` |
| `job_sessions` | encode-job output (`mei_bytes`, `stem`, `manifest`) — replaces the old in-memory `_sessions` dict + `MANIFEST_DIR` tempfiles |

Schema is migrated forward via `_migrate_db()` in `auth_api.py` — new columns are `ALTER TABLE ADD COLUMN IF NOT EXISTS` guarded with `DuplicateColumn` catch.

---

## Running locally

**Five processes run together** — the frontend, its backend, the Interactive Classifier service (the IC step iframes it), the text-finding service, and a Celery worker (predict/encode jobs). Redis must also be running as the Celery broker:

| Port | Process | Role |
|---|---|---|
| `5173` | Vite dev server (landing-page) | Open this in the browser; proxies `/api`, `/neon`, `/Neon-gh` → `:8001` |
| `8001` | landing-page FastAPI (`uvicorn`) | `/api/*`; also reaches IC/text-service server-to-server |
| `8000` | IC service (`ic/` submodule) | IC REST API **and** the built IC SPA (served single-origin) |
| `8002` | text-finding service (`text-service/`) | wraps the `mothra-text` pipeline; called from `:8001` |
| — | Celery worker | runs `/predict`, `/encode-upload`, `/encode-batch` jobs; no port of its own |

Redis (default `redis://localhost:6379/0`, override via `CELERY_BROKER_URL`) is
**only the Celery task broker** — it holds no application state. All job
status/progress/results live in Postgres (`jobs`/`job_events`/`job_uploads`/`job_sessions`,
see schema above), so restarting Redis doesn't lose any job history.

### One command (recommended)

```bash
./dev.sh          # starts all five; Ctrl-C tears all five down cleanly
./dev.sh -b       # rebuild the IC frontend bundle first (do this after the
                  #   ic/ submodule's frontend changes — else the iframe
                  #   serves a stale build)
./dev.sh -f       # free the ports first if something is stuck on them
./dev.sh -h       # help
```

Ports are overridable (`WEB_PORT=3000 ./dev.sh`). The script assumes the venvs
already exist (`ic/api/.venv`, `landing-page/scripts/.venv`, `text-service/.venv`)
and `landing-page/node_modules` is installed — it runs them, it doesn't create
them. It also soft-warns (doesn't block) if Redis isn't reachable at
`CELERY_BROKER_URL`.

### Manual (one terminal each)

```bash
# Terminal 1 — IC service (:8000)
cd ic/api && HOST=127.0.0.1 PORT=8000 .venv/bin/ic-api

# Terminal 2 — text-finding service (:8002)
cd text-service && .venv/bin/uvicorn main:app --port 8002

# Terminal 3 — landing-page backend (:8001)
cd landing-page/scripts && source .venv/bin/activate
uvicorn main:app --reload --port 8001

# Terminal 4 — Celery worker (predict/encode jobs)
cd landing-page/scripts && source .venv/bin/activate
celery -A celery_app.celery_app worker --loglevel=info --pool=threads --concurrency=2

# Terminal 5 — landing-page frontend (:5173)
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
`IC_API_URL`, `IC_PUBLIC_URL`, `TEXT_API_URL`, `CELERY_BROKER_URL`.

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

`POST /api/projects/{id}/predict`, `POST /api/encode-upload`, and
`POST /api/encode-batch` no longer stream results directly — each returns
`{"job_id": ...}` immediately after enqueuing a Celery task
(`tasks_predict.py` / `tasks_encode.py`), and the frontend connects separately
to `GET /api/jobs/{job_id}/stream` to watch progress. That endpoint
(`jobs_api.py`) polls the `job_events` table and re-emits the same
`data: {...}\n\n` SSE frames the old in-request generator used to `yield` —
`ProcessingPage.tsx` didn't need to change at all, only the two-step
kickoff-then-stream wiring in `AppRouter.tsx` (via `apiFetchJobStream()` in
`lib/apiFetch.ts`). Redis is purely the Celery broker; Postgres (`jobs`/`job_events`)
is the single source of truth for job status, so there's no Celery result
backend configured. Known gaps: no job cancellation (`revoke`) wired up, no
periodic cleanup of `job_uploads`/`job_sessions` rows yet, and a dead worker
is detected via a ~90s staleness timeout rather than immediately.

**The worker must run with `--pool=threads`, not Celery's default `prefork`.**
`prefork` works by `fork()`-ing a child process per worker slot; PyTorch (and
other native BLAS/OpenMP-using libraries pulled in by `ultralytics`) is not
fork-safe, and a `predict.run` task will segfault the forked child almost
immediately (`WorkerLostError: Worker exited prematurely: signal 11 (SIGSEGV)`)
the instant it touches a loaded YOLO model — confirmed by actually running a
predict job locally, not just by inspection. `--pool=threads` runs tasks in
threads within one process instead, avoiding the fork entirely; concurrency is
still real since PyTorch/numpy release the GIL during actual tensor ops.

### Deployment (Docker)

A `docker-compose.yml` at the repo root builds and runs `redis`, `ic`
(reuses the existing `ic/Dockerfile`), `text-service` (new `text-service/Dockerfile`,
build context is the repo root since it needs the sibling `mothra-text/`
submodule), and `backend`/`worker` (both from the same new `landing-page/Dockerfile`,
`worker` just overrides the `command:`). Before building:
`git submodule update --init --recursive` (submodules aren't checked out by a
plain clone) and `git lfs pull` (the medieval model `.pt` files are LFS
pointers until then). `DATABASE_URL`/`MOTHRA_SECRET` come from a root-level
`.env` (gitignored, separate from `landing-page/scripts/.env`) that Compose
auto-loads. This is now the only deployment path — the old `render.yaml`
Render Blueprint (single-service, no worker, no Redis) has been retired; it
predated the job queue and would never have processed predict/encode jobs
correctly. If a Render service was ever created from it, that needs to be
deleted/disconnected on Render's side too — removing the file doesn't
un-hook an existing deploy.

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

`scripts/stored_models` (locally-uploaded custom YOLO models, written by
`models_api.py`) is **not baked into the image** — `landing-page/.dockerignore`
excludes it, and `docker-compose.yml` mounts a named volume
(`stored_models:/app/scripts/stored_models`) on both `backend` and `worker`
so uploads persist across container restarts and are visible to whichever
service needs to read them.

**Resource requirements: at least 8GB RAM, 4 CPUs for the Docker host/VM.**
Confirmed by actually running a real predict job through the containers —
with the default Colima allocation (2GB), `text-service`'s Kraken/HTR
segmentation step got SIGKILL'd by the VM's OOM killer mid-request
(`docker compose ps` shows `Exited (137)`, easy to mistake for an
application bug rather than an OOM kill). `worker` (YOLO inference) and
`text-service` (Kraken segmentation + HTR) are the two memory-heavy
containers; `backend`/`ic`/`redis` are comparatively light. If you see a
service unexpectedly exit with code 137 mid-job, check available memory
before debugging application code.

`text-service`'s recognition model (Tridis, via `htrmopo`) is baked into
the image at build time, matching local dev's one-time manual
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
restarting dependents together after any redeploy remains the safer habit.

---

## Key files

| Path | Role |
|---|---|
| `landing-page/scripts/main.py` | FastAPI app, CORS, mounts routers |
| `landing-page/scripts/auth_api.py` | Auth endpoints, DB init (incl. job-queue tables), project CRUD, image storage |
| `landing-page/scripts/account_api.py` | Profile update, password change, account delete |
| `landing-page/scripts/config.py` / `config.yaml` | Centralized non-secret paths + service URLs, env-var overridable |
| `landing-page/scripts/celery_app.py` | Celery app instance/config; entrypoint for `celery -A celery_app.celery_app worker` |
| `landing-page/scripts/job_store.py` | Postgres-backed job state: create/status/events, staged uploads, encode session/manifest storage |
| `landing-page/scripts/jobs_api.py` | `GET /api/jobs/{id}/stream` — polls `job_events`, re-emits SSE frames |
| `landing-page/scripts/tasks_predict.py` / `tasks_encode.py` | Celery tasks: the actual YOLO inference / MEI-building work, run out-of-request |
| `landing-page/scripts/yolo_inference.py` | YOLO model loading/inference (`resolve_yolo_models`, `YoloModelSet`) shared by the predict task and `batch_api.py` |
| `landing-page/scripts/encode_api.py` | `POST /api/encode-upload` / `/encode-batch` — kickoff endpoints, enqueue Celery tasks |
| `landing-page/scripts/encode_to_mei.py` | Core encoding logic: parse XML, estimate staves, build MEI, validate |
| `landing-page/src/lib/apiFetch.ts` | `apiFetch`/`apiFetchJobStream` — auth-aware fetch wrapper + job kickoff-then-stream helper |
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
6. Send to Cantus Ultimus  →  view: "sending"
```

`stepsUnlocked` on the project record gates which steps are accessible. It increments as each step completes and is persisted via `PUT /api/projects/{id}`.

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
- `/predict`, `/encode-upload`, `/encode-batch` run as Celery tasks (`tasks_predict.py`/`tasks_encode.py`); kickoff endpoints validate synchronously, create a `jobs` row, enqueue, and return `{"job_id": ...}` — see **Job queue** above
- Encode job output (`mei_bytes`, `stem`, `manifest`) lives in the `job_sessions` Postgres table, read by `GET /mei/{id}` / `GET /manifest/{id}` — no more in-memory `_sessions` dict or `MANIFEST_DIR` tempfiles
- `/projects/{id}/text-batch/run` (`batch_api.py`) is still synchronous — not yet converted to the job queue

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

## Things that don't exist yet (planned)

- JWT refresh — 72h expiry with no `POST /api/auth/refresh`; sessions drop mid-workflow
- Job cancellation — clicking "cancel" in `ProcessingPage` stops the browser from reading the stream but doesn't stop the Celery worker (no `revoke(..., terminate=True)`)
- Cleanup of `job_uploads`/`job_sessions` rows — no TTL/periodic deletion yet, they accumulate
- `/projects/{id}/text-batch/run` still runs synchronously — not yet moved to the Celery job queue

## Things that have been implemented (no longer placeholders)

- **Job queue** — `/predict`, `/encode-upload`, `/encode-batch` run as Celery tasks with Postgres-backed status/progress (`jobs`/`job_events`); see **Job queue** above
- **Batch encoding** — `POST /api/encode-batch` (`batch_api.py`/`tasks_encode.py`) handles multiple XML+image pairs in one job
- **YOLO inference** — `POST /api/predict` is live; `ModelTab` `.h5` uploads are wired up
- **Stave detection** — `estimate_staves_from_glyphs()` in `encode_to_mei.py` uses real staff-line glyph clustering (primary) with neume Y-gap clustering as fallback; `parse_staves()` / `parse_yolo_stave_hints()` handle YOLO-format stave detections
- **SSE/streaming for encoding** — `ProcessingPage.tsx` streams real log lines; fake `setTimeout` timers are gone
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