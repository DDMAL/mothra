# Mothra landing page

The primary web application for Mothra: upload manuscript images, run them
through the Interactive Classifier and YOLO inference, find text, correct
output in Neon.js, and export to Cantus Ultimus.

This README covers **first-time setup and day-to-day running**. For
architecture, database schema, and coding conventions, see
[`../CLAUDE.md`](../CLAUDE.md).

---

## Architecture at a glance

Five processes run together in dev:

| Port   | Process                          | Role                                                         |
| ------ | --------------------------------- | -------------------------------------------------------------- |
| `5173` | Vite dev server (this folder)     | Open this in the browser; proxies `/api`, `/neon`, `/Neon-gh` → `:8001` |
| `8001` | landing-page FastAPI (`uvicorn`)  | `/api/*`; also calls `:8000` and `:8002` server-to-server      |
| `8000` | Interactive Classifier (`ic/`)    | IC REST API **and** the IC SPA served into an iframe            |
| `8002` | Text-finding service (`text-service/`) | Wraps the `mothra-text` pipeline; called from `:8001`     |
| —      | Celery worker                    | Runs `/predict`, `/encode-upload`, `/encode-batch` jobs; no port of its own |

`:8000` and `:8002` are separate git submodules/services, not part of this
folder, but they must be running for the Interactive Classifier and
text-finding steps to work.

The Celery worker also needs a running **Redis** (default
`redis://localhost:6379/0`) — it's only the task broker, not a data store;
job status/progress live in Postgres alongside everything else (see
[`../CLAUDE.md`](../CLAUDE.md) for the `jobs`/`job_events`/`job_uploads`/`job_sessions`
tables and how the job queue works end to end).

---

## Prerequisites

- **Node.js** (Vite + React frontend) — any recent LTS works.
- **Python 3.9+** for the landing-page backend venv.
- **Python 3.10** specifically for `text-service` (`kraken`/`htrflow` pin to
  it).
- **[uv](https://docs.astral.sh/uv/)** for the `ic/api` venv (`ic/` uses
  `uv.lock`, requires Python ≥3.11 — `uv` will fetch it if you don't have
  one).
- **yarn** (only needed once, to build the embedded Neon.js editor).
- **Redis** — the Celery broker for the job queue (`/predict`, `/encode-upload`,
  `/encode-batch`). Any local install works:

  ```bash
  brew install redis
  brew services start redis
  redis-cli ping   # should print PONG
  ```

- A **local Postgres database** — the backend just needs a plain `psycopg2`
  DSN, so any Postgres install works. If you don't already have one running:

  ```bash
  brew install postgresql@16
  brew services start postgresql@16
  createdb mothra_dev
  ```

  Local DSN would then be `postgresql://localhost/mothra_dev` (no
  user/password needed for the default local trust setup).

Check what you have:

```bash
node -v && python3 --version && python3.10 --version && uv --version && yarn -v && redis-cli ping
```

---

## First-time setup

Clone with submodules (or `git submodule update --init --recursive` if you
already cloned without `--recursive`):

```bash
git clone --recursive <repo-url>
cd mothra
```

### 1. Frontend deps

```bash
cd landing-page
npm install
```

### 2. Backend venv (`:8001`)

```bash
cd landing-page/scripts
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `landing-page/scripts/.env`:

```
DATABASE_URL=postgresql://localhost/mothra_dev
MOTHRA_SECRET=<any long random string>
```

- `DATABASE_URL` is **required** — the backend fails to start without it.
  Points at your local Postgres (per above) — tables are created
  automatically on first run (see `auth_api.py:init_db`).
- `MOTHRA_SECRET` is **required** too — it signs JWTs, refresh tokens, and
  Neon edit-session tokens. The backend/worker fail to start without it,
  rather than silently generating a random one each process start (which
  used to invalidate every existing login token on restart with no error to
  flag the misconfiguration).
- Optional overrides (all have working defaults, only set if you're running
  services on non-default ports/hosts): `STORAGE_QUOTA_MB` (default 500),
  `IC_API_URL` / `IC_PUBLIC_URL` (default `http://localhost:8000`),
  `TEXT_API_URL` (default `http://localhost:8002`), `CELERY_BROKER_URL`
  (default `redis://localhost:6379/0`), `ALLOWED_ORIGINS` (default
  `http://localhost:5173`, matching Vite's dev port -- not a wildcard).
  These (and a few filesystem paths) have their non-secret defaults in
  `landing-page/scripts/config.yaml` — `.env`/environment only needs to set
  them if you're overriding a default, and `DATABASE_URL`/`MOTHRA_SECRET`
  stay in `.env` since they're secrets.

### 3. Interactive Classifier venv (`:8000`)

```bash
cd ic/api
uv sync
```

### 4. Text-finding service venv (`:8002`)

First, make sure the `mothra-text` submodule is populated (it's empty if you
cloned without `--recursive`):

```bash
git submodule update --init mothra-text
```

Then create the venv and install dependencies:

```bash
cd text-service
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/DDMAL/volpiano-display-utilities.git
pip install -e ../mothra-text
```

The last line installs `mothra-text` into the venv so `main.py` can import
`run_pipeline`. If `mothra-text` has no `setup.py`/`pyproject.toml` yet,
run uvicorn with `PYTHONPATH=../mothra-text` prepended instead.

Then download the Tridis OCR recognition model (one-time, ~few hundred MB):

```bash
.venv/bin/htrmopo get 10.5281/zenodo.10788590
```

(`htrmopo` is a CLI binary, not a Python module — `python -m htrmopo` will fail.)

Without this, the text-service starts but runs in stub mode — the pipeline
completes but produces no syllables and text-finding appears to return empty
results with no error message.

### 5. Build the embedded Neon.js editor (one-time, or after `neon/` updates)

This is bundled into the frontend build (`npm run build` runs it
automatically), but if you're only running `npm run dev` you still need the
built assets present at least once:

```bash
cd landing-page
npm run build:neon
```

---

## Running locally

From the repo root, the launcher starts and stops all five processes together
(this includes the Celery worker — make sure Redis is running first, per
Prerequisites):

```bash
./dev.sh
```

Then open **http://localhost:5173**.

```bash
./dev.sh -b     # also rebuild the IC frontend bundle first — do this after
                #   pulling changes to the ic/ submodule's frontend, otherwise
                #   the iframe serves a stale build
./dev.sh -f     # free the ports first if something is already listening on them
./dev.sh -bf    # both
./dev.sh -h     # help
Ctrl-C          # stops all five cleanly
```

Ports are overridable via env vars: `WEB_PORT`, `API_PORT`, `IC_PORT`,
`TEXT_PORT` (e.g. `WEB_PORT=3000 ./dev.sh`). The Celery worker has no port of
its own; its broker URL is overridable via `CELERY_BROKER_URL`.

`dev.sh` checks for each venv/dependency before starting and tells you the
exact command to fix it if something's missing — if in doubt, just run it
and read the error. It also soft-warns (doesn't block startup) if Redis isn't
reachable.

### Running servers manually (one terminal each)

Only do this if you need to watch one service's output in isolation —
`./dev.sh` is the normal path.

```bash
# Terminal 1 — Interactive Classifier (:8000)
cd ic/api && HOST=127.0.0.1 PORT=8000 .venv/bin/ic-api

# Terminal 2 — text-finding service (:8002)
cd text-service && .venv/bin/uvicorn main:app --port 8002

# Terminal 3 — landing-page backend (:8001)
cd landing-page/scripts && source .venv/bin/activate
uvicorn main:app --reload --port 8001

# Terminal 4 — Celery worker (predict/encode jobs; needs Redis running)
cd landing-page/scripts && source .venv/bin/activate
celery -A celery_app.celery_app worker --loglevel=info --pool=threads --concurrency=2

# Terminal 5 — landing-page frontend (:5173)
cd landing-page && npm run dev
```

---

## Running with Docker Compose

A root-level `docker-compose.yml` builds and runs `redis`, `ic`, `text-service`,
and `backend`/`worker` (both from `landing-page/Dockerfile`, `worker` just
overrides the command) as containers instead of local processes.

### Docker prerequisites

These are easy to skip and each one fails in a confusing, non-obvious way if
missed — check all three before your first build:

- **`git-lfs`** — the bundled medieval model `.pt` files
  (`landing-page/scripts/assets/models/medieval/*.pt`) are stored via Git
  LFS. If `git-lfs` isn't installed, `git lfs pull` below silently does
  nothing useful (no error, but the files stay as tiny LFS pointer text
  instead of real model weights) and prediction fails at runtime with an
  unhelpful error. Install and register it **once per machine**, before
  cloning or pulling:

  ```bash
  brew install git-lfs
  git lfs install
  ```

- **`docker-buildx`** — required by Docker Compose to actually honor
  `landing-page/.dockerignore`. Without it, `docker compose build` silently
  falls back to Docker's legacy builder, which ignores `.dockerignore`
  entirely — the build context balloons to several GB (pulling in
  `node_modules`, Python `.venv`s, `scripts/stored_models`, etc.) and builds
  take far longer than they should, with no error telling you why. Confirmed
  by hitting this directly during testing. Check you have it:

  ```bash
  docker buildx version
  ```

  Docker Desktop bundles this already. On Colima (or any Docker CLI-only
  setup), install explicitly:

  ```bash
  brew install docker docker-compose docker-buildx colima
  ```

- **At least 8GB RAM / 4 CPUs allocated to the Docker host/VM.** `worker`
  (YOLO inference) and `text-service` (Kraken segmentation + HTR) are both
  memory-heavy; Colima's 2GB default isn't enough and causes `text-service`
  to get silently OOM-killed mid-request (`docker compose ps` shows
  `Exited (137)`, easy to mistake for an application bug). Confirmed by
  hitting this directly during testing. If using Colima:

  ```bash
  colima start --memory 8 --cpu 4
  ```

  (Already running with less? `colima stop` first, then start again with
  the flags above.)

### Build and run

From the repo root:

```bash
git submodule update --init --recursive   # if not already done
git lfs pull                              # medieval model .pt files are LFS pointers otherwise

# create a root-level .env (gitignored, separate from landing-page/scripts/.env):
#   DATABASE_URL=postgresql://host.docker.internal/mothra_dev
#   MOTHRA_SECRET=...

docker compose build
docker compose up
```

Open **http://localhost:8001** (the backend container serves the built
frontend directly — there's no separate `:5173` container).
`IC_API_URL`/`TEXT_API_URL` are wired to the compose
service names (`http://ic:8000`, `http://text-service:8002`) automatically;
Postgres itself isn't a compose service, so `DATABASE_URL` still points at
your local Postgres install (per Prerequisites above) — from inside a
container that means `host.docker.internal`, not `localhost`, or the
containers won't be able to reach a Postgres running on the host machine.

The `text-service` image is large (pulls in `torch`/`kraken`/`htrflow`) and
the first `docker compose build` will take a while.

**Prefer redeploying the whole stack together** (`docker compose up -d --build`)
over recreating a single service in isolation
(`docker compose up -d --no-deps --build text-service`) — doing the latter
while other services keep running can leave a dependent service (e.g.
`worker`) holding a stale connection to the now-gone old container. See
[`../CLAUDE.md`](../CLAUDE.md)'s Deployment section for the full story.

---

## Building for production

```bash
cd landing-page
npm run build       # builds the Neon submodule, then tsc + vite build
```

`npm run build:neon` requires `yarn` and sets
`NODE_OPTIONS=--openssl-legacy-provider` internally — no extra setup needed
beyond having `yarn` installed.

Other useful scripts:

```bash
npm run lint      # eslint
npm run format    # prettier --write src/
npm run preview   # preview a production build locally
```

---

## Troubleshooting

- **"missing landing-page venv" / "missing IC venv" / "missing text-finding
  venv" from `dev.sh`** — run the setup command it prints; it's the exact
  command from the relevant step above.
- **Backend crashes on startup with a `KeyError` for `DATABASE_URL`, or a
  `RuntimeError` for `MOTHRA_SECRET`** — you haven't created
  `landing-page/scripts/.env` yet, or it's missing one of these two required
  values. Both fail fast at import rather than falling back to a default.
- **IC iframe shows old/broken UI after pulling changes** — rebuild it with
  `./dev.sh -b`.
- **A port is already in use** — `./dev.sh -f`, or find and stop whatever's
  holding it: `lsof -nP -iTCP:<port> -sTCP:LISTEN`.
- **Text-finding completes with no syllables / "no text-finding results"**
  — the Tridis OCR model isn't installed. Run
  `.venv/bin/htrmopo get 10.5281/zenodo.10788590` inside `text-service`
  (step 4 above). Use the binary directly — `python -m htrmopo` will fail.
  Without the model the service silently uses stub mode.
- **`ModuleNotFoundError: No module named 'run_pipeline'` when starting
  `:8002`** — the `mothra-text` submodule isn't installed in the venv. Run
  `git submodule update --init mothra-text` from the repo root, then
  `pip install -e ../mothra-text` inside the `text-service` venv (step 4
  above).
- **Celery worker crashes on startup with `KeyError: 'DATABASE_URL'`** —
  `celery_app.py` loads `landing-page/scripts/.env` itself (via `load_dotenv()`)
  since, unlike the backend, nothing else in the worker's import chain does.
  If you see this, something upstream of that call got reordered — check
  `celery_app.py` still calls `load_dotenv()` before importing `config`/`celery`.
- **"missing celery in landing-page venv" from `dev.sh`** — same venv as the
  backend (step 2); just re-run `pip install -r requirements.txt` there.
- **Redis not reachable / predict-encode jobs never start** —
  `redis-cli ping` should print `PONG`. If not, `brew services start redis`
  (or however you installed it). `dev.sh` warns but doesn't block on this —
  the kickoff request will still return a `job_id`, but the Celery worker
  will never pick up the task.
- **Predict/encoding progress bar never moves / hangs forever** — check the
  `[worker]`-tagged log lines for a traceback, and/or query the `jobs` table
  directly (`SELECT status FROM jobs ORDER BY created_at DESC LIMIT 5`) to see
  if the job ever left `pending`. `GET /api/jobs/{id}/stream` also gives up
  and surfaces an error after ~90s of no new events if `status` is stuck on
  `running` (worker crash without a clean error event).
- **`docker compose build` takes forever / build context is several GB** —
  you're missing `docker-buildx`; Compose has silently fallen back to the
  legacy builder, which ignores `.dockerignore`. Check with
  `docker buildx version`, install with
  `brew install docker docker-compose docker-buildx colima` (Docker Desktop
  already bundles it).
- **Medieval preset prediction fails, or the bundled `.pt` files look tiny
  (a few hundred bytes of text, not real model weights)** — `git-lfs` isn't
  installed/registered, so `git lfs pull` did nothing. Run
  `brew install git-lfs && git lfs install` **once**, then `git lfs pull`
  again.
- **A container (usually `text-service`) exits with code 137 mid-job** —
  it was OOM-killed. Colima's default 2GB isn't enough; run
  `colima stop && colima start --memory 8 --cpu 4`.
- **A job silently hangs after redeploying just one container** (e.g.
  `docker compose up -d --no-deps --build text-service`) — a dependent
  service (usually `worker`) is holding a stale connection to the old,
  now-gone container. Restart it, or redeploy the whole stack together
  (`docker compose up -d --build`) instead of one service at a time.
