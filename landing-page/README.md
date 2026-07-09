# Mothra landing page

The primary web application for Mothra: upload manuscript images, run them
through the Interactive Classifier and YOLO inference, find text, correct
output in Neon.js, and export to Cantus Ultimus.

This README covers **first-time setup and day-to-day running**. For
architecture, database schema, and coding conventions, see
[`../CLAUDE.md`](../CLAUDE.md).

---

## Architecture at a glance

Four servers run together in dev:

| Port   | Process                          | Role                                                         |
| ------ | --------------------------------- | -------------------------------------------------------------- |
| `5173` | Vite dev server (this folder)     | Open this in the browser; proxies `/api`, `/neon` → `:8001`    |
| `8001` | landing-page FastAPI (`uvicorn`)  | `/api/*`; also calls `:8000` and `:8002` server-to-server      |
| `8000` | Interactive Classifier (`ic/`)    | IC REST API **and** the IC SPA served into an iframe            |
| `8002` | Text-finding service (`text-service/`) | Wraps the `mothra-text` pipeline; called from `:8001`     |

`:8000` and `:8002` are separate git submodules/services, not part of this
folder, but they must be running for the Interactive Classifier and
text-finding steps to work.

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
- A **Postgres database** — this project is developed against
  [Neon](https://neon.tech), but the backend only talks to it via a plain
  `psycopg2` DSN, so a local Postgres install works exactly the same way.
  If you don't already have one running:

  ```bash
  brew install postgresql@16
  brew services start postgresql@16
  createdb mothra_dev
  ```

  Local DSN would then be `postgresql://localhost/mothra_dev` (no
  user/password needed for the default local trust setup).

Check what you have:

```bash
node -v && python3 --version && python3.10 --version && uv --version && yarn -v
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
DATABASE_URL=postgresql://user:password@host/dbname
MOTHRA_SECRET=<any long random string>
```

- `DATABASE_URL` is **required** — the backend fails to start without it.
  Points at any Postgres (Neon or local, per above) — tables are created
  automatically on first run (see `auth_api.py:init_db`).
- `MOTHRA_SECRET` signs JWTs. If omitted, a random one is generated each
  process start, which invalidates every existing login token on restart —
  set a real value so sessions survive a backend restart.
- Optional overrides (all have working defaults, only set if you're running
  services on non-default ports/hosts): `STORAGE_QUOTA_MB` (default 500),
  `IC_API_URL` / `IC_PUBLIC_URL` (default `http://localhost:8000`),
  `TEXT_API_URL` (default `http://localhost:8002`), `ALLOWED_ORIGINS`
  (default `*`).

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

From the repo root, the launcher starts and stops all four servers together:

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
Ctrl-C          # stops all four cleanly
```

Ports are overridable via env vars: `WEB_PORT`, `API_PORT`, `IC_PORT`,
`TEXT_PORT` (e.g. `WEB_PORT=3000 ./dev.sh`).

`dev.sh` checks for each venv/dependency before starting and tells you the
exact command to fix it if something's missing — if in doubt, just run it
and read the error.

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

# Terminal 4 — landing-page frontend (:5173)
cd landing-page && npm run dev
```

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
- **Backend crashes on startup with a `KeyError` for `DATABASE_URL`** — you
  haven't created `landing-page/scripts/.env` yet.
- **Logged out every time you restart the backend** — set `MOTHRA_SECRET`
  in `.env` instead of relying on the auto-generated fallback.
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
