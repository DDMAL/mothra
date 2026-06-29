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

Schema is migrated forward via `_migrate_db()` in `auth_api.py` — new columns are `ALTER TABLE ADD COLUMN IF NOT EXISTS` guarded with `DuplicateColumn` catch.

---

## Running locally

**Three servers must be running simultaneously** — the frontend, its backend, and the Interactive Classifier service (the IC step iframes it):

| Port | Process | Role |
|---|---|---|
| `5173` | Vite dev server (landing-page) | Open this in the browser; proxies `/api`, `/neon`, `/Neon-gh` → `:8001` |
| `8001` | landing-page FastAPI (`uvicorn`) | `/api/*`; also reaches IC server-to-server |
| `8000` | IC service (`ic/` submodule) | IC REST API **and** the built IC SPA (served single-origin) |

### One command (recommended)

```bash
./dev.sh          # starts all three; Ctrl-C tears all three down cleanly
./dev.sh -b       # rebuild the IC frontend bundle first (do this after the
                  #   ic/ submodule's frontend changes — else the iframe
                  #   serves a stale build)
./dev.sh -f       # free the ports first if something is stuck on them
./dev.sh -h       # help
```

Ports are overridable (`WEB_PORT=3000 ./dev.sh`). The script assumes the venvs
already exist (`ic/api/.venv`, `landing-page/scripts/.venv`) and
`landing-page/node_modules` is installed — it runs them, it doesn't create them.

### Manual (one terminal each)

```bash
# Terminal 1 — IC service (:8000)
cd ic/api && HOST=127.0.0.1 PORT=8000 .venv/bin/ic-api

# Terminal 2 — landing-page backend (:8001)
cd landing-page/scripts && source .venv/bin/activate
uvicorn main:app --reload --port 8001

# Terminal 3 — landing-page frontend (:5173)
cd landing-page && npm run dev
```

Open `http://localhost:5173`. All `/api/*` calls proxy to the backend automatically.

**Environment** — `landing-page/scripts/.env`:
```
DATABASE_URL=postgresql://...   # Neon connection string
MOTHRA_SECRET=...               # JWT signing key
IC_API_URL=http://localhost:8000    # how :8001 reaches IC (server-to-server)
IC_PUBLIC_URL=http://localhost:8000 # how the browser/iframe reaches IC's SPA
```

---

## Key files

| Path | Role |
|---|---|
| `landing-page/scripts/main.py` | FastAPI app, CORS, mounts routers |
| `landing-page/scripts/auth_api.py` | Auth endpoints, DB init, project CRUD, image storage |
| `landing-page/scripts/account_api.py` | Profile update, password change, account delete |
| `landing-page/scripts/encode_api.py` | `POST /api/encode-upload` — GameraXML → MEI |
| `landing-page/scripts/encode_to_mei.py` | Core encoding logic: parse XML, estimate staves, build MEI, validate |
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
- Encoding sessions stored in `_sessions` dict in `encode_api.py` (in-memory, lost on restart); manifest also written to `MANIFEST_DIR` as `.jsonld` files
- No background task queue — encoding is synchronous in the request handler

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

- YOLO inference endpoint (`POST /api/predict`) — `ModelTab` uploads `.h5` files but nothing runs them
- Real stave detection — `estimate_staves_from_glyphs()` in `encode_to_mei.py` is a heuristic placeholder
- SSE/streaming for encoding progress — `ProcessingPage.tsx` uses fake `setTimeout` timers; logs arrive all at once
- Batch encoding — one XML+image pair at a time currently
- Annotation overlay viewer — `AnnotationsTab.tsx` shows filename cards only, no geometry rendered
