import logging
import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from auth_api import limiter, cleanup_stale_neon_manifests, get_db_conn, release_db_conn
from celery_app import celery_app
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from auth_api import router as auth_router
from projects_api import router as projects_router
from images_api import router as images_router
from mei_api import router as mei_router
from models_api import router as models_router
from encode_api import router as encode_router
from account_api import router as account_router
from inference_api import router as inference_router
from ic_api import router as ic_router, verify_ic_finalize_support
from text_api import router as text_router
from cantus_api import router as cantus_router
from batch_api import router as batch_router
from jobs_api import router as jobs_router
from job_store import cleanup_stale_sessions, cleanup_stale_uploads

app = FastAPI()
# No "*" default: an unset ALLOWED_ORIGINS used to silently allow every
# origin in the world rather than failing loud or falling back to something
# safe. k8s/configmap.yaml (prod/staging) already sets this explicitly; the
# fallback here only covers local dev, where Vite serves the frontend at
# :5173 -- not a wildcard.
#
# CodeRabbit finding (PR #238): bare .split(",") preserves whitespace, so
# "https://a.example, https://b.example" (a space after the comma, the
# natural way to write a list) registers the second origin with a leading
# space -- CORSMiddleware compares this literally against the Origin header,
# which never has leading whitespace, so that origin is silently rejected.
# Strip each entry, and fail fast on an empty one (leftover from a stray
# comma) rather than silently registering a "" origin that matches nothing.
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")]
if any(not o for o in ALLOWED_ORIGINS):
    raise RuntimeError(
        f"ALLOWED_ORIGINS has an empty entry after splitting on commas: {ALLOWED_ORIGINS!r} "
        "-- check for a leading/trailing/double comma."
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router, prefix="/api")
app.include_router(projects_router, prefix="/api")
app.include_router(images_router, prefix="/api")
app.include_router(mei_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(encode_router, prefix="/api")
app.include_router(account_router, prefix="/api")
app.include_router(inference_router, prefix="/api")
app.include_router(ic_router, prefix="/api")
app.include_router(text_router, prefix="/api")
app.include_router(cantus_router, prefix="/api")
app.include_router(batch_router, prefix="/api")
app.include_router(jobs_router, prefix="/api")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/healthz/live", include_in_schema=False)
def healthz_live():
    """Liveness only: confirms this process is actually serving HTTP (not
    just that the OS has a listener on the port, which is all a bare
    tcpSocket probe -- k8s/backend.yaml's probes before this PR -- can tell).
    Deliberately checks NO external dependency. A livenessProbe failure gets
    k8s to kill and restart the pod -- if it also failed on a transient
    Postgres/Redis blip (like /healthz below does), a downstream outage would
    cause a restart storm that does nothing to fix the actual outage. That
    check belongs in readinessProbe (via /healthz), which only pulls the pod
    out of load-balancing rotation instead of killing it."""
    return {"status": "ok"}

@app.get("/healthz", include_in_schema=False)
def healthz():
    """Readiness: confirms /healthz/live's process-alive signal AND that
    this backend's two real dependencies are reachable -- Postgres and the
    Celery broker (Redis). Point readinessProbe here, NOT livenessProbe --
    see healthz_live's docstring for why."""
    try:
        con = get_db_conn()
        try:
            cur = con.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
            cur.close()
        finally:
            release_db_conn(con)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"database unreachable: {exc}")

    try:
        conn = celery_app.connection()
        try:
            conn.ensure_connection(max_retries=1, interval_start=0, timeout=2)
        finally:
            conn.release()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"celery broker unreachable: {exc}")

    return {"status": "ok"}

# job_uploads/job_sessions also get a periodic Celery-beat sweep now
# (celery_app.py's beat_schedule, tasks_cleanup.py) -- this immediate
# call just means a fresh deploy doesn't wait a full beat interval for
# its first cleanup.
cleanup_stale_uploads()
cleanup_stale_sessions()
# Neon-editor manifest files live on this backend container's own local
# disk (not the shared stored_models NFS mount), so unlike the two calls
# above, this can only run from the backend process itself -- see
# auth_api.cleanup_stale_neon_manifests's docstring. mei_api.py's
# create_edit_session also calls it proactively on each new edit session.
cleanup_stale_neon_manifests()

# Before serving anything: confirm the IC this backend is configured against
# still supports exporting WITHOUT finalising the session (ic_api.py's
# `finalize=false`). An IC predating that parameter ignores it and finalises,
# stranding the user's corrections behind a terminal EXPORT session with no
# error raised anywhere -- so it is checked here rather than left to fail
# silently, one encoded page at a time. Raises IcIncompatible (aborting
# startup) only on a schema that positively shows the parameter missing; an
# unreachable or unreadable IC warns and continues. See
# verify_ic_finalize_support's docstring for why unreachable must not abort.
_ic_compat = verify_ic_finalize_support()
if _ic_compat != "ok":
    logging.getLogger(__name__).info("IC finalize-support check: %s", _ic_compat)

_neon_dir = Path(__file__).parent.parent / "public" / "neon"
if _neon_dir.exists():
    app.mount("/neon", StaticFiles(directory=str(_neon_dir), html=True), name="neon")
_neon_gh_dir = _neon_dir / "Neon-gh"
if _neon_gh_dir.exists():
    app.mount("/Neon-gh", StaticFiles(directory=str(_neon_gh_dir)), name="neon-gh")

DIST_DIR = Path(__file__).parent.parent / "dist"
# Guard on the actual build artifacts, not just dist/ — a partial dist/ (e.g.
# holding only the Neon submodule build, with no assets/ or index.html) exists
# in dev and would otherwise crash StaticFiles() at import. In dev the frontend
# is served by Vite on :5173, so the backend simply skips this mount.
if (DIST_DIR / "assets").is_dir() and (DIST_DIR / "index.html").is_file():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        return FileResponse(DIST_DIR / "index.html")