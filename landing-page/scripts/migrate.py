"""One-shot DB schema migration entrypoint -- mothra#220 row 31.

init_db()/_migrate_db() (auth_api.py) used to run as a side effect of simply
IMPORTING auth_api -- which happened at startup in both the backend and the
Celery worker (every process that transitively imports auth_api), so every
pod start re-ran the full migration, racing whichever other pod started at
the same time. That's why backend/worker were pinned to replicas=1 (see
k8s/backend.yaml's former comment) and why a fresh database could
CrashLoopBackOff once before self-healing (k8s/README.md's former "Known
follow-ups").

Run this script once per deploy instead -- a k8s Job (k8s/migrate-job.yaml,
wired into .github/workflows/build-images.yml to run and complete before
backend/worker are applied), or manually before starting dev.sh/docker-compose's
services. auth_api.py itself no longer calls these at import, so backend/worker
starting without this having run first will fail loudly (missing tables) rather
than silently each re-creating the schema themselves.
"""
from auth_api import init_db, _migrate_db

if __name__ == "__main__":
    init_db()
    _migrate_db()
    print("migration complete")
