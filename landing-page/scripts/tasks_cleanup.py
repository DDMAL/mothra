"""Celery task backing celery_app.py's beat_schedule -- periodic sweep of the
Postgres-backed staging/output tables (job_uploads, job_sessions,
neon_manifests), which used to only get cleaned up once, at backend startup
(main.py) -- or (neon_manifests, before mothra#230) not at all from here,
since NEON_MANIFESTS_DIR was backend-local disk a worker-run task couldn't
reach. See job_store.run_periodic_cleanup's docstring."""
import logging

from celery_app import celery_app
from job_store import run_periodic_cleanup

logger = logging.getLogger(__name__)


@celery_app.task(name="cleanup.run_periodic")
def run_periodic_cleanup_task():
    result = run_periodic_cleanup()
    logger.info("periodic cleanup: %s", result)
    return result
