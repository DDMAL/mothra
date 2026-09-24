"""Celery task backing celery_app.py's beat_schedule -- periodic sweep of the
Postgres-backed staging/output tables (job_uploads, job_sessions,
text_batch_zips), which used to only get cleaned up once, at backend startup
(main.py) or (for text_batch_zips) text-service's own process start. See
job_store.run_periodic_cleanup's docstring for why the backend-local Neon
manifest sweep (auth_api.cleanup_stale_neon_manifests) is deliberately NOT
included here -- it lives on the backend container's own disk, which this
task (running on the worker) can't reach."""
import logging

from celery_app import celery_app
from job_store import run_periodic_cleanup

logger = logging.getLogger(__name__)


@celery_app.task(name="cleanup.run_periodic")
def run_periodic_cleanup_task():
    result = run_periodic_cleanup()
    logger.info("periodic cleanup: %s", result)
    return result
