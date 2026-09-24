"""Postgres access for text-service's batch-download zips (mothra#230,
BATCH_DIR half).

text-service was deliberately DB-free until now -- BATCH_DIR
(tempfile.gettempdir()/mothra-text/batches) held finished batch zips on
whatever node-local disk this process happened to land on. That's fine for a
single always-on replica, but doesn't survive a pod reschedule and can't be
swept except by this process's own startup sweep.

The `text_batch_zips` table itself is created by landing-page's migrate.py
(auth_api.py's init_db(), mothra#220 row 31's one-shot migration Job) --
text-service has no migration infrastructure of its own, and this is the
same physical Postgres database, so schema ownership stays centralized there
rather than teaching a second service to run its own DDL at import (the
exact anti-pattern row 31 just removed from the backend). This module only
ever does INSERT/SELECT; DELETE (retention) lives in job_store.
cleanup_stale_batch_zips, run by the worker's existing Celery-beat sweep --
see that function's docstring.
"""
import os
from typing import Optional

from psycopg2 import pool as _pg_pool

_db_pool: Optional["_pg_pool.ThreadedConnectionPool"] = None


def _get_pool() -> "_pg_pool.ThreadedConnectionPool":
    global _db_pool
    if _db_pool is None:
        # Bare os.environ[...] -- raises KeyError, not a friendly error, if
        # DATABASE_URL isn't set. Mirrors landing-page/scripts/auth_api.py's
        # _get_pool(): fail fast, no silent fallback (there is no safe
        # default DB to fall back to).
        _db_pool = _pg_pool.ThreadedConnectionPool(
            minconn=1, maxconn=5, dsn=os.environ["DATABASE_URL"]
        )
    return _db_pool


def _get_conn():
    return _get_pool().getconn()


def _release_conn(con) -> None:
    _get_pool().putconn(con)


def batch_zip_put(batch_id: str, zip_bytes: bytes) -> None:
    con = _get_conn()
    try:
        cur = con.cursor()
        cur.execute(
            "INSERT INTO text_batch_zips (batch_id, zip_bytes) VALUES (%s, %s)",
            (batch_id, zip_bytes),
        )
        con.commit()
        cur.close()
    finally:
        _release_conn(con)


def batch_zip_get(batch_id: str) -> Optional[bytes]:
    con = _get_conn()
    try:
        cur = con.cursor()
        cur.execute("SELECT zip_bytes FROM text_batch_zips WHERE batch_id=%s", (batch_id,))
        row = cur.fetchone()
        cur.close()
        return bytes(row[0]) if row else None
    finally:
        _release_conn(con)
