import json
import uuid
from typing import Optional

from auth_api import get_db_conn, release_db_conn

def new_job_id() -> str:
    return uuid.uuid4().hex[:8]

def create_job(job_id: str, kind: str, project_id: Optional[int], *,
               params: Optional[dict]= None, retry_of: Optional[str] = None,
               attempt: int = 1, dedupe_seconds: int = 0) -> tuple[str, bool]:
    """Creates a new job row. If `dedupe_seconds` > 0 and a pending/running
    job of the same kind+project_id was already created within that window,
    returns that EXISTING job's id instead of inserting a new row - guards
    against a client sending the same kickoff request twice in quick
    succession (e.g. React StrictMode double-invoking a kickoff effect in
    dev), which would otherwise enqueue two independent Celery tasks doing
    the same real work (confirmed: two `predict`/`text_batch` jobs, each
    writing its own annotations row for the same image). Not applied by
    default (dedupe_seconds=0) since encode_upload/encode_batch stage raw
    upload bytes under a fresh id before calling this - deduping those would
    just orphan that staged data rather than avoid real duplicate work.
    Returns (job_id_to_use, is_new) - callers should only enqueue their
    Celery task when is_new is True.
    """
    con = get_db_conn()
    try:
        cur = con.cursor()
        if dedupe_seconds > 0 and project_id is not None:
            cur.execute(
                "SELECT job_id FROM jobs WHERE kind=%s AND project_id=%s"
                " AND status IN ('pending','running')"
                " AND created_at > now() - %s::interval"
                " ORDER BY created_at DESC LIMIT 1",
                (kind, project_id, f"{dedupe_seconds} seconds"),
            )
            row = cur.fetchone()
            if row:
                cur.close()
                return row[0], False
        cur.execute(
            "INSERT INTO jobs (job_id, kind, project_id, status, params, retry_of, attempt)"
            " VALUES (%s,%s,%s,'pending',%s,%s,%s)",
            (job_id, kind, project_id, json.dumps(params) if params is not None else None,
             retry_of, attempt),
        )
        con.commit()
        cur.close()
        return job_id, True
    finally:
        release_db_conn(con)

def claim_project_job(project_id: int, kind: str, *, job_id: str,
                       allowed_kinds: Optional[set[str]] = None,
                       params: Optional[dict] = None, retry_of: Optional[str] = None,
                       attempt: int = 1, dedupe_seconds: int = 0,
                       ) -> tuple[Optional[str], bool, Optional[dict]]:
    """Atomic version of "get_active_job_for_project() then create_job()" --
    that two-call, two-connection sequence has a real TOCTOU race: two
    concurrent kickoff requests for the same project can both see no active
    job and both insert, so two jobs end up running and writing project
    state at once. This does the active-job check and the insert as one
    statement sequence inside one transaction, serialized per-project via
    pg_advisory_xact_lock(project_id) -- held only for this transaction,
    auto-released on commit/rollback, so unrelated projects' kickoffs never
    contend with each other, and this connection is never left holding a
    lock past its own request.

    Returns (job_id_to_use, is_new, active_job). `active_job` is set (and
    job_id_to_use is None) when a job of a kind NOT in `allowed_kinds`
    (defaults to just `kind` itself) is already active -- callers should
    raise 409 using it. Unlike get_active_job_for_project's {"job_id",
    "kind", "status"}, this active_job has no "status" key -- every caller
    of this function only ever reads "job_id"/"kind" out of it for the 409
    message, so it was never added here.

    Otherwise behaves like create_job: `is_new` indicates whether the
    caller should actually enqueue a Celery task, and dedupe_seconds keeps
    its original meaning (collapse an exact same-kind duplicate kickoff
    within that window into the existing job's id)."""
    con = get_db_conn()
    try:
        cur = con.cursor()
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (project_id,))

        kinds = tuple(allowed_kinds or {kind})
        cur.execute(
            "SELECT job_id, kind FROM jobs WHERE project_id=%s"
            " AND status IN ('pending','running')"
            " AND kind <> ALL(%s)"
            " ORDER BY created_at DESC LIMIT 1",
            (project_id, list(kinds)),
        )
        row = cur.fetchone()
        active = {"job_id": row[0], "kind": row[1]} if row else None
        if active is not None:
            con.commit()  # nothing written yet; just releases the advisory lock
            return None, False, active

        if dedupe_seconds > 0:
            cur.execute(
                "SELECT job_id FROM jobs WHERE kind=%s AND project_id=%s"
                " AND status IN ('pending','running')"
                " AND created_at > now() - %s::interval"
                " ORDER BY created_at DESC LIMIT 1",
                (kind, project_id, f"{dedupe_seconds} seconds"),
            )
            row = cur.fetchone()
            if row:
                con.commit()
                return row[0], False, None

        cur.execute(
            "INSERT INTO jobs (job_id, kind, project_id, status, params, retry_of, attempt)"
            " VALUES (%s,%s,%s,'pending',%s,%s,%s)",
            (job_id, kind, project_id, json.dumps(params) if params is not None else None,
             retry_of, attempt),
        )
        con.commit()
        return job_id, True, None
    finally:
        release_db_conn(con)

def get_active_job_for_project(project_id: int) -> Optional[dict]:
    """Returns {"job_id": ..., "kind": ..., "status": ...} for the most recent
    pending/running job for this project, across ALL kinds, or None if there
    isn't one.

    Unlike create_job's dedupe_seconds (scoped to kind+project_id, and only a
    few-second window — meant to collapse an exact duplicate kickoff, e.g. a
    React StrictMode double-invoke), this has no time window and isn't
    scoped to one kind: it's a cross-kind guard used by predict/text-batch
    kickoff endpoints to prevent a DIFFERENT kind of job from starting while
    one is already running for the same project. Without it, a fast job
    with no long-running step of its own (e.g. text_batch, which never
    calls paco's classifier) can start and finish before a slow one (e.g.
    predict, when paco's classifier is running inside it) that's still
    genuinely in progress — and whichever job's own ProcessingPage mount
    happens to see "done" first drives navigation, with no idea the other
    job even exists.
    """
    con = get_db_conn()
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT job_id, kind, status FROM jobs WHERE project_id=%s"
            " AND status IN ('pending','running')"
            " ORDER BY created_at DESC LIMIT 1",
            (project_id,),
        )
        row = cur.fetchone()
        cur.close()
        return {"job_id": row[0], "kind": row[1], "status": row[2]} if row else None
    finally:
        release_db_conn(con)

def publish_event(job_id: str, event: dict) -> None:
    con = get_db_conn()
    try:
        cur = con.cursor()
        cur.execute(
            "INSERT INTO job_events (job_id, payload) VALUES (%s, %s)",
            (job_id, json.dumps(event)),
        )
        event_type = event.get("type")
        if event_type == "done":
            cur.execute("UPDATE jobs SET status='succeeded', updated_at=now() WHERE job_id=%s", (job_id,))
        elif event_type == "error":
            cur.execute("UPDATE jobs SET status='failed', updated_at=now() WHERE job_id=%s", (job_id,))
        elif event_type == "cancelled":
            cur.execute("UPDATE jobs SET status='cancelled', updated_at=now() WHERE job_id=%s", (job_id,))
        else:
            cur.execute(
                "UPDATE jobs SET status='running', updated_at=now() WHERE job_id=%s AND status='pending'",
                (job_id,),
            )
        con.commit()
        cur.close()
    finally:
        release_db_conn(con)

def get_events_since(job_id: str, last_id: int) -> list[tuple[int, str]]:
    """Returns [(event_row_id, json_text), ...] for id > last_id, ascending."""
    con = get_db_conn()
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT id, payload FROM job_events WHERE job_id=%s AND id > %s ORDER BY id ASC",
            (job_id, last_id),
        )
        rows = cur.fetchall()
        cur.close()
        # psycopg2 auto-decodes jsonb columns to dict; re-serialize to text
        # for the SSE frame, but tolerate a driver returning raw text too.
        return [(r[0], r[1] if isinstance(r[1], str) else json.dumps(r[1])) for r in rows]
    finally:
        release_db_conn(con)

def get_job_status(job_id: str) -> Optional[str]:
    con = get_db_conn()
    try:
        cur = con.cursor()
        cur.execute("SELECT status FROM jobs WHERE job_id=%s", (job_id,))
        row = cur.fetchone()
        cur.close()
        return row[0] if row else None
    finally:
        release_db_conn(con)


def stage_upload(upload_id: str, data: bytes) -> None:
    con = get_db_conn()
    try:
        cur = con.cursor()
        cur.execute("INSERT INTO job_uploads (upload_id, data) VALUES (%s, %s)", (upload_id, data))
        con.commit()
        cur.close()
    finally:
        release_db_conn(con)


def fetch_upload(upload_id: str) -> Optional[bytes]:
    con = get_db_conn()
    try:
        cur = con.cursor()
        cur.execute("SELECT data FROM job_uploads WHERE upload_id=%s", (upload_id,))
        row = cur.fetchone()
        cur.close()
        return bytes(row[0]) if row else None
    finally:
        release_db_conn(con)


def drop_upload(upload_id: str) -> None:
    con = get_db_conn()
    try:
        cur = con.cursor()
        cur.execute("DELETE FROM job_uploads WHERE upload_id=%s", (upload_id,))
        con.commit()
        cur.close()
    finally:
        release_db_conn(con)


def session_put(session_id: str, mei_bytes: bytes, stem: str, manifest: Optional[dict]) -> None:
    con = get_db_conn()
    try:
        cur = con.cursor()
        cur.execute(
            "INSERT INTO job_sessions (session_id, mei_bytes, stem, manifest) VALUES (%s, %s, %s, %s)",
            (session_id, mei_bytes, stem, json.dumps(manifest) if manifest is not None else None),
        )
        con.commit()
        cur.close()
    finally:
        release_db_conn(con)


def session_get(session_id: str) -> Optional[dict]:
    con = get_db_conn()
    try:
        cur = con.cursor()
        cur.execute("SELECT mei_bytes, stem FROM job_sessions WHERE session_id=%s", (session_id,))
        row = cur.fetchone()
        cur.close()
        return {"mei_bytes": bytes(row[0]), "stem": row[1]} if row else None
    finally:
        release_db_conn(con)


def manifest_get(session_id: str) -> Optional[dict]:
    con = get_db_conn()
    try:
        cur = con.cursor()
        cur.execute("SELECT manifest FROM job_sessions WHERE session_id=%s", (session_id,))
        row = cur.fetchone()
        cur.close()
        if not row or row[0] is None:
            return None
        return row[0] if isinstance(row[0], dict) else json.loads(row[0])
    finally:
        release_db_conn(con)

class JobCancelled(Exception):
    pass

def check_cancelled(job_id: str) -> None:
    if get_job_status(job_id) == "cancelled":
        raise JobCancelled()

def cleanup_stale_uplaods(max_age_days: int = 1) -> int:
    """job_uploads is ephemeral staging — a row is created right before enqueuing
    a Celery task and normally dropped within seconds once the task fetches it.
    A row surviving a day means the enqueue or task crashed before consuming it.
    1 day is dead-letter headroom, not a working retention window."""
    con = get_db_conn()
    try:
        cur = con.cursor()
        cur.execute(
            "DELETE FROM job_uploads WHERE created_at < NOW() - make_interval(days => %s)",
            (max_age_days,),
        )
        deleted = cur.rowcount
        con.commit()
        cur.close()
        return deleted
    finally:
        release_db_conn(con)

def cleanup_stale_sessions(max_age_days: int = 14) -> int:
    """job_sessions holds encode-job OUTPUT (mei_bytes/manifest) served on-demand
    by GET /mei/{id} and GET /manifest/{id} — a user may not download for days.
    14 days balances bounding BYTEA storage against not silently breaking a
    slow-to-return user's download link."""
    con = get_db_conn()
    try:
        cur = con.cursor()
        cur.execute(
            "DELETE FROM job_sessions WHERE created_at < NOW() - make_interval(days => %s)",
            (max_age_days,),
        )
        deleted = cur.rowcount
        con.commit()
        cur.close()
        return deleted
    finally:
        release_db_conn(con)