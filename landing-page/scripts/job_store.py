import json
import uuid
from typing import Optional

from auth_api import get_db_conn, release_db_conn

def new_job_id() -> str:
    return uuid.uuid4().hex[:8]

def create_job(job_id: str, kind: str, project_id: Optional[int], *,
               params: Optional[dict]= None, retry_of: Optional[str] = None,
               attempt: int = 1) -> None:
    con = get_db_conn()
    try:
        cur = con.cursor()
        cur.execute(
            "INSERT INTO jobs (job_id, kind, project_id, status, params, retry_of, attempt)"
            " VALUES (%s,%s,%s,'pending',%s,%s,%s)",
            (job_id, kind, project_id, json.dumps(params) if params is not None else None,
             retry_of, attempt),
        )
        con.commit()
        cur.close()
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