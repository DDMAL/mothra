from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
import psycopg2, psycopg2.errors

from auth_api import get_current_user, db_cursor, verify_password, hash_password, STORAGE_QUOTA_BYTES

router = APIRouter()

class UpdateUserBody(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None

@router.patch("/me")
def update_me(body: UpdateUserBody, user=Depends(get_current_user)):
    if not body.username and not body.email:
        return user
    with db_cursor() as (con, cur):
        try:
            if body.username:
                cur.execute("UPDATE users SET username = %s WHERE id = %s", (body.username, user["id"]))
            if body.email:
                cur.execute("UPDATE users SET email = %s WHERE id = %s", (body.email, user["id"]))
            con.commit()
        except psycopg2.errors.UniqueViolation:
            con.rollback()
            raise HTTPException(status_code=409, detail="username or email already taken")
        cur.execute(
            "SELECT id, username, email, first_name, last_name, created_at FROM users WHERE id=%s",
            (user["id"],)
        )
        row = cur.fetchone()
    return {"id": row[0], "username": row[1], "email": row[2], "firstName": row[3], "lastName": row[4], "createdAt": str(row[5])}

class ChangePasswordBody(BaseModel):
    old_password: str
    new_password: str

@router.patch("/me/password")
def change_password(body: ChangePasswordBody, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        cur.execute("SELECT password_hash FROM users WHERE id=%s", (user["id"],))
        row = cur.fetchone()
    if not row or not verify_password(body.old_password, row[0]):
        raise HTTPException(status_code=400, detail="old password is incorrect")
    with db_cursor() as (con, cur):
        cur.execute("UPDATE users SET password_hash=%s WHERE id=%s",
                    (hash_password(body.new_password), user["id"]))
        cur.execute("UPDATE refresh_tokens SET revoked_at=NOW() WHERE user_id=%s AND revoked_at IS NULL", (user["id"],))
        con.commit()
    return {"ok": True}

@router.get("/me/usage")
def get_usage(user=Depends(get_current_user)):
    uid = user["id"]
    with db_cursor() as (con, cur):
        cur.execute("""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE deleted_at IS NULL) AS active
            FROM projects WHERE user_id = %s
        """, (uid,))
        proj = cur.fetchone()

        cur.execute("""
            SELECT COUNT(*), COALESCE(SUM(octet_length(data)), 0)
            FROM project_images
            WHERE project_id IN (SELECT id FROM projects WHERE user_id = %s)
        """, (uid,))
        imgs = cur.fetchone()

        cur.execute("""
            SELECT
                COUNT(*),
                COALESCE(SUM(octet_length(xml_content)), 0),
                COUNT(*) FILTER (WHERE corrected = 1)
            FROM mei_files
            WHERE project_id IN (SELECT id FROM projects WHERE user_id = %s)
        """, (uid,))
        mei = cur.fetchone()

        cur.execute("""
            SELECT COUNT(*), COALESCE(SUM(octet_length(content)), 0)
            FROM project_logs
            WHERE project_id IN (SELECT id FROM projects WHERE user_id = %s)
        """, (uid,))
        logs = cur.fetchone()

    return {
        "projects": {"total": proj[0], "active": proj[1], "deleted": proj[0] - proj[1]},
        "images": {"count": imgs[0], "bytes": imgs[1]},
        "meiFiles": {"count": mei[0],  "bytes": mei[1],  "corrected": mei[2]},
        "logs": {"count": logs[0], "bytes": logs[1]},
        "quotaBytes": STORAGE_QUOTA_BYTES,

    }

@router.delete("/me")
def delete_me(user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        try:
            cur.execute("SELECT id FROM projects WHERE user_id=%s", (user["id"], ))
            pids = [r[0] for r in cur.fetchall()]
            for pid in pids:
                cur.execute("DELETE FROM annotations WHERE project_id=%s", (pid, ))
                cur.execute("DELETE FROM activity_log WHERE project_id=%s", (pid, ))
                cur.execute("DELETE FROM project_logs WHERE project_id=%s", (pid, ))
                cur.execute("DELETE FROM mei_files WHERE project_id=%s", (pid, ))
                cur.execute("DELETE FROM project_images WHERE project_id=%s", (pid, ))
                cur.execute("DELETE FROM project_models WHERE project_id=%s", (pid, ))
                cur.execute("DELETE FROM staffline_detections WHERE project_id=%s", (pid, ))
                cur.execute("DELETE FROM ic_xml_files WHERE project_id=%s", (pid, ))
            cur.execute("DELETE FROM projects WHERE user_id=%s", (user["id"], ))
            cur.execute("DELETE FROM users WHERE id=%s", (user["id"], ))
            con.commit()
        except Exception:
            con.rollback()
            raise HTTPException(status_code=500, detail="account deletion failed")
    return {"ok": True}
