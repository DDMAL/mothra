from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
import psycopg2, psycopg2.errors

from auth_api import get_current_user, get_db_conn, verify_password, hash_password

router = APIRouter()

class UpdateUserBody(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None

@router.patch("/me")
def update_me(body: UpdateUserBody, user=Depends(get_current_user)):
    if not body.username and not body.email:
        return user
    con = get_db_conn()
    cur = con.cursor()
    try:
        if body.username:
            cur.execute("UPDATE users SET username = %s WHERE id = %s", (body.username, user["id"]))
        if body.email:
            cur.execute("UPDATE users SET email = %s WHERE id = %s", (body.email, user["id"]))
        con.commit()
    except psycopg2.errors.UniqueViolation:
        con.rollback()
        cur.close()
        con.close()
        raise HTTPException(status_code=409, detail="username or email already taken")
    cur.execute(
        "SELECT id, username, email, first_name, last_name, created_at FROM users WHERE id=%s",
        (user["id"],)
    )
    row = cur.fetchone()
    cur.close()
    con.close()
    return {"id": row[0], "username": row[1], "email": row[2], "firstName": row[3], "lastName": row[4], "createdAt": str(row[5])}

class ChangePasswordBody(BaseModel):
    old_password: str
    new_password: str

@router.patch("/me/password")
def change_password(body: ChangePasswordBody, user=Depends(get_current_user)):
    con = get_db_conn()
    cur = con.cursor()
    cur.execute("SELEct password_hash FROM users WHERE id=%s", (user["id"],))
    row = cur.fetchone()
    cur.close(); con.close()
    if not row or not verify_password(body.old_password, row[0]):
        raise HTTPException(status_code=400, detail="old password is incorrect")
    con = get_db_conn(); cur = con.cursor()
    cur.execute("UPDATE users SET password_hash=%s WHERE id=%s",
                (hash_password(body.new_password), user["id"]))
    con.commit(); cur.close(); con.close()
    return {"ok": True}

@router.delete("/me")
def delete_me(user=Depends(get_current_user)):
    con = get_db_conn(); cur = con.cursor()
    cur.execute("SELECT id FROM projects WHERE user_id=%s", (user["id"], ))
    pids = [r[0] for r in cur.fetchall()]
    for pid in pids:
        cur.execute("DELETE FROM mei_files WHERE project_id=%s", (pid, ))
        cur.execute("DELETE FROM project_images WHERE project_id=%s", (pid, ))
        cur.execute("DELETE FOM project_models WHERE project_id=%s", (pid, ))
    cur.execute("DELETE FROM projects WHERE user_id=%s", (user["id"], ))
    cur.execute("DELETE FROM users WHERE id=%s", (user["id"], ))
    con.commit(); cur.close(); con.close()
    return {"ok": True}
