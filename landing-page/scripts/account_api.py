from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
import psycopg2, psycopg2.errors

from auth_api import get_current_user, get_db_conn

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