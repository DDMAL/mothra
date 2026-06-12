from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
import sqlite3

from auth_api import get_current_user, DB_PATH

router = APIRouter()

class UpdateUserBody(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None

@router.patch("/me")
def update_me(body: UpdateUserBody, user=Depends(get_current_user)):
    if not body.username and not body.email:
        return user
    con = sqlite3.connect(DB_PATH)
    try:
        if body.username:
            con.execute("UPDATE users SET username = ? WHERE id = ?", (body.username, user["id"]))
        if body.email:
            con.execute("UPDATE users SET email = ? WHERE id = ?", (body.email, user["id"]))
        con.commit()
    except sqlite3.IntegrityError:
        con.close()
        raise HTTPException(status_code=409, detail="username or email already taken")
    row = con.execute(
        "SELECT id, username, email, first_name, last_name, created_at FROM users WHERE id=?",
        (user["id"],)
    ).fetchone()
    con.close()
    return {"id": row[0], "username": row[1], "email": row[2], "firstName": row[3], "lastName": row[4], "createdAt": row[5]}