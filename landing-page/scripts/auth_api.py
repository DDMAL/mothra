from fastapi import APIRouter, Depends, HTTPException, Header, UploadFile, File as FAPIFile
from fastapi.responses import Response
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import psycopg2, psycopg2.extras, psycopg2.errors, os, secrets, json, mimetypes, hashlib, base64
import uuid as _uuid
from datetime import datetime, timedelta
from jose import jwt, JWTError
import bcrypt
import io, zipfile

router = APIRouter()


SECRET_KEY = os.environ.get("MOTHRA_SECRET", secrets.token_hex(32))
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 72


def get_db_conn():
    return psycopg2.connect(os.environ["DATABASE_URL"])

def init_db():
    con = get_db_conn()
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            first_name TEXT,
            last_name TEXT,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            name TEXT NOT NULL,
            steps_unlocked INTEGER DEFAULT 0,
            used_image_names TEXT DEFAULT '[]', 
            used_model_names TEXT DEFAULT '[]', 
            deleted_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_images (
            id TEXT PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id),
            name TEXT NOT NULL,
            mime_type TEXT,
            data BYTEA NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_models (
            id TEXT PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id),
            name TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mei_files (
            id TEXT PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id),
            name TEXT NOT NULL,
            xml_content TEXT,
            corrected INTEGER DEFAULT 0
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS activity_log (
                id SERIAL PRIMARY KEY,
                project_id INTEGER NOT NULL REFERENCES projects(id),
                action_type TEXT NOT NULL,
                detail TEXT DEFAULT '',
                created_at TIMESTAMPTZ DEFAULT NOW()
        )    
    """)
    con.commit()
    cur.close()
    con.close()

init_db()

# migrate existing DBs that predate used_model_names
def _migrate_db():
    con = get_db_conn()
    cur = con.cursor()
    try:
        cur.execute("ALTER TABLE projects ADD COLUMN used_model_names TEXT DEFAULT '[]'")
        con.commit()
    except psycopg2.errors.DuplicateColumn:
        con.rollback()  # column already exists
    finally:
        cur.close()
        con.close()

    con = get_db_conn()
    cur = con.cursor()
    try:
        cur.execute("ALTER TABLE projects ADD COLUMN last_opened_at TIMESTAMPTZ")
        con.commit()
    except psycopg2.errors.DuplicateColumn:
        con.rollback()
    finally:
        cur.close()
        con.close()

    con = get_db_conn()
    cur = con.cursor()
    try:
        cur.execute("ALTER TABLE projects ADD COLUMN is_pinned BOOLEAN DEFAULT FALSE")
        con.commit()
    except psycopg2.errors.DuplicateColumn:
        con.rollback()
    finally:
        cur.close()
        con.close()

_migrate_db()

def _pre_hash(pw: str) -> str:
    return base64.b64encode(hashlib.sha256(pw.encode("utf-8")).digest()).decode()

def hash_password(pw: str) -> str:
    return bcrypt.hashpw(_pre_hash(pw).encode(), bcrypt.gensalt()).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(_pre_hash(plain).encode(), hashed.encode())

def create_token(user_id: int) -> str:
    exp = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode({"sub": str(user_id), "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="not authenticated")
    token = authorization.removeprefix("Bearer ")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="invalid token")
    con = get_db_conn()
    cur = con.cursor()
    cur.execute(
        "SELECT id, username, email, first_name, last_name, created_at from users where id=%s", (user_id,)
    )
    row = cur.fetchone()
    cur.close()
    con.close()
    if not row:
        raise HTTPException(status_code=401, detail="user not found")
    return {"id": row[0], "username": row[1], "email": row[2], "firstName": row[3], 
            "lastName": row[4], "createdAt": str(row[5])}

class RegisterBody(BaseModel):
    username: str
    email: str
    first_name: str
    last_name: str
    password: str

class LoginBody(BaseModel):
    username: str # username or email
    password: str

@router.post("/register")
def register(body: RegisterBody):
    con = get_db_conn()
    cur = con.cursor()
    try: 
        cur.execute(
            "INSERT INTO users (username, email, first_name, last_name, password_hash)"
            " VALUES (%s,%s,%s,%s,%s) RETURNING id",
            (body.username, body.email, body.first_name, body.last_name, hash_password(body.password))
        )
        user_id = cur.fetchone()[0]
        con.commit()  
    except psycopg2.errors.UniqueViolation:
        con.rollback()
        cur.close()
        con.close()
        raise HTTPException(status_code=409, detail="username or email already taken")
    cur.close()
    con.close()
    return {
        "token": create_token(user_id),
        "user": {"id": user_id, "username": body.username, "email": body.email, "firstName": body.first_name}
    }

@router.post("/login")
def login(body: LoginBody):
    con = get_db_conn()
    cur = con.cursor()
    cur.execute(
        "SELECT id, username, email, first_name, password_hash FROM  users WHERE username=%s OR email=%s",
        (body.username, body.username)
    )
    row = cur.fetchone()
    cur.close()
    con.close()
    if not row or not verify_password(body.password, row[4]):
        raise HTTPException(status_code=401, detail="invalid credentials")
    return {
        "token": create_token(row[0]),
        "user": {"id": row[0], "username": row[1], "email": row[2], "firstName": row[3]}
    }

@router.get("/me")
def me(user=Depends(get_current_user)):
    return user

def _project_row_to_dict(cur, row, username):
    pid, name, steps, used_json, used_model_json, deleted_at, last_opened_at, is_pinned = row
    cur.execute("SELECT id, name FROM project_images WHERE project_id=%s", (pid,))
    images = [{"id": r[0], "name": r[1]} for r in cur.fetchall()]
    cur.execute("SELECT id, name FROM project_models WHERE project_id=%s", (pid,))
    models = [{"id": r[0], "name": r[1]} for r in cur.fetchall()]
    cur.execute("SELECT id, name, xml_content, corrected FROM mei_files WHERE project_id=%s", (pid,))
    mei = [{"id": r[0], "name": r[1], "xmlContent": r[2], "corrected": bool(r[3])}
           for r in cur.fetchall()]
    return {
        "id": pid, "name": name, "user": username,
        "stepsUnlocked": steps,
        "usedImageNames": json.loads(used_json),
        "usedModelNames": json.loads(used_model_json or "[]"),
        "images": images, "models": models, "meiFiles": mei,
        "annotations": [], "deletedAt": deleted_at,
        "lastOpenedAt": str(last_opened_at) if last_opened_at else None,
        "isPinned": bool(is_pinned),
    }

def _log_activity(cur, project_id: int, action_type: str, detail: str = ""):
    cur.execute(
        "INSERT INTO activity_log (project_id, action_type, detail) VALUES (%s, %s, %s)",
        (project_id, action_type, detail)
    )
@router.get("/projects")
def list_projects(user=Depends(get_current_user)):
    con = get_db_conn()
    cur = con.cursor()
    cur.execute(
        "SELECT id, name, steps_unlocked, used_image_names, used_model_names, deleted_at, " \
        " last_opened_at, is_pinned"
        " FROM projects WHERE user_id=%s",
        (user["id"],)
    )
    rows = cur.fetchall()
    result=[_project_row_to_dict(cur, row, user["username"]) for row in rows]
    cur.close()
    con.close()
    return result

@router.get("/projects/{project_id}/activity")
def get_activity(project_id: int, user=Depends(get_current_user)):
    con = get_db_conn(); cur = con.cursor()
    cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id, ))
    row = cur.fetchone()
    if not row or row[0] != user["id"]:
        cur.close(); con.close()
        raise HTTPException(status_code=404)
    cur.execute(
        "SELECT action_type, detail, created_at FROM activity_log"
        " WHERE project_id=%s ORDER BY created_at DESC LIMIT 100",
        (project_id,)
    )
    entries = [{"actionType": r[0], "detail": r[1], "createdAt": str(r[2])} for r in cur.fetchall()]
    cur.close(); con.close()
    return entries

class CreateProjectBody(BaseModel):
    name: str

@router.post("/projects")
def create_project(body: CreateProjectBody, user=Depends(get_current_user)):
    con = get_db_conn()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO projects (user_id, name) VALUES (%s,%s) RETURNING id", 
        (user["id"], body.name))
    pid = cur.fetchone()[0]
    con.commit()
    cur.close()
    con.close()
    return {"id": pid, "name": body.name, "user": user["username"],
            "images": [], "models": [], "meiFiles": [], "annotations": [],
            "stepsUnlocked": 0, "usedImageNames": [], "usedModelNames": [], "deletedAt": None}

class UpdateProjectBody(BaseModel):
    name: Optional[str] = None
    stepsUnlocked: Optional[int] = None
    usedImageNames: Optional[list] = None
    usedModelNames: Optional[list] = None
    deletedAt: Optional[str] = None
    lastOpenedAt: Optional[str] = None
    isPinned: Optional[bool] = None

@router.put("/projects/{project_id}")
def update_project(project_id: int, body: UpdateProjectBody, user=Depends(get_current_user)):
    con = get_db_conn()
    cur = con.cursor()
    cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id, ))
    row = cur.fetchone()

    if not row or row[0] != user["id"]:
        cur.close()
        con.close()
        raise HTTPException(status_code=404)
    if body.name is not None:
        cur.execute("UPDATE projects SET name=%s WHERE id=%s", (body.name, project_id))
    if body.stepsUnlocked is not None:
        cur.execute("UPDATE projects SET steps_unlocked=%s WHERE id=%s", (body.stepsUnlocked, project_id))
        _log_activity(cur, project_id, "step_unlocked", str(body.stepsUnlocked))
    if body.usedImageNames is not None:
        cur.execute("UPDATE projects SET used_image_names=%s WHERE id=%s",
                    (json.dumps(body.usedImageNames), project_id))
    if body.usedModelNames is not None:
        cur.execute("UPDATE projects SET used_model_names=%s WHERE id=%s",
                    (json.dumps(body.usedModelNames), project_id))
    if body.deletedAt is not None:
        cur.execute("UPDATE projects SET deleted_at=%s WHERE id=%s", (body.deletedAt, project_id))
    if body.lastOpenedAt is not None:
        cur.execute("UPDATE projects SET last_opened_at=%s WHERE id=%s",
                    (body.lastOpenedAt, project_id))
    if body.isPinned is not None:
        cur.execute("UPDATE projects SET is_pinned=%s WHERE id=%s", (body.isPinned, project_id))
    con.commit()
    cur.close()
    con.close()
    return {"ok": True}

@router.post("/projects/{project_id}/restore")
def restore_project(project_id: int, user=Depends(get_current_user)):
    con = get_db_conn()
    cur = con.cursor()
    cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id,))
    row = cur.fetchone()
    if not row or row[0] != user["id"]:
        cur.close()
        con.close()
        raise HTTPException(status_code=404)
    cur.execute("UPDATE projects SET deleted_at=NULL WHERE id=%s", (project_id, ))
    con.commit()
    cur.close()
    con.close()
    return {"ok": True}

@router.delete("/projects/{project_id}")
def permanently_delete_project(project_id: int, user=Depends(get_current_user)):
    con = get_db_conn(); cur = con.cursor()
    cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id, ))
    row = cur.fetchone()
    if not row or row[0] != user["id"]:
        cur.close(); con.close()
        raise HTTPException(status_code=404)
    cur.execute("DELETE FROM project_images WHERE project_id=%s", (project_id,))
    cur.execute("DELETE FROM project_models WHERE project_id=%s", (project_id,))
    cur.execute("DELETE FROM mei_files WHERE project_id=%s", (project_id,))
    cur.execute("DELETE FROM projects WHERE id=%s", (project_id,))
    con.commit()
    cur.close(); con.close()
    return {"ok": True}

# image endpoints

@router.post("/projects/{project_id}/images")
async def upload_image(project_id: int, file: UploadFile = FAPIFile(...), user=Depends(get_current_user)):
    con = get_db_conn()
    cur = con.cursor()
    cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id,))
    row = cur.fetchone()
    if not row or row[0] != user["id"]:
        cur.close()
        con.close()
        raise HTTPException(status_code=404)
    image_id = _uuid.uuid4().hex
    image_bytes = await file.read()
    mime_type = file.content_type or "image/png"
    cur.execute(
        "INSERT INTO project_images (id, project_id, name, mime_type, data) VALUES (%s,%s,%s,%s,%s)",
        (image_id, project_id, file.filename, mime_type, psycopg2.Binary(image_bytes))
    )
    _log_activity(cur, project_id, "image_imported", file.filename)
    con.commit()
    cur.close()
    con.close()
    return {"id": image_id, "name": file.filename}

@router.get("/images/{image_id}")
def get_image(image_id: str, user=Depends(get_current_user)):
    con = get_db_conn()
    cur = con.cursor()
    cur.execute("SELECT data, mime_type FROM project_images WHERE id=%s", (image_id, ))
    row = cur.fetchone()
    cur.close()
    con.close()
    if not row:
        raise HTTPException(status_code=404)
    return Response(content=bytes(row[0]), media_type=row[1] or "image/png")

@router.delete("/projects/{project_id}/images/{image_id}")
def delete_image(project_id: int, image_id: str, user=Depends(get_current_user)):
    con = get_db_conn()
    cur = con.cursor()
    cur.execute(
        "SELECT id FROM project_images WHERE id=%s AND project_id=%s", (image_id, project_id)
    )
    if cur.fetchone():
        cur.execute("DELETE FROM project_images WHERE id=%s", (image_id,))
        con.commit()
    cur.close()
    con.close()
    return {"ok": True}

# mei + model endpoints

class AddMeiBody(BaseModel):
    name: str
    xmlContent: str

@router.post("/projects/{project_id}/mei")
def add_mei(project_id: int, body: AddMeiBody, user=Depends(get_current_user)):
    con = get_db_conn()
    cur = con.cursor()
    cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id, ))
    row = cur.fetchone()
    if not row or row[0] != user["id"]:
        cur.close()
        con.close()
        raise HTTPException(status_code=404)
    mei_id = _uuid.uuid4().hex
    cur.execute(
        "INSERT INTO mei_files (id, project_id, name, xml_content) VALUES (%s,%s,%s,%s)",
        (mei_id, project_id, body.name, body.xmlContent))
    _log_activity(cur, project_id, "mei_produced", body.name)
    con.commit()
    cur.close()
    con.close()
    return {"id": mei_id}

class UpdateMeiBody(BaseModel):
    corrected: Optional[bool] = None

@router.patch("/projects/{project_id}/mei/{mei_id}")
def update_mei(project_id: int, mei_id: str, body: UpdateMeiBody, user=Depends(get_current_user)):
    con = get_db_conn(); cur = con.cursor()
    cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id, ))
    row = cur.fetchone()
    if not row or row[0] != user["id"]:
        cur.close(); con.close()
        raise HTTPException(status_code=404)
    if body.corrected is not None:
        cur.execute("UPDATE mei_files SET corrected=%s WHERE id=%s AND project_id=%s",
                    (1 if body.corrected else 0, mei_id, project_id))
        if body.corrected:
            cur.execute("SELECT name FROM mei_files WHERE id=%s", 
                        (mei_id,))
            name_row = cur.fetchone()
            _log_activity(cur, project_id, "mei_corrected", name_row[0] if name_row else "")
    con.commit()
    cur.close(); con.close()
    return {"ok": True}

# models

class AddModelBody(BaseModel):
    name: str

@router.post("/projects/{project_id}/models")
def add_model(project_id: int, body: AddModelBody, user=Depends(get_current_user)):
    con = get_db_conn()
    cur = con.cursor()
    cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id, ))
    row = cur.fetchone()
    if not row or row[0] != user["id"]:
        cur.close()
        con.close()
        raise HTTPException(status_code=404)
    model_id = _uuid.uuid4().hex
    cur.execute(
        "INSERT INTO project_models (id, project_id, name) VALUES (%s,%s,%s)",
        (model_id, project_id, body.name)
    )
    _log_activity(cur, project_id, "model_added", body.name)
    con.commit()
    cur.close()
    con.close()
    return {"id": model_id, "name": body.name}

@router.get("/projects/{project_id}/export")
def export_project(project_id: int, user=Depends(get_current_user)):
    con = get_db_conn(); cur = con.cursor()
    cur.execute("SELECT user_id, name FROM projects WHERE id=%s", (project_id, ))
    row = cur.fetchone()
    if not row or row[0] != user["id"]:
        cur.close(); con.close()
        raise HTTPException(status_code=404)
    project_name = row[1]
    cur.execute("SELECT name, mime_type, data FROM project_images WHERE project_id=%s", (project_id, ))
    images = cur.fetchall()
    cur.execute("SELECT name, xml_content FROM mei_files WHERE project_id=%s", (project_id, ))
    mei_files = cur.fetchall()
    cur.close(); con.close()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_name, _mime, data in images:
            zf.writestr(f"images/{img_name}", bytes(data))
        for mei_name, xml_content in mei_files:
            zf.writestr(f"mei/{mei_name}", xml_content or "")
    buf.seek(0)
    safe_name = project_name.replace(" ", "_")
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.zip"'}
    )