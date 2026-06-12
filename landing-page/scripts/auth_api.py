from fastapi import APIRouter, Depends, HTTPException, Header, UploadFile, File as FAPIFile
from fastapi.responses import Response
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import sqlite3, os, secrets, json, shutil, mimetypes, hashlib, base64
import uuid as _uuid
from datetime import datetime, timedelta
from jose import jwt, JWTError
import bcrypt

router = APIRouter()


SECRET_KEY = os.environ.get("MOTHRA_SECRET", secrets.token_hex(32))
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 72

DATA_DIR = Path.home() / ".mothra"
DATA_DIR.mkdir(exist_ok=True)
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "mothra.db"



def init_db():
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            first_name TEXT,
            last_name TEXT,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            name TEXT NOT NULL,
            steps_unlocked INTEGER DEFAULT 0,
            used_image_names TEXT DEFAULT '[]', -- JSON array
            deleted_at TEXT
        );
        CREATE TABLE IF NOT EXISTS project_images (
            id TEXT PRIMARY KEY, -- UUID
            project_id INTEGER REFERENCES projects(id),
            name TEXT NOT NULL,
            filename TEXT -- path under UPLOADS_DIR
        );
        CREATE TABLE IF NOT EXISTS project_models (
            id TEXT PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id),
            name TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS mei_files (
            id TEXT PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id),
            name TEXT NOT NULL,
            xml_content TEXT,
            corrected INTEGER DEFAULT 0
        );
    """)
    con.close()

init_db()

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
        payload = jwt.encode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_is = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="invalid token")
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT id, username, email, first_name, last_name from users where id=?", (user_id,)
    ).fetchone()
    con.close()
    if not row:
        raise HTTPException(status_code=401, detail="user not found")
    return {"id": row[0], "username": row[1], "email": row[2], "firstName": row[3], "lastName": row[4]}

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
    con = sqlite3.connect(DB_PATH)
    try: 
        con.execute(
            "INSERT INTO users (username, email, first_name, last_name, password_hash) VALUES (?,?,?,?,?)",
            (body.username, body.email, body.first_name, body.last_name, hash_password(body.password))
        )
        con.commit()
        user_id = con.execute("SELECT id FROM users WHERE username=?", (body.username,)).fetchone()[0]
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="username or email already taken")
    finally:
        con.close()
    return {
        "token": create_token(user_id),
        "user": {"id": user_id, "username": body.username, "email": body.email, "firstName": body.first_name}
    }

@router.post("/login")
def login(body: LoginBody):
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT id, username, email, first_name, password_hash FROM users WHERE username=? OR email=?",
        (body.username, body.username)
    ).fetchone()
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

def _project_row_to_dict(con, row, username):
    pid, name, steps, used_json, deleted_at = row
    images = [{"id": r[0], "name": r[1]} for r in
              con.execute("SELECT id, name FROM project_images WHERE project_id=?", (pid,))]
    models = [{"id": r[0], "name": r[1]} for r in
              con.execute("SELECT id, name FROM project_models WHERE project_id=?", (pid,))]
    mei = [{"id": r[0], "name": r[1], "xmlContent": r[2], "corrected": bool(r[3])} for r in
           con.execute("SELECT id, name, xml_content, corrected FROM mei_files WHERE project_id=?", (pid,))]
    return {
        "id": pid, "name": name, "user": username,
        "stepsUnlocked": steps, "usedImageNames": json.loads(used_json),
        "images": images, "models": models, "meiFiles": mei,
        "annotations": [], "deletedAt": deleted_at,
    }

@router.get("/projects")
def list_projects(user=Depends(get_current_user)):
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id, name, steps_unlocked, used_image_names, deleted_at FROM projects WHERE user_id=?",
        (user["id"],)
    ).fetchall()
    result=[_project_row_to_dict(con, row, user["username"]) for row in rows]
    con.close()
    return result

class CreateProjectBody(BaseModel):
    name: str

@router.post("/projects")
def create_project(body: CreateProjectBody, user=Depends(get_current_user)):
    con = sqlite3.connect(DB_PATH)
    cur = con.execute("INSERT INTO projects (user_id, name) VALUES (?,?)", (user["id"], body.name))
    con.commit()
    pid = cur.lastrowid
    con.close()
    return {"id": pid, "name": body.name, "user": user["username"],
            "images": [], "models": [], "meiFiles": [], "annotations": [],
            "stepsUnlocked": 0, "usedImageNames": [], "deletedAt": None}

class UpdateProjectBody(BaseModel):
    name: Optional[str] = None
    stepsUnlocked: Optional[int] = None
    usedImageNames: Optional[int] = None
    deletedAt: Optional[str] = None

@router.put("/projects/{project_id}")
def update_project(project_id: int, body: UpdateProjectBody, user=Depends(get_current_user)):
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT user_id FROM projects WHERE id=?", (project_id, )).fetchone()
    if not row or row[0] != user["id"]:
        raise HTTPException(status_code=404)
    if body.name is not None:
        con.execute("UPDATE projects SET name=? WHERE id=?", (body.name, project_id))
    if body.stepsUnlocked is not None:
        con.execute("UPDATE projects SET steps_unlocked=? WHERE id=?", (body.stepsUnlocked, project_id))
    if body.usedImageNames is not None:
        con.execute("UPDATE projects SET used_image_names=? WHERE id=?",
                    (json.dumps(body.usedImageNames), project_id))
    if body.deletedAt is not None:
        con.execute("UPDATE projects SET deleted_at=? WHERE id=?", (body.deletedAt, project_id))
    con.commit()
    con.close()
    return {"ok": True}

@router.post("/projects/{project_id}/restore")
def restore_project(project_id: int, user=Depends(get_current_user)):
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT user_id FROM projects WHERE id=?", (project_id,)).fetchone()
    if not row or row[0] != user["id"]:
        raise HTTPException(status_code=404)
    con.execute("UPDATE projects SET deleted_all=NULL WHERE id=?", (project_id, ))
    con.commit()
    con.close()
    return {"ok": True}

# image endpoints

@router.post("/projects/{project_id}/images")
async def upload_image(project_id: int, file: UploadFile = FAPIFile(...), user=Depends(get_current_user)):
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT user_id FROM projects WHERE id=?", (project_id,)).fetchone()
    if not row or row[0] != user["id"]:
        raise HTTPException(status_code=404)
    image_id = _uuid.uuid4().hex
    ext = Path(file.filename).suffix
    save_path = UPLOADS_DIR / f"{image_id}{ext}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    con.execute("INSERT INTO project_images (id, project_id, name, filename) VALUES (?,?,?,?)",
            (image_id, project_id, file.filename, save_path.name))
    con.commit()
    con.close()
    return {"id": image_id, "name": file.filename}

@router.get("/images/{image_id}")
def get_image(image_id: str, user=Depends(get_current_user)):
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT filename FROM project_images WHERE id=?", (image_id,)).fetchone()
    con.close()
    if not row:
        raise HTTPException(status_code=404)
    path = UPLOADS_DIR / row[0]
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return Response(content=path.read_bytes(), media_type=mime)

@router.delete("/projects/{project_id}/images/{image_id}")
def delete_image(project_id: int, image_id: str, user=Depends(get_current_user)):
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT filename FROM project_images WHERE id=? AND project_id=?", (image_id, project_id)
    ).fetchone()
    if row:
        (UPLOADS_DIR / row[0]).unlink(missing_ok=True)
        con.execute("DELETE FROM project_images WHERE id=?", (image_id,))
        con.commit()
    con.close()
    return {"ok": True}

# mei + model endpoints

class AddMeiBody(BaseModel):
    name: str
    xmlContent: str

@router.post("/projects/{project_id}/mei")
def add_mei(project_id: int, body: AddMeiBody, user=Depends(get_current_user)):
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT user_id FROM projects WHERE id=?", (project_id, )).fetchone()
    if not row or row[0] != user["id"]:
        raise HTTPException(status_code=404)
    mei_id = _uuid.uuid4().hex
    con.execute("INSERT INTO mei files (id, project_id, name, xml_content) VALUES (?,?,?,?)",
                (mei_id, project_id, body.name, body.xmlContent))
    con.commit()
    con.close()
    return {"id": mei_id}

# models

class AddModelBody(BaseModel):
    name: str

@router.post("/projects/{project_id}/models")
def add_model(project_id: int, body: AddModelBody, user=Depends(get_current_user)):
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT user_id FROM projects WHERE id=?", (project_id, )).fetchone()
    if not row or row[0] != user["id"]:
        raise HTTPException(status_code=404)
    model_id = _uuid.uuid4().hex
    con.execute("INSERT INTO project_models (id, project_id, name) VALUES (?,?,?)",
                (model_id, project_id, body.name))
    con.commit()
    con.close()
    return {"id": model_id, "name": body.name}