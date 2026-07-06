from fastapi import APIRouter, Depends, HTTPException, Header, Request, UploadFile, File as FAPIFile
from fastapi.responses import Response
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
from slowapi import Limiter
from slowapi.util import get_remote_address
import psycopg2, psycopg2.extras, psycopg2.errors, os, secrets, json, mimetypes, hashlib, base64
import uuid as _uuid
from datetime import datetime, timedelta
from jose import jwt, JWTError
import bcrypt
import io, zipfile

limiter = Limiter(key_func=get_remote_address)
# connection pooling
from psycopg2 import pool as _pg_pool

_db_pool: "_pg_pool.ThreadedConnectionPool | None" = None

def _get_pool() -> "_pg_pool.ThreadedConnectionPool":
    global _db_pool
    if _db_pool is None:
        _db_pool = _pg_pool.ThreadedConnectionPool(
            minconn=2, maxconn=15, dsn=os.environ["DATABASE_URL"]
        )
    return _db_pool

def get_db_conn():
    return _get_pool().getconn()

def release_db_conn(con) -> None:
    _get_pool().putconn(con)

router = APIRouter()

SECRET_KEY = os.environ.get("MOTHRA_SECRET", secrets.token_hex(32))
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 72
STORAGE_QUOTA_BYTES = int(os.getenv("STORAGE_QUOTA_MB", "500")) * 1024 * 1024

MODELS_DIR = Path(__file__).parent / "stored_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
NEON_MANIFESTS_DIR = Path(__file__).parent.parent / "public" / "neon" / "samples" / "manifests"
NEON_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

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
            corrected INTEGER DEFAULT 0,
            image_name TEXT
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
    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_logs (
                id SERIAL PRIMARY KEY,
                project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                log_type TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
        )    
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            id TEXT PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id),
            image_id TEXT,
            image_name TEXT NOT NULL,
            yolo_txt TEXT NOT NULL,
            model_id TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS text_alignments (
            id TEXT PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id),
            image_id TEXT,
            image_name TEXT NOT NULL,
            alignment_json TEXT NOT NULL,
            median_line_spacing REAL DEFAULT 0,
            syllable_count INTEGER DEFAULT 0,
            log_text TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    # performance: db indexes
    for _idx in [
        "CREATE INDEX IF NOT EXISTS idx_project_images_pid ON project_images(project_id)",
        "CREATE INDEX IF NOT EXISTS idx_project_models_pid ON project_models(project_id)",
        "CREATE INDEX IF NOT EXISTS idx_mei_files_pid      ON mei_files(project_id)",
        "CREATE INDEX IF NOT EXISTS idx_annotations_pid    ON annotations(project_id)",
        "CREATE INDEX IF NOT EXISTS idx_activity_log_pid   ON activity_log(project_id)",
        "CREATE INDEX IF NOT EXISTS idx_projects_user_id   ON projects(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_text_alignments_pid ON text_alignments(project_id)"
    ]:
        cur.execute(_idx)
    con.commit()
    cur.close()
    release_db_conn(con)

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
        release_db_conn(con)

    con = get_db_conn()
    cur = con.cursor()
    try:
        cur.execute("ALTER TABLE projects ADD COLUMN last_opened_at TIMESTAMPTZ")
        con.commit()
    except psycopg2.errors.DuplicateColumn:
        con.rollback()
    finally:
        cur.close()
        release_db_conn(con)

    con = get_db_conn()
    cur = con.cursor()
    try:
        cur.execute("ALTER TABLE projects ADD COLUMN is_pinned BOOLEAN DEFAULT FALSE")
        con.commit()
    except psycopg2.errors.DuplicateColumn:
        con.rollback()
    finally:
        cur.close()
        release_db_conn(con)

    con = get_db_conn()
    cur = con.cursor()
    try:
        cur.execute("ALTER TABLE projects ADD COLUMN created_at TIMESTAMPTZ DEFAULT NOW()")
        con.commit()
    except psycopg2.errors.DuplicateColumn:
        con.rollback()
    finally:
        cur.close()
        release_db_conn(con)

    con = get_db_conn()
    cur = con.cursor()
    try:
        cur.execute("ALTER TABLE project_images ADD COLUMN created_at TIMESTAMPTZ DEFAULT NOW()")
        con.commit()
    except psycopg2.errors.DuplicateColumn:
        con.rollback()
    finally:
        cur.close()
        release_db_conn(con)

    con = get_db_conn()
    cur = con.cursor()
    try:
        cur.execute("ALTER TABLE mei_files ADD COLUMN image_name TEXT")
        con.commit()
    except psycopg2.errors.DuplicateColumn:
        con.rollback()
    finally:
        cur.close()
        release_db_conn(con)

    con = get_db_conn()
    cur = con.cursor()
    try:
        cur.execute("ALTER TABLE project_models ADD COLUMN file_path TEXT")
        con.commit()
    except psycopg2.errors.DuplicateColumn:
        con.rollback()
    finally:
        cur.close()
        release_db_conn(con)

    con = get_db_conn()
    cur = con.cursor()
    try:
        cur.execute("ALTER TABLE projects ADD COLUMN used_annotation_names TEXT DEFAULT '[]'")
        con.commit()
    except psycopg2.errors.DuplicateColumn:
        con.rollback()
    finally:
        cur.close()
        release_db_conn(con)

    con = get_db_conn()
    cur = con.cursor()
    try:
        cur.execute("ALTER TABLE text_alignments ADD COLUMN log_text TEXT")
        con.commit()
    except psycopg2.errors.DuplicateColumn:
        con.rollback()
    finally:
        cur.close()
        release_db_conn(con)

    # startup cleanup of neon manifests to prevent excess accumulation
    import time as _time
    _now = _time.time()
    for _f in NEON_MANIFESTS_DIR.glob("*.jsonld"):
        if _now - _f.stat().st_mtime > 86400:
            _f.unlink(missing_ok=True)
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

def _make_edit_token(project_id: int, mei_id: str) -> str:
    payload = {
        "project_id": project_id, 
        "mei_id": mei_id,
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def _verify_edit_token(token: str, project_id: int, mei_id: str) -> bool:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("project_id") == project_id and payload.get("mei_id") == mei_id
    except Exception:
        return False

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
    release_db_conn(con)
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
@limiter.limit("5/minute")
def register(request: Request, body: RegisterBody):
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
        release_db_conn(con)
        raise HTTPException(status_code=409, detail="username or email already taken")
    cur.close()
    release_db_conn(con)
    return {
        "token": create_token(user_id),
        "user": {"id": user_id, "username": body.username, "email": body.email, "firstName": body.first_name}
    }

@router.post("/login")
@limiter.limit("10/minute")
def login(request: Request, body: LoginBody):
    con = get_db_conn()
    cur = con.cursor()
    cur.execute(
        "SELECT id, username, email, first_name, password_hash FROM  users WHERE username=%s OR email=%s",
        (body.username, body.username)
    )
    row = cur.fetchone()
    cur.close()
    release_db_conn(con)
    if not row or not verify_password(body.password, row[4]):
        raise HTTPException(status_code=401, detail="invalid credentials")
    return {
        "token": create_token(row[0]),
        "user": {"id": row[0], "username": row[1], "email": row[2], "firstName": row[3]}
    }

@router.post("/auth/refresh")
def refresh_token(user=Depends(get_current_user)):
    return {"access_token": create_token(user["id"]), "token_type": "bearer"}

@router.get("/me")
def me(user=Depends(get_current_user)):
    return user

def _project_row_to_dict(cur, row, username):
    pid, name, steps, used_json, used_model_json, deleted_at, last_opened_at, is_pinned, used_annotation_json = row
    cur.execute("SELECT id, name FROM project_images WHERE project_id=%s", (pid,))
    images = [{"id": r[0], "name": r[1]} for r in cur.fetchall()]
    cur.execute("SELECT id, name FROM project_models WHERE project_id=%s", (pid,))
    models = [{"id": r[0], "name": r[1]} for r in cur.fetchall()]
    cur.execute("SELECT id, name, xml_content, corrected, image_name FROM mei_files WHERE project_id=%s", (pid,))
    mei = [{"id": r[0], "name": r[1], "xmlContent": r[2], "corrected": bool(r[3]), "imageName": r[4]}
           for r in cur.fetchall()]
    cur.execute(
        "SELECT id, image_id, image_name FROM annotations WHERE project_id=%s", (pid,)
    )
    annotations = [
        {
            "id": r[0],
            "imageName": r[2],
            "imageSrc": f"/api/images/{r[1]}" if r[1] else None,
            "txtName": f"annotation-{r[0]}.txt",
            "jsonName": "",
        }
        for r in cur.fetchall()
    ]
    cur.execute(
         "SELECT id, image_id, image_name, median_line_spacing, syllable_count"
        " FROM text_alignments WHERE project_id=%s", (pid,)
    )
    text_alignments = [
        {"id": r[0], "imageName": r[2], "imageSrc": f"/api/images/{r[1]}" if r[1] else None, "medianLineSpacing": r[3], "syllableCount": r[4]}
        for r in cur.fetchall()
    ]
    return {
        "id": pid, "name": name, "user": username,
        "stepsUnlocked": steps,
        "usedImageNames": json.loads(used_json),
        "usedModelNames": json.loads(used_model_json or "[]"),
        "images": images, "models": models, "meiFiles": mei,
        "annotations": annotations, "deletedAt": deleted_at,
        "lastOpenedAt": str(last_opened_at) if last_opened_at else None,
        "isPinned": bool(is_pinned),
        "usedAnnotationNames": json.loads(used_annotation_json or "[]"),
        "textAlignments": text_alignments,
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
        "SELECT id, name, steps_unlocked, used_image_names, used_model_names, deleted_at, "
        " last_opened_at, is_pinned, used_annotation_names"
        " FROM projects WHERE user_id=%s",
        (user["id"],)
    )
    rows = cur.fetchall()
    if not rows:
        cur.close(); release_db_conn(con); return []
    
    pids = tuple(r[0] for r in rows)

    cur.execute("SELECT project_id, id, name FROM  project_images WHERE project_id IN %s", (pids,))
    images_by_pid: dict = {}
    for pid, iid, iname in cur.fetchall():
        images_by_pid.setdefault(pid, []).append({"id": iid, "name": iname})

    cur.execute("SELECT project_id, id, name FROM project_models WHERE project_id IN %s", (pids,))
    models_by_pid: dict = {}
    for pid, mid, mname in cur.fetchall():
        models_by_pid.setdefault(pid, []).append({"id": mid, "name": mname})

    cur.execute(
        "SELECT project_id, id, name, xml_content, corrected, image_name"
        " FROM mei_files WHERE project_id IN %s", (pids,)
    )
    mei_by_pid: dict = {}
    for pid, fid, fname, xml, corr, iname in cur.fetchall():
        mei_by_pid.setdefault(pid, []).append(
            {"id": fid, "name": fname, "xmlContent": xml, "corrected": bool(corr), "imageName": iname}
        )

    cur.execute(
        "SELECT project_id, id, image_id, image_name FROM annotations WHERE project_id IN %s",
        (pids,)
    )
    ann_by_pid: dict = {}
    for pid, aid, img_id, img_name in cur.fetchall():
        ann_by_pid.setdefault(pid, []).append({
            "id": aid, "imageName": img_name,
            "imageSrc": f"/api/images/{img_id}" if img_id else None,
            "txtName": f"annotation-{aid}.txt", "jsonName": "",
        })

    cur.execute(
        "SELECT project_id, id, image_id, image_name, median_line_spacing, syllable_count"
        " FROM text_alignments WHERE project_id IN %s", (pids,)
    )
    text_by_pid: dict = {}
    for pid, tid, img_id, img_name, spacing, syl_count in cur.fetchall():
        text_by_pid.setdefault(pid, []).append({
            "id": tid, "imageName": img_name,
            "imageSrc": f"/api/images/{img_id}" if img_id else None,
            "medianLineSpacing": spacing, "syllableCount": syl_count,
        })

    result = []
    for row in rows:
        pid = row[0]
        result.append({
            "id": pid, "name": row[1], "user": user["username"],
            "stepsUnlocked": row[2],
            "usedImageNames": json.loads(row[3]),
            "usedModelNames": json.loads(row[4] or "[]"),
            "images": images_by_pid.get(pid, []),
            "models": models_by_pid.get(pid, []),
            "meiFiles": mei_by_pid.get(pid, []),
            "annotations": ann_by_pid.get(pid, []),
            "deletedAt": row[5],
            "lastOpenedAt": str(row[6]) if row[6] else None,
            "isPinned": bool(row[7]),
            "usedAnnotationNames": json.loads(row[8] or "[]"),
            "textAlignments": text_by_pid.get(pid, []),
        })
    cur.close()
    release_db_conn(con)
    return result

@router.get("/projects/{project_id}")
def get_project(project_id: int, user=Depends(get_current_user)):
    con = get_db_conn(); cur = con.cursor()
    cur.execute(
        "SELECT id, name, steps_unlocked, used_image_names, used_model_names, deleted_at,"
        " last_opened_at, is_pinned, used_annotation_names"
        " FROM projects WHERE id=%s AND user_id=%s",
        (project_id, user["id"])
    )
    row = cur.fetchone()
    if not row:
        cur.close(); release_db_conn(con)
        raise HTTPException(status_code=404)
    result = _project_row_to_dict(cur, row, user["username"])
    cur.close(); release_db_conn(con)
    return result

@router.get("/projects/{project_id}/activity")
def get_activity(project_id: int, user=Depends(get_current_user)):
    con = get_db_conn(); cur = con.cursor()
    cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id, ))
    row = cur.fetchone()
    if not row or row[0] != user["id"]:
        cur.close(); release_db_conn(con)
        raise HTTPException(status_code=404)
    cur.execute(
        "SELECT action_type, detail, created_at FROM activity_log"
        " WHERE project_id=%s ORDER BY created_at DESC LIMIT 100",
        (project_id,)
    )
    entries = [{"actionType": r[0], "detail": r[1], "createdAt": str(r[2])} for r in cur.fetchall()]
    cur.close(); release_db_conn(con)
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
    release_db_conn(con)
    return {"id": pid, "name": body.name, "user": user["username"],
            "images": [], "models": [], "meiFiles": [], "annotations": [],
            "stepsUnlocked": 0, "usedImageNames": [], "usedModelNames": [], 
            "deletedAt": None, "usedAnnotationNames": []}

class UpdateProjectBody(BaseModel):
    name: Optional[str] = None
    stepsUnlocked: Optional[int] = None
    usedImageNames: Optional[list] = None
    usedModelNames: Optional[list] = None
    deletedAt: Optional[str] = None
    lastOpenedAt: Optional[str] = None
    isPinned: Optional[bool] = None
    usedAnnotationNames: Optional[list] = None

@router.put("/projects/{project_id}")
def update_project(project_id: int, body: UpdateProjectBody, user=Depends(get_current_user)):
    con = get_db_conn()
    cur = con.cursor()
    cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id, ))
    row = cur.fetchone()

    if not row or row[0] != user["id"]:
        cur.close()
        release_db_conn(con)
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
    if body.usedAnnotationNames is not None:
        cur.execute("UPDATE projects SET used_annotation_names=%s WHERE id=%s",
                    (json.dumps(body.usedAnnotationNames), project_id))
    con.commit()
    cur.close()
    release_db_conn(con)
    return {"ok": True}

@router.post("/projects/{project_id}/restore")
def restore_project(project_id: int, user=Depends(get_current_user)):
    con = get_db_conn()
    cur = con.cursor()
    cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id,))
    row = cur.fetchone()
    if not row or row[0] != user["id"]:
        cur.close()
        release_db_conn(con)
        raise HTTPException(status_code=404)
    cur.execute("UPDATE projects SET deleted_at=NULL WHERE id=%s", (project_id, ))
    con.commit()
    cur.close()
    release_db_conn(con)
    return {"ok": True}

@router.delete("/projects/{project_id}")
def permanently_delete_project(project_id: int, user=Depends(get_current_user)):
    con = get_db_conn(); cur = con.cursor()
    cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id, ))
    row = cur.fetchone()
    if not row or row[0] != user["id"]:
        cur.close(); release_db_conn(con)
        raise HTTPException(status_code=404)
    cur.execute("DELETE FROM annotations WHERE project_id=%s", (project_id,))
    cur.execute("DELETE FROM project_logs WHERE project_id=%s", (project_id,))
    cur.execute("DELETE FROM activity_log WHERE project_id=%s", (project_id,))
    cur.execute("DELETE FROM project_images WHERE project_id=%s", (project_id,))
    cur.execute("DELETE FROM project_models WHERE project_id=%s", (project_id,))
    cur.execute("DELETE FROM mei_files WHERE project_id=%s", (project_id,))
    cur.execute("DELETE FROM projects WHERE id=%s", (project_id,))
    con.commit()
    cur.close(); release_db_conn(con)
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
        release_db_conn(con)
        raise HTTPException(status_code=404)
    image_id = _uuid.uuid4().hex
    image_bytes = await file.read()

    cur.execute("""
        SELECT COALESCE(SUM(octet_length(data)), 0)
        FROM project_images
        WHERE project_id IN (SELECT id FROM projects WHERE user_id = %s)
    """, (user["id"], ))
    current_bytes = cur.fetchone()[0]

    if current_bytes + len(image_bytes) > STORAGE_QUOTA_BYTES:
        cur.close(); release_db_conn(con)
        raise HTTPException(
            status_code=413,
            detail=f"Storage quota exceeded ({STORAGE_QUOTA_BYTES // (1024*1024)} MB limit)"
        )
    
    mime_type = file.content_type or "image/png"
    cur.execute(
        "INSERT INTO project_images (id, project_id, name, mime_type, data) VALUES (%s,%s,%s,%s,%s)",
        (image_id, project_id, file.filename, mime_type, psycopg2.Binary(image_bytes))
    )
    _log_activity(cur, project_id, "image_imported", file.filename)
    con.commit()
    cur.close()
    release_db_conn(con)
    return {"id": image_id, "name": file.filename}

@router.get("/images/{image_id}")
def get_image(image_id: str, user=Depends(get_current_user)):
    con = get_db_conn()
    cur = con.cursor()
    cur.execute("SELECT data, mime_type FROM project_images WHERE id=%s "
                " AND project_id IN (SELECT id FROM projects WHERE user_id=%s)", 
                (image_id, user["id"] ))
    row = cur.fetchone()
    cur.close()
    release_db_conn(con)
    if not row:
        raise HTTPException(status_code=404)
    return Response(content=bytes(row[0]), media_type=row[1] or "image/png")

@router.get("/images/{image_id}/meta")
def get_image_meta(image_id: str, user=Depends(get_current_user)):
    con = get_db_conn()
    cur = con.cursor()
    cur.execute(
        "SELECT name, mime_type, octet_length(data), created_at FROM project_images "
        " WHERE id=%s AND project_id IN (SELECT id FROM projects WHERE user_id=%s)",
        (image_id, user["id"])
    )
    row = cur.fetchone()
    cur.close(); release_db_conn(con)
    if not row:
        raise HTTPException(status_code=404)
    return {
        "name": row[0],
        "mimeType": row[1] or "image/png",
        "sizeBytes": row[2],
        "createdAt": row[3].isoformat() if row[3] else None,
    }

@router.delete("/projects/{project_id}/images/{image_id}")
def delete_image(project_id: int, image_id: str, user=Depends(get_current_user)):
    con = get_db_conn()
    cur = con.cursor()
    cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id, ))
    row = cur.fetchone()
    if not row or row[0] != user["id"]:
        cur.close(); release_db_conn(con)
        raise HTTPException(status_code=404)
    cur.execute(
        "SELECT id FROM project_images WHERE id=%s AND project_id=%s", (image_id, project_id)
    )
    if not cur.fetchone():
        cur.close(); release_db_conn(con)
        raise HTTPException(status_code=404, detail="Image not found")
    cur.execute("DELETE FROM project_images WHERE id=%s", (image_id,))
    _log_activity(cur, project_id, "image_deleted", image_id)
    con.commit()
    cur.close()
    release_db_conn(con)
    return {"ok": True}

@router.delete("/projects/{project_id}/annotations/{annotation_id}")
def delete_annotation(project_id: int, annotation_id: str, user=Depends(get_current_user)):
    con = get_db_conn()
    cur = con.cursor()
    cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id,))
    row = cur.fetchone()
    if not row or row[0] != user["id"]:
        cur.close(); release_db_conn(con)
        raise HTTPException(status_code=404)
    cur.execute("DELETE FROM annotations WHERE id=%s AND project_id=%s", (annotation_id, project_id))
    con.commit()
    cur.close()
    release_db_conn(con)
    return {"ok": True}

@router.delete("/projects/{project_id}/mei/{mei_id}")
def delete_mei_file(project_id: int, mei_id: str, user=Depends(get_current_user)):
    con = get_db_conn()
    cur = con.cursor()
    cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id,))
    row = cur.fetchone()
    if not row or row[0] != user["id"]:
        cur.close(); release_db_conn(con)
        raise HTTPException(status_code=404)
    cur.execute("DELETE FROM mei_files WHERE id=%s AND project_id=%s", (mei_id, project_id))
    con.commit()
    cur.close()
    release_db_conn(con)
    return {"ok": True}

# mei + model endpoints

class AddMeiBody(BaseModel):
    name: str
    xmlContent: str
    imageName: Optional[str] = None
    logs: Optional[list[str]] = None

@router.post("/projects/{project_id}/mei")
def add_mei(project_id: int, body: AddMeiBody, user=Depends(get_current_user)):
    con = get_db_conn()
    cur = con.cursor()
    cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id, ))
    row = cur.fetchone()
    if not row or row[0] != user["id"]:
        cur.close()
        release_db_conn(con)
        raise HTTPException(status_code=404)
    mei_id = _uuid.uuid4().hex
    cur.execute(
        "INSERT INTO mei_files (id, project_id, name, xml_content, image_name) VALUES (%s,%s,%s,%s,%s)",
        (mei_id, project_id, body.name, body.xmlContent, body.imageName))
    if body.logs:
        content = "\n".join(body.logs)
        cur.execute(
            "INSERT INTO project_logs (project_id, log_type, content) VALUES (%s, %s, %s)",
            (project_id, "encoding", content)
        )
    _log_activity(cur, project_id, "mei_produced", body.name)
    con.commit()
    cur.close()
    release_db_conn(con)
    return {"id": mei_id}

class UpdateMeiBody(BaseModel):
    corrected: Optional[bool] = None
    xmlContent: Optional[str] = None

@router.patch("/projects/{project_id}/mei/{mei_id}")
def update_mei(project_id: int, mei_id: str, body: UpdateMeiBody, user=Depends(get_current_user)):
    con = get_db_conn(); cur = con.cursor()
    cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id, ))
    row = cur.fetchone()
    if not row or row[0] != user["id"]:
        cur.close(); release_db_conn(con)
        raise HTTPException(status_code=404)
    if body.xmlContent is not None:
        cur.execute("UPDATE mei_files SET xml_content=%s WHERE id=%s AND project_id=%s",
                    (body.xmlContent, mei_id, project_id))
    if body.corrected is not None:
        cur.execute("UPDATE mei_files SET corrected=%s WHERE id=%s AND project_id=%s",
                    (1 if body.corrected else 0, mei_id, project_id))
        if body.corrected:
            cur.execute("SELECT name FROM mei_files WHERE id=%s", 
                        (mei_id,))
            name_row = cur.fetchone()
            _log_activity(cur, project_id, "mei_corrected", name_row[0] if name_row else "")
    con.commit()
    cur.close(); release_db_conn(con)
    return {"ok": True}

@router.get("/projects/{project_id}/mei/{mei_id}/content")
def get_mei_content(project_id: int, mei_id: str, token: str):
    if not _verify_edit_token(token, project_id, mei_id):
        raise HTTPException(status_code=403, detail="invalid or expired edit token")
    con = get_db_conn(); cur = con.cursor()
    cur.execute("SELECT xml_content FROM mei_files WHERE id=%s AND project_id=%s", (mei_id, project_id))
    row = cur.fetchone()
    cur.close(); release_db_conn(con)
    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="MEI not found")
    return Response(content=row[0], media_type="application/xml")

@router.put("/projects/{project_id}/mei/{mei_id}/content")
async def put_mei_content(project_id: int, mei_id: str, token: str, request: Request):
    if not _verify_edit_token(token, project_id, mei_id):
        raise HTTPException(status_code=403, detail="invalid or expired edit token")
    xml_content = (await request.body()).decode("utf-8")
    con = get_db_conn(); cur = con.cursor()
    cur.execute("UPDATE mei_files SET xml_content=%s WHERE id=%s AND project_id=%s",
                (xml_content, mei_id, project_id))
    con.commit(); cur.close(); release_db_conn(con)
    return {"ok": True}

@router.post("/projects/{project_id}/mei/{mei_id}/edit-session")
def create_edit_session(project_id: int, mei_id: str, user=Depends(get_current_user)):
   
   # proactive cleanup of neon manifests to prevent excess accumulation
    import time as _time
    _now = _time.time()
    for _f in NEON_MANIFESTS_DIR.glob("*.jsonld"):
        if _now - _f.stat().st_mtime > 86400:
            _f.unlink(missing_ok=True)

    con = get_db_conn(); cur = con.cursor()
    cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id, ))
    row = cur.fetchone()
    if not row or row[0] != user["id"]:
        cur.close(); release_db_conn(con)
        raise HTTPException(status_code=404)
    cur.execute("SELECT name, image_name FROM mei_files WHERE id=%s AND project_id=%s",
                (mei_id, project_id))
    mei_row = cur.fetchone()
    if not mei_row: 
        cur.close(); release_db_conn(con)
        raise HTTPException(status_code=404, detail="MEI not found")
    mei_name, image_name = mei_row

    image_data_uri = None
    if image_name:
        cur.execute("SELECT data, mime_type FROM project_images WHERE project_id=%s AND name=%s",
                    (project_id, image_name))
        img_row = cur.fetchone()
        if img_row:
            img_data, mime_type = img_row
            mime = mime_type or "image/jpeg"
            image_data_uri = f"data:{mime};base64,{base64.b64encode(bytes(img_data)).decode()}"
    cur.close(); release_db_conn(con)

    edit_token = _make_edit_token(project_id, mei_id)
    session_id = _uuid.uuid4().hex[:8]
    manifest_id = str(_uuid.uuid4())
    annotation_id = str(_uuid.uuid4())

    content_url = f"/api/projects/{project_id}/mei/{mei_id}/content?token={edit_token}"
    image_ref = image_data_uri or ""
    manifest = {
        "@context": [
            "http://www.w3.org/ns/anno.jsonld",
            {
                "schema": "http://schema.org/",
                "title": "schema:name",
                "timestamp": "schema:dateModified",
                "image": {"@id": "schema:image", "@type": "@id"},
                "mei_annotations": {"@id": "Annotation", "@type": "@id", "@container": "@list"},
            },
        ],
        "@id": f"urn:uuid:{manifest_id}",
        "title": mei_name,
        "image": image_ref,
        "mei_annotations": [{
            "id": f"urn:uuid:{annotation_id}",
            "type": "Annotation",
            "body": content_url,
            "target": image_ref,
        }]
    }

    NEON_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    (NEON_MANIFESTS_DIR / f"{session_id}.jsonld").write_text(json.dumps(manifest))
    return {"session_id": session_id, "manifest_id": manifest_id}

# models

@router.post("/projects/{project_id}/models")
async def add_model(
    project_id: int, 
    file: UploadFile = FAPIFile(...), 
    user=Depends(get_current_user)
):
    con = get_db_conn()
    cur = con.cursor()
    cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id, ))
    row = cur.fetchone()
    if not row or row[0] != user["id"]:
        cur.close()
        release_db_conn(con)
        raise HTTPException(status_code=404)
    model_id = _uuid.uuid4().hex
    dest_dir = MODELS_DIR / str(project_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = dest_dir / f"{model_id}.pt"
    model_bytes = await file.read()
    file_path.write_bytes(model_bytes)
    cur.execute(
        "INSERT INTO project_models (id, project_id, name, file_path) VALUES (%s,%s,%s, %s)",
        (model_id, project_id, file.filename, str(file_path))
    )
    _log_activity(cur, project_id, "model_added", file.filename)
    con.commit()
    cur.close()
    release_db_conn(con)
    return {"id": model_id, "name": file.filename}

@router.delete("/projects/{project_id}/models/{model_id}")
def delete_model(project_id: int, model_id: str, user=Depends(get_current_user)):
    con = get_db_conn(); cur = con.cursor()
    cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id,))
    row = cur.fetchone()
    if not row or row[0] != user["id"]:
        cur.close(); release_db_conn(con)
        raise HTTPException(status_code=404)
    cur.execute(
        "SELECT file_path FROM project_models WHERE id=%s AND project_id=%s",
        (model_id, project_id)
    )
    row = cur.fetchone()
    if not row:
        cur.close(); release_db_conn(con)
        raise HTTPException(status_code=404, detail="model not found")
    file_path = row[0]
    if file_path:
        Path(file_path).unlink(missing_ok=True)
    cur.execute("DELETE FROM project_models WHERE id=%s", (model_id,))
    _log_activity(cur, project_id, "model_deleted", model_id)
    con.commit()
    cur.close(); release_db_conn(con)
    return {"ok": True}

@router.get("/projects/{project_id}/annotations/{annotation_id}")
async def get_annotation_txt(
    project_id: int,
    annotation_id: str,
    user=Depends(get_current_user),
):
    con = get_db_conn()
    cur = con.cursor()
    cur.execute(
        "SELECT a.yolo_txt, a.image_name"
        " FROM annotations a"
        " JOIN projects p ON p.id = a.project_id"
        " WHERE a.id = %s AND a.project_id = %s AND p.user_id = %s",
        (annotation_id, project_id, user["id"]),
    )
    row = cur.fetchone()
    cur.close(); release_db_conn(con)
    if not row:
        raise HTTPException(status_code=404)
    return {"yoloTxt": row[0], "imageName": row[1]}

@router.get("/projects/{project_id}/text-alignments/{alignment_id}")
async def get_text_alignment(
    project_id: int,
    alignment_id: str,
    user=Depends(get_current_user),
):
    con = get_db_conn()
    cur = con.cursor()
    cur.execute(
        "SELECT t.alignment_json, t.image_name, t.log_text"
        " FROM text_alignments t"
        " JOIN projects p ON p.id = t.project_id"
        " WHERE t.id = %s AND t.project_id = %s AND p.user_id = %s",
        (alignment_id, project_id, user["id"]),
    )
    row = cur.fetchone()
    cur.close(); release_db_conn(con)
    if not row:
        raise HTTPException(status_code=404)
    return {"alignmentJson": row[0], "imageName": row[1], "logText": row[2]}
    

@router.get("/projects/{project_id}/export")
def export_project(project_id: int, user=Depends(get_current_user)):
    con = get_db_conn(); cur = con.cursor()
    cur.execute("SELECT user_id, name FROM projects WHERE id=%s", (project_id, ))
    row = cur.fetchone()
    if not row or row[0] != user["id"]:
        cur.close(); release_db_conn(con)
        raise HTTPException(status_code=404)
    project_name = row[1]
    cur.execute("SELECT name, mime_type, data FROM project_images WHERE project_id=%s", (project_id, ))
    images = cur.fetchall()
    cur.execute("SELECT name, xml_content FROM mei_files WHERE project_id=%s", (project_id, ))
    mei_files = cur.fetchall()
    cur.close(); release_db_conn(con)

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

@router.post("/projects/{project_id}/duplicate")
def duplicate_project(project_id: int, current_user=Depends(get_current_user)):
    con = get_db_conn(); cur = con.cursor()
    cur.execute(
        "SELECT name FROM projects WHERE id = %s AND user_id = %s AND deleted_at IS NULL",
        (project_id, current_user["id"])
    )
    row = cur.fetchone()
    if not row:
        cur.close(); release_db_conn(con)
        raise HTTPException(status_code=404, detail="project not found")

    new_name = f"{row[0]} (copy)"
    now = datetime.utcnow()
    cur.execute(
        "INSERT INTO projects (user_id, name, steps_unlocked, last_opened_at, created_at)"
        " VALUES (%s, %s, 0, %s, %s) RETURNING id",
        (current_user["id"], new_name, now, now)
    )
    new_id = cur.fetchone()[0]

    cur.execute("SELECT name, mime_type, data FROM project_images WHERE project_id=%s", (project_id,))
    for img_name, mime, data in cur.fetchall():
        cur.execute(
            "INSERT INTO project_images (id, project_id, name, mime_type, data, created_at)"
            " VALUES (%s, %s, %s, %s, %s, %s)",
            (str(_uuid.uuid4()), new_id, img_name, mime, data, now)
        )

    import shutil

    cur.execute("SELECT name, file_path FROM project_models WHERE project_id=%s", (project_id,))
    for model_name, file_path in cur.fetchall():
        new_model_id = str(_uuid.uuid4())
        new_file_path = None
        if file_path and Path(file_path).exists():
            new_model_dir = MODELS_DIR / str(new_id)
            new_model_dir.mkdir(parents=True, exist_ok=True)
            new_file_path = str(new_model_dir / f"{new_model_id}.pt")
            shutil.copy2(file_path, new_file_path)
        cur.execute(
            "INSERT INTO project_models (id, project_id, name, file_path) VALUES (%s, %s, %s, %s)",
            (new_model_id, new_id, model_name, new_file_path)
        )

    con.commit()

    cur.execute(
        "SELECT id, name, steps_unlocked, used_image_names, used_model_names, deleted_at,"
        " last_opened_at, is_pinned, used_annotation_names"
        " FROM projects WHERE id=%s",
        (new_id,)
    )
    result = _project_row_to_dict(cur, cur.fetchone(), current_user["username"])
    cur.close(); release_db_conn(con)
    return result


@router.get("/projects/{project_id}/logs/download")
def download_project_logs(project_id: int, user=Depends(get_current_user)):
    con = get_db_conn(); cur = con.cursor()
    cur.execute("SELECT user_id, name FROM projects WHERE id=%s", (project_id, ))
    row = cur.fetchone()
    if not row or row[0] != user["id"]:
        cur.close(); release_db_conn(con)
        raise HTTPException(status_code=404)
    project_name = row[1]

    cur.execute(
        "SELECT action_type, detail, created_at FROM activity_log WHERE project_id=%s ORDER BY created_at ASC",
        (project_id,)
    )
    activity_rows = cur.fetchall()

    cur.execute(
        "SELECT content, created_at FROM project_logs WHERE project_id=%s AND log_type='encoding' ORDER BY created_at ASC",
        (project_id,)
    )
    encoding_rows = cur.fetchall()
    cur.close(); release_db_conn(con)

    activity_lines = [
        f"[{r[2]}] {r[0]}: {r[1]}" for r in activity_rows
    ] or ["no activity recorded"]
    activity_text = "\n".join(activity_lines)

    encoding_sections = []
    for content, created_at in encoding_rows:
        encoding_sections.append(f"--- Run: {created_at} --- \n{content}")
    encoding_text = "\n\n".join(encoding_sections) if encoding_sections else "no encoding logs recorded"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("activity_log.txt", activity_text)
        zf.writestr("encoding_logs.txt", encoding_text)
    buf.seek(0)
    safe_name = project_name.replace(" ", "_")
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}_logs.zip"'}
    )
