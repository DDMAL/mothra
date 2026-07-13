"""Model upload/delete endpoints and file-path resolution.

Split out of auth_api.py — models are their own concern (YOLO .pt files,
Kraken segmentation/recognition files), tagged by `kind` and stored on
local disk under auth_api.MODELS_DIR.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File as FAPIFile, Form
from pathlib import Path
from typing import Optional
import uuid as _uuid
import json
from pydantic import BaseModel

from auth_api import get_current_user, db_cursor, require_project_owner, _log_activity, MODELS_DIR

router = APIRouter()

ALLOWED_MODEL_KINDS = {"yolo", "segmentation", "recognition", "text_mask"}

class ClassMapBody(BaseModel):
    class_map: dict[str, str]


def get_model_file_path(cur, project_id: int, model_id: str, kind: str) -> Optional[tuple]:
    """Resolve a project_models row to (file_path, name), scoped to both
    project and kind so a client can't point e.g. a YOLO model id at the
    OCR slot. Returns None if not found, wrong kind, or file_path is empty.
    Takes a cursor (not a connection) so callers can reuse their own
    request-scoped transaction instead of checking out a second connection.
    """
    cur.execute(
        "SELECT file_path, name, class_map FROM project_models WHERE id=%s AND project_id=%s AND kind=%s",
        (model_id, project_id, kind),
    )
    row = cur.fetchone()
    if not row or not row[0]:
        return None
    return row[0], row[1], row[2]

import hashlib
from model_validation import inspect_yolo_checkpoint

@router.post("/projects/{project_id}/models")
async def add_model(
    project_id: int,
    file: UploadFile = FAPIFile(...),
    kind: str = Form("yolo"),
    user=Depends(get_current_user)
):
    if kind not in ALLOWED_MODEL_KINDS:
        raise HTTPException(status_code=400, detail=f"invalid model kind: {kind}")
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        model_id = _uuid.uuid4().hex
        dest_dir = MODELS_DIR / str(project_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(file.filename).suffix if file.filename else ""
        file_path = dest_dir / f"{model_id}{ext}"
        model_bytes = await file.read()
        file_path.write_bytes(model_bytes)

        class_map_json, names = None, None
        if kind == "yolo":
            try:
                inspection = inspect_yolo_checkpoint(str(file_path))
            except ValueError as e:
                file_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail=f"invalid YOLO model: {e}")
            names = inspection["names"]
            if inspection["class_map"] is not None:
                class_map_json = json.dumps(inspection["class_map"])
        
        file_hash = hashlib.sha256(model_bytes).hexdigest()
        cur.execute(
            "INSERT INTO project_models (id, project_id, name, file_path, kind, class_map, file_hash) VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (model_id, project_id, file.filename, str(file_path), kind, class_map_json, file_hash)
        )
        _log_activity(cur, project_id, "model_added", f"{file.filename} ({kind})")
        con.commit()
        return {
            "id": model_id, "name": file.filename, "kind": kind,
            "classMap": json.loads(class_map_json) if class_map_json else None,
            "needsClassMapping": kind == "yolo" and class_map_json is None,
            "rawClassNames": names if kind == "yolo" and class_map_json is None else None,
        }


@router.delete("/projects/{project_id}/models/{model_id}")
def delete_model(project_id: int, model_id: str, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        cur.execute(
            "SELECT file_path FROM project_models WHERE id=%s AND project_id=%s",
            (model_id, project_id)
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="model not found")
        file_path = row[0]
        if file_path:
            Path(file_path).unlink(missing_ok=True)
        cur.execute("DELETE FROM project_models WHERE id=%s", (model_id,))
        _log_activity(cur, project_id, "model_deleted", model_id)
        con.commit()
        return {"ok": True}

@router.put("/projects/{project_id}/models/{model_id}/class-map")
def set_class_map(project_id: int, model_id: str, body: ClassMapBody, user=Depends(get_current_user)):
    if not set(body.class_map.values()) <= {"text", "music", "staves"}:
        raise HTTPException(status_code=400, detail="class_map values must be text/music/staves")
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        cur.execute(
            "UPDATE project_models SET class_map=%s WHERE id=%s AND project_id=%s",
            (json.dumps(body.class_map), model_id, project_id)
        )
        con.commit()
        return {"ok": True}