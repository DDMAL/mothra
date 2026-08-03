"""Project image upload/fetch/delete endpoints."""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File as FAPIFile, Form
from fastapi.responses import Response
from typing import Optional
import psycopg2
import uuid as _uuid
from pydantic import BaseModel

from auth_api import get_current_user, db_cursor, require_project_owner, _log_activity, STORAGE_QUOTA_BYTES

router = APIRouter()

class UpdateImageBody(BaseModel): 
    folio: Optional[str] = None

@router.post("/projects/{project_id}/images")
async def upload_image(
    project_id: int,
    file: UploadFile = FAPIFile(...),
    folio: Optional[str] = Form(None),
    source_id: Optional[str] = Form(None),
    source_name: Optional[str] = Form(None),
    original_file: Optional[UploadFile] = FAPIFile(None),
    user=Depends(get_current_user),
):
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        image_bytes = await file.read()
        original_bytes = await original_file.read() if original_file else None

        # Re-uploading a file with the same name reuses the existing image_id
        # (updates its bytes in place) instead of minting a new row - otherwise
        # annotations/text-alignments tied to the old id are orphaned while a
        # second, independent set accumulates under the new id.
        cur.execute(
            "SELECT id, octet_length(data) + COALESCE(octet_length(original_data), 0)"
            " FROM project_images WHERE project_id=%s AND name=%s",
            (project_id, file.filename),
        )
        existing = cur.fetchone()
        image_id = existing[0] if existing else _uuid.uuid4().hex
        existing_bytes = existing[1] if existing else 0

        cur.execute("""
            SELECT COALESCE(SUM(octet_length(data) + COALESCE(octet_length(original_data), 0)), 0)
            FROM project_images
            WHERE project_id IN (SELECT id FROM projects WHERE user_id = %s)
        """, (user["id"], ))
        current_bytes = cur.fetchone()[0]

        new_bytes_len = len(image_bytes) + (len(original_bytes) if original_bytes else 0)
        if current_bytes - existing_bytes + new_bytes_len > STORAGE_QUOTA_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Storage quota exceeded ({STORAGE_QUOTA_BYTES // (1024*1024)} MB limit)"
            )

        mime_type = file.content_type or "image/png"
        original_data = psycopg2.Binary(original_bytes) if original_bytes else None
        original_mime_type = original_file.content_type if (original_file and original_bytes) else None
        if existing:
            cur.execute(
                "UPDATE project_images SET mime_type=%s, data=%s, folio=%s, source_id=%s, source_name=%s,"
                " original_data=%s, original_mime_type=%s WHERE id=%s",
                (mime_type, psycopg2.Binary(image_bytes), folio or None, source_id or None,
                 source_name or None, original_data, original_mime_type, image_id)
            )
        else:
            cur.execute(
                "INSERT INTO project_images (id, project_id, name, mime_type, data, folio, source_id, source_name,"
                " original_data, original_mime_type)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (image_id, project_id, file.filename, mime_type, psycopg2.Binary(image_bytes),
                 folio or None, source_id or None, source_name or None, original_data, original_mime_type)
            )
        _log_activity(cur, project_id, "image_imported", file.filename)
        con.commit()
        return {
            "id": image_id, "name": file.filename, "folio": folio or None,
            "sourceId": source_id or None, "sourceName": source_name or None,
        }


@router.get("/images/{image_id}")
def get_image(image_id: str, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        cur.execute("SELECT data, mime_type FROM project_images WHERE id=%s "
                    " AND project_id IN (SELECT id FROM projects WHERE user_id=%s)",
                    (image_id, user["id"] ))
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404)
    return Response(content=bytes(row[0]), media_type=row[1] or "image/png")

@router.get("/images/{image_id}/original")
def get_original_image(image_id: str, user=Depends(get_current_user)):
    """Serves the pre-resize original if one was stored, else falls back to
    the working copy - always returns a viewable image either way, so callers
    (Neon's edit-session bootstrap, the MEI diff view) never need to branch."""
    with db_cursor() as (con, cur):
        cur.execute(
            "SELECT original_data, original_mime_type, data, mime_type FROM project_images WHERE id=%s "
            " AND project_id IN (SELECT id FROM projects WHERE user_id=%s)",
            (image_id, user["id"]),
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404)
    original_data, original_mime_type, data, mime_type = row
    image_bytes = original_data if original_data is not None else data
    # Fall back to the working-copy mime_type for rows written before
    # original_mime_type existed — best effort, may mislabel pre-migration
    # originals whose format differs from their resized working copy.
    resolved_mime_type = (original_mime_type if original_data is not None else mime_type) or mime_type
    return Response(content=bytes(image_bytes), media_type=resolved_mime_type or "image/png")

@router.get("/images/{image_id}/meta")
def get_image_meta(image_id: str, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        cur.execute(
            "SELECT name, mime_type, octet_length(data), created_at FROM project_images "
            " WHERE id=%s AND project_id IN (SELECT id FROM projects WHERE user_id=%s)",
            (image_id, user["id"])
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404)
    return {
        "name": row[0],
        "mimeType": row[1] or "image/png",
        "sizeBytes": row[2],
        "createdAt": row[3].isoformat() if row[3] else None,
    }

@router.put("/projects/{project_id}/images/{image_id}")
def update_image(project_id: int, image_id: str, body: UpdateImageBody, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        cur.execute(
            "SELECT id FROM project_images WHERE id=%s AND project_id=%s", (image_id, project_id)
        )
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Image not found")
        cur.execute(
            "UPDATE project_images SET folio=%s WHERE id=%s",
            (body.folio or None, image_id),
        )
        con.commit()
        return {"ok": True, "folio": body.folio or None}
    


@router.delete("/projects/{project_id}/images/{image_id}")
def delete_image(project_id: int, image_id: str, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        cur.execute(
            "SELECT id FROM project_images WHERE id=%s AND project_id=%s", (image_id, project_id)
        )
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Image not found")
        cur.execute("DELETE FROM annotations WHERE project_id=%s AND image_id=%s", (project_id, image_id))
        cur.execute("DELETE FROM text_alignments WHERE project_id=%s AND image_id=%s", (project_id, image_id))
        cur.execute("DELETE FROM project_images WHERE id=%s", (image_id,))
        _log_activity(cur, project_id, "image_deleted", image_id)
        con.commit()
        return {"ok": True}
