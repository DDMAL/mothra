"""Project image upload/fetch/delete endpoints."""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File as FAPIFile
from fastapi.responses import Response
import psycopg2
import uuid as _uuid

from auth_api import get_current_user, db_cursor, require_project_owner, _log_activity, STORAGE_QUOTA_BYTES

router = APIRouter()


@router.post("/projects/{project_id}/images")
async def upload_image(project_id: int, file: UploadFile = FAPIFile(...), user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        image_id = _uuid.uuid4().hex
        image_bytes = await file.read()

        cur.execute("""
            SELECT COALESCE(SUM(octet_length(data)), 0)
            FROM project_images
            WHERE project_id IN (SELECT id FROM projects WHERE user_id = %s)
        """, (user["id"], ))
        current_bytes = cur.fetchone()[0]

        if current_bytes + len(image_bytes) > STORAGE_QUOTA_BYTES:
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
        return {"id": image_id, "name": file.filename}


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


@router.delete("/projects/{project_id}/images/{image_id}")
def delete_image(project_id: int, image_id: str, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        cur.execute(
            "SELECT id FROM project_images WHERE id=%s AND project_id=%s", (image_id, project_id)
        )
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Image not found")
        cur.execute("DELETE FROM project_images WHERE id=%s", (image_id,))
        _log_activity(cur, project_id, "image_deleted", image_id)
        con.commit()
        return {"ok": True}
