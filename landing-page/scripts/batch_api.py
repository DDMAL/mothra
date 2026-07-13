"""Batch text-alignment bridge - proxies to text-service's /batch-run,
wrapping mothra-text's run_chain.py (subprocess). Mirrors text_api.py's
single-image pattern: server-to-server call, project-ownership check
before touching image bytes, browser never talks to text-service directly.
"""
import json
import urllib.error
import urllib.request
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from auth_api import get_current_user, db_cursor, require_project_owner
from text_api import TEXT_API_URL, _stream_multipart

router = APIRouter()

def _project_image_by_id(cur, project_id: int, image_id: str) -> tuple[str, bytes, str]:
    cur.execute(
        "SELECT name, data, mime_type FROM project_images WHERE id=%s AND project_id=%s",
        (image_id, project_id),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"image {image_id} not found")
    return row[0], bytes(row[1]), (row[2] or "image/png")

class BatchRunBody(BaseModel):
    image_ids: list[str]
    folios: list[str]
    source_id: int
    segmentation_model: Optional[str] = None
    recognition_model: Optional[str] = None
    device: str = "cpu"
    column_count: Optional[int] = None
    column_bimodal_threshold: float = 0.5


@router.post("/projects/{project_id}/text-batch/run")
def run_text_batch(project_id: int, body: BatchRunBody, user=Depends(get_current_user)):
    if len(body.image_ids) != len(body.folios):
        raise HTTPException(status_code=400, detail="image_ids and folios must be the same length")
    
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        images = [_project_image_by_id(cur, project_id, iid) for iid in body.image_ids]

    fields = {
        "folios": json.dumps(body.folios),
        "source_id": str(body.source_id),
        "device": body.device,
        "column_bimodal_threshold": str(body.column_bimodal_threshold),
    }
    if body.segmentation_model:
        fields["segmentation_model"] = body.segmentation_model
    if body.recognition_model:
        fields["recognition_model"] = body.recognition_model
    if body.column_count is not None:
        fields["column_count"] = str(body.column_count)
    files = [("images", name, mime, data) for name, data, mime in images]

    def generate():
        def event(obj):
            return f"data: {json.dumps(obj)}\n\n"
        try:
            for line in _stream_multipart(f"{TEXT_API_URL}/batch-run", fields=fields, files=files, timeout=1800):
                if line.startswith("data: "):
                    yield event(json.loads(line[len("data: "):]))
        except urllib.error.URLError as exc:
            yield event({"type": "error", "message": f"text-service at {TEXT_API_URL} is unreachable: {exc}"})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@router.get("/projects/{project_id}/text-batch/{batch_id}/download")
def download_text_batch(project_id: int, batch_id: str, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
    try:
        with urllib.request.urlopen(f"{TEXT_API_URL}/batch-download/{batch_id}", timeout=60) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        raise HTTPException(status_code=exc.code, detail="batch result not found or expired") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"text-service unreachable: {exc}") from exc
    return Response(
        content=data, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="batch-{batch_id}.zip"'},
    )