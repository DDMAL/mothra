"""Text-finding bridge - proxies to the standalone text-service, which
wraps the mothra-text line-segmentation/OCR pipeline.

Mirrors ic_api.py's pattern: server-to-server call over plain HTTP (stdlib
urllib, no extra dependency), env var for the service URL, ownership check
on the project before touching its image bytes.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid as _uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from auth_api import get_current_user, get_db_conn, release_db_conn

router = APIRouter()

TEXT_API_URL = os.environ.get("TEXT_API_URL", "http://localhost:8002").rstrip("/")

def _project_image(project_id: int, image_name: str, user_id: int) -> tuple[str, bytes, str]:
    """Return (image_id, data, mime_type) for a project image the user owns."""
    con = get_db_conn()
    cur = con.cursor()
    try:
        cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="project not found")
        if row[0] != user_id:
            raise HTTPException(status_code=403, detail="not your project")
        cur.execute(
            "SELECT id, data, mime_type FROM project_images WHERE project_id=%s AND name=%s", (project_id, image_name),
        )
        img = cur.fetchone()
        if not img:
            raise HTTPException(status_code=404, detail="image not found")
        return img[0], bytes(img[1]), (img[2] or "image/png")
    finally:
        cur.close()
        release_db_conn(con)

def _stream_multipart(url: str, fields: dict[str, str], files: list[tuple], timeout: int=600):
    """POST multipart/form-data and yield decoded response lines (SSE passthrough)."""
    boundary = _uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        body += value.encode() + b"\r\n"
    for name, filename, ctype, data in files:
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        body += f"Content-Type: {ctype}\r\n\r\n".encode()
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(url, data=bytes(body), method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            yield raw_line.decode()

@router.post("/projects/{project_id}/text-finding/run")
def run_text_finding(project_id: int, image_name: str, user=Depends(get_current_user)):
    image_id, image_bytes, mime_type = _project_image(project_id, image_name, user["id"])
    
    def generate():
        def event(obj):
            return f"data: {json.dumps(obj)}\n\n"
        try:
            for line in _stream_multipart(f"{TEXT_API_URL}/run", fields={"folio": image_name}, files=[("image", image_name, mime_type, image_bytes)],
                                          ):
                if not line.startswith("data: "): 
                    continue
                ev = json.loads(line[len("data: "):])
                if ev.get("type") == "result":
                    alignment = ev["text_alignment"]
                    con = get_db_conn()
                    cur = con.cursor()
                    try:
                        aid = _uuid.uuid4().hex
                        cur.execute(
                            "INSERT INTO text_alignments"
                            " (id, project_id, image_id, image_name, alignment_json,"
                            " median_line_spacing, syllable_count)"
                            " VALUES (%s,%s,%s,%s,%s,%s,%s)",
                            (aid, project_id, image_id, image_name,
                                json.dumps(alignment),
                                alignment.get("median_line_spacing", 0.0),
                                len(alignment.get("syl_boxes", [])),
                            ),
                        )
                        con.commit()
                    finally:
                        cur.close()
                        release_db_conn(con)
                    ev = {**ev, "alignment_id": aid}
                yield event(ev)
        except urllib.error.URLError as exc:
            yield event({"type": "error", "message": f"text-service at {TEXT_API_URL} is unreachable: {exc}"})
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )