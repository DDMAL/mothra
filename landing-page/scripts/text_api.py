"""Text-finding bridge - proxies to the standalone text-service, which
wraps the mothra-text line-segmentation/OCR pipeline.

Mirrors ic_api.py's pattern: server-to-server call over plain HTTP (stdlib
urllib, no extra dependency), env var for the service URL, ownership check
on the project before touching its image bytes.
"""
from __future__ import annotations

import json
import io
import os
import urllib.error
import urllib.request
import uuid as _uuid
from typing import Optional
from config import TEXT_API_URL

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

from auth_api import get_current_user, db_cursor, require_project_owner

router = APIRouter()

MUSIC_CLASS_ID = 1
TEXT_CLASS_ID = 0 # mothra's raw YOLO numbering
MOTHRA_TEXT_MASK_CLASS_ID = 1 # mothra-text's OWN numbering

def _project_image(project_id: int, image_name: str, user_id: int) -> tuple[str, bytes, str]:
    """Return (image_id, data, mime_type) for a project image the user owns."""
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user_id)
        cur.execute(
            "SELECT id, data, mime_type FROM project_images WHERE project_id=%s AND name=%s", (project_id, image_name),
        )
        img = cur.fetchone()
        if not img:
            raise HTTPException(status_code=404, detail="image not found")
        return img[0], bytes(img[1]), (img[2] or "image/png")

def _music_boxes_for_image(project_id: int, image_name: str, image_bytes: bytes) -> list[list[float]]:
    """Return YOLO-detected music-region boxes in absolute pixel coords
    [xmin, ymin, xmax, ymax], or [] if there's no annotation yet.

    Best-effort — any failure here should not block text-finding.
    """
    try:
        with db_cursor() as (con, cur):
            cur.execute(
                "SELECT yolo_txt FROM annotations"
                " WHERE project_id=%s AND image_name=%s"
                " ORDER BY created_at DESC LIMIT 1",
                (project_id, image_name),
            )
            row = cur.fetchone()
        if not row or not row[0].strip():
            return []
        with Image.open(io.BytesIO(image_bytes)) as im:
            page_w, page_h = im.size
        boxes = []
        for line in row[0].strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls, cx, cy, bw, bh = (float(p) for p in parts[:5])
            except ValueError:
                continue
            if int(cls) != MUSIC_CLASS_ID:
                continue
            px_cx, px_cy = cx * page_w, cy * page_h
            px_bw, px_bh = bw * page_w, bh * page_h
            boxes.append([
                px_cx - px_bw / 2, px_cy - px_bh / 2,
                px_cx + px_bw / 2, px_cy + px_bh / 2,
            ])
        return boxes
    except Exception:
        return []

def _mask_json_for_image(project_id: int, image_name: str, image_bytes: bytes) -> Optional[str]:
    """Auto-derive a mothra-text mask JSON string from this image's own
    latest YOLO annotation row, using Mothra's raw "text" class (classId 0
    in Mothra's own numbering — unrelated to mothra-text's own classId 1
    for "text to keep" in its mask JSON; this function bridges the two).

    Returns None (not an empty-list JSON) when there's no annotation yet or
    no text boxes found, so callers can distinguish "no mask available"
    from "mask with zero boxes" — the latter would black out the entire
    image, which is never the intent when nothing was detected.

    Best-effort — any failure here should not block text-finding.
    """
    try:
        with db_cursor() as (con, cur):
            cur.execute(
                "SELECT yolo_txt FROM annotations"
                " WHERE project_id=%s AND image_name=%s"
                " ORDER BY created_at DESC LIMIT 1",
                (project_id, image_name),
            )
            row = cur.fetchone()
        if not row or not row[0].strip():
            return None
        with Image.open(io.BytesIO(image_bytes)) as im:
            page_w, page_h = im.size
        annotations = []
        for line in row[0].strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls, cx, cy, bw, bh = (float(p) for p in parts[:5])
            except ValueError:
                continue
            if int(cls) != TEXT_CLASS_ID:
                continue
            px_w, px_h = bw * page_w, bh * page_h
            px_x = (cx * page_w) - px_w / 2
            px_y = (cy * page_h) - px_h / 2
            annotations.append({
                "classId": MOTHRA_TEXT_MASK_CLASS_ID,
                "bbox": [px_x, px_y, px_w, px_h],
            })
        if not annotations:
            return None
        return json.dumps({"annotations": annotations})
    except Exception:
        return None

def _stream_multipart(url: str, fields: dict[str, str], files: list[tuple], timeout: int=120):
    # Default covers this module's own single-image call below (real runs
    # complete in well under a minute) — batch_api.py's multi-file batch
    # call passes its own much larger explicit timeout, unaffected by this.
    # Kept well below Celery's task-visibility timeout: urlopen's timeout is
    # a per-read socket timeout, not a hard deadline, so a peer that goes
    # unreachable mid-connection (e.g. its container gets recreated) can
    # otherwise tie up a worker thread for the full duration before Python
    # ever raises — confirmed by actually hitting this during Docker testing.
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

def stream_text_finding(
        project_id: int,
        image_id: str,
        image_name: str,
        image_bytes: bytes,
        mime_type: str,
        column_count: Optional[int] = None,
        segmentation_model: Optional[str] = None,
        recognition_model: Optional[str] = None,
        device: str = "cpu",
        column_bimodal_threshold: float = 0.5,
        masking_enabled: bool = True,
        mask_padding: int = 15,
        music_overlap_filter_enabled: bool = True,
        mask_json_override: Optional[str] = None,
        source_id: Optional[int] = None,
        folio_override: Optional[str] = None,
    ):
    """Run text-finding for one image, yielding raw event dicts (not
    SSE-formatted) and persisting the result to text_alignments on completion.

    Shared by the standalone endpoint below and by inference_api.py's
    combined processing pipeline, which calls this immediately after YOLO
    produces this same image's boxes — mothra-text's music-region filter
    (_music_boxes_for_image) is re-derived here from whatever annotation
    row is freshest at call time, so it picks up boxes YOLO just committed.
    """
    music_boxes = _music_boxes_for_image(project_id, image_name, image_bytes)
    mask_json = mask_json_override
    if masking_enabled and not mask_json:
        mask_json = _mask_json_for_image(project_id, image_name, image_bytes)
    collected_logs: list[str] = []
    fields = {
        "folio": folio_override or image_name,
        "music_boxes": json.dumps(music_boxes),
        "device": device,
        "column_bimodal_threshold": str(column_bimodal_threshold),
        "masking_enabled": "true" if masking_enabled else "false",
        "mask_padding": str(mask_padding),
        "music_overlap_filter_enabled": "true" if music_overlap_filter_enabled else "false",
    }
    if source_id is not None:
        fields["source_id"] = str(source_id)
    if column_count is not None:
        fields["column_count"] = str(column_count)
    if segmentation_model:
        fields["segmentation_model"] = segmentation_model
    if recognition_model:
        fields["recognition_model"] = recognition_model
    if mask_json:
        fields["mask_json"] = mask_json
    try:
        for line in _stream_multipart(
            f"{TEXT_API_URL}/run",
            fields=fields,
            files=[("image", image_name, mime_type, image_bytes)],
        ):
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[len("data: "):])
            if ev.get("type") == "log":
                collected_logs.append(ev.get("message", ""))
            if ev.get("type") == "result":
                alignment = ev["text_alignment"]
                with db_cursor() as (con, cur):
                    aid = _uuid.uuid4().hex
                    cur.execute(
                        "INSERT INTO text_alignments"
                        " (id, project_id, image_id, image_name, alignment_json,"
                        " median_line_spacing, syllable_count, log_text)"
                        " VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                        (aid, project_id, image_id, image_name,
                            json.dumps(alignment),
                            alignment.get("median_line_spacing", 0.0),
                            len(alignment.get("syl_boxes", [])),
                            "\n".join(collected_logs),
                        ),
                    )
                    con.commit()
                ev = {**ev, "alignment_id": aid}
            yield ev
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="ignore")
        try:
            detail = json.loads(detail).get("detail", detail)
        except json.JSONDecodeError:
            pass
        yield {"type": "error", "message": f"text-service rejected the request (HTTP {exc.code}): {detail}"}
    except urllib.error.URLError as exc:
        yield {"type": "error", "message": f"text-service at {TEXT_API_URL} is unreachable: {exc}"}


@router.post("/projects/{project_id}/text-finding/run")
def run_text_finding(project_id: int, image_name: str, column_count: Optional[int] = None,
    segmentation_model: Optional[str] = None,
    recognition_model: Optional[str] = None,
    device: str = "cpu",
    column_bimodal_threshold: float = 0.5, 
    user=Depends(get_current_user),
    masking_enabled: bool = True,
    mask_padding: int = 15,
    music_overlap_filter_enabled: bool = True,
    source_id: Optional[int] = None,
    folio: Optional[str] = None,
):
    """Stream single-image text-finding results for one project image over SSE.

    Fetches the named image, then forwards it to `stream_text_finding` and
    re-emits each yielded event as a `data: {...}\\n\\n` frame. Unlike the
    batch/predict paths this runs synchronously in-request, not as a Celery
    job — single-image text-finding is fast enough not to need the job queue.
    """
    image_id, image_bytes, mime_type = _project_image(project_id, image_name, user["id"])

    def generate():
        def event(obj):
            return f"data: {json.dumps(obj)}\n\n"
        for ev in stream_text_finding(project_id, image_id, image_name, image_bytes, mime_type, column_count=column_count,
            segmentation_model=segmentation_model,
            recognition_model=recognition_model,
            device=device,
            column_bimodal_threshold=column_bimodal_threshold,
            masking_enabled=masking_enabled,
            mask_padding=mask_padding,
            music_overlap_filter_enabled=music_overlap_filter_enabled,
            source_id=source_id,
            folio_override=folio,
        ):
            yield event(ev)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/projects/{project_id}/text-alignments/{alignment_id}")
async def get_text_alignment(
    project_id: int,
    alignment_id: str,
    user=Depends(get_current_user),
):
    with db_cursor() as (con, cur):
        cur.execute(
            "SELECT t.alignment_json, t.image_name, t.log_text"
            " FROM text_alignments t"
            " JOIN projects p ON p.id = t.project_id"
            " WHERE t.id = %s AND t.project_id = %s AND p.user_id = %s",
            (alignment_id, project_id, user["id"]),
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404)
    return {"alignmentJson": row[0], "imageName": row[1], "logText": row[2]}