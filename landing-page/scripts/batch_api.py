"""Batch text-alignment bridge - proxies to text-service's /batch-run,
which wraps mothra-text's run_pipeline.run() in a loop. Mirrors
text_api.py's single-image pattern: server-to-server call, project-
ownership check before touching image bytes, browser never talks to
text-service directly.

Also runs YOLO layer-separation per folio before handing off to
text-service (mirroring inference_api.py's /predict, which already does
"YOLO then text-finding" for single images) so batch folios end up with
the same annotations + text_alignments rows a single image gets after
going through "annotate" - ready to continue through the normal
per-image IC/encoding flow, since batch IC doesn't exist yet.
"""
import io
import json
import urllib.error
import urllib.request
import uuid as _uuid
import zipfile
from typing import Literal, Optional

import numpy as np
from PIL import Image
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from auth_api import get_db_conn, get_current_user, release_db_conn, db_cursor, require_project_owner
from text_api import TEXT_API_URL, _stream_multipart, _music_boxes_for_image, _mask_json_for_image
from inference_api import resolve_yolo_models, write_annotation

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
    masking_enabled: bool = True
    mask_padding: int = 15
    # YOLO layer-separation settings, mirroring inference_api.py's PredictBody
    model_preset: Literal["medieval", "printed", "custom"] = "medieval"
    model_id: Optional[str] = None
    yolo_confidence_threshold: float = 0.5
    yolo_device: str = "cpu"
    text_music_confidence_threshold: Optional[float] = None
    text_music_device: Optional[str] = None
    stave_confidence_threshold: Optional[float] = None
    stave_device: Optional[str] = None


@router.post("/projects/{project_id}/text-batch/run")
def run_text_batch(project_id: int, body: BatchRunBody, user=Depends(get_current_user)):
    if len(body.image_ids) != len(body.folios):
        raise HTTPException(status_code=400, detail="image_ids and folios must be the same length")
    if body.model_preset == "printed":
        raise HTTPException(status_code=400, detail="printed text detection is not available yet!")
    if body.model_preset == "custom" and not body.model_id:
        raise HTTPException(status_code=400, detail="model_id is required when model_preset is 'custom'")
    
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        images = [_project_image_by_id(cur, project_id, iid) for iid in body.image_ids]

    def generate():
        def event(obj):
            return f"data: {json.dumps(obj)}\n\n"
        con = get_db_conn()
        cur = con.cursor()
        try:
            yield event({"type": "stage", "name": "checking"})
            try:
                yolo_models, load_logs = resolve_yolo_models(
                    cur, project_id, body.model_preset, body.model_id,
                    body.yolo_confidence_threshold, body.yolo_device,
                    body.text_music_confidence_threshold, body.text_music_device,
                    body.stave_confidence_threshold, body.stave_device,
                )
            except (RuntimeError, ValueError) as e:
                yield event({"type": "error", "message": str(e)}); return
            for msg in load_logs:
                yield event({"type": "log", "message": msg})

            # layer separation - run YOLO per folio, write annotations, and
            # derive music_boxes/mask before handing off to text-service:
            # the exact sequence /predict already does per single image.
            music_boxes_by_index = []
            mask_json_by_index = []
            for i, (image_id, (name, data, mime)) in enumerate(zip(body.image_ids, images)):
                pil_img = Image.open(io.BytesIO(data)).convert("RGB")
                img_arr = np.array(pil_img)
                yolo_txt = yolo_models.infer(img_arr)
                write_annotation(
                    cur, con, project_id, image_id, name, yolo_txt,
                    yolo_models.stored_model_id, yolo_models.model_label, yolo_models.model_hash,
                )
                music_boxes_by_index.append(_music_boxes_for_image(project_id, name, data))
                mask_json_by_index.append(
                    _mask_json_for_image(project_id, name, data) if body.masking_enabled else None
                )
                n_detections = len(yolo_txt.splitlines()) if yolo_txt else 0
                yield event({"type": "log", "message": f"{name}: layer separation done ({n_detections} detection(s))"})
            yield event({"type": "stage_done", "name": "checking"})

            yield event({"type": "stage", "name": "validating"})
            yield event({"type": "log", "message": f"running Kraken segmentation + HTR across {len(body.folios)} folio(s) (Cantus-aligned mode, source {body.source_id})..."})
            if body.segmentation_model:
                yield event({"type": "log", "message": f"using custom segmentation model: {body.segmentation_model}"})
            if body.column_count:
                yield event({"type": "log", "message": f"column count forced to {body.column_count}"})
            yield event({"type": "stage_done", "name": "validating"})

            yield event({"type": "stage", "name": "processing"})
            fields = {
                "folios": json.dumps(body.folios),
                "source_id": str(body.source_id),
                "device": body.device,
                "column_bimodal_threshold": str(body.column_bimodal_threshold),
                "masking_enabled": "true" if body.masking_enabled else "false",
                "mask_padding": str(body.mask_padding),
                "music_boxes": json.dumps(music_boxes_by_index),
                "mask_json_list": json.dumps(mask_json_by_index),
            }
            if body.segmentation_model:
                fields["segmentation_model"] = body.segmentation_model
            if body.recognition_model:
                fields["recognition_model"] = body.recognition_model
            if body.column_count is not None:
                fields["column_count"] = str(body.column_count)
            files = [("images", name, mime, data) for name, data, mime in images]

            for line in _stream_multipart(f"{TEXT_API_URL}/batch-run", fields=fields, files=files, timeout=1800):
                if not line.startswith("data: "):
                    continue
                ev = json.loads(line[len("data: "):])
                if ev.get("type") == "folio_result":
                    idx = ev["image_index"]
                    image_id = body.image_ids[idx]
                    image_name = images[idx][0]
                    alignment = ev["text_alignment"]
                    aid = _uuid.uuid4().hex
                    cur.execute(
                        "INSERT INTO text_alignments"
                        " (id, project_id, image_id, image_name, alignment_json,"
                        " median_line_spacing, syllable_count, log_text)"
                        " VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                        (aid, project_id, image_id, image_name, json.dumps(alignment),
                         alignment.get("median_line_spacing", 0.0),
                         len(alignment.get("syl_boxes", [])), ""),
                    )
                    con.commit()
                    yield event({"type": "log", "message": f"{image_name}: {len(alignment.get('syl_boxes', []))} syllable(s) aligned"})
                    continue
                yield event(ev)
        except urllib.error.URLError as exc:
            yield event({"type": "error", "message": f"text-service at {TEXT_API_URL} is unreachable: {exc}"})
        except Exception as e:
            yield event({"type": "error", "message": str(e)})
        finally:
            cur.close(); release_db_conn(con)

    
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

@router.get("/projects/{project_id}/sources/{source_id}/export")
def export_source_annotations(project_id: int, source_id: str, user=Depends(get_current_user)):
    """Zips the persisted YOLO annotations + text-alignment JSON for every
    image tagged with this Cantus source, downloadable any time - not just
    right after a batch run (unlike /text-batch/{batch_id}/download above,
    this reads straight from the DB, so it works regardless of whether the
    images went through grid or batch upload, and survives restarts).
    """
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        cur.execute(
            "SELECT id, name, folio FROM project_images WHERE project_id=%s AND source_id=%s",
            (project_id, source_id),
        )
        source_images = cur.fetchall()
        if not source_images:
            raise HTTPException(status_code=404, detail="no images found for this source")

        buf = io.BytesIO()
        written = 0
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for image_id, image_name, folio in source_images:
                cur.execute(
                    "SELECT yolo_txt FROM annotations WHERE project_id=%s AND image_name=%s"
                    " ORDER BY created_at DESC LIMIT 1",
                    (project_id, image_name),
                )
                ann_row = cur.fetchone()
                cur.execute(
                    "SELECT alignment_json FROM text_alignments WHERE project_id=%s AND image_name=%s"
                    " ORDER BY created_at DESC LIMIT 1",
                    (project_id, image_name),
                )
                align_row = cur.fetchone()
                if not ann_row and not align_row:
                    continue
                payload = {
                    "imageName": image_name,
                    "folio": folio,
                    "yoloAnnotations": ann_row[0] if ann_row else None,
                    "textAlignment": json.loads(align_row[0]) if align_row and align_row[0] else None,
                }
                stem = folio or image_name
                zf.writestr(f"{stem}.json", json.dumps(payload, indent=2))
                written += 1
        if written == 0:
            raise HTTPException(status_code=404, detail="no annotation or text data found for this source yet")
    
    return Response(
        content=buf.getvalue(), media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="source-{source_id}-export.zip"'},
    )