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
import re
import sys
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

from job_store import create_job
from tasks_text_batch import run_text_batch_task
from auth_api import get_db_conn, get_current_user, release_db_conn, db_cursor, require_project_owner
from text_api import TEXT_API_URL, _stream_multipart, _music_boxes_for_image, _mask_json_for_image
from yolo_inference import resolve_yolo_models, write_annotation
from encode_to_mei import scale_facsimile, get_encoded_dimensions

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
    yolo_device: str = "auto"   # auto → GPU if present else CPU (resolved in yolo_inference)
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
        already_aligned = set()
        for iid in body.image_ids:
            cur.execute("SELECT 1 FROM text_alignments WHERE project_id=%s AND image_id=%s", (project_id, iid))
            if cur.fetchone():
                already_aligned.add(iid)
        pairs = [(iid, folio) for iid, folio in zip(body.image_ids, body.folios) if iid not in already_aligned]
        image_ids = [p[0] for p in pairs]
        folios = [p[1] for p in pairs]
    skipped_count = len(body.image_ids) - len(image_ids)

    new_id = _uuid.uuid4().hex[:8]
    task_body = {**body.model_dump(), "image_ids": image_ids, "folios": folios, "skipped_count": skipped_count}
    job_id, is_new = create_job(new_id, "text_batch", project_id,
                                 params={"project_id": project_id, "body": task_body},
                                 dedupe_seconds=5)
    if is_new:
        run_text_batch_task.apply_async(kwargs={"job_id": job_id, "project_id": project_id, "body": task_body}, task_id=job_id)
    return {"job_id": job_id}


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

def _get_cantus_siglum(source_id: str) -> Optional[str]:
    try:
        with urllib.request.urlopen(f"{TEXT_API_URL}/cantus-source/{source_id}", timeout=15) as resp:
            return json.loads(resp.read().decode()).get("siglum")
    except Exception as e:
        # Falls back to the "source-{id}" filename prefix either way (see
        # caller) — logged so a text-service outage/timeout here doesn't look
        # identical to "this Cantus source just has no siglum".
        print(f"[warn] could not fetch siglum for cantus source {source_id}: {e}", file=sys.stderr)
        return None
    
def _sanitize_stem(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s.strip()) if s else ""

def _cantus_bundle_readme(source_id: str, siglum: str, filenames: list[str]) -> str:
    files_list = "\n".join(f"  - {f}" for f in filenames)
    return (
        "This bundle contains corrected MEI files ready for Cantus Ultimus indexing.\n\n"
        f"To publish:\n"
        f"  1. Place these files under production_mei_files/{source_id}/ (DDMAL/production_mei_files repo)\n"
        f"  2. Commit and push to that repo's main branch\n"
        f"  3. Pull the updated submodule into the running Cantus Ultimus deployment\n"
        f"  4. Inside the Cantus Ultimus container, run:\n"
        f"     python manage.py index_manuscript_mei {source_id} --mei-dir production_mei_files/{source_id}\n"
        f"     (confirm {source_id} is the correct manuscript_id — it may differ from the CantusDB source ID)\n\n"
        f"Files in this bundle ({len(filenames)}):\n{files_list}\n"
    )

@router.get("/projects/{project_id}/sources/{source_id}/cantus-bundle")
def cantus_bundle(project_id: int, source_id: str, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        cur.execute("""
            SELECT m.name, m.xml_content, pi.folio, pi.original_width
            FROM mei_files m
            JOIN project_images pi ON pi.project_id = m.project_id AND pi.name = m.image_name
            WHERE m.project_id=%s AND pi.source_id=%s AND m.corrected=1
              AND m.xml_content IS NOT NULL
              AND m.created_at = (
                  SELECT MAX(m2.created_at) FROM mei_files m2
                  WHERE m2.project_id = m.project_id AND m2.image_name = m.image_name
              )
        """, (project_id, source_id))
        rows = cur.fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail=(
                "no corrected MEI files found for this Cantus source yet — "
                "correct at least one MEI file in Neon before sending to Cantus Ultimus"
            ))

        siglum = _get_cantus_siglum(source_id) or f"source-{source_id}"
        buf = io.BytesIO()
        used_stems, filenames = set(), []
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, xml_content, folio, original_width in rows:
                stem = _sanitize_stem(folio) or _sanitize_stem(name) or "unknown"
                unique_stem, n = stem, 2
                while unique_stem in used_stems:
                    unique_stem = f"{stem}-{n}"; n += 1
                used_stems.add(unique_stem)
                mei_filename = f"{siglum}_{unique_stem}.mei"

                xml_bytes = xml_content.encode("utf-8")
                if original_width:
                    dims = get_encoded_dimensions(xml_bytes)
                    if dims and dims[0]:
                        xml_bytes = scale_facsimile(xml_bytes, original_width / dims[0])
                        
                zf.writestr(mei_filename, xml_bytes)
                filenames.append(mei_filename)
            zf.writestr("README.txt", _cantus_bundle_readme(source_id, siglum, filenames))

    return Response(
        content=buf.getvalue(), media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="cantus-bundle-{source_id}.zip"'},
    )