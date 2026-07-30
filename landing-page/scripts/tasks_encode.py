import base64
import json
import mimetypes
import shutil
import tempfile
import uuid as _uuid
from pathlib import Path
from typing import Optional

from celery_app import celery_app
from job_store import publish_event, fetch_upload, drop_upload, session_put
from auth_api import get_db_conn, release_db_conn
from encode_to_mei import (
    parse_gamera_xml, assign_glyphs_to_staves, estimate_staves_from_glyphs,
    parse_yolo_stave_hints, build_mei, build_neon_manifest, validate_mei,
    image_dimensions,
)
import staffline_adapter

def _resolve_hints(project_id: Optional[int], image_name: Optional[str], page_w, page_h):
    """Looks up saved text-alignment + stave annotations for one image.
    Falls back to None/[]/None on any lookup failure or missing
    project_id/image_name, mirroring the try/except-pass behavior of the
    original inline generator code.

    Stave hints are resolved through a 3-tier fallback (this function only
    implements the first two; estimate_staves_from_glyphs is tier 3, applied
    by the caller when yolo_stave_hints is empty): the rich, per-line
    staffline_detections data (tier 1, when a project has it) wins over the
    coarser yolo_txt-geometry heuristic in parse_yolo_stave_hints (tier 2,
    today's only source, still the fallback for projects/images that predate
    this feature or whose staffline detection failed/was skipped)."""
    text_alignment = None
    yolo_stave_hints = []
    stave_source = None
    if project_id and image_name:
        con = get_db_conn()
        try:
            cur = con.cursor()
            try:
                cur.execute(
                    "SELECT alignment_json FROM text_alignments WHERE image_name=%s AND project_id=%s"
                    " ORDER BY created_at DESC LIMIT 1",
                    (image_name, project_id),
                )
                row = cur.fetchone()
                if row and row[0]:
                    text_alignment = json.loads(row[0])
            except Exception:
                pass
            try:
                cur.execute(
                    "SELECT jsomr_json FROM staffline_detections WHERE image_name=%s AND project_id=%s"
                    " AND status='succeeded' ORDER BY created_at DESC LIMIT 1",
                    (image_name, project_id),
                )
                row = cur.fetchone()
                if row and row[0]:
                    jsomr_records = row[0] if isinstance(row[0], list) else json.loads(row[0])
                    yolo_stave_hints = staffline_adapter.staves_from_jsomr(jsomr_records)
                    if yolo_stave_hints:
                        stave_source = "staffline_detection"
            except Exception:
                pass
            if not yolo_stave_hints:
                try:
                    cur.execute(
                        "SELECT yolo_txt FROM annotations WHERE image_name = %s AND project_id = %s "
                        "ORDER BY created_at DESC LIMIT 1",
                        (image_name, project_id),
                    )
                    row = cur.fetchone()
                    if row and row[0]:
                        yolo_stave_hints = parse_yolo_stave_hints(row[0], page_w, page_h)
                        if yolo_stave_hints:
                            stave_source = "yolo_annotation"
                except Exception:
                    pass
            cur.close()
        finally:
            release_db_conn(con)
    return text_alignment, yolo_stave_hints, stave_source

def _encode_one(publish, xml_bytes, xml_filename, image_bytes, image_filename,
                project_id, image_name, clef_shape, clef_line, item=None,
                include_name_fields=False):
    """Runs the checking/validating/processing pipeline for one XML+image
    pair, publishing the same event sequence the old synchronous generator
    yielded. Returns (session_id, result_payload)."""
    def ev(obj):
        if item is not None:
            obj = {**obj, "item": item}
        publish(obj)

    tmp_dir = Path(tempfile.mkdtemp())
    session_id = _uuid.uuid4().hex[:8]
    try:
        ev({"type": "stage", "name": "checking"})
        xml_path = tmp_dir / "uploaded.xml"
        xml_path.write_bytes(xml_bytes)
        ev({"type": "log", "message": f"parsing GameraXML: {xml_filename}"})
        glyphs = parse_gamera_xml(xml_path)
        ev({"type": "log", "message": f" {len(glyphs)} glyphs loaded"})

        page_w = page_h = 0
        image_data_uri = None
        if image_bytes:
            dims = image_dimensions(image_bytes[:65536])
            if dims:
                page_w, page_h = dims
                ev({"type": "log", "message": f"page size: {page_w}x{page_h}px (from {image_filename})"})
            else:
                ev({"type": "log", "message": f"warning: could not read dimensions from {image_filename}"})
            mime = mimetypes.guess_type(image_filename or "")[0] or "image/jpeg"
            image_data_uri = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
        if not (page_w and page_h):
            page_w = max((g.lrx for g in glyphs), default=800) + 10
            page_h = max((g.lry for g in glyphs), default=1200) + 10
            ev({"type": "log", "message": f"page size: {page_w}x{page_h}px (estimated)"})
        ev({"type": "stage_done", "name": "checking"})

        ev({"type": "stage", "name": "validating"})
        text_alignment, yolo_stave_hints, stave_source = _resolve_hints(project_id, image_name, page_w, page_h)
        if text_alignment:
            ev({"type": "log", "message": f" {len(text_alignment.get('syl_boxes', []))} syllable(s) from text-finding"})
        if yolo_stave_hints:
            staves = yolo_stave_hints
            label = "staffline detection" if stave_source == "staffline_detection" else "YOLO annotations"
            ev({"type": "log", "message": f" {len(staves)} stave(s) from {label}"})
        else:
            staves = estimate_staves_from_glyphs(glyphs, page_w, page_h)
            ev({"type": "log", "message": f" estimated {len(staves)} stave(s) from glyph positions"})
        glyphs_by_stave = assign_glyphs_to_staves(glyphs, staves)
        assigned = sum(len(v) for k, v in glyphs_by_stave.items() if k >= 0)
        ev({"type": "log", "message": f" {assigned} glyphs assigned to stave"})
        ev({"type": "stage_done", "name": "validating"})

        ev({"type": "stage", "name": "processing"})
        stem = Path(xml_filename).stem
        image_ref = Path(image_filename) if image_filename else Path("")
        mei_bytes_out = build_mei(
            glyphs_by_stave, staves, image_ref, page_w, page_h, stem,
            clef_shape=clef_shape or "C",
            clef_line=clef_line or 3,
            text_alignment=text_alignment,
        )
        validation_warnings = validate_mei(mei_bytes_out)
        for w in validation_warnings:
            ev({"type": "log", "message": f"[warn] {w}"})
        ev({"type": "log", "message": "MEI built successfully" if not validation_warnings else "MEI built with warnings"})
        mei_b64 = base64.b64encode(mei_bytes_out).decode()
        manifest = build_neon_manifest(mei_bytes_out, image_data_uri or str(image_ref), stem) if image_data_uri else None
        session_put(session_id, mei_bytes_out, stem, manifest)

        result = {"session_id": session_id, "mei_base64": mei_b64, "manifest": manifest}
        if include_name_fields:
            result["image_name"] = image_filename
            result["stem"] = stem
        ev({"type": "result", **result})
        ev({"type": "stage_done", "name": "processing"})
        return session_id, result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

@celery_app.task(name="encode.upload")
def run_encode_upload_task(job_id, xml_upload_id, xml_filename, image_upload_id,
                           image_filename, project_id, image_name, clef_shape, clef_line):
    def publish(obj):
        publish_event(job_id, obj)

    xml_bytes = fetch_upload(xml_upload_id)
    image_bytes = fetch_upload(image_upload_id) if image_upload_id else None
    try:
        _encode_one(publish, xml_bytes, xml_filename, image_bytes, image_filename, 
                    project_id, image_name, clef_shape, clef_line, include_name_fields=False)
        publish({"type": "done"})
    except Exception as e:
        publish({"type": "error", "message": str(e)})
    finally:
        drop_upload(xml_upload_id)
        if image_upload_id:
            drop_upload(image_upload_id)

@celery_app.task(name="encode.batch")
def run_encode_batch_task(job_id, items, project_id, clef_shape, clef_line):
    def publish(obj):
        publish_event(job_id, obj)
    
    succeeded, failed = [], []
    for i, item in enumerate(items):
        publish({"type": "item_start", "item": i, "total": len(items),
                 "name": item["image_filename"] or item["xml_filename"]})
        xml_bytes = fetch_upload(item["xml_upload_id"])
        image_bytes = fetch_upload(item["image_upload_id"])
        try:
            session_id, _ = _encode_one(
                publish, xml_bytes, item["xml_filename"], image_bytes, item["image_filename"],
                project_id, item["image_name"], clef_shape, clef_line,
                item=i, include_name_fields=True,
            )
            succeeded.append({"item": i, "session_id": session_id,
                               "name": item["image_filename"] or item["xml_filename"]})
            publish({"type": "item_done", "item": i, "session_id": session_id})
        except Exception as e:
            failed.append({"item": i, "name": item["image_filename"] or item["xml_filename"], "message": str(e)})
            publish({"type": "item_error", "item": i,
                     "name": item["image_filename"] or item["xml_filename"], "message": str(e)})
        finally:
            drop_upload(item["xml_upload_id"])
            drop_upload(item["image_upload_id"])
    publish({"type": "done", "total": len(items), "succeeded": succeeded, "failed": failed})