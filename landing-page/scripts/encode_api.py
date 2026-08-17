from fastapi import APIRouter, UploadFile, File as FAPIFile, Form
from fastapi.responses import Response, JSONResponse
from pathlib import Path
import sys
import uuid as _uuid
from typing import Optional
import xml.etree.ElementTree as ET

from config import MOCK_DATA_DIR
sys.path.insert(0, str(Path(__file__).parent))
from encode_to_mei import (
    parse_gamera_xml, parse_staves, assign_glyphs_to_staves, build_mei,
    build_neon_manifest, validate_mei,
)
from job_store import stage_upload, session_get, manifest_get, new_job_id, create_job
from tasks_encode import run_encode_upload_task, run_encode_batch_task

router = APIRouter()

@router.post("/encode")
def encode():
    glyphs = parse_gamera_xml(MOCK_DATA_DIR / "mock_page.xml")
    staves, image_w, image_h = parse_staves(MOCK_DATA_DIR / "mock_staves.json")
    glyphs_by_stave, staves = assign_glyphs_to_staves(glyphs, staves, image_w, image_h)
    image_path = MOCK_DATA_DIR / "mock_page.jpg"
    mei_bytes = build_mei(glyphs_by_stave, staves, image_path, image_w, image_h, "mock_page")
    return build_neon_manifest(mei_bytes, str(image_path), "mock_page")

@router.post("/encode-upload")
async def encode_upload(
    xml_file: UploadFile = FAPIFile(...),
    image_file: Optional[UploadFile] = FAPIFile(None),
    project_id: Optional[int] = Form(None),
    image_name: Optional[str] = Form(None),
    clef_shape: Optional[str] = Form(None),
    clef_line: Optional[int] = Form(None),
    notation_type: Optional[str] = Form(None),
    allow_synthetic_lines: bool = Form(False),
):
    xml_bytes = await xml_file.read()
    xml_filename = xml_file.filename or "uplaoded.xml"
    image_bytes = await image_file.read() if image_file else None
    image_filename = image_file.filename if image_file else None

    job_id = new_job_id()
    xml_upload_id = _uuid.uuid4().hex
    stage_upload(xml_upload_id, xml_bytes)
    image_upload_id = None
    if image_bytes:
        image_upload_id = _uuid.uuid4().hex
        stage_upload(image_upload_id, image_bytes)

    kwargs = {
        "xml_upload_id": xml_upload_id,
        "xml_filename": xml_filename,
        "image_upload_id": image_upload_id,
        "image_filename": image_filename,
        "project_id": project_id,
        "image_name": image_name,
        "clef_shape": clef_shape,
        "clef_line": clef_line,
        "notation_type": notation_type,
        "allow_synthetic_lines": allow_synthetic_lines,
    }
    create_job(job_id, "encode_upload", project_id, params=kwargs)  # dedupe_seconds=0 (default): always creates
    run_encode_upload_task.apply_async(kwargs={"job_id": job_id, **kwargs}, task_id=job_id)
    return JSONResponse({"job_id": job_id})

@router.post("/validate-mei")
async def validate_mei_endpoint(file: UploadFile = FAPIFile(...)):
    xml_bytes = await file.read()
    try:
        ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        return {"valid": False, "warnings": [f"XML parse error: {e}"]}
    warnings = validate_mei(xml_bytes)
    return {"valid": len(warnings) == 0, "warnings": warnings}

@router.get("/manifest/{session_id}")
def get_manifest(session_id: str):
    manifest = manifest_get(session_id)
    if manifest is not None:
        return JSONResponse(content=manifest)
    return JSONResponse(status_code=404, content={"error": "manifest not found"})

@router.get("/mei/{session_id}")
def get_mei(session_id: str):
    s = session_get(session_id)
    if s is None:
        return JSONResponse(status_code=404, content={"error": "not found"})
    return Response(
        content=s["mei_bytes"],
        media_type="application/xml",
        headers={"Content-Disposition": f'attachment; filename="{s["stem"]}.mei"'},
    )

@router.post("/encode-batch")
async def encode_batch(
    xml_files: list[UploadFile] = FAPIFile(...),
    image_files: list[UploadFile] = FAPIFile(...),
    image_names: Optional[list[str]] = Form(None),
    project_id: Optional[int] = Form(None),
    clef_shape: Optional[str] = Form(None),
    clef_line: Optional[int] = Form(None),
    notation_type: Optional[str] = Form(None),
    allow_synthetic_lines: bool = Form(False),
):
    if len(xml_files) != len(image_files):
        return JSONResponse(status_code=400, content={
            "error": f"xml_files ({len(xml_files)}) and image_files ({len(image_files)}) must be the same length",
        })
    if image_names is not None and len(image_names) != len(xml_files):
        return JSONResponse(status_code=400, content={
            "error": f"image_names ({len(image_names)}) must match xml_files ({len(xml_files)}) if provided",
        })
    
    items = []
    for i, (x, img) in enumerate(zip(xml_files, image_files)):
        xml_bytes = await x.read()
        img_bytes = await img.read()
        name = image_names[i] if image_names else (img.filename or "")
        xml_upload_id = _uuid.uuid4().hex
        image_upload_id = _uuid.uuid4().hex
        stage_upload(xml_upload_id, xml_bytes)
        stage_upload(image_upload_id, img_bytes)
        items.append({
            "xml_upload_id": xml_upload_id,
            "xml_filename": x.filename or f"item-{i}.xml",
            "image_upload_id": image_upload_id,
            "image_filename": img.filename,
            "image_name": name,
        })

    job_id = new_job_id()
    kwargs = {
        "items": items,
        "project_id": project_id,
        "clef_shape": clef_shape,
        "clef_line": clef_line,
        "notation_type": notation_type,
        "allow_synthetic_lines": allow_synthetic_lines,
    }
    create_job(job_id, "encode_batch", project_id, params=kwargs)  # dedupe_seconds=0 (default): always creates
    run_encode_batch_task.apply_async(kwargs={"job_id": job_id, **kwargs}, task_id=job_id)
    return JSONResponse({"job_id": job_id})