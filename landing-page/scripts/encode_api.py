from fastapi import APIRouter, Depends, UploadFile, File as FAPIFile, Form
from fastapi.responses import Response, JSONResponse
from pathlib import Path
import sys
import uuid as _uuid
from typing import Optional
import xml.etree.ElementTree as ET

from auth_api import get_current_user, db_cursor, require_project_owner
sys.path.insert(0, str(Path(__file__).parent))
from encode_to_mei import validate_mei
from job_store import stage_upload, session_get, manifest_get, session_project_id, new_job_id, create_job
from tasks_encode import run_encode_upload_task, run_encode_batch_task

router = APIRouter()

# The only two mapping CSVs encode_to_mei.py's resolve_neume_mapping() knows
# how to load (see mothra#210) -- anything else is a client mistake, not a
# valid notation the encoder can fall back on.
_VALID_NOTATION_TYPES = {"square", "hufnagel"}


def _check_notation_type(notation_type: Optional[str]) -> Optional[JSONResponse]:
    """Returns a 400 response if notation_type is set but not one of the
    bundled mapping CSVs -- called before stage_upload()/create_job() so an
    invalid value fails the request outright instead of staging uploads and
    creating a job that will only fail once a worker picks it up."""
    if notation_type is not None and notation_type not in _VALID_NOTATION_TYPES:
        return JSONResponse(status_code=400, content={
            "error": f"notation_type must be one of {sorted(_VALID_NOTATION_TYPES)}, got {notation_type!r}",
        })
    return None


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
    user=Depends(get_current_user),
):
    # project_id is optional (an ad-hoc encode with no project context is a
    # real, supported flow -- see AppRouter.tsx's encode-upload kickoff) --
    # only check ownership when a project is actually named.
    if project_id is not None:
        with db_cursor() as (con, cur):
            require_project_owner(cur, project_id, user["id"])
    if (err := _check_notation_type(notation_type)) is not None:
        return err

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
async def validate_mei_endpoint(file: UploadFile = FAPIFile(...), user=Depends(get_current_user)):
    xml_bytes = await file.read()
    try:
        ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        return {"valid": False, "warnings": [f"XML parse error: {e}"]}
    warnings = validate_mei(xml_bytes)
    return {"valid": len(warnings) == 0, "warnings": warnings}

def _check_session_owner(session_id: str, user_id: int) -> Optional[JSONResponse]:
    """Returns a 404/403 response if this session_id doesn't exist or belongs
    to a project this user doesn't own; None if the caller may proceed.
    A session with no recorded project_id (predates mothra#220's job_sessions
    migration) is allowed through rather than denied -- see auth_api.py's
    _ADDED_COLUMNS entry for job_sessions.project_id."""
    info = session_project_id(session_id)
    if not info["exists"]:
        return JSONResponse(status_code=404, content={"error": "not found"})
    if info["project_id"] is not None:
        with db_cursor() as (con, cur):
            require_project_owner(cur, info["project_id"], user_id)
    return None

@router.get("/manifest/{session_id}")
def get_manifest(session_id: str, user=Depends(get_current_user)):
    if (err := _check_session_owner(session_id, user["id"])) is not None:
        return err
    manifest = manifest_get(session_id)
    if manifest is not None:
        return JSONResponse(content=manifest)
    return JSONResponse(status_code=404, content={"error": "manifest not found"})

@router.get("/mei/{session_id}")
def get_mei(session_id: str, user=Depends(get_current_user)):
    if (err := _check_session_owner(session_id, user["id"])) is not None:
        return err
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
    # mothra#241: project_images.id per item, when the caller has one (the
    # batch IC->encode path does, via icQueue.ts's buildEncodePair -- the
    # ad-hoc single-image encode-upload path above has no id to give and
    # doesn't send this). Threaded through to tasks_encode.py so hint
    # resolution and the resulting mei_files row can match by id instead of
    # the not-necessarily-unique image name.
    image_ids: Optional[list[str]] = Form(None),
    project_id: Optional[int] = Form(None),
    clef_shape: Optional[str] = Form(None),
    clef_line: Optional[int] = Form(None),
    notation_type: Optional[str] = Form(None),
    allow_synthetic_lines: bool = Form(False),
    user=Depends(get_current_user),
):
    # see encode_upload above: project_id is optional, ownership is only
    # checked when one is actually given.
    if project_id is not None:
        with db_cursor() as (con, cur):
            require_project_owner(cur, project_id, user["id"])
    if len(xml_files) != len(image_files):
        return JSONResponse(status_code=400, content={
            "error": f"xml_files ({len(xml_files)}) and image_files ({len(image_files)}) must be the same length",
        })
    if image_names is not None and len(image_names) != len(xml_files):
        return JSONResponse(status_code=400, content={
            "error": f"image_names ({len(image_names)}) must match xml_files ({len(xml_files)}) if provided",
        })
    if image_ids is not None and len(image_ids) != len(xml_files):
        return JSONResponse(status_code=400, content={
            "error": f"image_ids ({len(image_ids)}) must match xml_files ({len(xml_files)}) if provided",
        })
    if (err := _check_notation_type(notation_type)) is not None:
        return err

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
            "image_id": image_ids[i] if image_ids else None,
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