from fastapi import APIRouter, UploadFile, File as FAPIFile, Form
from fastapi.responses import Response, JSONResponse, StreamingResponse
from datetime import datetime, timedelta

from pathlib import Path
import sys
import base64
import json
import mimetypes
import struct
import tempfile, shutil, uuid as _uuid
from typing import Optional
import xml.etree.ElementTree as ET


def _image_dimensions(header: bytes) -> Optional[tuple]:
    """Return (width, height) from the first bytes of a JPEG, PNG, or TIFF file."""
    # PNG: dimensions at bytes 16-24 of the IHDR chunk
    if header[:8] == b'\x89PNG\r\n\x1a\n':
        w, h = struct.unpack('>II', header[16:24])
        return w, h
    # JPEG: scan forward for SOF0/SOF1/SOF2 marker
    if header[:2] == b'\xff\xd8':
        i = 2
        while i < len(header) - 8:
            if header[i] != 0xff:
                break
            marker = header[i + 1]
            if marker in (0xC0, 0xC1, 0xC2):
                h, w = struct.unpack('>HH', header[i + 5:i + 9])
                return w, h
            seg_len = struct.unpack('>H', header[i + 2:i + 4])[0]
            i += 2 + seg_len
    # TIFF: little-endian (II) or big-endian (MM)
    if header[:2] in (b'II', b'MM'):
        bo = '<' if header[:2] == b'II' else '>'
        ifd_off = struct.unpack(bo + 'I', header[4:8])[0]
        if ifd_off + 2 > len(header):
            return None
        n = struct.unpack(bo + 'H', header[ifd_off:ifd_off + 2])[0]
        w = h = 0
        for j in range(n):
            off = ifd_off + 2 + j * 12
            if off + 12 > len(header):
                break
            tag, typ = struct.unpack(bo + 'HH', header[off:off + 4])
            if tag in (256, 257):
                fmt = bo + ('I' if typ == 4 else 'H')
                val = struct.unpack(fmt, header[off + 8:off + 8 + struct.calcsize(fmt)])[0]
                if tag == 256: w = val
                else: h = val
        if w and h:
            return w, h
    return None

_SESSION_TTL = timedelta(hours=1)
_sessions: dict[str, tuple[datetime, dict]] = {}

def _session_put(sid: str, data: dict) -> None:
    _sessions[sid] = (datetime.utcnow() + _SESSION_TTL, data)
    now = datetime.utcnow()
    for k in [k for k, (exp, _) in list(_sessions.items()) if exp < now]:
        _sessions.pop(k, None)
        (MANIFEST_DIR / f"{k}.jsonld").unlink(missing_ok=True)

def _session_get(sid: str) -> "dict | None":
    entry = _sessions.get(sid)
    if not entry:
        return None
    exp, data = entry
    if exp < datetime.utcnow():
        _sessions.pop(sid, None)
        (MANIFEST_DIR / f"{sid}.jsonld").unlink(missing_ok=True)
        return None
    return data

MANIFEST_DIR = Path(tempfile.gettempdir()) / "mothra_manifests"
MANIFEST_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from encode_to_mei import (
    parse_gamera_xml, parse_staves, assign_glyphs_to_staves,
    estimate_staves_from_glyphs, parse_yolo_stave_hints, build_mei, build_neon_manifest, validate_mei,
)
from auth_api import get_db_conn, release_db_conn

router = APIRouter()

MOCK_DIR = Path(__file__).parent / "mock_data"

@router.post("/encode")
def encode():
    glyphs = parse_gamera_xml(MOCK_DIR / "mock_page.xml")
    staves, image_w, image_h = parse_staves(MOCK_DIR / "mock_staves.json")
    glyphs_by_stave = assign_glyphs_to_staves(glyphs, staves)
    image_path = MOCK_DIR / "mock_page.jpg"
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
):
    # read file bytes before entering the sync generator
    xml_bytes = await xml_file.read()
    xml_filename = xml_file.filename or "uploaded.xml"
    image_bytes = await image_file.read() if image_file else None
    image_filename = image_file.filename if image_file else None

    def generate():
        def event(obj): return f"data: {json.dumps(obj)}\n\n"
        tmp_dir = Path(tempfile.mkdtemp())
        session_id = _uuid.uuid4().hex[:8]
        try:
            # stage: checking
            yield event({"type": "stage", "name": "checking"})
            xml_path = tmp_dir / "uploaded.xml"
            xml_path.write_bytes(xml_bytes)
            yield event({"type": "log", "message": f"parsing GameraXML: {xml_filename}"})
            glyphs = parse_gamera_xml(xml_path)
            yield event({"type": "log", "message": f" {len(glyphs)} glyphs loaded"})

            page_w = page_h = 0
            image_data_uri = None
            if image_bytes:
                dims = _image_dimensions(image_bytes[:65536])
                if dims:
                    page_w, page_h = dims
                    yield event({"type": "log", "message": f"page size: {page_w}×{page_h}px (from {image_filename})"})
                else:
                    yield event({"type": "log", "message": f"warning: could not read dimensions from {image_filename}"})
                mime = mimetypes.guess_type(image_filename or "")[0] or "image/jpeg"
                image_data_uri = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
            if not (page_w and page_h):
                page_w = max((g.lrx for g in glyphs), default=800) + 10
                page_h = max((g.lry for g in glyphs), default=1200) + 10
                yield event({"type": "log", "message": f"page size: {page_w}×{page_h}px (estimated)"})
            yield event({"type": "stage_done", "name": "checking"})

            # stage: validating
            yield event({"type": "stage", "name": "validating"})
            yolo_stave_hints = []
            if project_id and image_name:
                try:
                    con = get_db_conn()
                    cur = con.cursor()
                    cur.execute(
                        "SELECT yolo_txt FROM annotations WHERE image_name = %s AND project_id = %s "
                        "ORDER BY created_at DESC LIMIT 1",
                        (image_name, project_id),
                    )
                    row = cur.fetchone()
                    cur.close()
                    release_db_conn(con)
                    if row and row[0]:
                        yolo_stave_hints = parse_yolo_stave_hints(row[0], page_w, page_h)
                except Exception:
                    pass  # fall back to heuristic silently
            if yolo_stave_hints:
                staves = yolo_stave_hints
                yield event({"type": "log", "message": f" {len(staves)} stave(s) from YOLO annotations"})
            else:
                staves = estimate_staves_from_glyphs(glyphs, page_w, page_h)
                yield event({"type": "log", "message": f" estimated {len(staves)} stave(s) from glyph positions"})
            glyphs_by_stave = assign_glyphs_to_staves(glyphs, staves)
            assigned = sum(len(v) for k, v in glyphs_by_stave.items() if k >= 0)
            yield event({"type": "log", "message": f" {assigned} glyphs assigned to stave"})
            yield event({"type": "stage_done", "name": "validating"})

            #stage: processing
            yield event({"type": "stage", "name": "processing"})
            stem = Path(xml_filename).stem
            image_ref = Path(image_filename) if image_filename else Path("")
            mei_bytes_out = build_mei(
                glyphs_by_stave, staves, image_ref, page_w, page_h, stem,
                clef_shape=clef_shape or "C",
                clef_line=clef_line or 3,
            )
            validation_warnings = validate_mei(mei_bytes_out)
            for w in validation_warnings:
                yield event({"type": "log", "message": f"[warn] {w}"})
            yield event({"type": "log", "message": "MEI built successfully" if not validation_warnings else "MEI built with warnings"})
            mei_b64 = base64.b64encode(mei_bytes_out).decode()
            manifest = build_neon_manifest(mei_bytes_out, image_data_uri or str(image_ref), stem) if image_data_uri else None
            if manifest:
                (MANIFEST_DIR / f"{session_id}.jsonld").write_text(json.dumps(manifest))
            _session_put(session_id, {"mei_bytes": mei_bytes_out, "stem": stem})
            yield event({"type": "result", "session_id": session_id, "mei_base64": mei_b64, "manifest": manifest})
            yield event({"type": "stage_done", "name": "processing"})
            yield event({"type": "done"})
        except Exception as e:
            yield event({"type": "error", "message": str(e)})
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
    manifest_file = MANIFEST_DIR / f"{session_id}.jsonld"
    if manifest_file.exists():
        return JSONResponse(content=json.loads(manifest_file.read_text()))
    return JSONResponse(status_code=404, content={"error": "manifest not found"})


@router.get("/mei/{session_id}")
def get_mei(session_id: str):
    if _session_get(session_id) is None:
        return JSONResponse(status_code=404, content={"error": "not found"})
    s = _session_get(session_id)
    return Response(
        content=s["mei_bytes"],
        media_type="application/xml",
        headers={"Content-Disposition": f'attachment; filename="{s["stem"]}.mei"'},
    )

@router.post("/encode-batch")
async def encode_batch(
    xml_files: list[UploadFile] = FAPIFile(...),
    image_files: list[UploadFile] = FAPIFile(...),
    project_id: Optional[int] = Form(None),
    clef_shape: Optional[str] = Form(None),
    clef_line: Optional[int] = Form(None),
):
    
    # Read all bytes eagerly before entering the sync generator
    pairs = [
        (await x.read(), x.filename, await img.read(), img.filename)
        for x, img in zip(xml_files, image_files)
    ]

    def generate():
        def event(obj): return f"data: {json.dumps(obj)}\n\n"
        all_results = []
        for i, (xml_bytes, xml_fn, img_bytes, img_fn) in enumerate(pairs):
            yield event({"type": "item_start", "index": i, "name": img_fn, "total": len(pairs)})
            # run the same parse/encode logic as encode_upload's generate()
            # yield stage/log events with an extra "item": i field
            