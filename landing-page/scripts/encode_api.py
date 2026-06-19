from fastapi import APIRouter, UploadFile, File as FAPIFile
from fastapi.responses import Response, JSONResponse


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

_sessions: dict[str, dict] = {}
MANIFEST_DIR = Path(tempfile.gettempdir()) / "mothra_manifests"
MANIFEST_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from encode_to_mei import (
    parse_gamera_xml, parse_staves, assign_glyphs_to_staves,
    estimate_staves_from_glyphs, build_mei, build_neon_manifest, validate_mei,
)

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
    image_file: Optional[UploadFile] = FAPIFile(None)):
    session_id = _uuid.uuid4().hex[:8]
    tmp_dir = Path(tempfile.mkdtemp())
    logs: list[str] = []

    xml_path = tmp_dir / "uploaded.xml"
    with open(xml_path, "wb") as f:
        shutil.copyfileobj(xml_file.file, f)

    logs.append(f"parsing GameraXML: {xml_file.filename}")
    glyphs = parse_gamera_xml(xml_path)
    logs.append(f" {len(glyphs)} glyphs loaded")

    page_w = page_h = 0
    image_bytes: bytes = b""
    image_data_uri: Optional[str] = None
    if image_file:
        header = await image_file.read(65536)
        rest = await image_file.read()
        image_bytes = header + rest
        dims = _image_dimensions(header)
        if dims:
            page_w, page_h = dims
            logs.append(f"page size: {page_w}×{page_h}px (from {image_file.filename})")
        else:
            logs.append(f"warning: could not read dimensions from {image_file.filename}")
        mime = mimetypes.guess_type(image_file.filename)[0] or "image/jpeg"
        image_data_uri = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"

    if not (page_w and page_h):
        page_w = max((g.lrx for g in glyphs), default=800) + 10
        page_h = max((g.lry for g in glyphs), default=1200) + 10
        logs.append(f"page size: {page_w}×{page_h}px (estimated — upload image for exact bounds)")

    staves = estimate_staves_from_glyphs(glyphs, page_w, page_h)
    logs.append(f" estimated {len(staves)} stave(s) from glyph positions (no stave detection provided)")

    glyphs_by_stave = assign_glyphs_to_staves(glyphs, staves)
    assigned = sum(len(v) for k, v in glyphs_by_stave.items() if k >= 0)
    logs.append(f" {assigned} glyphs assigned to stave")

    stem = Path(xml_file.filename).stem
    image_ref = Path(image_file.filename) if image_file else Path("")
    mei_bytes = build_mei(glyphs_by_stave, staves, image_ref, page_w, page_h, stem)
    validation_warnings = validate_mei(mei_bytes)
    for w in validation_warnings:
        logs.append(f"[warn] {w}")
    logs.append("MEI built successfully" if not validation_warnings else "MEI built with warnings (see above)")
    logs.append("encoding complete!")

    mei_b64 = base64.b64encode(mei_bytes).decode()

    manifest = build_neon_manifest(mei_bytes, image_data_uri or str(image_ref), stem) if image_data_uri else None

    if manifest:
        (MANIFEST_DIR / f"{session_id}.jsonld").write_text(json.dumps(manifest))

    _sessions[session_id] = {
        "mei_bytes": mei_bytes,
        "stem": stem,
    }
    return {"session_id": session_id, "mei_base64": mei_b64, "logs": logs, "manifest": manifest}

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
    if session_id not in _sessions:
        return JSONResponse(status_code=404, content={"error": "not found"})
    s = _sessions[session_id]
    return Response(
        content=s["mei_bytes"],
        media_type="application/xml",
        headers={"Content-Disposition": f'attachment; filename={s["stem"]}.mei"'},
    )

