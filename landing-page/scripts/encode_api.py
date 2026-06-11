from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys
from fastapi.responses import Response, JSONResponse

import base64
import tempfile, shutil, uuid as _uuid
from fastapi import UploadFile, File as FAPIFile, Form
from typing import Optional

_sessions: dict[str, dict] = {}

sys.path.insert(0, str(Path(__file__).parent))
from encode_to_mei import (
    parse_gamera_xml, parse_staves, assign_glyphs_to_staves, build_mei, build_neon_manifest,
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

MOCK_DIR = Path(__file__).parent / "mock_data"

@app.post("/encode")
def encode():
    glyphs = parse_gamera_xml(MOCK_DIR / "mock_page.xml")
    staves, image_w, image_h = parse_staves(MOCK_DIR / "mock_staves.json")
    glyphs_by_stave = assign_glyphs_to_staves(glyphs, staves)
    image_path = MOCK_DIR / "mock_page.jpg"
    mei_bytes = build_mei(glyphs_by_stave, staves, image_path, image_w, image_h, "mock_page")
    return build_neon_manifest(mei_bytes, image_path, "mock_page")

@app.post("/encode-upload")
async def encode_upload(
    xml_file: UploadFile = FAPIFile(...),
    image_width: Optional[int] = Form(None),
    image_height: Optional[int] = Form(None)):
    session_id = _uuid.uuid4().hex[:8]
    tmp_dir = Path(tempfile.mkdtemp())
    logs: list[str] = []

    xml_path = tmp_dir / "uploaded.xml"
    with open(xml_path, "wb") as f:
        shutil.copyfileobj(xml_file.file, f)

    logs.append(f"parsing GameraXML: {xml_file.filename}")
    glyphs = parse_gamera_xml(xml_path)
    logs.append(f" {len(glyphs)} glyphs loaded")

    from encode_to_mei import StaveBbox
    if not (image_width and image_height):
        return JSONResponse(
            status_code=422,
            content={"error": "image width and image height are required for correct MEI surface bounds"},
        )
    page_w, page_h = image_width, image_height
    logs.append(f"page size: {page_w}×{page_h}px")

    staves = [StaveBbox(id="synth-0", ulx=0, uly=0, lrx=page_w, lry=page_h)]
    logs.append(f" using synthetic full-page stave (no stave detection provided)")

    glyphs_by_stave = assign_glyphs_to_staves(glyphs, staves)
    assigned = sum(len(v) for k, v in glyphs_by_stave.items() if k >= 0)
    logs.append(f" {assigned} glyphs assigned to stave")

    stem = Path(xml_file.filename).stem
    image_ref = Path(stem).with_suffix(".jpg")
    mei_bytes = build_mei(glyphs_by_stave, staves, image_ref, page_w, page_h, stem)
    logs.append("MEI built successfully")
    logs.append("encoding complete!")

    mei_b64 = base64.b64encode(mei_bytes).decode()


    _sessions[session_id] = {
        "mei_bytes": mei_bytes,
        "stem": stem
    }
    return {"session_id": session_id, "mei_base64": mei_b64, "logs": logs}


@app.get("/mei/{session_id}")
def get_mei(session_id: str):
    if session_id not in _sessions:
        return JSONResponse(status_code=404, content={"error": "not found"})
    s = _sessions[session_id]
    return Response(
        content=s["mei_bytes"],
        media_type="application/xml",
        headers={"Content-Disposition": f'attachment; filename={s["stem"]}.mei"'},
    )

