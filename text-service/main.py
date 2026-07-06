"""FastAPI wrapper exposing mothra-text's run_pipeline as an HTTP service.

Runs as its own process (own venv, port 8002 by default) — see dev.sh.
Mirrors the SSE event contract used by inference_api.py's /predict and
encode_api.py's /encode-upload: stage -> stage_done -> log -> result -> done.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

MOTHRA_TEXT_DIR = Path(__file__).resolve().parent.parent / "mothra-text"
sys.path.insert(0, str(MOTHRA_TEXT_DIR))
from run_pipeline import run, _build_pipeline_payload, _write_mei_json

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/run")
async def run_text_pipeline(
    image: UploadFile = File(...),
    folio: Optional[str] = Form(None),
):
    image_bytes = await image.read()
    image_filename = image.filename or "page.jpg"

    def generate():
        def event(obj):
            return f"data: {json.dumps(obj)}\n\n"

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            yield event({"type": "stage", "name": "checking"})
            image_path = tmp_dir / image_filename
            image_path.write_bytes(image_bytes)
            yield event({"type": "log", "message": f"loaded {image_filename}"})
            yield event({"type": "stage_done", "name": "checking"})

            yield event({"type": "stage", "name": "validating"})
            yield event({"type": "log", "message": "running Kraken segmentation + HTR (OCR-only mode)..."})
            yield event({"type": "stage_done", "name": "validating"})

            yield event({"type": "stage", "name": "processing"})
            collection, manifest = run(
                image_path=str(image_path),
                folio=folio,
                ocr_only_mode=True,
            )
            payload = _build_pipeline_payload(
                collection, str(image_path), manifest, folio=folio, mode="ocr_only",
            )
            mei_json_path = tmp_dir / "mei_alignment.json"
            _write_mei_json(payload, str(mei_json_path))
            text_alignment = json.loads(mei_json_path.read_text())
            n_syl = len(text_alignment.get("syl_boxes", []))
            yield event({"type": "log", "message": f"{n_syl} syllable(s) aligned"})
            yield event({"type": "stage_done", "name": "processing"})
            yield event({"type": "result", "text_alignment": text_alignment})
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