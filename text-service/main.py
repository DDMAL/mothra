"""FastAPI wrapper exposing mothra-text's run_pipeline as an HTTP service.

Runs as its own process (own venv, port 8002 by default) — see dev.sh.
Mirrors the SSE event contract used by inference_api.py's /predict and
encode_api.py's /encode-upload: stage -> stage_done -> log -> result -> done.
"""
import json
import logging
import queue
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

MOTHRA_TEXT_DIR = Path(__file__).resolve().parent.parent / "mothra-text"
sys.path.insert(0, str(MOTHRA_TEXT_DIR))
from run_pipeline import run, _build_pipeline_payload, _write_mei_json, _find_tridis_model

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# run_pipeline.main() resolves this as the --recognition-model CLI default
# (_DEFAULT_RECOGNITION_MODEL); run() itself defaults to None (stub mode, empty
# OCR text) since we call it directly instead of going through main()/argparse.
RECOGNITION_MODEL = _find_tridis_model()

def _bbox_overlap_ratio(line_bbox: list[float], music_bbox: list[float]) -> float:
    """Fraction of line_bbox's own area covered by music_bbox."""
    lx0, ly0, lx1, ly1 = line_bbox
    mx0, my0, mx1, my1 = music_bbox
    iw = max(0.0, min(lx1, mx1) - max(lx0, mx0))
    ih = max(0.0, min(ly1, my1) - max(ly0, my0))
    line_area = max(1.0, (lx1 - lx0) * (ly1 - ly0))
    return (iw * ih) / line_area

def filter_lines_over_music(lines: list[dict], music_boxes: list[list[float]], threshold: float = 0.3) -> tuple[list[dict], int]:
    """Drop detected text lines that mostly overlap a YOLO music region - BLLA over-segmentation artifacts from neume notation, not real chant text."""
    if not music_boxes:
        return lines, 0
    kept, dropped = [], 0
    for line in lines:
        if any(_bbox_overlap_ratio(line["bbox"], mb) > threshold for mb in music_boxes):
            dropped += 1
            continue
        kept.append(line)
    return kept, dropped


class _QueueLogHandler(logging.Handler):
    """Relays one request's log records onto a queue for SSE relay.

    Filtered by thread identity so concurrent /run requests (each with its
    own worker thread) don't cross-talk on the shared root logger.
    """

    def __init__(self, q: "queue.Queue[dict]"):
        super().__init__()
        self.q = q
        self.thread_ident: Optional[int] = None
        self.setFormatter(logging.Formatter("%(message)s"))
    
    def emit(self, record: logging.LogRecord) -> None:
        if self.thread_ident is None or record.thread != self.thread_ident:
            return
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        self.q.put({"type": "log", "message": f"[{record.levelname}] {message}"})


@app.post("/run")
async def run_text_pipeline(
    image: UploadFile = File(...),
    folio: Optional[str] = Form(None),
    music_boxes: Optional[str] = Form(None)
):
    image_bytes = await image.read()
    image_filename = image.filename or "page.jpg"
    parsed_music_boxes = json.loads(music_boxes) if music_boxes else []

    def generate():
        def event(obj):
            return f"data: {json.dumps(obj)}\n\n"

        tmp_dir = Path(tempfile.mkdtemp())
        log_queue: "queue.Queue[dict]" = queue.Queue()
        handler = _QueueLogHandler(log_queue)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        try:
            yield event({"type": "stage", "name": "checking"})
            image_path = tmp_dir / image_filename
            image_path.write_bytes(image_bytes)
            yield event({"type": "log", "message": f"loaded {image_filename}"})
            yield event({"type": "stage_done", "name": "checking"})

            yield event({"type": "stage", "name": "validating"})
            if RECOGNITION_MODEL:
                yield event({"type": "log", "message": f"running Kraken segmentation + HTR (OCR-only mode, model={Path(RECOGNITION_MODEL).name})..."})
            else:
                yield event({"type": "log", "message": "running Kraken segmentation + HTR (OCR-only mode, STUB — no recognition model installed, text will be empty)..."})
            yield event({"type": "stage_done", "name": "validating"})

            yield event({"type": "stage", "name": "processing"})
            result_holder: dict = {}

            def _worker():
                handler.thread_ident = threading.current_thread().ident
                try:
                    result_holder["value"] = run(
                        image_path=str(image_path),
                        folio=folio,
                        recognition_model=RECOGNITION_MODEL,
                        ocr_only_mode=True,
                    )
                except Exception as exc:
                    result_holder["error"] = exc
            
            worker = threading.Thread(target=_worker, daemon=True)
            worker.start()

            while worker.is_alive() or not log_queue.empty():
                try:
                    yield event(log_queue.get(timeout=0.2))
                except queue.Empty:
                    continue
            worker.join()
            if "error" in result_holder:
                raise result_holder["error"]
            collection, manifest = result_holder["value"]
            payload = _build_pipeline_payload(
                collection, str(image_path), manifest, folio=folio, mode="ocr_only",
            )
            if parsed_music_boxes:
                kept_lines, n_dropped = filter_lines_over_music(payload["lines"], parsed_music_boxes)
                payload["lines"] = kept_lines
                if n_dropped:
                    yield event({"type": "log", "message": f"dropped {n_dropped} line(s) overlapping YOLO music regions"})
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
            root_logger.removeHandler(handler)
            shutil.rmtree(tmp_dir, ignore_errors=True)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )