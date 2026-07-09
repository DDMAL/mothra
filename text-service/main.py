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
from PIL import Image as PILImage
from steps.mothra_mask import MothraImageMask

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

def _apply_mothra_mask(image_path: Path, mask_json: Optional[str], padding_px: int) -> Optional[Path]:
    """Black out everything except mothra-detected text regions in the image
    at image_path, replicating run_pipeline.py's main() CLI masking block
    (which only exists in the CLI entrypoint, not in run()).

    mask_json is JSON *content*, not a file path — MothraImageMask's
    constructor only accepts a path, so this writes it to a scratch temp
    file first and deletes that scratch file once the masker has parsed it.

    Returns the path to a new temp masked-image file, or None if mask_json
    was falsy (caller keeps using the original image_path). Caller owns
    deleting the returned path.
    """
    if not mask_json:
        return None
    tmp_json = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as _tmpf:
            _tmpf.write(mask_json)
            tmp_json = _tmpf.name
        img = PILImage.open(image_path).convert("RGB")
        masker = MothraImageMask(tmp_json, padding_px=padding_px)
        masked_img = masker.apply(img)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as _tmp:
            masked_img.save(_tmp.name)
            return Path(_tmp.name)
    finally:
        if tmp_json:
            Path(tmp_json).unlink(missing_ok=True)


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
    music_boxes: Optional[str] = Form(None),
    column_count: Optional[int] = Form(None),
    segmentation_model: Optional[str] = Form(None),
    recognition_model: Optional[str] = Form(None),
    device: str = Form("cpu"),
    column_bimodal_threshold: float = Form(0.5),
    masking_enabled: bool = Form(True),
    mask_padding: int = Form(15),
    mask_json: Optional[str] = Form(None),
):
    image_bytes = await image.read()
    image_filename = image.filename or "page.jpg"
    parsed_music_boxes = json.loads(music_boxes) if music_boxes else []

    # Empty/omitted recognition_model means "use text-service's own default"
    # (the auto-detected Tridis model, same as mothra-text's own CLI default
    # via _DEFAULT_RECOGNITION_MODEL) — NOT run()'s bare None (stub mode).
    effective_recognition_model = recognition_model or RECOGNITION_MODEL

    def generate():
        def event(obj):
            return f"data: {json.dumps(obj)}\n\n"

        tmp_dir = Path(tempfile.mkdtemp())
        log_queue: "queue.Queue[dict]" = queue.Queue()
        handler = _QueueLogHandler(log_queue)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        _mask_tmp: Optional[Path] = None
        try:
            yield event({"type": "stage", "name": "checking"})
            image_path = tmp_dir / image_filename
            image_path.write_bytes(image_bytes)
            yield event({"type": "log", "message": f"loaded {image_filename}"})
            yield event({"type": "stage_done", "name": "checking"})

            yield event({"type": "stage", "name": "validating"})
            if effective_recognition_model:
                yield event({"type": "log", "message": f"running Kraken segmentation + HTR (OCR-only mode, model={Path(effective_recognition_model).name})..."})
            else:
                yield event({"type": "log", "message": "running Kraken segmentation + HTR (OCR-only mode, STUB — no recognition model installed, text will be empty)..."})
            if segmentation_model:
                yield event({"type": "log", "message": f"using custom segmentation model: {segmentation_model}"})
            if column_count:
                yield event({"type": "log", "message": f"column count forced to {column_count}"})

            active_image_path = image_path
            if masking_enabled and mask_json:
                try:
                    _mask_tmp = _apply_mothra_mask(image_path, mask_json, mask_padding)
                except Exception as exc:
                    yield event({"type": "log", "message": f"text-region masking failed, continuing unmasked: {exc}"})
                if _mask_tmp:
                    active_image_path = _mask_tmp
                    yield event({"type": "log", "message": f"applied text-region mask (padding={mask_padding}px)"})
            elif masking_enabled and not mask_json:
                yield event({"type": "log", "message": "text-region masking enabled but no mask JSON available; running without masking"})
            else:
                yield event({"type": "log", "message": "text-region masking disabled; running without masking"})
            yield event({"type": "stage_done", "name": "validating"})

            yield event({"type": "stage", "name": "processing"})
            result_holder: dict = {}

            def _worker():
                handler.thread_ident = threading.current_thread().ident
                try:
                    result_holder["value"] = run(
                        image_path=str(active_image_path),
                        folio=folio,
                        segmentation_model=segmentation_model,
                        recognition_model=effective_recognition_model,
                        device=device,
                        column_bimodal_threshold=column_bimodal_threshold,
                        column_count=column_count,
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
                collection, str(active_image_path), manifest, folio=folio, mode="ocr_only",
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
            if _mask_tmp:
                _mask_tmp.unlink(missing_ok=True)
            shutil.rmtree(tmp_dir, ignore_errors=True)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )