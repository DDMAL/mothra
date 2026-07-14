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
import uuid as _uuid
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse

MOTHRA_TEXT_DIR = Path(__file__).resolve().parent.parent / "mothra-text"
sys.path.insert(0, str(MOTHRA_TEXT_DIR))
BATCH_DIR = Path(tempfile.gettempdir()) / "mothra-text/batches"
BATCH_DIR.mkdir(parents=True, exist_ok=True)

# startup cleanup of old batch zips to prevent excess accumulation
import time as _time
_now = _time.time()
for _f in BATCH_DIR.glob("*.zip"):
    if _now - _f.stat().st_mtime > 86400:
        _f.unlink(missing_ok=True)

from run_pipeline import run, _build_pipeline_payload, _write_mei_json, _find_tridis_model
from steps.gt_manifest import fetch_cantus_csv, make_output_stem
from steps.nw_chant_allocator import _folio_sort_key, read_folio_state
from PIL import Image as PILImage
from steps.mothra_mask import MothraImageMask

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# run_pipeline.main() resolves this as the --recognition-model CLI default
# (_DEFAULT_RECOGNITION_MODEL); run() itself defaults to None (stub mode, empty
# OCR text) since we call it directly instead of going through main()/argparse.
RECOGNITION_MODEL = _find_tridis_model()

import urllib.error
from datetime import datetime, timedelta

_CANTUS_CACHE_TTL = timedelta(hours=1)
_cantus_cache: dict[int, tuple[datetime, dict]] = {}

def _cantus_cache_get(source_id: int) -> Optional[dict]:
    entry = _cantus_cache.get(source_id)
    if not entry:
        return None
    exp, data = entry
    if exp < datetime.utcnow():
        _cantus_cache.pop(source_id, None)
        return None
    return data

def _cantus_cache_put(source_id: int, data: dict) -> None:
    _cantus_cache[source_id] = (datetime.utcnow() + _CANTUS_CACHE_TTL, data)
    now = datetime.utcnow()
    for k in [k for k, (exp, _) in list(_cantus_cache.items()) if exp < now]:
        _cantus_cache.pop(k, None)

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


@app.get("/cantus-source/{source_id}")
def get_cantus_source(source_id: int):
    cached = _cantus_cache_get(source_id)
    if cached:
        return cached
    try:
        rows = fetch_cantus_csv(source_id)
    except urllib.error.HTTPError as exc:
        raise HTTPException(status_code=404, detail=f"CantusDB source {source_id} not found") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach cantusdatabase.org: {exc}") from exc
    if not rows:
        raise HTTPException(status_code=404, detail=f"CantusDB source {source_id} has no chants")
    shelfmark = (rows[0].get("shelfmark") or "").strip()
    institution = (rows[0].get("holding_institution") or "").strip()
    name = f"{institution} - {shelfmark}" if institution and shelfmark else (institution or shelfmark or f"source {source_id}")
    folios = sorted(
        {(r.get("folio") or "").strip() for r in rows if (r.get("folio") or "").strip()},
        key=_folio_sort_key,
    )
    data = {"sourceId": str(source_id), "name": name, "folios": folios}
    _cantus_cache_put(source_id, data)
    return data

@app.post("/run")
async def run_text_pipeline(
    image: UploadFile = File(...),
    folio: Optional[str] = Form(None),
    source_id: Optional[int] = Form(None),
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
    if source_id is not None and not (folio or "").strip():
        raise HTTPException(status_code=400, detail="folio is required when source_id is given")

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
            if source_id is not None:
                yield event({"type": "log", "message": f"running Kraken segmentation + HTR (Cantus-aligned mode, source {source_id}, folio {folio})..."})
            elif effective_recognition_model:
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
                        source_id=source_id,
                        segmentation_model=segmentation_model,
                        recognition_model=effective_recognition_model,
                        device=device,
                        column_bimodal_threshold=column_bimodal_threshold,
                        column_count=column_count,
                        ocr_only_mode=(source_id is None),
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
                collection, str(active_image_path), manifest, folio=folio, mode=("ocr_only" if source_id is None else "cantus_aligned"),
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

@app.post("/batch-run")
async def run_text_batch(
    images: list[UploadFile] = File(...),
    folios: str = Form(...),
    source_id: int = Form(...),
    segmentation_model: Optional[str] = Form(None),
    recognition_model: Optional[str] = Form(None),
    device: str = Form("cpu"),
    column_count: Optional[int] = Form(None),
    column_bimodal_threshold: float = Form(0.5),
    music_boxes: Optional[str] = Form(None),
    mask_json_list: Optional[str] = Form(None),
    masking_enabled: bool = Form(True),
    mask_padding: int = Form(15),
):
    folio_list = json.loads(folios)
    if len(folio_list) != len(images):
        raise HTTPException(status_code=400, detail="folios count must match images count")
    if len(folio_list) < 2:
        raise HTTPException(status_code=400, detail="batch requires at least 2 folios -  use /run for a single image")
    
    # Parallel per-folio arrays, JSON-encoded the same way `folios` already
    # is — mirrors /run's single music_boxes/mask_json fields, pluralized.
    parsed_music_boxes = json.loads(music_boxes) if music_boxes else [[] for _ in folio_list]
    parsed_mask_json = json.loads(mask_json_list) if mask_json_list else [None for _ in folio_list]

    image_blobs = [(img.filename or f"page_{i}.jpg", await img.read()) for i, img in enumerate(images)]
    effective_recognition_model = recognition_model or RECOGNITION_MODEL

    def generate():
        def event(obj):
            return f"data: {json.dumps(obj)}\n\n"
        
        batch_id = _uuid.uuid4().hex
        tmp_in = Path(tempfile.mkdtemp(prefix="batch-in-"))
        tmp_out = Path(tempfile.mkdtemp(prefix="batch-out-"))
        log_queue: "queue.Queue[dict]" = queue.Queue()
        handler = _QueueLogHandler(log_queue)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        try:
            yield event({"type": "stage", "name": "checking"})
            image_paths = []
            for i, (filename, data) in enumerate(image_blobs):
                p = tmp_in / f"{i:03d}_{filename}"
                p.write_bytes(data)
                image_paths.append(str(p))
            csv_rows = fetch_cantus_csv(source_id)
            stems = [make_output_stem(csv_rows, f) for f in folio_list]
            yield event({"type": "log", "message": f"{len(image_paths)} image(s) staged for folios {folio_list[0]}–{folio_list[-1]}"})
            yield event({"type": "stage_done", "name": "checking"})

            yield event({"type": "stage", "name": "validating"})
            yield event({"type": "log", "message": f"running Kraken segmentation + HTR across {len(folio_list)} folio(s) (Cantus-aligned mode, source {source_id})..."})
            if segmentation_model:
                yield event({"type": "log", "message": f"using custom segmentation model: {segmentation_model}"})
            if column_count:
                yield event({"type": "log", "message": f"column count forced to {column_count}"})
            yield event({"type": "stage_done", "name": "validating"})

            yield event({"type": "stage", "name": "processing"})
            result_holder: dict = {"completed": 0}
            
            def _worker():
                handler.thread_ident = threading.current_thread().ident
                logger = logging.getLogger(__name__)
                prev_state = None
                try:
                    for i, (image_path, folio, stem) in enumerate(zip(image_paths, folio_list, stems)):
                        logger.info("Folio %d/%d: %s", i + 1, len(folio_list), folio)
                        active_image_path = Path(image_path)
                        mask_tmp = None
                        folio_mask_json = parsed_mask_json[i] if i < len(parsed_mask_json) else None
                        if masking_enabled and folio_mask_json:
                            try:
                                mask_tmp = _apply_mothra_mask(active_image_path, folio_mask_json, mask_padding)
                            except Exception as exc:
                                logger.info("folio %s: text-region masking failed, continuing unmasked: %s", folio, exc)
                            if mask_tmp:
                                active_image_path = mask_tmp
                        state_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
                        state_path = state_tmp.name
                        state_tmp.close()
                        try:
                            collection, manifest = run(
                                image_path=str(active_image_path),
                                folio=folio,
                                source_id=source_id,
                                segmentation_model=segmentation_model,
                                recognition_model=effective_recognition_model,
                                device=device,
                                column_bimodal_threshold=column_bimodal_threshold,
                                prev_folio_state=prev_state,
                                folio_state_out=state_path,
                                column_count=column_count,
                            )
                            payload = _build_pipeline_payload(
                                collection, str(active_image_path), manifest, folio=stem, mode="cantus_aligned",
                            )
                            folio_boxes = parsed_music_boxes[i] if i < len(parsed_music_boxes) else []
                            if folio_boxes:
                                kept_lines, n_dropped = filter_lines_over_music(payload["lines"], folio_boxes)
                                payload["lines"] = kept_lines
                                if n_dropped:
                                    logger.info("folio %s: dropped %d line(s) overlapping YOLO music regions", folio, n_dropped)
                            mei_json_path = tmp_out / f"{stem}.json"
                            _write_mei_json(payload, str(mei_json_path))
                            text_alignment = json.loads(mei_json_path.read_text())
                            # Relayed through the same log_queue the SSE loop
                            # below already drains — batch_api.py intercepts
                            # this event type to persist text_alignments per
                            # folio as it completes, not just at the end.
                            log_queue.put({
                                "type": "folio_result",
                                "image_index": i,
                                "folio": folio,
                                "text_alignment": text_alignment,
                            })
                            prev_state = read_folio_state(state_path)
                            result_holder["completed"] += 1
                        finally:
                            Path(state_path).unlink(missing_ok=True)
                            if mask_tmp:
                                mask_tmp.unlink(missing_ok=True)
                except Exception as exc:
                    result_holder["error"] = exc
                    result_holder["failed_folio"] = folio_list[result_holder["completed"]]
            
            worker = threading.Thread(target=_worker, daemon=True)
            worker.start()

            while worker.is_alive() or not log_queue.empty():
                try:
                    yield event(log_queue.get(timeout=0.2))
                except queue.Empty:
                    continue
            worker.join()

            if "error" in result_holder:
                n = len(folio_list)
                yield event({
                    "type": "error",
                    "message": (
                        f"Chain aborted at folio {result_holder['failed_folio']} "
                        f"({result_holder['completed'] + 1}/{n}): {result_holder['error']}"
                    ),
                })
                return
            
            output_files = sorted(tmp_out.glob("*.json"))
            if not output_files:
                yield event({"type": "error", "message": "batch completed but produced no output files"})
                return
            zip_path = BATCH_DIR / f"{batch_id}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in output_files:
                    zf.write(f, arcname=f.name)
            yield event({"type": "log", "message": f"{len(output_files)} folio(s) aligned"})
            yield event({"type": "stage_done", "name": "processing"})
            yield event({"type": "result", "batchId": batch_id, "fileCount": len(output_files)})
            yield event({"type": "done"})
        except Exception as e:
            yield event({"type": "error", "message": str(e)})
        finally:
            root_logger.removeHandler(handler)
            shutil.rmtree(tmp_in, ignore_errors=True)
            shutil.rmtree(tmp_out, ignore_errors=True)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/batch-download/{batch_id}")
def download_batch(batch_id: str):
    zip_path = BATCH_DIR / f"{batch_id}.zip"
    if not zip_path.is_file():
        raise HTTPException(status_code=404, detail="batch result not found or expired")
    return FileResponse(zip_path, media_type="application/zip", filename=f"batch-{batch_id}.zip")