"""FastAPI wrapper exposing mothra-text's run_pipeline as an HTTP service.

Runs as its own process (own venv, port 8002 by default) — see dev.sh.
Mirrors the SSE event contract used by inference_api.py's /predict and
encode_api.py's /encode-upload: stage -> stage_done -> log -> result -> done.
"""
import json
import os
import re
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
from run_chain import _are_contiguous

app = FastAPI()
# No "*" default -- see landing-page/scripts/main.py's identical fix (mothra#220
# row 27). text-service has no public ingress route (ClusterIP only, see
# k8s/text-service.yaml) so this middleware never actually sees real browser
# traffic today, but an unconditional wildcard is still the wrong default to
# leave in code for whenever that changes. Falls back to the same local-dev
# Vite origin as the backend when unset.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])

# run_pipeline.main() resolves this as the --recognition-model CLI default
# (_DEFAULT_RECOGNITION_MODEL); run() itself defaults to None (stub mode, empty
# OCR text) since we call it directly instead of going through main()/argparse.
RECOGNITION_MODEL = _find_tridis_model()

@app.get("/healthz", include_in_schema=False)
def healthz():
    """text-service has no DB/broker of its own -- unlike the backend's
    /healthz (mothra#220 row 29), there's no external dependency to check
    reachability of. Still a real improvement over the bare tcpSocket probe
    it replaces (k8s/text-service.yaml): confirms this process is actually
    serving HTTP, not just that the OS has a listener on the port. Always
    returns 200 -- a missing recognition_model means text-finding silently
    runs in stub mode (segmentation/YOLO still work, no OCR text), which is
    a real but degraded capability, not a reason to fail the probe and pull
    this pod out of rotation; recognition_model is reported for visibility."""
    return {"status": "ok", "recognition_model": RECOGNITION_MODEL is not None}

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

def filter_lines_over_music(lines: list[dict], music_boxes: list[list[float]], threshold: float = 0.3) -> tuple[list[dict], list[dict]]:
    """Drop detected text lines that mostly overlap a YOLO music region - BLLA over-segmentation artifacts from neume notation, not real chant text.

    Returns (kept, dropped) - the dropped line dicts themselves, not just a
    count, so callers can log exactly what was removed (issue #131 flagged this
    filter as an unauditable silent drop on manuscripts with interleaved text
    and music)."""
    if not music_boxes:
        return lines, []
    kept, dropped = [], []
    for line in lines:
        if any(_bbox_overlap_ratio(line["bbox"], mb) > threshold for mb in music_boxes):
            dropped.append(line)
            continue
        kept.append(line)
    return kept, dropped

def _write_mask_json_tmp(mask_json: Optional[str]) -> Optional[Path]:
    """Write mask_json content to a scratch temp file so run()'s own
    mothra_json_path/padding masking (mothra-text/run_pipeline.py) can read it,
    instead of text-service pre-masking the image itself. run() didn't support
    mothra_json_path when this integration was first written; now that it does,
    this is the sanctioned single implementation of the masking logic instead of
    a second hand-rolled copy that can drift from it over time.

    Returns the path to the temp JSON file, or None if mask_json is falsy.
    Caller owns deleting the returned path.
    """
    if not mask_json:
        return None
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as _tmpf:
        _tmpf.write(mask_json)
        return Path(_tmpf.name)

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
    m = re.search(r"\(([^)]+)\)$", institution) if institution else None
    institution_code = m.group(1) if m else institution
    siglum = f"{institution_code} {shelfmark}".strip() if institution_code and shelfmark else None
    data = {"sourceId": str(source_id), "name": name, "folios": folios, "siglum": siglum}
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
    music_overlap_filter_enabled: bool = Form(True),
    debug_mode: bool = Form(False),
):
    """Run Kraken segmentation + HTR over a single image and stream progress as SSE.

    Stages a mothra-mask JSON to a scratch temp file (if masking is enabled
    and provided) for `run()` to apply itself via `mothra_json_path`, invokes
    `run()` on a worker thread (relaying its logger through `_QueueLogHandler`
    onto this request's SSE stream), then optionally drops text lines that
    mostly overlap a YOLO music region via `filter_lines_over_music` before
    emitting the final `text_alignment`/`log_text` result event.
    """
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
        _mask_json_tmp: Optional[Path] = None
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

            mothra_json_path: Optional[str] = None
            if masking_enabled and mask_json:
                try:
                    _mask_json_tmp = _write_mask_json_tmp(mask_json)
                    mothra_json_path = str(_mask_json_tmp)
                    yield event({"type": "log", "message": f"text-region mask will be applied by run() (padding={mask_padding}px)"})
                except Exception as exc:
                    yield event({"type": "log", "message": f"text-region masking setup failed, continuing unmasked: {exc}"})
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
                        image_path=str(image_path),
                        folio=folio,
                        source_id=source_id,
                        segmentation_model=segmentation_model,
                        recognition_model=effective_recognition_model,
                        device=device,
                        column_bimodal_threshold=column_bimodal_threshold,
                        column_count=column_count,
                        ocr_only_mode=(source_id is None),
                        mothra_json_path=mothra_json_path,
                        padding=mask_padding,
                        music_boxes=parsed_music_boxes if music_overlap_filter_enabled else None,
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
            dropped_lines = getattr(collection, '_music_filter_dropped', [])
            if dropped_lines:
                yield event({"type": "log", "message": f"dropped {len(dropped_lines)} line(s) overlapping YOLO music regions before NW alignment"})
            payload = _build_pipeline_payload(
                collection, str(image_path), manifest, folio=folio, mode=("ocr_only" if source_id is None else "cantus_aligned"),
            )
            lines_pre_filter = list(payload["lines"]) if debug_mode else None
            mei_json_path = tmp_dir / "mei_alignment.json"
            _write_mei_json(payload, str(mei_json_path))
            text_alignment = json.loads(mei_json_path.read_text())
            n_syl = len(text_alignment.get("syl_boxes", []))
            yield event({"type": "log", "message": f"{n_syl} syllable(s) aligned"})
            yield event({"type": "stage_done", "name": "processing"})
            result_ev: dict = {"type": "result", "text_alignment": text_alignment}
            if debug_mode:
                mask_boxes = json.loads(mask_json).get("annotations", []) if mask_json else []
                result_ev["debug_data"] = {
                    "mothra_text": {
                        "mask": {
                            "box_count": len(mask_boxes),
                            "boxes": mask_boxes,
                        },
                        "run_params": {
                            "folio": folio,
                            "source_id": source_id,
                            "padding": mask_padding,
                            "column_bimodal_threshold": column_bimodal_threshold,
                            "column_count": column_count,
                            "ocr_only_mode": source_id is None,
                            "segmentation_model": segmentation_model,
                            "recognition_model": effective_recognition_model,
                            "device": device,
                        },
                        "lines_pre_filter": lines_pre_filter,
                        "music_filter": {
                            "enabled": music_overlap_filter_enabled,
                            "threshold": 0.30,
                            "lines_dropped": dropped_lines,
                        },
                    }
                }
            yield event(result_ev)
            yield event({"type": "done"})
        except Exception as e:
            yield event({"type": "error", "message": str(e)})
        finally:
            root_logger.removeHandler(handler)
            if _mask_json_tmp:
                _mask_json_tmp.unlink(missing_ok=True)
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
    music_overlap_filter_enabled: bool = Form(True),
    debug_mode: bool = Form(False),
):
    """Run Kraken segmentation + HTR over a batch of Cantus-aligned folios and
    stream progress as SSE.

    Same per-folio pipeline as `run_text_pipeline` (mask staging, `run()`,
    optional `filter_lines_over_music`), but iterated across `images`/`folios`
    with `prev_folio_state`/`folio_state_out` threaded from one folio to the
    next so `run()` can track chant allocation continuity across the batch.
    """
    folio_list = json.loads(folios)
    if len(folio_list) != len(images):
        raise HTTPException(status_code=400, detail="folios count must match images count")
    if len(folio_list) < 1:
        raise HTTPException(status_code=400, detail="batch requires at least 1 folio")
    
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
                        # infer_continuation=True (build_flat_text_and_anchors' own
                        # default) would otherwise let it independently scan the CSV
                        # for "the nearest preceding folio with a 77 break" and
                        # re-derive the same wrong continuation even after prev_state
                        # is reset below - it has no idea this folio's true physical
                        # predecessor was skipped in this batch, only that no explicit
                        # prev_folio_state was passed. Must be suppressed in the same
                        # branch, not just prev_state.
                        infer_continuation = True
                        if i > 0 and not _are_contiguous(folio_list[i - 1], folio):
                            # e.g. a folio was intentionally skipped in this batch (or is
                            # simply not the next physical page) - carrying the previous
                            # folio's leftover continuation words into this one would
                            # silently corrupt its alignment from the very first line.
                            # Mirrors run_chain.py's identical reset for the CLI chain.
                            logger.info(
                                "folio %s: not contiguous with previous folio %s, resetting FolioState",
                                folio, folio_list[i - 1],
                            )
                            prev_state = None
                            infer_continuation = False
                        mask_json_tmp = None
                        folio_mask_json = parsed_mask_json[i] if i < len(parsed_mask_json) else None
                        mothra_json_path = None
                        if masking_enabled and folio_mask_json:
                            try:
                                mask_json_tmp = _write_mask_json_tmp(folio_mask_json)
                                mothra_json_path = str(mask_json_tmp)
                            except Exception as exc:
                                logger.info("folio %s: text-region masking setup failed, continuing unmasked: %s", folio, exc)
                        state_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
                        state_path = state_tmp.name
                        state_tmp.close()
                        try:
                            folio_boxes = parsed_music_boxes[i] if i < len(parsed_music_boxes) else []
                            collection, manifest = run(
                                image_path=image_path,
                                folio=folio,
                                source_id=source_id,
                                segmentation_model=segmentation_model,
                                recognition_model=effective_recognition_model,
                                device=device,
                                column_bimodal_threshold=column_bimodal_threshold,
                                prev_folio_state=prev_state,
                                folio_state_out=state_path,
                                infer_continuation=infer_continuation,
                                column_count=column_count,
                                mothra_json_path=mothra_json_path,
                                padding=mask_padding,
                                music_boxes=folio_boxes if music_overlap_filter_enabled else None,
                            )
                            dropped_lines = getattr(collection, '_music_filter_dropped', [])
                            if dropped_lines:
                                logger.info(
                                    "folio %s: dropped %d line(s) overlapping YOLO music regions before NW alignment: %s",
                                    folio, len(dropped_lines), [dl["bbox"] for dl in dropped_lines],
                                )
                            payload = _build_pipeline_payload(
                                collection, image_path, manifest, folio=stem, mode="cantus_aligned",
                            )
                            lines_pre_filter = list(payload["lines"]) if debug_mode else None
                            mei_json_path = tmp_out / f"{stem}.json"
                            _write_mei_json(payload, str(mei_json_path))
                            text_alignment = json.loads(mei_json_path.read_text())
                            folio_result: dict = {
                                "type": "folio_result",
                                "image_index": i,
                                "folio": folio,
                                "text_alignment": text_alignment,
                            }
                            if debug_mode:
                                folio_mask_json = parsed_mask_json[i] if i < len(parsed_mask_json) else None
                                mask_boxes = json.loads(folio_mask_json).get("annotations", []) if folio_mask_json else []
                                folio_result["debug_data"] = {
                                    "mothra_text": {
                                        "mask": {"box_count": len(mask_boxes), "boxes": mask_boxes},
                                        "run_params": {
                                            "folio": folio, "source_id": source_id,
                                            "padding": mask_padding,
                                            "column_bimodal_threshold": column_bimodal_threshold,
                                            "column_count": column_count,
                                            "segmentation_model": segmentation_model,
                                            "recognition_model": effective_recognition_model,
                                            "device": device,
                                        },
                                        "lines_pre_filter": lines_pre_filter,
                                        "music_filter": {
                                            "enabled": music_overlap_filter_enabled,
                                            "threshold": 0.30,
                                            "lines_dropped": dropped_lines,
                                        },
                                    }
                                }
                            # Relayed through the same log_queue the SSE loop
                            # below already drains — batch_api.py intercepts
                            # this event type to persist text_alignments per
                            # folio as it completes, not just at the end.
                            log_queue.put(folio_result)
                            prev_state = read_folio_state(state_path)
                            result_holder["completed"] += 1
                        finally:
                            Path(state_path).unlink(missing_ok=True)
                            if mask_json_tmp:
                                mask_json_tmp.unlink(missing_ok=True)
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