from typing import Optional, Literal
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from pathlib import Path
from fastapi.responses import StreamingResponse
import json
import uuid as _uuid
import io
from medieval_models import resolve_medieval_model_paths, TEXT_MUSIC_CLASS_MAP, STAVE_CLASS_MAP

from auth_api import get_db_conn, get_current_user, release_db_conn, db_cursor, require_project_owner, _log_activity
from models_api import get_model_file_path
from text_api import stream_text_finding

router = APIRouter()

CATEGORY_TO_SLOT = {"text": 0, "music": 1, "staves": 2}

def _append_boxes(lines, inference, cls_map, threshold):
    if inference.boxes is None or not len(inference.boxes):
        return
    for box in inference.boxes:
        if float(box.conf[0]) < threshold:
            continue
        raw_cls = int(box.cls[0])
        cls = cls_map.get(raw_cls) if cls_map is not None else raw_cls
        if cls is None:
            continue
        x, y, w, h = box.xywhn[0].tolist()
        lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

class YoloModelSet:
    """A resolved YOLO model (or medieval pair) ready to run inference on
    one image at a time. Built by resolve_yolo_models(); shared by
    run_predict (single/grid) and batch_api.py's run_text_batch."""

    def __init__(self, medieval_models, class_maps, single_model, custom_cls_map,
                 tm_threshold, tm_device, st_threshold, st_device,
                 confidence_threshold, device,
                 stored_model_id, model_label, model_hash):
        self.medieval_models = medieval_models
        self.class_maps = class_maps
        self.single_model = single_model
        self.custom_cls_map = custom_cls_map
        self.tm_threshold = tm_threshold
        self.tm_device = tm_device
        self.st_threshold = st_threshold
        self.st_device = st_device
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.stored_model_id = stored_model_id
        self.model_label = model_label
        self.model_hash = model_hash

    def infer(self, img_arr) -> str:
        lines = []
        if self.medieval_models is not None:
            tm_model, st_model = self.medieval_models
            tm_map, st_map = self.class_maps
            _append_boxes(lines, tm_model(img_arr, device=self.tm_device, verbose=False)[0], tm_map, self.tm_threshold)
            _append_boxes(lines, st_model(img_arr, device=self.st_device, verbose=False)[0], st_map, self.st_threshold)
        else:
            _append_boxes(lines, self.single_model(img_arr, device=self.device, verbose=False)[0], self.custom_cls_map, self.confidence_threshold)
        return "\n".join(lines)


def resolve_yolo_models(
        cur, project_id: int, model_preset: str, model_id: Optional[str],
        confidence_threshold: float, device: str,
        text_music_confidence_threshold: Optional[float], text_music_device: Optional[str],
        stave_confidence_threshold: Optional[float], stave_device: Optional[str],
) -> tuple["YoloModelSet", list[str]]:
    """Loads the requested YOLO model(s), returns a ready-to-use
    YoloModelSet plus human-readable log lines describing what loaded.
    Raises RuntimeError (medieval preset unavailable) or ValueError (custom
    model not found) - callers decide how to surface that as an SSE error.
    """
    from ultralytics import YOLO

    tm_threshold = text_music_confidence_threshold if text_music_confidence_threshold is not None else confidence_threshold
    tm_device = text_music_device or device
    st_threshold = stave_confidence_threshold if stave_confidence_threshold is not None else confidence_threshold
    st_device = stave_device or device

    if model_preset == "medieval":
        tm_path, st_path = resolve_medieval_model_paths()
        model_set = YoloModelSet(
            medieval_models=(YOLO(tm_path), YOLO(st_path)),
            class_maps=(TEXT_MUSIC_CLASS_MAP, STAVE_CLASS_MAP),
            single_model=None, custom_cls_map=None,
            tm_threshold=tm_threshold, tm_device=tm_device,
            st_threshold=st_threshold, st_device=st_device,
            confidence_threshold=confidence_threshold, device=device,
            stored_model_id=model_preset,
            model_label="medieval manuscripts (text_music_detector_fulldata.pt + stave_detector_fulldata.pt)",
            model_hash=None,
        )
        return model_set, ["medieval manuscripts preset: loaded text/music + stave detectors"]
    
    model_row = get_model_file_path(cur, project_id, model_id, "yolo")
    if not model_row:
        raise ValueError("Model file not found")
    file_path, model_name, class_map_json, file_hash = model_row
    custom_cls_map = None
    if class_map_json:
        raw_map = json.loads(class_map_json)
        custom_cls_map = {int(k): CATEGORY_TO_SLOT[v] for k, v in raw_map.items()}
    model_set = YoloModelSet(
        medieval_models=None, class_maps=None,
        single_model=YOLO(file_path), custom_cls_map=custom_cls_map,
        tm_threshold=tm_threshold, tm_device=tm_device,
        st_threshold=st_threshold, st_device=st_device,
        confidence_threshold=confidence_threshold, device=device,
        stored_model_id=model_id,
        model_label=f"custom: {model_name}", model_hash=file_hash,
    )
    return model_set, [f"Model loaded: {model_name}"]

def write_annotation(cur, con, project_id: int, image_id: str, image_name: str,
                     yolo_txt: str, stored_model_id: str, model_label: str,
                     model_hash: Optional[str]) -> str:
    """Replaces this image's annotations row. Returns the new row's id."""
    cur.execute("DELETE FROM annotations WHERE project_id=%s AND image_id=%s", (project_id, image_id))
    ann_id = _uuid.uuid4().hex
    cur.execute(
        "INSERT INTO annotations (id, project_id, image_id, image_name, yolo_txt, model_id, model_label, model_hash)"
        " VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
        (ann_id, project_id, image_id, image_name, yolo_txt, stored_model_id, model_label, model_hash),
    )
    con.commit()
    return ann_id

class PredictBody(BaseModel):
    model_id: Optional[str] = None
    model_preset: Literal["medieval", "printed", "custom"] = "medieval"
    image_ids: list[str]
    confidence_threshold: float = 0.5
    device: str = "cpu"
    text_music_confidence_threshold: Optional[float] = None
    text_music_device: Optional[str] = None
    stave_confidence_threshold: Optional[float] = None
    stave_device: Optional[str] = None
    text_column_count: Optional[int] = None
    text_segmentation_model_id: Optional[str] = None
    text_recognition_model_id: Optional[str] = None
    text_device: str = "cpu"
    text_column_bimodal_threshold: float = 0.5
    text_masking_enabled: bool = True
    text_mask_padding: int = 15
    text_mask_model_id: Optional[str] = None
    text_source_id: Optional[int] = None

@router.post("/projects/{project_id}/predict")
async def run_predict(
    project_id: int,
    body: PredictBody,
    user=Depends(get_current_user),
):
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])

    if body.model_preset == "printed":
        raise HTTPException(status_code=400, detail="printed text detection is not available yet!")
    if body.model_preset == "custom" and not body.model_id: 
        raise HTTPException(status_code=400, detail="model_id is required when model_preset is 'custom'")
    
    model_preset = body.model_preset
    model_id = body.model_id
    image_ids = body.image_ids

    def generate():
        import numpy as np
        from PIL import Image

        def event(obj): return f"data: {json.dumps(obj)}\n\n"

        con = get_db_conn()
        cur = con.cursor()
        try:
            # stage 1 - checking
            yield event({"type": "stage", "name": "checking"})

            try:
                yolo_models, load_logs = resolve_yolo_models(
                     cur, project_id, model_preset, model_id,
                    body.confidence_threshold, body.device,
                    body.text_music_confidence_threshold, body.text_music_device,
                    body.stave_confidence_threshold, body.stave_device,
                )
            except RuntimeError as e:
                yield event({"type": "error", "message": str(e)}); return
            except ValueError as e:
                yield event({"type": "error", "message": str(e)}); return
            for msg in load_logs:
                yield event({"type": "log", "message": msg})

            seg_model_path = None
            if body.text_segmentation_model_id:
                seg_row = get_model_file_path(cur, project_id, body.text_segmentation_model_id, "segmentation")
                if seg_row:
                    seg_model_path = seg_row[0]
                    yield event({"type": "log", "message": f"Segmentation model: {seg_row[1]}"})
                else:
                    yield event({"type": "log", "message": "text-finding: custom segmentation model not found — using default"})
            rec_model_path = None
            if body.text_recognition_model_id:
                rec_row = get_model_file_path(cur, project_id,body.text_recognition_model_id, "recognition")
                if rec_row:
                    rec_model_path = rec_row[0]
                    yield event({"type": "log", "message": f"OCR model: {rec_row[1]}"})
                else:
                    yield event({"type": "log", "message": "text-finding: custom OCR model not found — using default"})
            mask_json_override = None
            if body.text_mask_model_id:
                mask_row = get_model_file_path(cur, project_id, body.text_mask_model_id, "text_mask")
                if mask_row:
                    try:
                        mask_json_override = Path(mask_row[0]).read_text(encoding="utf-8")
                        yield event({"type": "log", "message": f"text-region mask: {mask_row[1]}"})
                    except Exception:
                        yield event({"type": "log", "message": "text-finding: custom mask JSON could not be read — using auto-derived mask"})
                else:
                    yield event({"type": "log", "message": "text-finding: custom mask model not found — using auto-derived mask"})
            yield event({"type": "stage_done", "name": "checking"})

            # stage 2 - validation
            yield event({"type": "stage", "name": "validating"})
            images = []
            skipped = []
            for iid in image_ids:
                cur.execute("SELECT name, data, mime_type, folio FROM project_images WHERE id=%s AND project_id=%s",
                            (iid, project_id))
                r = cur.fetchone()
                if not r:
                    continue
                cur.execute("SELECT 1 FROM annotations WHERE project_id=%s AND image_id=%s", (project_id, iid))
                if cur.fetchone():
                    skipped.append(r[0])
                    continue
                images.append((iid, r[0], r[1], r[2] or "image/png", r[3]))
            yield event({"type": "log", "message": f"{len(images)} image(s) ready"})
            if skipped:
                yield event({"type": "log", "message": f"skipping {len(skipped)} already-annotated image(s): {', '.join(skipped)}"})
            yield event({"type": "stage_done", "name": "validating"})

            _log_activity(cur, project_id, "predict_run", f"{yolo_models.model_label} on {len(images)} image(s)")
            con.commit()

            # stage 3 - processing
            yield event({"type": "stage", "name": "processing"})
            results = []
            for image_id, image_name, image_data, mime_type, image_folio in images:
                yield event({"type": "log", "message": f"Processing {image_name}..."})
                pil_img = Image.open(io.BytesIO(bytes(image_data))).convert("RGB")
                img_arr = np.array(pil_img)
                yolo_txt = yolo_models.infer(img_arr)
                ann_id = write_annotation(
                    cur, con, project_id, image_id, image_name, yolo_txt,
                    yolo_models.stored_model_id, yolo_models.model_label, yolo_models.model_hash,
                )
                n_detections = len(yolo_txt.splitlines()) if yolo_txt else 0
                yield event({"type": "log", "message": f"{image_name}: {n_detections} detection(s)"})
                results.append({
                    "id": ann_id, "imageName": image_name,
                    "imageSrc": f"/api/images/{image_id}",
                    "txtName": f"annotation-{ann_id}.txt",
                    "jsonName": "", "detectionCount": n_detections,
                })

                # Hand off to mothra-text now that this image's YOLO boxes
                # exist — runs as part of this same processing stage, in
                # the same stream, so its logs show up right here. Failures
                # are downgraded to log lines: text-finding must never fail
                # the visible (music-path) pipeline.
                yield event({"type": "log", "message": f"{image_name}: starting text-finding..."})
                for text_ev in stream_text_finding(
                    project_id, image_id, image_name,
                    bytes(image_data), mime_type,
                    column_count=body.text_column_count,
                    segmentation_model=seg_model_path,
                    recognition_model=rec_model_path,
                    device=body.text_device,
                    column_bimodal_threshold=body.text_column_bimodal_threshold,
                    masking_enabled=body.text_masking_enabled,
                    mask_padding=body.text_mask_padding,
                    mask_json_override=mask_json_override,
                    source_id=body.text_source_id if image_folio else None,
                    folio_override=image_folio,
                ):
                    if text_ev.get("type") == "log":
                        yield event(text_ev)
                    elif text_ev.get("type") == "error":
                        yield event({"type": "log", "message": f"text-finding: {text_ev.get('message', 'failed')}"})
            yield event({"type": "stage_done", "name": "processing"})
            yield event({"type": "result", "annotations": results})
            yield event({"type": "done"})
        except Exception as e:
            con.rollback()
            yield event({"type": "error", "message": str(e)})
        finally: 
            cur.close(); release_db_conn(con)
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/projects/{project_id}/annotations/{annotation_id}")
def delete_annotation(project_id: int, annotation_id: str, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        cur.execute("DELETE FROM annotations WHERE id=%s AND project_id=%s", (annotation_id, project_id))
        con.commit()
        return {"ok": True}


@router.get("/projects/{project_id}/annotations/{annotation_id}")
async def get_annotation_txt(
    project_id: int,
    annotation_id: str,
    user=Depends(get_current_user),
):
    with db_cursor() as (con, cur):
        cur.execute(
            "SELECT a.yolo_txt, a.image_name"
            " FROM annotations a"
            " JOIN projects p ON p.id = a.project_id"
            " WHERE a.id = %s AND a.project_id = %s AND p.user_id = %s",
            (annotation_id, project_id, user["id"]),
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404)
    return {"yoloTxt": row[0], "imageName": row[1]}