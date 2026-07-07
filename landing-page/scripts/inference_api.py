from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import json
import uuid as _uuid
import io

from auth_api import get_db_conn, get_current_user, release_db_conn, get_model_file_path
from text_api import stream_text_finding

router = APIRouter()

class PredictBody(BaseModel):
    model_id: str
    image_ids: list[str]
    confidence_threshold: float = 0.5
    device: str = "cpu"
    text_column_count: Optional[int] = None
    text_segmentation_model_id: Optional[str] = None
    text_recognition_model_id: Optional[str] = None
    text_device: str = "cpu"
    text_column_bimodal_threshold: float = 0.5

@router.post("/projects/{project_id}/predict")
async def run_predict(
    project_id: int,
    body: PredictBody,
    user=Depends(get_current_user),
):
    con = get_db_conn()
    cur = con.cursor()
    cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id, ))
    row = cur.fetchone()
    if not row or row[0] != user["id"]:
        cur.close(); release_db_conn(con)
        raise HTTPException(status_code=404)
    
    model_id = body.model_id
    image_ids = body.image_ids

    def generate():
        from ultralytics import YOLO
        import numpy as np
        from PIL import Image

        def event(obj): return f"data: {json.dumps(obj)}\n\n"

        con = get_db_conn()
        cur = con.cursor()
        try:
            # stage 1 - checking
            yield event({"type": "stage", "name": "checking"})
            model_row = get_model_file_path(cur, project_id, model_id, "yolo")
            if not model_row: 
                yield event({"type": "error", "message": "Model file not found"}); return
            model = YOLO(model_row[0])
            yield event({"type": "log", "message": f"Model loaded: {model_row[1]}"})

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
                rec_row = get_model_file_path(cur, project_id, body.text_recognition_model_id, "recognition")
                if rec_row:
                    rec_model_path = rec_row[0]
                    yield event({"type": "log", "message": f"OCR model: {rec_row[1]}"})
                else:
                    yield event({"type": "log", "message": "text-finding: custom OCR model not found — using default"})

            yield event({"type": "stage_done", "name": "checking"})

            # stage 2 - validation
            yield event({"type": "stage", "name": "validating"})
            images = []
            for iid in image_ids:
                cur.execute("SELECT name, data, mime_type FROM project_images WHERE id=%s AND project_id=%s",
                            (iid, project_id))
                r = cur.fetchone()
                if r: images.append((iid, r[0], r[1], r[2] or "image/png"))
            yield event({"type": "log", "message": f"{len(images)} image(s) ready"})
            yield event({"type": "stage_done", "name": "validating"})

            # stage 3 - processing
            yield event({"type": "stage", "name": "processing"})
            results = []
            for image_id, image_name, image_data, mime_type in images:
                yield event({"type": "log", "message": f"Processing {image_name}..."})
                pil_img = Image.open(io.BytesIO(bytes(image_data))).convert("RGB")
                inference = model(np.array(pil_img), device=body.device, verbose=False)[0]
                lines = []
                if inference.boxes is not None and len(inference.boxes):
                    for box in inference.boxes:
                        if float(box.conf[0]) < body.confidence_threshold:
                            continue
                        cls = int(box.cls[0])
                        x, y, w, h = box.xywhn[0].tolist()
                        lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                yolo_txt = "\n".join(lines)
                ann_id = _uuid.uuid4().hex
                cur.execute("DELETE FROM annotations WHERE project_id=%s AND image_id=%s",
                            (project_id, image_id))
                cur.execute(
                    "INSERT INTO annotations (id, project_id, image_id, image_name, yolo_txt, model_id)"
                    " VALUES (%s,%s,%s,%s,%s,%s)",
                    (ann_id, project_id, image_id, image_name, yolo_txt, model_id)
                )
                con.commit()
                yield event({"type": "log", "message": f"{image_name}: {len(lines)} detection(s)"})
                results.append({
                    "id": ann_id, "imageName": image_name,
                    "imageSrc": f"/api/images/{image_id}",
                    "txtName": f"annotation-{ann_id}.txt",
                    "jsonName": "", "detectionCount": len(lines),
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