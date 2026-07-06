from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import json
import uuid as _uuid
import io

from auth_api import get_db_conn, get_current_user, release_db_conn

router = APIRouter()

class PredictBody(BaseModel):
    model_id: str
    image_ids: list[str]
    confidence_threshold: float = 0.5
    device: str = "cpu"

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
            cur.execute("SELECT file_path, name FROM project_models WHERE id=%s AND project_id=%s",
                        (model_id, project_id))
            model_row = cur.fetchone()
            if not model_row or not model_row[0]:
                yield event({"type": "error", "message": "Model file not found"}); return
            model = YOLO(model_row[0])
            yield event({"type": "log", "message": f"Model loaded: {model_row[1]}"})
            yield event({"type": "stage_done", "name": "checking"})

            # stage 2 - validatin
            yield event({"type": "stage", "name": "validating"})
            images = []
            for iid in image_ids:
                cur.execute("SELECT name, data FROM project_images WHERE id=%s AND project_id=%s", 
                            (iid, project_id))
                r = cur.fetchone()
                if r: images.append((iid, r[0], r[1]))
            yield event({"type": "log", "message": f"{len(images)} image(s) ready"})
            yield event({"type": "stage_done", "name": "validating"})

            # stage 3 - processing
            yield event({"type": "stage", "name": "processing"})
            results = []
            for image_id, image_name, image_data in images:
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