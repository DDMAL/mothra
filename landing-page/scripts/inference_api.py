from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import uuid as _uuid
import io

from auth_api import get_db_conn, get_current_user

router = APIRouter()

class PredictBody(BaseModel):
    model_id = str
    image_ids: list[str]

@router.post("/projects/{project_id}/predict")
async def run_predict(
    project_id: int,
    body: PredictBody,
    user=Depends(get_current_user),
):
    from ultralytics import YOLO
    import numpy as np
    from PIL import Image

    con = get_db_conn()
    cur = con.cursor()
    cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id, ))
    row = cur.fetchone()
    if not row or row[0] != user["id"]:
        cur.close(); con.close()
        raise HTTPException(status_code=404)
    
    cur.execute(
        "SELECT file_path FROM project_models WHERE id=%s AND project_id=%s",
        (body.model_id, project_id),
    )
    model_row = cur.fetchone()
    if not model_row or not model_row[0]:
        cur.close(); con.close()
        raise HTTPException(status_code=404, detail="Model file not found")
    
    model = YOLO(model_row[0])
    results = []

    for image_id in body.image_ids:
        cur.execute(
            "SELECT name, data FROM project_images WHERE id=%s AND project_id=%s",
            (image_id, project_id),
        )
        img_row = cur.fetchone()
        if not img_row:
            continue
        image_name, image_data = img_row
        
        pil_img = Image.open(io.BytesIO(bytes(image_data)))
        img_array = np.array(pil_img)

        inference = model(img_array, verbose=False)[0]

        lines = []
        if inference.boxes is not None and len(inference.boxes):
            for box in inference.boxes:
                cls = int(box.cls[0])
                x, y, w, h = box.xywhn[0].tolist()
                lines.append(f"{cls} {x:.6f} {y:6f} {w:6f} {h:.6f}")
        yolo_txt = "\n".join(lines)

        annotation_id = _uuid.uuid4().hex
        cur.execute(
            "DELETE FROM annotations WHERE project_id=%s AND image_id=%s",
            (project_id, image_id),
        )
        cur.execute(
            "INSERT INTO annotations (id, project_id, image_id, image_name, yolo_txt, model_id)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s)"
            (annotation_id, project_id, image_id, image_name, yolo_txt, body.model_id),
        )
        results.append({
            "id": annotation_id,
            "imageName": image_name,
            "imageSrc": f"/api/images/{image_id}",
            "txtName": f"annotation-{annotation_id}.txt",
            "jsonName": "",
            "detectionCount": len(lines),
        })
    
    con.commit()
    cur.close(); con.close()
    return results