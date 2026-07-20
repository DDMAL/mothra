from typing import Optional, Literal
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import uuid as _uuid

from auth_api import get_current_user, require_project_owner, db_cursor
from job_store import create_job
from tasks_predict import run_predict_task

router = APIRouter()

class PredictBody(BaseModel):
    model_id: Optional[str] = None
    model_preset: Literal["medieval", "printed", "custom"] = "medieval"
    image_ids: list[str]
    confidence_threshold: float = 0.5
    device: str = "auto"   # auto → GPU if present else CPU (resolved in yolo_inference)
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
    job_id = _uuid.uuid4().hex[:8]
    create_job(job_id, "predict", project_id)
    run_predict_task.apply_async(
        kwargs={"job_id": job_id, "project_id": project_id, "body": body.model_dump()},
        task_id=job_id,
    )
    return {"job_id": job_id}

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