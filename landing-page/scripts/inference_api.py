import io
from typing import Optional, Literal
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import numpy as np
from PIL import Image
import uuid as _uuid

from auth_api import get_current_user, require_project_owner, db_cursor
from config import SKIP_YOLO
from job_store import create_job
from tasks_predict import run_predict_task
import staffline_stage

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
    text_music_overlap_filter_enabled: bool = True
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

    # MOTHRA_SKIP_YOLO means this box can't run ultralytics, so refuse here
    # rather than enqueueing a job that can only die in the worker — the caller
    # gets a reason instead of a failed row. Only covers this endpoint; a retry
    # of an older job replays params straight to the task, which is why
    # resolve_yolo_models() carries its own guard.
    if SKIP_YOLO:
        raise HTTPException(
            status_code=503,
            detail="YOLO inference is disabled on this server (MOTHRA_SKIP_YOLO is "
                   "set, normally because ultralytics isn't installed). Set "
                   "VITE_SKIP_PREDICT=1 in landing-page/.env.local so the UI skips "
                   "this step, or unset MOTHRA_SKIP_YOLO once ultralytics is "
                   "installed.",
        )
    if body.model_preset == "printed":
        raise HTTPException(status_code=400, detail="printed text detection is not available yet!")
    if body.model_preset == "custom" and not body.model_id: 
        raise HTTPException(status_code=400, detail="model_id is required when model_preset is 'custom'")
    new_id = _uuid.uuid4().hex[:8]
    kwargs = {"job_id": new_id, "project_id": project_id, "body": body.model_dump()}
    job_id, is_new = create_job(new_id, "predict", project_id,
                                 params={k: v for k, v in kwargs.items() if k != "job_id"},
                                 dedupe_seconds=5)
    if is_new:
        run_predict_task.apply_async(kwargs=kwargs, task_id=job_id)
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


@router.get("/projects/{project_id}/stafflines/{detection_id}")
async def get_staffline_detection(
    project_id: int,
    detection_id: str,
    user=Depends(get_current_user),
):
    with db_cursor() as (con, cur):
        cur.execute(
            "SELECT s.jsomr_json, s.image_name, s.scale_unit, s.stave_count,"
            " s.mode_lines_per_stave, s.status"
            " FROM staffline_detections s"
            " JOIN projects p ON p.id = s.project_id"
            " WHERE s.id = %s AND s.project_id = %s AND p.user_id = %s",
            (detection_id, project_id, user["id"]),
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404)
    return {
        "jsomrJson": row[0], "imageName": row[1], "scaleUnit": row[2],
        "staveCount": row[3], "modeLinesPerStave": row[4], "status": row[5],
    }


def _load_image_and_yolo_for_detection(cur, project_id: int, detection_id: str, user_id: int):
    """Looks up a staffline_detections row's image_id/image_name, then loads
    that image's bytes and its CURRENT annotation's yolo_txt -- deliberately
    the latest annotations row by image_id, not the detection's own
    annotation_id, since a later re-annotate replaces that row entirely via
    write_annotation()'s delete+insert (see yolo_inference.write_annotation),
    which would leave an older detection's annotation_id pointing at
    nothing. Raises HTTPException(404) if the detection, image, or a
    current annotation isn't found."""
    cur.execute(
        "SELECT s.image_id, s.image_name"
        " FROM staffline_detections s"
        " JOIN projects p ON p.id = s.project_id"
        " WHERE s.id = %s AND s.project_id = %s AND p.user_id = %s",
        (detection_id, project_id, user_id),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="staffline detection not found")
    image_id, image_name = row

    cur.execute(
        "SELECT data FROM project_images WHERE id=%s AND project_id=%s",
        (image_id, project_id),
    )
    img_row = cur.fetchone()
    if not img_row:
        raise HTTPException(status_code=404, detail=f"image {image_id} not found")
    image_arr = np.array(Image.open(io.BytesIO(bytes(img_row[0]))).convert("RGB"))

    cur.execute(
        "SELECT id, yolo_txt FROM annotations WHERE project_id=%s AND image_id=%s"
        " ORDER BY created_at DESC LIMIT 1",
        (project_id, image_id),
    )
    ann_row = cur.fetchone()
    if not ann_row:
        raise HTTPException(status_code=404, detail=f"no current annotation for image {image_id}")
    annotation_id, yolo_txt = ann_row

    return image_id, image_name, annotation_id, image_arr, yolo_txt


@router.post("/projects/{project_id}/stafflines/{detection_id}/interpolate-preview")
def preview_staffline_interpolation(
    project_id: int,
    detection_id: str,
    user=Depends(get_current_user),
):
    """Computes what turning on interpolate_missing would produce for this
    image -- without persisting anything. Lets the frontend show the
    would-be result (dashed interpolated lines, same as any other
    source="interpolated" JSOMR record) before the user chooses to accept
    it via interpolate-confirm below. See staffline_stage.preview_interpolation's
    own docstring for why this is opt-in review-first rather than always-on:
    staff-finding/dox/STATUS.md flags interpolate_missing as "not yet
    validated across the corpus".

    Plain `def`, not `async def`: this does blocking, CPU-bound work
    (image decode + component filtering/centerline fitting) directly, so
    FastAPI needs to run it in its threadpool rather than on the event
    loop, where it would stall every other concurrent request for its
    duration."""
    with db_cursor() as (con, cur):
        _image_id, image_name, _ann_id, image_arr, yolo_txt = _load_image_and_yolo_for_detection(
            cur, project_id, detection_id, user["id"],
        )
    records = staffline_stage.preview_interpolation(image_name, image_arr, yolo_txt)
    if records is None:
        raise HTTPException(status_code=422, detail="nothing to interpolate for this image")
    return {"jsomrJson": records}


@router.post("/projects/{project_id}/stafflines/{detection_id}/interpolate-confirm")
def confirm_staffline_interpolation(
    project_id: int,
    detection_id: str,
    user=Depends(get_current_user),
):
    """Re-runs real staffline detection with interpolate_missing=True and
    persists it as a NEW staffline_detections row -- matches that table's
    existing accumulate-forever design (see
    documentation_allons-y/STAFFLINE_INTEGRATION_FOLLOWUPS.md's retention
    note, which already anticipated exactly this before/after-interpolation
    comparison use case), so the pre-interpolation detection this was
    previewed from is still there too, not overwritten.

    Deliberately re-runs rather than persisting whatever interpolate-preview
    returned verbatim: detection is deterministic given the same inputs, so
    this avoids trusting/re-validating a client-supplied JSOMR payload for
    something that writes to the DB.

    Plain `def`, not `async def` -- same reason as interpolate-preview
    above. Also, unlike run_staffline_detection() (which bundles compute and
    persist under one connection its own callers already hold for an entire
    per-image loop regardless), this route deliberately does NOT hold a
    database connection during the CPU-bound detection step: component
    filtering/centerline fitting on a real manuscript page can take real
    time, and several concurrent confirmations each pinning a pooled
    connection for that whole duration could exhaust the pool and block
    unrelated requests. So: load inputs (connection released before
    returning from the `with`), compute via compute_staffline_interpolation()
    with no connection held at all, then reacquire one just for the cheap
    persist_staffline_detection() write."""
    with db_cursor() as (con, cur):
        image_id, image_name, annotation_id, image_arr, yolo_txt = _load_image_and_yolo_for_detection(
            cur, project_id, detection_id, user["id"],
        )

    computed = staffline_stage.compute_staffline_interpolation(image_name, image_arr, yolo_txt)
    if computed is None:
        raise HTTPException(status_code=422, detail="nothing to interpolate for this image")

    with db_cursor() as (con, cur):
        new_detection_id = staffline_stage.persist_staffline_detection(
            cur, con, project_id, image_id, image_name, annotation_id, computed,
        )
        # Look up by the id persist_staffline_detection just generated and
        # inserted, not "the latest row for this image" -- created_at uses
        # DEFAULT NOW() (transaction-start time, so same-transaction inserts
        # can share an identical value) and id is a random uuid4, so neither
        # is a reliable insertion-order tiebreak on its own.
        cur.execute(
            "SELECT id, image_id, image_name, stave_count, mode_lines_per_stave, status"
            " FROM staffline_detections WHERE id=%s AND project_id=%s",
            (new_detection_id, project_id),
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=500, detail="interpolation did not produce a new detection")
    did, img_id, img_name, stave_count, mode_lines_per_stave, status = row
    return {
        "id": did, "imageName": img_name,
        "imageSrc": f"/api/images/{img_id}" if img_id else None,
        "staveCount": stave_count, "modeLinesPerStave": mode_lines_per_stave,
        "status": status,
    }