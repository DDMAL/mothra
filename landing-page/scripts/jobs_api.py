import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from celery_app import celery_app
from auth_api import get_current_user, require_project_owner, db_cursor
from job_store import get_events_since, get_job_status, publish_event, new_job_id, create_job
from tasks_predict import run_predict_task
from tasks_encode import run_encode_upload_task, run_encode_batch_task
from tasks_text_batch import run_text_batch_task

router = APIRouter()

POLL_INTERVAL_SECONDS = 0.5
STALE_JOB_TIMEOUT_SECONDS = 90
TERMINAL_TYPES = {"done", "error", "cancelled"}
TASK_BY_KIND = {
    "predict": run_predict_task,
    "encode_upload": run_encode_upload_task,
    "encode_batch": run_encode_batch_task,
    "text_batch": run_text_batch_task,
}

@router.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    async def generate():
        last_id = 0
        idle_seconds = 0.0
        while True:
            rows = await asyncio.to_thread(get_events_since, job_id, last_id)
            if rows:
                idle_seconds = 0.0
                for row_id, payload in rows:
                    last_id = row_id
                    yield f"data: {payload}\n\n"
                    if json.loads(payload).get("type") in TERMINAL_TYPES:
                        return
            else:
                status = await asyncio.to_thread(get_job_status, job_id)
                if status is None:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'job not found'})}\n\n"
                    return
                idle_seconds += POLL_INTERVAL_SECONDS
                if status == "running" and idle_seconds > STALE_JOB_TIMEOUT_SECONDS:
                    # Persist the failure via the same path a task would use
                    # (publish_event flips jobs.status to 'failed') — otherwise
                    # the DB stays stuck on 'running' forever after a worker
                    # crash, and a reconnecting client would silently wait out
                    # another full staleness timeout to rediscover the same
                    # conclusion instead of seeing it immediately.
                    error_event = {"type": "error", "message": "job appears to have stalled (worker may have crashed)"}
                    await asyncio.to_thread(publish_event, job_id, error_event)
                    yield f"data: {json.dumps(error_event)}\n\n"
                    return
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@router.get("/jobs/{job_id}")
async def get_job(job_id: str, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        cur.execute("SELECT project_id, status FROM jobs WHERE job_id=%s", (job_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="job not found")
        project_id, status = row
        if project_id is not None:
            require_project_owner(cur, project_id, user["id"])
    return {"job_id": job_id, "status": status}

@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        cur.execute("SELECT project_id, status FROM jobs WHERE job_id=%s", (job_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="job not found")
        project_id, status = row
        if project_id is not None:
            require_project_owner(cur, project_id, user["id"])
        if status in ("succeeded", "failed", "cancelled"):
            return {"ok": True, "status": status}

    await asyncio.to_thread(celery_app.control.revoke, job_id, terminate=True, signal="SIGTERM")
    await asyncio.to_thread(publish_event, job_id, {"type": "cancelled", "message": "cancelled by user"})
    return {"ok": True}

@router.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        cur.execute("SELECT kind, project_id, status, params, attempt FROM jobs WHERE job_id=%s", (job_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="job not found")
        kind, project_id, status, params, attempt = row
        if project_id is not None:
            require_project_owner(cur, project_id, user["id"])
    if status != "failed":
        raise HTTPException(status_code=400, detail="only failed jobs can be retried")
    if params is None:
        raise HTTPException(status_code=409, detail="this job predates retry support and has no stored parameters")
    task = TASK_BY_KIND.get(kind)
    if task is None:
        raise HTTPException(status_code=400, detail=f"unknown job kind '{kind}'")
    
    new_id = new_job_id()
    create_job(new_id, kind, project_id, params=params, retry_of=job_id, attempt=(attempt or 1) + 1)
    task.apply_async(kwargs={**params, "job_id": new_id}, task_id=new_id)
    return {"job_id": new_id}