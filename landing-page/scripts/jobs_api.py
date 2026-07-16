import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from job_store import get_events_since, get_job_status, publish_event

router = APIRouter()

POLL_INTERVAL_SECONDS = 0.5
STALE_JOB_TIMEOUT_SECONDS = 90
TERMINAL_TYPES = {"done", "error"}

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