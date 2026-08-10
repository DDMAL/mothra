import io
import threading
from pathlib import Path

from celery_app import celery_app
from job_store import publish_event, check_cancelled, JobCancelled
from auth_api import get_db_conn, release_db_conn, _log_activity
from yolo_inference import resolve_yolo_models, write_annotation
from models_api import get_model_file_path
from text_api import stream_text_finding
from staffline_stage import run_staffline_detection, has_class, STAFFLINE_CLASS_ID
from paco_api import classify_stafflines, abort_classify_request, PacoClassifierError

# How often the main thread polls check_cancelled() while waiting on the
# background classifier thread (see _run_medieval_inference below).
_CANCEL_POLL_INTERVAL_S = 0.5


def _run_medieval_inference(yolo_models, img_arr, image_bytes, mime_type, image_name, publish, job_id=None):
    """Medieval-preset-only per-image inference: runs the text/music YOLO
    pass and a classifier-then-stave-YOLO pass concurrently, since the
    Paco-classifier call is the slow half of this and the two passes are
    otherwise independent until their outputs are concatenated below.

    The background thread touches NO shared DB state (no cur/con) — i
    only calls classify_stafflines (plain HTTP) and yolo_models.infer_staves
    (a pure in-process model call). write_annotation/run_staffline_detection
    stay strictly main-thread-only, after this function returns. All
    success/failure state is written into stave_result for the main thread
    to read AFTER .join() — a bare threading.Thread does not propagate
    exceptions to its caller on its own, so wrapping try/except around
    thread.join() instead of inside _stave_pipeline would silently break
    the fallback path below.

    Returns (yolo_txt, staffline_source_arr) — the merged text+music+stave
    YOLO lines, and whichever image array the stave boxes were actually
    detected against, so run_staffline_detection crops the SAME image the
    stave model saw (not always the raw page).

    If `job_id` is given, waits on the background thread by polling
    check_cancelled() every _CANCEL_POLL_INTERVAL_S instead of an
    unconditional thread.join() — a cancel request seen mid-classifier-call
    aborts the in-flight paco-classifier-service connection (via
    abort_classify_request) so this raises JobCancelled within about a
    second, instead of blocking for up to classify_stafflines's own
    DEFAULT_TIMEOUT (180s) on a job nobody cares about anymore. That same
    abort now also stops the TensorFlow inference actually running
    server-side, not just this worker thread: paco-classifier-service polls
    for exactly this disconnect and cancels cooperatively between patches
    (see paco_api.py's module docstring and recognition_engine's
    should_cancel param) — it's no longer a fire-and-forget abandoned call.
    """
    import cv2
    import numpy as np

    stave_result = {}
    conn_holder: dict = {}

    def _stave_pipeline():
        try:
            stafflines_png, _background_png = classify_stafflines(
                image_bytes, mime_type, conn_holder=conn_holder,
            )
            arr = cv2.imdecode(np.frombuffer(stafflines_png, np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                raise PacoClassifierError("could not decode stafflines PNG from paco-classifier-service")
            if arr.shape[:2] != img_arr.shape[:2]:
                raise PacoClassifierError(f"classifier output {arr.shape[:2]} != page {img_arr.shape[:2]}")
            stave_result["yolo_txt"] = yolo_models.infer_staves(arr)
            stave_result["source_arr"] = arr
        except Exception as e:
            stave_result["error"] = e

    thread = threading.Thread(target=_stave_pipeline, daemon=True)
    thread.start()
    tm_txt = yolo_models.infer_text_music(img_arr)

    if job_id is None:
        thread.join()
    else:
        # Heartbeat: the classifier call alone can legitimately run past the
        # SSE stream's 90s stale-job timeout (jobs_api.py's
        # STALE_JOB_TIMEOUT_SECONDS), which fires purely on "no new
        # job_events row for 90s" — it can't tell a slow classifier call from
        # a dead worker. Without a periodic publish() here, a page that takes
        # >90s falsely reports "job appears to have stalled" to the client
        # and flips jobs.status to 'failed', even though this thread is
        # still running fine and will finish and publish its own real
        # success events moments later. Piggybacking on the existing
        # cancel-poll loop (rather than adding a second timer) keeps this
        # cheap and resets the stream's idle counter well under the timeout.
        _HEARTBEAT_INTERVAL_S = 20.0
        elapsed_s = 0.0
        while thread.is_alive():
            try:
                check_cancelled(job_id)
            except JobCancelled:
                abort_classify_request(conn_holder)
                thread.join(timeout=5)
                raise
            thread.join(timeout=_CANCEL_POLL_INTERVAL_S)
            elapsed_s += _CANCEL_POLL_INTERVAL_S
            if elapsed_s >= _HEARTBEAT_INTERVAL_S:
                elapsed_s = 0.0
                publish({"type": "log", "message": f"{image_name}: staffline classifier still running..."})

        # The loop above only checks cancellation BETWEEN polls -- the
        # thread can finish (is_alive() goes False) in the same instant a
        # cancel request lands, right after the loop's own last
        # check_cancelled() passed. Recheck once more here, before trusting
        # stave_result/publishing anything from it, so a job cancelled in
        # that exact window doesn't still write an annotation as if it
        # completed normally.
        check_cancelled(job_id)

    if "error" in stave_result:
        publish({
            "type": "log",
            "message": f"{image_name}: staffline classifier unavailable ({stave_result['error']}) — falling back to raw-image stave detection",
        })
        st_txt = yolo_models.infer_staves(img_arr)
        source_arr = img_arr
    else:
        st_txt = stave_result["yolo_txt"]
        source_arr = stave_result["source_arr"]

    return "\n".join(filter(None, [tm_txt, st_txt])), source_arr

@celery_app.task(name="predict.run")
def run_predict_task(job_id, project_id, body):
    """Celery task backing `POST /api/projects/{id}/predict`.

    Loads the requested YOLO model set, runs layer-separation inference over
    the project's images, writes the resulting annotations, and (if
    text-finding params are present in `body`) streams text-finding via
    `stream_text_finding`. Progress/log/error events are published to
    `job_events` via `publish_event` for `GET /api/jobs/{id}/stream` to relay.
    """
    import numpy as np
    from PIL import Image

    def publish(obj):
        publish_event(job_id, obj)

    con = get_db_conn()
    cur = con.cursor()
    try:
        publish({"type": "stage", "name": "checking"})
        try:
            yolo_models, load_logs = resolve_yolo_models(
                cur, project_id, body["model_preset"], body.get("model_id"),
                body["confidence_threshold"], body["device"],
                body.get("text_music_confidence_threshold"), body.get("text_music_device"),
                body.get("stave_confidence_threshold"), body.get("stave_device"),
            )
        except (RuntimeError, ValueError) as e:
            publish({"type": "error", "message": str(e)})
            return
        for msg in load_logs:
            publish({"type": "log", "message": msg})

        seg_model_path = None
        if body.get("text_segmentation_model_id"):
            seg_row = get_model_file_path(cur, project_id, body["text_segmentation_model_id"], "segmentation")
            if seg_row:
                seg_model_path = seg_row[0]
                publish({"type": "log", "message": f"Segmentation model: {seg_row[1]}"})
            else:
                publish({"type": "log", "message": "text-finding: custom segmentation model not found — using default"})
        rec_model_path = None
        if body.get("text_recognition_model_id"):
            rec_row = get_model_file_path(cur, project_id, body["text_recognition_model_id"], "recognition")
            if rec_row:
                rec_model_path = rec_row[0]
                publish({"type": "log", "message": f"OCR model: {rec_row[1]}"})
            else:
                publish({"type": "log", "message": "text-finding: custom OCR model not found — using default"})
        mask_json_override = None
        if body.get("text_mask_model_id"):
            mask_row = get_model_file_path(cur, project_id, body["text_mask_model_id"], "text_mask")
            if mask_row:
                try:
                    mask_json_override = Path(mask_row[0]).read_text(encoding="utf-8")
                    publish({"type": "log", "message": f"text-region mask: {mask_row[1]}"})
                except Exception:
                    publish({"type": "log", "message": "text-finding: custom mask JSON could not be read — using auto-derived mask"})
            else:
                publish({"type": "log", "message": "text-finding: custom mask model not found — using auto-derived mask"})
        publish({"type": "stage_done", "name": "checking"})

        publish({"type": "stage", "name": "validating"})
        images = []
        skipped = []
        for iid in body["image_ids"]:
            cur.execute("SELECT name, data, mime_type, folio FROM project_images WHERE id=%s AND project_id=%s",
                        (iid, project_id))
            r = cur.fetchone()
            if not r:
                continue
            cur.execute("SELECT 1 FROM annotations WHERE project_id=%s AND image_id=%s", (project_id, iid))
            has_annotation = cur.fetchone() is not None
            cur.execute("SELECT 1 FROM text_alignments WHERE project_id=%s AND image_id=%s", (project_id, iid))
            has_text_alignment = cur.fetchone() is not None
            # annotation and text-finding are independent steps — an image tha
            # already has one but not the other (e.g. a job that died between the
            # two, or a race from concurrent duplicate jobs) must still run
            # whichever step is missing, not be skipped wholesale.
            if has_annotation and has_text_alignment:
                skipped.append(r[0])
                continue
            images.append((iid, r[0], r[1], r[2] or "image/png", r[3], has_annotation, has_text_alignment))
        publish({"type": "log", "message": f"{len(images)} image(s) ready"})
        if skipped:
            publish({"type": "log", "message": f"skipping {len(skipped)} already fully-processed image(s): {', '.join(skipped)}"})
        publish({"type": "stage_done", "name": "validating"})

        _log_activity(cur, project_id, "predict_run", f"{yolo_models.model_label} on {len(images)} image(s)")
        con.commit()

        publish({"type": "stage", "name": "processing"})
        results = []
        text_debug_mode = body.get("text_debug_mode", False)
        text_debug_data: dict = {}
        for image_id, image_name, image_data, mime_type, image_folio, has_annotation, has_text_alignment in images:
            check_cancelled(job_id)
            pil_img = Image.open(io.BytesIO(bytes(image_data))).convert("RGB")
            img_arr = np.array(pil_img)
            if has_annotation:
                publish({"type": "log", "message": f"{image_name}: already annotated — skipping YOLO"})
                # yolo_txt/ann_id are still needed below for the staffline-detection
                # check (has_text_alignment can be False even when has_annotation is
                # True -- see the comment in the validating stage above), so fetch
                # the annotation a prior run already wrote instead of leaving these
                # unset.
                cur.execute(
                    "SELECT id, yolo_txt FROM annotations WHERE project_id=%s AND image_id=%s"
                    " ORDER BY created_at DESC LIMIT 1",
                    (project_id, image_id),
                )
                row = cur.fetchone()
                if row is not None:
                    ann_id, yolo_txt = row
                else:
                    # Annotation was deleted between the validating stage and here
                    # (e.g. a concurrent duplicate job) -- fall through to a fresh
                    # YOLO run instead of crashing on an empty unpack.
                    has_annotation = False
                    publish({"type": "log", "message": f"{image_name}: annotation disappeared since validation — re-running YOLO"})
            if not has_annotation:
                publish({"type": "log", "message": f"Processing {image_name}..."})
                if yolo_models.medieval_models is not None:
                    yolo_txt, staffline_source_arr = _run_medieval_inference(
                        yolo_models, img_arr, bytes(image_data), mime_type, image_name, publish,
                        job_id=job_id,
                    )
                else:
                    yolo_txt = yolo_models.infer(img_arr)
                    staffline_source_arr = img_arr
                ann_id = write_annotation(
                    cur, con, project_id, image_id, image_name, yolo_txt,
                    yolo_models.stored_model_id, yolo_models.model_label, yolo_models.model_hash,
                )
                n_detections = len(yolo_txt.splitlines()) if yolo_txt else 0
                publish({"type": "log", "message": f"{image_name}: {n_detections} detection(s)"})
                results.append({
                    "id": ann_id, "imageName": image_name,
                    "imageSrc": f"/api/images/{image_id}",
                    "txtName": f"annotation-{ann_id}.txt",
                    "jsonName": "", "detectionCount": n_detections,
                })
            else:
                staffline_source_arr = img_arr  # reused annotation -- no fresh classifier run to redo

            # Staffline detection is gated only on has_class (fresh stave-class
            # boxes to work from), never on has_text_alignment -- an image can
            # have has_text_alignment=True (text-finding already ran) and
            # has_annotation=False (YOLO just produced brand-new boxes in this
            # same iteration) at the same time, and those new boxes still need
            # a staffline_detections row. Ordered before the has_text_alignmen
            # check below so its continue can never skip this block.
            if has_class(yolo_txt, STAFFLINE_CLASS_ID):
                publish({"type": "log", "message":
                    f"[trace] {image_name}: stave-class boxes came from model"
                    f" '{yolo_models.model_label}' (hash {yolo_models.model_hash or 'n/a'})"})
                redetect_fn = (
                    yolo_models.infer_staves_raw_boxes
                    if yolo_models.medieval_models is not None else None
                )
                for sf_ev in run_staffline_detection(
                    job_id, cur, con, project_id, image_id, image_name, ann_id, staffline_source_arr, yolo_txt,
                    redetect_fn=redetect_fn,
                ):
                    if sf_ev.get("type") == "error":
                        publish({"type": "log", "message": f"staffline-detection: {sf_ev.get('message', 'failed')}"})
                    else:
                        publish(sf_ev)

            if has_text_alignment:
                publish({"type": "log", "message": f"{image_name}: text already found — skipping text-finding"})
                continue

            publish({"type": "log", "message": f"{image_name}: starting text-finding..."})
            for text_ev in stream_text_finding(
                project_id, image_id, image_name,
                bytes(image_data), mime_type,
                column_count=body.get("text_column_count"),
                segmentation_model=seg_model_path,
                recognition_model=rec_model_path,
                device=body.get("text_device", "cpu"),
                column_bimodal_threshold=body.get("text_column_bimodal_threshold", 0.5),
                masking_enabled=body.get("text_masking_enabled", True),
                mask_padding=body.get("text_mask_padding", 15),
                music_overlap_filter_enabled=body.get("text_music_overlap_filter_enabled", True),
                mask_json_override=mask_json_override,
                source_id=body.get("text_source_id") if image_folio else None,
                folio_override=image_folio,
                debug_mode=text_debug_mode,
            ):
                if text_ev.get("type") == "log":
                    publish(text_ev)
                elif text_ev.get("type") == "error":
                    publish({"type": "log", "message": f"text-finding: {text_ev.get('message', 'failed')}"})
                elif text_ev.get("type") == "result" and text_ev.get("debug_data"):
                    text_debug_data[image_name] = text_ev["debug_data"]
        publish({"type": "stage_done", "name": "processing"})
        result_event: dict = {"type": "result", "annotations": results}
        if text_debug_data:
            result_event["text_debug_data"] = text_debug_data
        publish(result_event)
        publish({"type": "done"})
    except JobCancelled:
        con.rollback()
        return
    except Exception as e:
        con.rollback()
        publish({"type": "error", "message": str(e)})
    finally:
        cur.close()
        release_db_conn(con)
