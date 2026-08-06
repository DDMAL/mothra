import io
from pathlib import Path

from celery_app import celery_app
from job_store import publish_event, check_cancelled, JobCancelled
from auth_api import get_db_conn, release_db_conn, _log_activity
from yolo_inference import resolve_yolo_models, write_annotation
from models_api import get_model_file_path
from text_api import stream_text_finding
from staffline_stage import run_staffline_detection, has_class, STAFFLINE_CLASS_ID


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
            # annotation and text-finding are independent steps — an image that
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
                yolo_txt = yolo_models.infer(img_arr)
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

            # Staffline detection is gated only on has_class (fresh stave-class
            # boxes to work from), never on has_text_alignment -- an image can
            # have has_text_alignment=True (text-finding already ran) and
            # has_annotation=False (YOLO just produced brand-new boxes in this
            # same iteration) at the same time, and those new boxes still need
            # a staffline_detections row. Ordered before the has_text_alignment
            # check below so its continue can never skip this block.
            if has_class(yolo_txt, STAFFLINE_CLASS_ID):
                publish({"type": "log", "message":
                    f"[trace] {image_name}: stave-class boxes came from model"
                    f" '{yolo_models.model_label}' (hash {yolo_models.model_hash or 'n/a'})"})
                for sf_ev in run_staffline_detection(
                    job_id, cur, con, project_id, image_id, image_name, ann_id, img_arr, yolo_txt,
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
