import io
from pathlib import Path

from celery_app import celery_app
from job_store import publish_event, check_cancelled, JobCancelled
from auth_api import get_db_conn, release_db_conn, _log_activity
from yolo_inference import resolve_yolo_models, write_annotation
from models_api import get_model_file_path
from text_api import stream_text_finding


@celery_app.task(name="predict.run")
def run_predict_task(job_id, project_id, body):
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
        for image_id, image_name, image_data, mime_type, image_folio, has_annotation, has_text_alignment in images:
            check_cancelled(job_id)
            if has_annotation:
                publish({"type": "log", "message": f"{image_name}: already annotated — skipping YOLO"})
            else:
                publish({"type": "log", "message": f"Processing {image_name}..."})
                pil_img = Image.open(io.BytesIO(bytes(image_data))).convert("RGB")
                img_arr = np.array(pil_img)
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
            ):
                if text_ev.get("type") == "log":
                    publish(text_ev)
                elif text_ev.get("type") == "error":
                    publish({"type": "log", "message": f"text-finding: {text_ev.get('message', 'failed')}"})
        publish({"type": "stage_done", "name": "processing"})
        publish({"type": "result", "annotations": results})
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
