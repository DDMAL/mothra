"""Celery task for /text-batch/run — reproduces the pipeline that used to
live inline in batch_api.py's generate() SSE closure, now off the request
thread. image_ids/folios have already been deduped against existing
text_alignments rows by the kickoff endpoint before this task is enqueued."""
import io
import json
import urllib.error
import uuid as _uuid
from pathlib import Path

import numpy as np
from PIL import Image

from celery_app import celery_app
from job_store import publish_event, check_cancelled, JobCancelled
from auth_api import get_db_conn, release_db_conn, _log_activity
from models_api import get_model_file_path
from text_api import TEXT_API_URL, _stream_multipart, _music_boxes_for_image, _mask_json_for_image
from yolo_inference import resolve_yolo_models, write_annotation
from staffline_stage import run_staffline_detection, has_class, STAFFLINE_CLASS_ID

@celery_app.task(name="text_batch.run")
def run_text_batch_task(job_id, project_id, body):
    """Celery task backing `POST /api/projects/{id}/text-batch/run`.

    Runs YOLO layer-separation over each image (for music-box/mask context),
    then forwards the batch to the text-service's `/batch-run` endpoint and
    relays its SSE progress as `job_events` rows via `publish_event`.
    """
    pending_logs: list[str] = []

    def publish(obj):
        publish_event(job_id, obj)
        if obj.get("type") == "log":
            pending_logs.append(obj.get("message", ""))

    con = get_db_conn()
    cur = con.cursor()
    try:
        image_ids = body["image_ids"]
        folios = body["folios"]
        skipped_count = body.get("skipped_count", 0)
        if skipped_count:
            publish({"type": "log", "message": f"skipping {skipped_count} image(s) that already have text alignments"})
        if not image_ids:
            publish({"type": "log", "message": "nothing to do — all selected images already have text alignments"})
            publish({"type": "done"})
            return

        images = []
        for iid in image_ids:
            # NOTE (SF-2, ALPHA_TRANSITION_PLAN.md): tasks_predict.py's own
            # image fetch now prefers original_data over this resized
            # working copy, since staffline_stage.py's JSOMR is resolution-
            # sensitive. That fix was NOT backported here on purpose -- this
            # is a different job type, wasn't part of the measured #213
            # regression, and expanding the behavior change here multiplies
            # verification surface without a measured justification. If a
            # #213-class regression is ever reported for the text-batch
            # path specifically, this is the place to revisit.
            cur.execute("SELECT name, data, mime_type FROM project_images WHERE id=%s AND project_id=%s", (iid, project_id))
            row = cur.fetchone()
            if not row:
                raise RuntimeError(f"image {iid} not found")
            images.append((row[0], bytes(row[1]), row[2] or "image/png"))

        publish({"type": "stage", "name": "checking"})
        try:
            yolo_models, load_logs = resolve_yolo_models(
                cur, project_id, body["model_preset"], body.get("model_id"),
                body["yolo_confidence_threshold"], body["yolo_device"],
                body.get("text_music_confidence_threshold"), body.get("text_music_device"),
                body.get("stave_confidence_threshold"), body.get("stave_device"),
            )
        except (RuntimeError, ValueError) as e:
            publish({"type": "error", "message": str(e)})
            return
        for msg in load_logs:
            publish({"type": "log", "message": msg})

        mask_json_override = None
        if body.get("mask_model_id"):
            mask_row = get_model_file_path(cur, project_id, body["mask_model_id"], "text_mask")
            if mask_row:
                try:
                    mask_json_override = Path(mask_row[0]).read_text(encoding="utf-8")
                    publish({"type": "log", "message": f"text-region mask: {mask_row[1]}"})
                except Exception:
                    publish({"type": "log", "message": "text-finding: custom mask JSON could not be read — using auto-derived mask"})
            else:
                publish({"type": "log", "message": "text-finding: custom mask model not found — using auto-derived mask"})

        seg_model_path = None
        if body.get("segmentation_model"):
            seg_row = get_model_file_path(cur, project_id, body["segmentation_model"], "segmentation")
            if seg_row:
                seg_model_path = seg_row[0]
                publish({"type": "log", "message": f"Segmentation model: {seg_row[1]}"})
            else:
                publish({"type": "log", "message": "text-finding: custom segmentation model not found — using default"})

        rec_model_path = None
        if body.get("recognition_model"):
            rec_row = get_model_file_path(cur, project_id, body["recognition_model"], "recognition")
            if rec_row:
                rec_model_path = rec_row[0]
                publish({"type": "log", "message": f"OCR model: {rec_row[1]}"})
            else:
                publish({"type": "log", "message": "text-finding: custom OCR model not found — using default"})

        music_boxes_by_index, mask_json_by_index = [], []
        for image_id, (name, data, mime) in zip(image_ids, images):
            check_cancelled(job_id)
            pil_img = Image.open(io.BytesIO(data)).convert("RGB")
            img_arr = np.array(pil_img)
            yolo_txt = yolo_models.infer(img_arr)
            ann_id = write_annotation(cur, con, project_id, image_id, name, yolo_txt,
                             yolo_models.stored_model_id, yolo_models.model_label, yolo_models.model_hash)
            music_boxes_by_index.append(_music_boxes_for_image(project_id, name, data))
            if mask_json_override is not None:
                mask_json_by_index.append(mask_json_override)
            else:
                mask_json_by_index.append(_mask_json_for_image(project_id, name, data) if body["masking_enabled"] else None)
            n = len(yolo_txt.splitlines()) if yolo_txt else 0
            publish({"type": "log", "message": f"{name}: layer separation done ({n} detection(s))"})

            # Mirrors tasks_predict.py's own predict-job pipeline: this path
            # also runs fresh YOLO layer-separation per image and previously
            # never called staffline_stage.py, so batch-run images got no
            # staffline_detections row at all (see
            # documentation_allons-y/STAFFLINE_INTEGRATION_FOLLOWUPS.md's
            # "batch_api.py's text-batch-run path" bullet).
            if has_class(yolo_txt, STAFFLINE_CLASS_ID):
                publish({"type": "log", "message":
                    f"[trace] {name}: stave-class boxes came from model"
                    f" '{yolo_models.model_label}' (hash {yolo_models.model_hash or 'n/a'})"})
                for sf_ev in run_staffline_detection(
                    job_id, cur, con, project_id, image_id, name, ann_id, img_arr, yolo_txt,
                    # SF-6: this path never has a classifier choice -- img_arr
                    # is always the raw page (see the PIL decode above).
                    # Explicit rather than relying on the default so this
                    # stays correct if the default ever changes.
                    source_label="raw_page",
                    # CodeRabbit PR #219: this path's own image fetch above
                    # (SELECT name, data, mime_type) never reads
                    # original_data -- always the resized working copy.
                    storage_variant="working_copy",
                ):
                    if sf_ev.get("type") == "error":
                        publish({"type": "log", "message": f"staffline-detection: {sf_ev.get('message', 'failed')}"})
                    else:
                        publish(sf_ev)
        publish({"type": "stage_done", "name": "checking"})

        publish({"type": "stage", "name": "validating"})
        publish({"type": "log", "message": f"running Kraken segmentation + HTR across {len(folios)} folio(s) (Cantus-aligned mode, source {body['source_id']})..."})
        if body.get("column_count"):
            publish({"type": "log", "message": f"column count forced to {body['column_count']}"})
        publish({"type": "stage_done", "name": "validating"})

        _log_activity(cur, project_id, "text_batch_run",
                       f"{yolo_models.model_label} on {len(folios)} folio(s) (source {body['source_id']})")
        con.commit()

        publish({"type": "stage", "name": "processing"})
        text_debug_data: dict = {}
        fields = {
            "folios": json.dumps(folios), "source_id": str(body["source_id"]), "device": body["device"],
            "column_bimodal_threshold": str(body["column_bimodal_threshold"]),
            "masking_enabled": "true" if body["masking_enabled"] else "false",
            "mask_padding": str(body["mask_padding"]),
            "music_overlap_filter_enabled": "true" if body.get("music_overlap_filter_enabled", True) else "false",
            "debug_mode": "true" if body.get("debug_mode", False) else "false",
            "music_boxes": json.dumps(music_boxes_by_index),
            "mask_json_list": json.dumps(mask_json_by_index),
        }
        if seg_model_path:
            fields["segmentation_model"] = seg_model_path
        if rec_model_path:
            fields["recognition_model"] = rec_model_path
        if body.get("column_count") is not None:
            fields["column_count"] = str(body["column_count"])

        # mothra#236: first item's start -- there's no earlier per-item signal
        # from the text-service to key off of, so this one fires
        # unconditionally before the stream starts (images is always
        # non-empty here; an empty image_ids list already returned early
        # above).
        publish({"type": "item_start", "item": 0, "total": len(images), "name": images[0][0]})
        for line in _stream_multipart(f"{TEXT_API_URL}/batch-run", fields=fields,
                                      files=[("images", n, m, d) for n, d, m in images], timeout=1800):
            check_cancelled(job_id)
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[len("data: "):])
            if ev.get("type") == "folio_result":
                idx = ev["image_index"]
                image_id = image_ids[idx]
                image_name = images[idx][0]
                alignment = ev["text_alignment"]
                aid = _uuid.uuid4().hex
                syl_count = len(alignment.get("syl_boxes", []))
                publish({"type": "log", "message": f"{image_name}: {syl_count} syllable(s) aligned"})
                log_text = "\n".join(pending_logs)
                pending_logs = []
                cur.execute(
                    "INSERT INTO text_alignments"
                    " (id, project_id, image_id, image_name, alignment_json,"
                    " median_line_spacing, syllable_count, log_text)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    (aid, project_id, image_id, image_name, json.dumps(alignment),
                     alignment.get("median_line_spacing", 0.0), syl_count, log_text),
                )
                con.commit()
                if ev.get("debug_data"):
                    text_debug_data[image_name] = ev["debug_data"]
                # mothra#236: routes this batch's per-item timing into the
                # same avgItemMsRef/ETA mechanism tasks_encode.py's
                # run_encode_batch_task already drives (ProcessingPage.tsx's
                # stage_done handler records the sample once it sees
                # "processing" complete for an item). Assumes folio_result
                # events arrive in increasing image_index order (true today
                # -- the text-service processes the submitted list
                # sequentially); if that ever changes, the bar/ETA just
                # degrade to "roughly right" rather than crashing.
                publish({"type": "item_done", "item": idx})
                publish({"type": "stage_done", "name": "processing"})
                if idx + 1 < len(images):
                    publish({"type": "item_start", "item": idx + 1, "total": len(images), "name": images[idx + 1][0]})
                continue
            if ev.get("type") == "result":
                if text_debug_data:
                    ev = {**ev, "text_debug_data": text_debug_data}
            publish(ev)  # forwards "result" {batchId, fileCount} and "done" unchanged
    except JobCancelled:
        con.rollback()
        return
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="ignore")
        try:
            detail = json.loads(detail).get("detail", detail)
        except json.JSONDecodeError:
            pass
        publish({"type": "error", "message": f"text-service rejected the request (HTTP {exc.code}): {detail}"})
    except urllib.error.URLError as exc:
        publish({"type": "error", "message": f"text-service at {TEXT_API_URL} is unreachable: {exc}"})
    except Exception as e:
        con.rollback()
        publish({"type": "error", "message": str(e)})
    finally:
        cur.close()
        release_db_conn(con)