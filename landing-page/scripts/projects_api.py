"""Project CRUD, export/duplicate, and activity/log-download endpoints.

`_build_project_dict` is the single source of truth for the project-to-dict
shape shared by `get_project` (single-project query) and `list_projects`
(batched query) - previously these reimplemented the same shaping twice.
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
from datetime import datetime
import io
import json
import shutil
import uuid as _uuid
import zipfile

from auth_api import get_current_user, db_cursor, require_project_owner, _log_activity, MODELS_DIR
from job_store import get_active_job_for_project

router = APIRouter()


def _build_project_dict(pid, name, username, steps, used_json, used_model_json,
                         deleted_at, last_opened_at, is_pinned, used_annotation_json,
                         images, models, mei, annotations, text_alignments, cantus_source_id,
                         stafflines, ic_xml_files, is_tutorial=False):
    return {
        "id": pid, "name": name, "user": username,
        "stepsUnlocked": steps,
        # CodeRabbit (#241 follow-up): renamed from usedImageNames -- the
        # column still stores a JSON string array, but its elements are now
        # project_images.id values, not names, so duplicate-named uploads
        # can be selected/removed/run independently instead of every
        # same-named row being treated as one unit.
        "usedImageIds": json.loads(used_json or "[]"),
        "usedModelNames": json.loads(used_model_json or "[]"),
        "images": images, "models": models, "meiFiles": mei,
        "annotations": annotations, "deletedAt": deleted_at,
        "lastOpenedAt": str(last_opened_at) if last_opened_at else None,
        "isPinned": bool(is_pinned),
        "usedAnnotationNames": json.loads(used_annotation_json or "[]"),
        "textAlignments": text_alignments,
        "cantusSourceId": cantus_source_id,
        "stafflines": stafflines,
        # Metadata only -- an exported GameraXML page runs to megabytes
        # (one RLE mask per glyph), so the body is fetched per file from
        # ic_api.py's /ic-xml/{id} endpoints rather than shipped with
        # every project list request the way meiFiles' xmlContent is.
        "icXmlFiles": ic_xml_files,
        "isTutorial": bool(is_tutorial),
    }


def _map_annotation_row(aid, img_id, img_name, model_label=None):
    return {
        "id": aid, "imageId": img_id, "imageName": img_name,
        "imageSrc": f"/api/images/{img_id}" if img_id else None,
        "txtName": f"annotation-{aid}.txt", "jsonName": "",
        "modelLabel": model_label,
    }


def _map_ic_xml_row(xid, img_id, img_name, name, glyph_count, byte_size, created_at):
    return {
        "id": xid, "imageId": img_id, "imageName": img_name, "name": name,
        "imageSrc": f"/api/images/{img_id}" if img_id else None,
        "glyphCount": glyph_count, "byteSize": byte_size,
        "createdAt": str(created_at) if created_at else None,
    }


def _image_src(img_id, storage_variant):
    """Build the imageSrc URL for a row carrying absolute-pixel coordinates
    (syl_boxes, JSOMR bounding_box/centerline_page). These are computed
    against whichever project_images column storage_variant names -- serving
    the other one back to the frontend leaves its scaleX/scaleY (naturalWidth/
    naturalHeight of whatever imageSrc loads, divided into the displayed
    size) computed against the wrong resolution, drifting every box further
    off the further it sits from the top-left corner (mothra#260). See
    text_api.py's stream_text_finding / staffline_stage.py's
    run_staffline_detection docstrings for how storage_variant gets set."""
    if not img_id:
        return None
    return f"/api/images/{img_id}/original" if storage_variant == "original" else f"/api/images/{img_id}"


def _map_text_alignment_row(tid, img_id, img_name, spacing, syl_count, storage_variant="working_copy"):
    return {
        "id": tid, "imageName": img_name,
        "imageSrc": _image_src(img_id, storage_variant),
        "medianLineSpacing": spacing, "syllableCount": syl_count,
    }


def _map_staffline_row(did, img_id, img_name, stave_count, mode_lines_per_stave, status, has_classifier_image=False,
                        has_classifier_fallback=False, classifier_error=None, storage_variant="working_copy",
                        has_background_image=False):
    return {
        "id": did, "imageName": img_name,
        "imageSrc": _image_src(img_id, storage_variant),
        "staveCount": stave_count, "modeLinesPerStave": mode_lines_per_stave,
        "status": status,
        "hasClassifierImage": bool(has_classifier_image),
        # hasClassifierFallback (derived from settings_json.source_label, not
        # from classifier_error) so pre-existing fallback rows still surface
        # the flag even though they predate classifier_error and have no
        # stored reason -- see staffline_stage.py's run_staffline_detection
        # docstring.
        "hasClassifierFallback": bool(has_classifier_fallback),
        "classifierError": classifier_error,
        # mothra#286: paco-classifier's OTHER output layer (background-only),
        # alongside the stafflines-only one hasClassifierImage exposes --
        # see inference_api.py's GET .../background-image.
        "hasBackgroundImage": bool(has_background_image),
    }

def _project_row_to_dict(cur, row, username):
    """Build one project's full API dict (images/models/MEI/annotations/text
    alignments) from its `projects` row, issuing the per-project child-table
    queries needed to assemble it. Images are ordered by `created_at, id` so
    upload order is stable across requests rather than following whatever
    order Postgres happens to return rows in.
    """
    pid, name, steps, used_json, used_model_json, deleted_at, last_opened_at, is_pinned, used_annotation_json, cantus_source_id, is_tutorial = row
    cur.execute(
        "SELECT id, name, folio, source_id, source_name FROM project_images"
        " WHERE project_id=%s ORDER BY created_at ASC, id ASC", (pid,)
    )
    images = [{"id": r[0], "name": r[1], "folio": r[2], "sourceId": r[3], "sourceName": r[4]} for r in cur.fetchall()]
    cur.execute("SELECT id, name, COALESCE(kind, 'yolo') FROM project_models WHERE project_id=%s", (pid,))
    models = [{"id": r[0], "name": r[1], "kind": r[2]} for r in cur.fetchall()]
    # created_at ASC, and carried through as createdAt: mei_files is
    # append-only, so a page encoded more than once has several rows and the
    # frontend has to be able to tell which is current (utils/mei.ts's
    # latestMeiPerImage -- what the Neon editor lists).
    cur.execute("SELECT id, name, xml_content, corrected, image_name, stave_source, image_id, created_at"
                " FROM mei_files WHERE project_id=%s ORDER BY created_at ASC", (pid,))
    mei = [{"id": r[0], "name": r[1], "xmlContent": r[2], "corrected": bool(r[3]), "imageName": r[4],
            "staveSource": r[5], "imageId": r[6], "createdAt": str(r[7]) if r[7] else None}
           for r in cur.fetchall()]
    cur.execute("SELECT id, image_id, image_name, model_label FROM annotations WHERE project_id=%s", (pid,))
    annotations = [_map_annotation_row(r[0], r[1], r[2], r[3]) for r in cur.fetchall()]
    cur.execute(
        "SELECT id, image_id, image_name, median_line_spacing, syllable_count, storage_variant"
        " FROM text_alignments WHERE project_id=%s ORDER BY created_at ASC", (pid,)
    )
    text_alignments = [_map_text_alignment_row(r[0], r[1], r[2], r[3], r[4], r[5]) for r in cur.fetchall()]
    cur.execute(
        "SELECT id, image_id, image_name, stave_count, mode_lines_per_stave, status,"
        " classifier_image IS NOT NULL,"
        " settings_json->>'source_label' = 'raw_page_fallback', settings_json->>'classifier_error',"
        " settings_json->>'storage_variant',"
        " background_image IS NOT NULL"
        " FROM staffline_detections WHERE project_id=%s ORDER BY created_at ASC", (pid,)
    )
    stafflines = [_map_staffline_row(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10]) for r in cur.fetchall()]
    cur.execute(
        "SELECT id, image_id, image_name, name, glyph_count, octet_length(xml_content), created_at"
        " FROM ic_xml_files WHERE project_id=%s ORDER BY created_at ASC", (pid,)
    )
    ic_xml_files = [_map_ic_xml_row(*r) for r in cur.fetchall()]
    return _build_project_dict(
        pid, name, username, steps, used_json, used_model_json, deleted_at,
        last_opened_at, is_pinned, used_annotation_json,
        images, models, mei, annotations, text_alignments, cantus_source_id,
        stafflines, ic_xml_files, is_tutorial=is_tutorial,
    )


@router.get("/projects")
def list_projects(user=Depends(get_current_user)):
    """List all of the current user's projects with their child rows.

    Batches the images/models/MEI/annotations/text-alignments lookups across
    all of the user's project ids (one query per child table instead of one
    per project) to avoid N+1 queries, then reassembles each project's dict
    via `_build_project_dict`. Images are ordered by `created_at, id` for the
    same reason as `_project_row_to_dict`.
    """
    with db_cursor() as (con, cur):
        cur.execute(
            "SELECT id, name, steps_unlocked, used_image_names, used_model_names, deleted_at, "
            " last_opened_at, is_pinned, used_annotation_names, cantus_source_id, is_tutorial"
            " FROM projects WHERE user_id=%s",
            (user["id"],)
        )
        rows = cur.fetchall()
        if not rows:
            return []

        pids = tuple(r[0] for r in rows)

        cur.execute(
            "SELECT project_id, id, name, folio, source_id, source_name FROM project_images"
            " WHERE project_id IN %s ORDER BY created_at ASC, id ASC",
            (pids,),
        )
        images_by_pid: dict = {}
        for pid, iid, iname, ifolio, isourceid, isourcename in cur.fetchall():
            images_by_pid.setdefault(pid, []).append(
                {"id": iid, "name": iname, "folio": ifolio, "sourceId": isourceid, "sourceName": isourcename}
            )

        cur.execute("SELECT project_id, id, name, COALESCE(kind, 'yolo') FROM project_models WHERE project_id IN %s", (pids,))
        models_by_pid: dict = {}
        for pid, mid, mname, mkind in cur.fetchall():
            models_by_pid.setdefault(pid, []).append({"id": mid, "name": mname, "kind": mkind})

        cur.execute(
            "SELECT project_id, id, name, xml_content, corrected, image_name, stave_source, image_id, created_at"
            " FROM mei_files WHERE project_id IN %s ORDER BY created_at ASC", (pids,)
        )
        mei_by_pid: dict = {}
        for pid, fid, fname, xml, corr, iname, stave_source, iid, created_at in cur.fetchall():
            mei_by_pid.setdefault(pid, []).append(
                {"id": fid, "name": fname, "xmlContent": xml, "corrected": bool(corr), "imageName": iname,
                 "staveSource": stave_source, "imageId": iid,
                 "createdAt": str(created_at) if created_at else None}
            )

        cur.execute(
            "SELECT project_id, id, image_id, image_name, model_label FROM annotations WHERE project_id IN %s",
            (pids,)
        )
        ann_by_pid: dict = {}
        for pid, aid, img_id, img_name, model_label in cur.fetchall():
            ann_by_pid.setdefault(pid, []).append(_map_annotation_row(aid, img_id, img_name, model_label))

        cur.execute(
            "SELECT project_id, id, image_id, image_name, median_line_spacing, syllable_count, storage_variant"
            " FROM text_alignments WHERE project_id IN %s ORDER BY created_at ASC", (pids,)
        )
        text_by_pid: dict = {}
        for pid, tid, img_id, img_name, spacing, syl_count, storage_variant in cur.fetchall():
            text_by_pid.setdefault(pid, []).append(
                _map_text_alignment_row(tid, img_id, img_name, spacing, syl_count, storage_variant)
            )

        cur.execute(
            "SELECT project_id, id, image_id, image_name, stave_count, mode_lines_per_stave, status,"
            " classifier_image IS NOT NULL,"
            " settings_json->>'source_label' = 'raw_page_fallback', settings_json->>'classifier_error',"
            " settings_json->>'storage_variant',"
            " background_image IS NOT NULL"
            " FROM staffline_detections WHERE project_id IN %s ORDER BY created_at ASC", (pids,)
        )
        stafflines_by_pid: dict = {}
        for (pid, did, img_id, img_name, stave_count, mode_lines_per_stave, status, has_classifier_image,
             has_classifier_fallback, classifier_error, storage_variant, has_background_image) in cur.fetchall():
            stafflines_by_pid.setdefault(pid, []).append(
                _map_staffline_row(did, img_id, img_name, stave_count, mode_lines_per_stave, status,
                                    has_classifier_image, has_classifier_fallback, classifier_error, storage_variant,
                                    has_background_image)
            )

        cur.execute(
            "SELECT project_id, id, image_id, image_name, name, glyph_count,"
            " octet_length(xml_content), created_at"
            " FROM ic_xml_files WHERE project_id IN %s ORDER BY created_at ASC", (pids,)
        )
        ic_xml_by_pid: dict = {}
        for row_ in cur.fetchall():
            ic_xml_by_pid.setdefault(row_[0], []).append(_map_ic_xml_row(*row_[1:]))

        result = [
            _build_project_dict(
                row[0], row[1], user["username"], row[2], row[3], row[4], row[5], row[6], row[7], row[8],
                images=images_by_pid.get(row[0], []),
                models=models_by_pid.get(row[0], []),
                mei=mei_by_pid.get(row[0], []),
                annotations=ann_by_pid.get(row[0], []),
                text_alignments=text_by_pid.get(row[0], []),
                cantus_source_id=row[9],
                stafflines=stafflines_by_pid.get(row[0], []),
                ic_xml_files=ic_xml_by_pid.get(row[0], []),
                is_tutorial=row[10],
            )
            for row in rows
        ]
    return result


@router.get("/projects/{project_id}")
def get_project(project_id: int, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        cur.execute(
            "SELECT id, name, steps_unlocked, used_image_names, used_model_names, deleted_at,"
            " last_opened_at, is_pinned, used_annotation_names, cantus_source_id, is_tutorial"
            " FROM projects WHERE id=%s AND user_id=%s",
            (project_id, user["id"])
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404)
        result = _project_row_to_dict(cur, row, user["username"])
    return result


@router.get("/projects/{project_id}/activity")
def get_activity(project_id: int, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        cur.execute(
            "SELECT action_type, detail, created_at FROM activity_log"
            " WHERE project_id=%s ORDER BY created_at DESC LIMIT 100",
            (project_id,)
        )
        return [{"actionType": r[0], "detail": r[1], "createdAt": str(r[2])} for r in cur.fetchall()]

@router.get("/projects/{project_id}/active-job")
def get_project_active_job(project_id: int, user=Depends(get_current_user)):
    """Exposes job_store.get_active_job_for_project over HTTP so the project
    page can discover a still-running predict/encode/text-batch job after a
    reload or from a different tab than the one that kicked it off --
    activeJobs.ts's in-memory registry only covers the same-tab session that
    actually called the kickoff endpoint. Returns null when nothing is
    currently pending/running for this project."""
    with db_cursor() as (_con, cur):
        require_project_owner(cur, project_id, user["id"])
    active = get_active_job_for_project(project_id)
    if active is None:
        return None
    return {
        "job_id": active["job_id"],
        "kind": active["kind"],
        "status": active["status"],
        "created_at": active["created_at"].isoformat() if active["created_at"] else None,
    }

class CreateProjectBody(BaseModel):
    name: str

@router.post("/projects")
def create_project(body: CreateProjectBody, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        cur.execute(
            "INSERT INTO projects (user_id, name) VALUES (%s,%s) RETURNING id",
            (user["id"], body.name))
        pid = cur.fetchone()[0]
        con.commit()
    # Every list-shaped field the project payload carries is spelled out,
    # even though a fresh project has none of them: the frontend puts this
    # object straight into its projects state (useProjectMutations'
    # createProject) rather than re-fetching, so an omitted key lands as
    # `undefined` on a Project and the tab reading it crashes on `.length`.
    return {"id": pid, "name": body.name, "user": user["username"],
            "images": [], "models": [], "meiFiles": [], "annotations": [],
            "stepsUnlocked": 0, "usedImageIds": [], "usedModelNames": [],
            "deletedAt": None, "usedAnnotationNames": [], "cantusSourceId": None,
            "textAlignments": [], "stafflines": [], "icXmlFiles": [], "isTutorial": False}


class UpdateProjectBody(BaseModel):
    name: Optional[str] = None
    stepsUnlocked: Optional[int] = None
    usedImageIds: Optional[list] = None
    usedModelNames: Optional[list] = None
    deletedAt: Optional[str] = None
    lastOpenedAt: Optional[str] = None
    isPinned: Optional[bool] = None
    usedAnnotationNames: Optional[list] = None
    cantusSourceId: Optional[str] = None

@router.put("/projects/{project_id}")
def update_project(project_id: int, body: UpdateProjectBody, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        if body.name is not None:
            cur.execute("UPDATE projects SET name=%s WHERE id=%s", (body.name, project_id))
        if body.stepsUnlocked is not None:
            cur.execute("UPDATE projects SET steps_unlocked=%s WHERE id=%s", (body.stepsUnlocked, project_id))
            _log_activity(cur, project_id, "step_unlocked", str(body.stepsUnlocked))
        if body.usedImageIds is not None:
            cur.execute("UPDATE projects SET used_image_names=%s WHERE id=%s",
                        (json.dumps(body.usedImageIds), project_id))
        if body.usedModelNames is not None:
            cur.execute("UPDATE projects SET used_model_names=%s WHERE id=%s",
                        (json.dumps(body.usedModelNames), project_id))
        if body.deletedAt is not None:
            cur.execute("UPDATE projects SET deleted_at=%s WHERE id=%s", (body.deletedAt, project_id))
        if body.lastOpenedAt is not None:
            cur.execute("UPDATE projects SET last_opened_at=%s WHERE id=%s",
                        (body.lastOpenedAt, project_id))
        if body.isPinned is not None:
            cur.execute("UPDATE projects SET is_pinned=%s WHERE id=%s", (body.isPinned, project_id))
        if body.usedAnnotationNames is not None:
            cur.execute("UPDATE projects SET used_annotation_names=%s WHERE id=%s",
                        (json.dumps(body.usedAnnotationNames), project_id))
        if body.cantusSourceId is not None:
            cur.execute("UPDATE projects SET cantus_source_id=%s WHERE id=%s", (body.cantusSourceId, project_id))
        con.commit()
        return {"ok": True}


@router.post("/projects/{project_id}/restore")
def restore_project(project_id: int, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        cur.execute("UPDATE projects SET deleted_at=NULL WHERE id=%s", (project_id, ))
        con.commit()
        return {"ok": True}


@router.delete("/projects/{project_id}")
def permanently_delete_project(project_id: int, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        cur.execute("DELETE FROM annotations WHERE project_id=%s", (project_id,))
        cur.execute("DELETE FROM project_logs WHERE project_id=%s", (project_id,))
        cur.execute("DELETE FROM activity_log WHERE project_id=%s", (project_id,))
        cur.execute("DELETE FROM project_images WHERE project_id=%s", (project_id,))
        cur.execute("DELETE FROM project_models WHERE project_id=%s", (project_id,))
        cur.execute("DELETE FROM mei_files WHERE project_id=%s", (project_id,))
        cur.execute("DELETE FROM text_alignments WHERE project_id=%s", (project_id,))
        # staffline_detections accumulates forever by design (see CLAUDE.md's
        # Database schema table) but still carries a project_id FK, which
        # would otherwise block this same DELETE FROM projects below.
        cur.execute("DELETE FROM staffline_detections WHERE project_id=%s", (project_id,))
        cur.execute("DELETE FROM ic_xml_files WHERE project_id=%s", (project_id,))
        # IC persists its sessions (incl. page-image BYTEA) in a table it
        # owns; purge this project's rows too. Guarded by to_regclass since
        # the table only exists once the IC service has run against this DB.
        cur.execute("SELECT to_regclass('ic_sessions')")
        if cur.fetchone()[0] is not None:
            cur.execute("DELETE FROM ic_sessions WHERE project_id=%s", (project_id,))
        cur.execute("DELETE FROM projects WHERE id=%s", (project_id,))
        con.commit()
        return {"ok": True}


_ZIP_NAME_SAFE = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._- ()"
)


def _zip_entry_name(name, fallback: str) -> str:
    """Reduce a stored row name to a single safe path segment.

    project_images.name is whatever multipart filename the client sent --
    Starlette hands `UploadFile.filename` over verbatim, and images_api.py
    stores it as-is -- so a name can carry `/`, `\\`, or `..` components.
    Written into a ZIP entry unchanged those become a Zip Slip: `zipfile`'s
    own extract() sanitizes, but plenty of other extractors don't, and the
    archive is a file the user hands around. Take the basename, then keep
    only an allow-list of filename characters, falling back for a name that
    survives as empty or as a pure traversal component.
    """
    base = str(name or "").replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = "".join(ch if ch in _ZIP_NAME_SAFE else "_" for ch in base).strip()
    if not cleaned or set(cleaned) <= {"."}:
        return fallback
    return cleaned


@router.get("/projects/{project_id}/export")
def export_project(project_id: int, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        cur.execute("SELECT name FROM projects WHERE id=%s", (project_id, ))
        project_name = cur.fetchone()[0]
        cur.execute("SELECT name, mime_type, data FROM project_images WHERE project_id=%s", (project_id, ))
        images = cur.fetchall()
        cur.execute("SELECT name, xml_content FROM mei_files WHERE project_id=%s", (project_id, ))
        mei_files = cur.fetchall()
        cur.execute("SELECT name, xml_content FROM ic_xml_files WHERE project_id=%s", (project_id, ))
        ic_xml_files = cur.fetchall()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (img_name, _mime, data) in enumerate(images):
            zf.writestr(f"images/{_zip_entry_name(img_name, f'image_{i}')}", bytes(data))
        for i, (mei_name, xml_content) in enumerate(mei_files):
            zf.writestr(f"mei/{_zip_entry_name(mei_name, f'file_{i}.mei')}", xml_content or "")
        # BYTEA, so these go in verbatim -- the archived classifier XML is
        # the encoder's exact input, which is the point of keeping it. (The
        # mei/ entries above are TEXT: mothra generates that XML itself and
        # always as UTF-8.)
        for i, (xml_name, xml_bytes) in enumerate(ic_xml_files):
            zf.writestr(f"ic-xml/{_zip_entry_name(xml_name, f'file_{i}.xml')}",
                        bytes(xml_bytes) if xml_bytes else b"")
    buf.seek(0)
    safe_name = project_name.replace(" ", "_")
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.zip"'}
    )


@router.post("/projects/{project_id}/duplicate")
def duplicate_project(project_id: int, current_user=Depends(get_current_user)):
    """Clone a project's raw inputs (images, custom models) as a fresh,
    unstarted project -- not a full snapshot.

    steps_unlocked resets to 0 deliberately: also cloning annotations/
    mei_files/text_alignments/staffline_detections would leave those rows
    orphaned and unreachable (nothing would re-point the UI at them once
    steps_unlocked says "nothing done yet"). A duplicate is a clean rerun
    starting point, not a copy of pipeline progress.
    """
    with db_cursor() as (con, cur):
        cur.execute(
            "SELECT name FROM projects WHERE id = %s AND user_id = %s AND deleted_at IS NULL",
            (project_id, current_user["id"])
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="project not found")

        new_name = f"{row[0]} (copy)"
        now = datetime.utcnow()
        cur.execute(
            "INSERT INTO projects (user_id, name, steps_unlocked, last_opened_at, created_at)"
            " VALUES (%s, %s, 0, %s, %s) RETURNING id",
            (current_user["id"], new_name, now, now)
        )
        new_id = cur.fetchone()[0]

        cur.execute("SELECT name, mime_type, data FROM project_images WHERE project_id=%s", (project_id,))
        for img_name, mime, data in cur.fetchall():
            cur.execute(
                "INSERT INTO project_images (id, project_id, name, mime_type, data, created_at)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                (str(_uuid.uuid4()), new_id, img_name, mime, data, now)
            )

        cur.execute("SELECT name, file_path, kind FROM project_models WHERE project_id=%s", (project_id,))
        for model_name, file_path, model_kind in cur.fetchall():
            new_model_id = str(_uuid.uuid4())
            new_file_path = None
            if file_path and Path(file_path).exists():
                new_model_dir = MODELS_DIR / str(new_id)
                new_model_dir.mkdir(parents=True, exist_ok=True)
                ext = Path(file_path).suffix
                new_file_path = str(new_model_dir / f"{new_model_id}{ext}")
                shutil.copy2(file_path, new_file_path)
            cur.execute(
                "INSERT INTO project_models (id, project_id, name, file_path, kind) VALUES (%s, %s, %s, %s, %s)",
                (new_model_id, new_id, model_name, new_file_path, model_kind)
            )

        con.commit()

        cur.execute(
            "SELECT id, name, steps_unlocked, used_image_names, used_model_names, deleted_at,"
            " last_opened_at, is_pinned, used_annotation_names"
            " FROM projects WHERE id=%s",
            (new_id,)
        )
        result = _project_row_to_dict(cur, cur.fetchone(), current_user["username"])
    return result


@router.get("/projects/{project_id}/logs/download")
def download_project_logs(project_id: int, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        cur.execute("SELECT name FROM projects WHERE id=%s", (project_id, ))
        project_name = cur.fetchone()[0]

        cur.execute(
            "SELECT action_type, detail, created_at FROM activity_log WHERE project_id=%s ORDER BY created_at ASC",
            (project_id,)
        )
        activity_rows = cur.fetchall()

        cur.execute(
            "SELECT content, created_at FROM project_logs WHERE project_id=%s AND log_type='encoding' ORDER BY created_at ASC",
            (project_id,)
        )
        encoding_rows = cur.fetchall()

    activity_lines = [
        f"[{r[2]}] {r[0]}: {r[1]}" for r in activity_rows
    ] or ["no activity recorded"]
    activity_text = "\n".join(activity_lines)

    encoding_sections = []
    for content, created_at in encoding_rows:
        encoding_sections.append(f"--- Run: {created_at} --- \n{content}")
    encoding_text = "\n\n".join(encoding_sections) if encoding_sections else "no encoding logs recorded"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("activity_log.txt", activity_text)
        zf.writestr("encoding_logs.txt", encoding_text)
    buf.seek(0)
    safe_name = project_name.replace(" ", "_")
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}_logs.zip"'}
    )
