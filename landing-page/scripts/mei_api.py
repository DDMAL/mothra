"""MEI file CRUD, Neon batch-editor edit-session bootstrap, and the
token-authed raw-content endpoints the embedded Neon editor iframe uses."""
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
import base64
import json
import uuid as _uuid
import sys

from auth_api import (
    get_current_user, db_cursor, require_project_owner, _log_activity,
    NEON_MANIFESTS_DIR, _make_edit_token, _verify_edit_token, get_latest_text_alignment
)
import encode_to_mei

router = APIRouter()


class AddMeiBody(BaseModel):
    name: str
    xmlContent: str
    imageName: Optional[str] = None
    logs: Optional[list[str]] = None

@router.post("/projects/{project_id}/mei")
def add_mei(project_id: int, body: AddMeiBody, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        mei_id = _uuid.uuid4().hex
        cur.execute(
            "INSERT INTO mei_files (id, project_id, name, xml_content, image_name) VALUES (%s,%s,%s,%s,%s)",
            (mei_id, project_id, body.name, body.xmlContent, body.imageName))
        if body.logs:
            content = "\n".join(body.logs)
            cur.execute(
                "INSERT INTO project_logs (project_id, log_type, content) VALUES (%s, %s, %s)",
                (project_id, "encoding", content)
            )
        _log_activity(cur, project_id, "mei_produced", body.name)
        con.commit()
        return {"id": mei_id}


class UpdateMeiBody(BaseModel):
    corrected: Optional[bool] = None
    xmlContent: Optional[str] = None

@router.patch("/projects/{project_id}/mei/{mei_id}")
def update_mei(project_id: int, mei_id: str, body: UpdateMeiBody, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        if body.xmlContent is not None:
            cur.execute("UPDATE mei_files SET xml_content=%s WHERE id=%s AND project_id=%s",
                        (body.xmlContent, mei_id, project_id))
        if body.corrected is not None:
            cur.execute("UPDATE mei_files SET corrected=%s WHERE id=%s AND project_id=%s",
                        (1 if body.corrected else 0, mei_id, project_id))
            if body.corrected:
                cur.execute("SELECT name FROM mei_files WHERE id=%s",
                            (mei_id,))
                name_row = cur.fetchone()
                _log_activity(cur, project_id, "mei_corrected", name_row[0] if name_row else "")
        con.commit()
        return {"ok": True}


@router.delete("/projects/{project_id}/mei/{mei_id}")
def delete_mei_file(project_id: int, mei_id: str, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        cur.execute("DELETE FROM mei_files WHERE id=%s AND project_id=%s", (mei_id, project_id))
        con.commit()
        return {"ok": True}


@router.get("/projects/{project_id}/mei/{mei_id}/content")
def get_mei_content(project_id: int, mei_id: str, token: str):
    if not _verify_edit_token(token, project_id, mei_id):
        raise HTTPException(status_code=403, detail="invalid or expired edit token")
    with db_cursor() as (con, cur):
        cur.execute("SELECT xml_content FROM mei_files WHERE id=%s AND project_id=%s", (mei_id, project_id))
        row = cur.fetchone()
    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="MEI not found")
    return Response(content=row[0], media_type="application/xml")


@router.put("/projects/{project_id}/mei/{mei_id}/content")
async def put_mei_content(project_id: int, mei_id: str, token: str, request: Request):
    if not _verify_edit_token(token, project_id, mei_id):
        raise HTTPException(status_code=403, detail="invalid or expired edit token")
    xml_content = (await request.body()).decode("utf-8")
    with db_cursor() as (con, cur):
        cur.execute("UPDATE mei_files SET xml_content=%s WHERE id=%s AND project_id=%s",
                    (xml_content, mei_id, project_id))
        con.commit()
    return {"ok": True}


@router.post("/projects/{project_id}/mei/{mei_id}/edit-session")
def create_edit_session(project_id: int, mei_id: str, user=Depends(get_current_user)):

   # proactive cleanup of neon manifests to prevent excess accumulation
    import time as _time
    _now = _time.time()
    for _f in NEON_MANIFESTS_DIR.glob("*.jsonld"):
        if _now - _f.stat().st_mtime > 86400:
            _f.unlink(missing_ok=True)

    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        cur.execute("SELECT name, image_name, xml_content, corrected FROM mei_files"
                    " WHERE id=%s AND project_id=%s", (mei_id, project_id))
        mei_row = cur.fetchone()
        if not mei_row:
            raise HTTPException(status_code=404, detail="MEI not found")
        mei_name, image_name, xml_content, corrected = mei_row

        image_data_uri = None
        image_bytes = None
        if image_name:
            cur.execute(
                "SELECT data, original_data, original_mime_type, mime_type FROM project_images"
                " WHERE project_id=%s AND name=%s",
                (project_id, image_name))
            img_row = cur.fetchone()
            if img_row:
                img_data, original_data, original_mime_type, mime_type = img_row
                image_bytes = bytes(original_data if original_data is not None else img_data)
                # original_data (when present) can be a different format than
                # the resized working copy (e.g. PNG vs. the resize's JPEG) —
                # use its own mime type, falling back to the working copy's
                # for rows written before original_mime_type existed.
                mime = (original_mime_type if original_data is not None else mime_type) or mime_type or "image/jpeg"
                image_data_uri = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"

        # Silently re-sync syllable text/order against mothra-text before
        # Neon ever sees this MEI — but never touch a file a human has
        # already corrected (see verify_and_correct_syllables's docstring
        # for what this can and can't fix).
        if not corrected and xml_content and image_bytes:
            text_alignment = get_latest_text_alignment(cur, project_id, image_name)
            if text_alignment:
                dims = encode_to_mei.image_dimensions(image_bytes)
                if dims:
                    image_w, image_h = dims
                    corrected_bytes, correction_logs = encode_to_mei.verify_and_correct_syllables(
                        xml_content.encode(), text_alignment, image_w, image_h,
                    )
                    if correction_logs:
                        xml_content = corrected_bytes.decode()
                        cur.execute("UPDATE mei_files SET xml_content=%s WHERE id=%s AND project_id=%s",
                                    (xml_content, mei_id, project_id))
                        con.commit()
                        for line in correction_logs:
                            print(f"[mei-verify] mei_id={mei_id}: {line}", file=sys.stderr)

    edit_token = _make_edit_token(project_id, mei_id)
    session_id = _uuid.uuid4().hex[:8]
    manifest_id = str(_uuid.uuid4())
    annotation_id = str(_uuid.uuid4())

    content_url = f"/api/projects/{project_id}/mei/{mei_id}/content?token={edit_token}"
    image_ref = image_data_uri or ""
    manifest = {
        "@context": [
            "http://www.w3.org/ns/anno.jsonld",
            {
                "schema": "http://schema.org/",
                "title": "schema:name",
                "timestamp": "schema:dateModified",
                "image": {"@id": "schema:image", "@type": "@id"},
                "mei_annotations": {"@id": "Annotation", "@type": "@id", "@container": "@list"},
            },
        ],
        "@id": f"urn:uuid:{manifest_id}",
        "title": mei_name,
        "image": image_ref,
        "mei_annotations": [{
            "id": f"urn:uuid:{annotation_id}",
            "type": "Annotation",
            "body": content_url,
            "target": image_ref,
        }]
    }

    NEON_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    (NEON_MANIFESTS_DIR / f"{session_id}.jsonld").write_text(json.dumps(manifest))
    return {"session_id": session_id, "manifest_id": manifest_id}
