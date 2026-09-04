"""MEI file CRUD, Neon batch-editor edit-session bootstrap, and the
token-authed raw-content endpoints the embedded Neon editor iframe uses."""
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Literal, Optional
import base64
import uuid as _uuid
import sys
import xml.etree.ElementTree as ET

from auth_api import (
    get_current_user, db_cursor, require_project_owner, _log_activity,
    _make_edit_token, _verify_edit_token, get_latest_text_alignment,
)
from job_store import neon_manifest_put
import encode_to_mei

router = APIRouter()


def _restore_notation_subtype(previous_xml_content: Optional[str], incoming_xml: str) -> str:
    """Neon's own Verovio-compatibility shim (stripHufnagelForVerovio, in the
    landing-page/neon submodule's ConvertMei.ts) normalizes a loaded file's
    <staffDef notationtype="neume.square"|"neume.hufnagel"> down to the bare
    "neume" before caching it, and Neon's save flow (updateDatabase()) PUTs
    that same stripped copy back here -- so saving any edit in Neon would
    otherwise silently lose which notation the file was encoded in, and the
    file would reopen as Square (NeonBatchEditor.tsx's applyNotationTypeFont
    has no subtype left to read). The neon submodule is tracked clean
    against upstream with no local patches, so the fix lives on this side
    of the PUT instead: if the incoming save has been stripped to the
    generic form, re-stamp whatever subtype the previously-stored copy had.
    """
    if not previous_xml_content:
        return incoming_xml
    try:
        prev_root = ET.fromstring(previous_xml_content.encode("utf-8"))
    except ET.ParseError:
        return incoming_xml
    prev_staff_def = prev_root.find(f".//{{{encode_to_mei.MEI_NS}}}staffDef")
    prev_notation = prev_staff_def.get("notationtype") if prev_staff_def is not None else None
    if prev_notation not in ("neume.square", "neume.hufnagel"):
        return incoming_xml

    try:
        incoming_root = ET.fromstring(incoming_xml.encode("utf-8"))
    except ET.ParseError:
        return incoming_xml
    incoming_staff_def = incoming_root.find(f".//{{{encode_to_mei.MEI_NS}}}staffDef")
    if incoming_staff_def is None or incoming_staff_def.get("notationtype") != "neume":
        # Missing <staffDef>, or one that already carries a real subtype
        # (or anything else) -- never overwrite an explicit value with a
        # stale one from the previous revision.
        return incoming_xml
    incoming_staff_def.set("notationtype", prev_notation)
    ET.register_namespace("", encode_to_mei.MEI_NS)
    return ET.tostring(incoming_root, encoding="unicode")

# Matches types.ts's StaveSource union and the tags tasks_encode.py's
# 3-tier fallback actually produces (see auth_api.py's mei_files.stave_source
# migration comment) -- a Literal here keeps this column's contents aligned
# with what the frontend badge (MeiViewerModal.tsx) knows how to render,
# instead of accepting any string a client sends.
StaveSource = Literal[
    "staffline_detection", "yolo_annotation", "glyph_estimate",
    "glyph_estimate_unresolved_lines", "glyph_estimate_synthetic_lines",
    "placeholder_no_glyphs",
]


class AddMeiBody(BaseModel):
    name: str
    xmlContent: str
    # mothra#241: project_images.id of the source page, when the caller has
    # one (the batch IC->encode path does; the ad-hoc single-image
    # encode-upload path doesn't, see tasks_encode.py's _encode_one). Lets
    # create_edit_session/getImageProgress match by id instead of the
    # not-necessarily-unique imageName.
    imageId: Optional[str] = None
    imageName: Optional[str] = None
    logs: Optional[list[str]] = None
    staveSource: Optional[StaveSource] = None

@router.post("/projects/{project_id}/mei")
def add_mei(project_id: int, body: AddMeiBody, user=Depends(get_current_user)):
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        # CodeRabbit: mei_files.image_id has no FK/composite constraint tying
        # it to project_images(project_id, id) -- require_project_owner only
        # confirms the caller owns project_id, not that body.imageId (an
        # arbitrary client-supplied string) actually names one of ITS
        # images. Reject rather than silently store a dangling/cross-project
        # reference that create_edit_session/getImageProgress would later
        # just fail to resolve (confusing, not a real leak -- its own lookup
        # is already scoped by project_id too -- but a real bug on the
        # frontend side deserves a loud 400, not a silent "no image" later).
        if body.imageId is not None:
            cur.execute(
                "SELECT 1 FROM project_images WHERE id=%s AND project_id=%s",
                (body.imageId, project_id),
            )
            if cur.fetchone() is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"imageId {body.imageId!r} does not belong to project {project_id}",
                )
        mei_id = _uuid.uuid4().hex
        cur.execute(
            "INSERT INTO mei_files (id, project_id, name, xml_content, image_name, stave_source, image_id)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (mei_id, project_id, body.name, body.xmlContent, body.imageName, body.staveSource, body.imageId))
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
        cur.execute("SELECT xml_content FROM mei_files WHERE id=%s AND project_id=%s", (mei_id, project_id))
        row = cur.fetchone()
        xml_content = _restore_notation_subtype(row[0] if row else None, xml_content)
        cur.execute("UPDATE mei_files SET xml_content=%s WHERE id=%s AND project_id=%s",
                    (xml_content, mei_id, project_id))
        con.commit()
    return {"ok": True}


@router.post("/projects/{project_id}/mei/{mei_id}/edit-session")
def create_edit_session(project_id: int, mei_id: str, user=Depends(get_current_user)):
    # Manifests now live in Postgres (neon_manifests, mothra#230), swept by
    # the worker's periodic Celery-beat cleanup (job_store.run_periodic_cleanup)
    # like everything else -- no more proactive per-request sweep needed here.
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        cur.execute("SELECT name, image_name, xml_content, corrected, image_id FROM mei_files"
                    " WHERE id=%s AND project_id=%s", (mei_id, project_id))
        mei_row = cur.fetchone()
        if not mei_row:
            raise HTTPException(status_code=404, detail="MEI not found")
        mei_name, image_name, xml_content, corrected, mei_image_id = mei_row

        image_data_uri = None
        image_bytes = None
        image_id = None
        # mothra#241: prefer the MEI row's own recorded image_id over a name
        # match -- image_name alone is not unique within a project once
        # duplicate-named uploads are allowed (see mei_files.image_id's
        # migration comment in auth_api.py). mei_image_id is None for rows
        # written before the encode round-trip threaded an id through, so
        # this falls back to the pre-existing name match unchanged for those.
        img_row = None
        if mei_image_id:
            cur.execute(
                "SELECT id, data, original_data, original_mime_type, mime_type FROM project_images"
                " WHERE id=%s AND project_id=%s",
                (mei_image_id, project_id))
            img_row = cur.fetchone()
        elif image_name:
            cur.execute(
                "SELECT id, data, original_data, original_mime_type, mime_type FROM project_images"
                " WHERE project_id=%s AND name=%s",
                (project_id, image_name))
            img_row = cur.fetchone()
        if img_row:
            image_id, img_data, original_data, original_mime_type, mime_type = img_row
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
            # The re-sync itself is an improvement, not a precondition for
            # opening Neon -- encode_to_mei.image_dimensions can raise
            # struct.error on a truncated/corrupt stored image, and
            # verify_and_correct_syllables's ET.fromstring can raise
            # ET.ParseError on XML that predates the current structure.
            # Either must degrade to "skip the correction, open the file as-is"
            # rather than 500 an edit session that opened fine before this.
            try:
                text_alignment, alignment_storage_variant = get_latest_text_alignment(
                    cur, project_id, image_name, image_id, include_storage_variant=True,
                )
                dims = encode_to_mei.image_dimensions(image_bytes) if text_alignment else None
                if dims:
                    image_w, image_h = dims
                    # image_bytes/dims is original_data (pre-upload-resize) when present, else the
                    # working copy (img_data) -- see the img_row unpack above. text_alignment's
                    # syl_boxes are absolute-pixel in WHICHEVER of those two the predict job actually
                    # read (alignment_storage_variant, mothra#260 -- SF-2 made that "original" whenever
                    # original_data exists, no longer unconditionally the working copy as this comment
                    # used to assume). Only rescale against working_dims when syl_boxes are actually in
                    # that frame; when they're "original" and image_bytes is itself original_data (the
                    # common case whenever original_data exists), the two frames already match and
                    # comparing against the working copy's dims would rescale a box that needs no
                    # rescaling at all. Degrades to factor=1.0 on both axes (today's behavior) if the
                    # working copy's header can't be read -- image_dimensions returns None rather
                    # than raising. X and Y factors are computed independently, not from one shared
                    # ratio -- imageResize.ts rounds width/height separately after one scalar shrink,
                    # so the two axes' ratios can differ slightly even for a visually uniform resize.
                    image_bytes_variant = "original" if original_data is not None else "working_copy"
                    factor_x = factor_y = 1.0
                    if alignment_storage_variant != image_bytes_variant:
                        working_dims = encode_to_mei.image_dimensions(bytes(img_data)) if img_data is not None else None
                        factor_x = (image_w / working_dims[0]) if working_dims and working_dims[0] else 1.0
                        factor_y = (image_h / working_dims[1]) if working_dims and working_dims[1] else 1.0
                    scaled_alignment = encode_to_mei.scale_text_alignment(text_alignment, factor_x, factor_y)
                    corrected_bytes, correction_logs = encode_to_mei.verify_and_correct_syllables(
                        xml_content.encode(), scaled_alignment, image_w, image_h,
                    )
                    if correction_logs:
                        xml_content = corrected_bytes.decode()
                        cur.execute("UPDATE mei_files SET xml_content=%s WHERE id=%s AND project_id=%s",
                                    (xml_content, mei_id, project_id))
                        con.commit()
                        for line in correction_logs:
                            print(f"[mei-verify] mei_id={mei_id}: {line}", file=sys.stderr)
            except Exception as e:
                con.rollback()
                print(f"[mei-verify] mei_id={mei_id}: skipped re-sync: {e!r}", file=sys.stderr)

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

    neon_manifest_put(session_id, manifest, project_id=project_id)
    return {"session_id": session_id, "manifest_id": manifest_id}
