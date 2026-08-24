"""Interactive Classifier (IC) bridge.

Connects the mothra workflow to the standalone Interactive Classifier
vendored at the repo's top-level ``ic/`` submodule and run as its own
service (default ``http://localhost:8000``).

The flow is server-driven so the GameraXML never has to round-trip
through the embedded IC iframe:

1. ``POST /api/projects/{id}/ic/start`` — read a project page image from
   the DB, generate bbox annotations for it, and create an IC session via
   IC's ``POST /sessions``. Returns the session id plus the deep-link URL
   the frontend embeds in an iframe (``{IC}/?session=<id>&embed=1``).
2. ``POST /api/ic/{session_id}/complete`` — call IC's
   ``POST /sessions/{id}/complete`` and hand the resulting GameraXML back
   to the frontend, which feeds it into the existing encode flow.
   Exported with ``finalize=false`` so the session stays editable and
   resumable afterwards (see :func:`ic_complete`). The GameraXML itself is
   filed under the page by the encode job, not here -- see
   ``ic_xml_store.py``.

IC is reached over plain HTTP with the stdlib ``urllib`` (no extra
dependency); calls are server-to-server, so IC's CORS allowlist is
irrelevant.
"""
from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from config import IC_API_URL, IC_PUBLIC_URL

from auth_api import get_current_user, db_cursor, require_project_owner

router = APIRouter()


def _music_only_yolo_lines(yolo_txt: str) -> str:
    """Keep only class_id==1 ('music') lines from a merged text/music/staves
    YOLO annotation blob before handing candidate glyphs to IC — IC classifies
    neume shapes into subtypes, and 'text'(0)/'staves'(2) boxes aren't neumes.
    Confirmed on a real page that unfiltered text-class boxes get classified
    into neume shapes and end up as spurious <neume> elements in the final
    MEI (see medieval_models.py's merged 0=text/1=music/2=staves convention)."""
    kept = []
    for line in yolo_txt.strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        if class_id == 1:
            kept.append(line)
    return "\n".join(kept)

# ---------------------------------------------------------------------------
# Bounding boxes
# ---------------------------------------------------------------------------

def generate_bboxes(image_bytes: bytes, project_id: int, image_id: str, image_name: str) -> tuple[bytes, str]:
    """Return ``(annotation_bytes, format)`` for ``ic_start()``.

    Uses stored YOLO detections when available; falls back to a coarse
    placeholder grid so the IC step is always exercisable without a prior
    predict run.
    """
    with db_cursor() as (con, cur):
        # mothra#241: match by image_id, not image_name -- a same-named
        # duplicate upload now gets its own row/id, and image_name alone is
        # not unique within a project (see auth_api.py's
        # get_latest_text_alignment docstring for the same rule).
        # CodeRabbit: the "image_id IS NULL" arm is defensive insurance
        # against a hypothetical legacy row with no image_id ever recorded,
        # not a known real gap -- an exact id match always wins, so this
        # can't fall back onto a different same-named image's annotation.
        cur.execute(
            "SELECT yolo_txt FROM annotations"
            " WHERE project_id=%s AND (image_id=%s OR (image_id IS NULL AND image_name=%s))"
            " ORDER BY created_at DESC LIMIT 1",
            (project_id, image_id, image_name),
        )
        row = cur.fetchone()
        if row and row[0].strip():
            music_only = _music_only_yolo_lines(row[0])
            if music_only:
                return music_only.encode(), "yolo"

    try:
        from PIL import Image  # available in the mothra venv

        with Image.open(io.BytesIO(image_bytes)) as im:
            page_w, page_h = im.size
    except Exception:
        # Fall back to a nominal page size if the image can't be decoded;
        # the boxes are placeholders regardless.
        page_w, page_h = 1000, 1400

    cols, rows = 6, 8
    margin_x = page_w // (cols + 2)
    margin_y = page_h // (rows + 2)
    box_w = max(1, page_w // (cols * 2))
    box_h = max(1, page_h // (rows * 2))
    step_x = (page_w - 2 * margin_x) // cols
    step_y = (page_h - 2 * margin_y) // rows

    annotations = []
    for r in range(rows):
        for c in range(cols):
            annotations.append(
                {
                    "id": uuid.uuid4().hex,
                    "classId": 2,  # Neumes
                    "bbox": [
                        margin_x + c * step_x,
                        margin_y + r * step_y,
                        box_w,
                        box_h,
                    ],
                }
            )
    return json.dumps({"annotations": annotations}).encode(), "json"


# ---------------------------------------------------------------------------
# Minimal multipart HTTP client (stdlib only)
# ---------------------------------------------------------------------------


def _post_multipart(url: str, fields: dict[str, str], files: list[tuple], timeout: int = 120):
    """POST ``multipart/form-data`` and return ``(status, body_bytes)``.

    ``files`` is a list of ``(field_name, filename, content_type, data)``.
    """
    boundary = uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        body += value.encode() + b"\r\n"
    for name, filename, ctype, data in files:
        body += f"--{boundary}\r\n".encode()
        body += (
            f'Content-Disposition: form-data; name="{name}"; '
            f'filename="{filename}"\r\n'
        ).encode()
        body += f"Content-Type: {ctype}\r\n\r\n".encode()
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(url, data=bytes(body), method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def _post_empty(url: str, timeout: int = 120):
    """POST with no body and return ``(status, body_bytes, headers)``."""
    req = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read(), resp.headers


def _get(url: str, timeout: int = 30):
    """GET and return ``(status, body_bytes)``. Raises ``HTTPError`` on 4xx/5xx."""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def _delete(url: str, timeout: int = 30):
    """DELETE and return ``(status, body_bytes)``. Raises ``HTTPError`` on 4xx/5xx."""
    req = urllib.request.Request(url, method="DELETE")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def _ic_unreachable(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=502,
        detail=f"Interactive Classifier at {IC_API_URL} is unreachable: {exc}",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


class IcStartRequest(BaseModel):
    imageName: str
    # mothra#241 / CodeRabbit: image_name alone is not unique within a
    # project once duplicate-named uploads are allowed -- prefer this when
    # the caller has it (both frontend call sites do, since they always
    # start from a real ProjectImage). Optional so any caller that only
    # sends imageName still resolves via the name fallback below.
    imageId: Optional[str] = None


def _project_image(
    project_id: int, image_name: str, user_id: int, image_id: Optional[str] = None,
) -> tuple[str, bytes, str, bool]:
    """Return ``(image_id, data, mime_type, ic_synthetic)`` for a project image the user owns.

    Prefers an exact ``image_id`` match when given -- ``image_name`` alone is
    not unique within a project once duplicate-named uploads are allowed
    (mothra#241), so a name-only lookup here could resolve to an arbitrary
    same-named image before its bytes/id are ever handed to IC. ``ic_synthetic``
    is the mothra#220 DL-1 placeholder-grid flag (see _set_project_image_synthetic).
    """
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user_id)
        if image_id:
            cur.execute(
                "SELECT id, data, mime_type, ic_synthetic FROM project_images"
                " WHERE project_id=%s AND id=%s",
                (project_id, image_id),
            )
        else:
            cur.execute(
                "SELECT id, data, mime_type, ic_synthetic FROM project_images"
                " WHERE project_id=%s AND name=%s",
                (project_id, image_name),
            )
        img = cur.fetchone()
        if not img:
            raise HTTPException(status_code=404, detail="image not found")
        return img[0], bytes(img[1]), (img[2] or "image/png"), bool(img[3])


def _set_project_image_synthetic(project_id: int, image_id: str, synthetic: bool) -> None:
    """Persist the mothra#220 DL-1 synthetic flag so it survives IC session resume."""
    with db_cursor() as (con, cur):
        cur.execute(
            "UPDATE project_images SET ic_synthetic=%s WHERE id=%s AND project_id=%s",
            (synthetic, image_id, project_id),
        )
        con.commit()


@router.post("/projects/{project_id}/ic/start")
def ic_start(project_id: int, body: IcStartRequest, user=Depends(get_current_user)):
    """Resume — or stage — this project page in IC and return its deep-link.

    Reads the page image from the DB, then:

    * If IC has a saved, still-editable session for this exact
      ``(project_id, image_id)``, return a ``?session=…`` deep-link so the
      user resumes their prior work (the session id is echoed back and also
      re-broadcast via postMessage when the SPA opens it).
    * Otherwise generate bbox annotations and *stage* them via IC's
      ``POST /staging`` (tagged with the project + image id so the created
      session is resumable next time), returning the ``?staged=…`` URL. The
      session is created by the user on IC's create-session screen and its
      id comes back via postMessage.
    """
    image_id, image_bytes, mime_type, ic_synthetic = _project_image(
        project_id, body.imageName, user["id"], image_id=body.imageId
    )

    # Resume a previously-saved session for this page, if one exists.
    try:
        _, raw = _get(
            f"{IC_API_URL}/sessions/lookup"
            f"?project_id={project_id}"
            f"&image_id={urllib.parse.quote(str(image_id))}"
        )
        existing = json.loads(raw).get("session_id")
    except urllib.error.HTTPError:
        existing = None  # 404 = nothing to resume
    except urllib.error.URLError as exc:
        raise _ic_unreachable(exc)

    if existing:
        return {
            "session_id": existing,
            "ic_url": f"{IC_PUBLIC_URL}/?session={existing}&embed=1",
            "resumed": True,
            # mothra#220 DL-1: IC's own /sessions/lookup doesn't carry this
            # flag, so read it back from where ic_start() persisted it the
            # first time this page was staged (see the fresh-staging branch
            # below) -- otherwise a placeholder-grid session would silently
            # lose its "no prediction ran" warning on resume/reload.
            "synthetic": ic_synthetic,
        }

    # Nothing saved — stage the page + bboxes fresh.
    annotations, ann_format = generate_bboxes(image_bytes, project_id, image_id, body.imageName)
    try:
        status, raw = _post_multipart(
            f"{IC_API_URL}/staging",
            fields={
                "annotations_format": ann_format,
                "project_id": str(project_id),
                "image_id": str(image_id),
            },
            files=[
                ("page_image", body.imageName, mime_type, image_bytes),
                ("annotations", "annotations.json", "application/json", annotations),
            ],
        )
    except urllib.error.URLError as exc:
        raise _ic_unreachable(exc)

    if status >= 400:
        raise HTTPException(status_code=502, detail=f"IC /staging failed ({status}): {raw[:500]!r}")

    staging_id = json.loads(raw).get("staging_id")
    if not staging_id:
        raise HTTPException(status_code=502, detail="IC /staging returned no staging id")

    # mothra#220 DL-1: ann_format is "json" only on generate_bboxes()'s
    # placeholder-grid fallback (no real YOLO detections exist yet for this
    # page) -- surfaced so InteractiveClassifier.tsx can show an unmissable
    # "no prediction ran" banner instead of a fabricated grid silently
    # looking like real detections. Persisted on project_images so a later
    # resume (the `existing` branch above) can still see it -- IC's own
    # session doesn't carry it.
    synthetic = ann_format == "json"
    _set_project_image_synthetic(project_id, image_id, synthetic)

    return {
        "staging_id": staging_id,
        "ic_url": f"{IC_PUBLIC_URL}/?staged={staging_id}&embed=1",
        "resumed": False,
        "synthetic": synthetic,
    }


class IcCompleteRequest(BaseModel):
    """Optional page context for an export.

    Only used to check the caller owns the project the page belongs to (see
    :func:`ic_complete`'s note on why session_id alone isn't enough).
    Omitted entirely by any caller that just wants the XML back -- the
    export itself doesn't need it, since IC resolves everything from
    ``session_id``. ``imageId``/``imageName`` are along for the ride so a
    future caller has the page identity to hand without another round trip.
    """
    projectId: Optional[int] = None
    imageId: Optional[str] = None
    imageName: Optional[str] = None


@router.post("/ic/{session_id}/complete")
def ic_complete(
    session_id: str,
    body: Optional[IcCompleteRequest] = None,
    user=Depends(get_current_user),
):
    """Export an IC session's GameraXML (base64), leaving it editable.

    The frontend turns this into a ``File`` and feeds it to the existing
    ``/api/encode-upload`` flow.

    Exported with ``finalize=false``: IC's default transitions the session
    to ``EXPORT``, which is terminal and read-only, and which its own
    ``/sessions/lookup`` treats as *not* resumable -- so encoding a page
    used to silently retire the session behind it, and re-entering the IC
    step got a blank new one instead of the corrections. Here the XML is a
    mid-pipeline artefact, not the end of the session's life: the user
    encodes, sees the result in Neon, and can come back through "manage IC
    sessions" to keep correcting the same page. (Requires an ``ic/``
    carrying the ``finalize`` parameter; an older IC ignores the unknown
    query param and finalises, which is the pre-existing behaviour rather
    than an error.)

    ``body`` is optional page context; when it names a project, the caller
    must own it. The exported XML is deliberately *not* filed under the page
    here -- ``tasks_encode.py`` does that as it parses the encoder's input
    (see ic_xml_store.py), so "Classifier XML" means "what was encoded" and
    covers every path into the encoder rather than only this bridge.

    Note: without ``body.projectId`` this only checks that *some* user is
    logged in, not that they own the project session_id belongs to -- IC's
    own session store is the real authority here, but within mothra's own
    trust model any authenticated user who learns another user's session_id
    could export it. Inconsistent with the rest of this file; revisit if
    session_id ever becomes guessable/discoverable in practice.
    """
    if body is not None and body.projectId is not None:
        with db_cursor() as (con, cur):
            require_project_owner(cur, body.projectId, user["id"])
    try:
        status, raw, _headers = _post_empty(
            f"{IC_API_URL}/sessions/{session_id}/complete?page=true&finalize=false"
        )
    except urllib.error.HTTPError as exc:
        # IC maps an unknown/torn-down session to 404.
        raise HTTPException(status_code=exc.code, detail=f"IC complete failed: {exc.read()[:500]!r}")
    except urllib.error.URLError as exc:
        raise _ic_unreachable(exc)

    if status >= 400:
        raise HTTPException(status_code=502, detail=f"IC complete failed ({status})")

    return {
        "session_id": session_id,
        "xml_base64": base64.b64encode(raw).decode(),
        "filename": f"ic-session-{session_id}.xml",
    }


# ---------------------------------------------------------------------------
# Stored classifier XML ("Generated files" tab)
# ---------------------------------------------------------------------------
#
# Written by the encode job (ic_xml_store.py); read here. The list itself
# rides along with the project payload (projects_api.py's
# _project_row_to_dict / list_projects), like annotations and stafflines --
# only the XML body, which is megabytes per page, is fetched on demand.


def _ic_xml_row(project_id: int, xml_id: str, user_id: int) -> tuple[str, str]:
    """Return ``(name, xml_content)`` for one stored file the user owns."""
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user_id)
        cur.execute(
            "SELECT name, xml_content FROM ic_xml_files WHERE id=%s AND project_id=%s",
            (xml_id, project_id),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="IC XML not found")
        return row[0], row[1]


@router.get("/projects/{project_id}/ic-xml/{xml_id}")
def get_ic_xml(project_id: int, xml_id: str, user=Depends(get_current_user)):
    """The stored GameraXML as JSON, for the in-app viewer."""
    name, xml_content = _ic_xml_row(project_id, xml_id, user["id"])
    return {"id": xml_id, "name": name, "xmlContent": xml_content}


@router.get("/projects/{project_id}/ic-xml/{xml_id}/download")
def download_ic_xml(project_id: int, xml_id: str, user=Depends(get_current_user)):
    """The stored GameraXML as a file download."""
    name, xml_content = _ic_xml_row(project_id, xml_id, user["id"])
    return Response(
        content=xml_content,
        media_type="application/xml",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@router.delete("/projects/{project_id}/ic-xml/{xml_id}")
def delete_ic_xml(project_id: int, xml_id: str, user=Depends(get_current_user)):
    """Drop one stored export.

    Only removes mothra's copy -- the IC session it came from is untouched
    and still resumable, so re-exporting the page brings the file back.
    """
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        cur.execute(
            "DELETE FROM ic_xml_files WHERE id=%s AND project_id=%s",
            (xml_id, project_id),
        )
        con.commit()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Saved-session management (frontend iframes IC's own resume UI)
# ---------------------------------------------------------------------------


@router.get("/projects/{project_id}/ic/manage-url")
def ic_manage_url(project_id: int, user=Depends(get_current_user)) -> dict:
    """Return the deep-link mothra iframes to manage this project's IC sessions.

    All the list/resume/delete logic already lives in IC's own SPA + REST
    (``GET/DELETE /sessions``); mothra just needs the URL to embed. Because
    ``IC_PUBLIC_URL`` is server-side config (it differs between local dev and
    Docker), the frontend can't build this itself — same reason
    :func:`ic_start` returns ``ic_url`` rather than the SPA hardcoding it.

    The ``project_id`` scopes IC's otherwise-global session list to this
    project (see IC's ``GET /sessions?project_id=``); the ownership check here
    guards against embedding another user's project's sessions.
    """
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
    return {
        "ic_url": f"{IC_PUBLIC_URL}/?manage=1&project_id={project_id}&embed=1",
    }


def _ic_project_sessions(project_id: int, timeout: int = 5) -> list[dict]:
    """IC's saved sessions for one project, as IC returns them.

    Short timeout on purpose: both callers run on a page open, and a down IC
    must not park a request thread on the default 30s. Errors are the
    caller's to interpret rather than something to swallow into an empty list
    -- "no sessions" and "IC is down" must stay distinguishable, or an
    unreachable IC would silently look like a project with nothing saved.
    """
    try:
        _, raw = _get(f"{IC_API_URL}/sessions?project_id={project_id}", timeout=timeout)
    except urllib.error.URLError as exc:
        raise _ic_unreachable(exc)
    try:
        sessions = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="IC returned a malformed session list")
    if not isinstance(sessions, list):
        raise HTTPException(status_code=502, detail="IC returned a malformed session list")
    return sessions


@router.get("/projects/{project_id}/ic/session-count")
def ic_session_count(project_id: int, user=Depends(get_current_user)) -> dict:
    """How many saved IC sessions this project has.

    Sessions live in IC's store, not mothra's DB, so whether a project has any
    is not derivable here — the project page needs this to decide whether to
    offer "manage IC sessions". It can't key that off ``steps_unlocked``:
    sessions can exist while that is still 0 (the ``VITE_SKIP_PREDICT`` dev
    path goes straight to IC, and a batch text-finding job that fails after
    its YOLO stage leaves annotations behind without advancing the step), and
    those are exactly the cases where a stale session most needs clearing.
    """
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
    return {"count": len(_ic_project_sessions(project_id))}


@router.get("/projects/{project_id}/ic/sessions")
def ic_sessions(project_id: int, user=Depends(get_current_user)) -> list[dict]:
    """This project's saved IC sessions, for mothra's own session picker.

    IC's `?manage=1` page (iframed by `IcSessionsModal`) covers listing and
    deleting, but it opens one session at a time. The IC step page instead
    lets the user pick *several* pages to reopen in one go -- the filmstrip
    is built to hold more than one -- which needs the list in mothra's own
    UI, where it can be resolved against the project's images and shown with
    their thumbnails. Camel-cased here (IC's own payload is snake_case) to
    match every other mothra API shape.
    """
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
    return [
        {
            "sessionId": s.get("id"),
            "state": s.get("state"),
            "sourceName": s.get("source_name") or "",
            "imageId": s.get("image_id"),
            "glyphCount": s.get("n_glyphs"),
            "updatedAt": s.get("updated_at"),
        }
        for s in _ic_project_sessions(project_id)
        if isinstance(s, dict) and s.get("id")
    ]


# ---------------------------------------------------------------------------
# Batch training set + "queue all" (parent-page-driven, no per-page iframe)
# ---------------------------------------------------------------------------


@router.get("/ic/training-presets")
def ic_training_presets(user=Depends(get_current_user)) -> list[str]:
    """Proxy IC's built-in training-set preset list to the IC parent page.

    Lets mothra's IC parent page render the same preset checkboxes IC's own
    create-session screen shows, so a training set can be picked once and
    applied to every page via :func:`ic_auto_queue`.
    """
    try:
        _, raw = _get(f"{IC_API_URL}/training-presets")
    except urllib.error.URLError as exc:
        raise _ic_unreachable(exc)
    try:
        presets = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return presets if isinstance(presets, list) else []


@router.post("/projects/{project_id}/ic/auto-queue")
async def ic_auto_queue(
    project_id: int,
    imageName: Annotated[str, Form()],
    # mothra#241 / CodeRabbit: see IcStartRequest.imageId's comment -- same
    # disambiguation, optional Form field so this stays backward compatible.
    imageId: Annotated[Optional[str], Form()] = None,
    training_presets: Annotated[Optional[str], Form()] = None,
    training_files: Annotated[Optional[List[UploadFile]], File()] = None,
    user=Depends(get_current_user),
):
    """Classify one project page with a shared training set → GameraXML.

    The server-side half of the "queue all available" batch path. Instead of
    opening the IC iframe for each page, mothra generates the page's bboxes,
    creates an IC session seeded with the caller's training set (which runs a
    classify round at ingest), and completes it — all server-to-server —
    returning the GameraXML the frontend turns into a ``File`` for the encode
    queue. Mirrors IC's in-iframe "queue page" (auto-export), just driven from
    the parent so every page can be queued in one pass.

    ``training_presets`` is a JSON-encoded ``list[str]`` of built-in preset
    filenames (see :func:`ic_training_presets`); ``training_files`` are
    optional GameraXML (.xml) uploads. At least one training source should be
    present, since classify needs a non-empty training pool — the frontend
    enforces this before calling.
    """
    image_id, image_bytes, mime_type, _ic_synthetic = _project_image(
        project_id, imageName, user["id"], image_id=imageId
    )
    annotations, ann_format = generate_bboxes(image_bytes, project_id, image_id, imageName)

    fields = {"annotations_format": ann_format}
    if training_presets:
        fields["training_presets"] = training_presets
    files = [
        ("page_image", imageName, mime_type, image_bytes),
        ("annotations", "annotations.json", "application/json", annotations),
    ]
    for tf in training_files or []:
        files.append(
            ("training_files", tf.filename or "training.xml", "application/xml", await tf.read())
        )

    # 1. Create + classify the session in one call (training set → classify).
    try:
        status, raw = _post_multipart(f"{IC_API_URL}/sessions", fields=fields, files=files)
    except urllib.error.URLError as exc:
        raise _ic_unreachable(exc)
    if status >= 400:
        raise HTTPException(status_code=502, detail=f"IC /sessions failed ({status}): {raw[:500]!r}")
    session_id = json.loads(raw).get("id")
    if not session_id:
        raise HTTPException(status_code=502, detail="IC /sessions returned no session id")

    # 2. Finalise → GameraXML. Unlike ic_complete above this *does*
    # finalise: the session was created here, server-side, purely to run
    # one classify round, and POST /sessions (unlike /staging) takes no
    # project/image id, so IC could never map it back to a page for
    # "manage IC sessions" to resume anyway. Leaving it in CLASSIFYING
    # would only accumulate unreachable sessions.
    try:
        c_status, c_raw, _headers = _post_empty(
            f"{IC_API_URL}/sessions/{session_id}/complete?page=true"
        )
    except urllib.error.HTTPError as exc:
        raise HTTPException(status_code=exc.code, detail=f"IC complete failed: {exc.read()[:500]!r}")
    except urllib.error.URLError as exc:
        raise _ic_unreachable(exc)
    if c_status >= 400:
        raise HTTPException(status_code=502, detail=f"IC complete failed ({c_status})")

    stem = imageName.rsplit(".", 1)[0] if "." in imageName else imageName
    return {
        "session_id": session_id,
        "xml_base64": base64.b64encode(c_raw).decode(),
        "filename": f"{stem}.xml",
    }
