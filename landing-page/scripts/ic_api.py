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
import urllib.request
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth_api import get_current_user, get_db_conn

router = APIRouter()

# Where this backend reaches the IC API (server-to-server) and where the
# browser/iframe reaches the IC SPA. Kept separate so a deployment can
# point them at different origins; for local dev both are :8000.
IC_API_URL = os.environ.get("IC_API_URL", "http://localhost:8000").rstrip("/")
IC_PUBLIC_URL = os.environ.get("IC_PUBLIC_URL", "http://localhost:8000").rstrip("/")


# ---------------------------------------------------------------------------
# Bounding boxes
# ---------------------------------------------------------------------------
#
# IC cannot create a session without a bbox annotation document. Real YOLO
# inference (POST /api/predict) does not exist in mothra yet, so this is a
# PLACEHOLDER that lays a coarse grid of boxes over the page just so the
# embed → classify → encode loop is exercisable end to end. Swap the body
# of generate_bboxes() for the real detector output (MOTHRA JSON / YOLO TXT)
# when it lands — nothing else in this module changes.

def generate_bboxes(image_bytes: bytes) -> bytes:
    """Return a MOTHRA-JSON bbox document (bytes) for ``image_bytes``.

    STUB. Emits a coarse grid of boxes covering the page so a created IC
    session has glyphs to display. ``classId`` 2 == Neumes (see IC's
    ``ingest._MOTHRA_CLASS_TO_CATEGORY``). The schema is
    ``{"annotations": [{"id", "classId", "bbox": [ulx, uly, w, h]}, ...]}``.
    """
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
    return json.dumps({"annotations": annotations}).encode()


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


def _project_image(project_id: int, image_name: str, user_id: int) -> tuple[bytes, str]:
    """Return ``(data, mime_type)`` for a project image the user owns."""
    con = get_db_conn()
    cur = con.cursor()
    try:
        cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="project not found")
        if row[0] != user_id:
            raise HTTPException(status_code=403, detail="not your project")
        cur.execute(
            "SELECT data, mime_type FROM project_images WHERE project_id=%s AND name=%s",
            (project_id, image_name),
        )
        img = cur.fetchone()
        if not img:
            raise HTTPException(status_code=404, detail="image not found")
        return bytes(img[0]), (img[1] or "image/png")
    finally:
        cur.close()
        con.close()


@router.post("/projects/{project_id}/ic/start")
def ic_start(project_id: int, body: IcStartRequest, user=Depends(get_current_user)):
    """Stage one project page + its bboxes in IC and return its deep-link.

    Reads the page image from the DB, generates bbox annotations for it,
    and *stages* them via IC's ``POST /staging`` — the session itself is
    created by the user on IC's create-session screen (so they can add
    training data and pick a vocabulary). Returns the ``?staged=…`` URL the
    frontend embeds; the created session id comes back via postMessage.
    """
    image_bytes, mime_type = _project_image(project_id, body.imageName, user["id"])
    annotations = generate_bboxes(image_bytes)

    try:
        status, raw = _post_multipart(
            f"{IC_API_URL}/staging",
            fields={"annotations_format": "json"},
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

    return {
        "staging_id": staging_id,
        "ic_url": f"{IC_PUBLIC_URL}/?staged={staging_id}&embed=1",
    }


@router.post("/ic/{session_id}/complete")
def ic_complete(session_id: str, user=Depends(get_current_user)):
    """Finalise an IC session and return its GameraXML (base64).

    The frontend turns this into a ``File`` and feeds it to the existing
    ``/api/encode-upload`` flow.
    """
    try:
        status, raw, _headers = _post_empty(f"{IC_API_URL}/sessions/{session_id}/complete")
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
