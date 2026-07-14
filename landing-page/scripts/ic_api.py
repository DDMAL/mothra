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
import urllib.parse
import urllib.request
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth_api import get_current_user, db_cursor, require_project_owner

router = APIRouter()

# Where this backend reaches the IC API (server-to-server) and where the
# browser/iframe reaches the IC SPA. Kept separate so a deployment can
# point them at different origins; for local dev both are :8000.
IC_API_URL = os.environ.get("IC_API_URL", "http://localhost:8000").rstrip("/")
IC_PUBLIC_URL = os.environ.get("IC_PUBLIC_URL", "http://localhost:8000").rstrip("/")


# ---------------------------------------------------------------------------
# Bounding boxes
# ---------------------------------------------------------------------------

def generate_bboxes(image_bytes: bytes, project_id: int, image_name: str) -> tuple[bytes, str]:
    """Return ``(annotation_bytes, format)`` for ``ic_start()``.

    Uses stored YOLO detections when available; falls back to a coarse
    placeholder grid so the IC step is always exercisable without a prior
    predict run.
    """
    with db_cursor() as (con, cur):
        cur.execute(
            "SELECT yolo_txt FROM annotations"
            " WHERE project_id=%s AND image_name=%s"
            " ORDER BY created_at DESC LIMIT 1",
            (project_id, image_name),
        )
        row = cur.fetchone()
        if row and row[0].strip():
            return row[0].encode(), "yolo"

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


def _project_image(
    project_id: int, image_name: str, user_id: int
) -> tuple[str, bytes, str]:
    """Return ``(image_id, data, mime_type)`` for a project image the user owns."""
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user_id)
        cur.execute(
            "SELECT id, data, mime_type FROM project_images"
            " WHERE project_id=%s AND name=%s",
            (project_id, image_name),
        )
        img = cur.fetchone()
        if not img:
            raise HTTPException(status_code=404, detail="image not found")
        return img[0], bytes(img[1]), (img[2] or "image/png")


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
    image_id, image_bytes, mime_type = _project_image(
        project_id, body.imageName, user["id"]
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
        }

    # Nothing saved — stage the page + bboxes fresh.
    annotations, ann_format = generate_bboxes(image_bytes, project_id, body.imageName)
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

    return {
        "staging_id": staging_id,
        "ic_url": f"{IC_PUBLIC_URL}/?staged={staging_id}&embed=1",
        "resumed": False,
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
