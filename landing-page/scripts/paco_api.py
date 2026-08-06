"""
Staffline/background layer-separation bridge — proxies to the standalone
paco-classifier-service, which wraps the Paco_classifier submodule's
TensorFlow auto-encoder pixel classifier. Mirrors text_api.py's/
cantus_api.py's pattern: server-to-server call over plain HTTP (stdlib
urllib, no extra dependency), config-driven service URL.

Kept synchronous and dependency-free on purpose — tasks_predict.py calls
this from inside a background threading.Thread (see its own module docs),
not from an async context, and this module must never touch the calling
task's shared psycopg2 connection.
"""
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
import uuid as _uuid

from config import PACO_API_URL

DEFAULT_TIMEOUT = 180

class PacoClassifierError(RuntimeError):
    """Raised on any failure talking to paco-classifier-service. Callers
    (tasks_predict.py) catch this broadly and fall back to raw-image stave
    detection rather than failing the whole predict job."""

def classify_stafflines(
    image_bytes: bytes, mime_type: str, timeout: int = DEFAULT_TIMEOUT,
) -> tuple[bytes, bytes]:
    """POSTs one page image to paco-classifier-service's /classify.

    Returns (stafflines_png_bytes, background_png_bytes) — both full-page-
    resolution RGBA PNGs (alpha=0 outside each layer's mask; masked-out
    pixels are also forced to pure white — see paco-classifier-service's
    own _layer_to_rgba_png docstring for why that matters to callers that
    later drop the alpha channel).
    """
    boundary = _uuid.uuid4().hex
    body = bytearray()
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="image"; filename="page.png"\r\n'
    body += f"Content-Type: {mime_type}\r\n\r\n".encode()
    body += image_bytes + b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{PACO_API_URL}/classify", data=bytes(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="ignore")
        try:
            detail = json.loads(detail).get("detail", detail)
        except json.JSONDecodeError:
            pass
        raise PacoClassifierError(
            f"paco-classifier-service rejected the request (HTTP {exc.code}): {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise PacoClassifierError(
            f"paco-classifier-service at {PACO_API_URL} is unreachable: {exc}"
        ) from exc

    return (
        base64.b64decode(payload["stafflines_png_base64"]),
        base64.b64decode(payload["background_png_base64"]),
    )