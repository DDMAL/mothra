"""
Staffline/background layer-separation bridge — proxies to the standalone
paco-classifier-service, which wraps the Paco_classifier submodule's
TensorFlow auto-encoder pixel classifier. Mirrors text_api.py's/
cantus_api.py's pattern: server-to-server call over plain HTTP, config-
driven service URL.

Kept synchronous and dependency-free on purpose — tasks_predict.py calls
this from inside a background threading.Thread (see its own module docs),
not from an async context, and this module must never touch the calling
task's shared psycopg2 connection.

Uses raw http.client rather than urllib.request.urlopen (unlike text_api.py/
cantus_api.py's simpler bridges) so the live connection can be exposed via
`conn_holder` and forcibly aborted from another thread — see
classify_stafflines's docstring. paco-classifier-service's /classify is a
fully synchronous endpoint with no cancellation concept of its own (it
blocks for the whole TensorFlow inference, up to DEFAULT_TIMEOUT seconds),
so this is the only way a job-cancel request can make this call stop
blocking the calling thread promptly instead of running to completion
regardless.
"""
from __future__ import annotations

import base64
import http.client
import json
import socket
import uuid as _uuid
from typing import Optional
from urllib.parse import urlsplit

from config import PACO_API_URL

DEFAULT_TIMEOUT = 180

class PacoClassifierError(RuntimeError):
    """Raised on any failure talking to paco-classifier-service. Callers
    (tasks_predict.py) catch this broadly and fall back to raw-image stave
    detection rather than failing the whole predict job."""

def classify_stafflines(
    image_bytes: bytes,
    mime_type: str,
    timeout: int = DEFAULT_TIMEOUT,
    conn_holder: Optional[dict] = None,
) -> tuple[bytes, bytes]:
    """POSTs one page image to paco-classifier-service's /classify.

    Returns (stafflines_png_bytes, background_png_bytes) — both full-page-
    resolution RGBA PNGs (alpha=0 outside each layer's mask; masked-out
    pixels are also forced to pure white — see paco-classifier-service's
    own _layer_to_rgba_png docstring for why that matters to callers that
    later drop the alpha channel).

    `conn_holder`, if given, is a plain dict this function stores its live
    http.client connection into (`conn_holder["conn"] = conn`) before
    making the request. A caller running this call on a background thread
    can then call `abort_classify_request(conn_holder)` from another
    thread at any point to forcibly interrupt it — whatever's currently
    blocking (connect/request/getresponse/read) raises, this function lets
    that exception propagate as a plain PacoClassifierError, and the
    caller is expected to already know it triggered the abort (see
    tasks_predict.py's _run_medieval_inference).
    """
    boundary = _uuid.uuid4().hex
    body = bytearray()
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="image"; filename="page.png"\r\n'
    body += f"Content-Type: {mime_type}\r\n\r\n".encode()
    body += image_bytes + b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    parsed = urlsplit(PACO_API_URL)
    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    default_port = 443 if parsed.scheme == "https" else 80
    conn = conn_cls(parsed.hostname, parsed.port or default_port, timeout=timeout)
    if conn_holder is not None:
        conn_holder["conn"] = conn

    try:
        conn.request(
            "POST", "/classify", body=bytes(body),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        resp = conn.getresponse()
        raw = resp.read()
    except (OSError, http.client.HTTPException) as exc:
        raise PacoClassifierError(
            f"paco-classifier-service at {PACO_API_URL} is unreachable or the request "
            f"was aborted: {exc}"
        ) from exc
    finally:
        conn.close()

    if resp.status >= 400:
        detail = raw.decode(errors="ignore")
        try:
            detail = json.loads(detail).get("detail", detail)
        except json.JSONDecodeError:
            pass
        raise PacoClassifierError(
            f"paco-classifier-service rejected the request (HTTP {resp.status}): {detail}"
        )

    payload = json.loads(raw.decode())
    return (
        base64.b64decode(payload["stafflines_png_base64"]),
        base64.b64decode(payload["background_png_base64"]),
    )

def abort_classify_request(conn_holder: dict) -> None:
    """Forcibly abort an in-flight classify_stafflines() call, given the
    same `conn_holder` dict passed to it. Safe to call even if the
    connection hasn't been established yet, or has already finished/closed
    on its own — this is a best-effort nudge, not a guarantee the remote
    TensorFlow inference stops (paco-classifier-service has no way to be
    told that; see this module's docstring). `sock.shutdown()` before
    `close()` is what actually unblocks a thread currently parked in
    getresponse()/read() on this connection — closing the fd alone isn't
    reliably enough to interrupt a blocking read on every platform."""
    conn = conn_holder.get("conn")
    if conn is None:
        return
    sock = getattr(conn, "sock", None)
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    try:
        conn.close()
    except OSError:
        pass