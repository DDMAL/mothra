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
classify_stafflines's docstring. Aborting here is also what makes
cancellation effective server-side, not just locally: paco-classifier-
service's /classify polls for exactly this disconnect (its own
request.is_disconnected() loop) and uses it to stop the TensorFlow
inference between patches via recognition_engine.process_image_msae()'s
should_cancel param, instead of running the whole page to completion for a
result nobody will read. Without this abort, that /classify call would
otherwise block the calling thread for up to DEFAULT_TIMEOUT seconds
regardless of whether Mothra's own job was already cancelled.
"""
from __future__ import annotations

import base64
import http.client
import json
import logging
import socket
import uuid as _uuid
from typing import Optional
from urllib.parse import urlsplit

from config import PACO_API_URL

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 180

# Small, flat failure-category taxonomy — not meant to be exhaustive, just
# enough for tasks_predict.py to log/report/persist *why* a fallback
# happened instead of only *that* one happened.
CATEGORY_TIMEOUT = "timeout"
CATEGORY_UNREACHABLE = "unreachable"
CATEGORY_HTTP_ERROR = "http_error"
CATEGORY_MALFORMED_RESPONSE = "malformed_response"

class PacoClassifierError(RuntimeError):
    """Raised on any failure talking to paco-classifier-service. Callers
    (tasks_predict.py) catch this broadly and fall back to raw-image stave
    detection rather than failing the whole predict job. `category` is one
    of the CATEGORY_* constants above, read by tasks_predict.py to build a
    short human-readable reason it logs/publishes/persists — see
    _classifier_error_reason() there."""

    def __init__(self, message: str, category: str = CATEGORY_UNREACHABLE):
        super().__init__(message)
        self.category = category

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
    except socket.timeout as exc:
        # socket.timeout (== TimeoutError) is itself an OSError subclass, so
        # this must be caught ahead of the broader OSError branch below to
        # be distinguishable at all.
        logger.warning(
            "paco-classifier-service at %s timed out after %ss: %s",
            PACO_API_URL, timeout, exc, exc_info=True,
        )
        raise PacoClassifierError(
            f"paco-classifier-service at {PACO_API_URL} timed out after {timeout}s: {exc}",
            category=CATEGORY_TIMEOUT,
        ) from exc
    except (OSError, http.client.HTTPException) as exc:
        logger.warning(
            "paco-classifier-service at %s is unreachable or the request was aborted: %s",
            PACO_API_URL, exc, exc_info=True,
        )
        raise PacoClassifierError(
            f"paco-classifier-service at {PACO_API_URL} is unreachable or the request "
            f"was aborted: {exc}",
            category=CATEGORY_UNREACHABLE,
        ) from exc
    finally:
        conn.close()

    # Only 2xx is success -- a 3xx (e.g. a redirect the client never follows)
    # used to fall through to the payload-parsing block below, where it
    # failed as a "malformed response" instead of the http_error it actually
    # is. CodeRabbit finding on #252.
    if not (200 <= resp.status < 300):
        detail = raw.decode(errors="ignore")
        try:
            parsed_detail = json.loads(detail)
        except json.JSONDecodeError:
            pass
        else:
            # A JSON body that isn't an object (a bare list, string, number,
            # or null -- all valid JSON) has no .get(); only dicts do.
            if isinstance(parsed_detail, dict):
                detail = parsed_detail.get("detail", detail)
        logger.warning(
            "paco-classifier-service rejected the request (HTTP %s): %s",
            resp.status, detail,
        )
        raise PacoClassifierError(
            f"paco-classifier-service rejected the request (HTTP {resp.status}): {detail}",
            category=CATEGORY_HTTP_ERROR,
        )

    try:
        payload = json.loads(raw.decode())
        return (
            base64.b64decode(payload["stafflines_png_base64"], validate=True),
            base64.b64decode(payload["background_png_base64"], validate=True),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        # ValueError also covers base64.b64decode's binascii.Error (incl. what
        # validate=True raises for non-base64 characters, instead of silently
        # ignoring them). TypeError covers a payload that's valid JSON but not
        # the expected shape -- a bare list/string/null (payload["..."] raises
        # TypeError, not KeyError, on a list; base64.b64decode(None, ...) also
        # raises TypeError) rather than the dict-with-missing-key case KeyError
        # alone catches. Previously unguarded -- a malformed 2xx response
        # raised a raw, uncategorized exception here.
        logger.warning(
            "paco-classifier-service returned a malformed response: %s", exc, exc_info=True,
        )
        raise PacoClassifierError(
            f"paco-classifier-service returned a malformed response: {exc}",
            category=CATEGORY_MALFORMED_RESPONSE,
        ) from exc

def abort_classify_request(conn_holder: dict) -> None:
    """Forcibly abort an in-flight classify_stafflines() call, given the
    same `conn_holder` dict passed to it. Safe to call even if the
    connection hasn't been established yet, or has already finished/closed
    on its own. This does double duty: it unblocks the LOCAL thread
    immediately, and the resulting disconnect is also what
    paco-classifier-service's /classify polls for to cancel the remote
    TensorFlow inference cooperatively (see this module's docstring) — so
    this is a real cancellation trigger for both sides now, not just a
    local nudge. `sock.shutdown()` before `close()` is what actually
    unblocks a thread currently parked in getresponse()/read() on this
    connection — closing the fd alone isn't reliably enough to interrupt a
    blocking read on every platform."""
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