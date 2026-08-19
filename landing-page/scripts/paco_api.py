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

/classify streams its response as SSE-shaped `data: {...}\n\n` lines rather
than one blocking JSON body (mothra#247 follow-up) — a "progress" event per
sliding-window row (see recognition_engine.process_image_msae's
progress_callback param) so the real "row N of total" signal the TF loop
already produces in-process can reach tasks_predict.py's job-progress
reporting instead of only a time-based guess, followed by exactly one
terminal "result" or "error" event. http.client's HTTPResponse is read
line-by-line (`resp.readline()`) for this rather than one `resp.read()`,
same underlying socket, no new dependency.
"""
from __future__ import annotations

import base64
import http.client
import json
import logging
import socket
import uuid as _uuid
from typing import Callable, Optional
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
# The service committed to a 200 stream and then reported its OWN failure
# mid-computation (a "type": "error" SSE event -- e.g. a TF inference
# exception) -- distinct from CATEGORY_HTTP_ERROR (which means the request
# itself was rejected before any streaming started). Added when the SSE
# streaming protocol (mothra#247) was reconciled with this categorization
# system: neither existed at the same time before, so there was no prior
# category for this case to reuse.
CATEGORY_SERVICE_ERROR = "service_error"

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
    progress_callback: Optional[Callable[[int, int], None]] = None,
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

    `progress_callback`, if given, is called as progress_callback(row,
    total) once per "progress" SSE event the service emits while the
    sliding-window pass is running (mothra#247 follow-up) — real,
    deterministic progress from the TF inference loop itself, not a
    time-based guess. Optional and defaults to None so every existing
    caller keeps working unchanged.
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

        # Only 2xx is success -- a 3xx (e.g. a redirect the client never
        # follows) used to fall through to the streaming block below, where
        # it read as an empty/garbage stream and surfaced as a generic
        # "stream ended without a result" instead of the http_error it
        # actually is. CodeRabbit finding on #252.
        if not (200 <= resp.status < 300):
            # A pre-stream validation failure (bad upload, oversized image) —
            # the service raises HTTPException for these BEFORE it commits to
            # a streaming response, so this is still a plain JSON error body,
            # read in one shot rather than line-by-line.
            raw = resp.read()
            detail = raw.decode(errors="ignore")
            try:
                parsed_detail = json.loads(detail)
            except json.JSONDecodeError:
                pass
            else:
                # A JSON body that isn't an object (a bare list, string,
                # number, or null -- all valid JSON) has no .get(); only
                # dicts do. CodeRabbit finding on #252.
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

        # A 200 response is a stream of SSE `data: {...}\n\n` lines: zero or
        # more "progress" events, then exactly one terminal "result" or
        # "error" event. Read line-by-line rather than resp.read() so
        # progress events are visible as they arrive instead of only once
        # the whole page finishes classifying.
        result_payload = None
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data: "):
                continue
            try:
                ev = json.loads(line[len("data: "):])
            except json.JSONDecodeError as exc:
                logger.warning(
                    "paco-classifier-service sent an unparseable SSE line: %s", exc, exc_info=True,
                )
                raise PacoClassifierError(
                    f"paco-classifier-service sent an unparseable SSE line: {exc}",
                    category=CATEGORY_MALFORMED_RESPONSE,
                ) from exc
            ev_type = ev.get("type")
            if ev_type == "progress":
                if progress_callback is not None:
                    progress_callback(ev.get("row", 0), ev.get("total", 0))
            elif ev_type == "error":
                # The service committed to the stream (status already sent
                # as 200) and then failed mid-computation -- a distinct
                # failure mode from the pre-stream HTTP-error case above,
                # which never got this far.
                detail = ev.get("detail", "unknown error")
                logger.warning(
                    "paco-classifier-service reported an error: %s", detail,
                )
                raise PacoClassifierError(
                    f"paco-classifier-service reported an error: {detail}",
                    category=CATEGORY_SERVICE_ERROR,
                )
            elif ev_type == "result":
                result_payload = ev
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

    if result_payload is None:
        raise PacoClassifierError(
            "paco-classifier-service's response stream ended without a result",
            category=CATEGORY_MALFORMED_RESPONSE,
        )
    try:
        return (
            base64.b64decode(result_payload["stafflines_png_base64"], validate=True),
            base64.b64decode(result_payload["background_png_base64"], validate=True),
        )
    except (KeyError, TypeError, ValueError) as exc:
        # ValueError also covers base64.b64decode's binascii.Error (incl. what
        # validate=True raises for non-base64 characters, instead of silently
        # ignoring them). TypeError covers a result event that's valid JSON
        # but not the expected shape -- e.g. one of these fields is a list or
        # null (result_payload["..."] raises TypeError, not KeyError, when
        # result_payload itself isn't a dict; base64.b64decode(None, ...) also
        # raises TypeError) rather than the dict-with-missing-key case KeyError
        # alone catches. Previously unguarded -- a malformed result event
        # raised a raw, uncategorized exception here. CodeRabbit finding on #252.
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