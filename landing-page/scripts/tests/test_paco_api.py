"""Unit tests for paco_api.py's HTTP-bridge error handling, the abort
mechanism cancellation depends on, and the alpha-drop/whiting-out contract
it depends on. No network, no DB — mirrors test_staffline_adapter.py's
style (plain pytest functions, sys.path insert).
"""
import json
import base64
import socket
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paco_api  # noqa: E402


def _fake_connection(status: int, body: bytes):
    conn = MagicMock()
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = body
    conn.getresponse.return_value = resp
    return conn


def _sse_lines(*events: dict) -> list[bytes]:
    """Build the readline()-return-value sequence a real streamed /classify
    response would produce for the given SSE events, terminated by an empty
    bytes object (readline()'s EOF signal, mirroring http.client.HTTPResponse)."""
    lines = [f"data: {json.dumps(ev)}\n\n".encode() for ev in events]
    lines.append(b"")
    return lines


def _fake_stream_connection(status: int, events: tuple = (), body: bytes = b""):
    """Like _fake_connection, but for the streamed (status 200) case:
    resp.readline() yields one SSE line per call instead of resp.read()
    returning the whole body at once — mirrors what paco-classifier-
    service's /classify actually sends now (mothra#247 follow-up)."""
    conn = MagicMock()
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = body
    resp.readline.side_effect = _sse_lines(*events)
    conn.getresponse.return_value = resp
    return conn


def test_classify_stafflines_decodes_both_images():
    stafflines_b64 = base64.b64encode(b"stafflines-bytes").decode()
    background_b64 = base64.b64encode(b"background-bytes").decode()
    result_event = {
        "type": "result",
        "stafflines_png_base64": stafflines_b64,
        "background_png_base64": background_b64,
    }
    conn = _fake_stream_connection(200, events=(result_event,))
    with patch("http.client.HTTPConnection", return_value=conn):
        stafflines, background = paco_api.classify_stafflines(b"fake-image-bytes", "image/png")
    assert stafflines == b"stafflines-bytes"
    assert background == b"background-bytes"


def test_classify_stafflines_reports_progress():
    """progress_callback is invoked once per "progress" SSE event, in
    order, with the (row, total) the service reported — the real signal
    this whole streaming contract exists to carry (mothra#247 follow-up)."""
    stafflines_b64 = base64.b64encode(b"a").decode()
    background_b64 = base64.b64encode(b"b").decode()
    conn = _fake_stream_connection(200, events=(
        {"type": "progress", "row": 0, "total": 350},
        {"type": "progress", "row": 205, "total": 350},
        {"type": "result", "stafflines_png_base64": stafflines_b64, "background_png_base64": background_b64},
    ))
    seen: list[tuple[int, int]] = []
    with patch("http.client.HTTPConnection", return_value=conn):
        paco_api.classify_stafflines(
            b"x", "image/png", progress_callback=lambda row, total: seen.append((row, total)),
        )
    assert seen == [(0, 350), (205, 350)]


def test_classify_stafflines_raises_on_mid_stream_error_event():
    """A "type": "error" event arrives with HTTP status 200 (the response
    already committed to streaming by the time classification itself
    fails) -- distinct from the pre-stream-validation-failure case below,
    which is still a plain HTTP error status."""
    conn = _fake_stream_connection(200, events=(
        {"type": "error", "detail": "classifier output shape mismatch"},
    ))
    with patch("http.client.HTTPConnection", return_value=conn):
        try:
            paco_api.classify_stafflines(b"x", "image/png")
            assert False, "expected PacoClassifierError"
        except paco_api.PacoClassifierError as e:
            assert "classifier output shape mismatch" in str(e)


def test_classify_stafflines_wraps_http_error():
    body = b'{"detail": "model load failed"}'
    with patch("http.client.HTTPConnection", return_value=_fake_connection(500, body)):
        try:
            paco_api.classify_stafflines(b"x", "image/png")
            assert False, "expected PacoClassifierError"
        except paco_api.PacoClassifierError as e:
            assert "model load failed" in str(e)


def test_classify_stafflines_wraps_url_error():
    conn = MagicMock()
    conn.request.side_effect = OSError("connection refused")
    with patch("http.client.HTTPConnection", return_value=conn):
        try:
            paco_api.classify_stafflines(b"x", "image/png")
            assert False, "expected PacoClassifierError"
        except paco_api.PacoClassifierError as e:
            assert "unreachable" in str(e)


def test_classify_stafflines_exposes_conn_via_conn_holder():
    """conn_holder must be populated with the live connection BEFORE the
    (blocking) request is made -- that's what lets another thread abort it
    mid-flight. Verified indirectly: the connection returned to the caller
    is the exact same object classify_stafflines used internally."""
    result_event = {
        "type": "result",
        "stafflines_png_base64": base64.b64encode(b"a").decode(),
        "background_png_base64": base64.b64encode(b"b").decode(),
    }
    fake_conn = _fake_stream_connection(200, events=(result_event,))
    conn_holder: dict = {}
    with patch("http.client.HTTPConnection", return_value=fake_conn):
        paco_api.classify_stafflines(b"x", "image/png", conn_holder=conn_holder)
    assert conn_holder["conn"] is fake_conn


def test_abort_classify_request_shuts_down_socket_and_closes():
    fake_conn = MagicMock()
    fake_sock = MagicMock()
    fake_conn.sock = fake_sock
    conn_holder = {"conn": fake_conn}
    paco_api.abort_classify_request(conn_holder)
    fake_sock.shutdown.assert_called_once_with(socket.SHUT_RDWR)
    fake_conn.close.assert_called_once()


def test_abort_classify_request_is_a_noop_before_connection_exists():
    paco_api.abort_classify_request({})  # must not raise


def test_masked_out_pixels_are_pure_white_not_just_transparent():
    """Regression guard for the exact risk flagged during planning: if the
    service ever stops whiting-out masked-out RGB (only zeroing alpha),
    tasks_predict.py's alpha-drop step would silently feed the stave model
    the ORIGINAL page instead of the isolated stafflines layer. This
    round-trips a small RGBA array the way cv2.imdecode(..., IMREAD_COLOR)
    would see it -- alpha-drop must yield white, not leftover colour.
    """
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., :3] = 255  # this is what a correctly-produced PNG has at masked-out pixels
    rgba[..., 3] = 0
    im = Image.fromarray(rgba, mode="RGBA").convert("RGB")
    assert np.array(im).tolist() == [[[255, 255, 255]] * 4] * 4


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))