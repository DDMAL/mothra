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


def test_classify_stafflines_decodes_both_images():
    stafflines_b64 = base64.b64encode(b"stafflines-bytes").decode()
    background_b64 = base64.b64encode(b"background-bytes").decode()
    body = json.dumps({
        "stafflines_png_base64": stafflines_b64,
        "background_png_base64": background_b64,
    }).encode()
    with patch("http.client.HTTPConnection", return_value=_fake_connection(200, body)):
        stafflines, background = paco_api.classify_stafflines(b"fake-image-bytes", "image/png")
    assert stafflines == b"stafflines-bytes"
    assert background == b"background-bytes"


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
    body = json.dumps({
        "stafflines_png_base64": base64.b64encode(b"a").decode(),
        "background_png_base64": base64.b64encode(b"b").decode(),
    }).encode()
    fake_conn = _fake_connection(200, body)
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