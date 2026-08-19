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


def test_classify_stafflines_treats_3xx_as_http_error():
    """A redirect must be categorized http_error, not fall through to the
    payload-parsing block below and get misreported as a malformed response
    (CodeRabbit finding on #252 -- only 200-299 is success)."""
    body = b'{"detail": "moved"}'
    with patch("http.client.HTTPConnection", return_value=_fake_connection(302, body)):
        try:
            paco_api.classify_stafflines(b"x", "image/png")
            assert False, "expected PacoClassifierError"
        except paco_api.PacoClassifierError as e:
            assert e.category == paco_api.CATEGORY_HTTP_ERROR
            assert "moved" in str(e)


def test_classify_stafflines_http_error_with_non_dict_json_body():
    """An error body that's valid JSON but not an object (a bare list here)
    used to raise AttributeError from .get() on the parsed value instead of
    PacoClassifierError -- CodeRabbit finding on #252."""
    body = b'["nginx", "502 Bad Gateway"]'
    with patch("http.client.HTTPConnection", return_value=_fake_connection(502, body)):
        try:
            paco_api.classify_stafflines(b"x", "image/png")
            assert False, "expected PacoClassifierError"
        except paco_api.PacoClassifierError as e:
            assert e.category == paco_api.CATEGORY_HTTP_ERROR


def test_classify_stafflines_malformed_payload_is_a_list_not_a_dict():
    """A 2xx body that's valid JSON but not the expected object shape used to
    raise a raw TypeError (payload["..."] on a list) instead of
    PacoClassifierError -- CodeRabbit finding on #252."""
    body = b'["unexpected", "shape"]'
    with patch("http.client.HTTPConnection", return_value=_fake_connection(200, body)):
        try:
            paco_api.classify_stafflines(b"x", "image/png")
            assert False, "expected PacoClassifierError"
        except paco_api.PacoClassifierError as e:
            assert e.category == paco_api.CATEGORY_MALFORMED_RESPONSE


def test_classify_stafflines_rejects_non_base64_with_validate():
    """base64.b64decode(..., validate=True) rejects non-alphabet characters
    instead of silently ignoring them -- CodeRabbit finding on #252."""
    body = json.dumps({
        "stafflines_png_base64": "not-valid-base64!!!",
        "background_png_base64": base64.b64encode(b"b").decode(),
    }).encode()
    with patch("http.client.HTTPConnection", return_value=_fake_connection(200, body)):
        try:
            paco_api.classify_stafflines(b"x", "image/png")
            assert False, "expected PacoClassifierError"
        except paco_api.PacoClassifierError as e:
            assert e.category == paco_api.CATEGORY_MALFORMED_RESPONSE


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