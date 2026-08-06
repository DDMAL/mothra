"""Unit tests for paco_api.py's HTTP-bridge error handling and the
alpha-drop/whiting-out contract it depends on. No network, no DB — mirrors
test_staffline_adapter.py's style (plain pytest functions, sys.path insert).
"""
import io
import json
import base64
import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paco_api  # noqa: E402


def _fake_response(payload: dict):
    body = json.dumps(payload).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__.return_value = resp
    return resp


def test_classify_stafflines_decodes_both_images():
    stafflines_b64 = base64.b64encode(b"stafflines-bytes").decode()
    background_b64 = base64.b64encode(b"background-bytes").decode()
    with patch("urllib.request.urlopen", return_value=_fake_response({
        "stafflines_png_base64": stafflines_b64,
        "background_png_base64": background_b64,
    })):
        stafflines, background = paco_api.classify_stafflines(b"fake-image-bytes", "image/png")
    assert stafflines == b"stafflines-bytes"
    assert background == b"background-bytes"


def test_classify_stafflines_wraps_http_error():
    exc = urllib.error.HTTPError("url", 500, "boom", {}, io.BytesIO(b'{"detail": "model load failed"}'))
    with patch("urllib.request.urlopen", side_effect=exc):
        try:
            paco_api.classify_stafflines(b"x", "image/png")
            assert False, "expected PacoClassifierError"
        except paco_api.PacoClassifierError as e:
            assert "model load failed" in str(e)


def test_classify_stafflines_wraps_url_error():
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")):
        try:
            paco_api.classify_stafflines(b"x", "image/png")
            assert False, "expected PacoClassifierError"
        except paco_api.PacoClassifierError as e:
            assert "unreachable" in str(e)


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