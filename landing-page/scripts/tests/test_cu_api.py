"""cu_api.py -- the Cantus Ultimus deposit bridge.

Imports cleanly under CI's minimal dependency set (pytest + Pillow + PyYAML +
staff-finding, see .github/workflows/tests.yml): cu_api.py deliberately imports
no fastapi/pydantic/psycopg2, which is the whole reason it raises its own
CuError instead of HTTPException. Only config.py comes along, and that needs
nothing but PyYAML.

Follows test_staffline_adapter.py's convention: sys.path insert, bare module
import, plain pytest functions, no fixtures/conftest.
"""
import io
import json
import sys
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cu_api  # noqa: E402


# --- helpers -------------------------------------------------------------

class _FakeResponse:
    """Minimal stand-in for urlopen's context manager."""

    def __init__(self, status: int, body: str):
        self.status = status
        self._body = body.encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code: int, body, headers=None):
    msg = Message()
    for key, value in (headers or {}).items():
        msg[key] = value
    payload = body if isinstance(body, str) else json.dumps(body)
    return urllib.error.HTTPError(
        "https://cu.example/api/mei-submissions/",
        code,
        "error",
        msg,
        io.BytesIO(payload.encode("utf-8")),
    )


def _capture(monkeypatch, result):
    """Patch cu_api's urlopen, recording the Request it was handed.

    `result` is either a _FakeResponse to return or an exception to raise.
    """
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["request"] = req
        seen["timeout"] = timeout
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(cu_api.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(cu_api, "CU_DEPOSIT_TOKEN", "tok-abc123")
    monkeypatch.setattr(cu_api, "CU_API_URL", "https://cu.example")
    return seen


_RECORD = {
    "id": 7,
    "manuscript_id": 123723,
    "folio_number": "001r",
    "status": "PENDING",
    "status_display": "Pending review",
    "review_note": "",
    "comment": "",
    "submitter": "someone",
    "submitted_at": "2026-09-15T10:00:00Z",
    "reviewed_at": None,
}


def _submit(**kwargs):
    defaults = dict(
        manuscript_id=123723,
        folio_number="001r",
        mei="<mei/>",
        submitter="someone",
    )
    defaults.update(kwargs)
    return cu_api.submit_mei(**defaults)


# --- the credential ------------------------------------------------------

def test_submit_sends_the_deposit_token_as_a_token_header(monkeypatch):
    """The whole point of the module: CU authenticates Mothra-the-application
    by a DRF token, and this is the codebase's first outbound credential."""
    seen = _capture(monkeypatch, _FakeResponse(201, json.dumps(_RECORD)))
    _submit()
    req = seen["request"]
    assert req.get_header("Authorization") == "Token tok-abc123"
    assert req.get_header("Content-type") == "application/json"
    assert req.get_method() == "POST"
    assert req.full_url == "https://cu.example/api/mei-submissions/"


def test_submit_sends_cus_exact_field_names(monkeypatch):
    """MEISubmissionCreateSerializer is snake_case; Mothra is camelCase
    everywhere else, so this is an easy and silent thing to get wrong."""
    seen = _capture(monkeypatch, _FakeResponse(201, json.dumps(_RECORD)))
    _submit(comment="from project Foo")
    body = json.loads(seen["request"].data.decode())
    assert body == {
        "manuscript_id": 123723,
        "folio_number": "001r",
        "mei": "<mei/>",
        "submitter": "someone",
        "comment": "from project Foo",
    }


def test_unset_token_is_reported_as_not_configured_without_calling_out(monkeypatch):
    """A deployment that never submits is valid, so the token is not required
    at import. The failure must name the fix and must not reach the network."""
    called = []
    monkeypatch.setattr(
        cu_api.urllib.request, "urlopen",
        lambda *a, **k: called.append(1),
    )
    monkeypatch.setattr(cu_api, "CU_DEPOSIT_TOKEN", "")
    with pytest.raises(cu_api.CuError) as exc:
        _submit()
    assert exc.value.category == cu_api.CATEGORY_NOT_CONFIGURED
    assert "create_deposit_user" in str(exc.value)
    assert not called, "must not attempt a request without a token"


# --- 201 vs 200, the idempotency distinction -----------------------------

def test_201_is_reported_as_a_new_submission(monkeypatch):
    _capture(monkeypatch, _FakeResponse(201, json.dumps(_RECORD)))
    assert _submit()["alreadyPending"] is False


def test_200_is_reported_as_already_pending(monkeypatch):
    """CU answers 200 with the EXISTING review when byte-identical content is
    already pending. Both are successes, but reporting the second as a fresh
    submission would make a re-run look like it doubled the review queue."""
    _capture(monkeypatch, _FakeResponse(200, json.dumps(_RECORD)))
    result = _submit()
    assert result["alreadyPending"] is True
    assert result["id"] == 7


def test_submit_passes_through_cus_record_fields(monkeypatch):
    _capture(monkeypatch, _FakeResponse(201, json.dumps(_RECORD)))
    result = _submit()
    assert result["status"] == "PENDING"
    assert result["folio_number"] == "001r"


# --- rejections ----------------------------------------------------------

def test_400_surfaces_cus_own_detail(monkeypatch):
    _capture(monkeypatch, _http_error(400, {"detail": "No manuscript with id 9."}))
    with pytest.raises(cu_api.CuError) as exc:
        _submit()
    assert exc.value.category == cu_api.CATEGORY_REJECTED
    assert str(exc.value) == "No manuscript with id 9."


def test_400_serializer_errors_keep_the_field_name(monkeypatch):
    """CU's most common rejection is a folio problem, and "folio_number: ..."
    is far more useful than the bare message when one page of a submit-all
    fails among many."""
    _capture(monkeypatch, _http_error(
        400, {"folio_number": ["Manuscript 1 has no folio 999r."]},
    ))
    with pytest.raises(cu_api.CuError) as exc:
        _submit()
    assert "folio_number: Manuscript 1 has no folio 999r." == str(exc.value)


def test_non_json_error_body_falls_back_to_the_raw_text(monkeypatch):
    """An nginx/proxy error page is not JSON; it must not become a traceback."""
    _capture(monkeypatch, _http_error(502, "<html>502 Bad Gateway</html>"))
    with pytest.raises(cu_api.CuError) as exc:
        _submit()
    assert exc.value.category == cu_api.CATEGORY_SERVICE_ERROR
    assert "502 Bad Gateway" in str(exc.value)


def test_401_and_403_are_a_deployment_fault_not_a_user_error(monkeypatch):
    for code in (401, 403):
        _capture(monkeypatch, _http_error(code, {"detail": "Invalid token."}))
        with pytest.raises(cu_api.CuError) as exc:
            _submit()
        assert exc.value.category == cu_api.CATEGORY_UNAUTHORIZED
        assert "token" in str(exc.value).lower()


# --- the throttle --------------------------------------------------------

def test_429_reports_the_retry_after_window(monkeypatch):
    """CU throttles deposits at 300/hour, shared across every Mothra user, so
    a large submit-all can genuinely exhaust it."""
    _capture(monkeypatch, _http_error(
        429, {"detail": "Request was throttled."}, {"Retry-After": "612"},
    ))
    with pytest.raises(cu_api.CuError) as exc:
        _submit()
    assert exc.value.category == cu_api.CATEGORY_THROTTLED
    assert exc.value.retry_after == 612
    assert "612s" in str(exc.value)


def test_429_without_a_parseable_retry_after_still_reports_throttling(monkeypatch):
    """Retry-After may legally be an HTTP-date, which we only use to phrase a
    message -- an unparseable value must not mask the throttle itself."""
    _capture(monkeypatch, _http_error(
        429, {"detail": "throttled"}, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"},
    ))
    with pytest.raises(cu_api.CuError) as exc:
        _submit()
    assert exc.value.category == cu_api.CATEGORY_THROTTLED
    assert exc.value.retry_after is None


# --- transport failures --------------------------------------------------

def test_unreachable_names_the_url_it_tried(monkeypatch):
    _capture(monkeypatch, urllib.error.URLError("Name or service not known"))
    with pytest.raises(cu_api.CuError) as exc:
        _submit()
    assert exc.value.category == cu_api.CATEGORY_UNREACHABLE
    assert "https://cu.example" in str(exc.value)


def test_timeout_is_distinct_from_unreachable(monkeypatch):
    """"CU is slow" and "CU is gone or misaddressed" call for different
    responses from whoever is debugging."""
    _capture(monkeypatch, urllib.error.URLError(TimeoutError("timed out")))
    with pytest.raises(cu_api.CuError) as exc:
        _submit()
    assert exc.value.category == cu_api.CATEGORY_TIMEOUT


def test_a_bare_timeouterror_is_also_caught(monkeypatch):
    """urlopen raises socket.timeout directly in some paths rather than
    wrapping it in URLError."""
    _capture(monkeypatch, TimeoutError("timed out"))
    with pytest.raises(cu_api.CuError) as exc:
        _submit()
    assert exc.value.category == cu_api.CATEGORY_TIMEOUT


def test_a_2xx_that_is_not_json_is_malformed_not_a_crash(monkeypatch):
    _capture(monkeypatch, _FakeResponse(201, "<html>hello</html>"))
    with pytest.raises(cu_api.CuError) as exc:
        _submit()
    assert exc.value.category == cu_api.CATEGORY_MALFORMED_RESPONSE


def test_an_explicit_timeout_is_always_passed_to_urlopen(monkeypatch):
    """urlopen's default is no timeout at all, which would let one dead
    connection hold a request open indefinitely."""
    seen = _capture(monkeypatch, _FakeResponse(201, json.dumps(_RECORD)))
    _submit()
    assert seen["timeout"] == cu_api.DEFAULT_TIMEOUT


# --- the read endpoints --------------------------------------------------

def test_list_submissions_filters_by_submitter(monkeypatch):
    seen = _capture(monkeypatch, _FakeResponse(200, json.dumps([_RECORD])))
    records = cu_api.list_submissions("someone")
    assert records == [_RECORD]
    assert seen["request"].full_url == (
        "https://cu.example/api/mei-submissions/?submitter=someone"
    )
    assert seen["request"].get_method() == "GET"


def test_a_submitter_name_is_url_encoded(monkeypatch):
    """Mothra usernames are user-chosen and are not restricted to URL-safe
    characters."""
    seen = _capture(monkeypatch, _FakeResponse(200, "[]"))
    cu_api.list_submissions("a b&c")
    assert "submitter=a+b%26c" in seen["request"].full_url


def test_a_paginated_collection_is_unwrapped(monkeypatch):
    """DRF returns a bare array unless pagination is configured. Which of CU's
    endpoints paginate is CU's decision to change, and guessing wrong would
    silently show an empty picker rather than failing."""
    _capture(monkeypatch, _FakeResponse(
        200, json.dumps({"count": 1, "results": [{"id": 1}]}),
    ))
    assert cu_api.list_manuscripts() == [{"id": 1}]


def test_an_empty_body_is_an_empty_collection(monkeypatch):
    _capture(monkeypatch, _FakeResponse(200, ""))
    assert cu_api.list_manuscripts() == []


def test_list_folios_targets_cus_solr_backed_folio_set(monkeypatch):
    seen = _capture(monkeypatch, _FakeResponse(200, json.dumps(
        [{"number": "001r", "image_uri": "https://img/1"}],
    )))
    folios = cu_api.list_folios(123723)
    assert folios[0]["image_uri"] == "https://img/1"
    assert seen["request"].full_url == (
        "https://cu.example/folio-set/manuscript/123723/"
    )


def test_a_collection_that_is_not_a_list_is_malformed(monkeypatch):
    _capture(monkeypatch, _FakeResponse(200, json.dumps({"unexpected": "shape"})))
    with pytest.raises(cu_api.CuError) as exc:
        cu_api.list_manuscripts()
    assert exc.value.category == cu_api.CATEGORY_MALFORMED_RESPONSE


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
