"""Bridge to Cantus Ultimus's MEI deposit inbox.

CU (DDMAL/cantus) exposes `POST`/`GET /api/mei-submissions/` — a DRF
`ListCreateAPIView` that files one folio's MEI as a PENDING record for a human
admin to review. Nothing a submission carries reaches CU's public site until an
admin publishes it, so this module can only ever fill a review queue.

Two deliberate shape choices, both worth keeping:

1. **No fastapi import.** This module raises its own `CuError` rather than
   `HTTPException`, and `cu_submission_api.py` translates the category into a
   status code at the edge. That is what makes it testable: CI installs only
   pytest/Pillow/PyYAML/staff-finding (.github/workflows/tests.yml), so a module
   that imports fastapi cannot be imported by a test at all. Same split, and the
   same reason, as neon_manifest.py vs. the routers that use it.
2. **urllib, not http.client.** paco_api.py drops to raw http.client because it
   needs to expose a live socket for cross-thread abort mid-inference. Nothing
   here streams or needs aborting — the payload is one document and the reply is
   one JSON object — so this follows cantus_api.py's simpler urllib shape.

This is the first place in the codebase that sends a credential *outbound*.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from config import CU_API_URL, CU_DEPOSIT_TOKEN

# CU's own work per request is an XML parse and an INSERT. 30s is generous for
# that. Keep it modest on purpose: urlopen's timeout is a PER-READ socket
# timeout, not a hard deadline (see text_api.py's _stream_multipart, where a
# 600s value let an unreachable peer tie up a worker thread for ten minutes
# before Python raised), so this bounds how long one dead connection can hold
# a request open.
DEFAULT_TIMEOUT = 30

# Flat failure taxonomy, mirroring paco_api.py's CATEGORY_* convention. The
# router maps these to status codes; the frontend shows the message.
CATEGORY_NOT_CONFIGURED = "not_configured"
CATEGORY_TIMEOUT = "timeout"
CATEGORY_UNREACHABLE = "unreachable"
CATEGORY_UNAUTHORIZED = "unauthorized"
CATEGORY_REJECTED = "rejected"
CATEGORY_THROTTLED = "throttled"
CATEGORY_SERVICE_ERROR = "service_error"
CATEGORY_MALFORMED_RESPONSE = "malformed_response"


class CuError(RuntimeError):
    """Any failure talking to Cantus Ultimus.

    `category` is one of the CATEGORY_* constants. `retry_after` is set only
    for CATEGORY_THROTTLED, carrying CU's own Retry-After seconds when it sent
    one, so the caller can say when to try again instead of just that it failed.
    """

    def __init__(
        self,
        message: str,
        category: str = CATEGORY_UNREACHABLE,
        retry_after: Optional[int] = None,
    ):
        super().__init__(message)
        self.category = category
        self.retry_after = retry_after


def _require_token() -> str:
    if not CU_DEPOSIT_TOKEN:
        raise CuError(
            "CU_DEPOSIT_TOKEN is not set, so this deployment cannot submit to "
            "Cantus Ultimus. Generate a token on CU with "
            "`manage.py create_deposit_user` and set it in the backend's "
            "environment.",
            CATEGORY_NOT_CONFIGURED,
        )
    return CU_DEPOSIT_TOKEN


def _detail_from_error(exc: urllib.error.HTTPError) -> str:
    """CU's own explanation of a rejection, which is the actionable part.

    DRF serializer errors come back as {"field": ["message", ...]} and plain
    APIException as {"detail": "..."}; both are worth surfacing verbatim rather
    than flattening to "request failed". Falls back to the raw body, then to
    the status line, when the response is not JSON at all (an nginx error page,
    say).
    """
    try:
        raw = exc.read().decode(errors="ignore")
    except Exception:
        return f"HTTP {exc.code}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip() or f"HTTP {exc.code}"
    if isinstance(parsed, dict):
        if "detail" in parsed:
            return str(parsed["detail"])
        # Serializer errors: join every field's messages, keeping the field
        # name, since "folio_number: ..." is far more useful than the message
        # alone when a submit-all reports one failure among many.
        parts = []
        for field, messages in parsed.items():
            if isinstance(messages, (list, tuple)):
                parts.append(f"{field}: {'; '.join(str(m) for m in messages)}")
            else:
                parts.append(f"{field}: {messages}")
        if parts:
            return " | ".join(parts)
    return raw.strip() or f"HTTP {exc.code}"


def _retry_after_seconds(exc: urllib.error.HTTPError) -> Optional[int]:
    raw = None
    try:
        raw = exc.headers.get("Retry-After")
    except Exception:
        return None
    if not raw:
        return None
    try:
        return int(float(str(raw).strip()))
    except (TypeError, ValueError):
        # Retry-After may legally be an HTTP-date. We only use this to phrase a
        # message, so an unparseable value is simply not shown.
        return None


def _request(
    method: str,
    path: str,
    *,
    payload: Optional[dict] = None,
    query: Optional[dict] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[int, Any]:
    """One authenticated call to CU, returning (status_code, decoded JSON).

    The status is returned rather than discarded because CU's deposit endpoint
    uses 201-vs-200 to mean "filed" vs "identical submission already pending",
    which is the one case where the body alone cannot tell the caller what
    happened.

    Raises CuError for every failure mode, so callers never see a urllib
    exception.
    """
    token = _require_token()
    url = f"{CU_API_URL}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"

    data = None
    req = urllib.request.Request(url, method=method)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        req.data = data
        req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("Authorization", f"Token {token}")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode(errors="ignore")
            status = resp.status
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise CuError(
                "Cantus Ultimus rejected this deployment's deposit token "
                f"({exc.code}). It may be wrong, revoked, or missing the "
                "add_meisubmission/view_meisubmission permissions.",
                CATEGORY_UNAUTHORIZED,
            ) from exc
        if exc.code == 429:
            retry_after = _retry_after_seconds(exc)
            when = f" Try again in {retry_after}s." if retry_after else ""
            raise CuError(
                "Cantus Ultimus is rate-limiting deposits (its cap is 300 per "
                f"hour, shared by everyone submitting from Mothra).{when}",
                CATEGORY_THROTTLED,
                retry_after=retry_after,
            ) from exc
        if 400 <= exc.code < 500:
            raise CuError(_detail_from_error(exc), CATEGORY_REJECTED) from exc
        raise CuError(
            f"Cantus Ultimus returned {exc.code}: {_detail_from_error(exc)}",
            CATEGORY_SERVICE_ERROR,
        ) from exc
    except urllib.error.URLError as exc:
        # socket.timeout arrives wrapped in URLError for urlopen, and is worth
        # distinguishing from "no route to host" -- one means CU is slow, the
        # other that it is gone or misaddressed.
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, TimeoutError):
            raise CuError(
                f"Cantus Ultimus at {CU_API_URL} did not respond within "
                f"{timeout}s.",
                CATEGORY_TIMEOUT,
            ) from exc
        raise CuError(
            f"Cantus Ultimus at {CU_API_URL} is unreachable: {reason}",
            CATEGORY_UNREACHABLE,
        ) from exc
    except TimeoutError as exc:
        raise CuError(
            f"Cantus Ultimus at {CU_API_URL} did not respond within {timeout}s.",
            CATEGORY_TIMEOUT,
        ) from exc

    if not body:
        return status, None
    try:
        return status, json.loads(body)
    except json.JSONDecodeError as exc:
        raise CuError(
            f"Cantus Ultimus returned a {status} that is not JSON: "
            f"{body[:200]!r}",
            CATEGORY_MALFORMED_RESPONSE,
        ) from exc


def submit_mei(
    *,
    manuscript_id: int,
    folio_number: str,
    mei: str,
    submitter: str,
    comment: str = "",
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """File one folio's MEI for review, returning CU's status record.

    CU answers 201 for a newly-filed submission and **200 when an identical
    document is already pending for that folio** -- its serializer hashes the
    content and returns the existing review rather than opening a second one.
    Both are successes, but they are not the same event: a "submit all" that
    reported every already-pending page as newly submitted would look like it
    had duplicated the whole queue. The distinction is surfaced as
    `alreadyPending` so the caller can say which happened.

    urllib raises on any non-2xx, so reaching the return means CU accepted it.
    """
    status, record = _request(
        "POST",
        "/api/mei-submissions/",
        payload={
            "manuscript_id": manuscript_id,
            "folio_number": folio_number,
            "mei": mei,
            "submitter": submitter,
            "comment": comment or "",
        },
        timeout=timeout,
    )
    if not isinstance(record, dict):
        raise CuError(
            f"Cantus Ultimus accepted the submission but returned "
            f"{type(record).__name__}, not a submission record.",
            CATEGORY_MALFORMED_RESPONSE,
        )
    # Both bodies come from the same serializer, so the status code is the only
    # thing that says which happened. 201 => this call created the review;
    # anything else 2xx (CU sends 200) => an identical document was already
    # pending for this folio and CU returned that existing review untouched.
    return {**record, "alreadyPending": status != 201}


def _get_list(path: str, what: str, *, query=None, timeout: int) -> list:
    """GET a CU collection, tolerating both DRF list shapes.

    DRF returns a bare JSON array unless pagination is configured, in which
    case the array is under "results". None of the three endpoints read here
    paginate today, but which of them does is CU's decision to change, not
    Mothra's, and guessing wrong would silently return an empty picker.
    """
    _, records = _request("GET", path, query=query, timeout=timeout)
    if records is None:
        return []
    if isinstance(records, dict) and isinstance(records.get("results"), list):
        return records["results"]
    if not isinstance(records, list):
        raise CuError(
            f"Cantus Ultimus returned {type(records).__name__}, not a list of "
            f"{what}.",
            CATEGORY_MALFORMED_RESPONSE,
        )
    return records


def list_submissions(submitter: str, timeout: int = DEFAULT_TIMEOUT) -> list:
    """Every submission CU holds for one submitter, newest first.

    The submitter is always the authenticated Mothra user's own username -- see
    cu_submission_api.py, which never lets a client choose it.
    """
    return _get_list(
        "/api/mei-submissions/",
        "submissions",
        query={"submitter": submitter},
        timeout=timeout,
    )


def list_manuscripts(timeout: int = DEFAULT_TIMEOUT) -> list:
    """CU's manuscripts, for the submission page's picker.

    NOTE: CU's ManuscriptList filters `public=True`, while its deposit
    serializer deliberately does NOT -- "manuscripts being processed by an OMR
    pipeline are normally still unpublished, and that is the case this endpoint
    exists to serve". So a manuscript can be perfectly valid to submit against
    and still be absent from this list. That is why the submission page also
    accepts a manuscript id typed by hand.
    """
    return _get_list("/manuscripts/", "manuscripts", timeout=timeout)


def list_folios(manuscript_id: int, timeout: int = DEFAULT_TIMEOUT) -> list:
    """One manuscript's folios, as Solr documents with `number` and `image_uri`.

    `image_uri` is the load-bearing field: CU refuses a deposit for a folio that
    has none ("not mapped to an image yet, so its notation could not be
    displayed"), so the caller uses it to disable those folios up front rather
    than discovering them one 400 at a time during a submit-all.
    """
    return _get_list(
        f"/folio-set/manuscript/{manuscript_id}/", "folios", timeout=timeout
    )
