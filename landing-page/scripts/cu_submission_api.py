"""Submitting corrected MEI to Cantus Ultimus's deposit inbox.

The pipeline's step 6 used to end at a zip download (batch_api.py's
cantus-bundle) that a human carried to a CU maintainer by hand, because CU had
no write API. It has one now, so these endpoints file a page's MEI directly into
CU's review queue as a PENDING record. An admin on CU's side publishes,
requests a correction, or refuses it; nothing here can make anything public.

The zip export stays -- it is still the route for anything CU cannot accept.

Shape notes:
  - The HTTP talking lives in cu_api.py, which imports no fastapi so it can be
    unit-tested under CI's minimal dependency set. This module is the thin edge
    that maps its CuError categories onto status codes.
  - There is deliberately no mothra-side submissions table. Status is read
    straight from CU on request, keyed on (manuscript_id, folio_number), and
    matched back to pages by the frontend. See CLAUDE.md's Cantus Ultimus
    submission section.
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import NoReturn, Optional

import cu_api
from auth_api import get_current_user, db_cursor, require_project_owner, _log_activity

router = APIRouter()

# How each failure talking to CU is reported to the browser. The distinction
# that matters is 400 (the user can fix this: wrong folio, unmapped folio,
# unparseable MEI) versus everything else (nobody using the app can fix it).
_STATUS_BY_CATEGORY = {
    cu_api.CATEGORY_REJECTED: 400,
    cu_api.CATEGORY_THROTTLED: 429,
    cu_api.CATEGORY_NOT_CONFIGURED: 503,
    cu_api.CATEGORY_TIMEOUT: 504,
    cu_api.CATEGORY_UNAUTHORIZED: 502,
    cu_api.CATEGORY_UNREACHABLE: 502,
    cu_api.CATEGORY_SERVICE_ERROR: 502,
    cu_api.CATEGORY_MALFORMED_RESPONSE: 502,
}


def _raise_for(exc: cu_api.CuError) -> NoReturn:
    """Re-raise a CuError as the HTTPException the frontend should see.

    Retry-After is echoed on a throttle so the browser is told when to come
    back, rather than only that it was refused.
    """
    status = _STATUS_BY_CATEGORY.get(exc.category, 502)
    headers = None
    if exc.category == cu_api.CATEGORY_THROTTLED and exc.retry_after:
        headers = {"Retry-After": str(exc.retry_after)}
    raise HTTPException(status_code=status, detail=str(exc), headers=headers) from exc


@router.get("/cu/manuscripts")
def cu_manuscripts(user=Depends(get_current_user)):
    """CU's manuscripts, for the submission page's picker.

    Note this reaches CU's ManuscriptList, which filters public=True -- while
    CU's deposit serializer deliberately does not, since a manuscript being
    OMR'd is normally still unpublished. A manuscript can therefore be valid to
    submit against and still be missing here, which is why the submission page
    also accepts a manuscript id typed by hand.
    """
    try:
        return cu_api.list_manuscripts()
    except cu_api.CuError as exc:
        _raise_for(exc)


@router.get("/cu/manuscripts/{manuscript_id}/folios")
def cu_folios(manuscript_id: int, user=Depends(get_current_user)):
    """One manuscript's folios, with the image_uri that decides submittability.

    CU refuses a deposit for a folio with no image_uri, so the frontend uses
    this to disable those up front instead of discovering them one failed page
    at a time during a submit-all.
    """
    try:
        return cu_api.list_folios(manuscript_id)
    except cu_api.CuError as exc:
        _raise_for(exc)


@router.get("/cu/submissions")
def cu_submissions(user=Depends(get_current_user)):
    """Every CU submission belonging to the calling user.

    The submitter is taken from the JWT, never from a query parameter: CU's GET
    will happily filter by any submitter it is given, so letting a client choose
    would expose other people's submissions and review notes.
    """
    try:
        return cu_api.list_submissions(user["username"])
    except cu_api.CuError as exc:
        _raise_for(exc)


class CuSubmitBody(BaseModel):
    manuscriptId: int
    folioNumber: str
    comment: Optional[str] = None


@router.post("/projects/{project_id}/mei/{mei_id}/cu-submit")
def cu_submit(project_id: int, mei_id: str, body: CuSubmitBody,
              user=Depends(get_current_user)):
    """File one page's MEI with Cantus Ultimus for review."""
    with db_cursor() as (con, cur):
        require_project_owner(cur, project_id, user["id"])
        # Only this MEI row is read -- the caller has already resolved which
        # revision it wants (the frontend uses latestMeiPerImage, matching the
        # Neon editor), and re-deriving "latest" here would silently submit a
        # different document than the one the user was looking at.
        cur.execute(
            "SELECT m.name, m.xml_content, m.corrected, m.image_name, p.name"
            " FROM mei_files m JOIN projects p ON p.id = m.project_id"
            " WHERE m.id=%s AND m.project_id=%s",
            (mei_id, project_id),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="MEI not found")
        mei_name, xml_content, corrected, image_name, project_name = row
        if not xml_content:
            raise HTTPException(
                status_code=400,
                detail=f"{mei_name} has no MEI content to submit.",
            )

    folio = body.folioNumber.strip()
    if not folio:
        raise HTTPException(status_code=400, detail="a folio number is required")

    # CU's reviewer sees only manuscript, folio, submitter, timestamp and an MEI
    # download -- the comment is the single channel for anything else. Whether a
    # human corrected the page in Neon is the part a reviewer most needs and
    # cannot otherwise get, since Mothra submits uncorrected MEI too.
    provenance = (
        f"Mothra project {project_name!r}, page {image_name or mei_name}, "
        f"{'corrected in Neon' if corrected else 'NOT yet corrected in Neon'}."
    )
    note = (body.comment or "").strip()
    comment = f"{note}\n\n{provenance}" if note else provenance

    try:
        record = cu_api.submit_mei(
            manuscript_id=body.manuscriptId,
            folio_number=folio,
            mei=xml_content,
            submitter=user["username"],
            comment=comment,
        )
    except cu_api.CuError as exc:
        _raise_for(exc)

    with db_cursor() as (con, cur):
        _log_activity(
            cur, project_id, "cu_submitted",
            f"{image_name or mei_name} → CU manuscript {body.manuscriptId} f. {folio}",
        )
        con.commit()

    # 200 rather than 201 even for a newly-filed submission: what was created
    # lives on CU, not here, so there is no mothra resource to point a 201 at.
    # `alreadyPending` carries CU's own created-vs-deduplicated distinction.
    return JSONResponse(record)
