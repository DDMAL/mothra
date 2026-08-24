"""Storage for the GameraXML an encode run consumed (``ic_xml_files``).

Written by ``tasks_encode.py`` as the encoder parses its input, so the
project page's "Generated files → Classifier XML" tab shows the artefact
the pipeline actually used, at the moment it was used. Deliberately *not*
written at IC-export time: the export bridge (``ic_api.py``'s
``ic_complete``) can also be reached without an encode ever following, and
IC's own in-iframe export streams straight to the browser without mothra
seeing it at all -- so keying off "was encoded" is the only rule that holds
for every path into the encoder (interactive IC, auto-classified IC, and
the step-3 "upload XML output" path).

Reads live in ``ic_api.py`` (the ``/projects/{id}/ic-xml/{id}`` endpoints);
this module is import-light on purpose, since the Celery worker pulls it in.
"""
from __future__ import annotations

import uuid
from typing import Optional

import psycopg2

from auth_api import get_db_conn, release_db_conn


def store_ic_xml(
    project_id: Optional[int],
    image_id: Optional[str],
    image_name: Optional[str],
    xml_bytes: bytes,
    glyph_count: Optional[int] = None,
    session_id: Optional[str] = None,
) -> bool:
    """File ``xml_bytes`` as this page's current classifier XML.

    Returns whether a row was written. A missing ``project_id`` or
    ``image_name`` is a no-op, not an error: an ad-hoc encode with no
    project context (``/api/encode-upload`` without ``project_id``) has no
    project page to surface the file on.

    Delete-then-insert per page (like ``annotations``, unlike
    ``staffline_detections``' accumulate-forever history): a re-encode
    supersedes the page's previous XML rather than being a second data
    point worth comparing against. Matching prefers ``image_id`` --
    ``image_name`` alone is not unique within a project once duplicate-named
    uploads are allowed (mothra#241) -- and falls back to the name only when
    no id was recorded. Delete and insert share one transaction, so a
    failure can't leave the page with its old XML dropped and no new one in
    its place.

    ``xml_bytes`` is stored verbatim, in a BYTEA column. It used to be
    ``.decode("utf-8", "replace")``-ed into a TEXT column, which silently
    replaced every byte of a non-UTF-8 GameraXML with U+FFFD -- and the
    encode itself still succeeded, since ``ET.parse`` honours the document's
    own ``encoding=`` declaration, so the only casualty was the archived
    artefact nobody looks at until they need it. Only the JSON viewer
    endpoint decodes (``ic_api.py``'s ``get_ic_xml``); the download and the
    project-export zip serve these bytes unchanged.

    Never raises: this is a side artefact of an encode job, and losing it
    must not fail the encode the user is actually waiting on. The caller
    logs the ``False`` return instead.
    """
    if not project_id or not image_name:
        return False
    stem = image_name.rsplit(".", 1)[0] if "." in image_name else image_name
    con = None
    cur = None
    try:
        # Acquisition is inside the try like everything else: a pool that is
        # exhausted or cannot reach Postgres raises here, and "never raises"
        # has to hold for that too -- otherwise the encode the user is
        # waiting on dies over its side artefact.
        con = get_db_conn()
        cur = con.cursor()
        if image_id:
            cur.execute(
                "DELETE FROM ic_xml_files WHERE project_id=%s AND image_id=%s",
                (project_id, image_id),
            )
        else:
            cur.execute(
                "DELETE FROM ic_xml_files WHERE project_id=%s AND image_id IS NULL"
                " AND image_name=%s",
                (project_id, image_name),
            )
        cur.execute(
            "INSERT INTO ic_xml_files"
            " (id, project_id, image_id, image_name, name, xml_content, glyph_count, session_id)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                uuid.uuid4().hex, project_id, image_id, image_name,
                f"{stem}.xml", psycopg2.Binary(xml_bytes),
                glyph_count, session_id,
            ),
        )
        con.commit()
        return True
    except Exception:
        return False
    finally:
        if cur is not None:
            cur.close()
        if con is not None:
            release_db_conn(con)
