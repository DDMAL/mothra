"""One-shot seed script -- creates (or resets) the "tutorial template" demo
project: a fixture project, owned by its own throwaway fixture user, that
ships 2 manuscript pages plus one paired MEI file so a later frontend feature
can present a ready-made example without a real user having produced one.

Run manually, after `python migrate.py` has added `projects.is_tutorial_template`:

    cd landing-page/scripts && source .venv/bin/activate
    python migrate.py            # picks up the new column -- only needs to
                                  # run once per database
    python seed_demo_project.py

Deliberately does NOT touch `ic_xml_files` or the `ic-session-*.xml` fixture
under assets/demo_fixtures/files/ -- reserved for a later frontend
IC-training-set feature; mothra has no DB table for "IC training set" today
(see CLAUDE.md's Workflow pipeline step 2).

Safe to re-run: deletes and recreates the template project from scratch each
time, so it always ends up as exactly these 2 images + 1 MEI file, nothing
accumulated from a previous run. The template user (mothra-tutorial-template)
is created once and reused; it has a throwaway random password and is never
meant to log in.

Note: re-running this only updates the TEMPLATE. A user who already has
their own cloned tutorial project (tutorial_store.py's ensure_tutorial_project
only ever clones once per account) keeps whatever the template looked like
at clone time -- delete their `projects` row where is_tutorial=TRUE to force
a fresh clone on next login.
"""
# .env has to be loaded before importing auth_api -- auth_api reads
# DATABASE_URL/MOTHRA_SECRET at import time (see migrate.py's own comment).
from dotenv import load_dotenv
load_dotenv()

import mimetypes
import sys
import uuid

import psycopg2

from auth_api import get_db_conn, release_db_conn, hash_password
from config import DEMO_FIXTURES_DIR

TEMPLATE_USERNAME = "mothra-tutorial-template"
TEMPLATE_EMAIL = "tutorial-template@mothra.internal"
TEMPLATE_PROJECT_NAME = "Mothra Tutorial Demo Project"

IMAGES_DIR = DEMO_FIXTURES_DIR / "images"
FILES_DIR = DEMO_FIXTURES_DIR / "files"

DEMO_PAGES = [
    # Antiphonal does double duty in the guided tutorial -- both the
    # "process a page" example and the IC-classifying example (see
    # tutorial/tutorialSteps.ts's TUTORIAL_IMAGE_NAMES). A third fixture
    # image (Aarau_MsMurF2_6v.jpg) originally covered "process a page"
    # separately, kept out of the IC step entirely -- dropped in favor of
    # this simpler two-image project.
    ("Antiphonal_1v_hfngl.jpg", None),
    ("CDN-Hsmu_M2149.L4_097r_demo.png", "CDN-Hsmu_M2149.L4_097r_demo.mei"),
]

def _check_fixtures_exist() -> None:
    """Fail fast with a clear message instead of a mid-run traceback if the
    fixture assets aren't in place yet."""
    missing = []
    if not IMAGES_DIR.is_dir():
        missing.append(str(IMAGES_DIR))
    if not FILES_DIR.is_dir():
        missing.append(str(FILES_DIR))
    for image_filename, mei_filename in DEMO_PAGES:
        image_path = IMAGES_DIR / image_filename
        if not image_path.is_file():
            missing.append(str(image_path))
        if mei_filename:
            mei_path = FILES_DIR / mei_filename
            if not mei_path.is_file():
                missing.append(str(mei_path))
    if missing:
        print("Missing demo fixture file(s) - add these before running:", file=sys.stderr)
        for path in missing:
            print(f" {path}", file=sys.stderr)
        sys.exit(1)

def _ensure_template_user(cur) -> int:
    cur.execute("SELECT id FROM users WHERE username=%s", (TEMPLATE_USERNAME,))
    row = cur.fetchone()
    if row:
        return row[0]
    # This account is never meant to log in -- the password is discarded
    # immediately, nothing stores or returns it.
    cur.execute(
        "INSERT INTO users (username, email, first_name, last_name, password_hash)"
        " VALUES (%s,%s,%s,%s,%s) RETURNING id",
        (
            TEMPLATE_USERNAME,
            TEMPLATE_EMAIL,
            "Mothra",
            "Tutorial Template",
            hash_password(uuid.uuid4().hex),
        ),
    )
    return cur.fetchone()[0]

def _reset_template_project(cur, user_id: int) -> int:
    cur.execute(
        "SELECT id FROM projects WHERE user_id=%s AND is_tutorial_template=TRUE",
        (user_id,),
    )
    row = cur.fetchone()
    if row:
        old_project_id = row[0]
        cur.execute("DELETE FROM mei_files WHERE project_id=%s", (old_project_id,))
        cur.execute("DELETE FROM project_images WHERE project_id=%s", (old_project_id,))
        cur.execute("DELETE FROM projects WHERE id=%s", (old_project_id,))

    cur.execute(
        "INSERT INTO projects (user_id, name, is_tutorial_template) VALUES (%s,%s,TRUE) RETURNING id",
        (user_id, TEMPLATE_PROJECT_NAME),
    )
    return cur.fetchone()[0]

def _insert_image(cur, project_id: int, filename: str) -> str:
    data = (IMAGES_DIR / filename).read_bytes()
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    image_id = uuid.uuid4().hex
    cur.execute(
        "INSERT INTO project_images (id, project_id, name, mime_type, data)"
        " VALUES (%s,%s,%s,%s,%s)",
        (image_id, project_id, filename, mime_type, psycopg2.Binary(data)),
    )
    return image_id

def _insert_mei(cur, project_id: int, image_id: str, image_filename: str, mei_filename: str) -> None:
    # mei_files.name follows the real convention set by useEncodingFlow.ts
    # (`${stem}.mei`) -- the fixture's own filename already is that stem+.mei.
    xml_content = (FILES_DIR / mei_filename).read_text(encoding="utf-8")
    cur.execute(
        "INSERT INTO mei_files (id, project_id, name, xml_content, image_name, stave_source, image_id)"
        " VALUES (%s,%s,%s,%s,%s,%s,%s)",
        (uuid.uuid4().hex, project_id, mei_filename, xml_content, image_filename, None, image_id),
    )

def main() -> None:
    _check_fixtures_exist()

    con = get_db_conn()
    cur = con.cursor()
    try:
        user_id = _ensure_template_user(cur)
        project_id = _reset_template_project(cur, user_id)
        for image_filename, mei_filename in DEMO_PAGES:
            image_id = _insert_image(cur, project_id, image_filename)
            if mei_filename:
                _insert_mei(cur, project_id, image_id, image_filename, mei_filename)
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        cur.close()
        release_db_conn(con)

    print(
        f"Seeded tutorial template project id={project_id} "
        f"(user_id={user_id}, {len(DEMO_PAGES)} images)."
    )


if __name__ == "__main__":
    main()