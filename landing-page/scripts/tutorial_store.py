"""Auto-provisions each user's own copy of the tutorial project.

Called from auth_api.py's login() and register() with their own already-open
(con, cur) -- a brand-new user gets tokens straight from register() without
ever calling login(), so both call sites need the same idempotent check.

Mirrors ic_xml_store.py's store_ic_xml() shape: never raises. A missing
template (e.g. seed_demo_project.py hasn't been run in this environment) or
any DB error mid-clone must never break authentication -- it just means no
tutorial project shows up this time, silently, and the next login tries
again.
"""
import json
import traceback
import uuid

CLONE_PROJECT_NAME = "Mothra Tutorial Demo Project"

FULLY_UNLOCKED_STEPS = 4

def ensure_tutorial_project(con, cur, user_id: int) -> None:
    try:
        cur.execute(
            "SELECT 1 FROM projects WHERE user_id=%s AND is_tutorial=TRUE", (user_id,)
        )
        if cur.fetchone():
            return  # already has one

        cur.execute("SELECT id FROM projects WHERE is_tutorial_template=TRUE")
        template_row = cur.fetchone()
        if not template_row:
            return  # template not seeded in this environment -- skip quietly
        template_project_id = template_row[0]

        cur.execute(
            "INSERT INTO projects (user_id, name, is_tutorial, steps_unlocked)"
            " VALUES (%s,%s,TRUE,%s) RETURNING id",
            (user_id, CLONE_PROJECT_NAME, FULLY_UNLOCKED_STEPS),
        )
        new_project_id = cur.fetchone()[0]

        cur.execute(
            "SELECT id, name, mime_type, data FROM project_images"
            " WHERE project_id=%s ORDER BY created_at ASC, id ASC",
            (template_project_id,),
        )
        image_id_map = {}
        for old_id, name, mime_type, data in cur.fetchall():
            new_id = uuid.uuid4().hex
            cur.execute(
                "INSERT INTO project_images (id, project_id, name, mime_type, data)"
                " VALUES (%s,%s,%s,%s,%s)",
                (new_id, new_project_id, name, mime_type, data),
            )
            image_id_map[old_id] = new_id

        cur.execute(
            "SELECT name, xml_content, image_name, stave_source, image_id"
            " FROM mei_files WHERE project_id=%s ORDER BY created_at ASC",
            (template_project_id,),
        )
        for name, xml_content, image_name, stave_source, old_image_id in cur.fetchall():
            cur.execute(
                "INSERT INTO mei_files (id, project_id, name, xml_content, image_name, stave_source, image_id)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (uuid.uuid4().hex, new_project_id, name, xml_content, image_name, stave_source,
                 image_id_map.get(old_image_id)),
            )

        cur.execute(
            "UPDATE projects SET used_image_names=%s WHERE id=%s",
            (json.dumps(list(image_id_map.values())), new_project_id),
        )
        con.commit()
    except Exception:
        con.rollback()
        traceback.print_exc()