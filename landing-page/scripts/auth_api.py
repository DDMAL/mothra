from fastapi import APIRouter, Depends, HTTPException, Header, Request
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
from slowapi import Limiter
from slowapi.util import get_remote_address
from config import MODELS_DIR, NEON_MANIFESTS_DIR
import psycopg2, psycopg2.errors, os, secrets, hashlib, base64
from datetime import datetime, timedelta
from jose import jwt, JWTError
from contextlib import contextmanager
import bcrypt
import json

limiter = Limiter(key_func=get_remote_address)
# connection pooling
from psycopg2 import pool as _pg_pool

_db_pool: Optional["_pg_pool.ThreadedConnectionPool"] = None

def _get_pool() -> "_pg_pool.ThreadedConnectionPool":
    global _db_pool
    if _db_pool is None:
        # Bare os.environ[...] -- raises KeyError, not a friendly error, if
        # landing-page/scripts/.env doesn't exist/doesn't set this yet.
        # psycopg2 just needs a DSN, so any local Postgres works -- see
        # ../README.md's "Prerequisites" section for the install
        # (brew install postgresql@16 && createdb mothra_dev, then
        # DATABASE_URL=postgresql://localhost/mothra_dev).
        _db_pool = _pg_pool.ThreadedConnectionPool(
            minconn=2, maxconn=15, dsn=os.environ["DATABASE_URL"]
        )
    return _db_pool

def get_db_conn():
    return _get_pool().getconn()

def require_project_owner(cur, project_id: int, user_id: int) -> None:
    cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id, ))
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="project not found")
    if row[0] != user_id:
        raise HTTPException(status_code=403, detail="not your project")

def release_db_conn(con) -> None:
    _get_pool().putconn(con)

router = APIRouter()

def _require_secret_key() -> str:
    """MOTHRA_SECRET signs every access/refresh token and Neon edit-session
    token. This used to fall back to a fresh secrets.token_hex(32) so a bare
    local checkout with no .env still ran -- but that fallback evaluates at
    import time, so any process started without MOTHRA_SECRET set (backend,
    worker, a future replica) got its OWN random secret, silently
    invalidating every previously-issued token on that process's next
    restart with no error to flag it as a misconfiguration rather than
    "random logouts after every deploy." Fails fast instead, like
    DATABASE_URL above."""
    key = os.environ.get("MOTHRA_SECRET")
    if not key:
        raise RuntimeError(
            "MOTHRA_SECRET is not set. Set it in landing-page/scripts/.env "
            "(see ../README.md's Prerequisites) or the process environment "
            "before starting the backend/worker -- there is no safe default, "
            "since a randomly-generated one would silently invalidate every "
            "issued token on the next restart."
        )
    return key

SECRET_KEY = _require_secret_key()
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 72
STORAGE_QUOTA_BYTES = int(os.getenv("STORAGE_QUOTA_MB", "500")) * 1024 * 1024
REFRESH_TOKEN_EXPIRE_DAYS = 30

MODELS_DIR.mkdir(parents=True, exist_ok=True)
NEON_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

def cleanup_stale_neon_manifests(max_age_seconds: int = 86400) -> int:
    """Deletes exported Neon-editor manifest files (NEON_MANIFESTS_DIR/*.jsonld)
    older than max_age_seconds. This lives on the backend container's own
    ephemeral local disk (not the stored_models NFS share), unlike
    job_store.cleanup_stale_uploads/cleanup_stale_sessions -- a Celery task run
    by the worker executes on a different pod/filesystem and would have no
    effect on these files, so this has to be invoked from the backend process
    itself (see main.py). Consolidates what used to be two separately
    maintained inline copies of this same loop (auth_api.py's _migrate_db and
    mei_api.py's create_edit_session -- mothra#220 row 28)."""
    import time as _time
    now = _time.time()
    deleted = 0
    for f in NEON_MANIFESTS_DIR.glob("*.jsonld"):
        try:
            if now - f.stat().st_mtime > max_age_seconds:
                f.unlink(missing_ok=True)
                deleted += 1
        except FileNotFoundError:
            pass
    return deleted

@contextmanager
def db_cursor():
    con = get_db_conn()
    cur = con.cursor()
    try:
        yield con, cur
    finally:
        cur.close()
        release_db_conn(con)

def init_db():
    con = get_db_conn()
    cur = con.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                first_name TEXT,
                last_name TEXT,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                name TEXT NOT NULL,
                steps_unlocked INTEGER DEFAULT 0,
                -- mothra#241 follow-up: column name is legacy (kept as-is to
                -- avoid a migration -- it's an opaque JSON string blob either
                -- way), but its contents are now project_images.id values,
                -- not names -- see projects_api.py's usedImageIds field.
                used_image_names TEXT DEFAULT '[]',
                used_model_names TEXT DEFAULT '[]',
                deleted_at TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS project_images (
                id TEXT PRIMARY KEY,
                project_id INTEGER REFERENCES projects(id),
                name TEXT NOT NULL,
                mime_type TEXT,
                data BYTEA NOT NULL,
                folio TEXT,
                source_id TEXT,
                source_name TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS project_models (
                id TEXT PRIMARY KEY,
                project_id INTEGER REFERENCES projects(id),
                name TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS mei_files (
                id TEXT PRIMARY KEY,
                project_id INTEGER REFERENCES projects(id),
                name TEXT NOT NULL,
                xml_content TEXT,
                corrected INTEGER DEFAULT 0,
                image_name TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS activity_log (
                    id SERIAL PRIMARY KEY,
                    project_id INTEGER NOT NULL REFERENCES projects(id),
                    action_type TEXT NOT NULL,
                    detail TEXT DEFAULT '',
                    created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS project_logs (
                    id SERIAL PRIMARY KEY,
                    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    log_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id TEXT PRIMARY KEY,
                project_id INTEGER REFERENCES projects(id),
                image_id TEXT,
                image_name TEXT NOT NULL,
                yolo_txt TEXT NOT NULL,
                model_id TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS text_alignments (
                id TEXT PRIMARY KEY,
                project_id INTEGER REFERENCES projects(id),
                image_id TEXT,
                image_name TEXT NOT NULL,
                alignment_json TEXT NOT NULL,
                median_line_spacing REAL DEFAULT 0,
                syllable_count INTEGER DEFAULT 0,
                log_text TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        # Accumulate-forever (never overwritten), unlike annotations' delete-then-
        # insert -- so a future re-run (e.g. once interpolate_missing is validated
        # and flipped on) stays comparable against history without a schema change.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS staffline_detections (
                id TEXT PRIMARY KEY,
                project_id INTEGER REFERENCES projects(id),
                image_id TEXT,
                image_name TEXT NOT NULL,
                annotation_id TEXT,
                jsomr_json JSONB NOT NULL,
                scale_unit REAL,
                stave_count INTEGER,
                mode_lines_per_stave INTEGER,
                settings_json JSONB NOT NULL,
                status TEXT NOT NULL DEFAULT 'succeeded',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # The GameraXML the Interactive Classifier exported for a page --
        # the artefact mothra's encoder consumes, surfaced in the project
        # page's "Generated files" tab. One current row per page
        # (delete-then-insert on re-export, like annotations), since a
        # re-export supersedes the previous XML for that page rather than
        # adding a comparable data point the way staffline_detections'
        # accumulate-forever history does.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ic_xml_files (
                id TEXT PRIMARY KEY,
                project_id INTEGER REFERENCES projects(id),
                image_id TEXT,
                image_name TEXT NOT NULL,
                name TEXT NOT NULL,
                -- BYTEA, not TEXT: this is the encoder's verbatim input, and
                -- the step-3 "upload XML output" path accepts any file the
                -- user has. A GameraXML declaring `encoding="ISO-8859-1"`
                -- parses fine (ET.parse honours the declaration) but its
                -- accented bytes are not valid UTF-8, so decoding it to store
                -- it as TEXT replaced them with U+FFFD -- irreversibly, and
                -- only in the archived artefact, never in the encode the user
                -- was watching. Storing bytes keeps the download and the zip
                -- export byte-identical to what was encoded.
                xml_content BYTEA NOT NULL,
                glyph_count INTEGER,
                session_id TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # job queue
        cur.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                project_id INTEGER,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS job_events (
                id SERIAL PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                payload JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS job_uploads (
                upload_id TEXT PRIMARY KEY,
                data BYTEA NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS job_sessions (
                session_id TEXT PRIMARY KEY,
                mei_bytes BYTEA NOT NULL,
                stem TEXT NOT NULL,
                manifest JSONB,
                project_id INTEGER,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # performance: db indexes
        for _idx in [
            "CREATE INDEX IF NOT EXISTS idx_project_images_pid ON project_images(project_id)",
            "CREATE INDEX IF NOT EXISTS idx_project_models_pid ON project_models(project_id)",
            "CREATE INDEX IF NOT EXISTS idx_mei_files_pid      ON mei_files(project_id)",
            "CREATE INDEX IF NOT EXISTS idx_annotations_pid    ON annotations(project_id)",
            "CREATE INDEX IF NOT EXISTS idx_activity_log_pid   ON activity_log(project_id)",
            "CREATE INDEX IF NOT EXISTS idx_projects_user_id   ON projects(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_text_alignments_pid ON text_alignments(project_id)",
            "CREATE INDEX IF NOT EXISTS idx_staffline_detections_lookup"
            " ON staffline_detections(project_id, image_name, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_ic_xml_files_pid ON ic_xml_files(project_id)",
            "CREATE INDEX IF NOT EXISTS idx_job_events_job_id ON job_events(job_id, id)",
            "CREATE INDEX IF NOT EXISTS idx_jobs_status        ON jobs(status)",
        ]:
            cur.execute(_idx)
        con.commit()
    except (psycopg2.errors.DuplicateTable, psycopg2.errors.DuplicateObject, psycopg2.errors.UniqueViolation):
        # backend and worker both run init_db() independently at import, and
        # CI/CD redeploys both on every push to main - two processes racing
        # on CREATE TABLE/INDEX IF NOT EXISTS against a not-yet-existing
        # schema can raise a UniqueViolation on the pg_type catalog rather
        # than the expected DuplicateTable, since the "IF NOT EXISTS"
        # existence check isn't atomic across concurrent sessions. Harmless
        # to roll back and move on: the winner just created the identical
        # schema this loser was about to create anyway.
        con.rollback()
    finally:
        cur.close()
        release_db_conn(con)

# init_db()/_migrate_db() no longer run as a side effect of importing this
# module (mothra#220 row 31) -- previously this ran at import time in BOTH
# the backend and the Celery worker (every process that transitively imports
# auth_api), racing each other on a fresh database and occasionally
# CrashLoopBackOff-ing once before self-healing (see k8s/README.md's former
# "Known follow-ups"). migrate.py is now the one explicit entrypoint that
# calls both, run once per deploy as a k8s Job (k8s/migrate-job.yaml) before
# backend/worker start -- see that file and .github/workflows/build-images.yml.

# Columns added to tables that predate them, as (table, column, definition).
#
# Each entry still has to be idempotent *without raising* -- migrate.py can be
# re-run manually, and a Job can retry on failure. These used to be bare
# `ALTER TABLE ... ADD COLUMN` statements using `except DuplicateColumn` as
# control flow. The application handled that fine, but Postgres logs a
# server-side ERROR for a failed statement before the client ever sees it
# (log_min_error_statement = error), so each pod start wrote ~26 false ERRORs
# into the production database log — ~1900 accumulated lines in which a real
# error would have been invisible. `ADD COLUMN IF NOT EXISTS` reaches the same
# end state silently.
#
# Identifiers here are hardcoded literals, never user input.
_ADDED_COLUMNS = [
    ("projects",        "used_model_names",      "TEXT DEFAULT '[]'"),
    ("projects",        "last_opened_at",        "TIMESTAMPTZ"),
    ("projects",        "is_pinned",             "BOOLEAN DEFAULT FALSE"),
    ("projects",        "created_at",            "TIMESTAMPTZ DEFAULT NOW()"),
    ("projects",        "used_annotation_names", "TEXT DEFAULT '[]'"),
    ("projects",        "cantus_source_id",      "TEXT"),
    ("projects",        "is_tutorial_template",  "BOOLEAN DEFAULT FALSE"),

    ("project_images",  "created_at",            "TIMESTAMPTZ DEFAULT NOW()"),
    ("project_images",  "folio",                 "TEXT"),
    ("project_images",  "source_id",             "TEXT"),
    ("project_images",  "source_name",           "TEXT"),
    ("project_images",  "original_data",         "BYTEA"),
    # The working-copy `mime_type` column can't be reused for original_data:
    # a client-side resize (imageResize.ts) always re-encodes the working
    # copy as JPEG, while original_data keeps whatever format the source
    # file actually was (PNG, TIFF, ...) — serving/embedding original_data
    # under the working copy's mime_type mislabels it.
    ("project_images",  "original_mime_type",    "TEXT"),
    # Set by ic_api.py's ic_start() the first time a page is staged into IC,
    # so the "synthetic placeholder grid, not real detections" warning
    # (mothra#220 DL-1) survives session resume/reload -- IC's own
    # /sessions/lookup response doesn't carry this flag, so mothra has to
    # remember it itself rather than only returning it on the fresh-staging
    # path. NULL/FALSE for images staged before this column existed.
    ("project_images",  "ic_synthetic",          "BOOLEAN"),

    ("mei_files",       "image_name",            "TEXT"),
    # Records which of tasks_encode.py's 3-tier stave-source fallback actually
    # produced this MEI's zones ("staffline_detection" / "yolo_annotation" /
    # "glyph_estimate" / "glyph_estimate_unresolved_lines" /
    # "glyph_estimate_synthetic_lines" / "placeholder_no_glyphs") -- see
    # CLAUDE.md's "Staffline detection" section. NULL for MEI files encoded
    # before this column existed.
    ("mei_files",       "stave_source",          "TEXT"),
    # project_images.id of the source page this MEI was encoded from.
    # Preferred over image_name by imageStep.ts's getImageProgress() and by
    # mei_api.py's create_edit_session, since image_name alone is not unique
    # within a project once duplicate-named uploads are allowed (mothra#241).
    # NULL for rows written before the encode round-trip threaded an id
    # through (see tasks_encode.py's _encode_one/run_encode_batch_task) and
    # for the single-image "manual encode-upload" path, which has no id to
    # give (ICCompletionTestPage.tsx picks a raw local file, not a
    # project_images row) — both fall back to the old image_name match.
    ("mei_files",       "image_id",              "TEXT"),

    ("project_models",  "file_path",             "TEXT"),
    ("project_models",  "kind",                  "TEXT DEFAULT 'yolo'"),
    ("project_models",  "class_map",             "TEXT"),
    ("project_models",  "file_hash",             "TEXT"),

    ("annotations",     "model_label",           "TEXT"),
    ("annotations",     "model_hash",            "TEXT"),
    ("staffline_detections", "classifier_image",      "BYTEA"),
    ("staffline_detections", "classifier_image_mime", "TEXT DEFAULT 'image/png'"),
    # mothra#286: paco-classifier's OTHER output layer (background-only),
    # stored the same way as classifier_image above (verbatim bytes, its own
    # mime sibling) so both of its raw output layers can be viewed
    # side-by-side with the original page -- see staffline_stage.py's
    # persist_staffline_detection and inference_api.py's
    # GET .../background-image.
    ("staffline_detections", "background_image",      "BYTEA"),
    ("staffline_detections", "background_image_mime", "TEXT DEFAULT 'image/png'"),

    ("text_alignments", "log_text",              "TEXT"),
    # Which project_images column syl_boxes' absolute-pixel ul/lr coords were
    # computed against ("original" / "working_copy") -- mirrors
    # staffline_detections' settings_json.storage_variant (CodeRabbit PR #219),
    # added for the same reason: SF-2 made tasks_predict.py's text-finding call
    # run against original_data whenever it's present, but projects_api.py's
    # imageSrc used to always point at the resized working copy regardless,
    # so the "Detected text" viewer scaled these boxes against the wrong
    # image's naturalWidth/naturalHeight (mothra#260). Default 'working_copy'
    # matches every caller that doesn't read original_data at all
    # (tasks_text_batch.py, the standalone /text-finding/run endpoint) and
    # every pre-existing row written before this column existed.
    ("text_alignments", "storage_variant",       "TEXT DEFAULT 'working_copy'"),

    # jobs : retry lineage + stored kickoff params (needed by cancel/retry)
    ("jobs",            "params",                "JSONB"),
    ("jobs",            "retry_of",              "TEXT REFERENCES jobs(job_id)"),
    ("jobs",            "attempt",               "INTEGER NOT NULL DEFAULT 1"),

    # job_sessions.project_id backs encode_api.py's /manifest, /mei ownership
    # checks (mothra#220 row 25) -- NULL on rows written before this column
    # existed, which encode_api.py treats as "no recorded owner, allow" rather
    # than a hard 403 (see get_manifest/get_mei) since job_sessions' 14-day
    # cleanup (job_store.cleanup_stale_sessions) already bounds how long any
    # ownerless row can exist.
    ("job_sessions",    "project_id",            "INTEGER"),
]


# migrate existing DBs that predate the columns above
def _migrate_db():
    con = get_db_conn()
    cur = con.cursor()
    try:
        for _table, _column, _definition in _ADDED_COLUMNS:
            cur.execute(
                f"ALTER TABLE {_table} ADD COLUMN IF NOT EXISTS {_column} {_definition}"
            )
        con.commit()
    except psycopg2.errors.DuplicateColumn:
        # `IF NOT EXISTS` isn't atomic across concurrent sessions — the same
        # race init_db() documents above (backend and worker migrating at
        # once) can still surface a duplicate here. Harmless: the other
        # process added the identical column. This is a safety net only; it
        # should never fire in steady state.
        con.rollback()
    finally:
        cur.close()
        release_db_conn(con)

    con = get_db_conn()
    cur = con.cursor()
    cur.execute("CREATE INDEX IF NOT EXISTS idx_project_models_kind ON project_models(project_id, kind)")
    con.commit()
    cur.close()
    release_db_conn(con)

    # mei_files: needed so the cantus-bundle export (section 8) can pick the
    # latest revision per image_name, mirroring how annotations/text_alignments
    # already do `ORDER BY created_at DESC LIMIT 1`.
    con = get_db_conn(); cur = con.cursor()
    try:
        cur.execute("ALTER TABLE mei_files ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ")
        # mei_files is append-only (add_mei always INSERTs a new row; existing
        # rows are only ever UPDATEd in place afterwards), so a manuscript
        # corrected/re-encoded more than once before this migration already has
        # several rows sharing one image_name. Backfilling them all with a flat
        # NOW() would tie every legacy revision together, and the cantus-bundle
        # export's `created_at = MAX(created_at)` lookup would then match ALL
        # of them instead of just the true latest one — silently bundling a
        # stale/uncorrected MEI alongside (or instead of) the real latest
        # revision. ctid approximates insertion order for this append-only
        # table, so use it to hand legacy rows distinct, order-preserving
        # timestamps before defaulting new inserts to NOW().
        #
        # The `created_at IS NULL` guard is load-bearing, not a micro-
        # optimisation. Before this function was made idempotent, the bare
        # `ADD COLUMN` above raised DuplicateColumn on every run after the
        # first, and that raise — aborting the transaction — was the only
        # thing stopping this backfill from re-running. Now that the ALTER
        # succeeds silently, an unqualified UPDATE would re-stamp EVERY row
        # with epoch+N on every backend/worker start, destroying the real
        # timestamps of rows inserted since and breaking the very MAX
        # (created_at) lookup this column exists to serve. Restricted to
        # never-backfilled rows it is a no-op on an already-migrated database
        # and still correct on one that predates the column.
        cur.execute("""
            WITH ordered AS (
                SELECT id, ROW_NUMBER() OVER (ORDER BY ctid) AS rn
                FROM mei_files WHERE created_at IS NULL
            )
            UPDATE mei_files m
            SET created_at = TIMESTAMPTZ 'epoch' + (ordered.rn * INTERVAL '1 second')
            FROM ordered
            WHERE m.id = ordered.id AND m.created_at IS NULL
        """)
        cur.execute("ALTER TABLE mei_files ALTER COLUMN created_at SET DEFAULT NOW()")
        con.commit()
    except psycopg2.errors.DuplicateColumn:
        con.rollback()
    finally:
        cur.close(); release_db_conn(con)

    # ic_xml_files.xml_content: TEXT -> BYTEA. Rows written before this stored
    # `xml_bytes.decode("utf-8", "replace")`, so a non-UTF-8 GameraXML (the
    # step-3 "upload XML output" path takes any file the user has) had its
    # bytes replaced with U+FFFD before ever reaching the table -- see the
    # CREATE TABLE comment in init_db(). convert_to(..., 'UTF8') reproduces
    # exactly the bytes that column was already serving, so this conversion
    # is byte-for-byte lossless for every existing row; it does NOT recover
    # rows already corrupted on the way in (that data is gone), it only stops
    # new ones from being corrupted.
    #
    # Guarded on the column's current type rather than run unconditionally:
    # ALTER COLUMN ... TYPE has no IF NOT EXISTS form and rewrites the whole
    # table, and backend and worker both migrate independently at import (see
    # below), so an unguarded version would rewrite this table on every pod
    # start. Once converted, data_type is 'bytea' and this is a no-op.
    con = get_db_conn(); cur = con.cursor()
    try:
        cur.execute("""
            SELECT data_type FROM information_schema.columns
            WHERE table_name='ic_xml_files' AND column_name='xml_content'
        """)
        _row = cur.fetchone()
        if _row and _row[0] == 'text':
            cur.execute(
                "ALTER TABLE ic_xml_files ALTER COLUMN xml_content"
                " TYPE BYTEA USING convert_to(xml_content, 'UTF8')"
            )
        con.commit()
    except psycopg2.errors.UndefinedTable:
        # Fresh database where init_db() hasn't created the table yet, or the
        # racing pod is mid-CREATE. init_db() makes it BYTEA outright.
        con.rollback()
    finally:
        cur.close(); release_db_conn(con)

    # backend and worker both run this migration independently at import
    # (see k8s/README.md's "Known follow-ups"), and CI/CD redeploys both on
    # every push to main — so unlike the ALTER TABLE blocks above, these
    # CREATE TABLE/INDEX IF NOT EXISTS statements need their own duplicate-
    # object guard too, or two pods racing on a not-yet-existing table/index
    # can crash one of them with an uncaught duplicate-relation error.
    con = get_db_conn(); cur = con.cursor()
    try:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_job_uploads_created_at ON job_uploads (created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_job_sessions_created_at ON job_sessions (created_at)")
        con.commit()
    except (psycopg2.errors.DuplicateTable, psycopg2.errors.DuplicateObject, psycopg2.errors.UniqueViolation):
        con.rollback()
    finally:
        cur.close(); release_db_conn(con)

    # refresh_tokens
    con = get_db_conn(); cur = con.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS refresh_tokens (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token_hash TEXT NOT NULL UNIQUE,
                expires_at TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                revoked_at TIMESTAMPTZ
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user_id ON refresh_tokens(user_id)")
        con.commit()
    except (psycopg2.errors.DuplicateTable, psycopg2.errors.DuplicateObject, psycopg2.errors.UniqueViolation):
        con.rollback()
    finally:
        cur.close(); release_db_conn(con)

def _pre_hash(pw: str) -> str:
    """SHA-256+base64 the password before bcrypt ever sees it.

    bcrypt silently truncates its input at 72 bytes, so without this two
    different long/multi-byte-UTF-8 passwords sharing the same first 72
    bytes would hash identically and both would authenticate. Collapsing to
    a fixed-size digest first means the whole password participates.
    """
    return base64.b64encode(hashlib.sha256(pw.encode("utf-8")).digest()).decode()

def hash_password(pw: str) -> str:
    return bcrypt.hashpw(_pre_hash(pw).encode(), bcrypt.gensalt()).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(_pre_hash(plain).encode(), hashed.encode())

def create_token(user_id: int) -> str:
    exp = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode({"sub": str(user_id), "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)

def _hash_token(raw: str) -> str:
    """Plain SHA-256, deliberately not bcrypt, for refresh-token storage.

    Refresh tokens are 256 bits of secrets.token_urlsafe(32), not a
    human-chosen password -- nothing brute-forces that, so bcrypt's slow
    salted hashing buys nothing here and would actively break the lookup:
    bcrypt salts randomly by design, so the same token would hash
    differently every time and refresh_tokens.token_hash could never be
    looked up by a plain WHERE equality the way it is in refresh_token()
    below. A deterministic hash is both sufficient and required.
    """
    return hashlib.sha256(raw.encode()).hexdigest()

def create_refresh_token(user_id: int) -> str:
    raw = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    with db_cursor() as (con, cur):
        cur.execute(
            "INSERT INTO refresh_tokens (user_id, token_hash, expires_at) VALUES (%s, %s, %s)",
            (user_id, _hash_token(raw), expires_at),
        )
        con.commit()
    return raw

def _make_edit_token(project_id: int, mei_id: str) -> str:
    payload = {
        "project_id": project_id,
        "mei_id": mei_id,
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def _verify_edit_token(token: str, project_id: int, mei_id: str) -> bool:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("project_id") == project_id and payload.get("mei_id") == mei_id
    except Exception:
        return False

def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="not authenticated")
    token = authorization.removeprefix("Bearer ")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="invalid token")
    with db_cursor() as (con, cur):
        cur.execute(
            "SELECT id, username, email, first_name, last_name, created_at from users where id=%s", (user_id,)
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="user not found")
    return {"id": row[0], "username": row[1], "email": row[2], "firstName": row[3],
            "lastName": row[4], "createdAt": str(row[5])}

def _log_activity(cur, project_id: int, action_type: str, detail: str = ""):
    cur.execute(
        "INSERT INTO activity_log (project_id, action_type, detail) VALUES (%s, %s, %s)",
        (project_id, action_type, detail)
    )

class RegisterBody(BaseModel):
    username: str
    email: str
    first_name: str
    last_name: str
    password: str

class LoginBody(BaseModel):
    username: str # username or email
    password: str

@router.post("/register")
@limiter.limit("5/minute")
def register(request: Request, body: RegisterBody):
    with db_cursor() as (con, cur):
        try:
            cur.execute(
                "INSERT INTO users (username, email, first_name, last_name, password_hash)"
                " VALUES (%s,%s,%s,%s,%s) RETURNING id",
                (body.username, body.email, body.first_name, body.last_name, hash_password(body.password))
            )
            user_id = cur.fetchone()[0]
            con.commit()
        except psycopg2.errors.UniqueViolation:
            con.rollback()
            raise HTTPException(status_code=409, detail="username or email already taken")
    return {
        "token": create_token(user_id),
        "refresh_token": create_refresh_token(user_id),
        "user": {"id": user_id, "username": body.username, "email": body.email, "firstName": body.first_name}
    }

@router.post("/login")
@limiter.limit("10/minute")
def login(request: Request, body: LoginBody):
    with db_cursor() as (con, cur):
        cur.execute(
            "SELECT id, username, email, first_name, password_hash FROM  users WHERE username=%s OR email=%s",
            (body.username, body.username)
        )
        row = cur.fetchone()
    if not row or not verify_password(body.password, row[4]):
        raise HTTPException(status_code=401, detail="invalid credentials")
    return {
        "token": create_token(row[0]),
        "refresh_token": create_refresh_token(row[0]),
        "user": {"id": row[0], "username": row[1], "email": row[2], "firstName": row[3]}
    }

@router.post("/auth/refresh")
def refresh_token(x_refresh_token: str = Header(None, alias="X-Refresh-Token")):
    if not x_refresh_token:
        raise HTTPException(status_code=401, detail="missing refresh token")
    token_hash = _hash_token(x_refresh_token)
    with db_cursor() as (con, cur):
        cur.execute(
            "SELECT id, user_id, expires_at, revoked_at FROM refresh_tokens WHERE token_hash=%s",
            (token_hash,),
        )
        row = cur.fetchone()
        if not row or row[3] is not None or row[2] < datetime.utcnow():
            raise HTTPException(status_code=401, detail="invalid or expired refresh token")
        cur.execute("UPDATE refresh_tokens SET revoked_at=NOW() WHERE id=%s", (row[0],))
        con.commit()
    user_id = row[1]
    return {
        "access_token": create_token(user_id),
        "refresh_token": create_refresh_token(user_id),  # rotation: old is now revoked, above
        "token_type": "bearer",
    }

@router.get("/me")
def me(user=Depends(get_current_user)):
    return user

@router.post("/auth/logout")
def logout(x_refresh_token: str = Header(None, alias="X-Refresh-Token"), user=Depends(get_current_user)):
    if x_refresh_token:
        with db_cursor() as (con, cur):
            cur.execute(
                "UPDATE refresh_tokens SET revoked_at=NOW() WHERE token_hash=%s AND user_id=%s AND revoked_at IS NULL",
                (_hash_token(x_refresh_token), user["id"]),
            )
            con.commit()
    return {"ok": True}

def get_latest_text_alignment(cur, project_id: int, image_name: str,
                               image_id: Optional[str] = None,
                               include_storage_variant: bool = False):
    """Return the most recently created text_alignments row's parsed
    alignment_json for this image, or None when there's no row, or the
    stored JSON isn't a dict. Single source of truth for "what is
    mothra-text's current syllable data for this image" -- shared by
    tasks_encode.py's _resolve_hints() (feeds a fresh encode) and
    mei_api.py's create_edit_session (re-verifies an existing one).

    `image_name` alone is NOT unique within a project -- project_images has
    no DB-level uniqueness constraint on `name` (only an app-level dedup
    check at upload time), and a project can run /text/run or
    /text-batch/run more than once for the same name. When the caller can
    supply the row's actual `image_id` (project_images.id -- the same
    identifier batch_api.py already keys text_alignments lookups on),
    resolve by that instead so a same-named-but-different image can't
    return the wrong alignment. Falls back to the old image_name-only
    lookup when the caller doesn't have an image_id available (e.g. a
    freshly-uploaded file with no persisted project_images row yet).

    A real database failure is NOT one of those "no data" cases -- it's
    distinguished from a missing row/malformed JSON and re-raised (after
    rolling back the connection, since a failed query otherwise leaves the
    pooled connection in "current transaction is aborted" state for
    whatever runs next on it). Callers that want this function's old
    swallow-everything behavior wrap the call in their own try/except, the
    same way tasks_encode.py's _resolve_hints() already does for its two
    neighboring lookups.

    include_storage_variant (mothra#260): when True, returns
    (alignment, storage_variant) instead of just alignment -- "None" cases
    become (None, None). storage_variant is the row's own
    "original"/"working_copy" tag (see text_api.py's stream_text_finding),
    needed by a caller that has to know which project_images column
    syl_boxes' absolute-pixel coords were computed against before comparing
    them to some other image's dimensions -- mei_api.py's create_edit_session
    is the only caller that opts in; tasks_encode.py's _resolve_hints() only
    ever reads syl_boxes' text/order, never their absolute coords, so it
    doesn't need this and stays on the plain default."""
    try:
        cols = "alignment_json, storage_variant" if include_storage_variant else "alignment_json"
        if image_id:
            cur.execute(
                f"SELECT {cols} FROM text_alignments WHERE image_id=%s AND project_id=%s"
                " ORDER BY created_at DESC LIMIT 1",
                (image_id, project_id),
            )
        else:
            cur.execute(
                f"SELECT {cols} FROM text_alignments WHERE image_name=%s AND project_id=%s"
                " ORDER BY created_at DESC LIMIT 1",
                (image_name, project_id),
            )
        row = cur.fetchone()
    except Exception:
        cur.connection.rollback()
        raise
    empty = (None, None) if include_storage_variant else None
    if not row or not row[0]:
        return empty
    try:
        alignment = json.loads(row[0])
    except (TypeError, ValueError):
        return empty
    alignment = alignment if isinstance(alignment, dict) else None
    if alignment is None:
        return empty
    if include_storage_variant:
        return alignment, (row[1] or "working_copy")
    return alignment