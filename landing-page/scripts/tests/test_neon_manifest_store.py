"""Unit tests for job_store.py's neon_manifest_put/neon_manifest_get/
cleanup_stale_neon_manifests (mothra#230, NEON_MANIFESTS_DIR half) -- the
Postgres-backed replacement for NEON_MANIFESTS_DIR/{session_id}.jsonld disk
files.

job_store.py imports get_db_conn/release_db_conn from auth_api at module
level, and auth_api.py imports psycopg2/fastapi/slowapi/jose/bcrypt -- none
of which are installed by CI's "DB-independent scripts tests" step (see
.github/workflows/tests.yml). Stubs a minimal auth_api stand-in in
sys.modules before loading job_store.py, add-if-missing onto whatever's
already there -- mirroring test_image_src_storage_variant.py's documented
reasoning: sys.modules is process-global across the whole pytest session, so
whichever test file's stub installs first "wins," and every attribute here
is add-if-missing so this can run in any collection order without starving
another test's own stub of something IT needs.

Unlike every other file in this directory, this one actually wants the REAL
job_store.py (it's what's under test), not a fake stand-in for it -- but at
least one other file (test_image_src_storage_variant.py) installs a FAKE
sys.modules["job_store"], and alphabetically collects before this file, so a
plain `import job_store` here could silently bind that fake instead of the
real module. Loaded via importlib under a private module-registry key
instead, sidestepping the shared "job_store" sys.modules slot entirely.

No DB, no FastAPI app.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS_DIR))


def _install_auth_api_stub():
    if "auth_api" not in sys.modules:
        sys.modules["auth_api"] = types.ModuleType("auth_api")
    auth_api_stub = sys.modules["auth_api"]
    # job_store.py itself only needs get_db_conn/release_db_conn -- the rest
    # of this superset matches test_image_src_storage_variant.py's, purely
    # so an already-installed stub from that file (or vice versa) satisfies
    # both without either clobbering the other.
    for name, val in {
        "get_current_user": lambda: None,
        "db_cursor": lambda: None,
        "require_project_owner": lambda *a, **k: None,
        "MODELS_DIR": Path("/tmp"),
        "get_db_conn": lambda: None,
        "release_db_conn": lambda con: None,
        "get_latest_text_alignment": lambda *a, **k: None,
    }.items():
        if not hasattr(auth_api_stub, name):
            setattr(auth_api_stub, name, val)


def _load_real_job_store():
    _install_auth_api_stub()
    spec = importlib.util.spec_from_file_location(
        "_job_store_under_test", _SCRIPTS_DIR / "job_store.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Registered under a private key, not "job_store" -- see module
    # docstring for why the shared slot isn't safe to use here.
    sys.modules["_job_store_under_test"] = module
    spec.loader.exec_module(module)
    return module


job_store = _load_real_job_store()


class FakeCursor:
    """Records every execute() call and answers fetchone() with whatever a
    test pre-configures via `next_fetchone`."""

    def __init__(self):
        self.queries = []
        self.next_fetchone = None

    def execute(self, sql, params=()):
        self.queries.append((sql, params))

    def fetchone(self):
        return self.next_fetchone

    @property
    def rowcount(self):
        return 1

    def close(self):
        pass


class FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True


def _use_cursor(cursor: FakeCursor) -> None:
    """Points this loaded job_store instance's get_db_conn at a fresh
    cursor for this test."""
    job_store.get_db_conn = lambda: FakeConn(cursor)
    job_store.release_db_conn = lambda con: None


def test_neon_manifest_put_inserts_json_encoded_manifest():
    cur = FakeCursor()
    _use_cursor(cur)

    job_store.neon_manifest_put("abc123", {"foo": "bar"}, project_id=5)

    assert len(cur.queries) == 1
    sql, params = cur.queries[0]
    assert "INSERT INTO neon_manifests" in sql
    assert params == ("abc123", json.dumps({"foo": "bar"}), 5)


def test_neon_manifest_put_defaults_project_id_to_none():
    cur = FakeCursor()
    _use_cursor(cur)

    job_store.neon_manifest_put("abc123", {"foo": "bar"})

    _, params = cur.queries[0]
    assert params[2] is None


def test_neon_manifest_get_returns_dict_when_column_already_decoded():
    # psycopg2 auto-decodes JSONB columns to dict -- the common case.
    cur = FakeCursor()
    cur.next_fetchone = ({"foo": "bar"},)
    _use_cursor(cur)

    assert job_store.neon_manifest_get("abc123") == {"foo": "bar"}


def test_neon_manifest_get_decodes_json_string_column():
    # Defensive path mirroring manifest_get's (job_sessions) same fallback --
    # covers a driver/column configuration that hands back the raw string.
    cur = FakeCursor()
    cur.next_fetchone = ('{"foo": "bar"}',)
    _use_cursor(cur)

    assert job_store.neon_manifest_get("abc123") == {"foo": "bar"}


def test_neon_manifest_get_returns_none_when_missing():
    cur = FakeCursor()
    cur.next_fetchone = None
    _use_cursor(cur)

    assert job_store.neon_manifest_get("does-not-exist") is None


def test_cleanup_stale_neon_manifests_deletes_by_age():
    cur = FakeCursor()
    _use_cursor(cur)

    deleted = job_store.cleanup_stale_neon_manifests(max_age_days=1)

    assert deleted == 1  # FakeCursor.rowcount is fixed at 1
    sql, params = cur.queries[0]
    assert "DELETE FROM neon_manifests" in sql
    assert params == (1,)


def test_run_periodic_cleanup_includes_neon_manifests():
    cur = FakeCursor()
    _use_cursor(cur)
    job_store.cleanup_stale_uploads = lambda: 0
    job_store.cleanup_stale_sessions = lambda: 0

    result = job_store.run_periodic_cleanup()

    assert "neon_manifests_deleted" in result
