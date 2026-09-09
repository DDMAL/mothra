"""Pure helper for building the Neon batch-editor manifest's "title" field.

Kept dependency-free on purpose (no fastapi/auth_api import), mirroring
paco_api.py's/staffline_adapter.py's split of DB/framework-independent
logic out of the routers that use it -- mei_api.py pulls in auth_api (and
so fastapi, psycopg2, jose, bcrypt, slowapi, plus a hard MOTHRA_SECRET
requirement at import time), which the CI "DB-independent scripts tests"
step (.github/workflows/tests.yml) does not install, so a helper meant to
be unit tested from tests/ can't live directly in mei_api.py.
"""


def neon_manifest_title(mei_name: str) -> str:
    """Strip mei_files.name's own ".mei" suffix before handing it to Neon
    as the manifest title.

    mei_files.name is stored as "{stem}.mei" (useEncodingFlow.ts) because
    MeiTab.tsx's export button uses it directly as a whole filename. Neon's
    own "Get MEI" download handler (neon/src/utils/EditControls.ts) always
    appends ".mei" to the manifest's title on top of whatever's there
    (matching encode_to_mei.build_neon_manifest's bare "title": stem for the
    CLI/standalone path) -- so handing Neon a title that already ends in
    ".mei" doubles it into "{stem}.mei.mei" on download (mothra#318).
    """
    return mei_name[:-4] if mei_name.lower().endswith(".mei") else mei_name
