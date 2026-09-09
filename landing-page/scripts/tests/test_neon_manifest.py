"""Unit tests for neon_manifest.py's title helper.

Regression coverage for mothra#318: mei_files.name is stored as "{stem}.mei"
(useEncodingFlow.ts, correct for MeiTab.tsx's export button, which uses it
directly as a whole filename), but mei_api.py's create_edit_session used to
hand that same already-suffixed name straight to Neon as the manifest
title. Neon's own "Get MEI" download handler always appends ".mei" to the
title on top of whatever's there, so a title of "{stem}.mei" downloaded as
"{stem}.mei.mei". neon_manifest_title strips mei_files.name's own suffix
back off before it becomes the manifest title, matching the bare
"title": stem that encode_to_mei.build_neon_manifest already passes for the
CLI/standalone path.

No network, no DB -- mirrors test_paco_api.py's/test_staffline_adapter.py's
style (plain pytest functions, sys.path insert, dependency-free module
under test -- see neon_manifest.py's own docstring for why this couldn't
just be a helper inside mei_api.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neon_manifest import neon_manifest_title  # noqa: E402


def test_strips_mei_suffix():
    assert neon_manifest_title("MS234_005v.mei") == "MS234_005v"


def test_case_insensitive_suffix():
    assert neon_manifest_title("MS234_005v.MEI") == "MS234_005v"


def test_leaves_bare_stem_untouched():
    assert neon_manifest_title("MS234_005v") == "MS234_005v"


def test_leaves_unrelated_dot_untouched():
    # a stem that itself contains a dot but doesn't end in ".mei" must not
    # be truncated
    assert neon_manifest_title("MS234.005v") == "MS234.005v"
