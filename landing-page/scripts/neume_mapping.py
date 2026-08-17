"""neume_mapping.py — CSV-driven neume classification -> MEI <nc> component
templates, replacing encode_to_mei.py's old hardcoded _NEUME_NC_MAP dict.

Each CSV row (see landing-page/scripts/assets/mei_encoding/*.csv, vendored
from ic/core/data/train/csv-{square_notation,hufnagel}_neume_level_newest.csv maps 
one IC classification string to a literal MEI <neume> XML snippet, one <nc> per note component.
This module parses that snippet into a list of NcTemplate objects (raw XML
attributes + a parsed @intm interval + whether a <liquescent/> child is
present) rather than re-encoding a second hardcoded schema in Python — the
whole point of moving to CSVs is that a new attribute or notation type
shouldn't need a code change here.

The `width` column (bbox-splitting for a <zone> per component; see the
issue's discussion) is parsed, kept on NeumeEntry.width, and consumed by 
encode_to_mei.py's _component_zone_ids(), with graceful fallback on mismatch.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from config import MEI_ENCODING_DIR

_INTM_RE = re.compile(r"^(-?\d+)[Ss]$")

@dataclass
class NcTemplate:
    """One <nc> component template, parsed from a CSV row's <neume> snippet."""
    attrs: dict[str, str] = field(default_factory=dict)  # tilt / curve / ligated / con / ... — passed through verbatim
    intm: int = 0             # parsed @intm delta in diatonic steps; 0 = "same pitch as previous" (or unused, for the first component)
    liquescent: bool = False  # whether this <nc> had a <liquescent/> child

@dataclass
class NeumeEntry:
    classification: str
    components: list[NcTemplate]
    width: str  # raw width column text (e.g. "1", "[1, 1]"); unused for now — see module docstring

def _parse_intm(raw: str | None) -> int:
    if not raw:
        return 0
    m = _INTM_RE.match(raw.strip())
    if not m:
        print(f"[neume_mapping] unrecognized @intm value {raw!r} - treating as 0", file=sys.stderr)
        return 0
    return int(m.group(1))

def _parse_neume_snippet(mei_xml: str, classification: str) -> list[NcTemplate] | None:
    """Parse a CSV row's <neume>...</neume> snippet into an ordered list of
    NcTemplate, one per direct-child <nc>. Returns None (caller should skip
    the row) if the XML doesn't parse, or if the root isn't <neume> at all —
    clef/custos/divLine/accid rows fall in the latter case and are
    intentionally not modeled here, since nothing consumes them yet."""
    try:
        root = ET.fromstring(mei_xml)
    except ET.ParseError as e:
        print(f"[neume_mapping] skipping {classification!r}: malformed XML in 'mei' column: {e}", file=sys.stderr)
        return None
    if root.tag != "neume":
        return None
    components = []
    for nc in root.findall("nc"):
        attrs = dict(nc.attrib)
        intm = _parse_intm(attrs.pop("intm", None))
        liquescent = nc.find("liquescent") is not None
        components.append(NcTemplate(attrs=attrs, intm=intm, liquescent=liquescent))
    if not components:
        print(f"[neume_mapping] skipping {classification!r}: <neume> has no <nc> children", file=sys.stderr)
        return None
    return components

def load_neume_mapping(csv_path: Path) -> dict[str, NeumeEntry]:
    """Load one mapping CSV (columns: name, classification, width, mei —
    see the module docstring) into {classification.lower(): NeumeEntry}.
    Rows with a blank classification, unparseable XML, or a non-<neume>
    root are skipped (logged, not raised) so one bad row can't take down
    the whole load."""
    mapping: dict[str, NeumeEntry] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            classification = (row.get("classification") or "").strip()
            mei_xml = (row.get("mei") or "").strip()
            if not classification or not mei_xml:
                continue
            components = _parse_neume_snippet(mei_xml, classification)
            if components is None:
                continue
            key = classification.lower()
            if key in mapping:
                print(f"[neume_mapping] duplicate classification {classification!r} in {csv_path.name} - keeping first", file=sys.stderr)
                continue
            mapping[key] = NeumeEntry(
                classification=classification,
                components=components,
                width=(row.get("width") or "").strip()
            )
    return mapping

def parse_width(raw: str) -> list[float] | None:
    """Parse a CSV row's width column into a list of relative horizontal
    weights, one per <nc> component (e.g. "[1, 1]" -> [1.0, 1.0]), for
    splitting a compound neume's glyph zone into side-by-side per-component
    sub-zones (see encode_to_mei.py's build_mei). Returns None for anything
    that isn't a bracketed list -- single-component rows use a bare scalar
    like "1" or "2" in the bundled CSVs, and there's nothing to split for
    one component, so a plain scalar is "no width data", not an error --
    or that fails to parse as JSON."""
    raw = (raw or "").strip()
    if not raw.startswith("["):
        return None
    try: 
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, list) or not parsed or not all(
        isinstance(v, (int, float)) for v in parsed
    ):
        return None
    return [float(v) for v in parsed]

_BUNDLED_CSVS = {
    "square": "square.csv",
    "hufnagel": "hufnagel.csv",
}

@lru_cache(maxsize=None)
def resolve_neume_mapping(notation_type: str) -> dict[str, NeumeEntry]:
    """Load (and cache) one of the bundled preset mapping CSVs by name.
    Valid names: 'square', 'hufnagel' — see _BUNDLED_CSVS. Raises
    ValueError for anything else; there's no custom-upload path yet."""
    filename = _BUNDLED_CSVS.get(notation_type)
    if filename is None:
        raise ValueError(
            f"Unknown notation_type {notation_type!r} — expected one of {sorted(_BUNDLED_CSVS)}"
        )
    return load_neume_mapping(MEI_ENCODING_DIR / filename)