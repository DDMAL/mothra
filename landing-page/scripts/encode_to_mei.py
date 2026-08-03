#!/usr/bin/env python3
"""
encode_to_mei.py - Convert GameraXML (output from standalone IC) + Mothra inference JSON (stave detections) into
a pitch-less MEI-Neume file and Neon manifest JSON-LD

Usage:
    python scrupts/encode_to_mei.py \
        --gamera-xml path/to/classified.xml \
        --mothra-json path/to/inference_output.json \
        --image path/to/image.jpg \
        [--output-dir encoding-outputs/] \
        [--manuscript "CH-Fco Ms. 2_006r"]
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import struct
import uuid
from PIL import Image as _PIL_Image
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Optional
import xml.etree.ElementTree as ET
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from extract_ms_id import extract_manuscript_id
except ImportError:
    def extract_manuscript_id(filename: str) -> str:
        return Path(filename).stem
    

MEI_NS = "http://www.music-encoding.org/ns/mei"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
STAVE_BUFFER_PX = 20
SYLLABLE_GAP_MULTIPLIER = 1.5
_SKIP_CLASS_FRAGMENTS = frozenset({"custos", "divline", "division"})

@dataclass
class Glyph:
    id: str
    ulx: int
    uly: int
    ncols: int
    nrows: int
    class_name: str
    confidence: float
    state: str # MANUAL | AUTOMATIC | UNCLASSIFIEd

    @property
    def lrx(self) -> int:
        return self.ulx + self.ncols
    
    @property
    def lry(self) -> int:
        return self.uly + self.nrows
    
    @property
    def cy(self) -> float:
        return self.uly + self.nrows / 2
    
@dataclass
class StaveBbox:
    id: str
    ulx: int
    uly: int
    lrx: int
    lry: int
    line_ys: list[float] = field(default_factory=list)

    @property
    def cy(self) -> float:
        return (self.uly + self.lry) / 2
    
def parse_gamera_xml(path: Path) -> list[Glyph]:
    tree = ET.parse(path)
    root = tree.getroot()
    glyphs = []
    for glyph_el in root.iter("glyph"):
        ulx = int(glyph_el.get("ulx", 0))
        uly = int(glyph_el.get("uly", 0))
        ncols= int(glyph_el.get("ncols", 1))
        nrows = int(glyph_el.get("nrows", 1))
        class_name = "UNCLASSIFIED"
        confidence = 0.0
        state = "UNCLASSIFIED"
        ids_el = glyph_el.find("ids")
        if ids_el is not None:
            state = ids_el.get("state", "UNCLASSIFIED")
            id_el = ids_el.find("id")
            if id_el is not None:
                class_name = id_el.get("name", "UNCLASSIFIED")
                confidence = float(id_el.get("confidence", 0.0))
        glyphs.append(Glyph(
            id=str(uuid.uuid4()).replace("-", "")[:12],
            ulx=ulx, uly=uly, ncols=ncols, nrows=nrows,
            class_name=class_name, confidence=confidence, state=state,
        ))
    return glyphs
    

def parse_staves(path: Path) -> tuple[list[StaveBbox], int, int]:
    with open(path) as f:
        data = json.load(f)
    image_w = data.get("imageWidth", 0)
    image_h = data.get("imageHeight", 0)
    staves = []
    for ann in data.get("annotations", []):
        class_id = ann.get("classId", ann.get("class_id", -1))
        # clsasId 3 = staves in annotator format, classId 2 = staves in raw YOLO format
        if class_id not in (2, 3):
            continue
        x, y, w, h = ann["bbox"]
        staves.append(StaveBbox(
            id=str(uuid.uuid4()).replace("-", "")[:12],
            ulx=int(x), uly=int(y),
            lrx=int(x+w), lry=int(y+h)
        ))
    staves.sort(key=lambda s: s.uly)
    return staves, image_w, image_h

def image_dimensions(header: bytes) -> Optional[tuple]:
    """Return (width, height) from the first bytes of a JPEG, PNG, or TIFF file."""
    if header[:8] == b'\x89PNG\r\n\x1a\n':
        w, h = struct.unpack('>II', header[16:24])
        return w, h
    if header[:2] == b'\xff\xd8':
        i = 2
        while i < len(header) - 8:
            if header[i] != 0xff:
                break
            marker = header[i + 1]
            if marker in (0xC0, 0xC1, 0xC2):
                h, w = struct.unpack('>HH', header[i + 5:i + 9])
                return w, h
            seg_len = struct.unpack('>H', header[i + 2:i + 4])[0]
            i += 2 + seg_len
    if header[:2] in (b'II', b'MM'):
        bo = '<' if header[:2] == b'II' else '>'
        ifd_off = struct.unpack(bo + 'I', header[4:8])[0]
        if ifd_off + 2 > len(header):
            return None
        n = struct.unpack(bo + 'H', header[ifd_off:ifd_off + 2])[0]
        w = h = 0
        for j in range(n):
            off = ifd_off + 2 + j * 12
            if off + 12 > len(header):
                break
            tag, typ = struct.unpack(bo + 'HH', header[off:off + 4])
            if tag in (256, 257):
                fmt = bo + ('I' if typ == 4 else 'H')
                val = struct.unpack(fmt, header[off + 8:off + 8 + struct.calcsize(fmt)])[0]
                if tag == 256: w = val
                else: h = val
        if w and h:
            return w, h
    return None

def assign_glyphs_to_staves(
        glyphs: list[Glyph], staves: list[StaveBbox], page_w: int, page_h: int
) -> tuple[dict[int, list[Glyph]], list[StaveBbox]]:
    """Assign each glyph to a stave.

    Detected staves (usually from YOLO stave-class detections) can miss real
    staff systems entirely — confirmed on a real manuscript page where only 4
    of 10 real staff-system rows had any stave-class detection at all. Rather
    than force every glyph onto whichever detected stave is nearest (which
    corrupts note ordering and pitch for every missed row, since glyphs from
    multiple distinct physical rows end up interleaved by x-position in one
    stave's bucket), this reconciles independently-clustered row-groups
    against the detected staves by Y-range overlap: a row that overlaps a
    detected stave is assigned to it as before; a row that overlaps nothing
    is a system the detector missed, and gets a synthesized stave of its own.

    Returns (glyphs_by_stave, staves) — `staves` may be LONGER than the input
    (synthesized entries appended at the end). Callers must build zones/
    <sb>/<clef> from the RETURNED staves list, not the one passed in.
    """
    staves = list(staves)
    if not staves:
        result: dict[int, list[Glyph]] = {-1: list(glyphs)}
        return result, staves

    result = {i: [] for i in range(len(staves))}
    row_groups = _cluster_glyphs_into_staves(glyphs, page_w, page_h, id_prefix="row")

    for row_stave, members in row_groups:
        best_idx, best_overlap = None, 0
        for i, stave in enumerate(staves):
            lo, hi = stave.uly - STAVE_BUFFER_PX, stave.lry + STAVE_BUFFER_PX
            overlap = min(row_stave.lry, hi) - max(row_stave.uly, lo)
            if overlap > best_overlap:
                best_overlap, best_idx = overlap, i
        if best_idx is not None:
            result[best_idx].extend(members)
        else:
            new_idx = len(staves)
            staves.append(row_stave)
            result[new_idx] = list(members)
    for idx in result:
        result[idx].sort(key=lambda g: g.ulx)
    return result, staves

def cluster_into_syllables(glyphs: list[Glyph], gap_mult: float = SYLLABLE_GAP_MULTIPLIER, ) -> list[list[Glyph]]:
    if not glyphs:
        return []
    threshold = gap_mult * median(g.ncols for g in glyphs)
    clusters: list[list[Glyph]] = [[glyphs[0]]]
    for glyph in glyphs[1:]:
        if glyph.ulx - clusters[-1][-1].lrx > threshold:
            clusters.append([glyph])
        else:
            clusters[-1].append(glyph)
    return clusters

def find_best_syllable_match(
        cluster: list[Glyph], syl_boxes: list[dict], used: set[int]
) -> Optional[int]:
    """Match a neume cluster to the syl_boxes entry (by index) with the largest
    x-overlap on the same row (both segmentations read the stave L-to-R).
    Entries already claimed by an earlier cluster (via `used`) are skipped —
    Neon's resize action mutates the one zone all referencing elements share,
    so two syllables pointing at the same zone would silently move together.
    Returns None when nothing overlaps, no alignment was supplied, or every
    candidate on this row is already used."""
    if not syl_boxes or not cluster:
        return None
    c_ulx = min(g.ulx for g in cluster)
    c_lrx = max(g.lrx for g in cluster)
    c_cy = sum(g.cy for g in cluster) / len(cluster)
    best_idx, best_overlap = None, 0
    for i, box in enumerate(syl_boxes):
        if i in used:
            continue
        b_ulx, b_uly = box["ul"]
        b_lrx, b_lry = box["lr"]
        b_cy = (b_uly + b_lry) / 2
        if abs(b_cy - c_cy) > (b_lry - b_uly) * 4:
            continue  # different row band
        overlap = min(c_lrx, b_lrx) - max(c_ulx, b_ulx)
        if overlap > best_overlap:
            best_idx, best_overlap = i, overlap
    return best_idx

def _syl_box_valid(box: dict, image_w: int, image_h: int) -> bool:
    """A syl_boxes entry is only usable as a zone if it's a well-formed box
    that actually falls within the page — guards against a stale/mismatched-
    resolution text_alignment row (see _resolve_hints in tasks_encode.py,
    which has no dimension cross-check against the image being encoded)."""
    try:
        ulx, uly = box["ul"]
        lrx, lry = box["lr"]
    except (KeyError, TypeError, ValueError):
        return False
    return 0 <= ulx < lrx <= image_w and 0 <= uly < lry <= image_h


def parse_yolo_stave_hints(yolo_txt: str, img_w: int, img_h: int) -> list[StaveBbox]:
    """Parse YOLO annotation text into staff-line glyphs, then cluster them
    into per-system StaveBbox groups via _staves_from_staff_lines — the same
    clustering already used for the GameraXML path. A single detected line
    is not a stave; ~4-5 of them grouped together are.
    """
    lines: list[Glyph] = []
    for i, line in enumerate(yolo_txt.strip().splitlines()):
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            _, cx, cy, bw, bh = (float(parts[0]), float(parts[1]), 
                                 float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            continue
        if bh == 0 or bw / bh <= 6:
            continue
        px_cx, px_cy = cx * img_w, cy * img_h
        px_bw, px_bh = bw * img_w, bh * img_h
        lines.append(Glyph(
            id=f"yolo-line-{i}",
            ulx=int(px_cx - px_bw / 2), uly=int(px_cy - px_bh / 2),
            ncols=max(1, int(px_bw)), nrows=max(1, int(px_bh)),
            class_name="staffline", confidence=1.0, state="AUTOMATIC",
        ))
    if not lines:
        return []
    return _staves_from_staff_lines(lines, img_w, img_h)

def _cluster_glyphs_into_staves(
        glyphs: list[Glyph], page_w: int, page_h: int, id_prefix: str = "auto"
) -> list[tuple[StaveBbox, list[Glyph]]]:
    """Cluster glyphs into stave-sized row groups by Y-center gap, pairing each
    synthesized StaveBbox with the exact glyphs that produced it. Shared by
    estimate_staves_from_glyphs's no-stave-data fallback and by
    assign_glyphs_to_staves's missed-stave recovery (see there)."""
    if not glyphs:
        return []
    neume_like = [g for g in glyphs if g.nrows > 0 and g.ncols / g.nrows < 8]
    if not neume_like:
        neume_like = glyphs

    avg_h = median(g.nrows for g in neume_like)
    # Tighter outlier cutoff (x2 instead of x3) reduces bridging by tall glyphs
    representative = [g for g in neume_like if g.nrows <= avg_h * 2] or neume_like

    sorted_glyphs = sorted(representative, key=lambda g: g.cy)
    gap_threshold = avg_h * 1.5

    rows: list[list[Glyph]] = [[sorted_glyphs[0]]]
    for i in range(1, len(sorted_glyphs)):
        if sorted_glyphs[i].cy - sorted_glyphs[i - 1].cy > gap_threshold:
            rows.append([sorted_glyphs[i]])
        else:
            rows[-1].append(sorted_glyphs[i])

    pad = max(5, int(avg_h * 0.3))
    groups: list[tuple[StaveBbox, list[Glyph]]] = []
    for i, row in enumerate(rows):
        h = max(g.lry for g in row) + pad - max(0, min(g.uly for g in row) - pad)
        est_uly = max(0, min(g.uly for g in row) - pad)
        stave = StaveBbox(
            id=f"{id_prefix}-{i}",
            ulx=max(0, min(g.ulx for g in row) - pad),
            uly=max(0, min(g.uly for g in row) - pad),
            lrx=min(page_w, max(g.lrx for g in row) + pad),
            lry=min(page_h, max(g.lry for g in row) + pad),
            line_ys=[est_uly + h * j / 3 for j in range(4)],
        )
        groups.append((stave, row))
    return groups


def estimate_staves_from_glyphs(
    glyphs: list[Glyph], page_w: int, page_h: int
) -> list[StaveBbox]:
    """Estimate stave bounding boxes from GameraXML glyphs.

    Primary strategy: cluster the detected staff lines (wide, flat glyphs with
    aspect ratio ≥ 8 spanning >20% of page width).  These are highly reliable
    stave anchors and are unaffected by text characters that fill inter-stave
    gaps and cause the neume-Y-clustering approach to collapse all staves into
    one.

    Fallback: if too few staff lines are available, cluster neume-like glyphs
    by Y-center gap (original approach with tighter outlier filtering).
    """
    if not glyphs:
        return [StaveBbox(id="synth-0", ulx=0, uly=0, lrx=page_w, lry=page_h)]

    # ── Strategy 1: use staff line glyphs ──────────────────────────────────
    # Staff lines span most of the page width and have ncols >> nrows.
    min_line_width = page_w * 0.2
    staff_lines = [
        g for g in glyphs
        if g.nrows > 0 and g.ncols / g.nrows >= 8 and g.ncols >= min_line_width
    ]

    if len(staff_lines) >= 4:
        staves = _staves_from_staff_lines(staff_lines, page_w, page_h)
        if staves:
            return staves

    # ── Strategy 2: neume Y-gap clustering (fallback) ──────────────────────
    return [stave for stave, _ in _cluster_glyphs_into_staves(glyphs, page_w, page_h)]


def _staves_from_staff_lines(
    staff_lines: list[Glyph], page_w: int, page_h: int
) -> list[StaveBbox]:
    """Build stave bboxes from clustered staff line glyphs."""
    sorted_lines = sorted(staff_lines, key=lambda g: g.cy)

    # Estimate typical intra-stave line spacing from the smaller half of gaps.
    gaps_in_order = [
        sorted_lines[i].cy - sorted_lines[i - 1].cy
        for i in range(1, len(sorted_lines))
    ]
    small_gaps = sorted(gaps_in_order)[: max(1, len(gaps_in_order) // 2)]
    typical_spacing = median(small_gaps)
    inter_stave_threshold = max(typical_spacing * 2.5, typical_spacing + 20)

    # First-pass: split into clusters at inter-stave gaps.
    clusters: list[list[Glyph]] = [[sorted_lines[0]]]
    for i in range(1, len(sorted_lines)):
        if gaps_in_order[i - 1] > inter_stave_threshold:
            clusters.append([sorted_lines[i]])
        else:
            clusters[-1].append(sorted_lines[i])

    # Second-pass: recursively split any cluster containing an internal gap
    # notably bigger than the typical intra-stave spacing. A single,
    # non-recursive split (the old "only clusters >5 lines, split once"
    # approach) misses staves whose detector emits paired partial detections
    # (e.g. top-half + bottom-half per stave) — confirmed on a real page
    # where a 6-line cluster spanning 3 real staves only ever got split once,
    # leaving 2 of them merged. Recursing until no sub-cluster has an
    # oversized gap resolves this without needing a fixed line-count gate.
    def _split_oversized(cluster: list[Glyph]) -> list[list[Glyph]]:
        if len(cluster) <= 1:
            return [cluster]
        c_sorted = sorted(cluster, key=lambda g: g.cy)
        c_gaps = [(c_sorted[i].cy - c_sorted[i - 1].cy, i)
                  for i in range(1, len(c_sorted))]
        split_gap, split_idx = max(c_gaps, key=lambda x: x[0])
        if split_gap <= typical_spacing * 1.8:
            return [cluster]
        return (_split_oversized(c_sorted[:split_idx])
                + _split_oversized(c_sorted[split_idx:]))
    
    split_clusters: list[list[Glyph]] = []
    for cluster in clusters:
        split_clusters.extend(_split_oversized(cluster))

    # Build bboxes: pad above/below so neumes that sit on the staff are included.
    staves = []
    for i, cluster in enumerate(split_clusters):
        if not cluster:
            continue
        top_y = min(g.cy for g in cluster)
        bot_y = max(g.cy for g in cluster)
        # Tight bounds: half a line-spacing above/below the outermost line centers.
        # Using cy (not uly/lry) prevents line thickness from inflating the zone.
        pad = max(5, int(typical_spacing * 0.5))
        _raw_ys = sorted(g.cy for g in cluster)
        _deduped: list[float] = []
        for _y in _raw_ys:
            if not _deduped or _y - _deduped[-1] > 5:
                _deduped.append(_y)
        staves.append(StaveBbox(
            id=f"auto-{i}",
            ulx=max(0, min(g.ulx for g in cluster)),
            uly=max(0, int(top_y - pad)),
            lrx=min(page_w, max(g.lrx for g in cluster)),
            lry=min(page_h, int(bot_y + pad)),
            line_ys=_deduped,
        ))
    return staves

@dataclass
class _NcSpec:
    tilt: str = ""       # "" plain | "n" virga stem | "se" inclinatum diamond
    quilisma: bool = False
    y_fraction: float = 0.5

# Ordered nc specs per neume type (one entry = one note component).
# Ichiro appends variant codes (e.g. "clivis2a") — _nc_specs_for strips them.
_NEUME_NC_MAP: dict[str, list[_NcSpec]] = {
    # ── single-note ────────────────────────────────────────────────────────
    "punctum":              [_NcSpec(y_fraction=0.5)],
    "virga":                [_NcSpec(tilt="n", y_fraction=0.5)],
    "quilisma":             [_NcSpec(quilisma=True, y_fraction=0.5)],
    "inclinatum":           [_NcSpec(tilt="se", y_fraction=0.5)],
    "oriscus":              [_NcSpec(y_fraction=0.5)],
    # ── two-note ───────────────────────────────────────────────────────────
    "podatus":              [_NcSpec(y_fraction=0.75), _NcSpec(y_fraction=0.25)],          # ascending (= pes)
    "pes":                  [_NcSpec(y_fraction=0.75), _NcSpec(y_fraction=0.25)],
    "clivis":               [_NcSpec(y_fraction=0.25), _NcSpec(y_fraction=0.75)],          # descending
    "distropha":            [_NcSpec(y_fraction=0.4), _NcSpec(y_fraction=0.6)],
    "bivirga":              [_NcSpec(tilt="n", y_fraction=0.4), _NcSpec(tilt="n", y_fraction=0.6)],
    # ── three-note ─────────────────────────────────────────────────────────
    "torculus":             [_NcSpec(y_fraction=0.75), _NcSpec(y_fraction=0.25), _NcSpec(y_fraction=0.75)],
    "porrectus":            [_NcSpec(y_fraction=0.25), _NcSpec(y_fraction=0.75), _NcSpec(y_fraction=0.25)],
    "scandicus":            [_NcSpec(y_fraction=0.83), _NcSpec(y_fraction=0.5), _NcSpec(tilt="n", y_fraction=0.17)],
    "climacus":             [_NcSpec(y_fraction=0.17), _NcSpec(tilt="se", y_fraction=0.5), _NcSpec(tilt="se", y_fraction=0.83)],
    "tristropha":           [_NcSpec(y_fraction=0.33), _NcSpec(y_fraction=0.5), _NcSpec(y_fraction=0.67)],
    "trivirga":             [_NcSpec(tilt="n", y_fraction=0.33), _NcSpec(tilt="n", y_fraction=0.5), _NcSpec(tilt="n", y_fraction=0.67)],
    # ── four-note ──────────────────────────────────────────────────────────
    "torculusresupinus":    [_NcSpec(y_fraction=0.75), _NcSpec(y_fraction=0.25), _NcSpec(y_fraction=0.75), _NcSpec(y_fraction=0.25)],
    "porrectusflexus":      [_NcSpec(y_fraction=0.25), _NcSpec(y_fraction=0.75), _NcSpec(y_fraction=0.25), _NcSpec(y_fraction=0.75)],
    "scandicusflexus":      [_NcSpec(y_fraction=0.8), _NcSpec(y_fraction=0.5), _NcSpec(y_fraction=0.2), _NcSpec(y_fraction=0.5)],
    "climacusresupinus":    [_NcSpec(y_fraction=0.17), _NcSpec(tilt="se", y_fraction=0.5), _NcSpec(tilt="se", y_fraction=0.83), _NcSpec(y_fraction=0.5)],
}

_NEUME_PREFIXES = ("neume--", "neume.", "neume_", "neume/")

def _nc_specs_for(class_name: str) -> list[_NcSpec]:
    """Map an Ichiro class name to an ordered list of nc specs.

    Handles prefixes ("neume.", "neume--") and variant suffixes ("clivis2a" → "clivis").
    Falls back to a single plain nc for unknown types.
    """
    name = class_name.lower().strip()
    for prefix in _NEUME_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    # Strip trailing variant code: digits + optional letter ("2a", "3b", "1")
    base = re.sub(r"\d+[a-z]?$", "", name).strip("_.-")
    return _NEUME_NC_MAP.get(name) or _NEUME_NC_MAP.get(base) or [_NcSpec()]

_PITCH_NOTES = ["c", "d", "e", "f", "g", "a", "b"]

def _pitch_from_step(step: int, clef_note: str = "c", clef_oct: int = 4) -> tuple[str, str]:
    """Diatonic step offset from clef note → (pname, oct).
    Positive step = below clef (lower pitch); negative = above (higher pitch).
    """
    clef_abs = clef_oct * 7 + _PITCH_NOTES.index(clef_note)
    note_abs = clef_abs - step
    return _PITCH_NOTES[note_abs % 7], str(note_abs // 7)

def _nc_pitch(nc_cy: float, line_ys: list[float], clef_line: int = 3) -> tuple[str, str]:
    """Return (pname, oct) for a note at nc_cy given staff line Y positions.

    line_ys must be sorted ascending (smallest Y = top of image = highest pitch).
    Falls back to ("a", "3") when line data is unavailable.
    """
    if len(line_ys) < 2:
        return "a", "3"
    spacings = [line_ys[i+1] - line_ys[i] for i in range(len(line_ys) - 1)]
    line_spacing = sum(spacings) / len(spacings)
    clef_idx = len(line_ys) - clef_line
    clef_y = line_ys[clef_idx]
    step = round((nc_cy - clef_y) / (line_spacing / 2))
    return _pitch_from_step(step)

def _tag(local: str) -> str:
    return f"{{{MEI_NS}}}{local}"

def build_mei(
    glyphs_by_stave: dict[int, list[Glyph]],
    staves: list[StaveBbox],
    image_path: Path,
    image_w: int,
    image_h: int,
    manuscript_name: str,
    syllable_gap_mult: float = SYLLABLE_GAP_MULTIPLIER,
    text_alignment: dict | None = None,
    clef_shape: str = "C",
    clef_line: int = 3,
) -> bytes:
    ET.register_namespace("", MEI_NS)
    mei = ET.Element(_tag("mei"), {"meiversion": "5.0.0-dev"})

    # meiHead

    head = ET.SubElement(mei, _tag("meiHead"))
    file_desc = ET.SubElement(head, _tag("fileDesc"))
    title_stmt = ET.SubElement(file_desc, _tag("titleStmt"))
    title_el = ET.SubElement(title_stmt, _tag("title"))
    title_el.text = manuscript_name
    pub_stmt = ET.SubElement(file_desc, _tag("pubStmt"))
    ET.SubElement(pub_stmt, _tag("unpub"))

    music = ET.SubElement(mei, _tag("music"))

    #fac simile
    facsimile = ET.SubElement(music, _tag("facsimile"), {"type": "transcription"})
    surface = ET.SubElement(facsimile, _tag("surface"), {
        XML_ID: "surface-0001",
        "ulx": "0",
        "uly": "0",
        "lrx": str(image_w),
        "lry": str(image_h),
    })
    ET.SubElement(surface, _tag("graphic"), {
        "target": str(image_path),
        "width": f"{image_w}px",
        "height": f"{image_h}px",
    })

    # stave zones (used by <sb @facs>) + clef zones (used by <clef @facs>)
    stave_zone_ids: dict[int, str] = {}
    clef_zone_ids: dict[int, str] = {}
    for i, stave in enumerate(staves):
        zone_id = f"sz-{stave.id}"
        ET.SubElement(surface, _tag("zone"), {
            XML_ID: zone_id,
            "type": "staff",
            "ulx": str(stave.ulx),
            "uly": str(stave.uly),
            "lrx": str(stave.lrx),
            "lry": str(stave.lry),
        })
        stave_zone_ids[i] = zone_id
        # Clef zone: left edge of stave, roughly square (height ≈ staff height)
        clef_zone_id = f"cz-{stave.id}"
        stave_h = max(stave.lry - stave.uly, 1)
        ET.SubElement(surface, _tag("zone"), {
            XML_ID: clef_zone_id,
            "ulx": str(stave.ulx),
            "uly": str(stave.uly),
            "lrx": str(stave.ulx + stave_h),
            "lry": str(stave.lry),
        })
        clef_zone_ids[i] = clef_zone_id

    all_glyphs = [g for idx in sorted(glyphs_by_stave) for g in glyphs_by_stave[idx]]
    for glyph in all_glyphs:
        ET.SubElement(surface, _tag("zone"), {
            XML_ID: f"z-{glyph.id}",
            "ulx": str(glyph.ulx),
            "uly": str(glyph.uly),
            "lrx": str(glyph.lrx),
            "lry": str(glyph.lry),
        })
    syl_boxes = text_alignment.get("syl_boxes", []) if text_alignment else []

    syl_zone_ids: dict[int, str] = {}
    if syl_boxes:
        invalid_idx = {
            i for i, box in enumerate(syl_boxes)
            if not _syl_box_valid(box, image_w, image_h)
        }
        if len(invalid_idx) > len(syl_boxes) / 2:
            print(
                f" [text-alignment] {len(invalid_idx)}/{len(syl_boxes)} syl_boxes "
                f"fall outside the {image_w}x{image_h} page (likely a stale or "
                "mismatched-resolution text-finding result) — syllable text will "
                "still be used, but without bounding boxes",
                file=sys.stderr,
            )
        else:
            for i, box in enumerate(syl_boxes):
                if i in invalid_idx:
                    continue
                ulx, uly = box["ul"]
                lrx, lry = box["lr"]
                zone_id = f"zone-syl-{i}"
                ET.SubElement(surface, _tag("zone"), {
                    XML_ID: zone_id,
                    "ulx": str(int(ulx)),
                    "uly": str(int(uly)),
                    "lrx": str(int(lrx)),
                    "lry": str(int(lry)),
                })
                syl_zone_ids[i] = zone_id

    # body
    body = ET.SubElement(music, _tag("body"))
    mdiv = ET.SubElement(body, _tag("mdiv"))
    score = ET.SubElement(mdiv, _tag("score"))

    # sb-based format: one <staffDef>, one <staff>, <sb> milestones per stave
    score_def = ET.SubElement(score, _tag("scoreDef"))
    staff_grp = ET.SubElement(score_def, _tag("staffGrp"))
    staff_grp_id = str(uuid.uuid4()).replace("-", "")[:12]
    ET.SubElement(staff_grp, _tag("staffDef"), {
        XML_ID: f"staffdef-{staff_grp_id}",
        "n": "1",
        "lines": "4",
        "notationtype": "neume",
        "clef.shape": clef_shape,
        "clef.line": str(clef_line),
    })

    section = ET.SubElement(score, _tag("section"))
    staff_id = str(uuid.uuid4()).replace("-", "")[:12]
    staff_el = ET.SubElement(section, _tag("staff"), {
        XML_ID: f"staff-{staff_id}",
        "n": "1",
    })
    layer_id = str(uuid.uuid4()).replace("-", "")[:12]
    layer = ET.SubElement(staff_el, _tag("layer"), {
        XML_ID: f"layer-{layer_id}",
        "n": "1",
    })
    pb_id = str(uuid.uuid4()).replace("-", "")[:12]
    ET.SubElement(layer, _tag("pb"), {
        XML_ID: f"pb-{pb_id}",
        "facs": "#surface-0001",
    })

    used_syl_boxes: set[int] = set()
    for stave_idx in sorted(k for k in glyphs_by_stave if k >= 0):
        staff_glyphs = glyphs_by_stave[stave_idx]
        if not staff_glyphs:
            continue
        # <sb> marks the start of each stave and links it to its zone
        zone_id = stave_zone_ids.get(stave_idx, str(stave_idx))
        sb_attrs: dict[str, str] = {XML_ID: f"sb-{zone_id}"}
        if stave_idx in stave_zone_ids:
            sb_attrs["facs"] = f"#{zone_id}"
        ET.SubElement(layer, _tag("sb"), sb_attrs)
        # Clef must follow each <sb>; @facs anchors it to the stave's left edge
        clef_id = str(uuid.uuid4()).replace("-", "")[:12]
        clef_attrs: dict[str, str] = {
            XML_ID: f"clef-{clef_id}",
            "shape": clef_shape,
            "line": str(clef_line),
        }
        if stave_idx in clef_zone_ids:
            clef_attrs["facs"] = f"#{clef_zone_ids[stave_idx]}"
        ET.SubElement(layer, _tag("clef"), clef_attrs)
        
        skip_ids = {
            g.id for g in staff_glyphs
            if (g.nrows > 0 and g.ncols / g.nrows >= 8)
            or any(frag in g.class_name.lower() for frag in _SKIP_CLASS_FRAGMENTS)
        }
        if skip_ids:
            skipped = [g for g in staff_glyphs if g.id in skip_ids]
            skip_types = ", ".join(sorted({g.class_name for g in skipped}))
            print(f" [stave {stave_idx}] skipping {len(skip_ids)} glyph(s): {skip_types}")
        neume_glyphs = [g for g in staff_glyphs if g.id not in skip_ids]
        for cluster in cluster_into_syllables(neume_glyphs, gap_mult=syllable_gap_mult):
            syllable_id = cluster[0].id
            syllable = ET.SubElement(layer, _tag("syllable"), {
                XML_ID: f"syllable-{syllable_id}",
            })
            syl_id = str(uuid.uuid4()).replace("-", "")[:12]
            syl_attrs: dict[str, str] = {XML_ID: f"syl-{syl_id}"}
            match_idx = find_best_syllable_match(cluster, syl_boxes, used_syl_boxes)
            if match_idx is not None:
                used_syl_boxes.add(match_idx)
                syl_text = syl_boxes[match_idx]["syl"]
                if match_idx in syl_zone_ids:
                    syl_attrs["facs"] = f"#{syl_zone_ids[match_idx]}"
            else:
                syl_text = "-"
            syl = ET.SubElement(syllable, _tag("syl"), syl_attrs)
            syl.text = syl_text
            for glyph in cluster:
                neume = ET.SubElement(syllable, _tag("neume"), {
                    XML_ID: f"neume-{glyph.id}",
                    "facs": f"#z-{glyph.id}",
                })
                stave = staves[stave_idx] if stave_idx < len(staves) else None
                line_ys = stave.line_ys if stave else []
                for j, spec in enumerate(_nc_specs_for(glyph.class_name)):
                    nc_id = glyph.id if j == 0 else f"{glyph.id}-{j}"
                    nc_cy = glyph.uly + spec.y_fraction * glyph.nrows
                    pname, oct_str = _nc_pitch(nc_cy, line_ys, clef_line)
                    nc_attrs: dict[str, str] = {
                        XML_ID: f"nc-{nc_id}",
                        "facs": f"#z-{glyph.id}",
                        "pname": pname,
                        "oct": oct_str,
                    }
                    if spec.tilt:
                        nc_attrs["tilt"] = spec.tilt
                    if spec.quilisma:
                        nc_attrs["quilisma"] = "true"
                    ET.SubElement(neume, _tag("nc"), nc_attrs)
    
    _XML_DECLARATION = '<?xml version="1.0" encoding="UTF-8"?>\n'
    _XML_MODEL_PI = (
        '<?xml-model href="https://music-encoding.org/schema/dev/mei-all.rng"'
        ' type="application/xml" schematypens="http://relaxng.org/ns/structure/1.0"?>\n'
        '<?xml-model href="https://music-encoding.org/schema/dev/mei-all.rng"'
        ' type="application/xml" schematypens="http://purl.oclc.org/dsdl/schematron"?>\n'
    )
    ET.indent(mei, space=" ")
    xml_str = _XML_DECLARATION + _XML_MODEL_PI + ET.tostring(mei, encoding="unicode")
    return xml_str.encode("utf-8")

def scale_facsimile(mei_bytes: bytes, factor: float) -> bytes:
    """Scale every facsimile zone coordinate by factor (port of Rodan mei_resize.py)."""
    if factor == 1.0:
        return mei_bytes    
    root = ET.fromstring(mei_bytes)
    for el in root.iter():
        for attr in ("ulx", "uly", "lrx", "lry"):
            val = el.get(attr)
            if val is not None:
                try:
                    el.set(attr, str(round(int(val) * factor)))
                except (ValueError, TypeError):
                    pass
        for attr in ("width", "height"):
            val = el.get(attr)
            if val is not None and val.endswith("px"):
                try:
                    el.set(attr, f"{round(int(val[:-2]) * factor)}px")
                except (ValueError, TypeError):
                    pass
    ET.register_namespace("", MEI_NS)
    ET.indent(root, space=" ")
    xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml_model_pi = (
        '<?xml-model href="https://music-encoding.org/schema/dev/mei-all.rng"'
        ' type="application/xml" schematypens="http://relaxng.org/ns/structure/1.0"?>\n'
        '<?xml-model href="https://music-encoding.org/schema/dev/mei-all.rng"'
        ' type="application/xml" schematypens="http://purl.oclc.org/dsdl/schematron"?>\n'
    )
    xml_str = xml_declaration + xml_model_pi + ET.tostring(root, encoding="unicode")
    return xml_str.encode("utf-8")
    
REQUIRED_MEI_VERSION = "5.0.0-dev"

def validate_mei(xml_bytes: bytes) -> list[str]:
    warnings = []
    root = ET.fromstring(xml_bytes)

    version = root.get("meiversion", "")
    if version != REQUIRED_MEI_VERSION:
        warnings.append(
            f"meiversion='{version}' — Neon requires '{REQUIRED_MEI_VERSION}' (schema will report INVALID)"
        )

    facsimiles = list(root.iter(_tag("facsimile")))
    for fac in facsimiles:
        if fac.get("type") != "transcription":
            warnings.append(
                "facsimile missing @type='transcription' — Verovio won't apply pixel coordinate conversion"
            )

    layers = list(root.iter(_tag("layer")))
    for layer in layers:
        children = list(layer)
        if not children or children[0].tag != _tag("pb"):
            warnings.append(
                "layer must start with <pb> before any <sb> — required by Neon's ConvertMei.ts"
            )
        pbs = [c for c in children if c.tag == _tag("pb")]
        for pb in pbs:
            if not pb.get("facs"):
                warnings.append("pb missing @facs pointing to <surface>")

    zones: dict[str, str] = {}
    for zone in root.iter(_tag("zone")):
        zid = zone.get(XML_ID, "")
        if zid:
            zones[zid] = zone.get("type", "")

    surfaces = list(root.iter(_tag("surface")))
    surface_bounds = None
    if surfaces:
        try:
            surface_bounds = tuple(
                int(surfaces[0].get(a, 0)) for a in ("ulx", "uly", "lrx", "lry")
            )
        except ValueError:
            pass


    def check_facs(el, label):
        facs = el.get("facs", "")
        if not facs:
            warnings.append(f"{label} missing @facs")
            return
        ref = facs.lstrip("#")
        if ref not in zones:
            warnings.append(f"{label} @facs '{facs}' does not resolve to any zone")
    
    #zone bounding boxes
    for zone in root.iter(_tag("zone")):
        zid = zone.get(XML_ID, "<no-id>")
        try:
            ulx, uly = int(zone.get("ulx", 0)), int(zone.get("uly", 0))
            lrx, lry = int(zone.get("lrx", 0)), int(zone.get("lry", 0))
        except ValueError:
            warnings.append(f"zone {zid}: non-integer coordinate")
            continue
        if ulx < 0 or uly < 0 or lrx < 0 or lry < 0:
            warnings.append(f"zone {zid}: negative coordinate ({ulx}, {uly}, {lrx}, {lry})")
        if ulx >= lrx or uly >= lry:
            warnings.append(f"zone {zid}: degenerate bbox ({ulx}, {uly})-({lrx}, {lry})")
        if surface_bounds and (
            ulx < surface_bounds[0] or uly < surface_bounds[1]
            or lrx > surface_bounds[2] or lry > surface_bounds[3]
        ):
            warnings.append(f"zone {zid}: extends outside surface bounds")

    # sb-based: exactly one <staff>, each <sb> must resolve to a type="staff" zone
    staves = list(root.iter(_tag("staff")))
    if not staves:
        warnings.append("no <staff> elements found - output is empty")
    sbs = list(root.iter(_tag("sb")))
    if not sbs:
        warnings.append("no <sb> elements found - stave zones not linked")
    for sb in sbs:
        sbid = sb.get(XML_ID, "?")
        facs = sb.get("facs", "")
        ref = facs.lstrip("#")
        if not ref:
            warnings.append(f"sb {sbid}: missing @facs")
        elif ref not in zones:
            warnings.append(f"sb {sbid}: @facs '{facs}' unresolved")
        elif zones[ref] != "staff":
            warnings.append(f"sb {sbid}: zone '{ref}' has type='{zones[ref]}', expected 'staff'")

    # syllable: xml:id required (Neon references syllables by id)
    for syllable in root.iter(_tag("syllable")):
        if not syllable.get(XML_ID):
            warnings.append("syllable missing xml:id")

    # neume: xml:id + facs
    neumes = list(root.iter(_tag("neume")))
    if not neumes:
        warnings.append("no <neume> elements found - no glyphs encoded")
    for neume in neumes:
        nid = neume.get(XML_ID, "")
        if not nid:
            warnings.append("neume missing xml:id")
        check_facs(neume, f"neume {nid or '?'}")

    # nc: xml:id + facs + pname + oct
    for nc in root.iter(_tag("nc")):
        ncid = nc.get(XML_ID, "")
        check_facs(nc, f"nc {ncid or '?'}")
        for attr in ("pname", "oct"):
            if not nc.get(attr):
                warnings.append(f"nc {ncid or '?'}: missing @{attr}")

    return warnings

def build_neon_manifest(mei_bytes: bytes, image_ref: str, stem: str) -> dict:
    mei_b64 = base64.b64encode(mei_bytes).decode()
    return {
        "@context": [
            "http://www.w3.org/ns/anno.jsonld",
            {
                "schema": "http://schema.org/",
                "title": "schema:name",
                "timestamp": "schema:dateModified",
                "image": {"@id": "schema:image", "@type": "@id"},
                "mei_annotations": {"@id": "Annotation", "@type": "@id", "@container": "@list"},
            },
        ],
        "@id": f"urn:uuid:{uuid.uuid4()}",
        "title": stem,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image": image_ref,
        "mei_annotations": [
            {
                "id": f"urn:uuid:{uuid.uuid4()}",
                "type": "Annotation",
                "body": f"data:application/mei+xml;base64,{mei_b64}",
                "target": image_ref,
            }
        ],
    }

def main():
    parser = argparse.ArgumentParser(
        description="GameraXML + Mothra inference JSON -> pitch-less MEI + Neon manifest."
    )
    parser.add_argument("--gamera-xml", required=True, type=Path, metavar="PATH")
    parser.add_argument("--mothra-json", required=True, type=Path, metavar="PATH")
    parser.add_argument("--image", required=True, type=Path, metavar="PATH")
    parser.add_argument("--output-dir", type=Path, default=Path("encoding-outputs"))
    parser.add_argument("--manuscript", type=str, default=None)
    parser.add_argument("--scale", type=float, default=1.0, metavar="FACTOR",
                        help="multiply all facsimile zone coordinates by this factor ")
    parser.add_argument("--syllable-gap-mult", type=float, default=SYLLABLE_GAP_MULTIPLIER,
                        metavar="FLOAT", help="gap-to-median-glyph-width ratio for syllable clustering (default: 1.5)",)
    args = parser.parse_args()

    stem = args.image.stem
    ms_name = args.manuscript or extract_manuscript_id(args.image.name)

    out_dir = args.output_dir / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing GameraXML: {args.gamera_xml}")
    glyphs = parse_gamera_xml(args.gamera_xml)
    print(f" {len(glyphs)} glyphs loaded")

    print(f"Parsing stave detections: {args.mothra_json}")
    staves, image_w, image_h = parse_staves(args.mothra_json)
    
    # Auto-detect scale: if --image is a different resolution than that of Mothra JSON, all coordinates need scaling
    auto_scale = 1.0
    try:
        with _PIL_Image.open(args.image) as _img:
            actual_w, _ = _img.size
        if image_w and image_w != actual_w:
            auto_scale = actual_w / image_w
    except Exception as e:
        print(f"[warn] could not read image dimensions for auto-scale: {e}", file=sys.stderr)
    
    scale = args.scale if args.scale is not None else auto_scale
    if scale != 1.0:
        source = "--scale override" if args.scale is not None else "auto-detected"
        print (f"Scaling facsimile coordinates by {scale:.4g} times ({source})")
    print(f"{len(staves)} staves found")

    glyphs_by_stave, staves = assign_glyphs_to_staves(glyphs, staves, image_w, image_h)
    assigned = sum(len(v) for k, v in glyphs_by_stave.items() if k >= 0)
    print(f" {assigned} glyphs assigned across {len([k for k in glyphs_by_stave if k >= 0])} staves")

    mei_bytes = build_mei(glyphs_by_stave, staves, args.image.resolve(), image_w, image_h, ms_name, syllable_gap_mult=args.syllable_gap_mult,)
    if scale != 1.0:
        mei_bytes = scale_facsimile(mei_bytes, scale)
    for w in validate_mei(mei_bytes):
        print(f"[warn] {w}", file=sys.stderr)

    mei_path = out_dir / f"{stem}.mei"
    mei_path.write_bytes(mei_bytes)
    print(f"MEI written: {mei_path}")

    manifest = build_neon_manifest(mei_bytes, str(args.image.resolve()), stem)
    manifest_path = out_dir / f"{stem}_manifest.jsonld"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written: {manifest_path}")
    print(f"\nLoad in Neon: editor.html?manifest={manifest_path.resolve()}")

if __name__ == "__main__":
    main()