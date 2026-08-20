#!/usr/bin/env python3
"""
encode_to_mei.py - Convert GameraXML (output from standalone IC) + Mothra inference JSON (stave detections) into
an MEI-Neume file and Neon manifest JSON-LD.

Every <nc> gets a pname/oct. Where it comes from is decided per glyph, by
_resolve_pitches:

  * normally from pitch_stage.py -- real pitch finding (algorithm #1 of the
    pitch-finding/ submodule), passed in as `pitch_map`: a per-class ink-centroid
    anchor on the glyph, the neume decomposed into its own notes from that one
    anchor, and the page's OWN detected clef glyph (whose measured staff line
    arrives as `clef_line_map` and is what this file then declares on <clef>);
  * otherwise from this file's older geometric placeholder -- the glyph bbox's
    vertical center converted to a diatonic step below an ASSUMED clef
    (clef_shape/clef_line, default C-clef/line 3), then @intm chaining. That is
    not a transcription, just something Verovio/Neon can place on the staff.

The placeholder still covers every glyph the algorithm reports it couldn't
resolve (no stave, no clef, no line coverage), and every deployment without a
pitch-finding checkout. A human corrects the result in Neon either way.

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

from neume_mapping import NcTemplate, SpecialEntry, resolve_neume_mapping, resolve_special_mapping, parse_width

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
STAFF_LINES = 4  # matches <staffDef lines="..."> below, and the stave zone's
                 # assumed line count (see build_mei's stave-zone construction)
_CLEF_PITCH_REF = {"C": ("c", 4), "F": ("f", 3)}
_XML_DECLARATION = '<?xml version="1.0" encoding="UTF-8"?>\n'
_XML_MODEL_PI = (
    '<?xml-model href="https://music-encoding.org/schema/dev/mei-all.rng"'
    ' type="application/xml" schematypens="http://relaxng.org/ns/structure/1.0"?>\n'
    '<?xml-model href="https://music-encoding.org/schema/dev/mei-all.rng"'
    ' type="application/xml" schematypens="http://purl.oclc.org/dsdl/schematron"?>\n'
)

def _serialize_mei(root: ET.Element) -> bytes:
    """Shared serialization tail for anything that produces/mutates a full
    MEI document (build_mei, scale_facsimile, verify_and_correct_syllables)."""
    ET.register_namespace("", MEI_NS)
    ET.indent(root, space=" ")
    xml_str = _XML_DECLARATION + _XML_MODEL_PI + ET.tostring(root, encoding="unicode")
    return xml_str.encode("utf-8")

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

def _typical_line_spacing(staves: list[StaveBbox]) -> Optional[float]:
    """Median intra-stave staff-line spacing across staves that carry real,
    multi-point line_ys — i.e. genuinely detected line positions, not the
    crude bbox-derived guess _cluster_glyphs_into_staves itself falls back
    to when a row has no real line data at all (that guess has no place
    contributing to an estimate of what "real" spacing looks like).

    Used by assign_glyphs_to_staves's missed-stave recovery path to give a
    reliable spacing figure to a row the detector missed entirely, instead
    of that row deriving its own (unreliable) spacing from its own glyphs'
    bounding box — see that function's docstring for why."""
    spacings: list[float] = []
    for stave in staves:
        ys = stave.line_ys
        if len(ys) >= 2:
            diffs = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
            spacings.append(median(diffs))
    return median(spacings) if spacings else None

def assign_glyphs_to_staves(
        glyphs: list[Glyph], staves: list[StaveBbox], page_w: int, page_h: int,
        allow_synthetic_lines: bool = False,
) -> tuple[dict[int, list[Glyph]], list[StaveBbox]]:
    """Assign each glyph to a stave.

    Two passes, in order:

    1. Direct claim: each glyph is assigned individually to whichever
       existing stave's own line band (uly..lry, padded by
       STAVE_BUFFER_PX) it falls nearest/within -- never to a stave more
       than STAVE_BUFFER_PX away. This is per-glyph, not per-row-blob, so a
       lyric-text glyph (or anything else) sitting between two
       independently-detected staves can never drag a REAL note that IS
       near its own stave along with it into the wrong bucket, and two
       real, adjacent, correctly-detected staves can never have their
       glyphs merged into just one of them and left the other with
       nothing -- confirmed as a real failure mode on a page where two
       genuinely-detected, Y-adjacent staves both ended up with zero
       glyphs after the OLD row-clustering-then-best-overlap-wins design
       let a lyric-text-bridged row blob's whole content get claimed by a
       third bucket instead.

    2. Whatever's left unclaimed by pass 1 (mostly lyric-text glyphs that
       don't sit close enough to any note-line band, plus -- crucially --
       every glyph belonging to a system the detector missed entirely) is
       clustered into rows exactly as before (_cluster_glyphs_into_staves),
       and EACH resulting row is matched to whichever existing stave its
       own aggregate bbox overlaps most (unchanged blob-level logic); a row
       that overlaps no existing stave at all is a genuinely missed system
       and gets a synthesized stave of its own, appended at the end of
       `staves`. This pass is unchanged from before this fix -- it only
       ever sees the reduced, already-narrowed-down leftover set, so it
       can no longer misroute a real note that pass 1 already claimed.

    That synthesized stave's line_ys (used for pitch computation -- see
    _step_from_y) would otherwise come purely from the missed row's own
    glyphs' bounding box (_cluster_glyphs_into_staves's crude fallback),
    which can be badly skewed by that row's own melodic range (a single
    unusually high/low note inflates the guessed staff height) -- confirmed
    on a real page as a visible, per-row-only vertical rendering offset
    (notes floating well above/below their real staff, since Neon/Verovio
    renders pitch, not raw pixel position). Passing this page's OWN
    already-reliable typical line spacing (from every OTHER, genuinely
    detected stave) into that fallback anchors the recovered row's
    estimated lines on real page geometry instead.

    allow_synthetic_lines is forwarded to that recovery clustering
    (_cluster_glyphs_into_staves) so a recovered stave's line_ys follows the
    same opt-in-fabrication rule as tier-3's own estimate_staves_from_glyphs
    fallback -- without this, a page could end up with tier-3 staves carrying
    synthetic pitch geometry while recovered staves silently didn't, even
    though the caller asked for synthetic lines everywhere.

    Returns (glyphs_by_stave, staves) — `staves` may be LONGER than the input
    (synthesized entries appended at the end). Callers must build zones/
    <sb>/<clef> from the RETURNED staves list, not the one passed in.
    """
    staves = list(staves)
    if not staves:
        result: dict[int, list[Glyph]] = {-1: list(glyphs)}
        return result, staves

    result: dict[int, list[Glyph]] = {i: [] for i in range(len(staves))}
    unclaimed: list[Glyph] = []

    def _claim(glyph: Glyph, candidates: list[StaveBbox]) -> Optional[int]:
        best_idx, best_dist = None, None
        for i, stave in candidates: 
            if stave.uly <= glyph.cy <= stave.lry:
                dist = 0.0
            else:
                dist = min(abs(glyph.cy - stave.uly), abs(glyph.cy - stave.lry))
            if dist <= STAVE_BUFFER_PX and (best_dist is None or dist < best_dist):
                best_dist, best_idx = dist, i
        return best_idx

    indexed_staves = list(enumerate(staves))
    for glyph in glyphs:
        best_idx = _claim(glyph, indexed_staves)
        if best_idx is not None:
            result[best_idx].append(glyph)
        else:
            unclaimed.append(glyph)

    if unclaimed:
        typical_spacing = _typical_line_spacing(staves)
        row_groups = _cluster_glyphs_into_staves(
            unclaimed, page_w, page_h, id_prefix="row", typical_line_spacing=typical_spacing,
            allow_synthetic_lines=allow_synthetic_lines,
        )

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

def _cluster_boxes_into_rows(boxes: list[dict], gap_mult: float = SYLLABLE_GAP_MULTIPLIER) -> list[list[dict]]:
    """Group one stave's syl_boxes into physical text rows by Y-gap —
    mirrors cluster_into_syllables's X-gap clustering, but on the other
    axis. A single stave's text can legitimately span more than one
    physical manuscript line: a long line wraps to a continuation row
    written directly above the next stave begins, still belonging (by
    this pipeline's manuscript convention, see _assign_boxes_to_staves)
    to the stave above it rather than the one below. Boxes are sorted by
    center-Y first so rows come out top-to-bottom; callers must sort each
    row left-to-right and concatenate rows in this order — never sort two
    rows' boxes together by X alone, or a wrapped row's leftmost token
    (written first on the page specifically because it's a continuation)
    sorts ahead of the row above it whenever it happens to sit further
    left, scrambling reading order (confirmed: this is exactly the failure
    _group_staves_by_row's docstring describes for pooled fragments, one
    axis over)."""
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda b: (b["ul"][1] + b["lr"][1]) / 2)
    heights = [b["lr"][1] - b["ul"][1] for b in ordered if b["lr"][1] > b["ul"][1]]
    threshold = gap_mult * median(heights) if heights else 0
    rows: list[list[dict]] = [[ordered[0]]]
    for box in ordered[1:]:
        prev_cy = (rows[-1][-1]["ul"][1] + rows[-1][-1]["lr"][1]) / 2
        cy = (box["ul"][1] + box["lr"][1]) / 2
        if cy - prev_cy > threshold:
            rows.append([box])
        else:
            rows[-1].append(box)
    return rows

def _assign_boxes_to_staves(staves: list[StaveBbox], boxes: list[dict]) -> dict[int, list[dict]]:
    """Assign each syl_box to a stave — a box strictly inside a stave's
    [uly, lry] band goes there; otherwise it goes to the nearest stave
    ABOVE it, never the nearest by raw symmetric distance. This matches
    this pipeline's manuscript convention: a stave's syllable text is
    always written directly below that same stave, with the next stave
    beginning below the text (stave1 -> text1 -> stave2 -> text2 -> ...) —
    so a box sitting between stave N and stave N+1 always belongs to
    stave N, even on the (real, observed) pages where the scribe left more
    space above the text than below it, which would otherwise put the box
    geometrically closer to stave N+1's top edge than to stave N's bottom
    edge. Only a box sitting above every stave (e.g. a rubric before the
    first system) falls back to nearest-by-distance. Every box is placed
    exactly once, deterministically — no 'best single match, everything
    else dropped' step here.

    Within a stave's bucket, boxes are ordered row-by-row (top to bottom,
    via _cluster_boxes_into_rows) then left-to-right within each row — NOT
    by a single flat X sort. A stave whose text wraps to a second physical
    row (see that helper's docstring) puts both rows' boxes in the same
    bucket; a flat X sort would interleave the two rows' tokens by raw
    X-position instead of preserving row order, which is a real observed
    bug (a wrapped row's continuation syllable sorting ahead of the row
    above it whenever it happens to sit further left on the page)."""
    result: dict[int, list[dict]] = {i: [] for i in range(len(staves))}
    if not staves:
        return result
    order = sorted(range(len(staves)), key=lambda i: staves[i].uly)
    for box in boxes:
        cy = (box["ul"][1] + box["lr"][1]) / 2
        inside = next((idx for idx in order if staves[idx].uly <= cy <= staves[idx].lry), None)
        if inside is not None:
            result[inside].append(box)
            continue
        above = [idx for idx in order if staves[idx].lry <= cy]
        if above:
            result[above[-1]].append(box)
            continue
        best_idx, best_dist = order[0], float("inf")
        for idx in order:
            stave = staves[idx]
            dist = min(abs(cy - stave.uly), abs(cy - stave.lry))
            if dist < best_dist:
                best_idx, best_dist = idx, dist
        result[best_idx].append(box)
    for idx, group in result.items():
        rows = _cluster_boxes_into_rows(group)
        for row in rows:
            row.sort(key=lambda b: b["ul"][0])
        result[idx] = [b for row in rows for b in row]
    return result

def _assign_glyphs_to_boxes(glyphs: list[Glyph], boxes: list[dict]) -> list[list[Glyph]]:
    """Assign each of one stave's neume glyphs to whichever syl_box (already
    sorted left to right) it's horizontally nearest to.

    This trusts mothra-text's own syllable segmentation as the authoritative
    syllable grouping, instead of independently re-deriving syllable
    boundaries from cluster_into_syllables's generic note-gap heuristic and
    then trying to reconcile the two after the fact (an earlier version of
    this fix did that, and on real melismatic pages either dropped every
    syl_box beyond a single best-overlap one, or — after that was fixed —
    still couldn't cleanly split a gap-clustered run into as many
    <syllable>s as there were real syl_boxes, and fell back to one big
    space-joined blob per cluster instead of one box each).

    A glyph whose center falls inside a box's own x-range goes there;
    anything else goes to the nearest box by X-distance. Every glyph lands
    in exactly one box's bucket — some buckets may end up empty (a syllable
    with no note under it), which is fine; see the caller.

    Containment is capped to boxes that aren't extreme width outliers
    relative to their row-mates (more than 4x the median syl_box width
    here). A stray/mis-detected syl_box (e.g. a quire mark or page
    signature mothra-text misreads as a short word) can end up with an
    anomalously wide X-range; since containment always wins outright over
    any other box's mere edge-distance, an oversized box's range can swallow
    every glyph on the whole stave into one syllable, leaving every real
    syl_box on that stave with none — confirmed as a real observed failure.
    An outlier-width box still competes for glyphs by ordinary edge-distance,
    it just loses its automatic containment win; this leaves every
    normal-width box's behavior (including legitimately wide multi-word
    boxes) completely unchanged."""
    if not boxes:
        return []
    result: list[list[Glyph]] = [[] for _ in boxes]
    widths = [b["lr"][0] - b["ul"][0] for b in boxes]
    typical_width = median(widths) if widths else 0
    outlier_width = 4 * typical_width if typical_width else float("inf")
    ranges = [(b["ul"][0], b["lr"][0]) for b in boxes]
    for glyph in sorted(glyphs, key=lambda g: g.ulx):
        cx = (glyph.ulx + glyph.lrx) / 2
        best_idx, best_dist = 0, float("inf")
        for i, (lo, hi) in enumerate(ranges):
            contains = lo <= cx <= hi and widths[i] <= outlier_width
            dist = 0 if contains else min(abs(cx - lo), abs(cx - hi))
            if dist < best_dist:
                best_idx, best_dist = i, dist
        result[best_idx].append(glyph)
    return result
def _build_syllable_units(
        neume_glyphs: list[Glyph],
        stave_boxes: list[dict],
        glyph_groups: list[list[Glyph]],
        syllable_gap_mult: float,
) -> list[tuple[str, Optional[dict], list[Glyph]]]:
    """Decide the final (syl_text, box, glyphs) triples for one stave, given
    its (possibly row-group-pooled-then-narrowed, see _stave_share_of_group)
    stave_boxes/glyph_groups. Extracted from build_mei's per-stave loop so
    build_mei and verify_and_correct_syllables compute this identically --
    one <syllable> per real syl_box when mothra-text supplied any for this
    stave, else a pure gap-based fallback with "-" text (this file's
    original pre-text-alignment behavior)."""
    if stave_boxes:
        return list(zip((b["syl"] for b in stave_boxes), stave_boxes, glyph_groups, strict=True))
    return [
        ("-", None, cluster)
        for cluster in cluster_into_syllables(neume_glyphs, gap_mult=syllable_gap_mult)
    ]

    
def _group_staves_by_row(staves: list[StaveBbox], n_detected_staves: int) -> list[set[int]]:
    """Connected components of stave indices that are fragments of one
    physical manuscript row. Exists ONLY so build_mei's syllable-matching
    prepass can recognize that N stave_idx buckets belong to the same row
    (assign_glyphs_to_staves's Y-gap glyph clustering can split one row's
    glyphs across multiple stave_idx values) — it must NOT be used for
    zone/clef/pitch building, which keeps keying off the un-grouped
    stave_idx exactly as before.

    A pair is only ever merged if AT LEAST ONE of the two indices is a
    stave assign_glyphs_to_staves itself synthesized — i.e. index >=
    n_detected_staves, a row_group whose Y-band didn't overlap ANY
    originally-detected stave (see that function's docstring: "a row that
    overlaps nothing is a system the detector missed, and gets a
    synthesized stave of its own"). Two originally-detected staves (both
    indices < n_detected_staves) are NEVER merged with each other, no
    matter how close together or similar in size — those are staff-line-
    detector-confirmed, independent physical systems, and pooling two of
    them (see _pool_group_syllable_data) reassigns their real syl_boxes by
    raw X-position with no Y discrimination at all, which can visibly
    interleave two unrelated manuscript lines' text.

    This function previously tried to infer "fragment-ness" purely from
    geometry (Y-adjacency plus bbox-height lopsidedness). That was
    empirically insufficient: confirmed on a real page where two
    genuinely distinct, independently detected rows were BOTH Y-adjacent
    within STAVE_BUFFER_PX AND lopsided in height, so the old check merged
    them — the resulting syllable sequence alternated between the two
    unrelated rows one-for-one (e.g. "mi cum nus dum sal pro..." was
    really "mi[row A] cum[row B] nus[row A] dum[row B] ..." zippered
    together by the shared X-sort). Tying the check to
    assign_glyphs_to_staves's own synthetic-stave signal instead of
    inferring "fragment-ness" from geometry is a strictly narrower, safer
    rescue: it may not catch every fragmentation case the geometric
    heuristic did (a fragment that DID overlap some other real detected
    stave, rather than getting promoted to synthetic, is not repaired
    here and falls back to "-" same as before this whole fix), but it can
    never merge two staves the detector itself vouched for independently."""
    n = len(staves)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    def _adjacent(a: StaveBbox, b: StaveBbox) -> bool:
        lo, hi = a.uly - STAVE_BUFFER_PX, a.lry + STAVE_BUFFER_PX
        return min(b.lry, hi) - max(b.uly, lo) > 0

    for i in range(n):
        for j in range(i + 1, n):
            involves_synthetic = i >= n_detected_staves or j >= n_detected_staves
            if involves_synthetic and _adjacent(staves[i], staves[j]):
                union(i, j)

    groups: dict[int, set[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), set()).add(i)
    return list(groups.values())

def _pool_group_syllable_data(
        group: set[int],
        boxes_by_stave: dict[int, list[dict]],
        neume_glyphs_by_stave: dict[int, list[Glyph]],
) -> tuple[list[dict], list[list[Glyph]]]:
    """For one row-group of 2+ stave_idx fragments of the same physical row
    (see _group_staves_by_row), pool every member's real syl_boxes (deduped
    by identity — _assign_boxes_to_staves already put each box in exactly
    one member's bucket) and every member's own neume glyphs, then run
    _assign_glyphs_to_boxes ONCE over the pooled sets so a fragment's
    glyphs can be matched against its row-mate's real syl_boxes instead of
    falling to the "-" no-text-alignment fallback. _assign_glyphs_to_boxes
    re-sorts glyphs by ulx internally, so pooling members' glyphs in any
    order and letting it re-sort keeps reading order correct for free.

    Returns (pooled_boxes, glyph_groups), x-sorted and index-aligned — same
    shape _assign_glyphs_to_boxes always returns."""
    pooled_boxes: list[dict] = []
    seen_ids: set[int] = set()
    for idx in sorted(group):
        for b in boxes_by_stave.get(idx, []):
            if id(b) not in seen_ids:
                seen_ids.add(id(b))
                pooled_boxes.append(b)
    pooled_boxes.sort(key=lambda b: b["ul"][0])
    pooled_glyphs = [g for idx in group for g in neume_glyphs_by_stave.get(idx, [])]
    return pooled_boxes, _assign_glyphs_to_boxes(pooled_glyphs, pooled_boxes)

def _stave_share_of_group(
        stave_idx: int,
        pooled_boxes: list[dict],
        pooled_glyph_groups: list[list[Glyph]],
        boxes_by_stave: dict[int, list[dict]],
        neume_glyphs_by_stave: dict[int, list[Glyph]],
) -> tuple[list[dict], list[list[Glyph]]]:
    """Cut a row-group's pooled (boxes, glyph_groups) down to just
    stave_idx's own share, so its glyphs still surface under ITS OWN
    <sb>/<clef> position in build_mei's layer — only the TEXT reachable
    across the fragment boundary changes, not where in the document each
    glyph's <syllable> lands.

    A pooled box with none of stave_idx's own glyphs is dropped, UNLESS the
    box is genuinely glyph-less across the WHOLE pooled group (a syllable
    with no note under it — legitimate, see _assign_glyphs_to_boxes's own
    docstring) AND stave_idx was that box's original (pre-grouping)
    boxes_by_stave owner. That second condition matters: _assign_boxes_to_
    staves places a box by pure Y-distance, so a box can be "native" to
    stave_idx while its glyphs (assigned by X-distance, within the group)
    actually belong to a *different* member. Re-including it for stave_idx
    whenever THAT stave merely has none of its glyphs — instead of only
    when NO member does — would emit the same real syllable twice: once
    correctly on the member that owns its glyphs, once as a spurious empty
    duplicate on the native stave. Gating on "glyph-less everywhere" keeps
    exactly one emission per real syl_box, total, exactly as happens today
    for an un-grouped stave."""
    own_ids = {g.id for g in neume_glyphs_by_stave.get(stave_idx, [])}
    native_box_ids = {id(b) for b in boxes_by_stave.get(stave_idx, [])}

    out_boxes: list[dict] = []
    out_glyph_groups: list[list[Glyph]] = []
    for box, glyphs_for_box in zip(pooled_boxes, pooled_glyph_groups, strict=True):
        own_share = [g for g in glyphs_for_box if g.id in own_ids]
        if own_share or (not glyphs_for_box and id(box) in native_box_ids):
            out_boxes.append(box)
            out_glyph_groups.append(own_share)
    return out_boxes, out_glyph_groups

def _text_box_valid(box: dict, image_w: int, image_h: int) -> bool:
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

def _filter_neume_glyphs(
        staff_glyphs: list[Glyph], stave_idx: int,
        special_mapping: dict[str, SpecialEntry], 
) -> list[Glyph]:
    """Drop staff-line glyphs and any glyph whose classification is one of
    the notation type's special (non-<neume>) mapping entries — clef,
    custos, divLine, accid — from one stave's glyph list before
    syllable-building. Extracted from build_mei's per-stave loop so the
    syllable-row-grouping prepass (see _group_staves_by_row) and the main
    loop share one skip-glyph definition instead of two copies drifting
    apart.

    Checking membership in special_mapping (the real classification
    strings, from the same CSV that drives _extract_special_glyphs' own
    encoding below) replaces an earlier substring-fragment check
    (_SKIP_CLASS_FRAGMENTS = {"custos", "divline", "division"}) that
    matched "custos" correctly but never matched "divisio.maxima"/
    "divisio.maior"/"divisio.finalis" (its "divline"/"division" fragments
    both contain an "n" the real classification strings don't), so a
    divisio glyph fell through into ordinary neume/<nc> encoding instead
    of being excluded here — see mothra#257."""
    skip_ids = {
        g.id for g in staff_glyphs
        if (g.nrows > 0 and g.ncols / g.nrows >= 8)
        or g.class_name.lower().strip() in special_mapping
    }
    if skip_ids:
        skipped = [g for g in staff_glyphs if g.id in skip_ids]
        skip_types = ", ".join(sorted({g.class_name for g in skipped}))
        print(f" [stave {stave_idx}] skipping {len(skip_ids)} glyph(s): {skip_types}")
    return [g for g in staff_glyphs if g.id not in skip_ids]

def _extract_special_glyphs(
    staff_glyphs: list[Glyph], 
    special_mapping: dict[str, SpecialEntry],
) -> list[tuple[Glyph, SpecialEntry]]:
    """The subset of one stave's raw glyphs that are clef/custos/divLine
    classifications (per special_mapping), paired with their parsed
    SpecialEntry, sorted left-to-right by ulx — the order build_mei's
    per-stave loop interleaves them into the <layer> alongside <syllable>s.
    A sibling to _filter_neume_glyphs's exclusion of the same glyphs from
    ordinary neume/<nc> encoding: that function says what to drop, this one
    says what to do with what was dropped.

    accid entries are deliberately excluded here (though still excluded
    from ordinary encoding by _filter_neume_glyphs, since they're also in
    special_mapping): an accidental correctly belongs nested inside the
    <nc> of the note it modifies, not floating as its own <layer> sibling
    like a clef/custos/divLine change does, and that nesting isn't
    implemented yet. accid classifications also aren't currently produced
    by either trained classifier, so this has no observable effect today —
    it only avoids fabricating an incorrect standalone element if/when that
    changes."""
    pairs: list[tuple[Glyph, SpecialEntry]] = []
    for g in staff_glyphs:
        entry = special_mapping.get(g.class_name.lower().strip())
        if entry is not None and entry.tag != "accid":
            pairs.append((g, entry))
    return sorted(pairs, key=lambda p: p[0].ulx)

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
        glyphs: list[Glyph], page_w: int, page_h: int, id_prefix: str = "auto",
        allow_synthetic_lines: bool = False,
        typical_line_spacing: Optional[float] = None,
) -> list[tuple[StaveBbox, list[Glyph]]]:
    """Cluster glyphs into stave-sized row groups by Y-center gap, pairing each
    synthesized StaveBbox with the exact glyphs that produced it. Shared by
    estimate_staves_from_glyphs's no-stave-data fallback and by
    assign_glyphs_to_staves's missed-stave recovery (see there).

    The resulting StaveBbox's bounding box is always real (derived from the
    glyph cluster itself), but its internal `line_ys` -- used only for pitch
    computation, via _nc_pitch -- has no real measured staff-line data behind
    it in this fallback. allow_synthetic_lines controls whether to fabricate
    plausible-looking line_ys anyway (the old, always-on behavior) or leave
    line_ys empty so _nc_pitch's own <2-entry guard degrades to an explicit
    unresolved default ("a", "3") instead of a confident-looking but invented
    pitch. Default False -- see estimate_staves_from_glyphs's own docstring
    for why.

    When allow_synthetic_lines is True, typical_line_spacing (assign_glyphs_
    to_staves's caller passes the page's own already-detected staves' real
    spacing — see _typical_line_spacing) further anchors each row's
    estimated line_ys on that reliable figure instead of deriving spacing
    from the row's own glyphs' bounding box, which a single unusually
    high/low note can badly skew. Left None (estimate_staves_from_glyphs's
    whole-page-fallback caller, where no real line data exists anywhere to
    borrow from) preserves the old bbox-derived guess exactly, since there's
    nothing better to use."""
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
        if not allow_synthetic_lines:
            line_ys = []
        elif typical_line_spacing:
            # Anchor on the row's own median note height (robust to a
            # single outlier note) using the page's REAL spacing, rather
            # than quartering this row's own bbox span — see docstring.
            # clef_line's default (3rd of 4 lines, index len(line_ys)-3=1)
            # sits just above center; a STAFF_LINES-point staff centered on
            # the middle offset keeps that same convention while using
            # reliable spacing, and stays correct if STAFF_LINES changes.
            center_y = median(g.cy for g in row)
            offset = (STAFF_LINES - 1) / 2
            line_ys = [center_y + typical_line_spacing * (j - offset) for j in range(STAFF_LINES)]
        else:
            line_ys = [est_uly + h * j / (STAFF_LINES - 1) for j in range(STAFF_LINES)]
        stave = StaveBbox(
            id=f"{id_prefix}-{i}",
            ulx=max(0, min(g.ulx for g in row) - pad),
            uly=max(0, min(g.uly for g in row) - pad),
            lrx=min(page_w, max(g.lrx for g in row) + pad),
            lry=min(page_h, max(g.lry for g in row) + pad),
            line_ys=line_ys,
        )
        groups.append((stave, row))
    return groups


def estimate_staves_from_glyphs(
    glyphs: list[Glyph], page_w: int, page_h: int, allow_synthetic_lines: bool = False,
) -> tuple[list[StaveBbox], str]:
    """Estimate stave bounding boxes from GameraXML glyphs. This is tier 3 of
    tasks_encode.py's 3-tier stave-source fallback (staffline_detection ->
    yolo_annotation -> this) -- the caller uses the returned tag as that
    tier's `stave_source` value.

    Primary strategy: cluster the detected staff lines (wide, flat glyphs with
    aspect ratio ≥ 8 spanning >20% of page width).  These are highly reliable
    stave anchors and are unaffected by text characters that fill inter-stave
    gaps and cause the neume-Y-clustering approach to collapse all staves into
    one. Tagged "glyph_estimate" -- real, page-specific geometry.

    Fallback: if too few staff lines are available, cluster neume-like glyphs
    by Y-center gap (original approach with tighter outlier filtering). The
    stave bounding box here is still real (from the glyph cluster), but its
    internal line positions are not measured -- see _cluster_glyphs_into_staves's
    own docstring for allow_synthetic_lines. Tagged "glyph_estimate_synthetic_lines"
    when allow_synthetic_lines fabricated plausible-looking line_ys, or
    "glyph_estimate_unresolved_lines" (default) when it didn't -- either way,
    distinct from "glyph_estimate" so this weaker case is never conflated with
    strategy 1's real per-page staff-line geometry.

    Zero-glyph case: no real geometry exists at all. Returns one hardcoded
    full-page StaveBbox tagged "placeholder_no_glyphs" -- callers should treat
    this as a warning-worthy result, not silent success.
    """
    if not glyphs:
        return [StaveBbox(id="synth-0", ulx=0, uly=0, lrx=page_w, lry=page_h)], "placeholder_no_glyphs"

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
            return staves, "glyph_estimate"

    # ── Strategy 2: neume Y-gap clustering (fallback) ──────────────────────
    staves = [
        stave for stave, _ in _cluster_glyphs_into_staves(
            glyphs, page_w, page_h, allow_synthetic_lines=allow_synthetic_lines,
        )
    ]
    detail = "glyph_estimate_synthetic_lines" if allow_synthetic_lines else "glyph_estimate_unresolved_lines"
    return staves, detail


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

_PITCH_NOTES = ["c", "d", "e", "f", "g", "a", "b"]

def _pitch_from_step(step: int, clef_note: str = "c", clef_oct: int = 4) -> tuple[str, str]:
    """Diatonic step offset from clef note → (pname, oct).
    Positive step = below clef (lower pitch); negative = above (higher pitch).
    """
    clef_abs = clef_oct * 7 + _PITCH_NOTES.index(clef_note)
    note_abs = clef_abs - step
    return _PITCH_NOTES[note_abs % 7], str(note_abs // 7)

def _line_spacing(line_ys: list[float]) -> Optional[float]:
    """Median gap between consecutive (sorted ascending) staff line
    positions, or None when there are fewer than 2 points to compare, or
    the points don't actually increase. Shared by _step_from_y (pitch
    assignment) and _stave_zone_bounds (the stave's rendered facsimile
    zone), so both always agree on what "one line-to-line spacing" means
    for a given stave — see _stave_zone_bounds's docstring for why that
    agreement matters."""
    if len(line_ys) < 2:
        return None
    spacings = [line_ys[i + 1] - line_ys[i] for i in range(len(line_ys) - 1)]
    spacing = median(spacings)
    return spacing if spacing > 0 else None

def _step_from_y(y: float, line_ys: list[float], clef_line: int = 3) -> Optional[int]:
    """Diatonic step (relative to the clef line) for a point at height y,
    given sorted ascending staff line Y positions. Returns None when there
    isn't enough line data to place it (len(line_ys) < 2) — callers fall
    back to a fixed default pitch in that case, same as this file's old
    per-component behaviour before @intm-chaining replaced the y_fraction
    heuristic (see neume_mapping.py and mothra#137).

    clef_y is anchored from the BOTTOM detected line and extrapolated by
    the (median) line spacing — clef_line follows MEI's @line convention
    of counting lines from the bottom of the staff — rather than indexing
    line_ys[len(line_ys) - clef_line] directly. That direct-index form
    silently assumed len(line_ys) always equals the real total line count
    for this stave; on a real page, per-stave detected line counts varied
    (2 to 7 for a nominal 4-line stave, from a mix of under-detection and
    duplicate left/right line-fragment sampling — see staffline_adapter's
    _dedupe_line_ys for the latter), so that assumption broke — even
    wrapping to the wrong end of the list via a negative index when a
    stave had fewer real lines than clef_line. Extrapolating from the
    bottom with the measured spacing degrades far more gracefully when a
    stave's real line count doesn't match the nominal expectation.
    Likewise, line_spacing uses the MEDIAN gap rather than the mean, so a
    handful of remaining anomalous gaps can't drag the spacing estimate
    (and therefore every note's diatonic step) off by a large margin."""
    line_spacing = _line_spacing(line_ys)
    if line_spacing is None:
        return None
    clef_y = line_ys[-1] - line_spacing * (clef_line - 1)
    return round((y - clef_y) / (line_spacing / 2))

def _stave_zone_bounds(stave: StaveBbox) -> tuple[int, int]:
    """(uly, lry) to use for this stave's <sb>-referenced facsimile zone.

    Verovio's own facsimile-transcription rendering (EditorToolkitNeume::
    Resize / SyncFromFacsimileFunctor's VisitPageEnd — confirmed against
    Verovio's own source; it isn't vendored in this repo) derives its
    per-note PIXEL spacing from this zone's height alone:
    zone_height / (STAFF_LINES - 1). That is a completely separate
    computation from _step_from_y's pitch assignment, which uses the real
    MEDIAN detected line-to-line gap — the two must agree, or a note's
    assigned diatonic step (correct in the abstract) renders at the wrong
    pixel distance from the clef. Confirmed on a real page: the union-of-
    detected-line-bounding-boxes zone heights `staves_from_jsomr()` (and
    the padded fallbacks elsewhere in this file) produce implied a
    per-line spacing anywhere from 0.7x to 2.2x the real spacing,
    visibly displacing every note on the affected staves — most visibly
    once a stave's zone was manually realigned with the real page in
    Neon (which only ever changes the zone's own geometry, never any
    note's pname/oct — see EditorToolkitNeume::Resize), suddenly exposing
    the mismatch that had been partly masked by the zone's prior,
    also-off position.

    When real line data is available (>=2 points), this returns a zone
    spanning exactly (STAFF_LINES - 1) * real_spacing, anchored on the
    bottommost detected line — matching _step_from_y's own bottom-
    anchored convention exactly, so Verovio's per-step pixel unit and
    _step_from_y's per-step diatonic assignment always agree. Falls back
    to the stave's own (possibly padded/union) uly/lry when there's no
    usable line data to anchor on — same case _step_from_y itself falls
    back to a fixed default pitch for, since there's no real spacing
    convention to reconcile with either way."""
    spacing = _line_spacing(stave.line_ys)
    if spacing is None:
        return stave.uly, stave.lry
    lry = stave.line_ys[-1]
    uly = lry - spacing * (STAFF_LINES - 1)
    return round(uly), round(lry)

def _component_pitches(
    anchor_step: Optional[int],
    components: list[NcTemplate],
    clef_note: str = "c",
    clef_oct: int = 4,
) -> list[tuple[str, str]]:
    """(pname, oct) for each component of one neume glyph.

    The FIRST component's pitch comes from anchor_step (the glyph's own
    position on the stave — see _step_from_y, called with the glyph's bbox
    center, matching the old default y_fraction=0.5 anchor). Every later
    component's pitch is the previous one's step minus its own @intm delta:
    @intm is positive when that note sits a step ABOVE the previous one
    (higher pitch), while _pitch_from_step's own convention is the
    opposite — positive step = LOWER pitch (see its docstring) — hence the
    subtraction. Verified against the CSVs' own ascending/descending
    neumes: podatus's second <nc intm="1S"/> must end up higher than its
    first; clivis's second <nc intm="-1S"/> must end up lower.

    Falls back to a fixed ("a", "3") for every component (matching this
    file's pre-existing fallback) when anchor_step is None.
    """
    if anchor_step is None:
        return [("a", "3")] * len(components)
    pitches = []
    step = anchor_step
    for j, comp in enumerate(components):
        if j > 0:
            step -= comp.intm
        pitches.append(_pitch_from_step(step, clef_note, clef_oct))
    return pitches

def _resolve_pitches(
    glyph: Glyph,
    components: list[NcTemplate],
    pitch_map: Optional[dict[str, list[tuple[str, str]]]],
    line_ys: list[float],
    clef_line: int,
    clef_note: str,
    clef_oct: int,
) -> list[tuple[str, str]]:
    """(pname, oct) per component for one neume/custos glyph.

    Three sources, strongest first:

    1. `pitch_map[glyph.id]` with one pitch per component -- pitch_stage.py's
       real pitch finding (algorithm #1 of the pitch-finding/ submodule):
       per-class ink-centroid anchor, interval decomposition, and the page's own
       detected clef. This is the normal path for a classified neume on a stave
       with line data.
    2. the same measurement for note 1 only, chained by @intm for the rest --
       when that stage resolved the glyph but to a different note count than
       this file's CSV `<nc>` list (see _chain_from_pitch).
    3. the geometric placeholder this file has always used: the glyph bbox's
       vertical center read as a diatonic step against the ASSUMED clef line
       (_step_from_y), then @intm chaining (_component_pitches). Reached
       per glyph, not per page -- a glyph the algorithm reported as
       missing_staff/missing_clef/no_line_coverage, an unclassified bbox, a
       deployment with no pitch-finding checkout, or MOTHRA_PITCH_FINDING=0.

    Only pitch differs between the three; the `<nc>` elements themselves, their
    attributes and their zones are the same either way.
    """
    mapped = pitch_map.get(glyph.id) if pitch_map else None
    if mapped:
        if len(mapped) == len(components):
            return list(mapped)
        return _chain_from_pitch(mapped[0], components)
    anchor_step = _step_from_y(glyph.cy, line_ys, clef_line)
    return _component_pitches(anchor_step, components, clef_note, clef_oct)

def _chain_from_pitch(
    first: tuple[str, str],
    components: list[NcTemplate],
) -> list[tuple[str, str]]:
    """(pname, oct) per component, chaining off an ALREADY-ABSOLUTE first pitch.

    The bridge for the case where pitch_stage.py resolved a glyph but with a
    different note count than this file's own `<nc>` list -- a repeated-note
    neume (neume.distropha and friends resolve to one pitch there, since every
    notehead of one carries the same pitch) or a class its interval table only
    knows as a single approximate note. Rather than throw the measured anchor
    away and fall back to _step_from_y for the whole glyph, keep it for note 1
    and place the rest with the same @intm chaining _component_pitches uses.

    @intm is positive for a note ABOVE the previous one, so the absolute
    diatonic index goes UP by intm -- the mirror of _component_pitches's
    subtraction, which works in _pitch_from_step's inverted step space (see
    both docstrings).
    """
    pname, oct_str = first
    try:
        abs_idx = int(oct_str) * 7 + _PITCH_NOTES.index(pname.lower())
    except (ValueError, TypeError):
        return [first] * len(components)
    pitches = [first]
    for comp in components[1:]:
        abs_idx += comp.intm
        pitches.append((_PITCH_NOTES[abs_idx % 7], str(abs_idx // 7)))
    return pitches

def _component_zone_ids(
    surface: ET.Element,
    glyph: Glyph,
    components: list[NcTemplate],
    width_raw: Optional[str],
) -> Optional[list[str]]:
    """For a multi-component neume (clivis, podatus, torculus, ...), split
    the glyph's own zone horizontally into one side-by-side sub-zone per
    component, proportional to the mapping CSV's width column (e.g.
    "[1, 1]" -> two equal-width halves), registering each as a new <zone>
    under `surface`. Returns None (caller falls back to the single shared
    "z-{glyph.id}" zone, today's pre-existing behaviour) whenever there's
    nothing to split: a single-component neume, no width column for this
    classification, or a width list that doesn't parse or whose length
    doesn't match the component count (e.g. square.csv's
    neume.scandicus22a/22b: a "[1, 2]"/2-weight width against 3 <nc>
    components -- a real, pre-existing inconsistency in the bundled CSV
    data, not something to silently paper over here).

    Only the horizontal (x) extent is split -- the full glyph height is
    kept for every sub-zone. Pitch is unaffected either way: pname/oct
    already come from @intm-chaining in _component_pitches, entirely
    independent of zone geometry -- these sub-zones only change where a
    component's bounding box points for facsimile display/click-to-
    correct in Neon."""
    if len(components) <= 1:
        return None
    weights = parse_width(width_raw) if width_raw else None
    if weights is None or len(weights) != len(components):
        return None
    total = sum(weights)
    if total <= 0:
        return None
    span = glyph.lrx - glyph.ulx
    # Precompute every rounded boundary before creating any <zone> at all.
    # A narrow glyph (span smaller than the component count) can round two
    # adjacent boundaries to the same x, which would register a zero-width
    # zone that <nc>'s @facs then points at -- worse than the shared-zone
    # fallback this function exists to avoid. Bail out to that fallback
    # instead of emitting invalid facsimile geometry.
    boundaries = [glyph.ulx]
    cumulative = 0.0
    for w in weights:
        cumulative += w
        boundaries.append(glyph.ulx + round(span * cumulative / total))
    for prev, nxt in zip(boundaries, boundaries[1:]):
        if nxt <= prev:
            return None
    zone_ids = []
    for j, (x, x_next) in enumerate(zip(boundaries, boundaries[1:])):
        zone_id = f"z-{glyph.id}-{j}"
        ET.SubElement(surface, _tag("zone"), {
            XML_ID: zone_id,
            "ulx": str(round(x)),
            "uly": str(glyph.uly),
            "lrx": str(round(x_next)),
            "lry": str(glyph.lry),
        })
        zone_ids.append(zone_id)
    return zone_ids
    
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
    notation_type: str = "square",
    n_detected_staves: Optional[int] = None,
    pitch_map: Optional[dict[str, list[tuple[str, str]]]] = None,
    clef_line_map: Optional[dict[str, int]] = None,
) -> bytes:
    ET.register_namespace("", MEI_NS)
    mei = ET.Element(_tag("mei"), {"meiversion": "5.0.0-dev"})
    neume_mapping = resolve_neume_mapping(notation_type)
    special_mapping = resolve_special_mapping(notation_type)
    missing_classes: set[str] = set()
    mismatched_widths: set[str] = set()

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
        zone_uly, zone_lry = _stave_zone_bounds(stave)
        zone_id = f"sz-{stave.id}"
        ET.SubElement(surface, _tag("zone"), {
            XML_ID: zone_id,
            "type": "staff",
            "ulx": str(stave.ulx),
            "uly": str(zone_uly),
            "lrx": str(stave.lrx),
            "lry": str(zone_lry),
        })
        stave_zone_ids[i] = zone_id
        
        ET.SubElement(surface, _tag("zone"), {
            XML_ID: f"sz-raw-{stave.id}",
            "type": "staff-raw",
            "ulx": str(stave.ulx),
            "uly": str(stave.uly),
            "lrx": str(stave.lrx),
            "lry": str(stave.lry),
        })
        # Clef zone: left edge of stave, roughly square (height ≈ staff height)
        clef_zone_id = f"cz-{stave.id}"
        stave_h = max(zone_lry - zone_uly, 1)
        ET.SubElement(surface, _tag("zone"), {
            XML_ID: clef_zone_id,
            "ulx": str(stave.ulx),
            "uly": str(zone_uly),
            "lrx": str(stave.ulx + stave_h),
            "lry": str(zone_lry),
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
    boxes_by_stave = _assign_boxes_to_staves(staves, syl_boxes) if syl_boxes else {}

    skip_zones = False
    if syl_boxes:
        invalid = sum(1 for b in syl_boxes if not _text_box_valid(b, image_w, image_h))
        if invalid > len(syl_boxes) / 2:
            skip_zones = True
            print(
                f" [text-alignment] {invalid}/{len(syl_boxes)} syl_boxes fall outside the "
                f"{image_w}x{image_h} page (likely a stale or mismatched-resolution "
                "text-finding result) — syllable text will still be used, but without "
                "bounding boxes",
                file=sys.stderr,
            )

    def _syl_zone(box: dict) -> Optional[str]:
        """Create a <zone> for one syl_box (or a synthesized union box, for
        the multi-box-no-clean-split fallback), unless zones are globally
        disabled for this page (see skip_zones above) or the box itself is
        out of bounds. Every call makes a fresh zone, so every <syl> gets
        its own — never shared across <syllable>s (Neon's resize action
        mutates the one zone all referencing elements share, so two
        syllables pointing at the same zone would silently move together)."""
        if skip_zones or not _text_box_valid(box, image_w, image_h):
            return None
        zone_id = f"zone-syl-{str(uuid.uuid4()).replace('-', '')[:12]}"
        ulx, uly = box["ul"]
        lrx, lry = box["lr"]
        ET.SubElement(surface, _tag("zone"), {
            XML_ID: zone_id,
            "ulx": str(int(ulx)), "uly": str(int(uly)),
            "lrx": str(int(lrx)), "lry": str(int(lry)),
        })
        return zone_id 

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
        "lines": str(STAFF_LINES),
        # Neon (landing-page/neon submodule)'s own font-selection code
        # (ConvertMei.ts's stripHufnagelForVerovio/setNotationTypeInMei)
        # keys off this exact "neume.square"/"neume.hufnagel" suffix on
        # <staffDef> to pick which notation font to render with -- a bare
        # "neume" (Neon's Verovio-compatibility working copy value, not a
        # real subtype) tells it nothing, so every generated MEI rendered
        # identically regardless of notation_type until this matched
        # Neon's own convention (mothra#210).
        "notationtype": (
            f"neume.{notation_type}" if notation_type in ("square", "hufnagel") else "neume"
        ),
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

    # Syllable-matching prepass: one physical manuscript row's glyphs can
    # end up split across multiple stave_idx buckets (assign_glyphs_to_staves's
    # Y-gap glyph clustering can fragment a row — see _group_staves_by_row's
    # docstring). When that happens, the row's real syl_boxes land on
    # whichever stave_idx got most of the row's glyphs via
    # _assign_boxes_to_staves's pure Y-distance test, leaving a fragment's
    # bucket with none — so without this, the fragment falls straight to
    # the "-" no-text-alignment fallback below even though its row-mate has
    # real text. Grouping fragments' boxes+glyphs before assignment lets a
    # fragment share its row-mate's real syl_boxes instead.
    #
    # n_detected_staves gates this to ONLY ever rescue a stave
    # assign_glyphs_to_staves itself synthesized (see _group_staves_by_row's
    # docstring for why two originally-detected staves must never merge).
    # Callers that don't know this (n_detected_staves=None) get the safest
    # possible default: len(staves), i.e. "trust every stave as originally
    # detected" — no merging happens at all, same as if this whole fix
    # didn't exist for that call.
    effective_n_detected = n_detected_staves if n_detected_staves is not None else len(staves)
    row_groups = _group_staves_by_row(staves, effective_n_detected)
    stave_to_group: dict[int, set[int]] = {i: g for g in row_groups for i in g}
    neume_glyphs_by_stave: dict[int, list[Glyph]] = {
        stave_idx: _filter_neume_glyphs(glyphs_by_stave[stave_idx], stave_idx, special_mapping)
        for stave_idx in sorted(k for k in glyphs_by_stave if k >= 0)
    }
    special_glyphs_by_stave: dict[int, list[tuple[Glyph, SpecialEntry]]] = {
        stave_idx: _extract_special_glyphs(glyphs_by_stave[stave_idx], special_mapping)
        for stave_idx in sorted(k for k in glyphs_by_stave if k >= 0)
    }
    group_pool: dict[frozenset, tuple[list[dict], list[list[Glyph]]]] = {
        frozenset(g): _pool_group_syllable_data(g, boxes_by_stave, neume_glyphs_by_stave)
        for g in row_groups if len(g) > 1
    }

    for stave_idx in sorted(
            (k for k in glyphs_by_stave if k >= 0),
            key=lambda k: (staves[k].uly, k),
    ):
        staff_glyphs = glyphs_by_stave[stave_idx]
        if not staff_glyphs and not boxes_by_stave.get(stave_idx):
            continue
        stave = staves[stave_idx] if stave_idx < len(staves) else None
        line_ys = stave.line_ys if stave else []
        # <sb> marks the start of each stave and links it to its zone
        zone_id = stave_zone_ids.get(stave_idx, str(stave_idx))
        sb_attrs: dict[str, str] = {XML_ID: f"sb-{zone_id}"}
        if stave_idx in stave_zone_ids:
            sb_attrs["facs"] = f"#{zone_id}"
        ET.SubElement(layer, _tag("sb"), sb_attrs)

        stave_clef_shape = clef_shape
        opening_clef_glyph_id: Optional[str] = None
        consumed_special_ids: set[str] = set()
        for g, entry in special_glyphs_by_stave.get(stave_idx, []):
            if entry.tag == "clef":
                stave_clef_shape = entry.attrs.get("shape", clef_shape)
                opening_clef_glyph_id = g.id
                consumed_special_ids.add(g.id)
                break
        clef_note, clef_oct = _CLEF_PITCH_REF.get(stave_clef_shape, ("c", 4))
        # Which staff line this stave's clef is DECLARED on. pitch_stage.py
        # measures it from the clef glyph's own ink centroid when it can (see
        # its _clef_line_from_step); the clef_line argument is the assumed
        # default for every stave it couldn't. This matters beyond the emitted
        # attribute: Verovio positions an <nc> from pname/oct against the
        # declared clef line, and _step_from_y's own placeholder pitch is
        # measured relative to it too, so both have to read the same value or
        # the stave's notes render off by whole lines.
        effective_clef_line = clef_line
        if opening_clef_glyph_id and clef_line_map:
            effective_clef_line = clef_line_map.get(opening_clef_glyph_id, clef_line)

        # Clef must follow each <sb>; @facs anchors it to the stave's left edge
        clef_id = str(uuid.uuid4()).replace("-", "")[:12]
        clef_attrs: dict[str, str] = {
            XML_ID: f"clef-{clef_id}",
            "shape": stave_clef_shape,
            "line": str(effective_clef_line),
        }
        if stave_idx in clef_zone_ids:
            clef_attrs["facs"] = f"#{clef_zone_ids[stave_idx]}"
        ET.SubElement(layer, _tag("clef"), clef_attrs)
        
        neume_glyphs = neume_glyphs_by_stave[stave_idx]
        group = stave_to_group.get(stave_idx, {stave_idx})
        if len(group) > 1:
            # This stave_idx is a fragment (or the main bucket) of a row
            # split across multiple stave_idx values — pull its share of
            # the row-group's pooled boxes/glyphs instead of just its own
            # (likely-empty) boxes_by_stave entry. See the prepass above.
            pooled_boxes, pooled_glyph_groups = group_pool[frozenset(group)]
            stave_boxes, glyph_groups = _stave_share_of_group(
                stave_idx, pooled_boxes, pooled_glyph_groups,
                boxes_by_stave, neume_glyphs_by_stave,
            )
        else:
            stave_boxes = boxes_by_stave.get(stave_idx, [])
            glyph_groups = _assign_glyphs_to_boxes(neume_glyphs, stave_boxes) if stave_boxes else []

        syllable_units = _build_syllable_units(neume_glyphs, stave_boxes, glyph_groups, syllable_gap_mult)

        # Interleave this stave's syllables with its non-opening special
        # (clef-change/custos/divLine) glyphs by X position, so a divLine
        # between two phrases (or a custos at the line's end) lands in its
        # real reading-order position in the <layer> instead of always
        # before or always after every syllable (mothra#257).
        layer_items: list[tuple[float, str, tuple]] = [
            (
                box["ul"][0] if box is not None else min((g.ulx for g in glyphs_in_syl), default=0.0),
                "syllable",
                (syl_text, box, glyphs_in_syl),
            )
            for syl_text, box, glyphs_in_syl in syllable_units
        ]
        layer_items += [
            (g.ulx, "special", (g, entry))
            for g, entry in special_glyphs_by_stave.get(stave_idx, [])
            if g.id not in consumed_special_ids
        ]
        layer_items.sort(key=lambda item: item[0])

        for _, kind, payload in layer_items:
            if kind == "syllable":
                syl_text, box, glyphs_in_syl = payload
                syllable_id = glyphs_in_syl[0].id if glyphs_in_syl else str(uuid.uuid4()).replace("-", "")[:12]
                syllable = ET.SubElement(layer, _tag("syllable"), {
                                XML_ID: f"syllable-{syllable_id}",
                            })
                syl_id = str(uuid.uuid4()).replace("-", "")[:12]
                syl_attrs: dict[str, str] = {XML_ID: f"syl-{syl_id}"}
                if box is not None:
                    zone_id = _syl_zone(box)
                    if zone_id is not None:
                        syl_attrs["facs"] = f"#{zone_id}"
                syl = ET.SubElement(syllable, _tag("syl"), syl_attrs)
                syl.text = syl_text
                for glyph in glyphs_in_syl:
                    neume = ET.SubElement(syllable, _tag("neume"), {
                        XML_ID: f"neume-{glyph.id}",
                        "facs": f"#z-{glyph.id}",
                    })
                    entry = neume_mapping.get(glyph.class_name.lower().strip())
                    if entry is not None:
                        components = entry.components
                    else:
                        components = [NcTemplate()]
                        missing_classes.add(glyph.class_name)
                    pitches = _resolve_pitches(
                        glyph, components, pitch_map, line_ys,
                        effective_clef_line, clef_note, clef_oct,
                    )
                    component_zone_ids = _component_zone_ids(
                        surface, glyph, components,
                        entry.width if entry is not None else None,
                    )
                    if len(components) > 1 and component_zone_ids is None:
                        mismatched_widths.add(glyph.class_name)
                    for j, (comp, (pname, oct_str)) in enumerate(zip(components, pitches, strict=True)):
                        nc_id = glyph.id if j == 0 else f"{glyph.id}-{j}"
                        zone_id = component_zone_ids[j] if component_zone_ids is not None else f"z-{glyph.id}"
                        nc_attrs: dict[str, str] = {
                            XML_ID: f"nc-{nc_id}",
                            "facs": f"#{zone_id}",
                            "pname": pname,
                            "oct": oct_str,
                        }
                        nc_attrs.update(comp.attrs)
                        nc_el = ET.SubElement(neume, _tag("nc"), nc_attrs)
                        if comp.liquescent:
                            ET.SubElement(nc_el, _tag("liquescent"))
            else:
                glyph, entry = payload
                if entry.tag == "clef":
                    attrs: dict[str, str] = {XML_ID: f"clef-{glyph.id}", "facs": f"#z-{glyph.id}"}
                    attrs.update(entry.attrs)
                    # A mid-stave clef change (this isn't the stave's opening
                    # clef -- that one already set stave_clef_shape/clef_note/
                    # clef_oct before this loop started) must also rebind the
                    # pitch reference: every <nc>/<custos> emitted after this
                    # point in reading order is read against THIS clef, not
                    # the one the stave opened with. That now includes the
                    # LINE it sits on: this used to stay on the stave's
                    # default because nothing measured a changed clef's
                    # position, but pitch_stage.py measures every clef glyph
                    # the same way, so when it has a value for this one, both
                    # the emitted @line and the placeholder's own reference
                    # move with it. Absent that, it falls back to the stave
                    # default exactly as before.
                    measured_line = (clef_line_map or {}).get(glyph.id)
                    if measured_line is not None:
                        effective_clef_line = measured_line
                    attrs.setdefault("line", str(effective_clef_line))
                    ET.SubElement(layer, _tag("clef"), attrs)
                    clef_note, clef_oct = _CLEF_PITCH_REF.get(
                        attrs.get("shape", stave_clef_shape), (clef_note, clef_oct)
                    )
                elif entry.tag == "custos":
                    pname, oct_str = _resolve_pitches(
                        glyph, [NcTemplate()], pitch_map, line_ys,
                        effective_clef_line, clef_note, clef_oct,
                    )[0]
                    attrs = {
                        XML_ID: f"custos-{glyph.id}",
                        "facs": f"#z-{glyph.id}",
                        "pname": pname,
                        "oct": oct_str,
                    }
                    attrs.update(entry.attrs)
                    ET.SubElement(layer, _tag("custos"), attrs)
                elif entry.tag == "divLine":
                    attrs = {XML_ID: f"divline-{glyph.id}", "facs": f"#z-{glyph.id}"}
                    attrs.update(entry.attrs)
                    ET.SubElement(layer, _tag("divLine"), attrs)
                

    if missing_classes:
        print(
            f" [neume-mapping:{notation_type}] {len(missing_classes)} classification(s) not found in "
            f"the mapping — encoded as a single plain <nc>: {', '.join(sorted(missing_classes))}",
            file=sys.stderr,
        )
    if mismatched_widths:
        print(
            f" [neume-mapping:{notation_type}] {len(mismatched_widths)} classification(s) have a width column "
            f"that doesn't match their component count — encoded with a shared (unsplit) zone: "
            f"{', '.join(sorted(mismatched_widths))}",
            file=sys.stderr,
        )

    return _serialize_mei(mei)

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
    return _serialize_mei(root)

def scale_text_alignment(text_alignment: Optional[dict], factor_x: float, factor_y: Optional[float] = None) -> Optional[dict]:
    """Scale a text_alignment's syl_boxes ul/lr coords by (factor_x, factor_y)
    -- counterpart to scale_facsimile, for when syl_boxes (always computed
    against the working-copy image) must be compared/written against
    geometry in a different pixel space.

    factor_x/factor_y are independent, NOT a single uniform scale: the
    client-side upload resize (imageResize.ts) rounds width and height
    separately after applying one scalar shrink factor, so
    image_w/working_w and image_h/working_h can come out numerically
    different (e.g. an odd source dimension rounds differently on each
    axis) even though the resize was visually uniform. Using one factor for
    both axes would skew Y coordinates by that rounding error. factor_y
    defaults to factor_x for callers that only have (or only need) one
    axis's ratio."""
    if factor_y is None:
        factor_y = factor_x
    if not text_alignment or (factor_x == 1.0 and factor_y == 1.0):
        return text_alignment

    def _scaled(box):
        try:
            ul = [box["ul"][0] * factor_x, box["ul"][1] * factor_y]
            lr = [box["lr"][0] * factor_x, box["lr"][1] * factor_y]
        except (KeyError, TypeError, IndexError):
            return box  # malformed box passes through unscaled, not dropped
        return {**box, "ul": ul, "lr": lr}

    scaled = dict(text_alignment)
    scaled["syl_boxes"] = [_scaled(b) for b in text_alignment.get("syl_boxes", [])]
    return scaled


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

    # stave section's very first <clef> (the one immediately following
    # <sb>), which has no detected-glyph facs to check when no clef glyph
    # was found on that stave -- see build_mei's opening-clef handling.
    for layer in layers:
        seen_opening_clef = False
        for child in layer:
            if child.tag == _tag("sb"):
                seen_opening_clef = False
            elif child.tag == _tag("clef"):
                if not seen_opening_clef:
                    seen_opening_clef = True
                    continue
                cid = child.get(XML_ID, "")
                if not cid:
                    warnings.append("clef missing xml:id")
                check_facs(child, f"clef {cid or '?'}")
            elif child.tag == _tag("custos"):
                cid = child.get(XML_ID, "")
                if not cid:
                    warnings.append("custos missing xml:id")
                check_facs(child, f"custos {cid or '?'}")
                for attr in ("pname", "oct"):
                    if not child.get(attr):
                        warnings.append(f"custos {cid or '?'}: missing @{attr}")
            elif child.tag == _tag("divLine"):
                did = child.get(XML_ID, "")
                if not did:
                    warnings.append("divLine missing xml:id")
                check_facs(child, f"divLine {did or '?'}")

    return warnings


def trace_stave_zone_parity(staves: list[StaveBbox], mei_bytes: bytes) -> list[str]:
    """Parse a just-built MEI's actual <zone type="staff"> elements back out
    and confirm they match the StaveBbox list build_mei() was handed for it
    -- same count, same coordinates. build_mei() writes each zone's
    ulx/uly/lrx/lry straight from the corresponding StaveBbox (zone id
    convention: `sz-{stave.id}`, see build_mei() itself), so any mismatch
    here would mean something between stave resolution and encoding silently
    altered the geometry Neon ends up displaying -- not something that
    should ever legitimately happen, hence a "[warn]"-prefixed message
    rather than a quiet one when it does.

    Returns a list of "[trace] ..." message strings for the caller to
    publish however it likes (a job-log event, a print, a test assertion --
    kept DB/Celery-independent on purpose so it's testable like the rest of
    this module, unlike tasks_encode.py which connects to Postgres at
    import time)."""
    try:
        root = ET.fromstring(mei_bytes)
        zones = {
            z.get(XML_ID): z
            for z in root.iter(_tag("zone"))
            if z.get("type") == "staff"
        }
    except ET.ParseError as e:
        return [f"[trace] [warn] could not parse MEI back out to verify zones: {e}"]

    if len(zones) != len(staves):
        return [
            f"[trace] [warn] stave/zone count mismatch: {len(staves)} StaveBbox(es) handed to"
            f" build_mei(), {len(zones)} type=\"staff\" zone(s) actually written"
        ]

    mismatches = []
    for stave in staves:
        zone = zones.get(f"sz-{stave.id}")
        if zone is None:
            mismatches.append(f"stave {stave.id}: no matching zone written")
            continue
        # float(), not int(): build_mei() writes str(stave.ulx) etc. verbatim,
        # and StaveBbox's own type hints (int) aren't enforced at runtime --
        # a fractional coordinate reaching here would otherwise raise
        # ValueError and abort the whole encode job over what's meant to be a
        # tracing-only check.
        try:
            written = tuple(float(zone.get(k)) for k in ("ulx", "uly", "lrx", "lry"))
        except (TypeError, ValueError):
            mismatches.append(f"stave {stave.id}: zone has unparsable coordinates")
            continue
        # Compare against _stave_zone_bounds(stave)'s (uly, lry), not the
        # raw stave.uly/stave.lry -- build_mei() deliberately writes the
        # zone's vertical bounds from _stave_zone_bounds (Verovio derives
        # its rendered note spacing from the zone's height, which needs to
        # match the stave's REAL measured line spacing, not its raw padded
        # bbox -- see _stave_zone_bounds's own docstring), so a real,
        # correct encode is EXPECTED to diverge here for exactly that one
        # dimension. This check still catches every other kind of silent
        # zone corruption (wrong ulx/lrx, a stale/mismatched stave.id, etc.)
        zone_uly, zone_lry = _stave_zone_bounds(stave)
        expected = tuple(
            float(v) for v in (stave.ulx, zone_uly, stave.lrx, zone_lry)
        )
        if written != expected:
            mismatches.append(f"stave {stave.id}: expected {expected}, zone has {written}")

    if mismatches:
        return [f"[trace] [warn] {len(mismatches)} stave zone(s) diverged from input: " + "; ".join(mismatches)]
    return [f"[trace] {len(staves)} stave zone(s) verified identical to build_mei()'s input"]


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

def _reconstruct_mei_state(
    root: ET.Element,
) -> tuple[list[StaveBbox], dict[int, list[Glyph]], dict[str, ET.Element],
           dict[int, list[tuple[Optional[str], tuple[str, ...], Optional[dict]]]],
           dict[int, list[tuple[float, ET.Element]]]]:
    """Reverse of what build_mei writes: walks an already-built MEI document
    back into (staves, glyphs_by_stave) -- the same shape build_mei
    originally took as input -- plus a glyph_id -> <neume> element map (so
    verify_and_correct_syllables can re-parent an existing, already
    pitch-computed <neume> element under a corrected <syllable> without
    recomputing pitch at all) and, per stave_idx, the CURRENT sequence of
    (syl_text, glyph_ids, syl_box) actually encoded, for comparison against
    a freshly recomputed one. syl_box (the syllable's own facs-linked zone,
    {"ul": [x, y], "lr": [x, y]}, or None if it has none) lets
    verify_and_correct_syllables's duplicate guard tell "this is the same
    physical mothra-text box, just re-bucketed" apart from "this is a
    different, legitimately repeated word" -- see that function's comments.

    Trusts the existing <sb>-delimited stave groupings as-is -- see
    verify_and_correct_syllables's docstring for why this does not redo
    _group_staves_by_row's cross-stave repair."""
    ns = f"{{{MEI_NS}}}"
    surface = root.find(f".//{ns}surface")
    zones: dict[str, ET.Element] = {}
    if surface is not None:
        for zone in surface.findall(f"{ns}zone"):
            zid = zone.get(XML_ID)
            if zid:
                zones[zid] = zone

    def _box(zone: ET.Element) -> tuple[int, int, int, int]:
        return (int(zone.get("ulx", 0)), int(zone.get("uly", 0)),
                int(zone.get("lrx", 0)), int(zone.get("lry", 0)))

    # dict preserves insertion order, and build_mei writes zones in
    # document order -- so this is already stave_idx order.

    staff_zone_ids = [zid for zid, z in zones.items() if z.get("type") == "staff"]
    staves: list[StaveBbox] = []
    zone_id_to_stave_idx: dict[str, int] = {}
    for i, zid in enumerate(staff_zone_ids):
        stave_id = zid.removeprefix("sz-")
        # Prefer the raw (pre-_stave_zone_bounds) box-assignment geometry
        # build_mei originally ran _assign_boxes_to_staves against, when
        # present -- see build_mei's "staff-raw" zone comment (mothra#208).
        # Falls back to the rendered zone's own (already Verovio-spacing-
        # distorted) bounds for MEI files encoded before this fix existed,
        # same as today's behavior.
        raw_zone = zones.get(f"sz-raw-{stave_id}")
        ulx, uly, lrx, lry = _box(raw_zone) if raw_zone is not None else _box(zones[zid])
        staves.append(StaveBbox(id=stave_id, ulx=ulx, uly=uly, lrx=lrx, lry=lry))
        zone_id_to_stave_idx[zid] = i

    layer = root.find(f".//{ns}layer")
    glyphs_by_stave: dict[int, list[Glyph]] = {i: [] for i in range(len(staves))}
    neume_elements: dict[str, ET.Element] = {}
    current_syllables: dict[int, list[tuple[Optional[str], tuple[str, ...]]]] = {
        i: [] for i in range(len(staves))
    }
    special_elements_by_stave: dict[int, list[tuple[float, ET.Element]]] = {
        i: [] for i in range(len(staves))
    }
    current_stave_idx: Optional[int] = None
    seen_opening_clef = False
    if layer is not None:
        for child in layer:
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "sb":
                facs = (child.get("facs") or "").lstrip("#")
                current_stave_idx = zone_id_to_stave_idx.get(facs)
                seen_opening_clef = False
            elif tag == "clef" and current_stave_idx is not None and not seen_opening_clef:
                # build_mei's unconditional per-stave opening clef -- left
                # untouched by verify_and_correct_syllables entirely (never
                # removed/re-inserted, unlike everything captured below).
                seen_opening_clef = True
            elif tag in ("clef", "custos", "divLine") and current_stave_idx is not None:
                facs = (child.get("facs") or "").lstrip("#")
                zone = zones.get(facs)
                x_key = float(zone.get("ulx", 0)) if zone is not None else 0.0
                special_elements_by_stave[current_stave_idx].append((x_key, child))
            elif tag == "syllable" and current_stave_idx is not None:
                syl_el = child.find(f"{ns}syl")
                syl_text = syl_el.text if syl_el is not None else None
                syl_box: Optional[dict] = None
                if syl_el is not None:
                    syl_facs = (syl_el.get("facs") or "").lstrip("#")
                    syl_zone = zones.get(syl_facs)
                    if syl_zone is not None:
                        ulx, uly, lrx, lry = _box(syl_zone)
                        syl_box = {"ul": [ulx, uly], "lr": [lrx, lry]}
                glyph_ids: list[str] = []
                for neume in child.findall(f"{ns}neume"):
                    facs = (neume.get("facs") or "").lstrip("#")
                    zone = zones.get(facs)
                    if zone is None or not facs.startswith("z-"):
                        continue
                    glyph_id = facs[len("z-"):]
                    ulx, uly, lrx, lry = _box(zone)
                    glyphs_by_stave[current_stave_idx].append(Glyph(
                        id=glyph_id, ulx=ulx, uly=uly,
                        ncols=max(lrx - ulx, 1), nrows=max(lry - uly, 1),
                        class_name="", confidence=1.0, state="AUTOMATIC",
                    ))
                    neume_elements[glyph_id] = neume
                    glyph_ids.append(glyph_id)
                current_syllables[current_stave_idx].append((syl_text, tuple(glyph_ids), syl_box))
    return staves, glyphs_by_stave, neume_elements, current_syllables, special_elements_by_stave

def _same_physical_box(a: dict, b: dict, min_iou: float = 0.3) -> bool:
    """True if boxes a/b are almost certainly the same underlying
    mothra-text detection (allowing for the sub-pixel rounding
    scale_text_alignment's independent X/Y factors can introduce), as
    opposed to two boxes that just happen to hold the same text -- e.g. a
    genuinely repeated word like "Alleluia" appearing twice on the page.
    Compared by intersection-over-union rather than exact coordinate
    equality for exactly that rounding-tolerance reason; verify_and_
    correct_syllables's duplicate guard needs this because a real
    fragment-pooling duplicate is the SAME box re-bucketed to the wrong
    stave, while a real repeated word is a DIFFERENT, spatially distant
    box that happens to share the same syl text."""
    ax0, ay0 = a["ul"]
    ax1, ay1 = a["lr"]
    bx0, by0 = b["ul"]
    bx1, by1 = b["lr"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    intersection = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(ax1 - ax0, 1) * max(ay1 - ay0, 1)
    area_b = max(bx1 - bx0, 1) * max(by1 - by0, 1)
    union = area_a + area_b - intersection
    return union > 0 and intersection / union >= min_iou

def verify_and_correct_syllables(
        mei_bytes: bytes,
        text_alignment: Optional[dict],
        image_w: int,
        image_h: int,
        syllable_gap_mult: float = SYLLABLE_GAP_MULTIPLIER,
) -> tuple[bytes, list[str]]:
    """Re-verify an already-built MEI's <syllable> text+order against
    mothra-text's CURRENT text_alignment for the same image, correcting it
    in place when they disagree (mothra-text wins). Called from
    mei_api.py's create_edit_session, right before Neon opens a
    not-yet-human-corrected MEI.

    Reuses _assign_boxes_to_staves/_assign_glyphs_to_boxes/
    _build_syllable_units -- the exact same logic build_mei itself uses --
    against glyph/stave geometry reconstructed straight out of the
    existing MEI (_reconstruct_mei_state), so a "correction" here can
    never disagree with what a fresh build_mei call would have produced
    for the same inputs.

    Scope boundary: does NOT re-run _group_staves_by_row's row-
    fragmentation repair -- see _reconstruct_mei_state's docstring; which
    staves were originally detected vs. synthesized isn't recoverable
    from a finished MEI. A glyph already in the wrong stave bucket is out
    of scope; a glyph in the RIGHT stave bucket with the wrong/misordered
    syllable text is exactly what this fixes.

    Returns (possibly-unchanged mei_bytes, list of human-readable
    correction log lines -- empty when nothing needed fixing)."""
    syl_boxes = text_alignment.get("syl_boxes", []) if text_alignment else []
    if not syl_boxes:
        return mei_bytes, []

    # Same stale/mismatched-resolution guard build_mei itself uses -- if
    # most boxes fall outside this page, don't risk a correction built on
    # bad coordinates.
    invalid = sum(1 for b in syl_boxes if not _text_box_valid(b, image_w, image_h))
    if invalid > len(syl_boxes) / 2:
        return mei_bytes, []

    root = ET.fromstring(mei_bytes)
    ns = f"{{{MEI_NS}}}"
    staves, glyphs_by_stave, neume_elements, current_syllables, special_elements_by_stave = _reconstruct_mei_state(root)
    surface = root.find(f".//{ns}surface")
    layer = root.find(f".//{ns}layer")
    if not staves or surface is None or layer is None:
        return mei_bytes, []

    boxes_by_stave = _assign_boxes_to_staves(staves, syl_boxes)
    staff_zone_id_by_idx = {i: f"sz-{stave.id}" for i, stave in enumerate(staves)}
    logs: list[str] = []

    # A box that's genuinely a fragment-pooled row-mate's (see
    # _pool_group_syllable_data) can be Y-bucketed here, unpooled, to a
    # DIFFERENT stave than the one whose glyphs it's actually paired with —
    # that stave then computes it as a real box with zero glyphs matched,
    # even though its true glyphs (and its already-correct <syllable>) live
    # on the row-mate. Detecting that specific case doesn't need re-running
    # the pooling (out of scope here, see below) -- it only needs to know
    # whether this exact text is ALREADY backed by real glyphs somewhere
    # else in the document; if so, this glyph-less copy is that duplicate,
    # not new content.
    old_boxes_with_glyphs_by_text: dict[str, list[dict]] = {}
    for stave_units in current_syllables.values():
        for syl_text, glyph_ids, old_box in stave_units:
            if glyph_ids and syl_text is not None and old_box is not None:
                old_boxes_with_glyphs_by_text.setdefault(syl_text, []).append(old_box)

    for stave_idx in range(len(staves)):
        neume_glyphs = sorted(glyphs_by_stave.get(stave_idx, []), key=lambda g: g.ulx)
        stave_boxes = boxes_by_stave.get(stave_idx, [])
        glyph_groups = _assign_glyphs_to_boxes(neume_glyphs, stave_boxes) if stave_boxes else []
        new_units = _build_syllable_units(neume_glyphs, stave_boxes, glyph_groups, syllable_gap_mult)

        # _reconstruct_mei_state's staves come from the WRITTEN <staffDef>
        # zone -- _stave_zone_bounds()'s tighter, Verovio-spacing-matched
        # band, not the raw stave bbox build_mei originally ran
        # _assign_boxes_to_staves against (see that function's docstring).
        # A real syl_box that landed correctly on this stave at encode time
        # can therefore fall just outside the tighter reconstructed band
        # here and get bucketed elsewhere, leaving this stave's own glyphs
        # with no box -- i.e. _build_syllable_units's "-" no-text fallback,
        # even though its row-mate had the real word right there and
        # nothing about mothra-text's data actually changed. Guard against
        # that specific regression: never let a "-" fallback overwrite an
        # existing real syl_text for the exact same glyph grouping -- only
        # a genuine glyph-membership change (row split/merge, reordering)
        # or a genuine text change should ever get written.
        #
        # NB: a unit with glyph_ids empty is NEVER the "-" fallback --
        # _build_syllable_units only emits "-" via cluster_into_syllables,
        # which clusters actual existing neume glyphs, so a "-" unit always
        # carries at least one. glyph_ids empty means stave_boxes was
        # non-empty and this specific real syl_box just has no note matched
        # to it in this stave-scoped, unpooled recheck: either a real "text
        # with nothing under it" box (legitimate, build_mei can produce
        # these), or a row-mate's fragment-pooled box this simplified check
        # can't see the glyph match for (see the module-level
        # old_texts_with_glyphs_anywhere comment above, and
        # _reconstruct_mei_state's docstring: redoing that pooling is out
        # of scope here). Distinguish the two by whether this exact text is
        # already backed by real glyphs somewhere else in the document —
        # if so it's that duplicate, drop it; otherwise it's genuinely new
        # data, keep it. An earlier version of this guard instead checked
        # only THIS stave's own old empty-glyph units, which silently
        # deleted real new syllables (and their boxes) the moment their
        # glyph-matching outcome changed for any reason, including
        # unrelated fixes upstream — too narrow. A version after that
        # dropped the check entirely, which let the fragment-pooled
        # duplicate case back in — too broad. This is the middle ground.
        old_units = current_syllables.get(stave_idx, [])
        old_text_by_glyphs = {glyph_ids: syl_text for syl_text, glyph_ids, _ in old_units}
        reconciled_units = []
        for syl_text, box, glyphs_in_syl in new_units:
            glyph_ids = tuple(g.id for g in glyphs_in_syl)
            if glyph_ids:
                old_text = old_text_by_glyphs.get(glyph_ids)
                if syl_text == "-" and old_text not in (None, "-"):
                    syl_text = old_text
            elif box is not None and any(
                _same_physical_box(box, old_box)
                for old_box in old_boxes_with_glyphs_by_text.get(syl_text, [])
            ):
                continue
            reconciled_units.append((syl_text, box, glyphs_in_syl))
        new_units = reconciled_units

        new_signature = [(syl_text, tuple(g.id for g in glyphs)) for syl_text, _, glyphs in new_units]
        old_signature = [(syl_text, glyph_ids) for syl_text, glyph_ids, _ in old_units]
        if new_signature == old_signature:
            continue # matches mothra-text

        children = list(layer)
        zone_id = staff_zone_id_by_idx[stave_idx]
        sb_idx = next(
            (i for i, c in enumerate(children)
             if c.tag == f"{ns}sb" and (c.get("facs") or "").lstrip("#") == zone_id),
             None,
        )
        if sb_idx is None:
            continue
        section_end = next(
            (i for i in range(sb_idx + 1, len(children)) if children[i].tag == f"{ns}sb"),
            len(children),
        )
        # Right after <sb> and the opening <clef>, when that clef is
        # present. A section whose opening clef was deleted downstream
        # (e.g. in Neon) must not shift the boundary into real content --
        # _reconstruct_mei_state only ever treats the FIRST clef/custos/
        # divLine after <sb> as "the opening clef, leave it alone"; if
        # that slot instead holds a real special element (opening clef
        # missing), _reconstruct_mei_state records it as an ordinary one
        # to preserve, and this boundary must include it in the removal
        # scan below or it gets re-inserted as a duplicate of itself.
        insert_at = sb_idx + 1
        if insert_at < section_end and children[insert_at].tag == f"{ns}clef":
            insert_at += 1

        # Remove this section's existing <syllable> elements (and their
        # syl-zone <zone> children, to avoid leaving orphaned zones behind)
        # AND any interspersed clef/custos/divLine special elements
        # (mothra#257) -- those are re-inserted below, merged back in by X
        # position with the freshly built syllables, rather than left in
        # place: leaving them in place while only removing <syllable>s and
        # then re-inserting all new syllables consecutively at insert_at
        # would silently push every special element after all the new
        # syllables, regardless of where it actually belongs.
        old_zone_ids: set[str] = set()
        special_els = special_elements_by_stave.get(stave_idx, [])
        special_el_ids = {id(el) for _, el in special_els}
        removed_special_ids: set[int] = set()
        for c in children[insert_at:section_end]:
            if c.tag == f"{ns}syllable":
                syl_el = c.find(f"{ns}syl")
                facs = (syl_el.get("facs") if syl_el is not None else None) or ""
                if facs.startswith("#"):
                    old_zone_ids.add(facs[1:])
                layer.remove(c)
            elif id(c) in special_el_ids:
                layer.remove(c)
                removed_special_ids.add(id(c))
        if old_zone_ids:
            for zone_el in list(surface.findall(f"{ns}zone")):
                if zone_el.get(XML_ID) in old_zone_ids:
                    surface.remove(zone_el)

        # Build fresh <syllable> elements (re-parenting the EXISTING
        # <neume> element for each glyph -- already correctly pitched,
        # untouched) and merge them back in, by X position, with the
        # preserved special elements removed above -- so a divLine/custos/
        # inline clef-change keeps its original position relative to the
        # syllables around it instead of always landing after all of them.
        merged: list[tuple[float, ET.Element]] = [
            (x, el) for x, el in special_els if id(el) in removed_special_ids
        ]
        for syl_text, box, glyphs_in_syl in new_units:
            syllable_id = glyphs_in_syl[0].id if glyphs_in_syl else str(uuid.uuid4()).replace("-", "")[:12]
            syllable = ET.Element(_tag("syllable"), {XML_ID: f"syllable-{syllable_id}"})
            syl_attrs: dict[str, str] = {XML_ID: f"syl-{str(uuid.uuid4()).replace('-', '')[:12]}"}
            if box is not None and _text_box_valid(box, image_w, image_h):
                syl_zone_id = f"zone-syl-{str(uuid.uuid4()).replace('-', '')[:12]}"
                ulx, uly = box["ul"]
                lrx, lry = box["lr"]
                ET.SubElement(surface, _tag("zone"), {
                    XML_ID: syl_zone_id,
                    "ulx": str(int(ulx)), "uly": str(int(uly)),
                    "lrx": str(int(lrx)), "lry": str(int(lry)),
                })
                syl_attrs["facs"] = f"#{syl_zone_id}"
            syl = ET.SubElement(syllable, _tag("syl"), syl_attrs)
            syl.text = syl_text
            for glyph in glyphs_in_syl:
                neume_el = neume_elements.get(glyph.id)
                if neume_el is not None:
                    syllable.append(neume_el)
            x_key = box["ul"][0] if box is not None else min((g.ulx for g in glyphs_in_syl), default=0.0)
            merged.append((x_key, syllable))

        for offset, (_, el) in enumerate(sorted(merged, key=lambda item: item[0])):
            layer.insert(insert_at + offset, el)

        logs.append(f" [verify-syllables] stave {stave_idx}: corrected "
                    f"{len(new_units)} syllable(s) to match text-finding")
    if not logs:
        return mei_bytes, []
    return _serialize_mei(root), logs


def main():
    parser = argparse.ArgumentParser(
        description="GameraXML + Mothra inference JSON -> MEI (placeholder pitch, human-corrected in Neon) + Neon manifest."
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
    parser.add_argument("--notation-type", type=str, default="square", choices=["square", "hufnagel"],
                        help="which bundled neume-to-MEI mapping CSV to use (default: square)")
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

    n_detected_staves = len(staves)
    glyphs_by_stave, staves = assign_glyphs_to_staves(glyphs, staves, image_w, image_h)
    assigned = sum(len(v) for k, v in glyphs_by_stave.items() if k >= 0)
    print(f" {assigned} glyphs assigned across {len([k for k in glyphs_by_stave if k >= 0])} staves")

    mei_bytes = build_mei(glyphs_by_stave, staves, args.image.resolve(), image_w, image_h, ms_name,
                          syllable_gap_mult=args.syllable_gap_mult, notation_type=args.notation_type,
                          n_detected_staves=n_detected_staves)
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