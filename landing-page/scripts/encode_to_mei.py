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

import argparse
import base64
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
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

def assign_glyphs_to_staves(
        glyphs: list[Glyph], staves: list[StaveBbox]
) -> dict[int, list[Glyph]]:
    result: dict[int, list[Glyph]] = {i: [] for i in range(len(staves))}
    if not staves:
        result[-1] = list(glyphs)
        return result
    for glyph in glyphs:
        cy = glyph.cy
        best_idx = None
        best_overlap = -1
        for i, stave in enumerate(staves):
            lo = stave.uly - STAVE_BUFFER_PX
            hi = stave.lry + STAVE_BUFFER_PX
            if lo <= cy <= hi:
                overlap = min(cy, hi) - max(cy, lo)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = i
        if best_idx is None:
            best_idx = min(range(len(staves)), key=lambda i: abs(staves[i].cy - cy))
        result[best_idx].append(glyph)
    for idx in result:
        result[idx].sort(key=lambda g: g.ulx)
    return result

def cluster_into_syllables(glyphs: list[Glyph]) -> list[list[Glyph]]:
    if not glyphs:
        return []
    threshold = SYLLABLE_GAP_MULTIPLIER * median(g.ncols for g in glyphs)
    clusters: list[list[Glyph]] = [[glyphs[0]]]
    for glyph in glyphs[1:]:
        if glyph.ulx - clusters[-1][-1].lrx > threshold:
            clusters.append([glyph])
        else:
            clusters[-1].append(glyph)
    return clusters

def estimate_staves_from_glyphs(
    glyphs: list[Glyph], page_w: int, page_h: int
) -> list[StaveBbox]:
    """Cluster glyphs by Y-center into approximate stave bounding boxes.

    GameraXML typically includes detected staff lines (very wide, very short
    glyphs with ncols/nrows >> 1) alongside actual neumes. Staff lines shift
    avg_h down and appear as bridging glyphs between rows in the sorted-cy
    sequence, collapsing all rows into one cluster.

    Fix: exclude staff-line-like glyphs (aspect ratio ≥ 8) from avg_h
    calculation and from clustering. Use consecutive-gap comparison on the
    sorted cy sequence: a gap > 1.5× avg neume height reliably falls between
    rows while staying below the ~79 px neume-only inter-row gap.
    """
    if not glyphs:
        return [StaveBbox(id="synth-0", ulx=0, uly=0, lrx=page_w, lry=page_h)]

    # Staff lines have ncols >> nrows (ratio ≥ 8); neumes are roughly square.
    # Exclude them so they don't distort avg_h or bridge inter-row gaps.
    neume_like = [g for g in glyphs if g.nrows > 0 and g.ncols / g.nrows < 8]
    if not neume_like:
        neume_like = glyphs

    avg_h = median(g.nrows for g in neume_like)

    # Also drop very tall outliers (large initials, decorations).
    representative = [g for g in neume_like if g.nrows <= avg_h * 3] or neume_like

    # Sort and split on CONSECUTIVE cy gaps.
    # Within a row these gaps are ≲ 44 px; between rows they are ≳ 79 px
    # (neume-only), so 1.5 × avg_h ≈ 72 px cleanly separates them.
    sorted_glyphs = sorted(representative, key=lambda g: g.cy)
    gap_threshold = avg_h * 1.5

    rows: list[list[Glyph]] = [[sorted_glyphs[0]]]
    for i in range(1, len(sorted_glyphs)):
        if sorted_glyphs[i].cy - sorted_glyphs[i - 1].cy > gap_threshold:
            rows.append([sorted_glyphs[i]])
        else:
            rows[-1].append(sorted_glyphs[i])

    pad = max(5, int(avg_h * 0.3))
    staves = []
    for i, row in enumerate(rows):
        staves.append(StaveBbox(
            id=f"auto-{i}",
            ulx=max(0, min(g.ulx for g in row) - pad),
            uly=max(0, min(g.uly for g in row) - pad),
            lrx=min(page_w, max(g.lrx for g in row) + pad),
            lry=min(page_h, max(g.lry for g in row) + pad),
        ))
    return staves

def _tag(local: str) -> str:
    return f"{{{MEI_NS}}}{local}"

def build_mei(
    glyphs_by_stave: dict[int, list[Glyph]],
    staves: list[StaveBbox],
    image_path: Path,
    image_w: int,
    image_h: int,
    manuscript_name: str,
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
    facsimile = ET.SubElement(music, _tag("facsimile"))
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

    # stave zones (used by <staff @facs>)
    stave_zone_ids: dict[int, str] = {}
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

    all_glyphs = [g for idx in sorted(glyphs_by_stave) for g in glyphs_by_stave[idx]]
    for glyph in all_glyphs:
        ET.SubElement(surface, _tag("zone"), {
            XML_ID: f"z-{glyph.id}",
            "ulx": str(glyph.ulx),
            "uly": str(glyph.uly),
            "lrx": str(glyph.lrx),
            "lry": str(glyph.lry),
        })

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
        "clef.shape": "C",
        "clef.line": "3",
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
        # Clef must follow each <sb> so Verovio has a reference frame per stave
        clef_id = str(uuid.uuid4()).replace("-", "")[:12]
        ET.SubElement(layer, _tag("clef"), {
            XML_ID: f"clef-{clef_id}",
            "shape": "C",
            "line": "3",
        })
        for cluster in cluster_into_syllables(staff_glyphs):
            syllable_id = cluster[0].id
            syllable = ET.SubElement(layer, _tag("syllable"), {
                XML_ID: f"syllable-{syllable_id}",
            })
            syl_id = str(uuid.uuid4()).replace("-", "")[:12]
            syl = ET.SubElement(syllable, _tag("syl"), {XML_ID: f"syl-{syl_id}"})
            syl.text = "-"
            for glyph in cluster:
                neume = ET.SubElement(syllable, _tag("neume"), {
                    XML_ID: f"neume-{glyph.id}",
                    "facs": f"#z-{glyph.id}",
                })
                ET.SubElement(neume, _tag("nc"), {
                    XML_ID: f"nc-{glyph.id}",
                    "pname": "a",
                    "oct": "3",
                })
    
    ET.indent(mei, space=" ")
    xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(mei, encoding="unicode")
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

    zones: dict[str, str] = {}
    for zone in root.iter(_tag("zone")):
        zid = zone.get(XML_ID, "")
        if zid:
            zones[zid] = zone.get("type", "")
    
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

    #nc: xml:id + pname + oct
    for nc in root.iter(_tag("nc")):
        ncid = nc.get(XML_ID, "")
        for attr in ("pname", "oct"):
            if not nc.get(attr):
                warnings.append(f"nc {ncid or '?'}: missing @{attr}")

    return warnings

def build_neon_manifest(mei_bytes: bytes, image_ref: str, stem: str) -> dict:
    mei_b64 = base64.b64encode(mei_bytes).decode()
    return {
        "@context": "https://ddmal.music.mcgill.ca/Neon/contexts/1/manifest.jsonld",
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
    print(f"{len(staves)} staves found")

    glyphs_by_stave = assign_glyphs_to_staves(glyphs, staves)
    assigned = sum(len(v) for k, v in glyphs_by_stave.items() if k >= 0)
    print(f" {assigned} glyphs assigned across {len([k for k in glyphs_by_stave if k >= 0])} staves")

    mei_bytes = build_mei(glyphs_by_stave, staves, args.image.resolve(), image_w, image_h, ms_name)
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