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
    mei = ET.Element(_tag("mei"), {"meiversion": "4.0.1"})

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

    n_staves = max(1, len([k for k in glyphs_by_stave if k >= 0]))
    score_def = ET.SubElement(score, _tag("scoreDef"))
    staff_grp = ET.SubElement(score_def, _tag("staffGrp"))
    for i in range(n_staves):
        ET.SubElement(staff_grp, _tag("staffDef"), {
            "n": str(i + 1),
            "lines": "4",
            "notationtype": "neume",
        })

    section = ET.SubElement(score, _tag("section"))

    for n, stave_idx in enumerate(sorted(k for k in glyphs_by_stave if k >= 0), start=1):
        staff_glyphs = glyphs_by_stave[stave_idx]
        if not staff_glyphs:
            continue
        staff_attrs: dict[str, str] = {"n": str(n)}
        if stave_idx in stave_zone_ids:
            staff_attrs["facs"] = f"#{stave_zone_ids[stave_idx]}"
        staff_el = ET.SubElement(section, _tag("staff"), staff_attrs)
        layer = ET.SubElement(staff_el, _tag("layer"), {"n": "1"})
        for cluster in cluster_into_syllables(staff_glyphs):
            syllable = ET.SubElement(layer, _tag("syllable"))
            syl = ET.SubElement(syllable, _tag("syl"))
            syl.text = "-"
            for glyph in cluster:
                neume = ET.SubElement(syllable, _tag("neume"))
                ET.SubElement(neume, _tag("nc"), {
                    XML_ID: f"nc-{glyph.id}",
                    "facs": f"#z-{glyph.id}",
                })
    
    ET.indent(mei, space=" ")
    xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(mei, encoding="unicode")
    return xml_str.encode("utf-8")

def build_neon_manifest(mei_bytes: bytes, image_path: Path, stem: str) -> dict:
    mei_b64 = base64.b64encode(mei_bytes).decode()
    image_ref = str(image_path)
    return {
        "@context": "https://ddmal.music.mcgill.ca/Neon/contexts/2/manifest.jsonld",
        "@id": f"{stem}-manifest",
        "title": stem,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image": image_ref,
        "mei_annotations": [
            {
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

    mei_path = out_dir / f"{stem}.mei"
    mei_path.write_bytes(mei_bytes)
    print(f"MEI written: {mei_path}")

    manifest = build_neon_manifest(mei_bytes, args.image.resolve(), stem)
    manifest_path = out_dir / f"{stem}_manifest.jsonld"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written: {manifest_path}")
    print(f"\nLoad in Neon: editor.html?manifest={manifest_path.resolve()}")

if __name__ == "__main__":
    main()