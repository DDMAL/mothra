"""
Run the whole staff-finding pipeline on one or more staff-line layers.

For each layer image:
  1. staff_finding.py (or staff_fragments.py): fragments, line/noise labels
       -> <name>_fragments.json, <name>_overlay.png, <name>_classified.png
  2. link_lines.py: lines, staves, completed curves
       -> <name>_lines.json, <name>_lines.png (drawn over the original page if found)
Then writes summary.csv with one row per page.

Usage:
    python run_pipeline.py layers/                       # every image in a folder
    python run_pipeline.py layers/*.png --originals pages/
    python run_pipeline.py layer.png --lines-per-staff 4
    python run_pipeline.py layers/ --link-args "--long-gap 10 --lat 0.3"

Originals are matched by name: the layer's file name minus a prefix such as
"stafflines_" or "stafflines2_" (change with --layer-prefix), any extension.
"""
import argparse
import csv
import json
import re
import shlex
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    import staff_finding as staff
except ImportError:
    import staff_fragments as staff
import link_lines

IMAGE_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


def collect(inputs):
    out = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            out.extend(sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS))
        elif p.exists():
            out.append(p)
        else:
            print(f"skipping {item}: not found", file=sys.stderr)
    return out


def find_original(layer, originals_dir, prefix_re):
    if not originals_dir:
        return None
    base = re.sub(prefix_re, "", layer.stem)
    for q in sorted(Path(originals_dir).iterdir()):
        if q.suffix.lower() in IMAGE_EXTS and q.stem in (base, layer.stem) and q != layer:
            return q
    return None


def summarize(lines_json):
    r = json.load(open(lines_json))
    staved = [L for L in r["lines"] if L["staff"] is not None]
    n_pts = sum(len(L["points"]) for L in staved)
    n_pred = sum(sum(L["predicted"]) for L in staved)
    return {
        "staff_space": r["staff_space"],
        "staves": len(r["staves"]),
        "lines_per_staff": " ".join(str(s["n_lines"]) for s in r["staves"]),
        "unstaved_lines": len(r["unstaved_line_ids"]),
        "unlinked_pieces": len(r["unlinked_fragment_ids"]),
        "gutters": len(r.get("barriers_x", [])),
        "predicted_share": round(n_pred / n_pts, 3) if n_pts else "",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Staff layers -> fragments -> lines and staves.")
    p.add_argument("inputs", nargs="+", help="staff-layer images and/or folders of them")
    p.add_argument("--originals", help="folder of original page images, for the overlays")
    p.add_argument("--layer-prefix", default=r"^stafflines\d*_",
                   help="regex removed from a layer's name to find its original "
                        "(default: '^stafflines\\d*_')")
    p.add_argument("--out-dir", default=str(HERE / "outputs"))
    p.add_argument("--invert", action="store_true", help="layer ink is light on dark")
    p.add_argument("--lines-per-staff", type=int, help="keep the N best-inked lines per staff")
    p.add_argument("--staff-args", default="", help='extra options for step 1, e.g. "--tol 0.2"')
    p.add_argument("--link-args", default="", help='extra options for step 2, e.g. "--lat 0.3"')
    a = p.parse_args(argv)

    layers = collect(a.inputs)
    if not layers:
        p.error("no input images found")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, layer in enumerate(layers, 1):
        print(f"\n[{i}/{len(layers)}] {layer.name}", file=sys.stderr)
        row = {"layer": str(layer)}
        try:
            step1 = [str(layer), "--out-dir", str(out)] + shlex.split(a.staff_args)
            if a.invert:
                step1.append("--invert")
            staff.main(step1)

            frag_json = out / f"{layer.stem}_fragments.json"
            original = find_original(layer, a.originals, a.layer_prefix)
            row["original"] = str(original) if original else ""
            step2 = [str(frag_json), "--layer", str(layer), "--out-dir", str(out)]
            if original:
                step2 += ["--image", str(original)]
            if a.invert:
                step2.append("--invert")
            if a.lines_per_staff:
                step2 += ["--lines-per-staff", str(a.lines_per_staff)]
            step2 += shlex.split(a.link_args)
            link_lines.main(step2)

            row.update(summarize(out / f"{layer.stem}_lines.json"))
            row["status"] = "ok"
        except SystemExit as e:          # argparse errors inside a step
            row["status"] = f"failed (exit {e.code})"
        except Exception as e:
            traceback.print_exc()
            row["status"] = f"failed: {type(e).__name__}: {e}"
        rows.append(row)

    cols = ["layer", "original", "status", "staff_space", "staves", "lines_per_staff",
            "unstaved_lines", "unlinked_pieces", "gutters", "predicted_share"]
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"\n{ok}/{len(rows)} pages done -> {out}/ (see summary.csv)", file=sys.stderr)


if __name__ == "__main__":
    main()