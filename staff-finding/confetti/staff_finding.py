"""
Turn a binary staff-line layer (e.g. Paco output) into described fragments
ready for heuristic linking.

Pipeline:
  mask -> light cleanup -> skeleton -> skeleton graph (skan)
       -> prune spurs / split at junctions -> keep near-horizontal paths
       -> per-fragment descriptors (endpoints, end tangents, thickness, y, length)

Requires: numpy, scipy, scikit-image, skan  (pip install skan)
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology
from skan import Skeleton


@dataclass
class Fragment:
    id: int
    path: np.ndarray            # (N, 2) ordered (row, col) skeleton pixels, left -> right
    thickness: float            # median stroke width in px (2 * distance transform)
    length: float               # skeleton path length in px
    left: np.ndarray            # (row, col) of left endpoint
    right: np.ndarray           # (row, col) of right endpoint
    left_dir: np.ndarray        # unit vector pointing OUT of the fragment at the left end
    right_dir: np.ndarray       # unit vector pointing OUT of the fragment at the right end
    angle_deg: float            # overall slope from a line fit
    is_confetti: bool = False   # too short for a reliable tangent
    meta: dict = field(default_factory=dict)


def clean_mask(mask, min_area=8, bridge_px=0):
    """Remove specks; optionally bridge tiny horizontal gaps.

    bridge_px: width of a 1-px-tall horizontal closing. Keep it well under the
    smallest real gap you care about, and never use a tall structuring element
    (it will merge adjacent staff lines).
    """
    m = mask.astype(bool)
    if bridge_px > 1:
        m = morphology.closing(m, np.ones((1, bridge_px), bool))
    lab, _ = ndi.label(m, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[lab]


def _end_direction(path, at_start, n):
    """Unit vector pointing outward from one end, fit over the last n pixels."""
    seg = path[:n] if at_start else path[-n:]
    if len(seg) < 2:
        return np.array([0.0, -1.0 if at_start else 1.0])
    # PCA direction of the end segment (robust to pixel staircasing)
    c = seg - seg.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    d = vt[0]
    # orient it outward: from interior toward the endpoint
    outward = seg[0] - seg[-1] if at_start else seg[-1] - seg[0]
    if np.dot(d, outward) < 0:
        d = -d
    return d / (np.linalg.norm(d) + 1e-9)


def extract_fragments(mask, *, min_area=8, bridge_px=0,
                      spur_factor=2.0, max_angle_deg=30.0,
                      confetti_len=None, tangent_px=None):
    """Return a list of Fragment objects from a binary staff-line mask.

    spur_factor:   branches ending in a tip and shorter than spur_factor * stroke
                   width are treated as skeleton spurs and dropped.
    max_angle_deg: paths steeper than this are discarded (stems, ascenders,
                   Paco bleed-through from neumes or text).
    confetti_len:  paths shorter than this are flagged is_confetti (default
                   3 * stroke width). Treat them as points, not as link nodes.
    tangent_px:    how many skeleton pixels to use for end tangents
                   (default ~4 * stroke width).
    """
    m = clean_mask(mask, min_area=min_area, bridge_px=bridge_px)
    if not m.any():
        return []

    dist = ndi.distance_transform_edt(m)
    skel = morphology.skeletonize(m)

    # global stroke width estimate, used to scale every other threshold
    stroke = float(np.median(2 * dist[skel])) if skel.any() else 2.0
    confetti_len = confetti_len or 3 * stroke
    tangent_px = int(tangent_px or max(4, round(4 * stroke)))

    sk = Skeleton(skel)
    n_paths = sk.n_paths
    deg = sk.degrees  # per-node degree in the skeleton graph

    frags = []
    for i in range(n_paths):
        coords = sk.path_coordinates(i)          # ordered along the path
        node_ids = sk.path(i)
        if len(coords) < 2:
            continue
        end_deg = (deg[node_ids[0]], deg[node_ids[-1]])
        length = float(sk.path_lengths()[i])

        # 1) spur pruning: a branch that dead-ends on one side and hits a
        #    junction on the other, and is short relative to stroke width
        is_tip_to_junction = (1 in end_deg) and any(d >= 3 for d in end_deg)
        if is_tip_to_junction and length < spur_factor * stroke:
            continue

        # 2) orientation filter: drop steep paths (stems / non-staff ink)
        c = coords - coords.mean(0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        dr, dc = vt[0]
        angle = np.degrees(np.arctan2(-dr, dc))  # image rows grow downward
        angle = (angle + 90) % 180 - 90          # fold to [-90, 90)
        if abs(angle) > max_angle_deg:
            continue

        # order left -> right
        if coords[0, 1] > coords[-1, 1]:
            coords = coords[::-1]

        rr, cc = coords[:, 0].astype(int), coords[:, 1].astype(int)
        thickness = float(np.median(2 * dist[rr, cc]))
        tp = min(tangent_px, len(coords))

        frags.append(Fragment(
            id=len(frags),
            path=coords,
            thickness=thickness,
            length=length,
            left=coords[0],
            right=coords[-1],
            left_dir=_end_direction(coords, True, tp),
            right_dir=_end_direction(coords, False, tp),
            angle_deg=float(angle),
            is_confetti=length < confetti_len,
            meta={"end_degrees": end_deg},
        ))
    return frags


def link_cost(a, b, staff_space, max_gap):
    """Cost of joining a's right end to b's left end (lower is better; inf = forbidden).

    Combines: gap length, lateral offset from a's extrapolated tangent,
    turning angle, and thickness mismatch.
    """
    v = b.left - a.right                     # (drow, dcol)
    gap = np.linalg.norm(v)
    if v[1] <= 0 or gap > max_gap:           # must move rightward, within reach
        return np.inf
    d = a.right_dir
    along = float(np.dot(v, d))
    lateral = abs(float(d[0] * v[1] - d[1] * v[0]))   # perpendicular distance
    if lateral > 0.35 * staff_space:          # would jump toward the next line
        return np.inf
    turn = np.degrees(np.arccos(np.clip(np.dot(a.right_dir, -b.left_dir), -1, 1)))
    thick = abs(a.thickness - b.thickness) / max(a.thickness, b.thickness)
    return (gap / max_gap) + 2.0 * (lateral / staff_space) + turn / 45.0 + 0.5 * thick + 0 * along


# ---------------------------------------------------------------- line vs noise

def _y_at(f, xs):
    """Row of fragment f's centerline at columns xs (clamped beyond its ends)."""
    cols = np.maximum.accumulate(f.path[:, 1].astype(float))
    return np.interp(xs, cols, f.path[:, 0].astype(float))


def estimate_staff_space(frags, strip_w=200, min_lag=None):
    """Staff-line spacing from the autocorrelation of the row profile of all
    fragment centerlines, computed in vertical strips (so slant and curvature
    don't smear it) and summed. Works when every fragment is short.

    Takes the first strong autocorrelation peak, which is one staff space;
    later peaks are its multiples or the gap between staves.
    """
    if not frags:
        return float("nan")
    pts = np.concatenate([f.path for f in frags]).astype(int)
    rows, cols = pts[:, 0], pts[:, 1]
    H = rows.max() + 1
    stroke = float(np.median([f.thickness for f in frags]))
    min_lag = int(min_lag or max(4, 3 * stroke))
    max_lag = H // 3
    acc = np.zeros(max_lag + 1)
    for c0 in range(0, cols.max() + 1, strip_w):
        sel = (cols >= c0) & (cols < c0 + strip_w)
        if sel.sum() < 20:
            continue
        prof = np.bincount(rows[sel], minlength=H).astype(float)
        prof = np.convolve(prof, np.ones(3) / 3, mode="same")
        prof -= prof.mean()
        full = np.correlate(prof, prof, mode="full")[H - 1:H + max_lag]
        if full[0] > 0:
            acc += full / full[0]
    seg = acc[min_lag:]
    if not np.isfinite(seg).any() or seg.max() <= 0:
        return float("nan")
    # local maxima, then the first one reaching half the strongest peak
    peaks = [i for i in range(1, len(seg) - 1)
             if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0]
    if not peaks:
        return float("nan")
    top = max(seg[i] for i in peaks)
    first = next(i for i in peaks if seg[i] >= 0.5 * top)
    # parabolic refinement for sub-pixel lag
    y0, y1, y2 = seg[first - 1], seg[first], seg[first + 1]
    denom = y0 - 2 * y1 + y2
    off = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    return float(first + min_lag + off)


def classify_fragments(frags, staff_space=None, tol=0.25, max_k=3,
                       anchor_len=None, reach=None, rounds=3):
    """Label each fragment 'line' or 'noise' by staff-lattice support.

    A fragment is supported by an anchor when the anchor runs parallel to it
    at k staff spaces (k = 1..max_k, within +/- tol * sqrt(k) * staff_space).
    Anchors start as the longer pieces that support each other; then every
    piece labelled 'line' joins them for a few rounds, so support spreads
    into stretches where all the pieces are short. Being on the same line
    (k = 0) does not count, so a row of collinear junk can't vouch for itself.

    'line'  = long piece with >= 1 supporting offset, or short piece with >= 2.
    'noise' = everything else (between-line bleed, margins, page debris).
    Sets f.meta['label'] and f.meta['support'] (sorted signed k offsets).
    Returns the staff_space used.
    """
    s = staff_space or estimate_staff_space(frags)
    stroke = float(np.median([f.thickness for f in frags]))
    # anchors only need a trustworthy y-position, not great length; keep the
    # bar low so pages broken into short pieces still have plenty of them
    anchor_len = anchor_len or max(4 * stroke, 0.5 * s)
    reach = reach if reach is not None else 2 * s

    def offsets(a, anchors):
        xs = np.linspace(a.left[1], a.right[1], min(7, max(2, len(a.path))))
        ya = _y_at(a, xs)
        ks = set()
        for b in anchors:
            if b is a:
                continue
            if b.right[1] < a.left[1] - reach or b.left[1] > a.right[1] + reach:
                continue
            dy = float(np.median(_y_at(b, xs) - ya))
            r = abs(dy) / s
            k = int(round(r))
            # spacing errors add up across lines, so allow more slack further out
            if 1 <= k <= max_k and abs(r - k) <= tol * np.sqrt(k):
                ks.add(int(np.sign(dy)) * k)
        return ks

    def label_all(anchors):
        for f in frags:
            ks = offsets(f, anchors)
            need = 1 if f.length >= anchor_len else 2
            f.meta["support"] = sorted(ks)
            f.meta["label"] = "line" if len(ks) >= need else "noise"

    # anchors must themselves be supported by other anchors
    anchors = [f for f in frags if f.length >= anchor_len]
    for _ in range(2):
        anchors = [a for a in anchors if offsets(a, anchors)]

    # then let confirmed pieces of any length act as anchors too, so support
    # spreads into regions where every piece is short (badly broken staves)
    label_all(anchors)
    for _ in range(rounds):
        grown = [f for f in frags if f.meta["label"] == "line"]
        if len(grown) == len(anchors):
            break
        anchors = grown
        label_all(anchors)
    return s


def save_classified(mask, frags, out_path):
    """Grey = mask, green = suspected staff-line piece, red = confetti/noise."""
    from skimage import io
    from skimage.morphology import dilation, disk
    h, w = mask.shape if mask is not None else (None, None)
    rgb = np.full(mask.shape + (3,), 255, np.uint8)
    rgb[mask] = (215, 215, 215)
    for label, colour in (("noise", (215, 30, 30)), ("line", (0, 160, 70))):
        layer = np.zeros(mask.shape, bool)
        for f in frags:
            if f.meta.get("label") == label:
                layer[f.path[:, 0].astype(int), f.path[:, 1].astype(int)] = True
        rgb[dilation(layer, disk(2))] = colour
    io.imsave(out_path, rgb, check_contrast=False)


# ---------------------------------------------------------------- I/O

def load_mask(path, invert=False, threshold=128):
    """Load a staff-line layer as a boolean mask (True = staff ink).

    - RGBA images (e.g. Paco/Rodan layer PNGs with a transparent background):
      foreground = alpha > 0.
    - Otherwise: grayscale, foreground = dark pixels (< threshold).
      Use invert=True if your layer is white ink on black.
    """
    from skimage import io
    img = io.imread(path)
    if img.ndim == 3 and img.shape[2] == 4 and (img[..., 3] < 255).any():
        return img[..., 3] > 0
    if img.ndim == 3:
        img = (img[..., :3].astype(float) @ [0.299, 0.587, 0.114])
    fg = img < threshold
    return ~fg if invert else fg


def fragments_to_json(frags):
    return [{
        "id": f.id,
        "length": round(f.length, 2),
        "thickness": round(f.thickness, 2),
        "angle_deg": round(f.angle_deg, 2),
        "left": [round(float(v), 1) for v in f.left],        # (row, col)
        "right": [round(float(v), 1) for v in f.right],
        "left_dir": [round(float(v), 4) for v in f.left_dir],
        "right_dir": [round(float(v), 4) for v in f.right_dir],
        "is_confetti": bool(f.is_confetti),
        "label": f.meta.get("label"),            # 'line' | 'noise' | None
        "support": f.meta.get("support", []),    # signed staff-space offsets of supporting lines
        "path": f.path.astype(int).tolist(),
    } for f in frags]


def save_overlay(mask, frags, out_path):
    """Grey = mask, green = fragment centerlines, orange = confetti, red dots = endpoints."""
    from skimage import io
    rgb = np.full(mask.shape + (3,), 255, np.uint8)
    rgb[mask] = (190, 190, 190)
    for f in frags:
        rr, cc = f.path[:, 0].astype(int), f.path[:, 1].astype(int)
        rgb[rr, cc] = (230, 140, 0) if f.is_confetti else (0, 150, 60)
        for r, c in (f.left.astype(int), f.right.astype(int)):
            rgb[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = (220, 0, 0)
    io.imsave(out_path, rgb, check_contrast=False)


DEFAULT_OUT_DIR = __import__("pathlib").Path(__file__).resolve().parent / "outputs"


def main(argv=None):
    import argparse, json, sys
    from pathlib import Path
    p = argparse.ArgumentParser(
        description="Extract staff-line fragments from binary staff layers. "
                    "For each input, writes <name>_fragments.json and "
                    "<name>_overlay.png to the output folder.")
    p.add_argument("images", nargs="*", help="one or more staff-line layers (PNG etc.)")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                   help="output folder (default: outputs/ next to this script)")
    p.add_argument("--no-overlay", action="store_true", help="skip the check PNGs")
    p.add_argument("--staff-space", type=float, default=None,
                   help="line spacing in px (default: estimated per image)")
    p.add_argument("--tol", type=float, default=0.25,
                   help="allowed deviation from k * staff space, as a fraction of it")
    p.add_argument("--invert", action="store_true", help="ink is light on dark background")
    p.add_argument("--threshold", type=int, default=128, help="grayscale ink threshold (non-RGBA input)")
    p.add_argument("--min-area", type=int, default=8)
    p.add_argument("--bridge-px", type=int, default=0)
    p.add_argument("--max-angle", type=float, default=30.0)
    p.add_argument("--demo", action="store_true", help="run the synthetic demo instead")
    a = p.parse_args(argv)

    if a.demo or not a.images:
        if not a.demo:
            p.print_usage(sys.stderr)
            sys.exit(2)
        return _demo()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for image in a.images:
        stem = Path(image).stem
        mask = load_mask(image, invert=a.invert, threshold=a.threshold)
        frags = extract_fragments(mask, min_area=a.min_area, bridge_px=a.bridge_px,
                                  max_angle_deg=a.max_angle)
        n_conf = sum(f.is_confetti for f in frags)
        stroke = np.median([f.thickness for f in frags]) if frags else float("nan")
        space = classify_fragments(frags, staff_space=a.staff_space, tol=a.tol) if frags else float("nan")
        n_line = sum(f.meta.get("label") == "line" for f in frags)
        json_path = out_dir / f"{stem}_fragments.json"
        with open(json_path, "w") as fh:
            json.dump({"image": str(image), "shape": list(mask.shape),
                       "params": {"min_area": a.min_area, "bridge_px": a.bridge_px,
                                  "max_angle": a.max_angle, "invert": a.invert,
                                  "threshold": a.threshold, "tol": a.tol},
                       "staff_space": round(float(space), 2),
                       "stroke": round(float(stroke), 2),
                       "fragments": fragments_to_json(frags)}, fh)
        if not a.no_overlay:
            save_overlay(mask, frags, out_dir / f"{stem}_overlay.png")
            save_classified(mask, frags, out_dir / f"{stem}_classified.png")
        print(f"{image}: {len(frags)} fragments ({n_conf} confetti), "
              f"{n_line} line / {len(frags) - n_line} noise, "
              f"stroke ~{stroke:.1f}px, staff space ~{space:.1f}px -> {out_dir}/{stem}_*",
              file=sys.stderr)


# ---------------------------------------------------------------- demo
def _demo():
    import json
    rng = np.random.default_rng(0)
    H, W, space = 220, 600, 32
    mask = np.zeros((H, W), bool)
    xs = np.arange(W)
    # four slightly slanted, gently curved staff lines, 3 px thick
    for k in range(4):
        y0 = 50 + k * space
        ys = y0 + 0.04 * xs + 4 * np.sin(xs / 140)
        for x, y in zip(xs, ys):
            mask[int(round(y)) - 1:int(round(y)) + 2, x] = True
    # break them into confetti with random gaps
    keep = np.ones(W, bool)
    for _ in range(40):
        s = rng.integers(0, W); keep[s:s + rng.integers(3, 18)] = False
    mask &= keep[None, :]
    # a neume stem crossing two lines (should be rejected / pruned)
    mask[60:120, 300:303] = True
    # speckle noise
    mask[rng.integers(0, H, 60), rng.integers(0, W, 60)] = True

    frags = extract_fragments(mask, min_area=6)
    n_conf = sum(f.is_confetti for f in frags)
    print(f"{len(frags)} fragments ({n_conf} confetti); "
          f"stroke ~{np.median([f.thickness for f in frags]):.1f}px")
    for f in frags[:5]:
        print(json.dumps({"id": f.id, "len": round(f.length, 1),
                          "angle": round(f.angle_deg, 1),
                          "left": f.left.round(1).tolist(),
                          "right": f.right.round(1).tolist(),
                          "thick": round(f.thickness, 1),
                          "confetti": f.is_confetti}))

    # quick linking sanity check: best successor for each non-confetti fragment
    big = [f for f in frags if not f.is_confetti]
    wrong = 0
    for a in big:
        costs = [(link_cost(a, b, space, max_gap=60), b) for b in big if b is not a]
        costs = [c for c in costs if np.isfinite(c[0])]
        if costs:
            _, b = min(costs, key=lambda t: t[0])
            line_a = round((a.right[0] - 0.04 * a.right[1] - 50) / space)
            line_b = round((b.left[0] - 0.04 * b.left[1] - 50) / space)
            wrong += line_a != line_b
    print(f"best-successor links that jump to a different staff line: {wrong}/{len(big)}")


if __name__ == "__main__":
    main()