"""
Link staff-line fragments into complete staff lines, then group lines into staves.

Reads the <name>_fragments.json files written by staff_fragments.py and writes,
to outputs/ next to this script:
    <name>_lines.json   staff lines as polylines, grouped into staves
    <name>_lines.png    overlay: one colour per stave, fitted curves on top

Usage:
    python link_lines.py outputs/page_fragments.json [more.json ...]
    python link_lines.py outputs/*_fragments.json --image page.png   # draw on the page

Pipeline:
  1. take fragments labelled 'line' (all fragments if there are no labels)
  2. short-range linking: join fragment ends that continue each other's
     direction, cheapest joins first, each end used at most once
  3. long-range linking: join the resulting chains across bigger gaps, using
     a direction fitted over several staff spaces of each chain
     and merge chains that overlap in x along the same row
  4. fit a smooth curve per line (binned medians, gaps interpolated)
  5. group lines into staves: lines about one staff space apart, overlapping
     in x, belong together
  6. complete each staff: measure line spacing column by column (vertical
     run lengths in the staff layer), then give every line a curve across the
     staff's full width, predicting positions where a line has no ink from its
     neighbours. Predicted stretches are flagged in the JSON and drawn pale.
"""
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "outputs"


# ------------------------------------------------------------------ loading

class Piece:
    """One fragment, with centerline points as (x, y) = (col, row)."""

    def __init__(self, j):
        p = np.asarray(j["path"], float)
        self.id = j["id"]
        self.xy = p[:, ::-1]                          # (x, y)
        self.xy = self.xy[np.argsort(self.xy[:, 0], kind="stable")]
        self.length = j["length"]
        self.thickness = j["thickness"]
        self.label = j.get("label")


def load_fragments(path):
    d = json.load(open(path))
    pieces = [Piece(j) for j in d["fragments"]]
    if any(p.label is not None for p in pieces):
        pieces = [p for p in pieces if p.label == "line"]
    s = d.get("staff_space")
    if not s or not np.isfinite(s):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from staff_finding import Fragment, estimate_staff_space
        except ImportError:
            from staff_fragments import Fragment, estimate_staff_space
        fr = [Fragment(j["id"], np.asarray(j["path"], float), j["thickness"],
                       j["length"], None, None, None, None, 0.0)
              for j in d["fragments"]]
        s = estimate_staff_space(fr)
    stroke = d.get("stroke") or float(np.median([p.thickness for p in pieces]))
    return d, pieces, float(s), float(stroke)


# ------------------------------------------------------------------ chains

class Chain:
    def __init__(self, pieces):
        self.pieces = list(pieces)
        self._pts = None

    @property
    def pts(self):
        if self._pts is None:
            pts = np.concatenate([p.xy for p in self.pieces])
            self._pts = pts[np.argsort(pts[:, 0], kind="stable")]
        return self._pts

    @property
    def x0(self):
        return self.pts[0, 0]

    @property
    def x1(self):
        return self.pts[-1, 0]

    def end_fit(self, side, span, fallback_slope):
        """(x, y, slope) at one end, from a line fit over `span` px of that end."""
        pts = self.pts
        if side == "right":
            sel = pts[pts[:, 0] >= pts[-1, 0] - span]
            x_end = pts[-1, 0]
        else:
            sel = pts[pts[:, 0] <= pts[0, 0] + span]
            x_end = pts[0, 0]
        if np.ptp(sel[:, 0]) >= 0.25 * span and len(sel) >= 5:
            slope, icpt = np.polyfit(sel[:, 0], sel[:, 1], 1)
            # don't let a short wobbly end produce a wild slope
            slope = float(np.clip(slope, fallback_slope - 0.1, fallback_slope + 0.1))
            y_end = float(np.median(sel[:, 1] - slope * sel[:, 0]) + slope * x_end)
        else:
            slope = fallback_slope
            y_end = float(np.median(sel[:, 1]))
        return float(x_end), y_end, slope


def local_slope_fn(pieces, s):
    """Median slope of the long pieces near (x, y); page median as fallback."""
    longs = [p for p in pieces if np.ptp(p.xy[:, 0]) >= 2 * s]
    data = []
    for p in longs:
        slope = np.polyfit(p.xy[:, 0], p.xy[:, 1], 1)[0]
        data.append((p.xy[:, 0].mean(), p.xy[:, 1].mean(), slope))
    data = np.array(data) if data else np.zeros((0, 3))
    page = float(np.median(data[:, 2])) if len(data) else 0.0

    def f(x, y):
        if not len(data):
            return page
        near = data[(np.abs(data[:, 1] - y) < 3 * s) & (np.abs(data[:, 0] - x) < 8 * s)]
        return float(np.median(near[:, 2])) if len(near) else page
    return f


def link(chains, s, *, max_gap, min_gap, span, lat_tol, slope_at, barriers=()):
    """Greedy end-to-end linking: cheapest valid joins first, each chain end
    used once. Returns the merged chains."""
    ends = []
    for i, c in enumerate(chains):
        yr = float(np.median(c.pts[:, 1]))
        ends.append((c.end_fit("right", span, slope_at(c.x1, yr)),
                     c.end_fit("left", span, slope_at(c.x0, yr))))

    cand = []
    for i, a in enumerate(chains):
        ax, ay, aslope = ends[i][0]
        for j, b in enumerate(chains):
            if i == j:
                continue
            bx, by, bslope = ends[j][1]
            dx = bx - ax
            if dx < min_gap or dx > max_gap:
                continue
            # b must not mostly overlap a horizontally
            if b.x1 <= a.x1:
                continue
            # never bridge a gutter / text-block edge
            if any(min(bx, x1) - max(ax, x0) > 0.5 * (x1 - x0) for x0, x1 in barriers):
                continue
            # predicted y from both sides, averaged
            lat = 0.5 * (abs(by - (ay + aslope * dx)) + abs(ay - (by - bslope * dx)))
            if lat > lat_tol:
                continue
            turn = abs(np.arctan(aslope) - np.arctan(bslope))
            cand.append((max(dx, 0) / s + 3.0 * lat / s + 2.0 * turn, i, j))
    cand.sort()

    nxt, prv = {}, {}
    parent = list(range(len(chains)))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for _, i, j in cand:
        if i in nxt or j in prv or find(i) == find(j):
            continue
        nxt[i], prv[j] = j, i
        parent[find(i)] = find(j)

    merged = []
    for i in range(len(chains)):
        if i in prv:
            continue
        seq, k = [], i
        while True:
            seq.extend(chains[k].pieces)
            if k not in nxt:
                break
            k = nxt[k]
        merged.append(Chain(seq))
    return merged


def find_barriers(pieces, s, min_width=0.75, max_frac=0.08):
    """Vertical bands (x0, x1) with almost no staff-line ink anywhere on the
    page, wider than min_width staff spaces and with ink on both sides:
    gutters, text-block edges. Lines are never joined across one."""
    if not pieces:
        return []
    W = int(max(p.xy[:, 0].max() for p in pieces)) + 2
    occ = np.zeros(W)
    for p in pieces:
        occ[int(p.xy[:, 0].min()):int(p.xy[:, 0].max()) + 1] += 1
    used = occ[occ > 0]
    thresh = max(1.0, max_frac * float(np.median(used))) if len(used) else 1.0
    empty = occ <= thresh
    first, last = np.argmax(occ > thresh), W - 1 - np.argmax(occ[::-1] > thresh)
    bars, x = [], first
    while x <= last:
        if empty[x]:
            x0 = x
            while x <= last and empty[x]:
                x += 1
            if x - x0 >= min_width * s:
                bars.append((float(x0), float(x)))
        x += 1
    return bars


# ------------------------------------------------------------------ fitting

def fit_curve(pts, step):
    """Smooth polyline through a chain: median y per x-bin, gaps interpolated,
    light smoothing. Returns (N, 2) array of (x, y)."""
    x0, x1 = pts[0, 0], pts[-1, 0]
    if x1 - x0 < step:
        return np.array([[x0, np.median(pts[:, 1])], [x1, np.median(pts[:, 1])]])
    edges = np.arange(x0, x1 + step, step)
    centres, ys = [], []
    idx = np.digitize(pts[:, 0], edges)
    for k in np.unique(idx):
        sel = pts[idx == k]
        centres.append(sel[:, 0].mean())
        ys.append(np.median(sel[:, 1]))
    centres, ys = np.array(centres), np.array(ys)
    xs = np.arange(x0, x1 + 1e-6, step)
    if xs[-1] < x1:
        xs = np.append(xs, x1)
    yi = np.interp(xs, centres, ys)
    if len(yi) >= 5:
        k = np.ones(5) / 5
        pad = np.pad(yi, 2, mode="edge")
        yi = np.convolve(pad, k, mode="valid")
    return np.column_stack([xs, yi])


def y_on(curve, xs):
    return np.interp(xs, curve[:, 0], curve[:, 1])


# ------------------------------------------------------------------ staves

def group_staves(lines, s, lo=0.6, hi=1.45, min_overlap=None):
    """Connected components of 'lines about one staff space apart and
    overlapping in x'. Returns list of staves (lists of line indices,
    top to bottom), and the leftover single lines."""
    n = len(lines)
    min_overlap = min_overlap or 2 * s
    adj = [[] for _ in range(n)]
    for i in range(n):
        ci = lines[i]["curve"]
        for j in range(i + 1, n):
            cj = lines[j]["curve"]
            a, b = max(ci[0, 0], cj[0, 0]), min(ci[-1, 0], cj[-1, 0])
            if b - a < min(min_overlap, 0.5 * min(np.ptp(ci[:, 0]), np.ptp(cj[:, 0]))):
                continue
            xs = np.linspace(a, b, 15)
            dy = abs(float(np.median(y_on(cj, xs) - y_on(ci, xs))))
            if lo * s <= dy <= hi * s:
                adj[i].append(j)
                adj[j].append(i)
    seen, groups = set(), []
    for i in range(n):
        if i in seen:
            continue
        stack, comp = [i], []
        while stack:
            k = stack.pop()
            if k in seen:
                continue
            seen.add(k)
            comp.append(k)
            stack.extend(adj[k])
        groups.append(comp)
    staves = [g for g in groups if len(g) >= 2]
    singles = [g[0] for g in groups if len(g) == 1]
    for g in staves:
        g.sort(key=lambda k: float(np.median(lines[k]["curve"][:, 1])))
    staves.sort(key=lambda g: (float(np.median(lines[g[0]]["curve"][:, 1])),
                               float(lines[g[0]]["curve"][0, 0])))
    return staves, singles


def merge_overlapping(chains, s, lat_tol, barriers=()):
    """Merge chains that run along the same row while overlapping in x
    (end-to-end linking can't join those)."""
    changed = True
    while changed:
        changed = False
        curves = [fit_curve(c.pts, step=s / 2) for c in chains]
        for i in range(len(chains)):
            for j in range(i + 1, len(chains)):
                a, b = max(chains[i].x0, chains[j].x0), min(chains[i].x1, chains[j].x1)
                if b - a <= 0:
                    continue
                if any(x0 < b and x1 > a for x0, x1 in barriers):
                    continue
                xs = np.linspace(a, b, 9)
                dy = float(np.median(np.abs(y_on(curves[i], xs) - y_on(curves[j], xs))))
                # pieces of one line interleave; two parallel tracings both
                # have ink in the same columns -> leave those to clean_staves
                bi = set(np.floor(chains[i].pts[(chains[i].pts[:, 0] >= a) & (chains[i].pts[:, 0] <= b), 0] / (s / 4)).astype(int))
                bj = set(np.floor(chains[j].pts[(chains[j].pts[:, 0] >= a) & (chains[j].pts[:, 0] <= b), 0] / (s / 4)).astype(int))
                shared = len(bi & bj) / max(1, min(len(bi), len(bj)))
                # ...except tracks only a few px apart: one thick line split in two
                if dy <= 0.12 * s or (dy <= lat_tol and shared < 0.3):
                    chains[i] = Chain(chains[i].pieces + chains[j].pieces)
                    del chains[j]
                    changed = True
                    break
            if changed:
                break
    return chains


def _y_ref(line, x):
    """Line's y at x, extrapolated with a straight-line fit (for ordering
    lines of different extents)."""
    c = line["curve"]
    if c[0, 0] <= x <= c[-1, 0]:
        return float(y_on(c, [x])[0])
    a, b = np.polyfit(c[:, 0], c[:, 1], 1)
    return float(a * x + b)


def _pair_dy(la, lb):
    """Median vertical distance lb - la over their x-overlap (extrapolated
    straight-line estimate if they don't overlap)."""
    a, b = max(la["curve"][0, 0], lb["curve"][0, 0]), min(la["curve"][-1, 0], lb["curve"][-1, 0])
    if b - a > 0:
        xs = np.linspace(a, b, 15)
        return float(np.median(y_on(lb["curve"], xs) - y_on(la["curve"], xs)))
    x = 0.5 * (a + b)
    return _y_ref(lb, x) - _y_ref(la, x)


def _merge_line_dicts(la, lb, s):
    pts = np.concatenate([la["pts"], lb["pts"]])
    pts = pts[np.argsort(pts[:, 0], kind="stable")]
    span = pts[-1, 0] - pts[0, 0] + 1
    bins = np.unique(np.floor((pts[:, 0] - pts[0, 0]) / (s / 2)))
    return {"curve": fit_curve(pts, step=s / 2), "pts": pts,
            "fragment_ids": sorted(la["fragment_ids"] + lb["fragment_ids"]),
            "coverage": float(min(1.0, len(bins) * (s / 2) / span))}


def _ink(l):
    return np.ptp(l["curve"][:, 0]) * l["coverage"]


def clean_staves(lines, staves, s, merge_below=0.4, drop_below=0.7, split_above=1.3):
    """Check each staff's internal spacing and repair it.

    With m = the staff's typical line spacing, consecutive lines that are
      < merge_below * m apart are the same line traced twice -> merged;
      < drop_below  * m apart: one is a stray -> the one with less ink leaves;
      > split_above * m apart: a gap between two staves -> split there.
    Returns (staves as lists of line dicts, stray line dicts).
    """
    out, strays = [], []
    for g in staves:
        group = [lines[k] for k in g]
        changed = True
        while changed and len(group) >= 2:
            changed = False
            xref = float(np.median(np.concatenate([l["curve"][:, 0] for l in group])))
            group.sort(key=lambda l: _y_ref(l, xref))
            d = np.array([_pair_dy(a, b) for a, b in zip(group[:-1], group[1:])])
            ok = d[(d >= 0.6 * s) & (d <= 1.45 * s)]
            m = float(np.median(ok)) if len(ok) else s
            for k, dk in enumerate(d):
                if dk < merge_below * m:
                    a, b = group[k], group[k + 1]
                    # same line only if they don't both have ink in the same place
                    lo, hi = max(a["pts"][0, 0], b["pts"][0, 0]), min(a["pts"][-1, 0], b["pts"][-1, 0])
                    if hi - lo < 2 * s:
                        group[k] = _merge_line_dicts(a, b, s)
                        del group[k + 1]
                        changed = True
                        break
                if dk < drop_below * m:
                    k_out = k if _ink(group[k]) < _ink(group[k + 1]) else k + 1
                    strays.append(group.pop(k_out))
                    changed = True
                    break
        if len(group) < 2:
            strays.extend(group)
            continue
        # split at between-staff gaps
        xref = float(np.median(np.concatenate([l["curve"][:, 0] for l in group])))
        d = np.array([_pair_dy(a, b) for a, b in zip(group[:-1], group[1:])])
        ok = d[(d >= 0.6 * s) & (d <= 1.45 * s)]
        m = float(np.median(ok)) if len(ok) else s
        cur = [group[0]]
        for l, dk in zip(group[1:], d):
            if dk > split_above * m:
                if len(cur) >= 2:
                    out.append(cur)
                else:
                    strays.extend(cur)
                cur = [l]
            else:
                cur.append(l)
        if len(cur) >= 2:
            out.append(cur)
        else:
            strays.extend(cur)
    return out, strays


# ------------------------------------------------------------------ staff layer

def load_layer(path, invert=False, threshold=128):
    """Staff-line layer as a boolean mask (True = ink). Same rules as
    staff_fragments.load_mask: alpha > 0 for transparent PNGs, else dark pixels."""
    from skimage import io
    img = io.imread(path)
    if img.ndim == 3 and img.shape[2] == 4 and (img[..., 3] < 255).any():
        return img[..., 3] > 0
    if img.ndim == 3:
        img = img[..., :3].astype(float) @ [0.299, 0.587, 0.114]
    fg = img < threshold
    return ~fg if invert else fg


def column_spacings(mask, xs, ylo, yhi, s, lo=0.6, hi=1.4):
    """Vertical run-length measurement of line spacing, column by column.

    In each column, find the ink runs between ylo and yhi; the distance between
    the centres of consecutive runs is one line spacing when it falls within
    [lo, hi] * s. Doubled gaps (a missing line) and specks fall outside and are
    ignored. Returns (x, spacing) samples.
    """
    H, W = mask.shape
    out = []
    for x, a, b in zip(xs, ylo, yhi):
        xi = int(round(x))
        if not 0 <= xi < W:
            continue
        a, b = max(0, int(a)), min(H, int(b) + 1)
        col = mask[a:b, xi].astype(np.int8)
        if col.sum() == 0:
            continue
        d = np.diff(np.concatenate([[0], col, [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        centres = (starts + ends - 1) / 2.0
        gaps = np.diff(centres)
        for g in gaps[(gaps >= lo * s) & (gaps <= hi * s)]:
            out.append((x, float(g)))
    return np.array(out) if out else np.zeros((0, 2))


def complete_staff(lines, idx, s, mask=None, step=None, pool=2.0):
    """Give every line in one staff a curve over the staff's full width.

    Where a line has ink, its own fitted curve is kept. Where it doesn't, its
    position is predicted from the staff's other lines at that x, using the
    local spacing (from column run lengths in the staff layer when available,
    otherwise from the lines' own curves). Stretches with no ink on any line are
    interpolated along the line. Adds 'full' (N x 2) and 'predicted' (N bools)
    to each line dict, and returns the staff's x-range and median spacing.
    """
    step = step or s / 2
    L = [lines[k] for k in idx]
    x0s = sorted(l["curve"][0, 0] for l in L)
    x1s = sorted(l["curve"][-1, 0] for l in L)
    # staff extent: where at least two lines agree it starts / ends
    sx0 = x0s[1] if len(x0s) > 1 else x0s[0]
    sx1 = x1s[-2] if len(x1s) > 1 else x1s[-1]
    # compute over the union so no line's own ink is cut off; trimmed below
    ux0, ux1 = x0s[0], x1s[-1]
    grid = np.arange(ux0, ux1 + 1e-6, step)
    n = len(L)

    # Y[i, g] = line i's own y where it has ink near grid column g, else NaN
    Y = np.full((n, len(grid)), np.nan)
    for i, l in enumerate(L):
        px = l["pts"][:, 0]
        has = np.zeros(len(grid), bool)
        bins = np.clip(np.round((px - ux0) / step).astype(int), 0, len(grid) - 1)
        has[np.unique(bins)] = True
        own = y_on(l["curve"], grid)
        Y[i, has] = own[has]

    # local spacing per grid column
    spacing = np.full(len(grid), np.nan)
    samples = np.zeros((0, 2))
    if mask is not None:
        top = np.array([y_on(L[0]["curve"], [x])[0] for x in grid])
        bot = np.array([y_on(L[-1]["curve"], [x])[0] for x in grid])
        cols = np.arange(int(ux0), int(ux1) + 1, max(1, int(s / 8)))
        samples = column_spacings(mask, cols,
                                  np.interp(cols, grid, top) - 0.5 * s,
                                  np.interp(cols, grid, bot) + 0.5 * s, s)
    if len(samples) == 0:
        # fall back to the lines' own curves where neighbours both have ink
        sm = []
        for g in range(len(grid)):
            have = np.flatnonzero(~np.isnan(Y[:, g]))
            for a, b in zip(have[:-1], have[1:]):
                sm.append((grid[g], (Y[b, g] - Y[a, g]) / (b - a)))
        samples = np.array(sm) if sm else np.zeros((0, 2))
    staff_med = float(np.median(samples[:, 1])) if len(samples) else s
    for g, x in enumerate(grid):
        near = samples[np.abs(samples[:, 0] - x) <= pool * s, 1] if len(samples) else []
        spacing[g] = np.median(near) if len(near) >= 3 else staff_med

    # P[i, g]: where line i "should" be at column g, from its nearest
    # neighbours (at most 2 lines away) that have ink there
    P = np.full_like(Y, np.nan)
    for g in range(len(grid)):
        have = np.flatnonzero(~np.isnan(Y[:, g]))
        for i in range(n):
            others = have[(have != i) & (np.abs(have - i) <= 2)]
            if not len(others):
                continue
            w = 1.0 / np.abs(others - i)
            P[i, g] = np.sum(w * (Y[others, g] + (i - others) * spacing[g])) / np.sum(w)

    full = Y.copy()
    for i in range(n):
        own = ~np.isnan(Y[i])
        both = own & ~np.isnan(P[i])
        gap = ~own & ~np.isnan(P[i])
        if both.any() and gap.any():
            # the line's own offset from the prediction, carried across gaps so
            # fills sit flush with the ink; capped so one bad column can't throw it
            r = np.clip((Y[i] - P[i])[both], -0.3 * staff_med, 0.3 * staff_med)
            resid = np.interp(grid, grid[both], r)
            full[i, gap] = P[i, gap] + resid[gap]
        elif gap.any():
            full[i, gap] = P[i, gap]
        ok = ~np.isnan(full[i])
        if ok.any():
            full[i, ~ok] = np.interp(grid[~ok], grid[ok], full[i, ok])

    pred_mask = np.isnan(Y)
    # smooth predicted stretches (they jump when the set of inked neighbours changes)
    win = max(3, int(round(2 * s / step)) | 1)
    for i in range(n):
        if not pred_mask[i].any() or len(grid) < win:
            continue
        y = full[i]
        pad = np.pad(y, win // 2, mode="edge")
        med = np.array([np.median(pad[k:k + win]) for k in range(len(y))])
        sm = np.convolve(np.pad(med, win // 2, mode="edge"), np.ones(win) / win, mode="valid")
        full[i] = np.where(pred_mask[i], sm, y)
    # predicted points may never cross or crowd a neighbour
    gapmin = 0.6 * spacing
    for i in range(1, n):
        p = pred_mask[i]
        full[i, p] = np.maximum(full[i, p], full[i - 1, p] + gapmin[p])
    for i in range(n - 2, -1, -1):
        p = pred_mask[i]
        full[i, p] = np.minimum(full[i, p], full[i + 1, p] - gapmin[p])

    for i, l in enumerate(L):
        pred = pred_mask[i]
        y = full[i]
        # each line spans the staff extent, or its own ink if that reaches further
        lo = min(sx0, l["curve"][0, 0]) - 1e-6
        hi = max(sx1, l["curve"][-1, 0]) + 1e-6
        keep = (grid >= lo) & (grid <= hi)
        l["full"] = np.column_stack([grid[keep], y[keep]])
        l["predicted"] = pred[keep]
    return (float(sx0), float(sx1)), staff_med


def trim_staff(lines, staff, n_keep):
    """Keep the n_keep lines with the most ink (span x coverage); the rest
    leave the staff. Returns (kept, dropped)."""
    if not n_keep or len(staff) <= n_keep:
        return staff, []
    ink = {k: np.ptp(lines[k]["curve"][:, 0]) * lines[k]["coverage"] for k in staff}
    keep = sorted(staff, key=lambda k: -ink[k])[:n_keep]
    keep.sort(key=lambda k: float(np.median(lines[k]["curve"][:, 1])))
    return keep, [k for k in staff if k not in keep]


# ------------------------------------------------------------------ main pass

def build_lines(pieces, s, stroke, *, short_gap=3.0, long_gap=8.0, lat=0.25,
                min_line=2.0, mask=None, lines_per_staff=None, complete=True):
    slope_at = local_slope_fn(pieces, s)
    barriers = find_barriers(pieces, s)
    chains = [Chain([p]) for p in pieces]
    # 1) short range: fragment to fragment, direction from each piece's own end
    chains = link(chains, s, max_gap=short_gap * s, min_gap=-2 * stroke,
                  span=1.5 * s, lat_tol=lat * s, slope_at=slope_at, barriers=barriers)
    # 2) long range: chain to chain, direction fitted over a longer stretch.
    #    Run twice so chains grown in the first round can bridge again.
    for _ in range(2):
        chains = link(chains, s, max_gap=long_gap * s, min_gap=-0.5 * s,
                      span=4 * s, lat_tol=lat * s, slope_at=slope_at, barriers=barriers)
        # 3) chains lying along the same row but overlapping in x -> one line
        chains = merge_overlapping(chains, s, lat * s, barriers)

    lines, leftovers = [], []
    for c in chains:
        if c.x1 - c.x0 < min_line * s:
            leftovers.extend(p.id for p in c.pieces)
            continue
        curve = fit_curve(c.pts, step=s / 2)
        covered = sum(np.ptp(p.xy[:, 0]) + 1 for p in c.pieces)
        lines.append({"curve": curve, "pts": c.pts,
                      "fragment_ids": sorted(p.id for p in c.pieces),
                      "coverage": float(min(1.0, covered / (c.x1 - c.x0 + 1)))})
    staves, singles = group_staves(lines, s)
    # check/repair each staff's internal spacing, then re-index all lines
    staff_groups, strays = clean_staves(lines, staves, s)
    single_lines = [lines[k] for k in singles] + strays
    lines = [l for g in staff_groups for l in g] + single_lines
    staves, k = [], 0
    for g in staff_groups:
        staves.append(list(range(k, k + len(g))))
        k += len(g)
    singles = list(range(k, len(lines)))
    if lines_per_staff:
        trimmed = []
        for g in staves:
            keep, drop = trim_staff(lines, g, lines_per_staff)
            trimmed.append(keep)
            singles.extend(drop)
        staves = trimmed
    staff_info = []
    for g in staves:
        if complete:
            xr, med = complete_staff(lines, g, s, mask)
        else:
            xr, med = (float(min(lines[k]["curve"][0, 0] for k in g)),
                       float(max(lines[k]["curve"][-1, 0] for k in g))), s
        staff_info.append({"x_range": xr, "spacing": med})
    return lines, staves, singles, leftovers, barriers, staff_info


def _ranges(xs, flags):
    """[[x0, x1], ...] for runs of True in flags."""
    out, start = [], None
    for x, f in zip(xs, flags):
        if f and start is None:
            start = x
        if not f and start is not None:
            out.append([round(float(start), 1), round(float(prev), 1)])
            start = None
        prev = x
    if start is not None:
        out.append([round(float(start), 1), round(float(prev), 1)])
    return out


def to_json(meta, lines, staves, singles, leftovers, s, staff_info=None):
    out_lines = []
    stave_of = {k: si for si, g in enumerate(staves) for k in g}
    pos_in = {k: pi for g in staves for pi, k in enumerate(g)}
    for k, L in enumerate(lines):
        c = L["full"] if k in stave_of and "full" in L else L["curve"]
        pred = L["predicted"] if k in stave_of and "predicted" in L else np.zeros(len(c), bool)
        out_lines.append({
            "id": k,
            "staff": stave_of.get(k),              # None = not part of any stave
            "line_in_staff": pos_in.get(k),        # 0 = top line
            "x_range": [round(float(c[0, 0]), 1), round(float(c[-1, 0]), 1)],
            "coverage": round(L["coverage"], 3),   # share of the span covered by ink
            "points": np.round(c, 1).tolist(),     # [[x, y], ...] image coords
            "predicted": [bool(v) for v in pred],  # per point: True = no ink, inferred
            "predicted_ranges": _ranges(c[:, 0], pred),
            "fragment_ids": L["fragment_ids"],
        })
    return {
        "image": meta.get("image"),
        "shape": meta.get("shape"),
        "staff_space": round(s, 2),
        "coords": "points are [x, y] = [column, row] in image pixels",
        "staves": [{"id": si, "line_ids": g, "n_lines": len(g),
                    **({"x_range": [round(v, 1) for v in staff_info[si]["x_range"]],
                        "spacing": round(staff_info[si]["spacing"], 2)}
                       if staff_info else {})}
                   for si, g in enumerate(staves)],
        "lines": out_lines,
        "unstaved_line_ids": singles,
        "unlinked_fragment_ids": leftovers,
    }


PALETTE = [(0, 114, 178), (213, 94, 0), (0, 158, 115), (204, 121, 167),
           (230, 159, 0), (86, 180, 233), (120, 80, 160), (160, 120, 40)]


def draw(result, pieces, shape, out_path, background=None):
    from PIL import Image, ImageDraw
    H, W = shape
    if background is not None:
        img = Image.open(background).convert("L").convert("RGB")
        if img.size != (W, H):
            img = img.resize((W, H))
        img = Image.blend(img, Image.new("RGB", (W, H), "white"), 0.55)
    else:
        img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    s = result["staff_space"]
    w = max(3, int(round(s / 10)))
    for p in pieces:
        d.line([tuple(q) for q in p.xy[:: max(1, len(p.xy) // 50)]] if len(p.xy) > 1
               else [tuple(p.xy[0]), tuple(p.xy[0])], fill=(185, 185, 185), width=w)
    for L in result["lines"]:
        col = PALETTE[L["staff"] % len(PALETTE)] if L["staff"] is not None else (0, 0, 0)
        pale = tuple(int(c + 0.6 * (255 - c)) for c in col)
        pts = [tuple(q) for q in L["points"]]
        pred = L.get("predicted") or [False] * len(pts)
        lw = max(2, w // 2 + 1)
        for (p, q), pr in zip(zip(pts[:-1], pts[1:]), pred[1:]):
            d.line([p, q], fill=pale if pr else col, width=lw)
        for (x, y) in (pts[0], pts[-1]):
            r = w
            d.ellipse([x - r, y - r, x + r, y + r], outline=col, width=2)
    for st in result["staves"]:
        first = result["lines"][st["line_ids"][0]]
        x, y = first["points"][0]
        d.text((max(0, x - 6 * w), y - 3 * w), f"S{st['id']}", fill=(0, 0, 0))
    img.save(out_path)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Link fragments into staff lines and staves.")
    p.add_argument("fragments", nargs="+", help="*_fragments.json from staff_fragments.py")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--image", help="draw the overlay on this image instead of white "
                                   "(only with a single input)")
    p.add_argument("--short-gap", type=float, default=3.0,
                   help="max gap between fragments, in staff spaces (default 3)")
    p.add_argument("--long-gap", type=float, default=8.0,
                   help="max gap between chains, in staff spaces (default 8)")
    p.add_argument("--lat", type=float, default=0.25,
                   help="max sideways miss when joining, in staff spaces (default 0.25)")
    p.add_argument("--min-line", type=float, default=2.0,
                   help="drop chains shorter than this many staff spaces (default 2)")
    p.add_argument("--staff-space", type=float, help="override the value in the JSON")
    p.add_argument("--layer", help="staff-line layer image, for column spacing "
                                   "(default: the image path stored in the JSON, if it exists)")
    p.add_argument("--invert", action="store_true", help="layer ink is light on dark")
    p.add_argument("--lines-per-staff", type=int,
                   help="keep only the N best-inked lines per staff (e.g. 4)")
    p.add_argument("--no-complete", action="store_true",
                   help="don't extend/fill lines across the staff width")
    a = p.parse_args(argv)
    if (a.image or a.layer) and len(a.fragments) > 1:
        p.error("--image / --layer need exactly one input")

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in a.fragments:
        meta, pieces, s, stroke = load_fragments(path)
        if a.staff_space:
            s = a.staff_space
        layer = a.layer or meta.get("image")
        mask = None
        if layer and Path(layer).exists() and not a.no_complete:
            mask = load_layer(layer, invert=a.invert)
            if list(mask.shape) != list(meta["shape"]):
                print(f"  {layer}: size {mask.shape} != {meta['shape']}, "
                      f"not using it", file=sys.stderr)
                mask = None
        lines, staves, singles, leftovers, barriers, staff_info = build_lines(
            pieces, s, stroke, short_gap=a.short_gap, long_gap=a.long_gap,
            lat=a.lat, min_line=a.min_line, mask=mask,
            lines_per_staff=a.lines_per_staff, complete=not a.no_complete)
        result = to_json(meta, lines, staves, singles, leftovers, s, staff_info)
        result["barriers_x"] = [[round(a, 1), round(b, 1)] for a, b in barriers]
        stem = Path(path).stem.removesuffix("_fragments")
        with open(out_dir / f"{stem}_lines.json", "w") as fh:
            json.dump(result, fh)
        draw(result, pieces, meta["shape"], out_dir / f"{stem}_lines.png", a.image)
        sizes = [st["n_lines"] for st in result["staves"]]
        print(f"{path}: {len(pieces)} pieces -> {len(lines)} lines, "
              f"{len(staves)} staves (lines per staff: {sizes}), "
              f"{len(singles)} unstaved lines, {len(leftovers)} unlinked pieces, "
              f"{len(barriers)} gutter(s), spacing from "
              f"{'layer columns' if mask is not None else 'line curves'} "
              f"-> {out_dir}/{stem}_lines.*", file=sys.stderr)


if __name__ == "__main__":
    main()