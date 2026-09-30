"""Frame -> columns -> straightened strips -> sheet masks -> counts. ONNX + OpenCV + NumPy only.

The four stages, each usable on its own (see the README for a worked example):
  1. SheetCounter.segment_columns(frame, typ)      column instances + complete/cropped per column
  2. SheetCounter.straighten_columns(frame, polys) one upright strip per complete column
  3. SheetCounter.edge_input(strip) -> sheet_probability(x) > SHEET_THR
                                                   sheet foreground (the two edge models averaged)
  4. SheetCounter.count_sheets(fg)                 the count; sheet_masks() grows one mask per counted sheet
SheetCounter.process(frame, typ) runs all four plus the self-check (c.extra["uncertain"]).

Per-type rules:
  type2, type3  complete/cropped from the column model's 4th output channel
  type10        complete/cropped from geometry: a column reaching the top or bottom of the whole
                stack is cropped (more reliable than the model on unseen type10 frames)
Other column types run with the type3 settings; they are untested (no labels exist).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import onnxruntime as ort

from .flows import follow_flows
from .straighten import (CUT_DIS, CUT_DIS_EX, CUT_DIS_EX2, POSUN_MASKY, _rotate, _straight_angle, _x_limits,
                         to_native_canvas)

MEAN, STD = 0.45, 0.23
COLUMN_SIZE = 256                 # column model input (letterboxed)
COLUMN_MIN_AREA = 25              # at 256
SHEET_THR = 0.50                  # sheet foreground threshold
BAND_FRAC = 0.50                  # count in the centre 50 % of the column width
SHEET_MIN_AREA = 6
REL_FLOOR = 0.25                  # ignore band blobs < 25 % of the crop's median blob
END_FLOOR = 0.50                  # first/last blob must reach 50 % of the median (drops end slivers)
EDGE_INPUT_W, EDGE_PAD = 74, 6    # centre 74 px of the column width, 6 px pad
GEO_TOL = 6                       # type10 geometric good/cropped tolerance (px)


def _session(path: str) -> ort.InferenceSession:
    prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    return ort.InferenceSession(path, providers=prov)


def _to_input(img_bgr: np.ndarray) -> np.ndarray:
    x = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def letterbox(img, size, interp=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    px, py = (size - nw) // 2, (size - nh) // 2
    out = np.zeros((size, size) + img.shape[2:], img.dtype)
    out[py:py + nh, px:px + nw] = r
    return out, s, px, py


def _unletter(a, s, px, py, shape, interp):
    H, W = shape
    h, w = int(round(H * s)), int(round(W * s))
    return cv2.resize(a[py:py + h, px:px + w].astype(np.float32), (W, H), interpolation=interp)


@dataclass
class Column:
    mask: np.ndarray                  # bool, frame resolution
    polygon: np.ndarray               # int32 (N, 2), frame coordinates
    cropped: bool
    p_cropped: float                  # the class head's mean P(cropped) (informational)
    count: int | None = None
    crop: np.ndarray | None = None    # straightened column (native width, 2050 canvas)
    sheet_masks: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


class SheetCounter:
    def __init__(self, models_dir: str):
        self.col = _session(os.path.join(models_dir, "column_flow.onnx"))
        edge_files = sorted(f for f in os.listdir(models_dir) if f.startswith("sheet_edges") and f.endswith(".onnx"))
        if not edge_files:
            raise FileNotFoundError(f"no sheet_edges*.onnx in {models_dir}")
        self.edges = [_session(os.path.join(models_dir, f)) for f in edge_files]

    # ---- 1. columns -------------------------------------------------------------------
    def segment_columns(self, frame: np.ndarray, typ: str = "type3") -> list[Column]:
        H, W = frame.shape[:2]
        imgL, s, px, py = letterbox(frame, COLUMN_SIZE)
        o = self.col.run(None, {"input": _to_input(imgL)})[0][0]
        prob = _sigmoid(o[0])
        lab = follow_flows(prob > 0.5, o[1:3])
        pc = _sigmoid(o[3]) if o.shape[0] >= 4 else None
        labN = _unletter(lab, s, px, py, (H, W), cv2.INTER_NEAREST).astype(np.int32)
        pcN = _unletter(pc, s, px, py, (H, W), cv2.INTER_LINEAR) if pc is not None else None
        ids = [i for i in range(1, int(lab.max()) + 1) if (lab == i).sum() >= COLUMN_MIN_AREA]
        cols = []
        for i in ids:
            m = labN == i
            if not m.any():
                continue
            p = float(pcN[m].mean()) if pcN is not None else 0.0
            cols.append(Column(mask=m, polygon=_polygon(m), cropped=p > 0.5, p_cropped=p))
        if typ == "type10" and cols:               # geometric complete/cropped
            yb = [np.where(c.mask.any(axis=1))[0] for c in cols]
            top, bot = min(y.min() for y in yb), max(y.max() for y in yb)
            for c, y in zip(cols, yb):
                c.cropped = bool(y.min() <= top + GEO_TOL or y.max() >= bot - GEO_TOL)
        cols.sort(key=lambda c: float(np.where(c.mask)[1].mean()))       # left to right
        return cols

    # ---- 2. straighten ----------------------------------------------------------------
    @staticmethod
    def straighten_columns(frame: np.ndarray, polys: list[np.ndarray]) -> list[np.ndarray | None]:
        """Production straightener (TosaI66 BAD_ELIMINATE geometry): one rotation angle per frame
        from the GOOD columns, each column cut out, background blacked, rotated 180 deg, placed on
        a 2050-px canvas at native width. None where a column cannot be cut."""
        if not polys:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        straight = any(p[:, 0].min() <= 1 and p[:, 0].max() >= W - 1 for p in polys)
        ang = _straight_angle([cv2.minAreaRect(p) for p in polys], W, straight)
        rot = _rotate(frame, ang)
        RW = rot.shape[1]
        out = []
        for p in polys:
            single = np.zeros_like(gray); cv2.fillPoly(single, [p], 255)
            outline = np.zeros_like(gray); cv2.drawContours(outline, [p], -1, 255, 1)
            rm, ro = _rotate(single, ang), _rotate(outline, ang)
            lmin1, lmax1 = _x_limits(POSUN_MASKY, rm, ro)
            lmin2, _ = _x_limits(-POSUN_MASKY, rm, ro)
            if lmin1[0] == 0 and lmin2[0] == 0:
                out.append(None); continue
            a, b = sorted((lmin2[1], lmax1[1]))
            d = b - a
            cut = CUT_DIS if d > 220 else CUT_DIS_EX if d > 120 else CUT_DIS_EX2
            x0, x1 = int(np.clip(a - cut, 0, RW - 1)), int(np.clip(b + cut, 0, RW))
            if x1 <= x0:
                out.append(None); continue
            crop = rot[:, x0:x1].copy()
            crop[rm[:, x0:x1] == 0] = 0
            out.append(to_native_canvas(cv2.rotate(crop, cv2.ROTATE_180)))
        return out

    # ---- 3. sheets --------------------------------------------------------------------
    @staticmethod
    def edge_input(crop: np.ndarray) -> np.ndarray:
        """Tight crop (content + 6 px) -> centre 74 px of the width (+ pads). The exact model input."""
        ys, xs = np.where(crop.max(axis=2) > 0)
        if ys.size == 0:
            return crop
        y0, y1 = max(0, ys.min() - EDGE_PAD), min(crop.shape[0], ys.max() + 1 + EDGE_PAD)
        x0, x1 = max(0, xs.min() - EDGE_PAD), min(crop.shape[1], xs.max() + 1 + EDGE_PAD)
        img = crop[y0:y1, x0:x1]
        target = EDGE_INPUT_W + 2 * EDGE_PAD
        if img.shape[1] > target:
            a = (img.shape[1] - target) // 2
            img = img[:, a:a + target]
        return img

    def sheet_probabilities(self, x: np.ndarray) -> list[np.ndarray]:
        """Sheet-foreground probability of each edge model separately."""
        h, w = x.shape[:2]
        ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
        xp = cv2.copyMakeBorder(x, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
        inp = _to_input(xp)
        return [_sigmoid(s.run(None, {"input": inp})[0][0, 0, :h, :w]) for s in self.edges]

    def sheet_probability(self, x: np.ndarray) -> np.ndarray:
        """Mean sheet-foreground probability over the edge models (averaging removes seed noise)."""
        return np.mean(self.sheet_probabilities(x), axis=0)

    # ---- 4. count ---------------------------------------------------------------------
    @staticmethod
    def count_sheets(fg: np.ndarray) -> tuple[int, np.ndarray, tuple[int, int]]:
        """(count, band seed labels 1..N top-down, band x-range). Mask count == count by construction."""
        a, b = _band(fg)
        seeds = np.zeros(fg.shape, np.int32)
        sub = fg[:, a:b].astype(np.uint8)
        if sub.size == 0:
            return 0, seeds, (a, b)
        n, cc, st, cen = cv2.connectedComponentsWithStats(sub, connectivity=8)
        if n <= 1:
            return 0, seeds, (a, b)
        areas = st[1:, cv2.CC_STAT_AREA]
        keep = areas >= SHEET_MIN_AREA
        if keep.any():
            keep &= areas >= REL_FLOOR * np.median(areas[keep])
        if END_FLOOR and keep.sum() > 2:
            med = np.median(areas[keep])
            idx = np.where(keep)[0]
            idx = idx[np.argsort(cen[1:, 1][idx])]
            for i in (idx[0], idx[-1]):
                if areas[i] < END_FLOOR * med:
                    keep[i] = False
        order = sorted(np.where(keep)[0], key=lambda i: cen[i + 1, 1])
        for new_id, i in enumerate(order, 1):
            seeds[:, a:b][cc == i + 1] = new_id
        return len(order), seeds, (a, b)

    @staticmethod
    def sheet_masks(fg: np.ndarray, seeds: np.ndarray, ab: tuple[int, int]) -> np.ndarray:
        """Grow each counted band seed outward one pixel column at a time (edges/grow.py): no
        sheet can take a neighbour's id, and the number of masks equals the count."""
        lab = seeds.copy()
        a, b = ab
        _propagate(lab, fg, range(b, fg.shape[1]), +1)
        _propagate(lab, fg, range(a - 1, -1, -1), -1)
        return lab

    # ---- everything -------------------------------------------------------------------
    def process(self, frame: np.ndarray, typ: str = "type3", masks: bool = True, check: bool = True) -> list[Column]:
        """check=True adds a self-check per complete column (c.extra["uncertain"], ["why"]): the
        count is recomputed by each edge model alone and from the outline grown by 1 px; any
        disagreement marks the column uncertain -> the user should take another photo."""
        cols = self.segment_columns(frame, typ)
        goods = [c for c in cols if not c.cropped]
        crops = self.straighten_columns(frame, [c.polygon for c in goods])
        grown = []
        if check and goods:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            grown = self.straighten_columns(frame, [_polygon(cv2.dilate(c.mask.astype(np.uint8), k) > 0) for c in goods])
        for gi, (c, crop) in enumerate(zip(goods, crops)):
            c.crop = crop
            if crop is None:
                c.extra.update(uncertain=True, why="could not straighten")
                continue
            x = self.edge_input(crop)
            probs = self.sheet_probabilities(x)
            fg = np.mean(probs, axis=0) > SHEET_THR
            c.count, seeds, ab = self.count_sheets(fg)
            c.extra["edge_input"] = x
            if check:
                why = []
                singles = [self.count_sheets(p > SHEET_THR)[0] for p in probs] if len(probs) > 1 else []
                if any(n != c.count for n in singles):
                    why.append(f"edge models disagree {singles}")
                if gi < len(grown) and grown[gi] is not None:
                    ng = self.count_sheets(self.sheet_probability(self.edge_input(grown[gi])) > SHEET_THR)[0]
                    if ng != c.count:
                        why.append(f"column end ambiguous ({ng} with a 1 px wider outline)")
                c.extra.update(uncertain=bool(why), why="; ".join(why))
            if masks:
                c.sheet_masks = self.sheet_masks(fg, seeds, ab)
        return cols


def _polygon(m: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    return cv2.approxPolyDP(c, 1.0, True)[:, 0, :].astype(np.int32)


def _band(fg: np.ndarray, frac: float = BAND_FRAC) -> tuple[int, int]:
    xs = np.where(fg.any(axis=0))[0]
    if xs.size == 0:
        return 0, fg.shape[1]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cx, half = 0.5 * (x0 + x1), 0.5 * (x1 - x0) * frac
    return max(x0, int(round(cx - half))), min(x1, int(round(cx + half)))


def _runs(col: np.ndarray):
    idx = np.flatnonzero(np.diff(np.r_[0, col.view(np.int8), 0]))
    return list(zip(idx[::2], idx[1::2]))


def _propagate(lab: np.ndarray, fg: np.ndarray, xs, step: int) -> None:
    for x in xs:
        prev = lab[:, x - step]
        if not prev.any():
            continue
        for y0, y1 in _runs(fg[:, x]):
            seg = prev[y0:y1]
            ids = [i for i in np.unique(seg) if i]
            if not ids:
                continue
            if len(ids) == 1:
                lab[y0:y1, x] = ids[0]
                continue
            centres = {i: float(np.where(seg == i)[0].mean()) + y0 for i in ids}
            ys = np.arange(y0, y1, dtype=np.float32)
            keys = np.array(list(centres))
            d = np.abs(ys[:, None] - np.array([centres[k] for k in keys])[None, :])
            lab[y0:y1, x] = keys[d.argmin(axis=1)]


def vote(counts_by_frame: list[list[tuple[float, float, int | None]]], window: int = 5,
         match_px: float = 80.0) -> list[list[int | None]]:
    """Frame voting for a run of consecutive frames from the fixed camera.
    Input per frame: [(cx, cy, count), ...] for its good columns. Output: the voted count per
    column = most common count of the nearest column (<= match_px) over +-window//2 frames;
    ties keep the frame's own count."""
    from collections import Counter
    half = window // 2
    out = []
    for i, cols in enumerate(counts_by_frame):
        res = []
        for cx, cy, n in cols:
            vals = []
            for j in range(max(0, i - half), min(len(counts_by_frame), i + half + 1)):
                cand = [(abs(ox - cx) ** 2 + abs(oy - cy) ** 2, on) for ox, oy, on in counts_by_frame[j]]
                if cand:
                    d2, on = min(cand, key=lambda t: t[0])
                    if d2 <= match_px ** 2:
                        vals.append(on)
            c = Counter(vals).most_common()
            res.append(c[0][0] if c and (len(c) == 1 or c[0][1] > c[1][1]) else n)
        out.append(res)
    return out
