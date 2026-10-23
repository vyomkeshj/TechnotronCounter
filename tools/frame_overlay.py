"""Run the shipped pipeline on full frames and draw BOTH the column masks and the sheet masks
back onto the original photo (sheet masks are mapped from the straightened strip to the frame).

    python -m tools.frame_overlay <folder> [...] --out <dir> [--labelled-only]

Per frame writes <name>_overlay.jpg (full resolution) and prints predicted count vs the number
of labelled "edge" polygons inside each labelled good_column (when a LabelMe json exists).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inference"))
from sheetcount import pipeline as P  # noqa: E402
from sheetcount.straighten import (CUT_DIS, CUT_DIS_EX, CUT_DIS_EX2, POSUN_MASKY,  # noqa: E402
                                   CANVAS_H, _rotate, _straight_angle, _x_limits)


def straighten_with_coords(frame, polys):
    """Same geometry as SheetCounter.straighten_columns, plus for each strip a (H, W, 2) map of
    frame (x, y) per strip pixel (-1 where empty). None where a column cannot be cut."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    straight = any(p[:, 0].min() <= 1 and p[:, 0].max() >= W - 1 for p in polys)
    ang = _straight_angle([cv2.minAreaRect(p) for p in polys], W, straight)
    rot = _rotate(frame, ang)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    coord = np.dstack([xx + 1, yy + 1])                      # +1 so 0 means "outside"
    rc = np.dstack([_rotate(coord[..., i], ang, flags=cv2.INTER_NEAREST) for i in range(2)])
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
        crop = rot[:, x0:x1].copy(); crop[rm[:, x0:x1] == 0] = 0
        cc = rc[:, x0:x1].copy(); cc[rm[:, x0:x1] == 0] = 0
        crop, cc = cv2.rotate(crop, cv2.ROTATE_180), cv2.rotate(cc, cv2.ROTATE_180)
        ys, xs = np.where(crop.max(axis=2) > 0)               # to_native_canvas placement
        if ys.size == 0 or ys.max() + 1 - ys.min() > CANVAS_H:
            out.append(None); continue
        y0, y1, xa, xb = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        top = min(max(0, y0 - max(0, (crop.shape[0] - CANVAS_H) // 2)), CANVAS_H - (y1 - y0))
        can = np.zeros((CANVAS_H, xb - xa, 3), np.uint8); can[top:top + y1 - y0] = crop[y0:y1, xa:xb]
        ccan = np.zeros((CANVAS_H, xb - xa, 2), np.float32); ccan[top:top + y1 - y0] = cc[y0:y1, xa:xb]
        out.append((can, ccan - 1))
    return out


def edge_window(crop):
    """The (y0, y1, x0, x1) window SheetCounter.edge_input cuts from a strip."""
    ys, xs = np.where(crop.max(axis=2) > 0)
    y0, y1 = max(0, ys.min() - P.EDGE_PAD), min(crop.shape[0], ys.max() + 1 + P.EDGE_PAD)
    x0, x1 = max(0, xs.min() - P.EDGE_PAD), min(crop.shape[1], xs.max() + 1 + P.EDGE_PAD)
    target = P.EDGE_INPUT_W + 2 * P.EDGE_PAD
    if x1 - x0 > target:
        x0 += (x1 - x0 - target) // 2; x1 = x0 + target
    return y0, y1, x0, x1


def palette(n, seed=7):
    return np.random.RandomState(seed).randint(70, 255, (n + 1, 3)).astype(np.uint8)


def labelled_counts(js, cols):
    """For each predicted complete column: number of labelled edges whose centroid is inside it."""
    if not os.path.exists(js):
        return None
    shapes = json.load(open(js, encoding="utf-8"))["shapes"]
    cents = [np.mean(s["points"], axis=0) for s in shapes if s["label"] == "edge"]
    res = []
    for c in cols:
        H, W = c.mask.shape
        res.append(sum(1 for x, y in cents if 0 <= int(y) < H and 0 <= int(x) < W and c.mask[int(y), int(x)]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labelled-only", action="store_true")
    ap.add_argument("--models", default=None, help="models folder (default: inference/models)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sc = P.SheetCounter(a.models or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inference", "models"))
    files = []
    for i in a.inputs:
        files += sorted(glob.glob(os.path.join(i, "*.jpg"))) if os.path.isdir(i) else [i]
    if a.labelled_only:
        files = [f for f in files if os.path.exists(os.path.splitext(f)[0] + ".json")]
    for f in files:
        frame = cv2.imread(f)
        name = os.path.splitext(os.path.basename(f))[0]
        m = re.search(r"type\d+", name)
        typ = m.group(0) if m else "type3"
        cols = sc.segment_columns(frame, typ)
        goods = [c for c in cols if not c.cropped]
        strips = straighten_with_coords(frame, [c.polygon for c in goods])
        out = frame.copy()
        H, W = frame.shape[:2]
        # column masks: light tint + thick outline
        for c in cols:
            col = (0, 0, 255) if c.cropped else (0, 200, 0)
            out[c.mask] = (out[c.mask] * 0.75 + np.array(col) * 0.25).astype(np.uint8)
        sheet_lab = np.zeros((H, W), np.int32)
        counts, nid = [], 0
        for c, st in zip(goods, strips):
            if st is None:
                counts.append(None); continue
            can, cmap = st
            ey0, ey1, ex0, ex1 = edge_window(can)
            x = can[ey0:ey1, ex0:ex1]
            fg = sc.sheet_probability(x) > P.SHEET_THR
            n, seeds, ab = sc.count_sheets(fg)
            lab = sc.sheet_masks(fg, seeds, ab)
            counts.append(n)
            rr, cc = np.where(lab > 0)
            fx = cmap[ey0 + rr, ex0 + cc, 0].round().astype(int)
            fy = cmap[ey0 + rr, ex0 + cc, 1].round().astype(int)
            ok = (fx >= 0) & (fy >= 0) & (fx < W) & (fy < H)
            sheet_lab[fy[ok], fx[ok]] = lab[rr[ok], cc[ok]] + nid
            nid += int(lab.max())
        # fill the 1-px holes left by forward mapping
        filled = cv2.dilate(sheet_lab.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
        sheet_lab = np.where(sheet_lab > 0, sheet_lab, filled)
        pal = palette(max(nid, 1))
        sm = sheet_lab > 0
        out[sm] = (out[sm] * 0.35 + pal[sheet_lab[sm]] * 0.65).astype(np.uint8)
        lt = max(2, W // 640)
        for c in cols:
            cv2.polylines(out, [c.polygon], True, (0, 0, 255) if c.cropped else (0, 200, 0), lt)
        gt = labelled_counts(os.path.splitext(f)[0] + ".json", goods)
        fs = W / 1400
        for i, (c, n) in enumerate(zip(goods, counts)):
            ys, xs = np.where(c.mask)
            txt = f"{'?' if n is None else n}" + (f" (label {gt[i]})" if gt else "")
            org = (int(xs.mean()) - int(60 * fs), int(ys.min()) - 10 if ys.min() > 60 else int(ys.max()) + int(40 * fs))
            cv2.putText(out, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), lt * 4)
            cv2.putText(out, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), lt)
        cv2.imwrite(os.path.join(a.out, f"{name}_overlay.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"{name}: {len(cols)} columns ({len(goods)} complete) counts={counts} labelled={gt}", flush=True)


if __name__ == "__main__":
    main()
