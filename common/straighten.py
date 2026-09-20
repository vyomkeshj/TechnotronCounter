"""Straighten labelled columns into vertical single-column crops (for edge labelling).

Vendored from TosaI66/BAD_ELIMINATE.py (ColumnEliminator: _straight_angle,
_rotate, _conture_x_limits, _split_columns), with the GUI, the binary-mask
polygon extraction, the geometric good/bad filter and the shapely-based column
separation removed: here the columns ARE the labeller's good_column polygons.
The geometry is unchanged, so crops match the ones the existing edge labels
were drawn on. Verified: angles identical to the original on OpenCV 4.13.

Output of straighten_frame(): one crop per good column, left to right, full
rotated-frame height, background blacked out, rotated 180 deg like the original.
Then to_native_canvas() puts each on the fixed-height canvas the edge dataset
stores (native width, never resampled).
"""
from __future__ import annotations

import json
import statistics

import cv2
import numpy as np

from common.labelme import column_label

DO_NOT_STRAIGHT_MIN_WIDTH = 0.98
POSUN_MASKY = 2          # mask shift used to locate each vertical edge
LINE_MASK = 8            # vertical dilation divisor for the edge line mask
CUT_DIS, CUT_DIS_EX, CUT_DIS_EX2 = -45, -25, -10   # inward edge correction by column width
CANVAS_H = 2050          # stored edge crops: fixed height, native width


def _straight_angle(rects, img_w, straight_plates):
    if not rects:
        return 0.0
    ang1, ang2 = [], []
    if len(rects) <= 3:
        for box in rects:
            if box[1][1] > DO_NOT_STRAIGHT_MIN_WIDTH * img_w and straight_plates:
                return box[2] - 90
            if box[1][0] > DO_NOT_STRAIGHT_MIN_WIDTH * img_w and straight_plates:
                return box[2]
    for box in rects:
        (ang1 if box[1][0] > box[1][1] else ang2).append(box[2] + 90 if box[1][0] > box[1][1] else box[2])
    if len(ang1) != len(ang2):
        return statistics.mean(ang1 if len(ang1) > len(ang2) else ang2)
    if ang1 and ang1[0] == 180:
        return statistics.mean(ang2)
    if ang2 and ang2[0] == 180:
        return statistics.mean(ang1)
    if len(ang1) == 1:
        return ang1[0] if ang1[0] > ang2[0] else ang2[0]
    return float(max(ang1 + ang2)) if (ang1 or ang2) else 0.0


def _rotate(image, alfa):
    m, n = image.shape[:2]
    M = cv2.getRotationMatrix2D((n / 2, m / 2), alfa, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    bw, bh = int(m * sin + n * cos), int(m * cos + n * sin)
    M[0, 2] += bw / 2 - n / 2
    M[1, 2] += bh / 2 - m / 2
    return cv2.warpAffine(image, M, (bw, bh))


def _x_limits(shift, rot_mask, rot_outline):
    rows, cols = rot_mask.shape[:2]
    shifted = cv2.warpAffine(rot_mask, np.float32([[1, 0, shift], [0, 1, 0]]), (cols, rows))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, int(rows / LINE_MASK))))
    line = 255 - cv2.dilate(shifted, k)
    sub = cv2.bitwise_or(line, line, mask=rot_outline)
    white = np.where(sub != 0)
    if white[1].size == 0:
        return (0, 0), (0, 0)
    i0, i1 = np.argmin(white[1]), np.argmax(white[1])
    return (white[0][i0], white[1][i0]), (white[0][i1], white[1][i1])


def straighten_frame(image: np.ndarray, column_json: str) -> list[np.ndarray]:
    shapes = json.load(open(column_json, encoding="utf-8"))["shapes"]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    polys = [np.round(np.array(s["points"], np.float64)).astype(np.int32)
             for s in shapes if column_label(s["label"]) == "good_column" and len(s["points"]) >= 3]
    if not polys:
        return []
    straight = any(p[:, 0].min() <= 1 and p[:, 0].max() >= W - 1 for p in polys)
    ang = _straight_angle([cv2.minAreaRect(p) for p in polys], W, straight)
    rot = _rotate(image, ang)
    RH, RW = rot.shape[:2]
    out = []
    for p in polys:
        single = np.zeros_like(gray); cv2.fillPoly(single, [p], 255)
        outline = np.zeros_like(gray); cv2.drawContours(outline, [p], -1, 255, 1)
        rm, ro = _rotate(single, ang), _rotate(outline, ang)
        lmin1, lmax1 = _x_limits(POSUN_MASKY, rm, ro)
        lmin2, _ = _x_limits(-POSUN_MASKY, rm, ro)
        if lmin1[0] == 0 and lmin2[0] == 0:
            continue
        a, b = sorted((lmin2[1], lmax1[1]))
        d = b - a
        cut = CUT_DIS if d > 220 else CUT_DIS_EX if d > 120 else CUT_DIS_EX2
        x0, x1 = int(np.clip(a - cut, 0, RW - 1)), int(np.clip(b + cut, 0, RW))
        if x1 <= x0:
            continue
        crop = rot[:, x0:x1].copy()
        crop[rm[:, x0:x1] == 0] = 0
        out.append((x0, cv2.rotate(crop, cv2.ROTATE_180)))
    return [c for _, c in sorted(out, key=lambda t: t[0])]


def to_native_canvas(crop: np.ndarray, out_h: int = CANVAS_H) -> np.ndarray | None:
    """Content columns only (native width, never resampled) on a fixed-height canvas,
    vertical position preserved. None if empty or taller than the canvas."""
    ys, xs = np.where(crop.max(axis=2) > 0)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h = y1 - y0
    if h > out_h:
        return None
    top = min(max(0, y0 - max(0, (crop.shape[0] - out_h) // 2)), out_h - h)
    canvas = np.zeros((out_h, x1 - x0, 3), np.uint8)
    canvas[top:top + h] = crop[y0:y1, x0:x1]
    return canvas
