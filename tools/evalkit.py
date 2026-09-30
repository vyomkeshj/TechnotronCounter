"""Shared helpers for the evaluation tools and the edge pseudo-labeller.

- the held-out test frames and their expected sheet counts
- the production straightener, returning one strip per polygon in input order
- loading the inference package (inference/) with any models folder
"""
from __future__ import annotations

import json
import os
import re
import sys

import cv2
import numpy as np

import paths
from common.labelme import column_label
from common.straighten import (CUT_DIS, CUT_DIS_EX, CUT_DIS_EX2, POSUN_MASKY, _rotate, _straight_angle,
                               _x_limits, to_native_canvas)

INFERENCE = os.path.join(paths.ROOT, "inference")
MODELS = os.path.join(INFERENCE, "models")
TYPES = ("type2", "type3", "type10")


def expected(stem: str) -> int:
    """The true sheet count: type3 50, type10 24, type2 the number in the file name (50 or 30).
    (For other types the file-name number is the batch size, NOT the sheet count.)"""
    t = re.search(r"(type\d+)", stem).group(1)
    if t == "type2":
        return int(re.search(r"_(\d+)pieces", stem).group(1))
    return {"type3": 50, "type10": 24}[t]


def type_of(stem: str) -> str:
    return re.search(r"(type\d+)", stem).group(1)


def test_frames() -> list[str]:
    """The held-out test set: the column split's val frames of type2/3/10 (whole capture sessions
    132812 + 134020 for type2, 131412 for type3; frames 9-14 of the only type10 session)."""
    return [s for s in json.load(open(paths.COLUMNS_SPLIT))["val"] if re.search(r"type(2|3|10)_", s)]


def frame_path(stem: str) -> str:
    return os.path.join(paths.COLUMNS_DATA, "quality", type_of(stem), stem + ".jpg")


def good_polygons(stem: str) -> list[np.ndarray]:
    """The human-labelled complete ("good") column polygons of a frame."""
    js = frame_path(stem)[:-4] + ".json"
    return [np.round(np.array(s["points"])).astype(np.int32)
            for s in json.load(open(js, encoding="utf-8"))["shapes"] if column_label(s["label"]) == "good_column"]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    i = int((a & b).sum())
    return i / int((a | b).sum()) if i else 0.0


def straighten_polys(image: np.ndarray, polys: list[np.ndarray], angle_polys=None) -> list:
    """The production straightener, one strip per polygon IN INPUT ORDER (None where it cannot cut).
    The rotation angle comes from `angle_polys` (production: the frame's complete columns)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    ap = polys if angle_polys is None else angle_polys
    straight = any(p[:, 0].min() <= 1 and p[:, 0].max() >= W - 1 for p in ap)
    ang = _straight_angle([cv2.minAreaRect(p) for p in ap], W, straight)
    rot = _rotate(image, ang)
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


def ckpt_path(x: str) -> str:
    """A run name under runs/ (e.g. edges_seed) or a checkpoint path -> the checkpoint file."""
    if os.path.isfile(x):
        return x
    for c in (os.path.join(paths.RUNS, x, "last.pt"), os.path.join(paths.RUNS, x, "stage2", "last.pt")):
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(x)


def sheet_counter(models: str | None = None):
    """The inference package's SheetCounter, with the shipped models or another models folder."""
    if INFERENCE not in sys.path:
        sys.path.insert(0, INFERENCE)
    from sheetcount.pipeline import SheetCounter
    return SheetCounter(models or MODELS)
