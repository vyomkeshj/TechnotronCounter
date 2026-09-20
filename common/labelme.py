"""LabelMe json <-> instance label maps, for both label kinds.

COLUMN labels (frames):  one polygon per column, label "good_column" | "cropped_column"
EDGE labels (crops):     one polygon per sheet edge, label "edge";
                         optional "background" polygons are painted to 0 afterwards
Both: shape_type "polygon", image beside the json with the same stem.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

COLUMN_LABELS = ("good_column", "cropped_column")
EDGE, BACKGROUND = "edge", "background"
IMAGE_EXTS = (".jpg", ".png", ".jpeg")


def image_for(json_path: str) -> str | None:
    stem = os.path.splitext(json_path)[0]
    for ext in IMAGE_EXTS:
        if os.path.exists(stem + ext):
            return stem + ext
    d = json.load(open(json_path, encoding="utf-8"))
    alt = os.path.join(os.path.dirname(json_path), str(d.get("imagePath", "")))
    return alt if os.path.isfile(alt) else None


def column_label(raw: str) -> str:
    """The close-up set spells labels 'good column'/'cropped column'; the operational
    and quality sets use underscores. Same meaning -- normalise."""
    return raw.strip().replace(" ", "_")


def _poly(points) -> np.ndarray:
    return np.round(np.asarray(points, np.float64)).astype(np.int32)


def column_labels(json_path: str, hw: tuple[int, int] | None = None):
    """(instance map int32, is_cropped[id] float32 LUT) for a column frame.
    Unknown labels raise: a silently-ignored label is a silently-missing column."""
    d = json.load(open(json_path, encoding="utf-8"))
    H, W = int(d["imageHeight"]), int(d["imageWidth"])
    lab = np.zeros((H, W), np.int32)
    shapes = d.get("shapes", [])
    lut = np.zeros(len(shapes) + 1, np.float32)
    for i, s in enumerate(shapes, 1):
        name = column_label(s["label"])
        if name not in COLUMN_LABELS:
            raise ValueError(f"{json_path}: unexpected label {s['label']!r} (want {COLUMN_LABELS})")
        cv2.fillPoly(lab, [_poly(s["points"])], i)
        lut[i] = 1.0 if name == "cropped_column" else 0.0
    if hw is not None and tuple(hw) != (H, W):
        lab = cv2.resize(lab, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return lab, lut


def edge_labels(json_path: str) -> np.ndarray:
    """Instance map int32 for a sheet-edge crop; ids renumbered top-down."""
    d = json.load(open(json_path, encoding="utf-8"))
    H, W = int(d["imageHeight"]), int(d["imageWidth"])
    lab = np.zeros((H, W), np.int32)
    edges = [_poly(s["points"]) for s in d.get("shapes", []) if s["label"] == EDGE]
    other = {s["label"] for s in d.get("shapes", [])} - {EDGE, BACKGROUND}
    if other:
        raise ValueError(f"{json_path}: unexpected labels {sorted(other)} (want 'edge'/'background')")
    for new_id, (_, p) in enumerate(sorted(((float(p[:, 1].mean()), k) for k, p in enumerate(edges))), 1):
        cv2.fillPoly(lab, [edges[p]], new_id)
    bg = [_poly(s["points"]) for s in d.get("shapes", []) if s["label"] == BACKGROUND]
    if bg:
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, bg, 1)
        lab[m > 0] = 0
    return lab


def write_polygons(path: str, lab: np.ndarray, label: str, image_name: str,
                   offset: tuple[int, int] = (0, 0), canvas_hw: tuple[int, int] | None = None) -> None:
    """Instance map -> LabelMe json (largest external contour per instance)."""
    ox, oy = offset
    H, W = canvas_hw or lab.shape
    shapes = []
    for i in range(1, int(lab.max()) + 1):
        cs, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        if len(c) >= 3:
            shapes.append({"label": label, "points": [[float(x + ox), float(y + oy)] for x, y in c[:, 0]],
                           "group_id": None, "description": "model", "shape_type": "polygon", "flags": {}})
    json.dump({"version": "5.5.0", "flags": {}, "shapes": shapes, "imagePath": image_name,
               "imageData": None, "imageHeight": int(H), "imageWidth": int(W)}, open(path, "w"), indent=1)
