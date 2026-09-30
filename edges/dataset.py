"""Edge (sheet) data: straightened single-column crops with per-sheet polygons.

STORED FORMAT (data/edges/<type>/):  <frame>_column_<k>.png  + .json
  native straightened column, full native width, black background, fixed height
  (2050 for crops made by edges.make_crops; the older type3 crops are ~2050-2075).
  Label: one "edge" polygon per sheet; optional "background" polygons.
  PNGs without a json are unlabelled and ignored for training.

MODEL INPUT: tight crop (image > 0, pad 6) -> width fitted to INPUT_W = 74 px:
  center  keep the middle 74 px, native pixels            (inference default)
  squash  resize the width to 74 px, height untouched     (keeps the whole column)
Height is NEVER rescaled at inference: the model is locked to sheet pitch
(~10-14 px on types 2/3/10), which already matches across types. Training mixes
both width modes so one model serves either input.

TRAINING WINDOWS (all measured on type3, see README):
  aniso flow target, wide pitch jitter sy 0.5-3.5 (without it the model is
  pitch-locked: 2x pitch -> MAE 5.2), fragment aug p=0.5 (pieces floated on black
  with labels cut from the FULL column -- fixed a +-1 flutter), +-6 deg only
  (180 deg flip measured worse on human labels), strong photometric.
"""
from __future__ import annotations

import glob
import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import paths
from common import labelme, splits
from common.flows import flow_targets
from common.imageio import degrade, photometric, to_tensor

cv2.setNumThreads(0)
SIZE = 256
INPUT_W = 74
PAD = 6
WIDTH_MODES = {"center": 0.5, "squash": 0.5}


def frame_of(stem: str) -> str:
    return stem.rsplit("_column_", 1)[0]


def discover() -> tuple[list[dict], dict[str, int]]:
    items, unlabelled = [], {}
    for png in sorted(glob.glob(os.path.join(paths.EDGES_DATA, "*", "*.png"))):
        typ = os.path.basename(os.path.dirname(png))
        js = png[:-4] + ".json"
        if not os.path.exists(js):
            unlabelled[typ] = unlabelled.get(typ, 0) + 1
            continue
        stem = os.path.splitext(os.path.basename(png))[0]
        items.append({"stem": stem, "frame": frame_of(stem), "type": typ, "image": png, "json": js})
    return items, unlabelled


def sync_split(items: list[dict], write: bool = True) -> dict:
    split = json.load(open(paths.EDGES_SPLIT)) if os.path.exists(paths.EDGES_SPLIT) else {}
    new_split, notes = splits.assign(sorted({it["frame"] for it in items}), split, splits.val_sessions(paths.COLUMNS_SPLIT))
    for n in notes:
        print("[split]", n)
    if write and notes:
        json.dump(new_split, open(paths.EDGES_SPLIT, "w"), indent=1)
    return new_split


def tight(img: np.ndarray, lab: np.ndarray | None = None, pad: int = PAD):
    ys, xs = np.where(img.max(axis=2) > 0)
    if ys.size == 0:
        return img, lab
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + 1 + pad)
    return img[y0:y1, x0:x1], (None if lab is None else lab[y0:y1, x0:x1])


def fit_width(img: np.ndarray, lab: np.ndarray | None, mode: str, width: int = INPUT_W):
    """Width normalisation on a TIGHT crop (the pad counts as part of the width)."""
    h, w = img.shape[:2]
    target = width + 2 * PAD
    if w <= target:
        return img, lab
    if mode == "center":
        a = (w - target) // 2
        return img[:, a:a + target], (None if lab is None else lab[:, a:a + target])
    if mode == "squash":
        return (cv2.resize(img, (target, h), interpolation=cv2.INTER_AREA),
                None if lab is None else cv2.resize(lab, (target, h), interpolation=cv2.INTER_NEAREST))
    raise ValueError(mode)


def model_input(img: np.ndarray, mode: str = "center") -> np.ndarray:
    """Stored crop (any width) -> the exact image the model sees at inference."""
    return fit_width(*tight(img), mode)[0]


def local_lighting_v2(img: np.ndarray, rng=random) -> np.ndarray:
    """Label-FREE local lighting. (An augmentation must never use the label: a version that
    dimmed glare on unlabelled pixels painted the label boundary into the image and hurt.)

    Everything here is computed from the image alone:
      glare     placed anywhere on the column, oriented along the window's dominant
                sheet direction (structure tensor), and dimmed where the image is
                locally dark -- grooves are dark, so they stay darker than sheets
      gradient  smooth illumination ramp
      shadow    soft dark band
    """
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    content = img.max(axis=2) > 0
    if content.sum() > 50 and rng.random() < 0.7:
        g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
        jxx, jyy, jxy = (gx * gx)[content].sum(), (gy * gy)[content].sum(), (gx * gy)[content].sum()
        ang = float(np.degrees(0.5 * np.arctan2(2 * jxy, jxx - jyy))) + 90.0   # along the sheets
        ys, xs = np.where(content)
        glare = np.zeros((h, w), np.float32)
        for _ in range(rng.randint(1, 3)):
            k = rng.randrange(len(ys))
            cv2.ellipse(glare, (int(xs[k]), int(ys[k])), (rng.randint(15, 60), rng.randint(3, 18)),
                        ang + rng.uniform(-10, 10), 0, 360, rng.uniform(0.6, 1.0), -1)
        glare = cv2.GaussianBlur(glare, (0, 0), rng.uniform(2, 6))
        lo, hi = np.percentile(g[content], (20, 80))
        bright = np.clip((g - lo) / max(hi - lo, 1.0), 0.3, 1.0)               # grooves -> 0.3
        f = f + (glare * bright * content)[..., None] * rng.uniform(80, 200)
    if rng.random() < 0.4:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        f = f * (1 + rng.uniform(-0.35, 0.35) * (xx / w - 0.5) * 2 + rng.uniform(-0.35, 0.35) * (yy / h - 0.5) * 2)[..., None]
    if rng.random() < 0.3:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        th = np.radians(rng.uniform(-60, 60))
        d = (xx - w / 2) * np.sin(th) + (yy - rng.uniform(0, h)) * np.cos(th)
        f = f * (1 - rng.uniform(0.3, 0.65) * np.exp(-(d / rng.uniform(8, 40)) ** 2))[..., None]
    return np.clip(f, 0, 255).astype(np.uint8)


def load(it: dict) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(it["image"], cv2.IMREAD_COLOR)
    lab = labelme.edge_labels(it["json"])
    if lab.shape != img.shape[:2]:
        raise ValueError(f"{it['json']}: label size {lab.shape} != image {img.shape[:2]}")
    return img, lab


class EdgeDataset(Dataset):
    def __init__(self, items: list[dict], repeats: int = 64, width_modes: dict = WIDTH_MODES,
                 lighting: str = ""):
        self.items = items * repeats
        self.modes, self.weights = list(width_modes), list(width_modes.values())
        self.lighting = lighting
        self._cache: dict = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        if it["image"] not in self._cache:
            self._cache[it["image"]] = tight(*load(it))
        img, lab = self._cache[it["image"]]
        img, lab = fit_width(img, lab, random.choices(self.modes, self.weights)[0])
        sy, sx = random.uniform(0.5, 3.5), random.uniform(0.8, 1.25)
        img = cv2.resize(img, None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)
        lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        h, w = lab.shape
        wi, wl = np.zeros((SIZE, SIZE, 3), np.uint8), np.zeros((SIZE, SIZE), np.int32)
        cw = min(w, SIZE)
        xs, xd = random.randint(0, max(0, w - SIZE)), random.randint(0, SIZE - cw)
        if random.random() < 0.5:                                  # fragment floated on black
            fh = min(h, random.randint(int(0.3 * SIZE), SIZE))
            y0, yd = random.randint(0, h - fh), random.randint(0, SIZE - fh)
        else:                                                      # window anchored at top
            y0 = random.randint(0, max(0, h - SIZE)); fh = min(h - y0, SIZE); yd = 0
        wi[yd:yd + fh, xd:xd + cw] = img[y0:y0 + fh, xs:xs + cw]
        wl[yd:yd + fh, xd:xd + cw] = lab[y0:y0 + fh, xs:xs + cw]
        if random.random() < 0.5:
            wi, wl = wi[:, ::-1], wl[:, ::-1]
        wi, wl = np.ascontiguousarray(wi), np.ascontiguousarray(wl)
        if random.random() < 0.6:
            M = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), random.uniform(-6, 6), 1.0)
            wi = cv2.warpAffine(wi, M, (SIZE, SIZE), flags=cv2.INTER_LINEAR)
            wl = cv2.warpAffine(wl, M, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
        if self.lighting == "v2":
            wi = local_lighting_v2(wi)
        wi = degrade(photometric(wi))
        return (to_tensor(wi), torch.from_numpy((wl > 0).astype(np.float32))[None],
                torch.from_numpy(flow_targets(wl, aniso=True)))


def build_val(items: list[dict], mode: str = "center") -> list[dict]:
    out = []
    for it in items:
        img, lab = fit_width(*tight(*load(it)), mode)
        out.append({**it, "img": img, "gt": lab, "n_gt": int(len(np.unique(lab)) - 1)})
    return out
