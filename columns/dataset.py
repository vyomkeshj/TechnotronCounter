"""Column data: discovery, pinned split, training dataset, validation set.

Instance frames: any LabelMe jpg+json under data/columns/<source>/ (or
<source>/<type>/), labels good_column / cropped_column. They supervise all
four output channels.
Legacy frames:  data/columns/legacy_fg/{images,masks}. Their masks fuse columns,
so they supervise the FOREGROUND channel only (valid=0 masks the flow and class
losses). Measured history: old instance-labelled data from another domain
(Data_1) HURT at every dose; these blobs help because they are only used for
the one thing they are right about.
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
from common.imageio import degrade, exposure, letterbox, photometric, to_tensor

cv2.setNumThreads(0)
SIZE = 256                                          # model input (letterboxed)
SCALE_LO = float(os.environ.get("COLUMN_SCALE_LO", "0.8"))  # scale-aug lower bound; columns.train --scale-lo sets it
WORK_CAP = 640            # decode once at <= 640 px long side; augmentation cost is O(pixels)


def discover() -> list[dict]:
    items = []
    for js in sorted(glob.glob(os.path.join(paths.COLUMNS_DATA, "**", "*.json"), recursive=True)):
        if os.path.basename(js) == "splits.json":
            continue
        img = labelme.image_for(js)
        if img is None:
            continue
        rel = os.path.relpath(os.path.dirname(js), paths.COLUMNS_DATA).replace(os.sep, "/")
        stem = os.path.splitext(os.path.basename(js))[0]
        group = splits.type_of(stem) if rel.startswith("quality") else rel.split("/")[0]
        items.append({"stem": stem, "image": img, "json": js, "source": rel, "group": group})
    return items


def legacy_items() -> list[dict]:
    out = []
    for m in sorted(glob.glob(os.path.join(paths.LEGACY_FG, "masks", "*.png"))):
        img = os.path.join(paths.LEGACY_FG, "images", os.path.splitext(os.path.basename(m))[0] + ".jpg")
        if os.path.exists(img):
            out.append({"stem": os.path.basename(img), "image": img, "mask": m})
    return out


def sync_split(items: list[dict], write: bool = True) -> dict:
    """Load the pinned split and assign any NEW frames (never re-rolls old ones)."""
    split = json.load(open(paths.COLUMNS_SPLIT)) if os.path.exists(paths.COLUMNS_SPLIT) else {}
    new_split, notes = splits.assign([it["stem"] for it in items], split, splits.val_sessions(paths.EDGES_SPLIT))
    for n in notes:
        print("[split]", n)
    if write and notes:
        json.dump(new_split, open(paths.COLUMNS_SPLIT, "w"), indent=1)
    return new_split


def _load_inst(it):
    img = cv2.imread(it["image"], cv2.IMREAD_COLOR)
    lab, lut = labelme.column_labels(it["json"], hw=img.shape[:2])
    for i in it.get("flip_good", ()):          # train-time label correction (--class-override)
        lut[i] = 0.0
    s = WORK_CAP / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return img, lab, lut


def _load_fg(it):
    img = cv2.imread(it["image"], cv2.IMREAD_COLOR)
    m = cv2.imread(it["mask"], cv2.IMREAD_GRAYSCALE)
    if m.shape[:2] != img.shape[:2]:
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    s = WORK_CAP / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return img, (m > 127).astype(np.int32), np.zeros(2, np.float32)


class ColumnDataset(Dataset):
    """Returns (x, fg, flow, valid, cls) with valid=0 for legacy fg-only frames.

    Augmentation = the measured winner: zoom jitter around the columns (65 %),
    h-flip, +-12 deg / 0.8-1.2 affine, strong photometric + degradation.
    NOT used, measured negative: full +-180 deg rotation (gravity prior is real).
    """

    def __init__(self, inst: list[dict], fg_only: list[dict], oversample_inst: int = 4,
                 negatives: list[str] = (), neg_frac: float = 0.0, bg_frac: float = 0.0, exposure_p: float = 0.0,
                 wide_p: float = 0.0, wide_scale_lo: float = 0.45, wide_rot: float = 20.0, persp: float = 0.15):
        """negatives: column-free images; neg_frac of the samples are one of them and bg_frac are
        background crops of a training frame that touch no labelled column. Both are trained as
        'no column anywhere' (fg = 0; flow and class losses masked). exposure_p: probability of the
        camera-exposure augmentation (common.imageio.exposure). wide_p: probability that a sample uses
        the WIDE camera placement (scale down to wide_scale_lo = camera farther away, roll +-wide_rot deg,
        pitch/yaw perspective up to persp); the rest keep the measured default (scale SCALE_LO..1.2,
        +-12 deg, no perspective) so the existing operating point stays the bulk of training."""
        self.items = [("inst", it) for it in inst] * oversample_inst + [("fg", it) for it in fg_only]
        self.inst = list(inst)
        self.negatives, self.neg_frac, self.bg_frac, self.exposure_p = list(negatives), neg_frac, bg_frac, exposure_p
        self.wide_p, self.wide_scale_lo, self.wide_rot, self.persp = wide_p, wide_scale_lo, wide_rot, persp
        self._cache: dict = {}

    def __len__(self):
        return len(self.items)

    def _base(self, kind, it):
        key = it["image"]
        if key not in self._cache:
            self._cache[key] = _load_inst(it) if kind == "inst" else _load_fg(it)
        img, lab, lut = self._cache[key]
        return img.copy(), lab.copy(), lut

    def _background_crop(self):
        """A crop of a training frame that touches no labelled column (>= 12 px away): the scene around the
        columns (beams, wood, walls, other stock) labelled 'no column'. None if no such crop was found."""
        for _ in range(4):
            img, lab, _ = self._base("inst", random.choice(self.inst))
            h, w = lab.shape
            near = cv2.dilate((lab > 0).astype(np.uint8), np.ones((25, 25), np.uint8))
            for _ in range(30):
                ch, cw = int(h * random.uniform(0.2, 0.7)), int(w * random.uniform(0.2, 0.7))
                if ch < 32 or cw < 32:
                    continue
                y0, x0 = random.randint(0, h - ch), random.randint(0, w - cw)
                if not near[y0:y0 + ch, x0:x0 + cw].any():
                    return img[y0:y0 + ch, x0:x0 + cw]
        return None

    def _negative(self):
        img = cv2.imread(random.choice(self.negatives), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        s = WORK_CAP / max(h, w)
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA); h, w = img.shape[:2]
        if random.random() < 0.5:                          # random sub-window: any scale, any aspect
            ch, cw = int(h * random.uniform(0.4, 1)), int(w * random.uniform(0.4, 1))
            y0, x0 = random.randint(0, h - ch), random.randint(0, w - cw)
            img = img[y0:y0 + ch, x0:x0 + cw]
        return img

    def __getitem__(self, idx):
        r = random.random()
        neg = None
        if r < self.bg_frac and self.inst:
            neg = self._background_crop()
        elif r < self.bg_frac + self.neg_frac and self.negatives:
            neg = self._negative()
        if neg is not None:
            kind, img, lut = "neg", np.ascontiguousarray(neg), np.zeros(2, np.float32)
            lab = np.zeros(img.shape[:2], np.int32)
        else:
            kind, it = self.items[idx]
            img, lab, lut = self._base(kind, it)
        ys, xs = np.where(lab > 0)
        if len(ys) > 50 and random.random() < 0.65:
            h, w = lab.shape
            bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
            y0 = max(0, ys.min() - int(bh * random.uniform(0.05, 0.6)))
            y1 = min(h, ys.max() + 1 + int(bh * random.uniform(0.05, 0.6)))
            x0 = max(0, xs.min() - int(bw * random.uniform(0.05, 0.6)))
            x1 = min(w, xs.max() + 1 + int(bw * random.uniform(0.05, 0.6)))
            if y1 - y0 >= 32 and x1 - x0 >= 32:
                img, lab = img[y0:y1, x0:x1], lab[y0:y1, x0:x1]
        if random.random() < 0.5:
            img, lab = img[:, ::-1], lab[:, ::-1]
        img, lab = np.ascontiguousarray(img), np.ascontiguousarray(lab)
        wide = self.wide_p and random.random() < self.wide_p
        if wide or random.random() < 0.7:
            h, w = lab.shape
            rot, lo = (self.wide_rot, self.wide_scale_lo) if wide else (12, SCALE_LO)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-rot, rot), random.uniform(lo, 1.2))
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
            lab = cv2.warpAffine(lab, M, (w, h), flags=cv2.INTER_NEAREST)
        if wide and self.persp and random.random() < 0.6:     # camera pitched / yawed: one side shrinks
            h, w = lab.shape
            f = random.uniform(-self.persp, self.persp)
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            if random.random() < 0.5:                           # pitch
                d = abs(f) * w / 2
                dst = np.float32([[d, 0], [w - d, 0], [w, h], [0, h]]) if f > 0 else np.float32([[0, 0], [w, 0], [w - d, h], [d, h]])
            else:                                               # yaw
                d = abs(f) * h / 2
                dst = np.float32([[0, d], [w, 0], [w, h], [0, h - d]]) if f > 0 else np.float32([[0, 0], [w, d], [w, h - d], [0, h]])
            Mp = cv2.getPerspectiveTransform(src, dst)
            img = cv2.warpPerspective(img, Mp, (w, h), flags=cv2.INTER_LINEAR)
            lab = cv2.warpPerspective(lab, Mp, (w, h), flags=cv2.INTER_NEAREST)
        if self.exposure_p and random.random() < self.exposure_p:
            img = exposure(img)
        img = degrade(photometric(img))
        img, _, _, _ = letterbox(img, SIZE, 0, cv2.INTER_LINEAR)
        lab, _, _, _ = letterbox(lab, SIZE, 0, cv2.INTER_NEAREST)
        valid = kind == "inst"
        fl = flow_targets(lab) if valid else np.zeros((2, SIZE, SIZE), np.float32)
        cls = lut[np.clip(lab, 0, len(lut) - 1)] if valid else np.zeros((SIZE, SIZE), np.float32)
        return (to_tensor(img), torch.from_numpy((lab > 0).astype(np.float32))[None], torch.from_numpy(fl),
                torch.tensor(1.0 if valid else 0.0), torch.from_numpy(cls.astype(np.float32))[None])


def build_val(items: list[dict]) -> list[dict]:
    """Letterboxed tensors + GT for evaluation (no augmentation)."""
    out = []
    for it in items:
        img = cv2.imread(it["image"], cv2.IMREAD_COLOR)
        lab, lut = labelme.column_labels(it["json"], hw=img.shape[:2])
        for i in it.get("flip_good", ()):      # label correction (--class-override): train as complete
            lut[i] = 0.0
        imgL, _, _, _ = letterbox(img, SIZE, 0, cv2.INTER_LINEAR)
        labL, _, _, _ = letterbox(lab, SIZE, 0, cv2.INTER_NEAREST)
        out.append({**it, "x": to_tensor(imgL), "gt": labL, "lut": lut})
    return out
