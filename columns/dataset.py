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
SIZE = int(os.environ.get("COLUMN_SIZE", "256"))   # model input (letterboxed); 512 = the high-resolution test
SCALE_LO = float(os.environ.get("COLUMN_SCALE_LO", "0.8"))  # scale-aug lower bound; columns.train --scale-lo sets it
WORK_CAP = max(640, int(2.5 * SIZE))   # decode once at <= this long side (640 at 256); augmentation cost is O(pixels)
# How LABELS are resized. "nearest" (cv2.INTER_NEAREST) samples each output pixel at the TOP-LEFT of its source
# block while images are resampled at the block CENTRE, so every model trained with it outputs masks shifted
# half a 256-px pixel down-right (~3.75 px at 1920 px, measured 2026-11-04). "exact" = INTER_NEAREST_EXACT,
# centre-aligned. columns.train --label-resize sets it (env var: DataLoader workers re-import this module).
LABEL_INTERP = cv2.INTER_NEAREST_EXACT if os.environ.get("COLUMN_LABEL_RESIZE", "nearest") == "exact" else cv2.INTER_NEAREST


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
        lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=LABEL_INTERP)
    return img, lab, lut


def _load_fg(it):
    img = cv2.imread(it["image"], cv2.IMREAD_COLOR)
    m = cv2.imread(it["mask"], cv2.IMREAD_GRAYSCALE)
    if m.shape[:2] != img.shape[:2]:
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=LABEL_INTERP)
    s = WORK_CAP / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=LABEL_INTERP)
    return img, (m > 127).astype(np.int32), np.zeros(2, np.float32)


class ColumnDataset(Dataset):
    """Returns (x, fg, flow, valid, cls) with valid=0 for legacy fg-only frames.

    Augmentation = the measured winner: zoom jitter around the columns (65 %),
    h-flip, +-12 deg / 0.8-1.2 affine, strong photometric + degradation.
    NOT used, measured negative: full +-180 deg rotation (gravity prior is real).
    """

    def __init__(self, inst: list[dict], fg_only: list[dict], oversample_inst: int = 4,
                 negatives: list[str] = (), neg_frac: float = 0.0, bg_frac: float = 0.0, exposure_p: float = 0.0,
                 wide_p: float = 0.0, wide_scale_lo: float = 0.45, wide_rot: float = 20.0, persp: float = 0.15,
                 occlude_p: float = 0.0, turn_p: float = 0.0, seam_r: int = 0, paste_p: float = 0.0):
        """negatives: column-free images; neg_frac of the samples are one of them and bg_frac are
        background crops of a training frame that touch no labelled column. Both are trained as
        'no column anywhere' (fg = 0; flow and class losses masked). exposure_p: probability of the
        camera-exposure augmentation (common.imageio.exposure). wide_p: probability that a sample uses
        the WIDE camera placement (scale down to wide_scale_lo = camera farther away, roll +-wide_rot deg,
        pitch/yaw perspective up to persp); the rest keep the measured default (scale SCALE_LO..1.2,
        +-12 deg, no perspective) so the existing operating point stays the bulk of training."""
        self.items = ([("inst", it) for it in inst for _ in range(it.get("rep", oversample_inst))]
                      + [("fg", it) for it in fg_only])         # it["rep"]: per-frame repeat (e.g. agent labels)
        self.inst = list(inst)
        self.negatives, self.neg_frac, self.bg_frac, self.exposure_p = list(negatives), neg_frac, bg_frac, exposure_p
        self.wide_p, self.wide_scale_lo, self.wide_rot, self.persp = wide_p, wide_scale_lo, wide_rot, persp
        self.occlude_p = occlude_p                # in-frame occlusion: a band hides part of a complete column
        self.turn_p = turn_p                      # camera turned 90/180/270 deg (portrait / landscape / upside down)
        self.seam_r = seam_r                      # seam map: pixels within seam_r (at 256) of 2 different columns
        self.paste_p = paste_p                    # a cropped piece placed just past a complete column's end
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

    def _occlude(self, img, lab, lut):
        """A band (strap / plank / board: texture from a negative image or plain paint) laid across a COMPLETE
        column, hiding 15-50 % of its length. Hidden pixels become background, every column the band touches is
        relabelled CROPPED and its visible pieces become separate instances (a hidden column is not countable)."""
        ids = [i for i in range(1, int(lab.max()) + 1) if i < len(lut) and lut[i] < 0.5 and (lab == i).sum() > 400]
        if not ids:
            return img, lab, lut
        m = lab == random.choice(ids)
        ys, xs = np.nonzero(m)
        P = np.stack([xs, ys], 1).astype(np.float32)
        mu = P.mean(0)
        _, _, vt = np.linalg.svd(P - mu, full_matrices=False)
        u = vt[0]
        t = (P - mu) @ u
        L = float(t.max() - t.min())
        bw = random.uniform(0.15, 0.5) * L
        c = mu + u * random.uniform(t.min() + bw / 2, t.max() - bw / 2)
        ang = np.arctan2(u[1], u[0]) + np.pi / 2 + np.radians(random.uniform(-25, 25))
        d = np.array([np.cos(ang), np.sin(ang)], np.float32)
        n = np.array([-d[1], d[0]], np.float32)
        wcol = float(np.ptp((P - mu) @ vt[1]))                 # the column's own width across its long axis
        half = wcol * random.uniform(0.6, 1.2)                 # the band crosses THIS column, not the whole frame
        poly = np.stack([c + d * half + n * bw / 2, c - d * half + n * bw / 2, c - d * half - n * bw / 2,
                         c + d * half - n * bw / 2]).round().astype(np.int32)
        occ = np.zeros(lab.shape, np.uint8)
        cv2.fillPoly(occ, [poly], 1)
        occ = occ > 0
        if self.negatives and random.random() < 0.6:
            tex = cv2.imread(random.choice(self.negatives), cv2.IMREAD_COLOR)
            tex = cv2.resize(tex, (lab.shape[1], lab.shape[0]))
        else:
            col = np.array([random.randint(10, 250) for _ in range(3)], np.float32)
            tex = np.clip(col + np.random.normal(0, random.uniform(2, 20), img.shape), 0, 255).astype(np.uint8)
        img = img.copy()
        img[occ] = tex[occ]
        lut = list(lut)
        out = lab.copy()
        out[occ] = 0
        for j in np.unique(lab[occ]):
            if j == 0:
                continue
            lost = (lab[occ] == j).sum() / (lab == j).sum()
            if lost < 0.05:
                continue
            n_cc, cc = cv2.connectedComponents((out == j).astype(np.uint8))
            for k in range(1, n_cc):
                piece = cc == k
                if piece.sum() < 30:
                    out[piece] = 0
                    continue
                if k > 1:
                    out[piece] = len(lut)
                    lut.append(1.0)
            if j < len(lut):
                lut[j] = 1.0
        return img, out, np.array(lut, np.float32)

    def _paste(self, img, lab, lut):
        """The fused-column failure, synthesised: a piece (15-50 % of the length) of another column of the same frame
        (columns in a frame are parallel, so a pure translation keeps it realistic) placed 2-12 px from a complete
        column, either BESIDE it along its long edge (type7 143956_003, type1 142210_025) or past one END (two pieces
        either side of a strap, type1 142016_019). The piece is a new CROPPED instance; columns it covers lose those
        pixels and become cropped if they lose > 5 %."""
        ids = [i for i in range(1, int(lab.max()) + 1) if i < len(lut) and (lab == i).sum() > 400]
        good = [i for i in ids if lut[i] < 0.5]
        if not good or len(ids) < 2:
            return img, lab, lut
        a = random.choice(good)
        donor = random.choice([i for i in ids if i != a])
        ys, xs = np.nonzero(lab == a)
        P = np.stack([xs, ys], 1).astype(np.float32)
        mu = P.mean(0)
        _, _, vt = np.linalg.svd(P - mu, full_matrices=False)
        u, v = vt[0], vt[1]
        ta, wa = (P - mu) @ u, float(np.ptp((P - mu) @ v))
        dy, dx = np.nonzero(lab == donor)
        Q = np.stack([dx, dy], 1).astype(np.float32)
        tq = (Q - mu) @ u
        L = float(tq.max() - tq.min())
        frac = random.uniform(0.15, 0.5)                       # the piece: one end of the donor column
        keep = tq <= tq.min() + frac * L if random.random() < 0.5 else tq >= tq.max() - frac * L
        Q = Q[keep]
        tq = (Q - mu) @ u
        gap = random.uniform(2, 12)
        lat = (Q - mu) @ v
        sa = (P - mu) @ v
        if random.random() < 0.5:                              # BESIDE the column, along its long edge
            shift_t = random.uniform(ta.min(), ta.max()) - tq.mean()
            if random.random() < 0.5:
                shift_v = sa.max() + gap - lat.min()
            else:
                shift_v = sa.min() - gap - lat.max()
        else:                                                  # past one END (two pieces either side of a strap)
            shift_t = ta.max() + gap - tq.min() if random.random() < 0.5 else ta.min() - gap - tq.max()
            shift_v = -lat.mean() + random.uniform(-0.3, 0.3) * wa
        Qn = np.round(Q + u * shift_t + v * shift_v).astype(np.int64)
        src = Q.astype(np.int64)
        h, w = lab.shape
        ok = (Qn[:, 0] >= 0) & (Qn[:, 0] < w) & (Qn[:, 1] >= 0) & (Qn[:, 1] < h)
        if ok.sum() < 200:
            return img, lab, lut
        Qn, src = Qn[ok], src[ok]
        if (lab[Qn[:, 1], Qn[:, 0]] == a).mean() > 0.02:      # must not cover the complete column itself
            return img, lab, lut
        img, out, lut = img.copy(), lab.copy(), list(lut)
        hit = np.unique(lab[Qn[:, 1], Qn[:, 0]])
        img[Qn[:, 1], Qn[:, 0]] = img[src[:, 1], src[:, 0]]
        new = len(lut)
        out[Qn[:, 1], Qn[:, 0]] = new
        lut.append(1.0)
        for j in hit:
            if j > 0 and j < new and (out == j).sum() < 0.95 * (lab == j).sum():
                lut[j] = 1.0
        return img, out, np.array(lut, np.float32)

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
            if kind == "inst" and self.occlude_p and random.random() < self.occlude_p:
                img, lab, lut = self._occlude(img, lab, lut)
            if kind == "inst" and self.paste_p and random.random() < self.paste_p:
                img, lab, lut = self._paste(img, lab, lut)
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
        if self.turn_p and random.random() < self.turn_p:
            code = random.choice((cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE))
            img, lab = cv2.rotate(np.ascontiguousarray(img), code), cv2.rotate(np.ascontiguousarray(lab), code)
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
        lab, _, _, _ = letterbox(lab, SIZE, 0, LABEL_INTERP)
        valid = kind == "inst"
        fl = flow_targets(lab) if valid else np.zeros((2, SIZE, SIZE), np.float32)
        cls = lut[np.clip(lab, 0, len(lut) - 1)] if valid else np.zeros((SIZE, SIZE), np.float32)
        seam = seam_map(lab, self.seam_r) if valid and self.seam_r else np.zeros((SIZE, SIZE), np.float32)
        return (to_tensor(img), torch.from_numpy((lab > 0).astype(np.float32))[None], torch.from_numpy(fl),
                torch.tensor(1.0 if valid else 0.0), torch.from_numpy(cls.astype(np.float32))[None],
                torch.from_numpy(seam)[None])


def seam_map(lab: np.ndarray, r: int) -> np.ndarray:
    """1 where a pixel lies within r px of two DIFFERENT columns (the thin gap between touching columns and both
    rims), else 0. The fused-column failure is the model smoothing across exactly these pixels."""
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    big = np.int32(1 << 20)
    hi = cv2.dilate(lab.astype(np.float32), k)                                  # largest id nearby
    lo = -cv2.dilate(-np.where(lab > 0, lab, big).astype(np.float32), k)        # smallest id nearby
    return ((hi > 0) & (lo < big) & (hi != lo)).astype(np.float32)


def build_val(items: list[dict]) -> list[dict]:
    """Letterboxed tensors + GT for evaluation (no augmentation)."""
    out = []
    for it in items:
        img = cv2.imread(it["image"], cv2.IMREAD_COLOR)
        lab, lut = labelme.column_labels(it["json"], hw=img.shape[:2])
        for i in it.get("flip_good", ()):      # label correction (--class-override): train as complete
            lut[i] = 0.0
        imgL, _, _, _ = letterbox(img, SIZE, 0, cv2.INTER_LINEAR)
        labL, _, _, _ = letterbox(lab, SIZE, 0, LABEL_INTERP)
        out.append({**it, "x": to_tensor(imgL), "gt": labL, "lut": lut})
    return out
