"""SCATTER ("frankenstein") scenes: edge patches thrown onto black at random
position, zoom and angle.

The question this exists to answer: can the model be made to detect a sheet edge from
its LOCAL APPEARANCE ALONE -- no column axis, no gravity, no "the stack runs down the
middle", no learned pitch anchored to the frame? Every positional and orientational
cue the production dataset quietly provides is destroyed on purpose.

    scene = black 256x256 canvas
          + N patches cut from labelled column crops
          + each independently scaled (zoom), rotated (any angle) and placed anywhere,
            allowed to run off the canvas edge, never overlapping another patch

TARGET IS FOREGROUND ONLY. Counting is meaningless here (a scene has no column and no
sheet total), so the objective is exactly what the user specified: where there is an
edge, the mask says edge; everywhere else, black.

THE PROPERTY THAT MAKES THIS A REAL TEST: the patch image carries the WHOLE column
strip -- sheet faces, the dark grooves between them, the stack cap -- while the label
marks only the labelled edge band. So "non-black pixel" is NOT the answer; the model
has to learn the band's own appearance. A model that cheats by segmenting "anything
lit" scores terribly.

Instance ids are kept distinct across patches (offset per patch) so a flow target can
still be built if wanted; the headline metric is fg IoU.
"""
from __future__ import annotations

import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from common.flows import flow_targets
from common.imageio import degrade, photometric, to_tensor
from edges.dataset import load, tight

cv2.setNumThreads(0)
SIZE = 256


def take_patch(img: np.ndarray, lab: np.ndarray, rng: random.Random):
    """A random rectangle of a tight-cropped column: a handful of sheets, any part."""
    H, W = lab.shape
    ph = rng.randint(24, max(25, min(H, 200)))
    pw = rng.randint(max(16, int(W * 0.35)), W)
    y0 = rng.randint(0, max(0, H - ph))
    x0 = rng.randint(0, max(0, W - pw))
    return img[y0:y0 + ph, x0:x0 + pw].copy(), lab[y0:y0 + ph, x0:x0 + pw].copy()


def warp(img: np.ndarray, lab: np.ndarray, rng: random.Random,
         zoom=(0.35, 2.2), rot=(-180.0, 180.0)):
    """Independent zoom (with mild anisotropy) then free rotation, canvas expanded."""
    s = rng.uniform(*zoom)
    sx, sy = s * rng.uniform(0.85, 1.18), s * rng.uniform(0.85, 1.18)
    nh, nw = max(2, int(lab.shape[0] * sy)), max(2, int(lab.shape[1] * sx))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    lab = cv2.resize(lab, (nw, nh), interpolation=cv2.INTER_NEAREST)
    ang = rng.uniform(*rot)
    M = cv2.getRotationMatrix2D((nw / 2, nh / 2), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    ow, oh = int(nh * sin + nw * cos) + 1, int(nh * cos + nw * sin) + 1
    M[0, 2] += ow / 2 - nw / 2
    M[1, 2] += oh / 2 - nh / 2
    img = cv2.warpAffine(img, M, (ow, oh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    lab = cv2.warpAffine(lab, M, (ow, oh), flags=cv2.INTER_NEAREST, borderValue=0)
    return img, lab


def scene(items: list[dict], rng: random.Random, size: int = SIZE, n_pieces=(2, 6),
          zoom=(0.35, 2.2), rot=(-180.0, 180.0), gap: int = 3, cache: dict | None = None):
    """(BGR uint8 scene, int32 instance labels). Patches never overlap each other."""
    canvas = np.zeros((size, size, 3), np.uint8)
    label = np.zeros((size, size), np.int32)
    busy = np.zeros((size, size), np.uint8)            # content already placed, dilated
    k = rng.randint(*n_pieces)
    nxt = 1
    for _ in range(k):
        it = rng.choice(items)
        if cache is not None:
            if it["image"] not in cache:
                cache[it["image"]] = tight(*load(it))
            src_img, src_lab = cache[it["image"]]
        else:
            src_img, src_lab = tight(*load(it))
        pi, pl = warp(*take_patch(src_img, src_lab, rng), rng, zoom, rot)
        ph, pw = pl.shape
        if ph >= size * 1.8 or pw >= size * 1.8:
            continue
        content = pi.max(axis=2) > 0
        if content.sum() < 40:
            continue
        for _try in range(24):                          # rejection-sample a free spot
            y = rng.randint(-ph // 3, size - ph + ph // 3)
            x = rng.randint(-pw // 3, size - pw + pw // 3)
            ys0, xs0 = max(0, -y), max(0, -x)
            ys1, xs1 = min(ph, size - y), min(pw, size - x)
            if ys1 <= ys0 or xs1 <= xs0:
                continue
            dy0, dx0 = y + ys0, x + xs0
            sub = content[ys0:ys1, xs0:xs1]
            if sub.sum() < 40:
                continue
            if busy[dy0:dy0 + sub.shape[0], dx0:dx0 + sub.shape[1]][sub].any():
                continue
            dst_i = canvas[dy0:dy0 + sub.shape[0], dx0:dx0 + sub.shape[1]]
            dst_l = label[dy0:dy0 + sub.shape[0], dx0:dx0 + sub.shape[1]]
            dst_i[sub] = pi[ys0:ys1, xs0:xs1][sub]
            pls = pl[ys0:ys1, xs0:xs1]
            ids = pls > 0
            dst_l[ids] = pls[ids] + nxt
            nxt += int(pls.max()) + 1
            m = np.zeros(busy.shape, np.uint8)
            m[dy0:dy0 + sub.shape[0], dx0:dx0 + sub.shape[1]][sub] = 1
            busy |= cv2.dilate(m, np.ones((gap * 2 + 1,) * 2, np.uint8))
            break
    # renumber 1..N so the flow target sees a dense id range
    out = np.zeros_like(label)
    for new, old in enumerate(sorted(set(np.unique(label)) - {0}), 1):
        out[label == old] = new
    return canvas, out


class ScatterDataset(Dataset):
    """Scenes generated on the fly; `length` is an epoch's worth of samples."""

    def __init__(self, items: list[dict], length: int = 20000, seed: int = 0, **kw):
        self.items, self.length, self.kw, self.seed = items, length, kw, seed
        self._cache: dict = {}

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = random.Random((self.seed * 1_000_003 + idx) & 0xFFFFFFFF)
        img, lab = scene(self.items, rng, cache=self._cache, **self.kw)
        img = degrade(photometric(img))
        return (to_tensor(img), torch.from_numpy((lab > 0).astype(np.float32))[None],
                torch.from_numpy(flow_targets(lab, aniso=True)))
