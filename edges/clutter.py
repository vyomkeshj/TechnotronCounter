"""Background clutter for the edge model's training windows (--clutter).

A sheet strip's ends often run past the stack (the column outline overshoots): lamps, a green-lit wall, the floor.
The model saw only black there in training (fragment windows float on black), so a bright band past the end
was counted as one more sheet (type7 143956: 24 vs 23). This fills the empty rows above/below the strip's content
with patches cut from TRAINING column frames where no labelled column is, so the label there is truthfully 0.
Placement depends on the window, never on the sheet labels.
"""
from __future__ import annotations

import json
import random

import cv2
import numpy as np

import paths
from columns.dataset import discover
from common import labelme

cv2.setNumThreads(0)

TYPES = ("type1", "type2", "type3", "type4", "type7", "type10")


def build_bank(n: int = 600, seed: int = 0) -> list[np.ndarray]:
    """n background patches (BGR uint8) from training frames, each overlapping labelled columns by < 2 %."""
    rng = random.Random(seed)
    train = set(json.load(open(paths.COLUMNS_SPLIT))["train"])
    frames = [it for it in discover() if it["stem"] in train and it["group"] in TYPES]
    bank, tries = [], 0
    while len(bank) < n and tries < 50 * n:
        tries += 1
        it = rng.choice(frames)
        img = cv2.imread(it["image"])
        H, W = img.shape[:2]
        lab, _ = labelme.column_labels(it["json"], hw=(H, W))
        for _ in range(8):
            w = rng.randint(40, 220)
            h = int(w * rng.uniform(0.5, 3.0))
            if h >= H or w >= W:
                continue
            x, y = rng.randint(0, W - w), rng.randint(0, H - h)
            if (lab[y:y + h, x:x + w] > 0).mean() < 0.02:
                bank.append(img[y:y + h, x:x + w].copy())
                if len(bank) >= n:
                    break
    return bank


def fill(wi: np.ndarray, rows: np.ndarray, x0: int, x1: int, bank: list[np.ndarray], rng=random) -> np.ndarray:
    """Paste a resized bank patch into wi[rows, x0:x1] (rows: bool mask over window rows)."""
    ys = np.flatnonzero(rows)
    if not len(ys) or x1 <= x0:
        return wi
    for y0, y1 in _runs(ys):
        p = rng.choice(bank)
        wi[y0:y1, x0:x1] = cv2.resize(p, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    return wi


def _runs(ys):
    start = prev = ys[0]
    for y in ys[1:]:
        if y != prev + 1:
            yield start, prev + 1
            start = y
        prev = y
    yield start, prev + 1
