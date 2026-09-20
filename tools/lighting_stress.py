"""Count robustness under lighting the model never trained on (val crops, centre input).

    python -m tools.lighting_stress <run> [<run> ...]

The perturbations are deliberately NOT the training augmentation (edges.dataset.
local_lighting), so a model cannot pass by memorising its own aug:
  spots     round glare spots, NOT sheet-aligned, applied to grooves too (x0.6)
  sidelight strong one-sided illumination ramp (x0.45 -> x1.5 across the column)
  blotch    large dark soft blotches (occluder shadows)
Each val crop gets 4 fixed-seed variants per kind. Scored: exact count, with the
shipped fixed 6 px blob floor and with a 25 % relative floor.
"""
import os
import sys

import cv2
import numpy as np
import torch

import paths
from common.unet import load_checkpoint
from edges.counter import MIN_AREA, THR, band, infer
from edges.dataset import build_val, discover, sync_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KINDS = ("clean", "spots", "sidelight", "blotch")


def perturb(img, kind, seed):
    if kind == "clean":
        return img
    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    content = (img.max(axis=2) > 0)[..., None]
    if kind == "spots":
        g = np.zeros((h, w), np.float32)
        for _ in range(rng.randint(2, 6)):
            cy, cx, r = rng.uniform(0, h), rng.uniform(0, w), rng.uniform(6, 22)
            g += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
        f = f + np.clip(g, 0, 1)[..., None] * rng.uniform(90, 170) * 0.6 / 0.6
    elif kind == "sidelight":
        t = xx / max(w - 1, 1) if rng.rand() < 0.5 else yy / max(h - 1, 1)
        lo, hi = (0.45, 1.5) if rng.rand() < 0.5 else (1.5, 0.45)
        f = f * (lo + (hi - lo) * t)[..., None]
    elif kind == "blotch":
        g = np.zeros((h, w), np.float32)
        for _ in range(rng.randint(1, 4)):
            cy, cx, r = rng.uniform(0, h), rng.uniform(0, w), rng.uniform(12, 40)
            g += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
        f = f * (1 - 0.6 * np.clip(g, 0, 1))[..., None]
    return (np.clip(f, 0, 255) * content).astype(np.uint8)


def count(fg, rel):
    a, b = band(fg)
    n, cc = cv2.connectedComponents(fg[:, a:b].astype(np.uint8), connectivity=8)
    areas = np.bincount(cc.ravel())[1:] if n > 1 else np.zeros(0, int)
    areas = areas[areas >= MIN_AREA]
    if rel and areas.size:
        areas = areas[areas >= rel * np.median(areas)]
    return int(areas.size)


def main():
    items, _ = discover()
    split = sync_split(items, write=False)
    val = build_val([it for it in items if it["frame"] in set(split["val"])], "center")
    print(f"{len(val)} val crops x {len(KINDS) - 1} lighting kinds x 4 seeds (+ clean)")
    for run in sys.argv[1:]:
        net = load_checkpoint(os.path.join(paths.RUNS, run, "last.pt"), DEVICE)
        res = {}
        for kind in KINDS:
            for seed in range(1 if kind == "clean" else 4):
                for j, it in enumerate(val):
                    fg = infer(net, perturb(it["img"], kind, 1000 * seed + j), DEVICE)[0] > THR
                    for rel in (None, 0.25):
                        key = (kind, rel, it["type"])
                        ok, tot = res.get(key, (0, 0))
                        res[key] = (ok + int(count(fg, rel) == it["n_gt"]), tot + 1)
        print(f"\n{run}")
        for rel in (None, 0.25):
            cells = []
            for kind in KINDS:
                parts = [f"{t[4:]} {res[(kind, rel, t)][0]}/{res[(kind, rel, t)][1]}"
                         for t in sorted({it['type'] for it in val})]
                cells.append(f"{kind}: " + ", ".join(parts))
            print(f"  floor {'25%' if rel else '6px':4s} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
