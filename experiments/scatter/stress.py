"""Counting under IMPERFECT conditions: tilt, zoom, and a column filling half the frame.

    python -m experiments.scatter.stress --runs edges_all3_s7 scatter_mix50

Clean straightened crops rank nothing -- every candidate scores ~100 % (the standing
"a saturated benchmark ranks nothing" rule), so this measures MARGIN.

WHOLE CROPS ONLY, and that is deliberate. A first version cut 12-sheet windows and
scored 46 % exact at ZERO rotation, which made every perturbation delta unreadable.
The cause is in the data, not the code: these sheets are tilted crescents, so ANY
horizontal cut crosses two or three of them and "how many sheets are in this window"
has no exact answer. On a whole crop the count is known exactly from the polygons.

Conditions (each preserves the true count):
  rot<θ>    rotate the whole crop, canvas expanded so nothing leaves the frame
  scale<s>  resize both axes
  pad2x     the column placed in the corner of a canvas twice as wide and tall, i.e.
            most of the frame is empty -- "only half the image has columns"

Counting rules, both on the same foreground:
  band  the SHIPPED rule: components inside the centre 50 % of the foreground WIDTH.
        It assumes the column is upright, so a tilt rotates the very axis it measures.
  cc    every component of the whole foreground, same area filters, NO axis assumption.

The model is run fully convolutionally on the perturbed image -- no tight crop, no
centre-74 fit -- because those preprocessing steps would undo the perturbation being
tested (tight() deletes padding; the centre fit re-imposes an axis).
"""
from __future__ import annotations

import argparse
import collections
import os

import cv2
import numpy as np
import torch

import paths
from common.unet import load_checkpoint
from edges.counter import MIN_AREA, REL_FLOOR, THR, band_count, infer
from edges.dataset import discover, load, sync_split, tight

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANGLES = (0, 3, 5, 10, 20, 45, 90)
SCALES = (0.6, 1.5)


def cc_count(fg: np.ndarray, min_area: int = MIN_AREA, rel_floor: float = REL_FLOOR) -> int:
    """Axis-free: connected components of the WHOLE foreground, same filters as band."""
    n, cc = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)
    if n <= 1:
        return 0
    areas = np.bincount(cc.ravel())[1:]
    areas = areas[areas >= min_area]
    if rel_floor and areas.size:
        areas = areas[areas >= rel_floor * np.median(areas)]
    return int(areas.size)


def rotate(img, ang):
    if ang % 360 == 0:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    ow, oh = int(h * sin + w * cos) + 1, int(h * cos + w * sin) + 1
    M[0, 2] += ow / 2 - w / 2
    M[1, 2] += oh / 2 - h / 2
    return cv2.warpAffine(img, M, (ow, oh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))


def conditions(img):
    yield "clean", img
    for a in ANGLES[1:]:
        yield f"rot{a}", rotate(img, a)
    for s in SCALES:
        yield f"scale{s}", cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    big = np.zeros((h * 2, w * 2, 3), img.dtype)
    big[:h, :w] = img
    yield "pad2x", big


@torch.no_grad()
def run(net, crops):
    res = collections.defaultdict(lambda: {"band": [], "cc": []})
    for img, gt in crops:
        for name, x in conditions(img):
            fg = infer(net, x, DEVICE)[0] > THR
            res[name]["band"].append(band_count(fg) - gt)
            res[name]["cc"].append(cc_count(fg) - gt)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--max-crops", type=int, default=16)
    args = ap.parse_args()

    items, _ = discover()
    sp = sync_split(items, write=False)
    val_items = [i for i in items if i["frame"] in set(sp["val"])]
    crops = []
    for it in val_items[: args.max_crops]:
        img, lab = tight(*load(it))
        crops.append((img, int(len([i for i in np.unique(lab) if i]))))
    print(f"{len(crops)} whole val crops, exact GT from the polygons "
          f"(counts {sorted({g for _i, g in crops})})\n")

    order = ["clean"] + [f"rot{a}" for a in ANGLES[1:]] + [f"scale{s}" for s in SCALES] + ["pad2x"]
    table = {}
    for name in args.runs:
        p = os.path.join(paths.RUNS, name, "last.pt")
        net = load_checkpoint(p if os.path.exists(p) else
                              os.path.join(paths.RUNS, name, "stage2", "last.pt"), DEVICE)
        table[name] = run(net, crops)

    for rule in ("band", "cc"):
        print(f"=== {rule} counter — exact % (MAE)")
        print(f"  {'condition':<10}" + "".join(f"{n:>26}" for n in args.runs))
        for k in order:
            row = f"  {k:<10}"
            for n in args.runs:
                e = np.array(table[n][k][rule], float)
                row += f"{(e == 0).mean():>18.0%} ({np.abs(e).mean():.2f})"
            print(row)
        print()


if __name__ == "__main__":
    main()
