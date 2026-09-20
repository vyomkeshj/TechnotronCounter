"""Does ignoring small band blobs (relative to the crop's median blob) fix miscounts?

    python -m tools.test_relative_blob_floor <run> [<run> ...]

Counts every labelled crop (all types) with centre input under: the shipped fixed
floor (6 px) and relative floors of 10/20/30/40 % of the median blob area.
Reports exact counts per type AND how many crops each rule breaks that were right.
"""
import glob
import os
import sys

import cv2
import numpy as np
import torch

import paths
from common import labelme
from common.unet import load_checkpoint
from edges.counter import MIN_AREA, THR, band, infer
from edges.dataset import fit_width, tight

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RULES = [("fixed 6px", None), ("rel 10%", 0.10), ("rel 20%", 0.20), ("rel 30%", 0.30), ("rel 40%", 0.40)]


def blob_areas(fg):
    a, b = band(fg)
    n, cc = cv2.connectedComponents(fg[:, a:b].astype(np.uint8), connectivity=8)
    return np.bincount(cc.ravel())[1:] if n > 1 else np.zeros(0, int)


def count(areas, rel):
    areas = areas[areas >= MIN_AREA]
    if rel is None or areas.size == 0:
        return int(areas.size)
    return int((areas >= rel * np.median(areas)).sum())


def main():
    crops = []
    args = sys.argv[1:]
    if args and args[0] == "--glob":                      # --glob <pattern> <expected> run ...
        pattern, expected, args = args[1], int(args[2]), args[3:]
        for p in sorted(glob.glob(pattern)):
            img = fit_width(*tight(cv2.imread(p)), "center")[0]
            crops.append((os.path.basename(os.path.dirname(p)), img, expected))
    else:
        for js in sorted(glob.glob(os.path.join(paths.EDGES_DATA, "*", "*.json"))):
            img, gt = fit_width(*tight(cv2.imread(js[:-5] + ".png"), labelme.edge_labels(js)), "center")
            crops.append((os.path.basename(os.path.dirname(js)), img, len(np.unique(gt)) - 1))
    for run in args:
        net = load_checkpoint(os.path.join(paths.RUNS, run, "last.pt"), DEVICE)
        areas = [blob_areas(infer(net, img, DEVICE)[0] > THR) for _, img, _ in crops]
        base = [count(a, None) == n for a, (_, _, n) in zip(areas, crops)]
        print(f"\n{run}")
        for name, rel in RULES:
            ok = [count(a, rel) == n for a, (_, _, n) in zip(areas, crops)]
            per = {t: f"{sum(o for o, c in zip(ok, crops) if c[0] == t)}/{sum(1 for c in crops if c[0] == t)}"
                   for t in sorted({c[0] for c in crops})}
            broke = sum(1 for o, b in zip(ok, base) if b and not o)
            fixed = sum(1 for o, b in zip(ok, base) if o and not b)
            print(f"  {name:10s} exact {sum(ok)}/{len(ok)} {per} | fixed {fixed}, broke {broke}")


if __name__ == "__main__":
    main()
