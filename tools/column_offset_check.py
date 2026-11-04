"""Is the column mask shifted? On the held-out labelled frames, every matched column's error pixels split by
side across the column (its minor axis, pointed down the image): spill past the upper/lower side and missed
upper/lower side. A centred mask has upper ~ lower; the models trained with nearest-resized labels spill
5-15x more past the LOWER side (the half-pixel label offset, 2026-11-04).

    python -m tools.column_offset_check [--models <dir>]
"""
from __future__ import annotations

import argparse
import collections
import re

import cv2
import numpy as np

from columns.dataset import discover, sync_split
from common import labelme
from tools.evalkit import iou, sheet_counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None)
    args = ap.parse_args()
    sc = sheet_counter(args.models)
    items = discover()
    val = set(sync_split(items, write=False)["val"])
    tot = collections.defaultdict(lambda: np.zeros(5))
    for it in items:
        if it["stem"] not in val:
            continue
        m = re.search(r"type\d+", it["stem"])
        typ = m.group(0) if m else "older"
        img = cv2.imread(it["image"])
        H, W = img.shape[:2]
        lab, _ = labelme.column_labels(it["json"], hw=(H, W))
        cols = sc.segment_columns(img, "type3")
        for g in range(1, lab.max() + 1):
            gm = lab == g
            if gm.sum() < 2000:
                continue
            c = max(cols, key=lambda c: iou(c.mask, gm), default=None)
            if c is None or iou(c.mask, gm) < 0.5:
                continue
            ys, xs = np.nonzero(gm)
            P = np.stack([xs, ys], 1).astype(np.float32)
            mu = P.mean(0)
            v = np.linalg.svd(P - mu, full_matrices=False)[2][1]
            if v[1] < 0:
                v = -v

            def side(mask):
                yy, xx = np.nonzero(mask)
                t = (np.stack([xx, yy], 1) - mu) @ v
                return (t < 0).sum(), (t >= 0).sum()
            eu, el = side(c.mask & ~gm)
            mu_, ml = side(gm & ~c.mask)
            for k in (typ, "ALL"):
                tot[k] += [eu, el, mu_, ml, gm.sum()]
    print(f"{'type':7s} {'spill upper':>12s} {'spill lower':>12s} {'miss upper':>11s} {'miss lower':>11s}   "
          f"(% of column area)  lower/upper spill ratio")
    for k in sorted(tot, key=lambda k: (k == "ALL", k)):
        eu, el, mu_, ml, a = tot[k]
        print(f"{k:7s} {100 * eu / a:11.2f}% {100 * el / a:11.2f}% {100 * mu_ / a:10.2f}% {100 * ml / a:10.2f}%   "
              f"{el / max(eu, 1):5.1f}x")


if __name__ == "__main__":
    main()
