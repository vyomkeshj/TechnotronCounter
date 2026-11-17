"""Column FUSION measured densely on the held-out frames (data/columns/splits.json "val"): every pair of labelled
columns that touch (within SEAM_R px at the model's 256 scale) is one trial. Per type:
  pairs        touching labelled pairs
  merged       pairs whose two columns land mostly (>= 25 % each) in ONE predicted column
  merged_cpl   merged pairs that include a COMPLETE column (that column is lost: miscounted or dropped)
  gap_fg       share of the labelled gap between touching columns that is painted as column (lower = cleaner seams)
Counting fused masks gives a handful of events; this gives hundreds of trials from the same labels.

    python -m tools.seam_check [--models <dir>] [--split val|train]
"""
from __future__ import annotations

import argparse
import collections
import json
import re

import cv2
import numpy as np

import paths
from columns.dataset import discover
from common import labelme
from tools.evalkit import sheet_counter

SEAM_R = 2          # px at 256 (the model's input scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None)
    ap.add_argument("--split", default="val", choices=["val", "train"])
    args = ap.parse_args()
    sc = sheet_counter(args.models)
    keep = set(json.load(open(paths.COLUMNS_SPLIT))[args.split])
    st = collections.defaultdict(collections.Counter)
    for it in discover():
        if it["stem"] not in keep:
            continue
        m = re.search(r"type\d+", it["stem"])
        typ = m.group(0) if m else it["group"]
        img = cv2.imread(it["image"])
        H, W = img.shape[:2]
        lab, lut = labelme.column_labels(it["json"], hw=(H, W))
        r = max(1, int(round(SEAM_R * max(H, W) / 256)))
        k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
        cols = sc.segment_columns(img, typ if typ.startswith("type") else "type3")
        pred = np.zeros((H, W), np.int32)
        for j, c in enumerate(cols, 1):
            pred[c.mask] = j
        ids = [i for i in range(1, int(lab.max()) + 1) if (lab == i).any()]
        dil = {i: cv2.dilate((lab == i).astype(np.uint8), k) > 0 for i in ids}
        best = {}
        for i in ids:
            g = lab == i
            p, n = np.unique(pred[g], return_counts=True)
            best[i] = [(int(a), int(b) / g.sum()) for a, b in zip(p, n) if a > 0]
        s = st[typ]
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = ids[x], ids[y]
                touch = dil[a] & dil[b]
                if not touch.any():
                    continue
                s["pairs"] += 1
                gap = touch & (lab == 0)
                s["gap_px"] += int(gap.sum())
                s["gap_fg"] += int((gap & (pred > 0)).sum())
                pa = {j for j, f in best[a] if f >= 0.25}
                pb = {j for j, f in best[b] if f >= 0.25}
                if pa & pb:
                    s["merged"] += 1
                    s["merged_cpl"] += int(lut[a] < 0.5 or lut[b] < 0.5)
        s["frames"] += 1
    print(f"{'type':9s} {'frames':>6s} {'pairs':>6s} {'merged':>7s} {'merged_cpl':>10s} {'gap_fg':>7s}")
    tot = collections.Counter()
    for t in sorted(st, key=lambda t: (len(t), t)):
        s = st[t]
        tot.update(s)
        print(f"{t:9s} {s['frames']:6d} {s['pairs']:6d} {s['merged']:7d} {s['merged_cpl']:10d} "
              f"{100 * s['gap_fg'] / max(1, s['gap_px']):6.1f}%")
    s = tot
    print(f"{'ALL':9s} {s['frames']:6d} {s['pairs']:6d} {s['merged']:7d} {s['merged_cpl']:10d} "
          f"{100 * s['gap_fg'] / max(1, s['gap_px']):6.1f}%")


if __name__ == "__main__":
    main()
