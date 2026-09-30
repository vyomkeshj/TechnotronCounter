"""Label-free check on frames NO model has ever trained on: every good_images frame of
type2/3/10 that has no column label (so it was never a training frame for either model;
pseudo labels were also only made from labelled column frames). Count vs product count.

    python -m tools.unseen_check [--models <dir>] [--name tag]

Expected with the shipped models: type2 ~73 % exact per frame, ~75 % with a 5-frame vote
(about 100 columns; most misses are +-1, many in the steep S-curve session 133033).

Per type: columns counted exactly (single frame, and with a 5-frame vote per session).
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import re
import sys

import cv2
import numpy as np

import paths
from tools.evalkit import expected, sheet_counter

GI = os.path.join(paths.DATA, "unlabelled")          # frames with no label of any kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="models folder (default: inference/models)")
    ap.add_argument("--name", default="shipped")
    args = ap.parse_args()
    sc = sheet_counter(args.models)
    from sheetcount.pipeline import vote
    labelled = {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(paths.COLUMNS_DATA, "**", "*.json"), recursive=True)}
    rows = []
    by_ses = collections.defaultdict(list)
    for typ in ("type2", "type3", "type10"):
        for p in sorted(glob.glob(os.path.join(GI, typ, "*.jpg"))):
            stem = os.path.basename(p)[:-4]
            if stem in labelled:
                continue
            img = cv2.imread(p)
            cols = [c for c in sc.process(img, typ, masks=False) if not c.cropped]
            exp = expected(stem)
            data = [(float(np.where(c.mask)[1].mean()), float(np.where(c.mask)[0].mean()), c.count) for c in cols]
            by_ses[(typ, stem.split("_")[1])].append((stem, exp, data))
    out = collections.defaultdict(lambda: [0, 0, 0])
    for (typ, ses), items in by_ses.items():
        voted = vote([d for _, _, d in items], 5)
        for (stem, exp, data), v in zip(items, voted):
            for (cx, cy, n), vn in zip(data, v):
                rows.append({"type": typ, "session": ses, "frame": stem, "cx": round(cx, 1), "count": n,
                             "voted": vn, "expected": exp})
                for k in (typ, f"{typ} {ses}"):
                    out[k][0] += n == exp; out[k][1] += vn == exp; out[k][2] += 1
    d = os.path.join(paths.RUNS, "_unseen_check", args.name)
    os.makedirs(d, exist_ok=True)
    if rows:
        with open(os.path.join(d, "columns.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for k in sorted(out):
        a, b, n = out[k]
        print(f"{k:16} columns {n:4}  exact {a:4} ({a / n:.1%})  voted {b:4} ({b / n:.1%})")


if __name__ == "__main__":
    main()
