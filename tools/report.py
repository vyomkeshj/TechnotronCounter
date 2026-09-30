"""Visual report: the inference package on held-out frames, one render per frame.

    python -m tools.report [--per-type 10] [--seed 0] [--models <dir>]

Frames: random held-out (never trained on) frames per type: the column split's val frames, plus
its excluded buffer frames for type10 (only 6 val frames exist). The blurred 131412_013 is skipped.
Render per frame:
  top     the frame, every column outlined: green = counted, yellow N? = uncertain (retake),
          red = cropped; the count on each counted column
  below   every counted column's straightened strip on its side (column top on the left):
          the image, then its sheet masks, every 10th sheet numbered; header = count / expected
Writes runs/_report/<type>__<frame>.jpg and summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys

import cv2
import numpy as np

import paths
from tools.evalkit import INFERENCE, expected, sheet_counter

OUT = os.path.join(paths.RUNS, "_report")
W = 1600
BLUR = {"20260626_131412_type3_-2_60pieces_013"}


def pick(per_type, seed):
    sp = json.load(open(paths.COLUMNS_SPLIT))
    rng = random.Random(seed)
    out = {}
    for typ in ("type2", "type3", "type10"):
        pool = [s for s in sp["val"] if f"{typ}_" in s]
        if typ == "type10":
            pool += [s for s in sp["excluded"] if f"{typ}_" in s]
        pool = sorted(s for s in pool if s not in BLUR)
        rng.shuffle(pool)
        out[typ] = sorted(pool[:per_type])
    return out


def strip_panel(c, exp, pal):
    x = c.extra["edge_input"]
    lab = c.sheet_masks
    ov = x.copy()
    m = lab > 0
    ov[m] = (0.3 * ov[m] + 0.7 * pal[lab[m] % len(pal)]).astype(np.uint8)
    both = np.hstack([x, np.zeros((x.shape[0], 3, 3), np.uint8), ov])
    both = cv2.rotate(both, cv2.ROTATE_90_COUNTERCLOCKWISE)[::-1]          # column top on the LEFT
    h = both.shape[0]
    scale = W / both.shape[1]
    both = cv2.resize(both, (W, max(1, int(h * scale))), interpolation=cv2.INTER_NEAREST)
    k0 = x.shape[1] + 3                                                     # rows of the mask half
    for k in range(10, int(lab.max()) + 1, 10):                            # number every 10th sheet
        ys, xs = np.where(lab == k)
        if ys.size:
            px = int(ys.mean() * scale)
            cv2.putText(both, str(k), (max(0, px - 8), both.shape[0] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)
    unsure = bool(c.extra.get("uncertain"))
    ok = c.count == exp
    colour = (0, 215, 255) if unsure else ((80, 255, 80) if ok else (80, 80, 255))
    head = np.zeros((24, W, 3), np.uint8)
    txt = f"column {c.extra.get('idx', '')}: {c.count} sheets (expected {exp})"
    if unsure:
        txt += f"  UNCERTAIN: {c.extra.get('why', '')}"
    cv2.putText(head, txt, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    return np.vstack([head, both, np.zeros((8, W, 3), np.uint8)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default=None)
    args = ap.parse_args()
    sc = sheet_counter(args.models)
    sys.path.insert(0, INFERENCE)
    import predict as pr
    pal = np.random.RandomState(7).randint(70, 255, (64, 3)).astype(np.uint8)
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for typ, stems in pick(args.per_type, args.seed).items():
        for stem in stems:
            img = cv2.imread(os.path.join(paths.COLUMNS_DATA, "quality", typ, stem + ".jpg"))
            cols = sc.process(img, typ, masks=True, check=True)
            exp = expected(stem)
            over = pr.render_overlay(img, cols, [c.count for c in cols])
            over = over[340:1080]                                            # the opening
            over = cv2.resize(over, (W, int(over.shape[0] * W / over.shape[1])))
            good = [c for c in cols if not c.cropped]
            for i, c in enumerate(good):
                c.extra["idx"] = i
            counts = [c.count for c in good]
            flags = sum(bool(c.extra.get("uncertain")) for c in good)
            head = np.zeros((34, W, 3), np.uint8)
            cv2.putText(head, f"{typ}  {stem}   counts {counts}   expected {exp}   uncertain {flags}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            panels = [head, over, np.zeros((10, W, 3), np.uint8)]
            panels += [strip_panel(c, exp, pal) for c in good if c.sheet_masks is not None]
            cv2.imwrite(os.path.join(OUT, f"{typ}__{stem}.jpg"), np.vstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 88])
            rows.append({"type": typ, "frame": stem, "expected": exp, "counts": " ".join(map(str, counts)),
                         "exact": sum(n == exp for n in counts), "columns": len(counts), "uncertain": flags,
                         "cropped": sum(c.cropped for c in cols)})
            print(f"{typ} {stem}: {counts} exp {exp} uncertain {flags}", flush=True)
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
