"""Render scatter scenes so a human can check the labels follow the pieces.

    python -m experiments.scatter.preview [--n 1] [--seed 0] [--out path.png]

Three panels per scene: the training input, the target mask, and the two overlaid.
The overlay is the one that matters -- the label must cover the sheet bands and NOT
the grooves between them, on every piece, at every zoom and angle.
"""
from __future__ import annotations

import argparse
import os
import random

import cv2
import numpy as np

from edges.dataset import discover
from experiments.scatter.dataset import scene

PANEL = ("training input (what the model sees)",
         "target (where there is an edge)",
         "overlay -- label must miss the grooves")


def strip(img: np.ndarray, lab: np.ndarray, zoom: int = 3) -> np.ndarray:
    fg = (lab > 0).astype(np.uint8)
    tgt = cv2.cvtColor(fg * 255, cv2.COLOR_GRAY2BGR)
    ov = img.copy()
    red = np.zeros_like(img)
    red[..., 2] = 255
    m = fg.astype(bool)
    ov[m] = (0.45 * ov[m] + 0.55 * red[m]).astype(np.uint8)
    cells = [cv2.resize(p, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
             for p in (img, tgt, ov)]
    h, w = cells[0].shape[:2]
    out = np.full((h + 34, w * 3 + 24, 3), 20, np.uint8)
    for i, c in enumerate(cells):
        x = i * (w + 12)
        out[34:34 + h, x:x + w] = c
        cv2.rectangle(out, (x - 1, 33), (x + w, 34 + h), (70, 70, 70), 1)
        cv2.putText(out, PANEL[i], (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (235, 235, 235), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zoom", type=int, default=3)
    ap.add_argument("--out", default="scatter_preview.png")
    args = ap.parse_args()

    items, _ = discover()
    rows = []
    for i in range(args.n):
        rng = random.Random(args.seed + i)
        img, lab = scene(items, rng)
        n = int(lab.max())
        cov = 100.0 * (lab > 0).sum() / lab.size
        lit = 100.0 * (img.max(axis=2) > 0).sum() / lab.size
        print(f"scene {i}: {n} labelled sheets | lit pixels {lit:.1f}% | labelled {cov:.1f}% "
              f"-> a 'segment anything lit' cheat would score IoU {cov / max(lit, 1e-6):.2f}")
        rows.append(strip(img, lab, args.zoom))
    out = rows[0] if len(rows) == 1 else np.vstack(rows)
    cv2.imwrite(args.out, out)
    print(f"-> {os.path.abspath(args.out)}  {out.shape[1]}x{out.shape[0]}")


if __name__ == "__main__":
    main()
