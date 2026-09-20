"""Render crops whose count differs from an expected value, with the culprit marked.

    python -m tools.render_count_failures <run> <glob> <expected> <out_dir>

Marks, inside the counted band: RED = blob far smaller than the crop's median blob
(the usual over-count culprit); YELLOW = a gap between consecutive blobs wider than
1.6x the median pitch (the usual under-count culprit -- two sheets merged).
"""
import glob
import os
import sys

import cv2
import numpy as np
import torch

import paths
from common.imageio import colorize
from common.unet import load_checkpoint
from edges.counter import MIN_AREA, REL_FLOOR, THR, band, band_count, infer
from edges.dataset import model_input

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    run, pattern, expected, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    os.makedirs(out, exist_ok=True)
    net = load_checkpoint(f"{paths.RUNS}/{run}/last.pt", DEVICE)
    for p in sorted(glob.glob(pattern)):
        img = model_input(cv2.imread(p), "center")
        fg = infer(net, img, DEVICE)[0] > THR
        n = band_count(fg)
        if n == expected:
            continue
        a, b = band(fg)
        ncc, cc = cv2.connectedComponents(fg[:, a:b].astype(np.uint8), connectivity=8)
        areas = np.bincount(cc.ravel())[1:]
        kept = [i for i in range(1, ncc) if areas[i - 1] >= MIN_AREA]
        med = np.median([areas[i - 1] for i in kept]) if kept else 0
        kept = [i for i in kept if areas[i - 1] >= REL_FLOOR * med]
        ys = {i: float(np.where(cc == i)[0].mean()) for i in kept}
        order = sorted(kept, key=lambda i: ys[i])
        pitches = np.diff([ys[i] for i in order]) if len(order) > 1 else np.array([1.0])
        medp = float(np.median(pitches)) if pitches.size else 1.0
        lab = np.zeros(fg.shape, np.int32)
        for k, i in enumerate(order, 1):
            lab[:, a:b][cc == i] = k
        vis = cv2.addWeighted(img, 0.4, colorize(lab)[:, :, ::-1], 0.6, 0)
        vis[lab == 0] = (img[lab == 0] * 0.45).astype(np.uint8)
        boxes = []
        for k, i in enumerate(order, 1):                      # small blobs
            if areas[i - 1] < 0.5 * med:
                yy, xx = np.where(lab == k)
                boxes.append(((0, 0, 255), yy.min(), yy.max(), "small blob"))
        for j, d in enumerate(pitches):                       # wide gaps
            if d > 1.6 * medp:
                boxes.append(((0, 230, 255), int(ys[order[j]]), int(ys[order[j + 1]]), "wide gap"))
        for col, y0, y1, _ in boxes:
            cv2.rectangle(vis, (a - 3, int(y0) - 3), (b + 2, int(y1) + 3), col, 1)
        ys_all = np.flatnonzero(img.max(axis=(1, 2)) > 0)
        if boxes:
            c = (boxes[0][1] + boxes[0][2]) // 2
            y0, y1 = max(ys_all[0], c - 90), min(ys_all[-1], c + 90)
        else:
            y0, y1 = ys_all[0], ys_all[0] + 180
        t = cv2.resize(np.hstack([img[y0:y1], np.full((y1 - y0, 3, 3), 255, np.uint8), vis[y0:y1]]),
                       None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
        head = np.full((40, t.shape[1], 3), 255, np.uint8)
        cv2.putText(head, f"{os.path.basename(p)[16:-4]}  count {n} / {expected}  ({', '.join(sorted({b[3] for b in boxes})) or 'no flag'})",
                    (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.imwrite(os.path.join(out, os.path.basename(p)[:-4] + ".jpg"), np.vstack([head, t]))
        print(f"{os.path.basename(p)[16:-4]}: count {n}/{expected}, flags {[b[3] for b in boxes]}")


if __name__ == "__main__":
    main()
