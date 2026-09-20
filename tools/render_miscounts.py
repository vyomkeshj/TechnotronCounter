"""Render the crops an edge model miscounts, with the culprit blobs marked.

    python -m tools.render_miscounts <run> <type> <out_dir> [mode]

Panels: input | counted band blobs | human edge labels. Blobs are matched to human
edges inside the band: a blob touching no human edge is SPURIOUS (red box); a human
edge covered by >1 blob is SPLIT (magenta box); a blob covering >1 human edge is a
MERGE (cyan box).
"""
import glob
import os
import sys

import cv2
import numpy as np
import torch

import paths
from common import labelme
from common.imageio import colorize
from common.unet import load_checkpoint
from edges.counter import THR, band, infer
from edges.counter import MIN_AREA
from edges.dataset import fit_width, tight

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    run, typ, out = sys.argv[1], sys.argv[2], sys.argv[3]
    mode = sys.argv[4] if len(sys.argv) > 4 else "center"
    os.makedirs(out, exist_ok=True)
    net = load_checkpoint(os.path.join(paths.RUNS, run, "last.pt"), DEVICE)
    n_bad = 0
    for js in sorted(glob.glob(os.path.join(paths.EDGES_DATA, typ, "*.json"))):
        raw = cv2.imread(js[:-5] + ".png")
        img, gt = fit_width(*tight(raw, labelme.edge_labels(js)), mode)
        prob, _ = infer(net, img, DEVICE)
        fg = prob > THR
        a, b = band(fg)
        n, cc = cv2.connectedComponents(fg[:, a:b].astype(np.uint8), connectivity=8)
        blobs = np.zeros(fg.shape, np.int32)
        sizes = np.bincount(cc.ravel())
        keep = [i for i in range(1, n) if sizes[i] >= MIN_AREA]
        for k, i in enumerate(keep, 1):
            blobs[:, a:b][cc == i] = k
        n_gt = len(np.unique(gt)) - 1
        if len(keep) == n_gt:
            continue
        n_bad += 1
        gtb = np.zeros_like(gt)
        gtb[:, a:b] = gt[:, a:b]
        boxes = []
        for k in range(1, len(keep) + 1):
            m = blobs == k
            ids = [i for i in np.unique(gtb[m]) if i]
            ys, xs = np.where(m)
            if not ids:
                boxes.append(((0, 0, 255), xs.min(), ys.min(), xs.max(), ys.max(), "spurious"))
            elif len(ids) > 1:
                boxes.append(((255, 255, 0), xs.min(), ys.min(), xs.max(), ys.max(), "merge"))
        for i in [i for i in np.unique(gtb) if i]:
            ks = [k for k in np.unique(blobs[gtb == i]) if k]
            if len(ks) > 1:
                ys, xs = np.where(gtb == i)
                boxes.append(((255, 0, 255), xs.min(), ys.min(), xs.max(), ys.max(), "split"))
        dim = (img * 0.35).astype(np.uint8)
        pb, pg = dim.copy(), dim.copy()
        pb[blobs > 0] = colorize(blobs)[blobs > 0][:, ::-1]
        pg[gt > 0] = colorize(gt)[gt > 0][:, ::-1]
        for p in (pb, pg):
            cv2.line(p, (a, 0), (a, p.shape[0] - 1), (255, 255, 255), 1)
            cv2.line(p, (b - 1, 0), (b - 1, p.shape[0] - 1), (255, 255, 255), 1)
        Z = 4
        panels = [cv2.resize(p, None, fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST) for p in (img, pb, pg)]
        for col, x0, y0, x1, y1, _ in boxes:
            for p in panels:
                cv2.rectangle(p, (int(x0 * Z) - 4, int(y0 * Z) - 4), (int(x1 * Z) + 4, int(y1 * Z) + 4), col, 3)
        ys = np.flatnonzero(img.max(axis=(1, 2)) > 0)
        y0, y1 = max(0, ys[0] * Z - 10), ys[-1] * Z + 10
        t = np.hstack([cv2.copyMakeBorder(p[y0:y1], 0, 0, 0, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                       for p in panels])
        kinds = ", ".join(sorted({bx[5] for bx in boxes})) or "unmatched"
        head = np.full((50, t.shape[1], 3), 255, np.uint8)
        cv2.putText(head, f"{os.path.basename(js)[16:-5]}  count {len(keep)} / human {n_gt}  ({kinds})",
                    (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        cv2.imwrite(os.path.join(out, os.path.basename(js)[:-5] + ".jpg"), np.vstack([head, t]))
        print(os.path.basename(js)[16:-5], "count", len(keep), "human", n_gt, kinds)
    print(n_bad, "miscounted ->", out)


if __name__ == "__main__":
    main()
