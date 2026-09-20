"""Why does a crop miscount? input | fg probability | counted band blobs, per model.

    python -m tools.blob_diagnostic <out.jpg> <expected> <crop.png> <run> [<run> ...]
"""
import sys

import cv2
import numpy as np
import torch

import paths  # noqa: F401  (keeps the import root consistent)
from common.imageio import colorize
from common.unet import load_checkpoint
from edges.counter import THR, band, band_count, infer
from edges.dataset import model_input

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    out, expected, crop, runs = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
    img = model_input(cv2.imread(crop), "center")
    ys = np.flatnonzero(img.max(axis=(1, 2)) > 0)
    y0, y1 = ys[0], min(ys[-1], ys[0] + 300)
    cols = [img[y0:y1]]
    heads = ["input"]
    for run in runs:
        net = load_checkpoint(f"{paths.RUNS}/{run}/last.pt", DEVICE)
        prob, _ = infer(net, img, DEVICE)
        fg = prob > THR
        a, b = band(fg)
        n, cc = cv2.connectedComponents(fg[:, a:b].astype(np.uint8), connectivity=8)
        blobs = np.zeros(fg.shape, np.int32)
        blobs[:, a:b] = cc
        heat = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        vis = (img * 0.3).astype(np.uint8)
        vis[blobs > 0] = colorize(blobs)[blobs > 0][:, ::-1]
        for p in (heat, vis):
            cv2.line(p, (a, 0), (a, p.shape[0] - 1), (255, 255, 255), 1)
            cv2.line(p, (b - 1, 0), (b - 1, p.shape[0] - 1), (255, 255, 255), 1)
        cols += [heat[y0:y1], vis[y0:y1]]
        heads += [f"{run[6:]} P(fg)", f"count {band_count(fg)} / {expected}"]
    Z = 4
    tiles = []
    for c, h in zip(cols, heads):
        t = cv2.resize(c, None, fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST)
        t = cv2.copyMakeBorder(t, 40, 0, 0, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(t, h, (4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        tiles.append(t)
    cv2.imwrite(out, np.hstack(tiles))


if __name__ == "__main__":
    main()
