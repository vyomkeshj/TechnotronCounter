"""Run an edge checkpoint over a folder of column crops (labelled or not).

    python -m edges.predict runs/edges_<name>/last.pt <crops_dir> <out_dir> [--mode center|squash]

Writes per crop: <stem>.jpg (input | coloured sheets, band lines), <stem>.json
(LabelMe "edge" polygons in MODEL-INPUT coordinates -- usable as pre-labels only
after checking), and predictions.csv with the band count and instance count.
If a human <stem>.json exists beside the crop, its edge count is added.
Use this to see how a model does on new data BEFORE labelling it.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import cv2
import numpy as np
import torch

from common import labelme
from common.imageio import colorize
from common.unet import load_checkpoint
from edges.counter import band, read
from edges.dataset import model_input

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("crops_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--mode", default="center", choices=["center", "squash"])
    a = ap.parse_args()
    net = load_checkpoint(a.checkpoint, DEVICE)
    os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    for p in sorted(glob.glob(os.path.join(a.crops_dir, "*.png"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        img = model_input(cv2.imread(p), a.mode)
        r = read(net, img, DEVICE)
        vis = cv2.addWeighted(img, 0.45, colorize(r["instances"])[:, :, ::-1], 0.55, 0)
        x0, x1 = band(r["fg"])
        cv2.line(vis, (x0, 0), (x0, vis.shape[0] - 1), (0, 0, 255), 1)
        cv2.line(vis, (x1 - 1, 0), (x1 - 1, vis.shape[0] - 1), (0, 0, 255), 1)
        cv2.imwrite(os.path.join(a.out_dir, stem + ".jpg"),
                    cv2.resize(np.hstack([img, vis]), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
        cv2.imwrite(os.path.join(a.out_dir, stem + "_input.png"), img)
        labelme.write_polygons(os.path.join(a.out_dir, stem + ".json"), r["instances"], "edge", stem + "_input.png")
        human = os.path.join(a.crops_dir, stem + ".json")
        n_h = int(len(np.unique(labelme.edge_labels(human))) - 1) if os.path.exists(human) else ""
        rows.append({"file": os.path.basename(p), "band_count": r["count"],
                     "instances": int(r["instances"].max()), "human": n_h})
    with open(os.path.join(a.out_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "band_count", "instances", "human"])
        w.writeheader(); w.writerows(rows)
    c = np.array([r["band_count"] for r in rows])
    print(f"{len(rows)} crops | band count median {np.median(c):.0f} [{c.min()}-{c.max()}] -> {a.out_dir}")


if __name__ == "__main__":
    main()
