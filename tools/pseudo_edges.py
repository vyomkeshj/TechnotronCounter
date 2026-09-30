"""Self-training data for the edge model (count-filtered pseudo labels).

    python -m tools.pseudo_edges [--model edges_seed] [--out data/edges_pseudo]

For every GOOD column of every COLUMN-TRAINING frame of type2/3/10 (human column polygons,
production straightener), run the edge model on the exact model input (centre-width crop),
read instances with the grow readout, and KEEP the crop only if its band count equals the
product count (type3 50, type10 24, type2 filename). The model never sees the product
count, so the filter is what stops it from teaching itself its own mistakes.

Written as <out>/<type>/<frame>_column_<k>.png + .json (LabelMe "edge" polygons) in the
same format as data/edges, but in a SEPARATE tree: human labels are never touched. Frames
already in data/edges (human edge labels) are skipped, and so is every frame of the held-out
test sessions, even if a split file were to change. Train on it with
`edges.train ... --extra-data data/edges_pseudo`.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re

import cv2
import numpy as np
import torch

import paths
from common import labelme
from common.unet import load_checkpoint
from edges.counter import REL_FLOOR, THR, band_count, infer
from edges.dataset import model_input
from edges.grow import grow_instances
from tools.evalkit import ckpt_path, expected, straighten_polys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_SESSIONS = {"132812", "134020", "131412"}          # + type10 frames 9-14 via the split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="edges_seed", help="the seed edge model (run name or checkpoint)")
    ap.add_argument("--out", default=os.path.join("data", "edges_pseudo"))
    args = ap.parse_args()
    net = load_checkpoint(ckpt_path(args.model), DEVICE); net.eval()
    cs = json.load(open(paths.COLUMNS_SPLIT))
    train_frames = set(cs["train"])
    human = {os.path.basename(p).rsplit("_column_", 1)[0] for p in glob.glob(os.path.join(paths.EDGES_DATA, "*", "*.json"))}
    out = os.path.join(paths.ROOT, args.out)
    stats = collections.Counter()
    for j in sorted(glob.glob(os.path.join(paths.COLUMNS_DATA, "quality", "*", "*.json"))):
        stem = os.path.basename(j)[:-5]
        typ = re.search(r"(type\d+)", stem).group(1)
        if stem in human or stem not in train_frames or stem.split("_")[1] in TEST_SESSIONS:
            continue
        img = cv2.imread(j[:-5] + ".jpg")
        shapes = json.load(open(j, encoding="utf-8"))["shapes"]
        good = [np.round(np.array(s["points"])).astype(np.int32) for s in shapes if s["label"] == "good_column"]
        if not good:
            continue
        exp = expected(stem)
        for k, crop in enumerate(straighten_polys(img, good)):
            if crop is None:
                continue
            x = model_input(crop, "center")
            fg = infer(net, x, DEVICE)[0] > THR
            n = band_count(fg)
            stats[(typ, "total")] += 1
            if n != exp:
                continue
            inst, m = grow_instances(fg, rel_floor=REL_FLOOR)
            if m != exp:
                continue
            stats[(typ, "kept")] += 1
            d = os.path.join(out, typ)
            os.makedirs(d, exist_ok=True)
            name = f"{stem}_column_{k}.png"
            cv2.imwrite(os.path.join(d, name), x)
            labelme.write_polygons(os.path.join(d, name[:-4] + ".json"), inst, labelme.EDGE, name)
    for typ in ("type2", "type3", "type10"):
        print(f"{typ}: kept {stats[(typ, 'kept')]} of {stats[(typ, 'total')]} crops")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
