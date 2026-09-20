"""Label-free check: do an edge model's counts match the piece count in the file name?

    python -m tools.count_vs_filename <type> <crops_glob> <run> [<run> ...]

Expected count per type (VISIBLE_EDGES):
  type2   the filename piece count -- verified: human labels are 50 on every
          50pieces crop and 30 on every 30pieces crop
  type3   always 50 (filename says 60 = stack size; human labels 50 on 71/72 crops)
  type10  always 24 (filename says 28; user-confirmed)

Each crop is counted with centre-crop and squash input, by the band counter and by
the flow instances, and bucketed by whether its capture session is held out in the
edge val split (a train-session hit is not evidence of generalisation).
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import cv2
import numpy as np
import torch

import paths
from common import splits
from common.unet import load_checkpoint
from edges.counter import read
from edges.dataset import model_input

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIECES = re.compile(r"_(\d+)pieces_")
VISIBLE_EDGES = {"type3": 50, "type10": 24}      # types whose filename count is NOT the visible count


def main():
    typ, pattern, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
    files = sorted(glob.glob(pattern))
    trained_frames = set(json.load(open(paths.EDGES_SPLIT)).get("train", []))
    crops = []
    for p in files:
        m = PIECES.search(os.path.basename(p))
        if m and f"_{typ}_" in os.path.basename(p):
            frame = os.path.basename(p).rsplit("_column_", 1)[0]
            img = cv2.imread(p)
            crops.append({"file": os.path.basename(p), "expected": VISIBLE_EDGES.get(typ, int(m.group(1))),
                          "val": frame not in trained_frames,        # True = frame never trained on
                          "img": {mode: model_input(img, mode) for mode in ("center", "squash")}})
    print(f"{len(crops)} {typ} crops from {pattern} | frames never trained on: {sum(c['val'] for c in crops)}")
    out = {}
    for run in runs:
        net = load_checkpoint(os.path.join(paths.RUNS, run, "last.pt"), DEVICE)
        for mode in ("center", "squash"):
            rows = []
            for c in crops:
                r = read(net, c["img"][mode], DEVICE)
                rows.append({"file": c["file"], "val": c["val"], "expected": c["expected"],
                             "band": r["count"], "inst": int(r["instances"].max())})
            out[f"{run}|{mode}"] = rows
            for bucket, sel in (("unseen frames", True), ("trained frames", False)):
                rs = [r for r in rows if r["val"] == sel]
                if not rs:
                    continue
                be = np.array([r["band"] - r["expected"] for r in rs])
                ie = np.array([r["inst"] - r["expected"] for r in rs])
                by = {e: f"{np.mean([r['band'] == e for r in rs if r['expected'] == e]):.0%}"
                      for e in sorted({r['expected'] for r in rs})}
                print(f"  {run:18s} {mode:6s} {bucket:14s} n={len(rs):3d} | band exact {np.mean(be == 0):4.0%} "
                      f"MAE {np.abs(be).mean():5.2f} (exact by piece count {by}) | masks exact {np.mean(ie == 0):4.0%} "
                      f"MAE {np.abs(ie).mean():5.2f}")
    dst = os.path.join(paths.RUNS, f"count_vs_filename_{typ}.json")
    json.dump(out, open(dst, "w"), indent=1)
    print("->", dst)


if __name__ == "__main__":
    main()
