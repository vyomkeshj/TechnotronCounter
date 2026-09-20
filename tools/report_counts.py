"""One count report per run: val, all labelled crops, and the unseen older sets.

    python -m tools.report_counts <run> [<run> ...]   -> prints + runs/_report_counts.csv

Count only (the project's measure of success). Uses the shipped counter defaults
(centre band + relative blob floor).
"""
from __future__ import annotations

import csv
import glob
import os
import sys

import cv2
import numpy as np
import torch

import paths
from common.unet import load_checkpoint
from edges.counter import THR, band_count, infer
from edges.dataset import build_val, discover, model_input, sync_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# The "unseen older" crops are NOT part of data/ -- they come from the colleague's
# cutter and live outside the pipeline. Point TECHNOTRON_OLD_CROPS at that tree to
# score them; without it those columns are SKIPPED, never silently reported as 0/0.
OLD = os.environ.get("TECHNOTRON_OLD_CROPS", r"D:\Projects\technotron\edge_dataset\good_images")
EXTRA = {                                          # name -> (glob, expected count)
    "old_t3_131412": (os.path.join(OLD, "type3", "columnsnew", "good", "*_131412_*.png"), 50),
    "old_t3_131612": (os.path.join(OLD, "type3", "columnsnew", "good", "*_131612_*.png"), 50),
    "old_t10": (os.path.join(OLD, "type10", "columnsnew", "good", "*.png"), 24),
}


def main():
    runs = sys.argv[1:]
    items, _ = discover()
    split = sync_split(items, write=False)
    val = build_val([it for it in items if it["frame"] in set(split["val"])], "center")
    labelled = {t: [it for it in items if it["type"] == t] for t in sorted({it["type"] for it in items})}
    extra = {k: (sorted(glob.glob(g)), n) for k, (g, n) in EXTRA.items()}
    for k in [k for k, (ps, _) in extra.items() if not ps]:
        print(f"[report_counts] SKIPPING {k}: no crops under {OLD} "
              f"(set TECHNOTRON_OLD_CROPS to that tree)", flush=True)
        del extra[k]
    rows = []
    for run in runs:
        net = load_checkpoint(os.path.join(paths.RUNS, run, "last.pt"), DEVICE)
        row = {"run": run}
        hit = {}
        for it in val:
            ok = band_count(infer(net, it["img"], DEVICE)[0] > THR) == it["n_gt"]
            d = hit.setdefault(it["type"], [0, 0]); d[0] += ok; d[1] += 1
        for t, (ok, n) in sorted(hit.items()):
            row[f"val_{t}"] = f"{ok}/{n}"
        for t, its in labelled.items():
            ok = 0
            for it in its:
                img, gt = build_val([it], "center")[0]["img"], build_val([it], "center")[0]["n_gt"]
                ok += int(band_count(infer(net, img, DEVICE)[0] > THR) == gt)
            row[f"all_{t}"] = f"{ok}/{len(its)}"
        for name, (ps, exp) in extra.items():
            ok = sum(int(band_count(infer(net, model_input(cv2.imread(p), "center"), DEVICE)[0] > THR) == exp)
                     for p in ps)
            row[name] = f"{ok}/{len(ps)}"
        rows.append(row)
        print(run, " | ".join(f"{k} {v}" for k, v in row.items() if k != "run"), flush=True)
    dst = os.path.join(paths.RUNS, "_report_counts.csv")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    write_header = not os.path.exists(dst)
    with open(dst, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print("->", dst)


if __name__ == "__main__":
    main()
