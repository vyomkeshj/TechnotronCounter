"""Sheet (edge) models on the HELD-OUT labelled strips of every type (data/edges/splits.json "val"), per type:
count exact / MAE vs the labelled sheets, merged masks (one mask over 2+ labelled sheets) and whole-sheet masks.
The checkpoints are averaged like the shipped pipeline (mean probability, threshold 0.5).

    python -m tools.edge_heldout_check <ckpt.pt> [<ckpt2.pt> ...]
"""
from __future__ import annotations

import argparse
import collections

import numpy as np

from common.unet import load_checkpoint
from edges.dataset import discover, sync_split
from tools.fragment_eval import DEVICE, score_strip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--rotate", type=float, default=0.0, help="rotate every strip first (margin stress)")
    a = ap.parse_args()
    import tools.fragment_eval as FE
    FE.ROTATE = a.rotate
    nets = [load_checkpoint(c, DEVICE) for c in a.ckpts]
    items, _ = discover()
    val = set(sync_split(items, write=False)["val"])
    by = collections.defaultdict(list)
    for it in items:
        if it["frame"] in val:
            r = score_strip(nets, {**it, "group": it["type"]})
            by[it["type"]].append(r)
            by["ALL"].append(r)
    print(f"{'type':7s} {'strips':>6s} {'exact':>9s} {'MAE':>5s} {'merged':>7s} {'whole':>6s}")
    for t in sorted(by, key=lambda t: (t == "ALL", len(t), t)):
        rows = by[t]
        sh = [s for r in rows for s in r["sheets"]]
        n_m = sum(r["n_masks"] for r in rows)
        print(f"{t:7s} {len(rows):6d} {sum(r['count_err'] == 0 for r in rows):4d}/{len(rows):<4d} "
              f"{np.mean([abs(r['count_err']) for r in rows]):5.2f} {100 * sum(r['merged'] for r in rows) / max(1, n_m):6.2f}% "
              f"{100 * np.mean([s['whole'] for s in sh]):5.1f}%")


if __name__ == "__main__":
    main()
