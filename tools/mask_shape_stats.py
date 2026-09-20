"""Mask SHAPE vs the human sheet labels, per type (grow readout).

    python -m tools.mask_shape_stats <run> [<run> ...]

IoU / recall / precision are per matched sheet. Thickness and x-extent say whether
a mask is thinner or shorter than the labelled sheet -- the two visible defects
after identity is fixed. 'fragmented' = instances whose mask is more than one
connected piece.
"""
import sys

import cv2
import numpy as np
import torch

import paths
from common.unet import load_checkpoint
from edges.counter import THR, infer
from edges.dataset import build_val, discover, sync_split
from edges.grow import grow_instances

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    items, _ = discover()
    split = sync_split(items, write=False)
    val = build_val([it for it in items if it["frame"] in set(split["val"])], "center")
    for run in sys.argv[1:]:
        net = load_checkpoint(f"{paths.RUNS}/{run}/last.pt", DEVICE)
        agg = {}
        for it in val:
            fg = infer(net, it["img"], DEVICE)[0] > THR
            lab, n = grow_instances(fg)
            gt = it["gt"]
            d = agg.setdefault(it["type"], {k: [] for k in ("iou", "rec", "prec", "w", "gw", "x", "gx", "frag", "miss")})
            for i in range(1, int(gt.max()) + 1):
                g = gt == i
                if not g.any():
                    continue
                ids = [v for v in np.unique(lab[g]) if v]
                if not ids:
                    d["miss"].append(1); d["iou"].append(0.0); d["rec"].append(0.0)
                    continue
                d["miss"].append(0)
                p = lab == max(ids, key=lambda v: int((lab[g] == v).sum()))
                d["iou"].append((g & p).sum() / max((g | p).sum(), 1))
                d["rec"].append((g & p).sum() / g.sum())
                d["prec"].append((g & p).sum() / max(p.sum(), 1))
                d["w"].append(p.sum() / max(len(np.unique(np.where(p)[1])), 1))
                d["gw"].append(g.sum() / max(len(np.unique(np.where(g)[1])), 1))
                d["x"].append(len(np.unique(np.where(p)[1])))
                d["gx"].append(len(np.unique(np.where(g)[1])))
            for i in range(1, n + 1):
                d["frag"].append(int(cv2.connectedComponents((lab == i).astype(np.uint8))[0] - 1 > 1))
        print(f"\n{run}")
        for t, d in sorted(agg.items()):
            m = {k: float(np.mean(v)) if v else 0.0 for k, v in d.items()}
            print(f"  {t:7s} IoU {m['iou']:.3f} | covers {100 * m['rec']:3.0f} % of the human sheet | "
                  f"precision {100 * m['prec']:3.0f} % | thickness {m['w']:4.1f} px vs human {m['gw']:4.1f} | "
                  f"length {m['x']:3.0f} px vs human {m['gx']:3.0f} | fragmented {100 * m['frag']:3.0f} % | "
                  f"sheets with no mask {100 * m['miss']:3.0f} %")


if __name__ == "__main__":
    main()
