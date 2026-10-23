"""Broken-mask audit for sheet (edge) models: is each labelled sheet ONE whole mask?

    python -m tools.fragment_eval <ckpt.pt> [<ckpt2.pt> ...] [--sets new_holdout,old_val] [--json out.json]

Several checkpoints are averaged exactly like the shipped pipeline (mean sheet probability,
threshold 0.5), then counted and grown into masks with the shipped inference code.

Per labelled sheet (human "edge" polygon, model-input coordinates):
  pieces    connected pieces of predicted foreground on the sheet (> 1 = BROKEN: gaps)
  coverage  share of the sheet's labelled pixels predicted as foreground
  whole     one final mask covers >= 80 % of the sheet and no other mask >= 10 %
Per strip: count error vs the number of labelled sheets.

Sets: new_holdout / new_train = data/oct26/edges split by HOLDOUT below (the holdout is
only for choosing the recipe; the final model trains on all of oct26), old_val = the pinned
val crops of data/edges (the old test frames).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

import paths
from common import labelme
from common.imageio import pad_to_multiple, to_tensor
from common.unet import load_checkpoint
from edges.dataset import discover, fit_width, sync_split, tight

sys.path.insert(0, os.path.join(paths.ROOT, "inference"))
from sheetcount.pipeline import SHEET_THR, SheetCounter  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OCT26 = os.path.join(paths.DATA, "oct26", "edges")
# frames held out of training while CHOOSING the recipe (whole capture bursts where possible)
HOLDOUT = ("2026-10-02_09-53", "2026-10-02_09-54", "2026-10-02_09-56",     # mainshots, later burst
           "20261002_103619_type3", "20261002_103338_type1")


def oct26_items(which: str) -> list[dict]:
    out = []
    for png in sorted(glob.glob(os.path.join(OCT26, "*", "*.png"))):
        js = png[:-4] + ".json"
        if not os.path.exists(js):
            continue
        stem = os.path.splitext(os.path.basename(png))[0]
        held = any(h in stem for h in HOLDOUT)
        if (which == "new_holdout") == held:
            out.append({"stem": stem, "frame": stem.rsplit("_column_", 1)[0],
                        "group": os.path.basename(os.path.dirname(png)), "image": png, "json": js})
    return out


def old_val_items() -> list[dict]:
    items, _ = discover()
    val = set(sync_split(items, write=False)["val"])
    return [{**it, "group": it["type"]} for it in items if it["frame"] in val]


@torch.no_grad()
def prob(nets, img):
    padded, h, w = pad_to_multiple(img, 16)
    x = to_tensor(padded)[None].to(DEVICE)
    return np.mean([torch.sigmoid(n(x)[0, 0]).cpu().numpy()[:h, :w] for n in nets], axis=0)


def score_strip(nets, it):
    img = cv2.imread(it["image"])
    lab = labelme.edge_labels(it["json"])
    img, lab = fit_width(*tight(img, lab), "center")
    return score_arrays(nets, it, img, lab)


def score_arrays(nets, it, img, lab):
    fg = prob(nets, img) > SHEET_THR
    count, seeds, ab = SheetCounter.count_sheets(fg)
    masks = SheetCounter.sheet_masks(fg, seeds, ab)
    sheets = []
    for i in np.unique(lab):
        if i == 0:
            continue
        g = lab == i
        n, cc, st, _ = cv2.connectedComponentsWithStats((fg & g).astype(np.uint8), connectivity=8)
        pieces = int((st[1:, cv2.CC_STAT_AREA] >= 3).sum())
        ids, cnt = np.unique(masks[g], return_counts=True)
        share = sorted((c / g.sum() for k, c in zip(ids, cnt) if k > 0), reverse=True)
        whole = bool(share and share[0] >= 0.8 and (len(share) < 2 or share[1] < 0.1))
        sheets.append({"pieces": pieces, "coverage": float((fg & g).sum() / g.sum()), "whole": whole})
    n_gt = len(sheets)
    return {"stem": it["stem"], "group": it["group"], "count": count, "n_gt": n_gt,
            "count_err": count - n_gt, "sheets": sheets}


def summarise(rows):
    sh = [s for r in rows for s in r["sheets"]]
    return {"strips": len(rows), "sheets": len(sh),
            "count_exact": sum(r["count_err"] == 0 for r in rows), "count_mae": float(np.mean([abs(r["count_err"]) for r in rows])),
            "broken_pct": 100 * float(np.mean([s["pieces"] > 1 for s in sh])),
            "missed_pct": 100 * float(np.mean([s["pieces"] == 0 for s in sh])),
            "coverage": float(np.mean([s["coverage"] for s in sh])),
            "whole_pct": 100 * float(np.mean([s["whole"] for s in sh]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--sets", default="new_holdout,old_val")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    nets = [load_checkpoint(c, DEVICE) for c in a.ckpts]
    out = {}
    for s in a.sets.split(","):
        items = old_val_items() if s == "old_val" else oct26_items(s)
        rows = [score_strip(nets, it) for it in items]
        out[s] = {"summary": summarise(rows), "rows": rows}
        m = out[s]["summary"]
        print(f"{s:12s} strips {m['strips']:3d} sheets {m['sheets']:5d} | count exact {m['count_exact']}/{m['strips']} "
              f"MAE {m['count_mae']:.2f} | broken {m['broken_pct']:.1f}% missed {m['missed_pct']:.1f}% "
              f"coverage {m['coverage']:.3f} | whole masks {m['whole_pct']:.1f}%")
        if s != "old_val":
            for r in rows:
                b = sum(x["pieces"] > 1 for x in r["sheets"])
                print(f"    {r['stem']:48s} count {r['count']:3d}/{r['n_gt']:3d}  broken sheets {b:3d}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
