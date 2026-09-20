"""Evaluate an edge checkpoint on the pinned val crops, per column type.

    python -m edges.evaluate runs/edges_<name>/last.pt [--mode center|squash] [--render out_dir]

count_*   the shipped band counter vs the number of human "edge" polygons
inst_*    per-sheet instances from the flow readout vs human polygons
Human labels mark sheet BANDS (~42 % of the column silhouette), so instIoU is
only comparable between models scored against the same human labels.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch

from common.imageio import colorize
from common.metrics import instance_metrics
from common.unet import load_checkpoint
from edges.counter import band, read

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(net, val: list[dict], render_dir: str | None = None) -> dict:
    net.eval()
    per: dict[str, list] = {}
    for it in val:
        r = read(net, it["img"], DEVICE)
        m = instance_metrics(r["instances"], it["gt"])
        row = {"count_err": r["count"] - it["n_gt"], "inst_count_err": m["n_pred"] - it["n_gt"],
               "inst_iou": m["inst_iou"]}
        per.setdefault(it["type"], []).append(row)
        per.setdefault("all", []).append(row)
        if render_dir:
            os.makedirs(render_dir, exist_ok=True)
            vis = cv2.addWeighted(it["img"], 0.45, colorize(r["instances"])[:, :, ::-1], 0.55, 0)
            a, b = band(r["fg"])
            cv2.line(vis, (a, 0), (a, vis.shape[0] - 1), (0, 0, 255), 1)
            cv2.line(vis, (b - 1, 0), (b - 1, vis.shape[0] - 1), (0, 0, 255), 1)
            panel = np.hstack([it["img"], vis, colorize(it["gt"])[:, :, ::-1]])
            cv2.imwrite(os.path.join(render_dir, f"{it['type']}__{it['stem']}__count{r['count']}_gt{it['n_gt']}.png"),
                        cv2.resize(panel, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
    out = {}
    for g, rows in per.items():
        ce = np.array([r["count_err"] for r in rows], float)
        ie = np.array([r["inst_count_err"] for r in rows], float)
        out[g] = {"n": len(rows), "count_mae": float(np.abs(ce).mean()), "count_exact": float((ce == 0).mean()),
                  "count_within1": float((np.abs(ce) <= 1).mean()), "count_bias": float(ce.mean()),
                  "inst_count_mae": float(np.abs(ie).mean()),
                  "inst_iou": float(np.mean([r["inst_iou"] for r in rows]))}
    return out


def preview(net, val: list[dict], n: int = 4) -> np.ndarray:
    """CHW grid of the first n val crops: input | instances, rotated to landscape."""
    tiles = []
    for it in val[:n]:
        r = read(net, it["img"], DEVICE)
        vis = cv2.addWeighted(it["img"], 0.45, colorize(r["instances"])[:, :, ::-1], 0.55, 0)
        t = cv2.rotate(np.hstack([it["img"], vis])[:, :, ::-1], cv2.ROTATE_90_COUNTERCLOCKWISE)
        tiles.append(cv2.resize(t, (1024, int(t.shape[0] * 1024 / t.shape[1]))))
    return np.vstack(tiles).transpose(2, 0, 1)


def main():
    from edges.dataset import build_val, discover, sync_split
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--mode", default="center", choices=["center", "squash"])
    ap.add_argument("--render", default=None)
    a = ap.parse_args()
    items, _ = discover()
    split = sync_split(items, write=False)
    val = build_val([it for it in items if it["frame"] in set(split["val"])], a.mode)
    res = evaluate(load_checkpoint(a.checkpoint, DEVICE), val, a.render)
    for g, m in sorted(res.items()):
        print(f"{g:8s} n={m['n']:3d} band count MAE {m['count_mae']:.2f} exact {m['count_exact']:.0%} "
              f"<=1 {m['count_within1']:.0%} bias {m['count_bias']:+.2f} | instances MAE {m['inst_count_mae']:.2f} "
              f"instIoU {m['inst_iou']:.3f}")
    if a.render:
        json.dump(res, open(os.path.join(a.render, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
