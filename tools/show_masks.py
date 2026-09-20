"""Look at the produced masks: whole column + tip zoom, human labels outlined.

    python -m tools.show_masks <run> <out_prefix> [n_per_type]

Left: input. Middle: grow-readout instances (white = instance-vs-instance contact).
Right: the same with the HUMAN sheet outlines drawn in white, so a mask that ends
short, spills past its sheet, or misses a sheet entirely is visible.
"""
import sys

import cv2
import numpy as np
import torch

import paths
from common.imageio import colorize
from common.unet import load_checkpoint
from edges.counter import THR, band, infer
from edges.dataset import build_val, discover, sync_split
from edges.grow import grow_instances

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def outlines(lab):
    m = np.zeros(lab.shape, np.uint8)
    for i in range(1, int(lab.max()) + 1):
        c, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(m, c, -1, 1, 1)
    return m > 0


def main():
    run, prefix = sys.argv[1], sys.argv[2]
    per = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    net = load_checkpoint(f"{paths.RUNS}/{run}/last.pt", DEVICE)
    items, _ = discover()
    split = sync_split(items, write=False)
    val = build_val([it for it in items if it["frame"] in set(split["val"])], "center")
    seen = {}
    for it in val:
        if seen.get(it["type"], 0) >= per:
            continue
        seen[it["type"]] = seen.get(it["type"], 0) + 1
        prob, _ = infer(net, it["img"], DEVICE)
        fg = prob > THR
        lab, n = grow_instances(fg)
        a, b = band(fg)
        vis = cv2.addWeighted(it["img"], 0.4, colorize(lab)[:, :, ::-1], 0.6, 0)
        vis[lab == 0] = (it["img"][lab == 0] * 0.5).astype(np.uint8)
        gt_vis = vis.copy()
        gt_vis[outlines(it["gt"])] = (255, 255, 255)
        ys = np.flatnonzero(it["img"].max(axis=(1, 2)) > 0)
        y0, y1 = ys[0], ys[-1] + 1
        panes = [it["img"][y0:y1], vis[y0:y1], gt_vis[y0:y1]]
        whole = np.hstack([np.hstack([p, np.full((p.shape[0], 3, 3), 255, np.uint8)]) for p in panes])
        whole = cv2.rotate(whole, cv2.ROTATE_90_CLOCKWISE)          # lay the column flat
        s = min(1900 / whole.shape[1], 3.0)
        cv2.imwrite(f"{prefix}_{it['type']}_whole.jpg", cv2.resize(whole, None, fx=s, fy=s))
        # tip zoom: 120 rows in the middle, left edge / right edge
        yc = y0 + (y1 - y0) // 2 - 60
        w = it["img"].shape[1]
        zones = [("left", 0, 24), ("right", w - 24, w)]
        cols = []
        for name, x0, x1 in zones:
            t = np.hstack([it["img"][yc:yc + 120, x0:x1], np.full((120, 2, 3), 255, np.uint8),
                           gt_vis[yc:yc + 120, x0:x1]])
            t = cv2.resize(t, None, fx=7, fy=7, interpolation=cv2.INTER_NEAREST)
            head = np.zeros((28, t.shape[1], 3), np.uint8)
            cv2.putText(head, f"{it['type']} {name} tip", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cols.append(np.vstack([head, t]))
        cv2.imwrite(f"{prefix}_{it['type']}_tips.jpg", np.hstack(cols))
        print(f"{it['type']:7s} {it['stem'][16:]}  count {n} / human {it['n_gt']}  band x {a}-{b}")


if __name__ == "__main__":
    main()
