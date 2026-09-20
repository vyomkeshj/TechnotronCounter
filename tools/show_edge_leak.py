"""Zoom the per-sheet masks at the strip's left edge / centre / right edge.

    python -m tools.show_edge_leak <run> <out.jpg> [rows]

Per crop: input | instances, at 3 x-zones. White pixels mark instance-vs-instance
CONTACT (two different ids adjacent) -- where sheets are claimed by their
neighbour. The count rule only uses the centre 50 % of the width, so a leak at
the tips changes the MASKS, not the count.
"""
import sys

import cv2
import numpy as np
import torch

import paths
from common import labelme
from common.imageio import colorize
from common.unet import load_checkpoint
from edges.counter import band, read
from edges.dataset import build_val, discover, sync_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def contact(lab):
    m = np.zeros(lab.shape, bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = lab[max(0, -dy):lab.shape[0] - max(0, dy), max(0, -dx):lab.shape[1] - max(0, dx)]
        b = lab[max(0, dy):lab.shape[0] - max(0, -dy), max(0, dx):lab.shape[1] - max(0, -dx)]
        touch = (a > 0) & (b > 0) & (a != b)
        m[max(0, -dy):lab.shape[0] - max(0, dy), max(0, -dx):lab.shape[1] - max(0, dx)] |= touch
    return m


def main():
    run, out = sys.argv[1], sys.argv[2]
    rows = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    net = load_checkpoint(f"{paths.RUNS}/{run}/last.pt", DEVICE)
    items, _ = discover()
    split = sync_split(items, write=False)
    val = build_val([it for it in items if it["frame"] in set(split["val"])], "center")
    panels = []
    for typ in ("type2", "type3", "type10"):
        it = next((v for v in val if v["type"] == typ), None)
        if it is None:
            continue
        r = read(net, it["img"], DEVICE)
        lab = r["instances"]
        a, b = band(r["fg"])
        vis = cv2.addWeighted(it["img"], 0.4, colorize(lab)[:, :, ::-1], 0.6, 0)
        vis[lab == 0] = (it["img"][lab == 0] * 0.5).astype(np.uint8)
        vis[contact(lab)] = (255, 255, 255)
        ys = np.flatnonzero(it["img"].max(axis=(1, 2)) > 0)
        y0 = ys[0] + (len(ys) - rows) // 2
        w = it["img"].shape[1]
        zones = [("left edge", 0, min(26, w)), ("centre (counted)", a, b), ("right edge", max(0, w - 26), w)]
        col = []
        for name, x0, x1 in zones:
            t = np.hstack([it["img"][y0:y0 + rows, x0:x1],
                           np.full((rows, 2, 3), 255, np.uint8),
                           vis[y0:y0 + rows, x0:x1]])
            t = cv2.resize(t, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
            head = np.zeros((34, t.shape[1], 3), np.uint8)
            cv2.putText(head, f"{typ} {name}", (4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            col.append(np.vstack([head, t]))
        h = max(c.shape[0] for c in col)
        panels.append(np.hstack([cv2.copyMakeBorder(c, 0, h - c.shape[0], 0, 14, cv2.BORDER_CONSTANT, value=(30, 30, 30))
                                 for c in col]))
        n_contact = int(contact(lab).sum())
        print(f"{typ}: {it['stem'][16:]} count {r['count']} / human {it['n_gt']} | contact px {n_contact}")
    W = max(p.shape[1] for p in panels)
    cv2.imwrite(out, np.vstack([cv2.copyMakeBorder(p, 0, 16, 0, W - p.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30))
                                for p in panels]))
    print("->", out)


if __name__ == "__main__":
    main()
