"""Look at what a scatter-trained model actually predicts.

    python -m experiments.scatter.show --run scatter_mix50 --scenes out1.png --strip out2.png

scenes : held-out scatter scenes -- input | target | prediction | errors (red = false
         positive, blue = missed). The errors panel is the one worth reading.
strip  : a REAL full column crop, rotated so the stack runs left-to-right and wrapped
         into rows, with the grow-readout instance masks and the resulting count, for
         this model and for the production champion side by side.
"""
from __future__ import annotations

import argparse
import os
import random

import cv2
import numpy as np
import torch

import paths
from common.unet import load_checkpoint
from edges.counter import REL_FLOOR, THR, band_count, infer
from edges.dataset import build_val, discover, model_input, sync_split
from edges.grow import grow_instances
from experiments.scatter.dataset import scene

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PALETTE = np.array([(232, 108, 62), (64, 176, 166), (226, 168, 40), (152, 104, 214),
                    (86, 152, 232), (214, 96, 128), (118, 190, 96)], np.uint8)


def ckpt(run: str) -> str:
    p = os.path.join(paths.RUNS, run, "last.pt")
    return p if os.path.exists(p) else os.path.join(paths.RUNS, run, "stage2", "last.pt")


def label_row(cells, titles, zoom, pad=12, header=26):
    cells = [cv2.resize(c, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST) for c in cells]
    h, w = cells[0].shape[:2]
    out = np.full((h + header, w * len(cells) + pad * (len(cells) - 1), 3), 20, np.uint8)
    for i, c in enumerate(cells):
        x = i * (w + pad)
        out[header:header + h, x:x + w] = c
        cv2.rectangle(out, (x - 1, header - 1), (x + w, header + h), (70, 70, 70), 1)
        cv2.putText(out, titles[i], (x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (235, 235, 235), 1, cv2.LINE_AA)
    return out


def scenes_figure(net, val_items, n, zoom, seed=1234):
    rows = []
    for i in range(n):
        img, lab = scene(val_items, random.Random(seed + i))
        gt = lab > 0
        pr = infer(net, img, DEVICE)[0] > THR
        tgt = cv2.cvtColor(gt.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        prd = cv2.cvtColor(pr.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        err = (img * 0.45).astype(np.uint8)
        err[pr & ~gt] = (60, 60, 255)                      # false positive
        err[gt & ~pr] = (255, 140, 60)                     # missed
        err[pr & gt] = (90, 220, 120)                      # hit
        iou = (pr & gt).sum() / max(1, (pr | gt).sum())
        rows.append(label_row([img, tgt, prd, err],
                              ["input", "target", "prediction",
                               f"green hit / red false pos / blue missed   IoU {iou:.2f}"], zoom))
    w = max(r.shape[1] for r in rows)
    return np.vstack([np.pad(r, ((0, 14), (0, w - r.shape[1]), (0, 0)), constant_values=20) for r in rows])


def strip_panel(net, img, title, zoom, seg, gap=10):
    prob = infer(net, img, DEVICE)[0]
    fg = prob > THR
    inst, n = grow_instances(fg, rel_floor=REL_FLOOR)
    cnt = band_count(fg)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = cv2.cvtColor((gray * 0.45).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for i in range(1, int(inst.max()) + 1):
        m = inst == i
        if m.any():
            vis[m] = (0.30 * vis[m] + 0.70 * PALETTE[(i - 1) % len(PALETTE)][::-1]).astype(np.uint8)
    vis = cv2.rotate(vis, cv2.ROTATE_90_CLOCKWISE)          # stack -> horizontal
    H, W = vis.shape[:2]
    chunks = [vis[:, s:s + seg] for s in range(0, W, seg)]
    chunks = [np.pad(c, ((0, 0), (0, seg - c.shape[1]), (0, 0))) if c.shape[1] < seg else c for c in chunks]
    rows = [cv2.resize(c, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST) for c in chunks]
    rh, rw = rows[0].shape[:2]
    out = np.full((26 + len(rows) * (rh + gap), rw, 3), 20, np.uint8)
    for i, r in enumerate(rows):
        out[26 + i * (rh + gap):26 + i * (rh + gap) + rh, :rw] = r
    cv2.putText(out, f"{title}   ->  count {cnt}, {n} masks", (2, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 220, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="scatter_mix50")
    ap.add_argument("--compare", default="edges_all3_s7")
    ap.add_argument("--scenes", default="")
    ap.add_argument("--strip", default="")
    ap.add_argument("--crop", default="", help="a specific crop png for the strip figure")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--zoom", type=int, default=2)
    ap.add_argument("--seg", type=int, default=520)
    args = ap.parse_args()

    items, _ = discover()
    sp = sync_split(items, write=False)
    val_items = [i for i in items if i["frame"] in set(sp["val"])]
    net = load_checkpoint(ckpt(args.run), DEVICE)

    if args.scenes:
        fig = scenes_figure(net, val_items, args.n, args.zoom)
        cv2.imwrite(args.scenes, fig)
        print(f"scenes -> {os.path.abspath(args.scenes)} {fig.shape[1]}x{fig.shape[0]}")

    if args.strip:
        src = args.crop or sorted(i["image"] for i in val_items if i["type"] == "type3")[0]
        img = model_input(cv2.imread(src), "center")
        gt = build_val([i for i in val_items if i["image"] == src], "center")
        n_gt = gt[0]["n_gt"] if gt else -1
        panels = [strip_panel(net, img, f"{args.run}", args.zoom, args.seg)]
        if args.compare:
            ref = load_checkpoint(ckpt(args.compare), DEVICE)
            panels.append(strip_panel(ref, img, f"{args.compare} (champion)", args.zoom, args.seg))
        w = max(p.shape[1] for p in panels)
        fig = np.vstack([np.pad(p, ((0, 16), (0, w - p.shape[1]), (0, 0)), constant_values=20) for p in panels])
        head = np.full((26, w, 3), 20, np.uint8)
        cv2.putText(head, f"{os.path.basename(src)}   human label: {n_gt} sheets", (2, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.imwrite(args.strip, np.vstack([head, fig]))
        print(f"strip -> {os.path.abspath(args.strip)}  GT {n_gt}")


if __name__ == "__main__":
    main()
