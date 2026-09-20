"""Flow readout vs grow-from-centre readout: leak, contact, coverage, count.

    python -m tools.measure_leak <run> [<run> ...] [--render out.jpg]

Scored against the HUMAN labels on the pinned val crops, per type:
  leak px/inst   pixels of a predicted sheet whose human label says a DIFFERENT
                 sheet (absolute px, not a fraction -- fractions are confounded
                 by mask width)
  contact px     pixels where two predicted instances touch (a groove should
                 separate every pair)
  covered %      share of the model's foreground that got an instance id
  count exact    unchanged-count check (the count must not move)
"""
from __future__ import annotations

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
from common.flows import follow_flows

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def contact_px(lab: np.ndarray) -> int:
    m = np.zeros(lab.shape, bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        h, w = lab.shape
        a = lab[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
        b = lab[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)]
        m[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)] |= (a > 0) & (b > 0) & (a != b)
    return int(m.sum())


def split_sheets(lab: np.ndarray, gt: np.ndarray, frac: float = 0.05) -> tuple[int, int]:
    """(human sheets covered by >1 predicted id, human sheets): the other direction of
    the error -- leak_px only catches a prediction holding two sheets."""
    bad = n = 0
    for i in range(1, int(gt.max()) + 1):
        m = (gt == i) & (lab > 0)
        if not m.any():
            continue
        n += 1
        counts = sorted((int((lab[m] == v).sum()) for v in np.unique(lab[m]) if v), reverse=True)
        if len(counts) > 1 and counts[1] > frac * sum(counts):
            bad += 1
    return bad, n


def leak_px(lab: np.ndarray, gt: np.ndarray) -> tuple[int, int]:
    """(leaked px, instances scored): per predicted sheet, px whose human id differs
    from that sheet's majority human id."""
    leaked = n = 0
    for i in range(1, int(lab.max()) + 1):
        m = (lab == i) & (gt > 0)
        if not m.any():
            continue
        vals, counts = np.unique(gt[m], return_counts=True)
        leaked += int(counts.sum() - counts.max())
        n += 1
    return leaked, n


def main():
    argv = sys.argv[1:]
    render = None
    if "--render" in argv:
        i = argv.index("--render")
        render = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    args = [a for a in argv if not a.startswith("--")]
    items, _ = discover()
    split = sync_split(items, write=False)
    val = build_val([it for it in items if it["frame"] in set(split["val"])], "center")
    for run in args:
        net = load_checkpoint(f"{paths.RUNS}/{run}/last.pt", DEVICE)
        acc = {}
        shots = []
        for it in val:
            prob, flow = infer(net, it["img"], DEVICE)
            fg = prob > THR
            flow_lab = follow_flows(fg, flow)
            grow_lab, n_grow = grow_instances(fg)
            for name, lab, n in (("flow", flow_lab, int(flow_lab.max())), ("grow", grow_lab, n_grow)):
                lk, ninst = leak_px(lab, it["gt"])
                sp, nsheets = split_sheets(lab, it["gt"])
                d = acc.setdefault((it["type"], name), dict(leak=0, inst=0, contact=0, cov=0.0, fgpx=0, exact=0, n=0,
                                                            split=0, sheets=0))
                d["split"] += sp; d["sheets"] += nsheets
                d["leak"] += lk; d["inst"] += ninst; d["contact"] += contact_px(lab)
                d["cov"] += float((lab > 0).sum()); d["fgpx"] += int(fg.sum())
                d["exact"] += int(n == it["n_gt"]); d["n"] += 1
            if render and len(shots) < 3 and it["type"] in ("type2", "type3", "type10"):
                if not any(s[0] == it["type"] for s in shots):
                    shots.append((it["type"], it["img"], flow_lab, grow_lab, band(fg)))
        print(f"\n{run}")
        for typ in sorted({t for t, _ in acc}):
            for name in ("flow", "grow"):
                d = acc[(typ, name)]
                print(f"  {typ:7s} {name:5s} leak {d['leak'] / max(d['inst'], 1):6.2f} px/inst | "
                      f"sheets split {100 * d['split'] / max(d['sheets'], 1):4.0f} % | "
                      f"contact {d['contact'] / d['n']:7.1f} px/crop | covered {100 * d['cov'] / max(d['fgpx'], 1):5.1f} % "
                      f"| count exact {d['exact']}/{d['n']}")
        if render and shots:
            rows = []
            for typ, img, fl, gr, (a, b) in shots:
                ys = np.flatnonzero(img.max(axis=(1, 2)) > 0)
                y0 = ys[0] + (len(ys) - 120) // 2
                panes = []
                for name, lab in (("flow readout", fl), ("grow readout", gr)):
                    v = cv2.addWeighted(img, 0.4, colorize(lab)[:, :, ::-1], 0.6, 0)
                    v[lab == 0] = (img[lab == 0] * 0.5).astype(np.uint8)
                    cm = np.zeros(lab.shape, bool)
                    for dy, dx in ((0, 1), (1, 0)):
                        h, w = lab.shape
                        p = lab[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
                        q = lab[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)]
                        cm[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)] |= (p > 0) & (q > 0) & (p != q)
                    v[cm] = (255, 255, 255)
                    t = cv2.resize(v[y0:y0 + 120], None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
                    head = np.zeros((30, t.shape[1], 3), np.uint8)
                    cv2.putText(head, f"{typ} {name}", (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                    panes.append(np.vstack([head, t]))
                src = cv2.resize(img[y0:y0 + 120], None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
                src = np.vstack([np.zeros((30, src.shape[1], 3), np.uint8), src])
                rows.append(np.hstack([src, np.full((src.shape[0], 8, 3), 255, np.uint8)] +
                                      [np.hstack([p, np.full((p.shape[0], 8, 3), 255, np.uint8)]) for p in panes]))
            W = max(r.shape[1] for r in rows)
            cv2.imwrite(render, np.vstack([cv2.copyMakeBorder(r, 0, 10, 0, W - r.shape[1],
                                                              cv2.BORDER_CONSTANT, value=(30, 30, 30)) for r in rows]))
            print("render ->", render)


if __name__ == "__main__":
    main()
