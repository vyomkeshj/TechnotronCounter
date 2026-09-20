"""Evaluate a column checkpoint on the pinned val frames, per group.

    python -m columns.evaluate runs/columns_<name>/stage2/last.pt [--render out_dir]

Groups: closeup, operational, type2, type3, type10 (whatever val contains), plus
'deploy' = pooled over the quality-dataset types (the deployment camera).
Class accuracy: mean P(cropped) over each GT instance's pixels vs its label.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch

from common.flows import follow_flows
from common.imageio import colorize, to_rgb_uint8
from common.metrics import aggregate, instance_metrics
from common.unet import load_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEPLOY_GROUPS = ("type2", "type3", "type10")


@torch.no_grad()
def predict(net, x: torch.Tensor):
    o = net(x[None].to(DEVICE))[0].cpu().numpy()
    fg = 1 / (1 + np.exp(-o[0])) > 0.5
    lab = follow_flows(fg, o[1:3])
    p_crop = 1 / (1 + np.exp(-o[3])) if o.shape[0] >= 4 else None
    return lab, p_crop


def evaluate(net, val: list[dict], render_dir: str | None = None) -> dict:
    net.eval()
    rows: dict[str, list] = {}
    for it in val:
        lab, p_crop = predict(net, it["x"])
        r = instance_metrics(lab, it["gt"])
        if p_crop is not None:
            ok = tot = 0
            for i in np.unique(it["gt"]):
                if i:
                    m = it["gt"] == i
                    ok += int((p_crop[m].mean() > 0.5) == bool(it["lut"][i]))
                    tot += 1
            r.update(cls_correct=ok, cls_total=tot)
        rows.setdefault(it["group"], []).append(r)
        if it["group"] in DEPLOY_GROUPS:
            rows.setdefault("deploy", []).append(r)
        if render_dir:
            os.makedirs(render_dir, exist_ok=True)
            rgb = to_rgb_uint8(it["x"])
            panel = np.hstack([rgb, colorize(it["gt"]), colorize(lab)])[:, :, ::-1]
            cv2.imwrite(os.path.join(render_dir, f"{it['group']}__{it['stem']}__n{r['n_pred']}_gt{r['n_gt']}.png"),
                        cv2.resize(panel, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    return {g: aggregate(rs) for g, rs in rows.items()}


def preview(net, val: list[dict], n_per_group: int = 2) -> np.ndarray:
    """CHW uint8 grid: input | GT | prediction, a couple of frames per group."""
    picked, seen = [], {}
    for it in val:
        if seen.get(it["group"], 0) < n_per_group:
            picked.append(it); seen[it["group"]] = seen.get(it["group"], 0) + 1
    rows = []
    for it in picked:
        lab, _ = predict(net, it["x"])
        rows.append(np.hstack([to_rgb_uint8(it["x"]), colorize(it["gt"]), colorize(lab)]))
    return np.vstack(rows).transpose(2, 0, 1)


def main():
    from columns.dataset import build_val, discover, sync_split
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--render", default=None)
    a = ap.parse_args()
    items = discover()
    split = sync_split(items, write=False)
    val = build_val([it for it in items if it["stem"] in set(split["val"])])
    res = evaluate(load_checkpoint(a.checkpoint, DEVICE), val, a.render)
    for g, m in sorted(res.items()):
        print(f"{g:12s} n={m['n']:3d} instIoU {m['inst_iou']:.3f} countAcc {m['count_acc']:.2f} "
              f"MAE {m['count_mae']:.2f} merge {m['merge_rate']:.3f}" + (f" cls {m['cls_acc']:.3f}" if "cls_acc" in m else ""))
    if a.render:
        json.dump(res, open(os.path.join(a.render, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
