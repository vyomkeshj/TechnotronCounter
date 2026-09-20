"""Promotion gate for the COLUMN model.

Same rule as `field.gate`: a candidate may only go live if no protected group drops by
more than the MEASURED replicate spread. Bands come from the two COLQ replicates
(`runs/COLQ_a_s2` / `COLQ_b_s2`, seeds 42 / 7, 2026-09-16):

    instIoU   0.903 / 0.896   spread 0.007  -> tol 0.02
    countAcc  0.85  / 0.80    spread 0.05   -> tol 0.05  (2 frames of 40)
    cls       0.953 / 0.953   spread 0.000  -> tol 0.02

Unlike the edge model, mask QUALITY is gated here and not just the count: these masks
feed the straightener, so a column mask that drifts produces a bad crop for every
downstream edge label. instIoU is a ranking metric for columns -- it is NOT for edges.
"""
from __future__ import annotations

import torch

from columns.dataset import build_val, discover, sync_split
from columns.evaluate import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOL = {"inst_iou": 0.02, "count_acc": 0.05, "cls_acc": 0.02}
KEYS = ("inst_iou", "count_acc", "cls_acc")


def build_sets(exclude_stems=()):
    """Pinned column val frames, minus the newly added ones (they are training data now)."""
    items = discover()
    split = sync_split(items, write=False)
    ex = set(exclude_stems)
    val_stems = set(split["val"]) - ex
    return build_val([it for it in items if it["stem"] in val_stems])


def score(net, val) -> dict:
    return {g: {k: float(m[k]) for k in KEYS if k in m} | {"n": int(m["n"])}
            for g, m in evaluate(net, val).items()}


def verdict(before: dict, after: dict, site_groups: set[str] | None = None):
    rows, passed = [], True
    for g in sorted(before):
        for k in KEYS:
            if k not in before[g] or k not in after.get(g, {}):
                continue
            b, a = before[g][k], after[g][k]
            d = a - b
            scoped = site_groups is None or g in site_groups or g == "deploy"
            if not scoped:
                status = "out of scope for this site"
            elif d < -TOL[k]:
                status = f"REGRESSED (> tol {TOL[k]})"
                passed = False
            else:
                status = f"ok (tol {TOL[k]})"
            rows.append({"set": f"{g}/{k}", "n": before[g]["n"], "before": b, "after": a,
                         "delta": d, "status": status})
    return passed, rows


def table(rows) -> str:
    w = max(len(r["set"]) for r in rows) + 2
    out = [f"{'group/metric':<{w}}{'n':>5}{'before':>9}{'after':>8}{'delta':>8}   status"]
    for r in rows:
        out.append(f"{r['set']:<{w}}{r['n']:>5}{r['before']:>9.3f}{r['after']:>8.3f}"
                   f"{r['delta']:>+8.3f}   {r['status']}")
    return "\n".join(out)
