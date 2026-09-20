"""Instance metrics (the project's evaluation protocol).

instance IoU = mean over GT instances of the best IoU with any prediction
count        = exact count accuracy and MAE
merge rate   = extra GT instances glued into one prediction / #GT

The val sets are small: single-checkpoint scores carry ~+-0.05 noise. Rank runs
by late-checkpoint medians or replicates, never by one peak.
"""
from __future__ import annotations

import numpy as np


def instance_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    gt_ids = [i for i in np.unique(gt) if i]
    pr_ids = [i for i in np.unique(pred) if i]
    K, P = len(gt_ids), len(pr_ids)
    pred_fg, gt_fg = pred > 0, gt > 0
    recalls, ious, owner = [], [], {}
    for gi in gt_ids:
        g = gt == gi
        recalls.append((g & pred_fg).sum() / (g.sum() + 1e-6))
        overl = pred[g]
        best, bj = 0.0, -1
        for pj in np.unique(overl[overl > 0]):
            p = pred == pj
            iou = (g & p).sum() / ((g | p).sum() + 1e-6)
            if iou > best:
                best, bj = iou, pj
        ious.append(best)
        if bj >= 0:
            owner.setdefault(bj, []).append(gi)
    merges = sum(len(v) - 1 for v in owner.values() if len(v) > 1)
    return {"inst_iou": float(np.mean(ious)) if ious else float(P == 0),
            "per_col_recall": float(np.mean(recalls)) if recalls else 1.0,
            "pix_iou": float((pred_fg & gt_fg).sum() / ((pred_fg | gt_fg).sum() + 1e-6)),
            "n_pred": P, "n_gt": K, "count_err": P - K,
            "merge_rate": merges / K if K else 0.0}


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    err = np.array([r["count_err"] for r in rows], float)
    out = {k: float(np.mean([r[k] for r in rows])) for k in ("inst_iou", "per_col_recall", "pix_iou", "merge_rate")}
    out.update(count_acc=float((err == 0).mean()), count_mae=float(np.abs(err).mean()),
               count_bias=float(err.mean()), within1=float((np.abs(err) <= 1).mean()), n=len(rows))
    if "cls_correct" in rows[0]:
        tot = sum(r["cls_total"] for r in rows)
        out["cls_acc"] = sum(r["cls_correct"] for r in rows) / max(1, tot)
    return out
