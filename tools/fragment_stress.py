"""Margin test for broken sheet masks: tools.fragment_eval under input perturbations.

    python -m tools.fragment_stress <ckptA.pt> <ckptB.pt> [--sets new_holdout,old_val] [--json out.json]

When the clean benchmark saturates (every candidate ~0 broken sheets), rank candidates by how
well they keep each sheet in ONE piece on degraded photos: gaussian noise, blur, JPEG,
low contrast, gamma, glare spots and a texture overlay. Perturbations are seeded and identical
for every candidate, and never look at the label.
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from common import labelme
from edges.dataset import fit_width, tight
from tools import fragment_eval as fe


def perturb(img: np.ndarray, kind: str, rng: np.random.RandomState) -> np.ndarray:
    f = img.astype(np.float32)
    m = img.max(axis=2) > 0
    if kind == "noise":
        f = f + rng.normal(0, 14, f.shape)
    elif kind == "blur":
        f = cv2.GaussianBlur(f, (0, 0), 1.3)
    elif kind == "jpeg":
        ok, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 25])
        f = cv2.imdecode(b, cv2.IMREAD_COLOR).astype(np.float32)
    elif kind == "contrast":
        f = (f - f[m].mean()) * 0.6 + f[m].mean()
    elif kind == "dark":
        f = 255 * (f / 255) ** 1.6
    elif kind == "glare":
        h, w = m.shape
        g = np.zeros((h, w), np.float32)
        for _ in range(max(1, h // 120)):
            cv2.circle(g, (rng.randint(0, w), rng.randint(0, h)), rng.randint(8, 20), 1.0, -1)
        f = f + cv2.GaussianBlur(g, (0, 0), 5)[..., None] * 140
    elif kind == "texture":
        t = cv2.GaussianBlur(rng.normal(0, 1, m.shape).astype(np.float32), (0, 0), 1.0)
        f = f * (1 + 0.35 * t / (t.std() + 1e-6))[..., None]
    out = np.clip(f, 0, 255).astype(np.uint8)
    out[~m] = 0
    return out


KINDS = ("noise", "blur", "jpeg", "contrast", "dark", "glare", "texture")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--sets", default="new_holdout,old_val")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    nets = [fe.load_checkpoint(c, fe.DEVICE) for c in a.ckpts]
    out = {}
    for s in a.sets.split(","):
        items = fe.old_val_items() if s == "old_val" else fe.oct26_items(s)
        data = []
        for it in items:
            img = cv2.imread(it["image"]); lab = labelme.edge_labels(it["json"])
            data.append((it, *fit_width(*tight(img, lab), "center")))
        for kind in KINDS:
            rng = np.random.RandomState(0)
            rows = []
            for it, img, lab in data:
                x = perturb(img, kind, rng)
                rows.append(fe.score_arrays(nets, it, x, lab))
            m = fe.summarise(rows)
            out[f"{s}/{kind}"] = m
            print(f"{s:12s} {kind:9s} broken {m['broken_pct']:5.1f}%  exact {m['count_exact']}/{m['strips']}  "
                  f"MAE {m['count_mae']:.2f}  coverage {m['coverage']:.3f}", flush=True)
        tot = [out[f"{s}/{k}"] for k in KINDS]
        print(f"{s:12s} MEAN      broken {np.mean([t['broken_pct'] for t in tot]):5.1f}%  "
              f"exact {np.mean([t['count_exact'] / t['strips'] for t in tot]):.1%}  MAE {np.mean([t['count_mae'] for t in tot]):.2f}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
