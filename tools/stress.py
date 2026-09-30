"""Robustness of the inference package to real-world variation: the held-out test frames under
controlled perturbations (photometric and small geometric), scored like tools.evaluate.

    python -m tools.stress [--models <dir>] [--only clean,zoom_0.9,...]

Geometric perturbations transform the labelled polygons with the image, so matching stays exact.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import cv2
import numpy as np

import paths
from common.labelme import column_label
from tools.evalkit import expected, iou, sheet_counter, test_frames


def affine(img, M):
    H, W = img.shape[:2]
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def perturbations():
    ident = None
    def rot(deg):
        return lambda img: (affine(img, cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), deg, 1.0)),
                            cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), deg, 1.0))
    def scale(f):
        return lambda img: (affine(img, cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 0, f)),
                            cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 0, f))
    def photo(fn):
        return lambda img: (fn(img), ident)
    def jpeg(q):
        return lambda img: cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    rng = np.random.RandomState(0)
    return {
        "clean": photo(lambda x: x),
        "darker x0.6": photo(lambda x: np.clip(x * 0.6, 0, 255).astype(np.uint8)),
        "brighter x1.4": photo(lambda x: np.clip(x * 1.4, 0, 255).astype(np.uint8)),
        "low contrast": photo(lambda x: np.clip((x - 128.0) * 0.6 + 128, 0, 255).astype(np.uint8)),
        "noise s10": photo(lambda x: np.clip(x + rng.normal(0, 10, x.shape), 0, 255).astype(np.uint8)),
        "jpeg q30": photo(jpeg(30)),
        "blur k3": photo(lambda x: cv2.GaussianBlur(x, (3, 3), 0)),
        "warm cast": photo(lambda x: np.clip(x * np.array([0.85, 1.0, 1.15]), 0, 255).astype(np.uint8)),
        "rotate -4": rot(-4), "rotate +4": rot(4),
        "zoom 0.9": scale(0.9), "zoom 0.95": scale(0.95), "zoom 1.05": scale(1.05), "zoom 1.1": scale(1.1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None)
    ap.add_argument("--only", default="", help="comma list of perturbation names")
    ap.add_argument("--name", default="shipped")
    args = ap.parse_args()
    sc = sheet_counter(args.models)
    frames = test_frames()
    data = []
    for stem in frames:
        typ = re.search(r"(type\d+)", stem).group(1)
        jpg = os.path.join(paths.COLUMNS_DATA, "quality", typ, stem + ".jpg")
        polys = [np.array(s["points"], np.float64) for s in json.load(open(jpg[:-4] + ".json", encoding="utf-8"))["shapes"]
                 if column_label(s["label"]) == "good_column"]
        data.append((stem, typ, cv2.imread(jpg), polys))
    results = {}
    only = [x.strip().replace("_", " ") for x in args.only.split(",") if x.strip()]
    for name, fn in perturbations().items():
        if only and name not in only:
            continue
        tot = {"type2": [0, 0], "type3": [0, 0], "type10": [0, 0]}
        for stem, typ, img, polys in data:
            im2, M = fn(img)
            H, W = im2.shape[:2]
            cols = sc.process(im2, typ, masks=False)
            exp = expected(stem)
            for p in polys:
                q = p if M is None else (np.c_[p, np.ones(len(p))] @ M.T)
                m = np.zeros((H, W), np.uint8)
                cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1)
                if m.sum() < 50:
                    continue                                  # column pushed out of the frame
                best = max(cols, key=lambda c: iou(c.mask, m > 0), default=None)
                ok = best is not None and iou(best.mask, m > 0) >= 0.5 and not best.cropped and best.count == exp
                tot[typ][0] += ok; tot[typ][1] += 1
        results[name] = tot
        a = sum(v[0] for v in tot.values()); n = sum(v[1] for v in tot.values())
        print(f"{name:14} " + "  ".join(f"{t} {v[0]:3}/{v[1]:<3}" for t, v in tot.items()) + f"  all {a / n:.3f}", flush=True)
    json.dump(results, open(os.path.join(paths.RUNS, f"_package_stress_{args.name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
