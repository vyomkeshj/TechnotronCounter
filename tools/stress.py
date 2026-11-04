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
    def tilt(f, axis):
        """Camera turned about the horizontal (pitch) or vertical (yaw) axis: one image side shrinks by f."""
        def go(img):
            H, W = img.shape[:2]
            src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
            d = f * (W if axis == "pitch" else H) / 2
            dst = (np.float32([[d, 0], [W - d, 0], [W, H], [0, H]]) if axis == "pitch"
                   else np.float32([[0, d], [W, 0], [W, H], [0, H - d]]))
            if f < 0:
                dst = (np.float32([[0, 0], [W, 0], [W + d, H], [-d, H]]) if axis == "pitch"
                       else np.float32([[0, 0], [W, -d], [W, H + d], [0, H]]))
            Mp = cv2.getPerspectiveTransform(src, dst)
            return cv2.warpPerspective(img, Mp, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), Mp
        return go
    def photo(fn):
        return lambda img: (fn(img), ident)
    def jpeg(q):
        return lambda img: cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    rng = np.random.RandomState(0)
    def exposure(ev):
        """Camera exposure off by `ev` stops: scale in linear light, clip (blown highlights / crushed shadows),
        back to 8-bit sRGB (dark frames keep few grey levels)."""
        def f(x):
            lin = (x.astype(np.float32) / 255.0) ** 2.2 * (2.0 ** ev)
            return np.round(255.0 * np.clip(lin, 0, 1) ** (1 / 2.2)).astype(np.uint8)
        return photo(f)
    return {
        "exposure -2.5EV": exposure(-2.5), "exposure -2EV": exposure(-2), "exposure -1EV": exposure(-1),
        "exposure +1EV": exposure(1), "exposure +1.5EV": exposure(1.5), "exposure +2EV": exposure(2),
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
        # wider camera placement (2026-11-03): farther away, rolled, tilted (pitch / yaw = perspective)
        "zoom 0.6": scale(0.6), "zoom 0.75": scale(0.75), "zoom 1.2": scale(1.2),
        "rotate -10": rot(-10), "rotate +10": rot(10), "rotate -20": rot(-20), "rotate +20": rot(20),
        "pitch 10%": tilt(0.10, "pitch"), "pitch -10%": tilt(-0.10, "pitch"), "yaw 10%": tilt(0.10, "yaw"),
        "yaw -10%": tilt(-0.10, "yaw"), "pitch 20%": tilt(0.20, "pitch"), "yaw 20%": tilt(0.20, "yaw"),
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
                q = p if M is None else (cv2.perspectiveTransform(p[None].astype(np.float32), M)[0] if M.shape == (3, 3)
                                         else np.c_[p, np.ones(len(p))] @ M.T)
                if M is not None and (q[:, 0].min() < 2 or q[:, 1].min() < 2 or q[:, 0].max() > W - 3 or q[:, 1].max() > H - 3):
                    continue                                  # column (partly) pushed out of the frame: no longer complete
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
