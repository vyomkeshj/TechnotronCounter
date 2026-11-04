"""False columns on column-FREE images: any column the package reports is a false positive.

    python -m tools.negative_check [--models <dir>] [--name tag]
    python -m tools.negative_check --build        # (re)make the sets from the downloads, once

Sets (paths.NEGATIVES = datasets/negatives_web):
  test/places   1 Places365 val image per scene category (365), never in train/
  test/bg_val   crops of HELD-OUT column frames (splits.json val + type8 018/021) >= 25 px from any
                labelled column: the plant scene itself
  test/misc     photos, skimage samples, synthetic patterns (stripes, checkers, noise)
  train/places  the other 90 Places images per category  -- columns.train --negatives
  train/misc    a disjoint set of photos / samples / synthetic patterns
Training background crops come from TRAINING frames on the fly (columns.train --bg-frac), never from here.
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
import random
import shutil

import cv2
import numpy as np

import paths
from tools.evalkit import sheet_counter

NEG = os.path.join(os.path.dirname(paths.ROOT), "datasets", "negatives_web")


def misc_regenerate(rnd):
    """60 synthetic patterns (stripes at any angle, checkers, gratings, noise, gradients, blurred blobs; the
    repetitive ones are what fool a sheet-column model) + the scikit-image sample photos; 1 in 3 to test."""
    rng = np.random.default_rng(0)
    out = []
    for i in range(60):
        h, w = rng.integers(300, 700), rng.integers(300, 900)
        kind = ("stripes", "checker", "grating", "noise", "gradient", "blobs")[i % 6]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        if kind in ("stripes", "grating"):
            a, p = rng.uniform(0, np.pi), rng.uniform(6, 30)
            t = np.sin((xx * np.cos(a) + yy * np.sin(a)) * 2 * np.pi / p)
            g = (t > 0).astype(np.float32) if kind == "stripes" else (t + 1) / 2
        elif kind == "checker":
            p = int(rng.integers(8, 60))
            g = (((yy // p) + (xx // p)) % 2).astype(np.float32)
        elif kind == "noise":
            g = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), float(rng.uniform(0.5, 4)))
        elif kind == "gradient":
            g = (xx / w * rng.uniform(0.3, 1) + yy / h * rng.uniform(0, 0.7))
        else:
            g = cv2.GaussianBlur(rng.random((h // 16 + 1, w // 16 + 1)).astype(np.float32), (0, 0), 1)
            g = cv2.resize(g, (int(w), int(h)))
        g = (g - g.min()) / (np.ptp(g) + 1e-6)
        col = np.dstack([g * rng.uniform(0.3, 1) + rng.uniform(0, 0.3) for _ in range(3)])
        out.append((f"syn_{kind}_{i}.jpg", np.clip(col * 255, 0, 255).astype(np.uint8)))
    try:
        from skimage import data as skd
        for name in ("astronaut", "coffee", "chelsea", "rocket", "camera", "coins", "brick", "grass", "gravel",
                     "rocket", "page", "text", "moon", "horse", "retina", "hubble_deep_field", "immunohistochemistry"):
            try:
                im = getattr(skd, name)()
            except Exception:
                continue
            im = np.dstack([im] * 3) if im.ndim == 2 else im[..., :3]
            out.append((f"sk_{name}.jpg", cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)))
    except ImportError:
        print("scikit-image not installed: misc set is synthetic only")
    for k, (name, img) in enumerate(out):
        split = "test" if k % 3 == 0 else "train"
        cv2.imwrite(os.path.join(NEG, split, "misc", name), img)


def build():
    from columns.dataset import discover, sync_split
    from common import labelme
    rnd = random.Random(0)
    lines = open(os.path.join(NEG, "meta", "places365_val.txt")).read().split()
    by_cat = collections.defaultdict(list)
    for name, cat in zip(lines[0::2], lines[1::2]):
        by_cat[int(cat)].append(name)
    for split in ("train", "test"):
        for sub in ("places", "misc", "bg_val"):
            shutil.rmtree(os.path.join(NEG, split, sub), ignore_errors=True)
            os.makedirs(os.path.join(NEG, split, sub), exist_ok=True)
    src = os.path.join(NEG, "places", "val_256")
    for cat, names in sorted(by_cat.items()):
        names = sorted(names)
        rnd.shuffle(names)
        shutil.copy(os.path.join(src, names[0]), os.path.join(NEG, "test", "places", names[0]))
        for n in names[10:]:                          # 1..9 kept out of both (spare test pool)
            shutil.copy(os.path.join(src, n), os.path.join(NEG, "train", "places", n))
    old_dir = os.path.join(paths.RUNS, "_wexp", "negatives")
    if os.path.isdir(old_dir):     # the original 2026-11-04 set (its frame crops dropped): exact same numbers
        for split, old in (("train", "fit"), ("test", "test")):
            for p in glob.glob(os.path.join(old_dir, old, "*.jpg")):
                if not os.path.basename(p).startswith("bg_"):
                    shutil.copy(p, os.path.join(NEG, split, "misc", os.path.basename(p)))
    else:                          # regenerate: synthetic patterns + scikit-image samples (no wallpaper photos)
        misc_regenerate(rnd)
    items = discover()
    val = set(sync_split(items, write=False)["val"])
    frames = [(it["image"], it["json"], it["stem"]) for it in items if it["stem"] in val]
    for js in glob.glob(os.path.join(paths.DATA, "type8_autolabel", "frames", "**", "*.json"), recursive=True):
        if "39pieces_018" in js or "39pieces_021" in js:
            frames.append((labelme.image_for(js), js, os.path.basename(js)[:-5]))
    n_bg = 0
    for img_p, js, stem in frames:
        img = cv2.imread(img_p)
        H, W = img.shape[:2]
        lab, _ = labelme.column_labels(js, hw=(H, W))
        near = cv2.dilate((lab > 0).astype(np.uint8), np.ones((51, 51), np.uint8))
        got = 0
        for _ in range(300):
            ch, cw = int(H * rnd.uniform(0.2, 0.6)), int(W * rnd.uniform(0.2, 0.6))
            y0, x0 = rnd.randint(0, H - ch), rnd.randint(0, W - cw)
            if not near[y0:y0 + ch, x0:x0 + cw].any():
                cv2.imwrite(os.path.join(NEG, "test", "bg_val", f"{stem}_{got}.jpg"), img[y0:y0 + ch, x0:x0 + cw])
                got += 1
                n_bg += 1
                if got == 2:
                    break
    for split in ("train", "test"):
        print(split, {s: len(os.listdir(os.path.join(NEG, split, s))) for s in ("places", "misc", "bg_val")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None)
    ap.add_argument("--name", default="shipped")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    if args.build:
        build()
        return
    sc = sheet_counter(args.models)
    st = collections.defaultdict(lambda: [0, 0, 0])
    for sub in ("places", "bg_val", "misc"):
        for p in sorted(glob.glob(os.path.join(NEG, "test", sub, "*"))):
            cols = sc.segment_columns(cv2.imread(p), "type3")
            for k in (sub, "ALL"):
                st[k][0] += 1
                st[k][1] += bool(cols)
                st[k][2] += len(cols)
    for k in ("places", "bg_val", "misc", "ALL"):
        n, fp, nc = st[k]
        print(f"{k:7s} images {n:4d}  with a false column {fp:4d} ({100 * fp / max(1, n):5.1f} %)  false columns {nc}")


if __name__ == "__main__":
    main()
