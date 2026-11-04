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
    for split, old in (("train", "fit"), ("test", "test")):    # earlier hand-made set, minus its frame crops
        for p in glob.glob(os.path.join(paths.RUNS, "_wexp", "negatives", old, "*.jpg")):
            if not os.path.basename(p).startswith("bg_"):
                shutil.copy(p, os.path.join(NEG, split, "misc", os.path.basename(p)))
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
