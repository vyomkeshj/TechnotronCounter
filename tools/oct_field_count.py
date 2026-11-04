"""Per-TYPE count check on the October field frames that NO model trained on (label-free, honest).

    python -m tools.oct_field_count [--models <dir>] [--name tag] [--exclude-labelled data/oct26_autolabel/frames ...]

Frames: every October jpg (goodshots, mainshots, type3). Excluded: frames with labels in data/oct26/frames (human,
trained on) or any --exclude-labelled folder (auto labels, trained on), and near-duplicates of those
(thumbnail NCC >= 0.97, same camera). Each complete column the pipeline reports is compared with the product
count of its session (data/oct26/pseudo_counts.json). Report per type: complete columns, exact %, MAE.
Writes runs/_oct_field/<name>.csv.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re

import cv2
import numpy as np

import paths
from tools.evalkit import sheet_counter

OCT = r"D:\Projects\technotron\datasets\october26_20261022\October26"


def thumb(p):
    g = cv2.cvtColor(cv2.resize(cv2.imread(p), (160, 120), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    t = g.astype(np.float32).ravel()
    return (t - t.mean()) / (t.std() + 1e-6)


def sharpness(img):
    """gradient energy / contrast in the centre: < 0.35 = motion-blurred / defocused (sheets not resolvable by eye)"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    c = g[int(H * .4):int(H * .8), int(W * .3):int(W * .6)]
    gx, gy = cv2.Sobel(c, cv2.CV_32F, 1, 0), cv2.Sobel(c, cv2.CV_32F, 0, 1)
    return float(np.mean(np.hypot(gx, gy)) / (c.std() + 1))


def type_of(stem):
    m = re.search(r"type\d+", stem)
    return m.group(0) if m else "mainshots"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None)
    ap.add_argument("--name", default="shipped")
    ap.add_argument("--pin", default=None, help="json list of frame stems: score exactly these (fixed set across runs)")
    ap.add_argument("--exclude-labelled", action="append",
                    default=[os.path.join(paths.DATA, "oct26", "frames"),
                             os.path.join(paths.DATA, "oct26_autolabel", "frames")])
    a = ap.parse_args()
    counts = json.load(open(os.path.join(paths.DATA, "oct26", "pseudo_counts.json")))
    train = []
    for d in a.exclude_labelled:
        train += glob.glob(os.path.join(d, "**", "*.jpg"), recursive=True)
    tstems = {os.path.splitext(os.path.basename(p))[0] for p in train}
    tth = [thumb(p) for p in train]
    frames = []
    pin = set(json.load(open(a.pin))) if a.pin else None
    if pin and pin & tstems:
        raise SystemExit(f"pinned frames are labelled (trained on): {sorted(pin & tstems)}")
    for p in sorted(glob.glob(os.path.join(OCT, "*", "*.jpg"))):
        s = os.path.splitext(os.path.basename(p))[0]
        if pin is not None:
            if s in pin:
                frames.append((s, p))
            continue
        if s in tstems:
            continue
        t = thumb(p)
        if any(float(t @ q) / len(t) >= 0.97 for q in tth):
            continue
        frames.append((s, p))
    sc = sheet_counter(a.models)
    rows = []
    agg = collections.defaultdict(list)
    for s, p in frames:
        exp = next((v for k, v in counts.items() if k in s), 50 if s.startswith("2026-10-02_") else None)   # mainshots: all 50
        typ = type_of(s)
        img = cv2.imread(p)
        sh = sharpness(img)
        cols = sc.process(img, typ if typ != "mainshots" else "type3", masks=False, check=True)
        for c in cols:
            if c.cropped:
                continue
            err = (c.count or 0) - exp                 # count None = the strip could not be cut: a miss
            rows.append((s, typ, exp, c.count, err, round(sh, 2)))
            agg[typ].append((err, sh >= 0.35))
    os.makedirs(os.path.join(paths.RUNS, "_oct_field"), exist_ok=True)
    with open(os.path.join(paths.RUNS, "_oct_field", a.name + ".csv"), "w") as f:
        f.write("frame,type,expected,count,err,sharpness\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    print(f"{'type':10s} {'frames':>6s} {'complete':>8s} {'exact':>7s} {'MAE':>6s}   | sharp frames only: {'complete':>8s} {'exact':>7s} {'MAE':>6s}")
    allr = []
    for typ in sorted(agg):
        e = np.array([v[0] for v in agg[typ]])
        k = np.array([v[1] for v in agg[typ]])
        allr += agg[typ]
        nf = len({r[0] for r in rows if r[1] == typ})
        es = e[k]
        print(f"{typ:10s} {nf:6d} {len(e):8d} {100 * np.mean(e == 0):6.1f}% {np.mean(np.abs(e)):6.2f}   | "
              f"{len(es):8d} {100 * np.mean(es == 0) if len(es) else 0:6.1f}% {np.mean(np.abs(es)) if len(es) else 0:6.2f}")
    e = np.array([v[0] for v in allr])
    k = np.array([v[1] for v in allr])
    print(f"{'all':10s} {len(frames):6d} {len(e):8d} {100 * np.mean(e == 0):6.1f}% {np.mean(np.abs(e)):6.2f}   | "
          f"{k.sum():8d} {100 * np.mean(e[k] == 0):6.1f}% {np.mean(np.abs(e[k])):6.2f}")

if __name__ == "__main__":
    main()
