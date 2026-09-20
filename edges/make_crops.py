"""Make straightened column crops for edge LABELLING from column-labelled frames.

    python -m edges.make_crops <frames_dir> [--out data/edges] [--dup-ncc 0.97]

<frames_dir>: LabelMe column frames (jpg + json with good_column/cropped_column),
searched recursively. Every GOOD column becomes one crop:
    data/edges/<type>/<frame>_column_<k>.png       k = left to right
native resolution, full width, fixed 2050 px height, background black.
Hand these PNGs to the labeller; labels come back as <same name>.json beside them.

Skipped, with the reason printed and written to <out>/make_crops_log.csv:
  * frames whose crops already exist in data/edges/<type>/ (never overwrite labels)
  * near-duplicate frames: thumbnail NCC >= --dup-ncc against an already-kept
    frame of the same capture session (the rack camera is fixed; neighbouring
    frames are near-identical and would just be labelled twice)
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import cv2
import numpy as np

import paths
from common import splits
from common.straighten import straighten_frame, to_native_canvas

cv2.setNumThreads(0)


def thumb(p):
    g = cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (192, 108), interpolation=cv2.INTER_AREA)
    t = g.astype(np.float32).ravel()
    return (t - t.mean()) / (t.std() + 1e-6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames_dir")
    ap.add_argument("--out", default=paths.EDGES_DATA)
    ap.add_argument("--dup-ncc", type=float, default=0.97)
    a = ap.parse_args()
    frames = []
    for js in sorted(glob.glob(os.path.join(a.frames_dir, "**", "*.json"), recursive=True)):
        jpg = js[:-5] + ".jpg"
        stem = os.path.splitext(os.path.basename(js))[0]
        if os.path.exists(jpg) and splits.session_of(stem):
            frames.append((stem, jpg, js))
    frames.sort(key=lambda f: (splits.session_of(f[0]), splits.index_of(f[0])))
    kept: dict[str, list] = {}
    log, n_crops = [], 0
    for stem, jpg, js in frames:
        typ, sess = splits.type_of(stem), splits.session_of(stem)
        out_dir = os.path.join(a.out, typ)
        if glob.glob(os.path.join(out_dir, f"{stem}_column_*.png")):
            log.append((stem, typ, "skip_exists", "")); continue
        th = thumb(jpg)
        best = max(((float(th @ t / len(th)), s) for s, t in kept.get(sess, [])), default=(0.0, ""))
        if best[0] >= a.dup_ncc:
            log.append((stem, typ, "skip_duplicate", f"ncc {best[0]:.3f} vs {best[1]}")); continue
        kept.setdefault(sess, []).append((stem, th))
        os.makedirs(out_dir, exist_ok=True)
        k = 0
        for crop in straighten_frame(cv2.imread(jpg), js):
            canvas = to_native_canvas(crop)
            if canvas is not None:
                cv2.imwrite(os.path.join(out_dir, f"{stem}_column_{k}.png"), canvas)
                k += 1
        n_crops += k
        log.append((stem, typ, "kept", f"{k} columns"))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "make_crops_log.csv"), "a", newline="") as f:
        csv.writer(f).writerows(log)
    for status in ("kept", "skip_duplicate", "skip_exists"):
        print(f"{status:15s} {sum(1 for r in log if r[2] == status)} frames")
    print(f"{n_crops} new crops -> {a.out}")


if __name__ == "__main__":
    main()
