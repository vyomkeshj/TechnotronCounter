"""COMBINED labelling -> edge crops WITH their labels, in one step.

    python -m edges.from_frames <frames_dir> [--out data/edges] [--dup-ncc 0.97]

The labeller draws, on the FULL frame in one LabelMe session, the good_column /
cropped_column polygons and then one "edge" polygon per sheet inside each good column.
This turns such frames straight into the edge dataset's stored format:

    data/edges/<type>/<frame>_column_<k>.png     straightened column, native width, 2050 canvas
    data/edges/<type>/<frame>_column_<k>.json    the edge polygons, in crop coordinates

The image goes through `common.straighten.straighten_frame_pairs`, which is the
production straightener with the label map carried alongside through the identical
geometry (same angle, same x-limits, same column mask, nearest-neighbour for the map).
So a crop made here is pixel-identical to one made by `edges.make_crops`, and its label
is the frame's edge polygons seen through the same transform. Frames WITHOUT edge
polygons still get their PNGs (labels come later, the old workflow).

Near-duplicate frames within a capture session are skipped, as in make_crops.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import cv2
import numpy as np

import paths
from common import labelme, splits
from common.straighten import straighten_frame_pairs, to_native_canvas_pair
from edges.make_crops import thumb


def frame_to_crops(jpg: str, js: str, out_dir: str, stem: str, overwrite_labels: bool = False):
    """Write <stem>_column_<k>.png (+ .json if the frame carries edges). Returns (n_png, n_json)."""
    img = cv2.imread(jpg)
    if img is None:
        return 0, 0
    lab = labelme.frame_edge_labels(js)
    if lab.shape != img.shape[:2]:
        raise ValueError(f"{js}: label size {lab.shape} != image {img.shape[:2]}")
    os.makedirs(out_dir, exist_ok=True)
    n_png = n_json = 0
    for k, (crop, clab) in enumerate(straighten_frame_pairs(img, lab, js)):
        pair = to_native_canvas_pair(crop, clab)
        if pair is None:
            continue
        canvas, lcan = pair
        png = os.path.join(out_dir, f"{stem}_column_{k}.png")
        cv2.imwrite(png, canvas)
        n_png += 1
        if lcan.max() > 0:
            jpath = png[:-4] + ".json"
            if overwrite_labels or not os.path.exists(jpath):
                labelme.write_polygons(jpath, lcan, labelme.EDGE, os.path.basename(png))
                n_json += 1
    return n_png, n_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames_dir")
    ap.add_argument("--out", default=paths.EDGES_DATA)
    ap.add_argument("--dup-ncc", type=float, default=0.97)
    ap.add_argument("--type", default=None, help="force the type folder (else from the filename)")
    a = ap.parse_args()
    frames = []
    for js in sorted(glob.glob(os.path.join(a.frames_dir, "**", "*.json"), recursive=True)):
        img = labelme.image_for(js)
        stem = os.path.splitext(os.path.basename(js))[0]
        if img:
            frames.append((stem, img, js))
    frames.sort(key=lambda f: (splits.session_of(f[0]) or "", splits.index_of(f[0]) or 0))
    kept: dict[str, list] = {}
    log, tot_png, tot_json = [], 0, 0
    for stem, jpg, js in frames:
        typ = a.type or splits.type_of(stem) or "typeX"
        sess = splits.session_of(stem) or "?"
        out_dir = os.path.join(a.out, typ)
        if glob.glob(os.path.join(out_dir, f"{stem}_column_*.png")):
            log.append((stem, typ, "skip_exists", "")); continue
        th = thumb(jpg)
        best = max(((float(th @ t / len(th)), s) for s, t in kept.get(sess, [])), default=(0.0, ""))
        if best[0] >= a.dup_ncc:
            log.append((stem, typ, "skip_duplicate", f"ncc {best[0]:.3f} vs {best[1]}")); continue
        kept.setdefault(sess, []).append((stem, th))
        n_png, n_json = frame_to_crops(jpg, js, out_dir, stem)
        tot_png += n_png; tot_json += n_json
        log.append((stem, typ, "kept", f"{n_png} crops, {n_json} labelled"))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "make_crops_log.csv"), "a", newline="") as f:
        csv.writer(f).writerows(log)
    for status in ("kept", "skip_duplicate", "skip_exists"):
        print(f"{status:15s} {sum(1 for r in log if r[2] == status)} frames")
    print(f"{tot_png} crops ({tot_json} with edge labels) -> {a.out}")


if __name__ == "__main__":
    main()
