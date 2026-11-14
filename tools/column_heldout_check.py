"""Column stage on the HELD-OUT frames of every labelled type (data/columns/splits.json "val"), per type:
labelled columns found (IoU >= 0.5), mean outline IoU, complete/cropped accuracy, both kinds of class error and
columns the model adds that match no label. Counts are not scored here (tools.evaluate does that for 2/3/10).

    python -m tools.column_heldout_check [--models <dir>]

Type7 has one session per product variant, so its held-out frames are a block of the same session
(2026-11-14 assignment): report type7 as SAME-SESSION.
"""
from __future__ import annotations

import argparse
import collections
import json
import re

import cv2

import paths
from columns.dataset import discover
from common import labelme
from tools.evalkit import iou, sheet_counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None)
    ap.add_argument("--turn", type=int, default=0, choices=[0, 90, 180, 270], help="turn every frame (and its labels)")
    args = ap.parse_args()
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(args.turn)
    sc = sheet_counter(args.models)
    val = set(json.load(open(paths.COLUMNS_SPLIT))["val"])
    st = collections.defaultdict(collections.Counter)
    for it in discover():
        if it["stem"] not in val:
            continue
        m = re.search(r"type\d+", it["stem"])
        typ = m.group(0) if m else "other"
        img = cv2.imread(it["image"])
        H, W = img.shape[:2]
        lab, lut = labelme.column_labels(it["json"], hw=(H, W))
        if code is not None:
            img, lab = cv2.rotate(img, code), cv2.rotate(lab, code)
        cols = sc.segment_columns(img, typ if typ.startswith("type") else "type3")
        used = set()
        for i in range(1, lab.max() + 1):
            g = lab == i
            s = st[typ]
            s["labelled"] += 1
            k = max(range(len(cols)), key=lambda k: iou(cols[k].mask, g), default=None)
            if k is None or iou(cols[k].mask, g) < 0.5:
                continue
            used.add(k)
            crop = lut[i] > 0.5
            s["found"] += 1
            s["iou"] += iou(cols[k].mask, g)
            s["class_ok"] += cols[k].cropped == crop
            s["complete_as_cropped"] += (not crop) and cols[k].cropped
            s["cropped_as_complete"] += crop and not cols[k].cropped
        st[typ]["extra"] += len(cols) - len(used)
        st[typ]["frames"] += 1
    print(f"{'type':7s} {'frames':>6s} {'labelled':>8s} {'found':>6s} {'IoU':>5s} {'class ok':>9s} "
          f"{'complete->crop':>14s} {'crop->complete':>14s} {'extra':>6s}")
    tot = collections.Counter()
    for t in sorted(st, key=lambda t: (len(t), t)):
        s = st[t]
        tot.update(s)
        print(f"{t:7s} {s['frames']:6d} {s['labelled']:8d} {s['found']:6d} {s['iou'] / max(1, s['found']):5.3f} "
              f"{100 * s['class_ok'] / max(1, s['found']):8.1f}% {s['complete_as_cropped']:14d} "
              f"{s['cropped_as_complete']:14d} {s['extra']:6d}")
    s = tot
    print(f"{'ALL':7s} {s['frames']:6d} {s['labelled']:8d} {s['found']:6d} {s['iou'] / max(1, s['found']):5.3f} "
          f"{100 * s['class_ok'] / max(1, s['found']):8.1f}% {s['complete_as_cropped']:14d} "
          f"{s['cropped_as_complete']:14d} {s['extra']:6d}")


if __name__ == "__main__":
    main()
