"""Column-stage audit on full frames that carry BOTH column and sheet-edge labels (data/oct26).

    python -m tools.column_field_eval [--models <dir>] [--sets holdout,train] [--json out.json]

Every labelled column is matched to the predicted column overlapping it most (IoU >= 0.5):
  good_ok        labelled complete, predicted complete
  good_cropped   labelled complete, predicted cropped (never counted -> a lost column)
  good_missed    labelled complete, no predicted column overlaps it
  false_good     predicted complete, but labelled cropped or unlabelled (counted although it should not be)
For good_ok columns: mask IoU with the label, and the sheet count of the full pipeline vs the
number of labelled edge polygons inside the labelled column ("count_exact").
Holdout = the same frames tools.fragment_eval holds out (whole capture bursts / sessions).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import cv2
import numpy as np

import paths
from common import labelme
from tools.evalkit import iou, sheet_counter
from tools.fragment_eval import HOLDOUT

FRAMES = os.path.join(paths.DATA, "oct26", "frames")


def frames(which: str) -> list[tuple[str, str]]:
    out = []
    for js in sorted(glob.glob(os.path.join(FRAMES, "*", "*.json"))):
        stem = os.path.splitext(os.path.basename(js))[0]
        if (which == "holdout") == any(h in stem for h in HOLDOUT):
            out.append((stem, js))
    return out


def edge_count_inside(js: str, mask: np.ndarray) -> int:
    shapes = json.load(open(js, encoding="utf-8"))["shapes"]
    H, W = mask.shape
    n = 0
    for s in shapes:
        if labelme.column_label(s["label"]) == labelme.EDGE:
            x, y = np.mean(s["points"], axis=0)
            n += 0 <= int(y) < H and 0 <= int(x) < W and bool(mask[int(y), int(x)])
    return n


def score_frame(sc, stem: str, js: str) -> dict:
    img = cv2.imread(labelme.image_for(js))
    m = re.search(r"type\d+", stem)
    cols = sc.process(img, m.group(0) if m else "type3", masks=False, check=True)
    lab, lut = labelme.column_labels(js, hw=img.shape[:2])
    gts = [(i, lab == i, lut[i] < 0.5) for i in np.unique(lab) if i > 0]
    r = {"stem": stem, "good_ok": 0, "good_cropped": 0, "good_missed": 0, "false_good": 0,
         "ious": [], "counts": []}
    used = set()
    for i, gm, is_good in gts:
        best, bj = 0.0, -1
        for j, c in enumerate(cols):
            v = iou(gm, c.mask)
            if v > best:
                best, bj = v, j
        if best >= 0.5:
            used.add(bj)
        if not is_good:
            if best >= 0.5 and not cols[bj].cropped:
                r["false_good"] += 1
            continue
        if best < 0.5:
            r["good_missed"] += 1
        elif cols[bj].cropped:
            r["good_cropped"] += 1
        else:
            r["good_ok"] += 1
            r["ious"].append(best)
            r["counts"].append((cols[bj].count, edge_count_inside(js, gm), bool(cols[bj].extra.get("uncertain"))))
    r["false_good"] += sum(1 for j, c in enumerate(cols) if j not in used and not c.cropped)
    return r


def summarise(rows: list[dict]) -> dict:
    tot = {k: sum(r[k] for r in rows) for k in ("good_ok", "good_cropped", "good_missed", "false_good")}
    labelled_good = tot["good_ok"] + tot["good_cropped"] + tot["good_missed"]
    counts = [c for r in rows for c in r["counts"]]
    tot.update(labelled_good=labelled_good,
               iou=float(np.mean([v for r in rows for v in r["ious"]])) if counts else 0.0,
               count_exact=sum(p == g for p, g, _ in counts),
               end_to_end=f"{sum(p == g for p, g, _ in counts)}/{labelled_good + tot['false_good']}")
    return tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=None)
    ap.add_argument("--sets", default="holdout,train")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    sc = sheet_counter(a.models)
    out = {}
    for s in a.sets.split(","):
        rows = [score_frame(sc, stem, js) for stem, js in frames(s)]
        m = summarise(rows)
        out[s] = {"summary": m, "rows": rows}
        print(f"{s:8s} labelled complete {m['labelled_good']:2d}: found {m['good_ok']}, called cropped "
              f"{m['good_cropped']}, missed {m['good_missed']} | false complete {m['false_good']} | "
              f"IoU {m['iou']:.3f} | count exact {m['count_exact']} | end-to-end {m['end_to_end']}")
        for r in rows:
            print(f"    {r['stem']:40s} ok {r['good_ok']} cropped {r['good_cropped']} missed {r['good_missed']} "
                  f"false {r['false_good']} counts {[(p, g) for p, g, _ in r['counts']]}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
