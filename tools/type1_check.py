"""Type1 (no labels yet): count every complete column of the June quality frames and compare with the
`_NNpieces_` number in the file name. Not a training type; a check of how the system behaves on it.

    python -m tools.type1_check [--models <dir>] [--name tag] [--render]

Type1 has three kinds of frame (user-confirmed 2026-11-03):
  side view  session 134502 (`_0_30pieces`): every column is cut off by the frame/beam -> the right
             answer is NO complete column ("complete" printed per session; target 0, MAE meaningless)
  50 pieces  sessions 141559, 141837, 141934 -> 50
  tilted 30  session 142210 -> 30
Output: one line per session (+ ALL over the counted kinds), runs/_type1/<name>/counts.csv, and with
--render a before-style overlay per frame (green = right count, amber = wrong, red = cropped).
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import re

import cv2
import numpy as np

import paths
from tools.evalkit import sheet_counter

FRAMES = os.path.join(os.path.dirname(paths.ROOT), "datasets", "quality_dataset_20260916", "quality_dataset",
                      "type1_quality", "passed")
SIDE_VIEW = "134502"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None)
    ap.add_argument("--name", default="shipped")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    sc = sheet_counter(args.models)                 # puts inference/ on sys.path
    from sheetcount.pipeline import vote
    out = os.path.join(paths.RUNS, "_type1", args.name)
    os.makedirs(out, exist_ok=True)
    by_ses, rows = collections.defaultdict(list), []
    for p in sorted(glob.glob(os.path.join(FRAMES, "*.jpg"))):
        stem = os.path.basename(p)[:-4]
        exp = int(re.search(r"_(\d+)pieces_", stem).group(1))
        img = cv2.imread(p)
        cols = sc.process(img, "type1", masks=args.render)
        good = [c for c in cols if not c.cropped]
        data = [(float(np.where(c.mask)[1].mean()), float(np.where(c.mask)[0].mean()), c.count) for c in good]
        by_ses[stem.split("_")[1]].append((stem, exp, data, len(cols)))
        if args.render:
            o = img.copy()
            for c in cols:
                col = (40, 40, 230) if c.cropped else ((60, 220, 60) if c.count == exp else (0, 190, 255))
                o[c.mask] = (0.55 * o[c.mask] + 0.45 * np.array(col)).astype(np.uint8)
                ys, xs = np.where(c.mask)
                cv2.putText(o, "crop" if c.cropped else str(c.count), (int(xs.mean()) - 20, int(ys.mean())), 0, 1.0,
                            (255, 255, 255), 3)
            cv2.imwrite(os.path.join(out, stem + ".jpg"), cv2.resize(o, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 85])
    tot = collections.Counter()
    for ses, items in sorted(by_ses.items()):
        voted = vote([d for _, _, d, _ in items], 5)
        s = collections.Counter()
        for (stem, exp, data, ncol), v in zip(items, voted):
            s["frames"] += 1
            for (cx, cy, n), vn in zip(data, v):
                s["good"] += 1
                s["exact"] += n == exp
                s["vexact"] += vn == exp
                s["ae"] += abs((n or 0) - exp)
                rows.append(dict(session=ses, frame=stem, cx=round(cx), count=n, voted=vn, expected=exp))
        if ses == SIDE_VIEW:
            print(f"{ses} side view: frames {s['frames']:3d}  columns called complete {s['good']:3d}  (target 0)")
            continue
        tot.update(s)
        print(f"{ses} exp {items[0][1]:2d}: frames {s['frames']:3d} complete {s['good']:3d}  exact {s['exact']:3d} "
              f"({100 * s['exact'] / max(1, s['good']):5.1f} %) voted {s['vexact']:3d}  MAE {s['ae'] / max(1, s['good']):.2f}")
    s = tot
    print(f"ALL counted kinds: complete {s['good']:3d}  exact {s['exact']:3d} ({100 * s['exact'] / max(1, s['good']):5.1f} %) "
          f"voted {s['vexact']:3d} ({100 * s['vexact'] / max(1, s['good']):5.1f} %)  MAE {s['ae'] / max(1, s['good']):.2f}")
    with open(os.path.join(out, "counts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, ["session", "frame", "cx", "count", "voted", "expected"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
