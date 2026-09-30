"""THE score: the inference package on the held-out test frames, per type.

    python -m tools.evaluate                       # shipped models (inference/models)
    python -m tools.evaluate --models <dir>        # any folder with column_flow.onnx + sheet_edges_*.onnx
    python -m tools.evaluate --vote 5              # also score 5-frame voting

Test set: 40 frames from capture sessions neither model trained on (tools.evalkit.test_frames),
169 human-labelled complete columns. Each labelled complete column is matched to the predicted
column overlapping it most (IoU >= 0.5) and ends in one bucket:
  exact    predicted complete and its count equals the expected count
  wrong    predicted complete, count differs
  cropped  predicted cropped (never counted)
  missed   no predicted column overlaps it
A predicted complete column matching no labelled complete column is a FALSE column; it is
harmful if its count is wrong. score = exact / (labelled complete + harmful false).
Self-check: of the columns the package does NOT flag uncertain, how many are exact.

Expected result with the shipped models:
  type2 19/20, type3 126/126, type10 23/23 (score 0.994, same with --vote 5);
  unflagged columns correct: 19/19, 126/126, 21/21.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os

import cv2
import numpy as np

import paths
from tools.evalkit import TYPES, expected, frame_path, good_polygons, iou, sheet_counter, test_frames, type_of


def vote_counts(rows: list[dict], window: int) -> dict:
    """Most common count of the same labelled column over +-window//2 neighbouring test frames of
    the same session (columns matched by nearest centre, <= 80 px). Ties keep the frame's own count."""
    by_frame = collections.defaultdict(list)
    for r in rows:
        by_frame[r["frame"]].append(r)
    frames = sorted(by_frame, key=lambda f: (f.split("_")[1], int(f.rsplit("_", 1)[1])))
    out = {}
    for f in frames:
        same = [g for g in frames if g.split("_")[1] == f.split("_")[1]]
        i = same.index(f)
        win = same[max(0, i - window // 2): i + window // 2 + 1]
        for r in by_frame[f]:
            vals = []
            for g in win:
                q = min(by_frame[g], key=lambda q: (q["cx"] - r["cx"]) ** 2 + (q["cy"] - r["cy"]) ** 2)
                if ((q["cx"] - r["cx"]) ** 2 + (q["cy"] - r["cy"]) ** 2) ** 0.5 <= 80:
                    vals.append(q["count"])
            c = collections.Counter(vals).most_common()
            out[(f, r["col"])] = c[0][0] if (len(c) == 1 or c[0][1] > c[1][1]) else r["count"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None)
    ap.add_argument("--vote", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(paths.RUNS, "_evaluate"))
    args = ap.parse_args()
    sc = sheet_counter(args.models)
    rows, false = [], collections.Counter()
    for stem in test_frames():
        typ = type_of(stem)
        img = cv2.imread(frame_path(stem))
        H, W = img.shape[:2]
        exp = expected(stem)
        cols = sc.process(img, typ, masks=False, check=True)
        used = set()
        for k, p in enumerate(good_polygons(stem)):
            m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [p], 1); m = m > 0
            j = max(range(len(cols)), key=lambda j: iou(cols[j].mask, m), default=None)
            r = {"type": typ, "frame": stem, "col": k, "expected": exp, "count": None, "uncertain": False,
                 "cx": float(p[:, 0].mean()), "cy": float(p[:, 1].mean())}
            if j is None or iou(cols[j].mask, m) < 0.5:
                r["bucket"] = "missed"
            elif cols[j].cropped:
                r["bucket"] = "cropped"
            else:
                used.add(j)
                r.update(count=cols[j].count, uncertain=bool(cols[j].extra.get("uncertain")),
                         bucket="exact" if cols[j].count == exp else "wrong")
            rows.append(r)
        for j, c in enumerate(cols):
            if not c.cropped and j not in used and c.count != exp:
                false[typ] += 1
    voted = vote_counts(rows, args.vote) if args.vote > 1 else {}

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "columns.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{'type':7}{'complete':>9}{'exact':>7}{'wrong':>7}{'cropped':>9}{'missed':>8}{'false':>7}{'score':>8}"
          f"{'unflagged correct':>19}" + (f"{'voted':>8}" if voted else ""))
    for typ in TYPES + ("all",):
        rs = [r for r in rows if typ == "all" or r["type"] == typ]
        b = collections.Counter(r["bucket"] for r in rs)
        fh_ = sum(false.values()) if typ == "all" else false[typ]
        score = b["exact"] / (len(rs) + fh_)
        unfl = [r for r in rs if r["bucket"] in ("exact", "wrong") and not r["uncertain"]]
        line = (f"{typ:7}{len(rs):9}{b['exact']:7}{b['wrong']:7}{b['cropped']:9}{b['missed']:8}{fh_:7}{score:8.3f}"
                f"{sum(r['bucket'] == 'exact' for r in unfl):>13}/{len(unfl):<5}")
        if voted:
            v = sum(voted.get((r["frame"], r["col"])) == r["expected"] for r in rs if r["bucket"] in ("exact", "wrong"))
            line += f"{v / (len(rs) + fh_):8.3f}"
        print(line)
    print(f"-> {args.out}/columns.csv")


if __name__ == "__main__":
    main()
