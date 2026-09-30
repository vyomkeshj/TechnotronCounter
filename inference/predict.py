"""Count the sheets in every complete column of stacked-sheet photos.

    python predict.py <image or folder> [...] --out results [--type type3] [--vote 5] [--save-crops]

--type    type2 | type3 | type10 (default: read "typeN" from each file name, else type3)
--vote N  treat the inputs (per folder, in file-name order) as consecutive frames of the fixed
          camera and report each column's most common count over N frames (recommended: 5)
Outputs in --out:
  results.csv                  one row per column: frame, column (left to right), class, count,
                               voted count, centre x/y, P(cropped)
  <frame>_overlay.jpg          the frame: green = complete column (counted), yellow "N?" = counted
                               but UNCERTAIN (retake the photo), red = cropped
  <frame>_col<k>.png           (--save-crops) the straightened column, each sheet coloured and
                               every 5th numbered
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sheetcount.pipeline import SheetCounter, vote  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def colours(n: int) -> np.ndarray:
    rng = np.random.RandomState(7)
    return rng.randint(70, 255, (n + 1, 3)).astype(np.uint8)


def render_overlay(frame, cols, counts):
    out = frame.copy()
    for c, n in zip(cols, counts):
        unsure = (not c.cropped) and c.extra.get("uncertain")
        col = (0, 0, 255) if c.cropped else ((0, 215, 255) if unsure else (0, 200, 0))
        tint = out[c.mask].astype(np.float32) * 0.6 + np.array(col, np.float32) * 0.4
        out[c.mask] = tint.astype(np.uint8)
        cv2.polylines(out, [c.polygon], True, col, 3)
        if not c.cropped:
            ys, xs = np.where(c.mask)
            txt = "?" if n is None else (f"{n}?" if unsure else str(n))
            org = (int(xs.mean()) - 20, int(ys.mean()) + 12)
            cv2.putText(out, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 7)
            cv2.putText(out, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    return out


def render_crop(c) -> np.ndarray:
    x = c.extra["edge_input"]
    lab = c.sheet_masks
    ov = x.copy()
    pal = colours(int(lab.max()))
    m = lab > 0
    ov[m] = (0.35 * ov[m] + 0.65 * pal[lab[m]]).astype(np.uint8)
    ov = cv2.resize(ov, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    for k in range(1, int(lab.max()) + 1):
        if k % 5 == 0 or k == 1:
            ys, xs = np.where(lab == k)
            cv2.putText(ov, str(k), (int(xs.max()) * 3 + 4, int(ys.mean()) * 3 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    pad = np.zeros((ov.shape[0], 40, 3), np.uint8)
    img = cv2.resize(x, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    return np.hstack([img, np.zeros((ov.shape[0], 6, 3), np.uint8), ov, pad])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="results")
    ap.add_argument("--type", default="", help="type2 | type3 | type10 (default: from the file name)")
    ap.add_argument("--vote", type=int, default=0)
    ap.add_argument("--save-crops", action="store_true")
    ap.add_argument("--models", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
    args = ap.parse_args()

    files = []
    for p in args.inputs:
        if os.path.isdir(p):
            files += sorted(f for f in glob.glob(os.path.join(p, "*")) if f.lower().endswith(EXTS))
        elif p.lower().endswith(EXTS):
            files.append(p)
    if not files:
        sys.exit("no images found")
    os.makedirs(args.out, exist_ok=True)
    sc = SheetCounter(args.models)

    rows, per_frame = [], []
    for f in files:
        frame = cv2.imread(f)
        if frame is None:
            print(f"skip (unreadable): {f}")
            continue
        m = re.search(r"(type\d+)", os.path.basename(f))
        typ = args.type or (m.group(1) if m else "type3")
        cols = sc.process(frame, typ, masks=args.save_crops)
        stem = os.path.splitext(os.path.basename(f))[0]
        good = [c for c in cols if not c.cropped]
        per_frame.append((f, stem, typ, cols))
        counts = [c.count for c in cols]
        cv2.imwrite(os.path.join(args.out, f"{stem}_overlay.jpg"), render_overlay(frame, cols, counts),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        if args.save_crops:
            for k, c in enumerate(good):
                if c.sheet_masks is not None:
                    cv2.imwrite(os.path.join(args.out, f"{stem}_col{k}.png"), render_crop(c))
        flags = sum(bool(c.extra.get("uncertain")) for c in good)
        print(f"{stem} [{typ}]: {len(good)} complete columns, counts {[c.count for c in good]}"
              + (f"  ({flags} UNCERTAIN: retake the photo)" if flags else ""), flush=True)

    voted = {}
    if args.vote > 1:                                   # per folder, in file-name order
        groups = {}
        for f, stem, typ, cols in per_frame:
            groups.setdefault(os.path.dirname(os.path.abspath(f)), []).append((stem, cols))
        for items in groups.values():
            data = [[(float(np.where(c.mask)[1].mean()), float(np.where(c.mask)[0].mean()), c.count)
                     for c in cols if not c.cropped] for _, cols in items]
            for (stem, _), v in zip(items, vote(data, args.vote)):
                voted[stem] = v

    with open(os.path.join(args.out, "results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "type", "column", "class", "count", "uncertain", "why", "voted_count", "centre_x",
                    "centre_y", "p_cropped"])
        for f, stem, typ, cols in per_frame:
            k = 0
            for c in cols:
                ys, xs = np.where(c.mask)
                vc = ""
                if not c.cropped:
                    vc = voted.get(stem, [None] * 999)[k] if stem in voted else ""
                    k += 1
                w.writerow([stem, typ, "" if c.cropped else k - 1, "cropped" if c.cropped else "good",
                            "" if c.count is None else c.count,
                            "" if c.cropped else int(bool(c.extra.get("uncertain"))), c.extra.get("why", ""),
                            "" if vc is None else vc,
                            round(float(xs.mean()), 1), round(float(ys.mean()), 1), round(c.p_cropped, 3)])
    print(f"-> {os.path.join(args.out, 'results.csv')}")


if __name__ == "__main__":
    main()
