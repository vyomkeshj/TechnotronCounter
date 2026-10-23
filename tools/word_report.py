"""Images for the Word visual report: 3 held-out frames per type.

    python -m tools.word_report
Writes runs/_word_report/<type>__<frame>__{image,columns,edges}.jpg and manifest.json.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

import paths
from tools.evalkit import INFERENCE, expected, sheet_counter
from tools.report import pick

OUT = os.path.join(paths.RUNS, "_word_report")


def edge_panel(cols, exp, pal):
    tiles = []
    for c in cols:
        x = c.extra["edge_input"]
        lab = c.sheet_masks
        ov = x.copy()
        m = lab > 0
        ov[m] = (0.3 * ov[m] + 0.7 * pal[lab[m] % len(pal)]).astype(np.uint8)
        pair = np.hstack([x, np.zeros((x.shape[0], 2, 3), np.uint8), ov])
        unsure = bool(c.extra.get("uncertain"))
        colour = (0, 215, 255) if unsure else ((80, 255, 80) if c.count == exp else (80, 80, 255))
        head = np.zeros((60, pair.shape[1], 3), np.uint8)
        txt = f"{c.count}?" if unsure else str(c.count)
        cv2.putText(head, txt, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.3, colour, 3)
        tiles.append(np.vstack([head, pair]))
        tiles.append(np.full((tiles[-1].shape[0], 14, 3), 255, np.uint8))
    h = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.zeros((h - t.shape[0], t.shape[1], 3), np.uint8)]) for t in tiles]
    return np.hstack(tiles[:-1])


def main():
    sc = sheet_counter(None)
    sys.path.insert(0, INFERENCE)
    import predict as pr
    pal = np.random.RandomState(7).randint(70, 255, (64, 3)).astype(np.uint8)
    os.makedirs(OUT, exist_ok=True)
    man = []
    for typ, stems in pick(3, 0).items():
        for stem in stems:
            img = cv2.imread(os.path.join(paths.COLUMNS_DATA, "quality", typ, stem + ".jpg"))
            cols = sc.process(img, typ, masks=True, check=True)
            exp = expected(stem)
            over = pr.render_overlay(img, cols, [c.count for c in cols])
            good = [c for c in cols if not c.cropped and c.sheet_masks is not None]
            base = os.path.join(OUT, f"{typ}__{stem}")
            cv2.imwrite(base + "__image.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            cv2.imwrite(base + "__columns.jpg", over, [cv2.IMWRITE_JPEG_QUALITY, 85])
            ep = edge_panel(good, exp, pal)
            cv2.imwrite(base + "__edges.jpg", ep, [cv2.IMWRITE_JPEG_QUALITY, 88])
            counts = [c.count for c in good]
            man.append({"type": typ, "frame": stem, "expected": exp, "counts": counts,
                        "exact": sum(n == exp for n in counts), "uncertain": sum(bool(c.extra.get("uncertain")) for c in good),
                        "cropped": sum(c.cropped for c in cols), "image": base + "__image.jpg",
                        "columns": base + "__columns.jpg", "edges": base + "__edges.jpg",
                        "edges_wh": [ep.shape[1], ep.shape[0]]})
            print(typ, stem, counts, exp, flush=True)
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
