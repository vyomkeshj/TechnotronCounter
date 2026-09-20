"""STAGE 2 of field adaptation: run the live COLUMN model over a site's frames and emit
straightened columns for the labeller.

    python -m field.crops --frames <dir> --out <dir> [--type type10] [--dup-ncc 0.97]

Every GOOD column the model finds becomes one crop, straightened and placed on the
fixed 2050 px canvas at native width -- the exact format the edge labeller already
works in, and the exact format `edges.make_crops` produces from HUMAN polygons.

How it stays identical to the human path: the model's instances are written out as a
LabelMe json (good_column / cropped_column, from the 4th channel's per-instance vote)
and then handed to the SAME `common.straighten.straighten_frame`. Nothing about the
geometry is reimplemented here, so a crop made from a predicted polygon and one made
from a drawn polygon differ only by the polygon.

Near-duplicate frames are skipped (thumbnail NCC within a capture session): the rack
camera is fixed, so neighbouring frames would just be labelled twice for nothing.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import tempfile

import cv2
import numpy as np
import torch

import paths
from common.imageio import letterbox, to_tensor
from common.straighten import straighten_frame, to_native_canvas
from common.unet import load_checkpoint
from columns.evaluate import predict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 256
MIN_AREA = 25                      # at 256, the readout's own floor


def thumb(p: str, n: int = 64):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    t = cv2.resize(im, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32)
    t -= t.mean()
    s = float(np.linalg.norm(t))
    return t / s if s > 1e-6 else None


def session_of(stem: str) -> str:
    parts = stem.split("_")
    return parts[1] if len(parts) > 1 else "?"


def type_of(stem: str, override: str | None) -> str:
    if override:
        return override
    m = re.search(r"(type\d+)", stem)
    return m.group(1) if m else "typeX"


@torch.no_grad()
def predict_polygons(net, img: np.ndarray) -> list[dict]:
    """LabelMe shapes from the column model, in ORIGINAL image coordinates."""
    H, W = img.shape[:2]
    imgL, s, px, py = letterbox(img, SIZE, 0, cv2.INTER_LINEAR)
    lab, p_crop = predict(net, to_tensor(imgL))
    shapes = []
    for i in range(1, int(lab.max()) + 1):
        m = (lab == i).astype(np.uint8)
        if int(m.sum()) < MIN_AREA:
            continue
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea).astype(np.float32)[:, 0, :]
        c = cv2.approxPolyDP(c.astype(np.int32), 1.0, True).astype(np.float32)[:, 0, :]
        if len(c) < 3:
            continue
        pts = c.copy()                                   # undo the letterbox: (x - pad) / scale
        pts[:, 0] = np.clip((pts[:, 0] - px) / s, 0, W - 1)
        pts[:, 1] = np.clip((pts[:, 1] - py) / s, 0, H - 1)
        cropped = bool(p_crop is not None and p_crop[lab == i].mean() > 0.5)
        shapes.append({"label": "cropped_column" if cropped else "good_column",
                       "points": [[float(a), float(b)] for a, b in pts],
                       "shape_type": "polygon", "group_id": None, "flags": {}})
    return shapes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="directory of site frames (jpg/png), labelled or not")
    ap.add_argument("--out", required=True, help="where to write <type>/<frame>_column_<k>.png")
    ap.add_argument("--type", default=None, help="force the type folder name (else read from the filename)")
    ap.add_argument("--dup-ncc", type=float, default=0.97)
    ap.add_argument("--ckpt", default=None, help="column checkpoint (default: the live one)")
    args = ap.parse_args()

    ckpt = args.ckpt
    if not ckpt:
        from field import registry
        ckpt = registry.checkpoint("columns")
    net = load_checkpoint(ckpt, DEVICE)
    print(f"column model {os.path.relpath(ckpt, paths.RUNS)}")

    frames = sorted(p for e in ("*.jpg", "*.jpeg", "*.png") for p in glob.glob(os.path.join(args.frames, e)))
    kept: dict[str, list] = {}
    n_crops = n_dup = 0
    for p in frames:
        stem = os.path.splitext(os.path.basename(p))[0]
        t = thumb(p)
        ses = session_of(stem)
        if t is not None:
            if any(float((t * k).sum()) >= args.dup_ncc for k in kept.get(ses, [])):
                n_dup += 1
                print(f"  {stem}: near-duplicate of a kept frame in session {ses}, skipped")
                continue
            kept.setdefault(ses, []).append(t)
        img = cv2.imread(p)
        if img is None:
            continue
        shapes = predict_polygons(net, img)
        good = sum(1 for s in shapes if s["label"] == "good_column")
        if not good:
            print(f"  {stem}: no good column found, skipped")
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            json.dump({"shapes": shapes, "imagePath": os.path.basename(p),
                       "imageHeight": img.shape[0], "imageWidth": img.shape[1]}, fh)
            tmp = fh.name
        try:
            crops = straighten_frame(img, tmp)
        finally:
            os.unlink(tmp)
        tdir = os.path.join(args.out, type_of(stem, args.type))
        os.makedirs(tdir, exist_ok=True)
        w = 0
        for k, c in enumerate(crops):
            canvas = to_native_canvas(c)
            if canvas is None:
                continue
            cv2.imwrite(os.path.join(tdir, f"{stem}_column_{k}.png"), canvas)
            w += 1
        n_crops += w
        print(f"  {stem}: {good} good columns -> {w} crops")
    print(f"\n{n_crops} crops written to {args.out} ({len(frames)} frames, {n_dup} near-duplicates skipped)")
    print("Hand this folder to the labeller; labels come back as <same name>.json beside each png.")


if __name__ == "__main__":
    main()
