"""One-off: copy every labelled source from the research tree into data/.

Run once from training_pipeline/:  python -m tools.bootstrap_data
Re-running is safe (identical files are skipped). This file is also the
PROVENANCE RECORD of where each data folder came from.

    data/columns/closeup/          datasets/New collection .../New collection   41 LabelMe frames
                                   (4000x2252 phone photos, per-column polygons)
    data/columns/operational/      datasets/column data                          18 frames
                                   (rack camera; the 14 frames re-labelled in the quality
                                   dataset are NOT copied -- the new labels supersede them)
    data/columns/quality/typeN/    datasets/quality_dataset_20260916             161 frames
                                   (types 2/3/10, passed + not_passed, flattened)
    data/columns/legacy_fg/        datasets/vid_20240213_100344_200frames +      200 frames
                                   data_2/200_fixed/masks (frames 1-60, hand-corrected)
                                   data_2/200_gen/all_columns_v16_200 (61-200, Gemini)
                                   Binary masks with fused columns -> foreground-only use.
    data/edges/type3/              datasets/good                                 72 human-labelled crops
    data/edges/type2/, type10/     datasets/edge_label_columns_20260916/         54 / 80 crops,
                                   type_<n>_cols_primitive                       awaiting labels
    data/columns/splits.json       pinned frame split, keyed by frame stem (common/splits.py)
    data/edges/splits.json         pinned SOURCE-frame split for crops (common/splits.py)
"""
from __future__ import annotations

import filecmp
import glob
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(HERE)
D = os.path.join(REPO, "datasets")
OUT = os.path.join(HERE, "data")

CLOSEUP = os.path.join(D, "New collection -20260709T205444Z-2-001", "New collection")
OPERATIONAL = os.path.join(D, "column data")
QUALITY = os.path.join(D, "quality_dataset_20260916", "quality_dataset")
LEGACY_FRAMES = os.path.join(D, "vid_20240213_100344_200frames")
LEGACY_FIXED = os.path.join(REPO, "data_2", "200_fixed", "masks")
LEGACY_GEN = os.path.join(REPO, "data_2", "200_gen", "all_columns_v16_200")
HUMAN_EDGES = os.path.join(D, "good")
PRIMITIVE = os.path.join(D, "edge_label_columns_20260916")
Q_TYPES = ("type2", "type3", "type10")

# Validation pins carried over from the research campaign (never re-rolled).
# closeup: newdata.split_new() seed 42; quality: qdata.build_split() session holdout.
RESEARCH_Q_SPLIT = os.path.join(REPO, "experiments", "column_unet", "splits_q2310.json")
EDGE_VAL_FRAMES = ["20260626_131412_type3_-2_60pieces_001", "20260626_131412_type3_-2_60pieces_005"]

n_copied = n_skipped = 0


def copy(src, dst):
    global n_copied, n_skipped
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and filecmp.cmp(src, dst, shallow=False):
        n_skipped += 1
        return
    shutil.copy2(src, dst)
    n_copied += 1


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def pairs(folder, ext=".jpg"):
    return [(j[:-5] + ext, j) for j in sorted(glob.glob(os.path.join(folder, "*.json")))
            if os.path.exists(j[:-5] + ext)]


def main():
    # ---------------- columns: quality dataset (types 2/3/10)
    q_hashes = set()
    for t in Q_TYPES:
        for split in ("passed", "not_passed"):
            for jpg, js in pairs(os.path.join(QUALITY, f"{t}_quality", split)):
                q_hashes.add(md5(jpg))
                dst = os.path.join(OUT, "columns", "quality", t)
                copy(jpg, os.path.join(dst, os.path.basename(jpg)))
                copy(js, os.path.join(dst, os.path.basename(js)))
    # ---------------- columns: close-up photos
    for jpg, js in pairs(CLOSEUP):
        copy(jpg, os.path.join(OUT, "columns", "closeup", os.path.basename(jpg)))
        copy(js, os.path.join(OUT, "columns", "closeup", os.path.basename(js)))
    # ---------------- columns: operational frames not superseded by a quality re-label
    superseded = []
    for jpg, js in pairs(OPERATIONAL):
        if md5(jpg) in q_hashes:
            superseded.append(os.path.basename(jpg))
            continue
        copy(jpg, os.path.join(OUT, "columns", "operational", os.path.basename(jpg)))
        copy(js, os.path.join(OUT, "columns", "operational", os.path.basename(js)))
    # ---------------- columns: legacy foreground-only frames
    src_rows = []
    for i in range(1, 201):
        fid = f"vid_20240213_100344_{i:04d}"
        mask = (os.path.join(LEGACY_FIXED, fid + ".png") if i <= 60
                else os.path.join(LEGACY_GEN, fid, "stage00_all_columns.png"))
        copy(os.path.join(LEGACY_FRAMES, fid + ".jpg"),
             os.path.join(OUT, "columns", "legacy_fg", "images", fid + ".jpg"))
        copy(mask, os.path.join(OUT, "columns", "legacy_fg", "masks", fid + ".png"))
        src_rows.append(f"{fid},{'hand_corrected' if i <= 60 else 'gemini'}")
    with open(os.path.join(OUT, "columns", "legacy_fg", "sources.csv"), "w") as f:
        f.write("frame,mask_source\n" + "\n".join(src_rows) + "\n")
    # ---------------- edges
    for png, js in pairs(HUMAN_EDGES, ".png"):
        if "_column_" not in os.path.basename(png):
            continue
        copy(png, os.path.join(OUT, "edges", "type3", os.path.basename(png)))
        copy(js, os.path.join(OUT, "edges", "type3", os.path.basename(js)))
    for n in ("2", "10"):
        for png in sorted(glob.glob(os.path.join(PRIMITIVE, f"type_{n}_cols_primitive", "*.png"))):
            copy(png, os.path.join(OUT, "edges", f"type{n}", os.path.basename(png)))
    # ---------------- pinned splits
    q = json.load(open(RESEARCH_Q_SPLIT))
    sys.path.insert(0, os.path.join(REPO, "experiments", "column_unet"))
    import newdata as N                                   # research module, read-only use
    val = {_stem(j) for j, _ in N.split_new()[1]} | {_stem(p[0]) for p in q["val"]}
    exc = {_stem(p[0]) for p in q["excluded"]}
    all_frames = {_stem(j) for j in glob.glob(os.path.join(OUT, "columns", "*", "*.json"))} | \
                 {_stem(j) for j in glob.glob(os.path.join(OUT, "columns", "quality", "*", "*.json"))}
    col_split = {
        "val": sorted(val), "excluded": sorted(exc), "train": sorted(all_frames - val - exc),
        "note": ("closeup val = research split_new (seed 42). quality val = held-out capture "
                 "sessions type2 132812+134020, type3 131412; type10 single session -> frames "
                 "9-14 with frames 8 and 15 excluded as buffer. Never re-roll."),
    }
    _dump(os.path.join(OUT, "columns", "splits.json"), col_split)
    edge_frames = {_stem(j).rsplit("_column_", 1)[0]
                   for j in glob.glob(os.path.join(OUT, "edges", "*", "*_column_*.json"))}
    _dump(os.path.join(OUT, "edges", "splits.json"), {
        "val": sorted(EDGE_VAL_FRAMES), "excluded": [], "train": sorted(edge_frames - set(EDGE_VAL_FRAMES)),
        "note": "type3 val = the 2 research-pinned frames never trained on by any earlier model.",
    })
    print(f"copied {n_copied}, unchanged {n_skipped}; operational superseded: {len(superseded)}")


def _stem(p):
    return os.path.splitext(os.path.basename(p))[0]


def _type(p):
    return os.path.basename(p).split("_")[2]


def _dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        print(f"keeping existing {path} (pinned splits are never overwritten)")
        return
    json.dump(obj, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
