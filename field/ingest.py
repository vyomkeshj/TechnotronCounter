"""Take a folder of newly labelled site data into data/, and CHECK IT FIRST.

Expected layout of the folder you point at:

    <folder>/frames/*.jpg|png + .json     LabelMe column polygons (good_column/cropped_column)
    <folder>/edges/type<N>/*.png + .json  LabelMe sheet polygons ("edge"), one per sheet

Either part may be absent. With only `frames/`, the straightened crops are generated
into <folder>/edges_to_label/ and you label those, then re-run.

WHY THE CONVENTION CHECK EXISTS. Measured 2026-09-20 on the transfer matrix: a
type3-only model counts type10 at 72/80, but adding type2 to it drops type10 to 1/80.
type2's 30-piece crops were labelled as THIN LINES while type3 and type10 use BANDS
(42 % and 52 % of the silhouette). A few crops labelled to a different convention can
poison the foreground the whole model shares -- exactly the failure a site adaptation
is most likely to introduce, because a new labeller has never seen the old crops.
So every incoming crop's coverage is compared with its type's existing median, and a
deviation is reported loudly BEFORE anything is trained.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import cv2
import numpy as np

import paths
from common import labelme
from edges.dataset import discover, frame_of, tight

COVERAGE_TOL = 0.40          # relative deviation from the type's median that earns a warning


def coverage(png: str, js: str) -> float:
    """Labelled pixels / lit pixels on the tight crop -- the 'band vs thin line' number."""
    img = cv2.imread(png)
    if img is None:
        return float("nan")
    lab = labelme.edge_label_map(js, img.shape[:2]) if hasattr(labelme, "edge_label_map") else None
    if lab is None:
        lab = np.zeros(img.shape[:2], np.int32)
        for sh in json.load(open(js)).get("shapes", []):
            if sh.get("label", "").strip().lower() == "edge":
                cv2.fillPoly(lab, [np.array(sh["points"], np.int32)], 1)
    img_t, lab_t = tight(img, lab)
    lit = int((img_t.max(axis=2) > 0).sum())
    return float((lab_t > 0).sum()) / max(1, lit)


def existing_coverage() -> dict[str, list[float]]:
    items, _ = discover()
    out: dict[str, list[float]] = {}
    for it in items:
        out.setdefault(it["type"], []).append(coverage(it["image"], it["json"]))
    return out


def check_convention(new: list[tuple[str, str, str]]) -> list[str]:
    """new = [(type, png, json)]. Returns warning lines."""
    base = {t: float(np.median(v)) for t, v in existing_coverage().items() if v}
    warn = []
    for typ, png, js in new:
        c = coverage(png, js)
        med = base.get(typ)
        stem = os.path.basename(png)
        if med is None:
            warn.append(f"  {stem}: type {typ} is NEW -- no convention to compare against "
                        f"(coverage {c:.0%}); the closest existing types are "
                        f"{ {k: f'{v:.0%}' for k, v in base.items()} }")
        elif abs(c - med) / med > COVERAGE_TOL:
            warn.append(f"  {stem}: coverage {c:.0%} vs {typ} median {med:.0%} "
                        f"({'thinner' if c < med else 'thicker'} labels) -- CONVENTION MISMATCH")
    return warn


def ingest(folder: str, site: str, do_columns=True, do_edges=True) -> dict:
    added = {"columns": [], "edges": [], "stems": []}
    # A stem is the identity of a frame across the whole pipeline (splits are keyed by it).
    # Re-adding one under a new folder would give it two entries and two split decisions.
    from columns.dataset import discover as _cdiscover
    have_cols = {it["stem"] for it in _cdiscover()}
    have_edges = {it["stem"] for it in discover()[0]}
    frames_dir = os.path.join(folder, "frames")
    if do_columns and os.path.isdir(frames_dir):
        dst = os.path.join(paths.COLUMNS_DATA, f"site_{site}")
        os.makedirs(dst, exist_ok=True)
        for js in sorted(glob.glob(os.path.join(frames_dir, "*.json"))):
            stem = os.path.splitext(os.path.basename(js))[0]
            if stem in have_cols:
                print(f"[ingest] {stem}: already in data/columns, skipped (rename it if it is "
                      f"genuinely a different frame)")
                continue
            img = next((p for e in (".jpg", ".png", ".jpeg")
                        if os.path.exists(p := os.path.join(frames_dir, stem + e))), None)
            if not img:
                print(f"[ingest] {stem}: json without an image, skipped")
                continue
            for src in (img, js):
                tgt = os.path.join(dst, os.path.basename(src))
                if not os.path.exists(tgt):
                    shutil.copy2(src, tgt)
            added["columns"].append(stem)

    # labels arrive either in edges/ (already had crops) or in edges_to_label/ (stage 2 wrote them)
    edge_dirs = [d for d in (os.path.join(folder, "edges"), os.path.join(folder, "edges_to_label"))
                 if os.path.isdir(d)]
    new_pairs = []
    if do_edges:
        for tdir in sorted(t for d in edge_dirs for t in glob.glob(os.path.join(d, "type*"))):
            typ = os.path.basename(tdir)
            dst = os.path.join(paths.EDGES_DATA, typ)
            os.makedirs(dst, exist_ok=True)
            for js in sorted(glob.glob(os.path.join(tdir, "*.json"))):
                stem = os.path.splitext(os.path.basename(js))[0]
                png = os.path.join(tdir, stem + ".png")
                if not os.path.exists(png):
                    print(f"[ingest] {stem}: json without a png, skipped")
                    continue
                if stem in have_edges:
                    print(f"[ingest] {stem}: already in data/edges, skipped")
                    continue
                new_pairs.append((typ, png, js))
                for src in (png, js):
                    tgt = os.path.join(dst, os.path.basename(src))
                    if not os.path.exists(tgt):
                        shutil.copy2(src, tgt)
                added["edges"].append(stem)
                added["stems"].append(stem)
    added["warnings"] = check_convention(new_pairs) if new_pairs else []
    added["types"] = sorted({t for t, _p, _j in new_pairs})
    added["frames"] = sorted({frame_of(s) for s in added["stems"]})
    return added


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--site", required=True)
    args = ap.parse_args()
    r = ingest(args.folder, args.site)
    print(f"columns {len(r['columns'])} frames | edges {len(r['edges'])} crops {r['types']}")
    for w in r["warnings"]:
        print("[CONVENTION]", w)


if __name__ == "__main__":
    main()
