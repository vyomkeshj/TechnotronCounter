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


def _type_of(stem: str) -> str:
    from common import splits
    return splits.type_of(stem) or "typeX"


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
    have_cols = {it["stem"]: it["json"] for it in _cdiscover()}
    have_edges = {it["stem"] for it in discover()[0]}
    frames_dir = os.path.join(folder, "frames")
    combined = []                                  # frames that ALSO carry edge polygons
    if do_columns and os.path.isdir(frames_dir):
        dst = os.path.join(paths.COLUMNS_DATA, f"site_{site}")
        os.makedirs(dst, exist_ok=True)
        for js in sorted(glob.glob(os.path.join(frames_dir, "*.json"))):
            stem = os.path.splitext(os.path.basename(js))[0]
            if stem in have_cols:
                # IDEMPOTENT for the same site: a --dry-run (or an earlier attempt) has
                # already copied this frame in. It is still THIS site's new frame, so it
                # must still be boosted in training and excluded from the gate sets.
                if os.path.normcase(os.path.dirname(os.path.abspath(have_cols[stem]))) == os.path.normcase(dst):
                    added["columns"].append(stem)
                    if labelme.has_edge_labels(js):
                        combined.append((stem, js))
                    continue
                print(f"[ingest] {stem}: already in data/columns (another source), skipped "
                      f"(rename it if it is genuinely a different frame)")
                continue
            if labelme.has_edge_labels(js):
                combined.append((stem, js))
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

    # COMBINED labelling: the frame json carries the edge polygons too. Derive the
    # straightened crops AND their edge labels straight into data/edges (same geometry as
    # the production straightener -- see edges/from_frames.py), then treat them exactly
    # like labels that arrived beside crops.
    new_pairs = []
    if do_edges and combined:
        from edges.from_frames import frame_to_crops
        for stem, js in combined:
            img = labelme.image_for(js)
            typ = _type_of(stem)
            out_dir = os.path.join(paths.EDGES_DATA, typ)
            if glob.glob(os.path.join(out_dir, f"{stem}_column_*.png")):
                n_png = len(glob.glob(os.path.join(out_dir, f"{stem}_column_*.png")))
                n_json = len(glob.glob(os.path.join(out_dir, f"{stem}_column_*.json")))
                print(f"[ingest] {stem}: edge crops already derived ({n_png} crops, {n_json} labelled) -- reusing")
            else:
                n_png, n_json = frame_to_crops(img, js, out_dir, stem)
            for jp in sorted(glob.glob(os.path.join(out_dir, f"{stem}_column_*.json"))):
                cstem = os.path.splitext(os.path.basename(jp))[0]
                new_pairs.append((typ, jp[:-5] + ".png", jp))
                added["edges"].append(cstem)
                added["stems"].append(cstem)
            print(f"[ingest] {stem}: combined frame -> {n_png} crops, {n_json} with edge labels -> data/edges/{typ}")

    # labels arrive either in edges/ (already had crops) or in edges_to_label/ (stage 2 wrote them)
    edge_dirs = [d for d in (os.path.join(folder, "edges"), os.path.join(folder, "edges_to_label"))
                 if os.path.isdir(d)]
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
                # already in data/edges is fine: it is in THIS site's folder, so it is this
                # site's crop whether or not an earlier run copied it (idempotent re-runs)
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
