"""FIELD ADAPTATION, all three stages. New site data in, better models out, gated.

    python -m field.adapt --site acme --from <folder>                 # stage 1 + 2, then stop
    python -m field.adapt --site acme --from <folder> --stage edges   # stage 3, after labelling
    python -m field.adapt --site acme --from <folder> --dry-run       # checks + baselines only
    python -m field.registry rollback edges|columns                   # undo a promotion

THE LOOP

  stage 1  columns   <folder>/frames/*.jpg + .json  (good_column / cropped_column)
                     -> ingest, fine-tune the COLUMN model from the live one, gate, promote
  stage 2  crops     run the (now improved) column model over every frame in <folder>/frames
                     -> <folder>/edges_to_label/type<N>/<frame>_column_<k>.png
                     STOP. The labeller draws one "edge" polygon per sheet on those PNGs.
  stage 3  edges     labels come back beside the PNGs
                     -> ingest, fine-tune the EDGE model, gate, promote

Each stage ends in the same three-way outcome: promote globally -> promote for this site
only (same checkpoint, gate scoped to the types the site runs) -> refuse and change
nothing. Auto-promote on a pass, previous kept for one-command rollback.

WHY GATE AT ALL (measured 2026-09-20, EXPERIMENTS_DATA_SCALING.md): growing type10 from
0 to 59 crops cost type2 3 crops of 54, monotonically; and a type2+type3 model scores
1/80 on type10 where a type3-only model scores 72/80. Adding data DOES break other
types, and the pinned val never shows it. The gate is what makes "better here, no worse
elsewhere" a checked claim instead of a hope.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch

import paths
from common.unet import load_checkpoint
from field import gate, gate_columns, ingest, registry

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _run(cmd: list[str], log: str) -> None:
    print("  " + " ".join(cmd[2:]), flush=True)
    with open(log, "w") as fh:
        p = subprocess.run(cmd, cwd=paths.ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise SystemExit(f"training failed, see {log}")


def _stems_file(tag: str, stems: list[str]) -> str:
    p = os.path.join(paths.RUNS, f"_boost_{tag}.txt")
    open(p, "w").write("\n".join(stems))
    return p


def _outcome(track: str, site: str, tag: str, run: str, passed: bool, rows, sets_scope,
             regate) -> str:
    """promote globally -> promote for the site -> refuse. Returns what happened."""
    if not passed and not sets_scope:
        # An empty scope protects NOTHING, so a site-scoped gate would pass vacuously.
        print(f"\nREFUSED: the global gate failed and this site's types could not be "
              f"determined, so a site-scoped gate would check nothing. Live model unchanged.")
        return "refused"
    if passed:
        registry.promote(track, run, note=f"site {site}")
        print(f"\nPROMOTED GLOBALLY: {run} is now the live {track} model "
              f"(`python -m field.registry rollback {track}` to undo)")
        return "global"
    print(f"\nglobal gate failed -> re-gating scoped to this site {sorted(sets_scope)}")
    passed2, rows2 = regate()
    print(rows2)
    if passed2:
        registry.promote_site(track, site, run, note="global gate failed")
        print(f"\nPROMOTED FOR SITE {site} ONLY: {run}. The global {track} model is untouched.")
        return "site"
    print(f"\nREFUSED: the candidate regresses something this site runs. Live model unchanged.")
    return "refused"


# ---------------------------------------------------------------- stage 1: columns
def stage_columns(args, r, tag: str) -> str:
    stems = r["columns"]
    if not stems:
        print("=== stage 1 (columns): no labelled frames in <folder>/frames, skipped")
        return "skipped"
    print(f"=== stage 1 (columns): {len(stems)} new labelled frames")
    val = gate_columns.build_sets(exclude_stems=stems)
    live = registry.checkpoint("columns")
    before = gate_columns.score(load_checkpoint(live, DEVICE), val)
    print(f"    live: {os.path.relpath(live, paths.RUNS)} | val {len(val)} frames")
    for g in sorted(before):
        print(f"      {g:<12} " + "  ".join(f"{k} {before[g][k]:.3f}" for k in gate_columns.KEYS
                                            if k in before[g]))
    if args.dry_run:
        return "dry-run"
    name = f"site_{args.site}_{tag}"
    _run([sys.executable, "-u", "-m", "columns.train", "--name", name, "--seed", "7",
          "--workers", str(args.workers), "--init", live, "--steps2", str(args.column_steps),
          "--lr", str(args.column_lr), "--boost-stems", _stems_file("col_" + tag, stems),
          "--boost-factor", str(args.boost_factor)],
         os.path.join(paths.RUNS, f"_adapt_columns_{tag}.log"))
    ckpt = os.path.join(paths.RUNS, f"columns_{name}", "stage2", "last.pt")
    after = gate_columns.score(load_checkpoint(ckpt, DEVICE), val)
    passed, rows = gate_columns.verdict(before, after)
    print(gate_columns.table(rows))
    # This site's groups come from the NEW COLUMN FRAMES, not from r["types"] (which is
    # the edge-crop types and is empty at this stage -- that emptiness silently disabled
    # the scoped gate before it was fixed).
    from common import splits as _sp
    groups = {t for t in (_sp.type_of(s) for s in stems) if t}
    print(f"    site groups from the new frames: {sorted(groups) or 'UNKNOWN'}")
    return _outcome("columns", args.site, tag, f"columns_{name}", passed, rows, groups,
                    lambda: (lambda p, rr: (p, gate_columns.table(rr)))(
                        *gate_columns.verdict(before, after, site_groups=groups)))


# ---------------------------------------------------------------- stage 2: crops
def stage_crops(args) -> str:
    frames = os.path.join(args.folder, "frames")
    out = os.path.join(args.folder, "edges_to_label")
    if not os.path.isdir(frames):
        print("=== stage 2 (crops): no <folder>/frames, skipped")
        return "skipped"
    print(f"=== stage 2 (crops): straightening every column the live model finds")
    _run([sys.executable, "-u", "-m", "field.crops", "--frames", frames, "--out", out]
         + (["--type", args.type] if args.type else []),
         os.path.join(paths.RUNS, f"_adapt_crops_{time.strftime('%Y%m%d_%H%M')}.log"))
    n = sum(len(fs) for _d, _s, fs in os.walk(out) if fs)
    print(f"    {n} crops in {out}")
    print(f"\nNEXT: label the PNGs in {out} (one 'edge' polygon per sheet, LabelMe),\n"
          f"      leave the .json files beside them, then run:\n"
          f"      python -m field.adapt --site {args.site} --from {args.folder} --stage edges")
    return "crops written"


# ---------------------------------------------------------------- stage 3: edges
def stage_edges(args, r, tag: str) -> str:
    if not r["stems"]:
        raise SystemExit("no labelled EDGE crops found. Run stage 2 first, label the crops it\n"
                         "writes to <folder>/edges_to_label, then re-run with --stage edges.")
    print(f"=== stage 3 (edges): {len(r['stems'])} new labelled crops {r['types']}")
    sets = gate.build_sets(exclude_stems=r["stems"])
    live = registry.checkpoint("edges")
    base_net = load_checkpoint(live, DEVICE)
    before = gate.score(base_net, sets, DEVICE)
    old_ok, new_n = gate.score_stems(base_net, r["stems"], DEVICE)
    print(f"    live: {os.path.relpath(live, paths.RUNS)} | new crops {old_ok}/{new_n} correct already")
    for k in sorted(before):
        print(f"      {k:<12} {before[k][0]}/{before[k][1]}")
    if args.dry_run:
        return "dry-run"
    if old_ok == new_n:
        print(f"\nNOTHING TO DO: the live model already counts all {new_n} new crops correctly.\n"
              f"Live model unchanged. Find the crops that ARE failing, without labels:\n"
              f"  python -m field.consistency '{os.path.join(args.folder, 'edges_to_label', '*', '*.png')}'")
        return "nothing to do"

    name = f"site_{args.site}_{tag}"
    _run([sys.executable, "-u", "-m", "edges.train", "--name", name, "--seed", "7",
          "--workers", str(args.workers), "--squash-weight", "0", "--lighting", "v2",
          "--init", live, "--lr", str(args.lr), "--steps", str(args.steps),
          "--boost-stems", _stems_file("edge_" + tag, r["stems"]),
          "--boost-factor", str(args.boost_factor)],
         os.path.join(paths.RUNS, f"_adapt_edges_{tag}.log"))
    ckpt = os.path.join(paths.RUNS, f"edges_{name}", "last.pt")
    net = load_checkpoint(ckpt, DEVICE)
    after = gate.score(net, sets, DEVICE)
    new_ok, _ = gate.score_stems(net, r["stems"], DEVICE)
    print(f"    NEW site crops: {old_ok}/{new_n} -> {new_ok}/{new_n}  "
          f"(training data -- evidence the adaptation took, not accuracy)")
    passed, rows = gate.verdict(before, after, sets)
    print(gate.table(rows))
    if new_ok <= old_ok:
        print(f"\nREFUSED: the new crops did not improve. Nothing promoted.")
        return "no improvement"
    types = set(r["types"])
    res = _outcome("edges", args.site, tag, f"edges_{name}", passed, rows, types,
                   lambda: (lambda p, rr: (p, gate.table(rr)))(
                       *gate.verdict(before, after, sets, site_types=types)))
    json.dump({"site": args.site, "run": f"edges_{name}", "new_stems": r["stems"],
               "before": before, "after": after, "new_crops": [old_ok, new_ok, new_n],
               "outcome": res, "rows": rows},
              open(os.path.join(paths.RUNS, f"_adapt_edges_{tag}.json"), "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True)
    ap.add_argument("--from", dest="folder", required=True)
    ap.add_argument("--stage", default="auto", choices=["auto", "columns", "crops", "edges"],
                    help="auto = columns then crops, then stop for labelling")
    ap.add_argument("--type", default=None, help="force the type folder name for generated crops")
    ap.add_argument("--steps", type=int, default=2500, help="edge fine-tune length (scratch is 8000)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--column-steps", type=int, default=1500, help="column fine-tune length (scratch is 4000)")
    ap.add_argument("--column-lr", type=float, default=5e-5)
    ap.add_argument("--boost-factor", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="proceed despite a label-convention mismatch")
    args = ap.parse_args()
    tag = time.strftime("%Y%m%d_%H%M")

    print(f"=== ingest {args.folder}")
    r = ingest.ingest(args.folder, args.site,
                      do_columns=args.stage in ("auto", "columns"),
                      do_edges=args.stage in ("auto", "edges"))
    print(f"    columns {len(r['columns'])} frames | edges {len(r['edges'])} crops {r['types']}")
    if r["warnings"]:
        print("    LABEL CONVENTION WARNINGS:")
        for w in r["warnings"]:
            print("   ", w)
        if not args.force:
            raise SystemExit("    refusing to train on labels that do not match their type's "
                             "convention.\n    Fix them, or re-run with --force if deliberate.")

    if args.stage in ("auto", "columns"):
        print("\n" + stage_columns(args, r, tag))
    if args.stage in ("auto", "crops") and not args.dry_run:
        print("\n" + stage_crops(args))
    if args.stage == "edges":
        print("\n" + stage_edges(args, r, tag))


if __name__ == "__main__":
    main()
