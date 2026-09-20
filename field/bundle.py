"""Build a self-contained SITE BUNDLE: everything needed to label, train and gate on a
machine that has never seen this repo.

    python -m field.bundle --out D:\\site_bundle
    python -m field.bundle --out D:\\site_bundle --no-legacy-fg    # -128 MB, see the warning

What goes in and why:

    code + paths.py + requirements.txt   the pipeline itself (all paths are relative)
    data/edges/**                        41 MB -- BOTH the edge training pool and the
                                         edge gate's protected sets; not separable
    data/columns/**                      the column training pool and the pinned val
                                         frames the column gate scores
    runs/<live edge>, runs/<live column>  the two checkpoints to fine-tune FROM
    onnx/                                deployment exports (optional, --no-onnx)

What stays behind: every other run under runs/ (experiment history, ~1.2 GB) and the
"unseen older" crops, which live outside the pipeline anyway and are already optional
(`TECHNOTRON_OLD_CROPS`; the gate skips them loudly when absent).

--no-legacy-fg drops data/columns/legacy_fg (128 MB of fg-only supervision). It makes
the bundle much smaller and it CHANGES THE TRAINING MIX: `columns.train --no-legacy` is
recorded as untested. The column gate will still catch a resulting regression, but
expect more refusals. Do not use it unless the size genuinely matters.
"""
from __future__ import annotations

import argparse
import os
import shutil

import paths
from field import registry

CODE = ["columns", "common", "edges", "field", "tools", "paths.py", "requirements.txt",
        "README.md", "STATE.md", "EXPERIMENTS_DATA_SCALING.md"]

SITE_README = """# Technotron site bundle

Self-contained. Nothing outside this folder is referenced.

## Setup (once)

Install Python 3.13 and torch for your CUDA version from https://pytorch.org, then:

    pip install -r requirements.txt

Run every command FROM THIS FOLDER.

## The loop

Put the frames you labelled (LabelMe, `good_column` / `cropped_column`) in
`<folder>/frames/`, then:

    python -m field.adapt --site <name> --from <folder>

That improves the column model (gated), then writes straightened columns to
`<folder>/edges_to_label/`. Label those (one `edge` polygon per sheet), leave the
.json files beside the PNGs, and run:

    python -m field.adapt --site <name> --from <folder> --stage edges

Useful:

    python -m field.adapt --site <name> --from <folder> --dry-run   # checks + baselines, trains nothing
    python -m field.consistency '<folder>/edges_to_label/*/*.png'   # which crops are failing, no labels needed
    python -m field.registry                                        # what is live
    python -m field.registry rollback edges                         # undo a promotion

## What "gated" means

A retrained model only goes live if it improves your new frames AND loses no more than
the measured seed noise on every set that was already working. If it regresses a type
this site does not run, it is registered for this site only. If it regresses one you do
run, nothing changes and you keep the old model. See STATE.md.

## Two things that are NOT errors

* "NOTHING TO DO: the live model already counts all N new crops correctly" -- label the
  crops that are actually failing; `field.consistency` finds them without labels.
* "SKIPPING old_t3 / old_t10" -- those evaluation sets are not part of this bundle. The
  gate runs without them.
"""


def copytree(src: str, dst: str, skip=()) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", *skip))


def size(p: str) -> int:
    if os.path.isfile(p):
        return os.path.getsize(p)
    return sum(os.path.getsize(os.path.join(r, f)) for r, _d, fs in os.walk(p) for f in fs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", default="full", choices=["full", "lean", "site"],
                    help="full = exact office behaviour (can promote globally); "
                         "lean = minus legacy_fg; "
                         "site = minus legacy_fg and closeup -- SITE-SCOPED MODELS ONLY")
    ap.add_argument("--no-legacy-fg", action="store_true", help="drop 128 MB of fg-only frames (changes training)")
    ap.add_argument("--no-onnx", action="store_true")
    args = ap.parse_args()
    drop = {"full": (), "lean": ("legacy_fg",), "site": ("legacy_fg", "closeup")}[args.profile]
    if args.no_legacy_fg and "legacy_fg" not in drop:
        drop = drop + ("legacy_fg",)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    report = []

    for c in CODE:
        s = os.path.join(paths.ROOT, c)
        if not os.path.exists(s):
            continue
        d = os.path.join(out, c)
        copytree(s, d) if os.path.isdir(s) else shutil.copy2(s, d)
        report.append((c, size(d)))

    copytree(paths.EDGES_DATA, os.path.join(out, "data", "edges"))
    report.append(("data/edges", size(os.path.join(out, "data", "edges"))))
    copytree(paths.COLUMNS_DATA, os.path.join(out, "data", "columns"), skip=drop)
    report.append(("data/columns", size(os.path.join(out, "data", "columns"))))

    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    for track in ("edges", "columns"):
        run = registry.active(track)
        # Only the checkpoint we fine-tune FROM. Stage-1 weights, event files and preview
        # images are ~18 MB of run history that the site never reads.
        src = registry.checkpoint(track)
        if not os.path.exists(src):
            raise SystemExit(f"live {track} checkpoint not found: {src}")
        dst = os.path.join(out, os.path.relpath(src, paths.ROOT))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        for extra in ("audit.json", "recipe.json"):
            e = os.path.join(os.path.dirname(src), extra)
            if os.path.exists(e):
                shutil.copy2(e, os.path.join(os.path.dirname(dst), extra))
        report.append((f"runs/{run}", size(os.path.dirname(dst))))
    shutil.copy2(registry.PATH, os.path.join(out, "runs", "_active.json")) \
        if os.path.exists(registry.PATH) else None

    if not args.no_onnx and os.path.isdir(os.path.join(paths.ROOT, "onnx")):
        copytree(os.path.join(paths.ROOT, "onnx"), os.path.join(out, "onnx"))
        report.append(("onnx", size(os.path.join(out, "onnx"))))

    note = SITE_README
    if drop:
        note += (f"\n## This bundle is profile `{args.profile}`\n\n"
                 f"Dropped from the training pool and the gate: {', '.join(drop)}.\n")
        if "closeup" in drop:
            note += ("\n**Because `closeup` is absent, this bundle can only produce SITE-SCOPED\n"
                     "models.** Those frames are neither replayed during fine-tuning nor scored by\n"
                     "the gate, so a drift away from the close-up camera domain would be invisible\n"
                     "here. Send the labelled folder to the office bundle if you want a model that\n"
                     "can be promoted globally.\n")
        if "legacy_fg" in drop:
            note += ("\nWithout `legacy_fg` the column training mix differs from the office\n"
                     "(`columns.train --no-legacy` is recorded as untested). Expect more refusals.\n")
    open(os.path.join(out, "SITE_README.md"), "w", encoding="utf-8").write(note)

    total = size(out)
    print(f"{'component':<34}{'MB':>9}")
    for name, n in sorted(report, key=lambda t: -t[1]):
        print(f"{name:<34}{n / 1e6:>9.1f}")
    print(f"{'-' * 43}\n{'TOTAL':<34}{total / 1e6:>9.1f}")
    print(f"\n-> {out}")
    print("Verify it before handing it over:")
    print(f"   cd {out} && python -m edges.train --name smoke --smoke --workers 2")
    print(f"   cd {out} && python -m columns.train --name smoke --smoke --workers 2")


if __name__ == "__main__":
    main()
