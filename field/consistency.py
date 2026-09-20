"""Label-free error detection on the customer's UNLABELLED frames.

With 3-4 labelled frames there is no honest held-out set for the new case. But the
rack camera is fixed and the stack does not change during a capture burst, so every
frame of one session must give the SAME count for the same column position. Any
disagreement inside a session is an error, and it costs nothing to measure -- no
labels, no second model, hundreds of frames.

    python -m field.consistency <crops_glob> [--run <name>]

Crops are named <date>_<session>_..._column_<k>.png, so (session, k) identifies one
physical column photographed repeatedly. Reports the modal count per group and every
frame that disagrees with it.
"""
from __future__ import annotations

import argparse
import collections
import glob
import os

import cv2
import torch

import paths
from common.unet import load_checkpoint
from edges.counter import THR, band_count, infer
from edges.dataset import model_input

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def group_of(stem: str) -> tuple[str, str]:
    parts = stem.split("_")
    session = parts[1] if len(parts) > 1 else "?"
    k = stem.rsplit("_column_", 1)[-1] if "_column_" in stem else "?"
    return session, k


def counts(net, paths_: list[str], device: str = DEVICE) -> dict[str, int]:
    out = {}
    for p in paths_:
        img = cv2.imread(p)
        if img is None:
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        out[stem] = band_count(infer(net, model_input(img, "center"), device)[0] > THR)
    return out


def check(counts_by_stem: dict[str, int]):
    """(summary rows, list of disagreeing stems)."""
    groups: dict[tuple[str, str], list[tuple[str, int]]] = collections.defaultdict(list)
    for stem, c in counts_by_stem.items():
        groups[group_of(stem)].append((stem, c))
    rows, bad = [], []
    for (ses, k), members in sorted(groups.items()):
        cs = [c for _s, c in members]
        mode = collections.Counter(cs).most_common(1)[0][0]
        off = [s for s, c in members if c != mode]
        bad += off
        rows.append({"session": ses, "column": k, "frames": len(members), "modal_count": mode,
                     "unstable": len(off), "range": f"{min(cs)}-{max(cs)}"})
    return rows, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("crops", help="glob of straightened column crops, e.g. 'data/edges/type3/*.png'")
    ap.add_argument("--run", default=None, help="run name under runs/ (default: the active edge model)")
    args = ap.parse_args()

    if args.run:
        ckpt = os.path.join(paths.RUNS, args.run, "last.pt")
    else:
        from field import registry
        ckpt = registry.checkpoint("edges")
    net = load_checkpoint(ckpt, DEVICE)
    ps = sorted(glob.glob(args.crops))
    print(f"{len(ps)} crops | model {os.path.relpath(ckpt, paths.RUNS)}")
    rows, bad = check(counts(net, ps))
    tot_f = sum(r["frames"] for r in rows)
    tot_u = sum(r["unstable"] for r in rows)
    print(f"{'session':>10}{'col':>5}{'frames':>8}{'count':>7}{'unstable':>10}   range")
    for r in rows:
        flag = "  <-- unstable" if r["unstable"] else ""
        print(f"{r['session']:>10}{r['column']:>5}{r['frames']:>8}{r['modal_count']:>7}"
              f"{r['unstable']:>10}   {r['range']}{flag}")
    print(f"\n{tot_f - tot_u}/{tot_f} frames agree with their session's modal count "
          f"({len(rows)} column-groups)")
    if bad:
        print("unstable frames:")
        for s in bad:
            print("  ", s)


if __name__ == "__main__":
    main()
