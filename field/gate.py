"""The promotion gate: may this candidate replace the model that is live?

Rule: promote only if every PROTECTED set stays within its MEASURED seed-noise band.
A drop larger than the band is a real regression; a drop inside it is indistinguishable
from re-running the same recipe with a different seed.

The bands come from running one recipe under seeds 7 and 42 (2026-09-20,
`EXPERIMENTS_DATA_SCALING.md`):

    all_type2      n=54    spread 1     -> tol 1
    all_type3      n=72    spread 0     -> tol 1
    all_type10     n=80    spread 0     -> tol 1
    old_t3         n=300   spread 6-10  -> tol 10
    old_t10        n=122   spread 23    -> NOT RANKABLE, excluded
    pinned val     n=50    saturated    -> informational only

Two things the gate deliberately does NOT do: it never scores the newly added crops as
part of a protected set (they are training data now -- `exclude_stems`), and it never
treats the pinned val as evidence, because 9 of the 12 study runs scored 50/50 on it.
"""
from __future__ import annotations

import glob
import os

import cv2
import numpy as np
import torch

import paths
from edges.counter import THR, band_count, infer
from edges.dataset import build_val, discover, model_input

OLD = os.environ.get("TECHNOTRON_OLD_CROPS", r"D:\Projects\technotron\edge_dataset\good_images")
OLD_SETS = {                                       # name -> (glob, expected count, type)
    "old_t3": (os.path.join(OLD, "type3", "columnsnew", "good", "*.png"), 50, "type3"),
    "old_t10": (os.path.join(OLD, "type10", "columnsnew", "good", "*.png"), 24, "type10"),
}
TOL = {"all_type2": 1, "all_type3": 1, "all_type10": 1, "old_t3": 10}
NOT_RANKABLE = {"old_t10": "seed spread 23/122 -- cannot rank"}


def build_sets(exclude_stems=()) -> dict:
    """name -> (kind, payload, type). `exclude_stems` drops the newly added crops."""
    items, _ = discover()
    ex = set(exclude_stems)
    sets: dict = {}
    for t in sorted({i["type"] for i in items}):
        its = [i for i in items if i["type"] == t and i["stem"] not in ex]
        if its:
            sets[f"all_{t}"] = ("labelled", build_val(its, "center"), t)
    for name, (g, exp, t) in OLD_SETS.items():
        ps = sorted(glob.glob(g))
        if ps:
            sets[name] = ("paths", (ps, exp), t)
        else:
            print(f"[gate] {name} unavailable (set TECHNOTRON_OLD_CROPS) -- not gated on it")
    return sets


@torch.no_grad()
def score(net, sets: dict, device: str) -> dict[str, tuple[int, int]]:
    out = {}
    for name, (kind, payload, _t) in sets.items():
        if kind == "labelled":
            ok = sum(int(band_count(infer(net, it["img"], device)[0] > THR) == it["n_gt"]) for it in payload)
            out[name] = (ok, len(payload))
        else:
            ps, exp = payload
            ok = sum(int(band_count(infer(net, model_input(cv2.imread(p), "center"), device)[0] > THR) == exp)
                     for p in ps)
            out[name] = (ok, len(ps))
    return out


@torch.no_grad()
def score_stems(net, stems, device: str) -> tuple[int, int]:
    """Exact-count score on specific crops (the newly added ones). Biased -- these are
    training data -- so it is evidence the adaptation took, never accuracy."""
    items, _ = discover()
    its = [i for i in items if i["stem"] in set(stems)]
    if not its:
        return 0, 0
    v = build_val(its, "center")
    return sum(int(band_count(infer(net, it["img"], device)[0] > THR) == it["n_gt"]) for it in v), len(v)


def verdict(before: dict, after: dict, sets: dict, site_types: set[str] | None = None):
    """(passed, rows). `site_types` scopes the gate to the types one site runs."""
    rows, passed = [], True
    for name in sorted(before):
        typ = sets[name][2]
        b, n = before[name]
        a, _ = after[name]
        d = a - b
        tol = TOL.get(name)
        if tol is None:
            status = f"not ranked ({NOT_RANKABLE.get(name, 'no band')})"
        elif site_types is not None and typ not in site_types:
            status = "out of scope for this site"
        elif d < -tol:
            status = f"REGRESSED (> tol {tol})"
            passed = False
        else:
            status = f"ok (tol {tol})"
        rows.append({"set": name, "n": n, "before": b, "after": a, "delta": d, "status": status})
    return passed, rows


def table(rows) -> str:
    w = max(len(r["set"]) for r in rows) + 2
    out = [f"{'set':<{w}}{'n':>5}{'before':>8}{'after':>7}{'delta':>7}   status"]
    for r in rows:
        out.append(f"{r['set']:<{w}}{r['n']:>5}{r['before']:>8}{r['after']:>7}{r['delta']:>+7}   {r['status']}")
    return "\n".join(out)
