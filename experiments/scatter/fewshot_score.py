"""Score every few-shot checkpoint on ALL THREE types: the new one, and the two it must
not have damaged.

    python -m experiments.scatter.fewshot_score            # -> runs/_fewshot_full.csv + table

The fine-tune trainer (fewshot.py) only scores the held-out type. This adds the
condition that matters just as much: accuracy on the OTHER two types, before vs after,
judged with the same measured tolerances as field/gate.py (all_type* 1 crop, old type3
10 crops; old type10 reported, NOT ranked -- seed spread 23/122).

  new     exact on every labelled crop of the held-out type that was not fine-tuned on
  others  exact on every labelled crop of each other type (they were replayed in full
          during the fine-tune, so this is the gate's all_type* convention), and the 300
          unseen older type3 crops
  pass    new type >= target AND no other set below its base by more than its tolerance

The base row (k=0) supplies the "before" for the others. Idempotent: rows already in the
csv are kept and skipped.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import cv2
import torch

import paths
from common.unet import load_checkpoint
from edges.counter import THR, band_count, infer
from edges.dataset import build_val, discover, model_input
from experiments.scatter.fewshot import BASES, OLDSETS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TYPES = ("type2", "type3", "type10")
TOL = {"type2": 1, "type3": 1, "type10": 1, "old_t3": 10}
OUT = os.path.join(paths.RUNS, "_fewshot_full.csv")


@torch.no_grad()
def exact(net, val):
    return sum(int(band_count(infer(net, it["img"], DEVICE)[0] > THR) == it["n_gt"]) for it in val)


@torch.no_grad()
def exact_paths(net, ps, exp):
    return sum(int(band_count(infer(net, model_input(cv2.imread(p), "center"), DEVICE)[0] > THR) == exp)
               for p in ps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=float, default=0.98, help="'perfect' = at least this exact fraction")
    args = ap.parse_args()

    items, _ = discover()
    by_type = {t: [it for it in items if it["type"] == t] for t in TYPES}
    vals = {t: build_val(by_type[t], "center") for t in TYPES}
    # OLDSETS is keyed by type; rename so the older unseen sets never collide with the
    # labelled per-type columns (a first version overwrote type3 72 with old type3 300)
    old = {{"type3": "old_t3", "type10": "old_t10"}[k]: (sorted(glob.glob(g)), exp)
           for k, (g, exp) in OLDSETS.items()}
    old = {k: v for k, v in old.items() if v[0]}

    done = {}
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)):
            done[r["run"]] = r

    rows = []
    for aud in sorted(glob.glob(os.path.join(paths.RUNS, "fewshot_*", "audit.json"))):
        name = os.path.basename(os.path.dirname(aud))
        if name in done:
            rows.append(done[name]); continue
        a = json.load(open(aud))
        T, K, app = a["held_out"], int(a["k"]), a["approach"]
        ck = os.path.join(paths.RUNS, name, "last.pt") if K > 0 else \
            os.path.join(paths.RUNS, BASES[app][T], "last.pt")
        picked = set()
        if K > 0:
            picked = set(torch.load(ck, map_location="cpu", weights_only=False).get("recipe", {}).get("picked", []))
        net = load_checkpoint(ck, DEVICE)
        r = {"run": name, "approach": app, "held_out": T, "k": K, "seed": a["seed"]}
        for t in TYPES:
            v = [it for it in vals[t] if it["stem"] not in picked] if t == T else vals[t]
            r[f"{t}_ok"], r[f"{t}_n"] = exact(net, v), len(v)
        for k, (ps, exp) in old.items():
            r[f"{k}_ok"], r[f"{k}_n"] = exact_paths(net, ps, exp), len(ps)
        rows.append(r)
        print(f"  scored {name}: " + " ".join(f"{t} {r[f'{t}_ok']}/{r[f'{t}_n']}" for t in TYPES)
              + " ".join(f" {k} {r[f'{k}_ok']}/{r[f'{k}_n']}" for k in old), flush=True)

    # verdicts: compare each row's OTHER types with its own base (k=0, same approach, same held-out)
    base = {(r["approach"], r["held_out"]): r for r in rows if int(r["k"]) == 0}
    for r in rows:
        b = base.get((r["approach"], r["held_out"]))
        T = r["held_out"]
        r["new_exact"] = int(r[f"{T}_ok"]) / max(1, int(r[f"{T}_n"]))
        reg = []
        if b:
            for t in TYPES:
                if t != T and int(r[f"{t}_ok"]) < int(b[f"{t}_ok"]) - TOL[t]:
                    reg.append(f"{t} {b[f'{t}_ok']}->{r[f'{t}_ok']}")
            if T != "type3" and "old_t3_ok" in r and int(r["old_t3_ok"]) < int(b["old_t3_ok"]) - TOL["old_t3"]:
                reg.append(f"old_t3 {b['old_t3_ok']}->{r['old_t3_ok']}")
        r["regressions"] = "; ".join(reg)
        r["pass"] = int(r["new_exact"] >= args.target and not reg)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    print(f"\n{'approach':<8}{'new type':<8}{'k':>3}{'new exact':>12}{'type2':>8}{'type3':>8}{'type10':>8}"
          f"{'old_t3':>9}{'old_t10':>9}  pass  regressions")
    for r in sorted(rows, key=lambda r: (r["held_out"], r["approach"], int(r["k"]))):
        print(f"{r['approach']:<8}{r['held_out']:<8}{int(r['k']):>3}{r['new_exact']:>12.1%}"
              f"{r['type2_ok']:>8}{r['type3_ok']:>8}{r['type10_ok']:>8}"
              f"{r.get('old_t3_ok', ''):>9}{r.get('old_t10_ok', ''):>9}  {'YES' if r['pass'] else '-':>4}  {r['regressions']}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
