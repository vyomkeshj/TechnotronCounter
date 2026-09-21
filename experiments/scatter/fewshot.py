"""NEW-TYPE TEST: how many crops of an unseen type does it take to fine-tune a model
trained on the OTHER two types up to perfect counting -- for both approaches.

    python -m experiments.scatter.fewshot --held-out type10 --k 4 --approach prod
    python -m experiments.scatter.fewshot --held-out type10 --k 4 --approach scatter
    python -m experiments.scatter.fewshot --held-out type10 --k 0 --approach prod   # the base itself

base      prod    : the production-only model trained without the held-out type
                    (edges_no_t2 / edges_no_t3 / edges_wlight2_s7 for type10)
          scatter : the mix50 scatter model trained without it (scatter_no_<t>)
fine-tune the FIELD RECIPE, unchanged, so the answer is what `field.adapt` needs on
          site: 2500 steps, lr 1e-4, cosine, the K new crops oversampled x12, the other
          two types replayed at full strength. prod fine-tunes on production windows;
          scatter fine-tunes on the 50/50 window+scatter mix (the K crops feed both).
K crops   frame-stratified (`edges.train.subsample_per_type`), from TRAIN frames only.
scored on EVERY labelled crop of the held-out type that is not one of the K, plus the
          300 unseen older type3 crops when the held-out type is type3 (old type10 is
          reported but NOT ranked: seed spread 23/122). The base never saw the type,
          so all of those are genuinely unseen.

One row per run -> runs/_fewshot.csv.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import paths
from common.unet import load_checkpoint, save_checkpoint
from edges.counter import THR, band_count, infer
from edges.dataset import EdgeDataset, build_val, discover, model_input, sync_split
from edges.train import subsample_per_type
from experiments.scatter.dataset import ScatterDataset
from experiments.scatter.train import MixDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASES = {
    "prod": {"type10": "edges_wlight2_s7", "type2": "edges_no_t2", "type3": "edges_no_t3"},
    "scatter": {"type10": "scatter_no_t10", "type2": "scatter_no_t2", "type3": "scatter_no_t3"},
}
OLD = os.environ.get("TECHNOTRON_OLD_CROPS", r"D:\Projects\technotron\edge_dataset\good_images")
OLDSETS = {"type3": (os.path.join(OLD, "type3", "columnsnew", "good", "*.png"), 50),
           "type10": (os.path.join(OLD, "type10", "columnsnew", "good", "*.png"), 24)}


@torch.no_grad()
def exact_on(net, val):
    ok = sum(int(band_count(infer(net, it["img"], DEVICE)[0] > THR) == it["n_gt"]) for it in val)
    return ok, len(val)


@torch.no_grad()
def exact_old(net, typ):
    if typ not in OLDSETS:
        return None
    ps, exp = sorted(glob.glob(OLDSETS[typ][0])), OLDSETS[typ][1]
    if not ps:
        return None
    ok = sum(int(band_count(infer(net, model_input(cv2.imread(p), "center"), DEVICE)[0] > THR) == exp)
             for p in ps)
    return ok, len(ps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--held-out", required=True, choices=["type2", "type3", "type10"])
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--approach", required=True, choices=["prod", "scatter"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--boost", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    T, K = args.held_out, args.k
    name = f"fewshot_{args.approach}_{T}_k{K:02d}_s{args.seed}"
    base = os.path.join(paths.RUNS, BASES[args.approach][T], "last.pt")

    items, _ = discover()
    split = sync_split(items, write=False)
    val_f, exc_f = set(split["val"]), set(split["excluded"])
    train_all = [it for it in items if it["frame"] not in val_f and it["frame"] not in exc_f]
    others = [it for it in train_all if it["type"] != T]
    picked = subsample_per_type([it for it in train_all if it["type"] == T], {T: K}, args.seed) if K else []
    picked_stems = {it["stem"] for it in picked}
    heldout = [it for it in items if it["type"] == T and it["stem"] not in picked_stems]
    val = build_val(heldout, "center")
    print(f"{name}: base {BASES[args.approach][T]} | +{len(picked)} {T} crops from "
          f"{len({it['frame'] for it in picked})} frames | scored on {len(val)} unseen {T} crops")

    net = load_checkpoint(base, DEVICE)
    out_dir = os.path.join(paths.RUNS, name)
    os.makedirs(out_dir, exist_ok=True)
    before, before_old = exact_on(net, val), exact_old(net, T)

    if K > 0:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        train = others + picked * args.boost
        prod = EdgeDataset(train, width_modes={"center": 1.0, "squash": 0.0}, lighting="v2")
        if args.approach == "scatter":
            ds = MixDataset(prod, ScatterDataset(train, length=100000, seed=args.seed), 0.5)
        else:
            ds = prod
        ld = DataLoader(ds, 32, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True,
                        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers > 0 else None)
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
        it_ = iter(ld)
        for step in range(1, args.steps + 1):
            try:
                x, fg, fl = next(it_)
            except StopIteration:
                it_ = iter(ld); x, fg, fl = next(it_)
            x, fg, fl = (t.to(DEVICE, non_blocking=True) for t in (x, fg, fl))
            out = net(x)
            bce = F.binary_cross_entropy_with_logits(out[:, :1], fg)
            p = torch.sigmoid(out[:, :1])
            dice = 1 - ((2 * (p * fg).sum((1, 2, 3)) + 1e-6) / ((p + fg).sum((1, 2, 3)) + 1e-6)).mean()
            mse = (((out[:, 1:3] - fl) ** 2) * fg).sum() / (fg.sum() * 2 + 1e-6)
            loss = bce + dice + 2.0 * mse
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
            if step % 500 == 0:
                net.eval()
                ok, n = exact_on(net, val)
                print(f"  {step:5d}/{args.steps} loss {float(loss):.3f} | unseen {T} {ok}/{n}", flush=True)
                net.train()
        net.eval()
        save_checkpoint(os.path.join(out_dir, "last.pt"), net, step=args.steps,
                        recipe=dict(approach=args.approach, held_out=T, k=K, seed=args.seed,
                                    base=BASES[args.approach][T], picked=sorted(picked_stems)))

    after, after_old = exact_on(net, val), exact_old(net, T)
    row = {"approach": args.approach, "held_out": T, "k": K, "seed": args.seed,
           "base": BASES[args.approach][T], "n_unseen": after[1],
           "before": before[0], "after": after[0],
           "old_before": before_old[0] if before_old else "", "old_after": after_old[0] if after_old else "",
           "old_n": after_old[1] if after_old else "", "picked_frames": len({it["frame"] for it in picked})}
    json.dump(row, open(os.path.join(out_dir, "audit.json"), "w"), indent=1)
    dst = os.path.join(paths.RUNS, "_fewshot.csv")
    new = not os.path.exists(dst)
    with open(dst, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"RESULT {name}: unseen {T} {before[0]} -> {after[0]} / {after[1]}"
          + (f" | old {before_old[0]} -> {after_old[0]} / {after_old[1]}" if after_old else ""))


if __name__ == "__main__":
    main()
