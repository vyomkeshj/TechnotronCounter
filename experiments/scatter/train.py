"""Train on SCATTER scenes and watch what it does to real-crop counting.

    python -m experiments.scatter.train --name scatter_only --mix 0.0
    python -m experiments.scatter.train --name scatter_mix  --mix 0.5

`--mix` is the fraction of batches drawn from the PRODUCTION window sampler; the rest
are scatter scenes. Everything else -- UNet base 16, 3 channels, BCE + Dice + 2*MSE on
the aniso flow, 8000 steps, AdamW 1e-3 cosine, batch 32 -- is the champion recipe
UNCHANGED, so the only variable is the data.

THREE THINGS ARE MEASURED AT EVERY EVAL, and the second and third are the point:

  scatter/*   fg IoU / precision / recall on held-out scenes built ONLY from val crops.
              Floor to beat: the "segment everything lit" cheat, printed at startup.
  real/*      the SHIPPED band counter on the real pinned val crops, per type, vs the
              human label count. This is the deviation the experiment is about.
  known/*     a fixed random sample of unseen older crops whose totals are known by
              construction (type3 = 50 sheets, type10 = 24). Never in any training set,
              and known without labels.

Leak control: scatter TRAINING scenes are cut only from train crops, scatter VAL scenes
only from val crops. A patch is a piece of a crop, so mixing the two would put val
pixels in training.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import paths
from common.tracking import Tracker
from common.unet import UNet, save_checkpoint
from edges.counter import THR, band_count, infer
from edges.dataset import EdgeDataset, build_val, discover, model_input, sync_split
from edges.evaluate import evaluate
from experiments.scatter.dataset import SIZE, ScatterDataset, scene

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MONITOR, GROUP = "Technotron Sheet Flow", "scatter-experiment"
OLD = os.environ.get("TECHNOTRON_OLD_CROPS", r"D:\Projects\technotron\edge_dataset\good_images")
KNOWN = {                                          # unseen, totals known by construction
    "t3": (os.path.join(OLD, "type3", "columnsnew", "good", "*.png"), 50),
    "t10": (os.path.join(OLD, "type10", "columnsnew", "good", "*.png"), 24),
}
N_KNOWN = 40                                       # per type, fixed random sample


class MixDataset(Dataset):
    def __init__(self, prod: EdgeDataset, scat: ScatterDataset, p_prod: float):
        self.prod, self.scat, self.p = prod, scat, p_prod

    def __len__(self):
        return len(self.scat)

    def __getitem__(self, i):
        if self.p > 0 and random.random() < self.p:
            return self.prod[random.randrange(len(self.prod))]
        return self.scat[i]


def make_val_scenes(val_items, n=48, seed=1234):
    out = []
    for i in range(n):
        img, lab = scene(val_items, random.Random(seed + i))
        out.append((img, (lab > 0)))
    return out


@torch.no_grad()
def score_scatter(net, scenes):
    """fg IoU / precision / recall over the held-out scenes."""
    inter = union = tp = pp = ap = 0
    for img, gt in scenes:
        pr = infer(net, img, DEVICE)[0] > THR
        inter += int((pr & gt).sum()); union += int((pr | gt).sum())
        tp += int((pr & gt).sum()); pp += int(pr.sum()); ap += int(gt.sum())
    return {"iou": inter / max(1, union), "precision": tp / max(1, pp), "recall": tp / max(1, ap)}


@torch.no_grad()
def score_known(net, known):
    out = {}
    for name, (ps, exp) in known.items():
        errs = []
        for p in ps:
            c = band_count(infer(net, model_input(cv2.imread(p), "center"), DEVICE)[0] > THR)
            errs.append(c - exp)
        e = np.array(errs, float)
        out[name] = {"exact": float((e == 0).mean()), "mae": float(np.abs(e).mean()),
                     "bias": float(e.mean()), "n": len(e)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--mix", type=float, default=0.0, help="fraction of batches from the production sampler")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--lighting", default="v2")
    ap.add_argument("--only-types", default="", help="comma list: train on these types ONLY (both the "
                                                    "production windows AND the scatter patches) -- "
                                                    "leave-one-type-out transfer test")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.eval_every, args.workers = 60, 30, 2

    items, _ = discover()
    split = sync_split(items, write=False)
    val_f, exc_f = set(split["val"]), set(split["excluded"])
    train = [it for it in items if it["frame"] not in val_f and it["frame"] not in exc_f]
    if args.only_types:
        keep = {t.strip() for t in args.only_types.split(",")}
        train = [it for it in train if it["type"] in keep]
        print(f"ONLY TYPES {sorted(keep)}: {len(train)} training crops -- held-out type(s) "
              f"{sorted({it['type'] for it in items} - keep)} appear NOWHERE in training")
    val_items = [it for it in items if it["frame"] in val_f]
    val = build_val(val_items, "center")
    print(f"scatter/{args.name}: {len(train)} train crops -> scatter scenes | "
          f"{len(val)} real val crops | mix {args.mix:.0%} production")

    known = {}
    for k, (g, exp) in KNOWN.items():
        ps = sorted(glob.glob(g))
        if ps:
            known[k] = (random.Random(0).sample(ps, min(N_KNOWN, len(ps))), exp)
            print(f"  known/{k}: {len(known[k][0])} unseen crops, expected {exp}")
        else:
            print(f"  known/{k}: unavailable (set TECHNOTRON_OLD_CROPS)")

    val_scenes = make_val_scenes(val_items)
    lit = np.mean([(i.max(axis=2) > 0).mean() for i, _ in val_scenes])
    cov = np.mean([g.mean() for _, g in val_scenes])
    print(f"  val scenes: {len(val_scenes)} | lit {lit:.1%} labelled {cov:.1%} "
          f"-> 'segment everything lit' cheat IoU {cov / lit:.3f}")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    scat = ScatterDataset(train, length=200000, seed=args.seed)
    prod = EdgeDataset(train, width_modes={"center": 1.0, "squash": 0.0}, lighting=args.lighting)
    ds = MixDataset(prod, scat, args.mix)
    ld = DataLoader(ds, args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                    pin_memory=True, persistent_workers=args.workers > 0,
                    prefetch_factor=4 if args.workers > 0 else None)

    net = UNet(out_ch=3).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    out_dir = os.path.join(paths.RUNS, f"scatter_{args.name}")
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(steps=args.steps, seed=args.seed, mix=args.mix, lighting=args.lighting,
               n_train=len(train), scene_size=SIZE)
    tr = Tracker(out_dir, f"scatter_{args.name}", MONITOR, GROUP, cfg, args.steps,
                 hero=("real_all/count_exact", "scatter/iou"))

    hist = []

    def batches():
        while True:
            yield from ld
    it_, run = batches(), {}
    status = "failed"
    try:
        for step in range(1, args.steps + 1):
            net.train()
            x, fg, fl = (t.to(DEVICE, non_blocking=True) for t in next(it_))
            out = net(x)
            bce = F.binary_cross_entropy_with_logits(out[:, :1], fg)
            p = torch.sigmoid(out[:, :1])
            dice = 1 - ((2 * (p * fg).sum((1, 2, 3)) + 1e-6) / ((p + fg).sum((1, 2, 3)) + 1e-6)).mean()
            mse = (((out[:, 1:3] - fl) ** 2) * fg).sum() / (fg.sum() * 2 + 1e-6)
            loss = bce + dice + 2.0 * mse
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
            for k, v in (("bce", bce), ("dice", dice), ("mse", mse), ("total", loss)):
                run[k] = run.get(k, 0.0) + float(v)
            if step % args.eval_every == 0 or step == args.steps:
                n = (step - 1) % args.eval_every + 1
                for k in run:
                    tr.add_scalar(f"loss/{k}", run[k] / n, step)
                run = {}
                net.eval()
                sc = score_scatter(net, val_scenes)
                real = evaluate(net, val)
                kn = score_known(net, known) if known else {}
                for k, v in sc.items():
                    tr.add_scalar(f"scatter/{k}", v, step)
                for g, m in real.items():
                    for k, v in m.items():
                        if k != "n":
                            tr.add_scalar(f"real_{g}/{k}", v, step)
                for g, m in kn.items():
                    for k, v in m.items():
                        if k != "n":
                            tr.add_scalar(f"known_{g}/{k}", v, step)
                a = real.get("all", {})
                row = {"step": step, "scatter_iou": sc["iou"],
                       "real_exact": a.get("count_exact", 0), "real_mae": a.get("count_mae", 0),
                       **{f"known_{g}_exact": m["exact"] for g, m in kn.items()},
                       **{f"known_{g}_mae": m["mae"] for g, m in kn.items()}}
                hist.append(row)
                print(f"  {step:5d}/{args.steps} loss {float(loss):.3f} | scatter IoU {sc['iou']:.3f} "
                      f"P {sc['precision']:.3f} R {sc['recall']:.3f} | REAL exact "
                      f"{a.get('count_exact', 0):.0%} MAE {a.get('count_mae', 0):.2f} | "
                      + " ".join(f"known/{g} {m['exact']:.0%} MAE {m['mae']:.2f}" for g, m in kn.items()),
                      flush=True)
        last = os.path.join(out_dir, "last.pt")
        save_checkpoint(last, net, step=args.steps, recipe=cfg)
        final = {"scatter": score_scatter(net, val_scenes),
                 "real": evaluate(net, val),
                 "known": score_known(net, known) if known else {},
                 "history": hist}
        json.dump(final, open(os.path.join(out_dir, "audit.json"), "w"), indent=1, default=float)
        tr.log_audit({f"final_scatter/{k}": v for k, v in final["scatter"].items()}
                     | {f"final_real_{g}/{k}": v for g, m in final["real"].items()
                        for k, v in m.items() if k != "n"}
                     | {f"final_known_{g}/{k}": v for g, m in final["known"].items()
                        for k, v in m.items() if k != "n"}, args.steps,
                     note="scatter/* = held-out scenes; real/* and known/* = the shipped band counter "
                          "on REAL crops. known/* totals are known by construction (t3=50, t10=24).")
        status = "finished"
    finally:
        tr.close(status)
    print(f"\nSHIP: {last}")


if __name__ == "__main__":
    main()
