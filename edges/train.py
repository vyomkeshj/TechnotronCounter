"""Train the sheet-edge model on human-labelled column crops (all types together).

    python -m edges.train --name v1                  # full recipe
    python -m edges.train --name smoke --smoke       # plumbing check

Model   UNet base 16, 3 channels (fg logit, anisotropic flow dy/dx to sheet centre)
Loss    BCE(fg) + Dice(fg) + 2.0 * MSE(flow | fg)
Steps   8000, AdamW lr 1e-3 cosine, batch 32 windows of 256 px. Ship LAST.
Audit   band count + flow instances vs human polygons on pinned val frames,
        per type, attached to the run and written to audit.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import paths
from common.tracking import Tracker
from common.unet import UNet, save_checkpoint
from edges.dataset import WIDTH_MODES, EdgeDataset, build_val, discover, sync_split
from edges.evaluate import evaluate, preview

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MONITOR, GROUP = "Technotron Sheet Flow", "pipeline-edges"


def subsample_per_type(train: list[dict], limits: dict[str, int], seed: int) -> list[dict]:
    """Keep N crops of each limited type, SPREAD ACROSS FRAMES (take one crop per frame
    round-robin) so a small budget is not all from one photo. Deterministic in `seed`."""
    rng = random.Random(seed)
    out = [it for it in train if it["type"] not in limits]
    for typ, n in limits.items():
        by_frame: dict[str, list] = {}
        for it in train:
            if it["type"] == typ:
                by_frame.setdefault(it["frame"], []).append(it)
        frames = sorted(by_frame)
        rng.shuffle(frames)
        for fs in by_frame.values():
            fs.sort(key=lambda i: i["stem"])
        picked, i = [], 0
        while len(picked) < n and any(by_frame.values()):
            f = frames[i % len(frames)]
            if by_frame[f]:
                picked.append(by_frame[f].pop(0))
            i += 1
            if i > 10000:
                break
        out += picked
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--squash-weight", type=float, default=WIDTH_MODES["squash"],
                    help="share of training windows width-fitted by squash (rest: centre crop)")
    ap.add_argument("--lighting", nargs="?", const="v1", default="", choices=["", "v1", "v2"],
                    help="local glare/gradient/shadow aug: v1 (label-guided, measured negative) or v2 (label-free)")
    ap.add_argument("--only-types", default="", help="comma list, e.g. type3 or type2,type3 (transfer study)")
    ap.add_argument("--limit-per-type", default="", help="comma list type=N, e.g. type2=8,type10=4 "
                                                         "(data-scaling study; frame-stratified, deterministic)")
    ap.add_argument("--exclude-session", action="append", default=[],
                    help="drop every crop of this capture session from TRAINING (honest "
                         "cross-session test: evaluate on that session afterwards)")
    ap.add_argument("--init", default="", help="start from this checkpoint instead of random "
                                               "(field adaptation: fine-tune the live model)")
    ap.add_argument("--lr", type=float, default=1e-3, help="peak LR; use ~1e-4 when --init is set")
    ap.add_argument("--boost-stems", default="", help="file of crop stems, one per line, to "
                                                      "OVERSAMPLE (a handful of new site crops is "
                                                      "otherwise <3 %% of batches and never learnt)")
    ap.add_argument("--boost-factor", type=int, default=12, help="how many times to repeat each boosted crop")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.eval_every, args.workers = 60, 30, 2

    items, unlabelled = discover()
    split = sync_split(items)
    val_f, exc_f = set(split["val"]), set(split["excluded"])
    train = [it for it in items if it["frame"] not in val_f and it["frame"] not in exc_f]
    if args.exclude_session:
        from common import splits as _sp
        before = len(train)
        train = [it for it in train if not any(s in (_sp.session_of(it["frame"]) or "") for s in args.exclude_session)]
        print(f"HOLDOUT {args.exclude_session}: dropped {before - len(train)} training crops")
    if args.only_types:
        keep = {t.strip() for t in args.only_types.split(",")}
        train = [it for it in train if it["type"] in keep]
        print(f"ONLY TYPES {sorted(keep)}: {len(train)} training crops")
    if args.limit_per_type:
        limits = {k.strip(): int(v) for k, v in (p.split("=") for p in args.limit_per_type.split(","))}
        train = subsample_per_type(train, limits, args.seed)
        print(f"LIMITS {limits}: {len(train)} training crops "
              f"{ {t: sum(1 for i in train if i['type'] == t) for t in sorted({i['type'] for i in train})} }")
    boosted = 0
    if args.boost_stems:
        want = {l.strip() for l in open(args.boost_stems) if l.strip()}
        extra = [it for it in train if it["stem"] in want] * (args.boost_factor - 1)
        missing = want - {it["stem"] for it in train}
        if missing:
            print(f"WARNING: {len(missing)} boosted stems are not in TRAIN (val/excluded?): "
                  f"{sorted(missing)[:3]}")
        train = train + extra
        boosted = len(extra)
        print(f"BOOST x{args.boost_factor} on {len(want) - len(missing)} stems: "
              f"+{boosted} repeats -> {len(train)} sampling slots "
              f"({100 * (boosted + len(want) - len(missing)) / max(1, len(train)):.0f} %% of batches)")
    val_items = [it for it in items if it["frame"] in val_f]
    vals = {m: build_val(val_items, m) for m in ("center", "squash")}   # every run is scored both ways
    val = vals["center"]
    by = lambda xs: {t: sum(1 for i in xs if i["type"] == t) for t in sorted({i["type"] for i in xs})}
    print(f"edges/{args.name}: train {len(train)} crops {by(train)} | val {len(val)} crops {by(val)} "
          f"| unlabelled (ignored) {unlabelled}")
    types_without_val = set(by(train)) - set(by(val))
    if types_without_val:
        print(f"WARNING: no val crops for {sorted(types_without_val)} -- their accuracy is unmeasured")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    modes = {"center": 1 - args.squash_weight, "squash": args.squash_weight}
    ld = DataLoader(EdgeDataset(train, width_modes=modes, lighting=args.lighting), args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=True, pin_memory=True, persistent_workers=args.workers > 0,
                    prefetch_factor=4 if args.workers > 0 else None)
    if args.init:
        from common.unet import load_checkpoint
        net = load_checkpoint(args.init, DEVICE).train()
        print(f"INIT from {args.init} (fine-tune, lr {args.lr})")
    else:
        net = UNet(out_ch=3).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    out_dir = os.path.join(paths.RUNS, f"edges_{args.name}")
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(steps=args.steps, seed=args.seed, lighting=args.lighting, n_train=len(train), n_val=len(val), width_modes=modes)
    tr = Tracker(out_dir, f"edges_{args.name}", MONITOR, GROUP, cfg, args.steps,
                 hero=("val_all/count_mae", "val_all/inst_iou"))

    def batches():
        while True:
            yield from ld
    it, run, status = batches(), {}, "failed"
    try:
        for step in range(1, args.steps + 1):
            net.train()
            x, fg, fl = (t.to(DEVICE, non_blocking=True) for t in next(it))
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
                tr.add_scalar("lr", sch.get_last_lr()[0], step)
                allres = {mode: evaluate(net, v) for mode, v in vals.items()}
                res = allres["center"]
                for mode, r_ in allres.items():
                    prefix = "val" if mode == "center" else f"val_{mode}"
                    for g, m in r_.items():
                        for k, v in m.items():
                            if k != "n":
                                tr.add_scalar(f"{prefix}_{g}/{k}", v, step)
                if val:
                    tr.add_image("preview/val", preview(net, val), step)
                a = res.get("all", {})
                print(f"  {step:5d}/{args.steps} loss {float(loss):.3f} | band count MAE {a.get('count_mae', 0):.2f} "
                      f"exact {a.get('count_exact', 0):.0%} | inst MAE {a.get('inst_count_mae', 0):.2f} "
                      f"instIoU {a.get('inst_iou', 0):.3f}", flush=True)
        last = os.path.join(out_dir, "last.pt")
        save_checkpoint(last, net, step=args.steps, recipe=cfg)
        allres = {mode: evaluate(net, v) for mode, v in vals.items()}
        json.dump(allres, open(os.path.join(out_dir, "audit.json"), "w"), indent=1)
        tr.log_audit({f"audit_{mode}_{g}/{k}": v for mode, r_ in allres.items() for g, m in r_.items()
                      for k, v in m.items() if k != "n"}, args.steps,
                     note="audit_<input mode>_<type>/* = LAST checkpoint vs human polygons on pinned val frames.")
        status = "finished"
    finally:
        tr.close(status)
    print(f"\nSHIP: {last}")
    for mode, r_ in allres.items():
        for g, m in sorted(r_.items()):
            print(f"  {mode:6s} input  {g:8s} n={m['n']:3d} count MAE {m['count_mae']:.2f} "
                  f"exact {m['count_exact']:.0%} inst MAE {m['inst_count_mae']:.2f} instIoU {m['inst_iou']:.3f}")


if __name__ == "__main__":
    main()
