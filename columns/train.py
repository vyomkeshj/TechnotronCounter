"""Train the column instance model (flow + good/cropped class), two stages.

    python -m columns.train --name v1                  # full recipe (~45 min on an RTX 4090)
    python -m columns.train --name smoke --smoke       # 2-minute plumbing check

Stage 1  3 channels (fg logit, flow dy, flow dx), from scratch, 12000 steps, lr 1e-3 cosine.
Stage 2  +1 channel P(cropped), initialised from stage-1 LAST, 4000 steps, lr 3e-4.
Loss     BCE(fg) + Dice(fg) + 2.0 * MSE(flow | fg & valid) [+ 1.0 * BCE(cls | fg & valid)]
Ship     runs/columns_<name>/stage2/last.pt  -- the LAST checkpoint, no selection.

Recipe facts this encodes (all measured, see README "Why the recipe is what it is"):
  * 3000 steps under-trains flow models; 4-fold CV showed the curve flat >= 8000.
  * checkpoint selection on an 18-50 frame val set picks noise (+-0.05): ship last.
  * TTA / SWA / weight averaging: neutral-to-negative for attractor fields.
Validation numbers in audit.json are honest ONLY for frames listed in splits.json
"val"; the in-training val/* curves are the same frames, used for curves only.
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
from columns.dataset import ColumnDataset, build_val, discover, legacy_items, sync_split
from columns.evaluate import evaluate, preview
from common.tracking import Tracker
from common.unet import UNet, save_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MONITOR, GROUP = "Technotron Column Flow", "pipeline-columns"


def dice_bce(logit, target):
    bce = F.binary_cross_entropy_with_logits(logit, target)
    p = torch.sigmoid(logit)
    dice = 1 - ((2 * (p * target).sum((1, 2, 3)) + 1e-6) / ((p + target).sum((1, 2, 3)) + 1e-6)).mean()
    return bce, dice


def run_stage(stage, out_dir, args, inst, fg_only, val, init=None):
    with_cls = stage == 2
    steps = args.steps2 if with_cls else args.steps1
    lr = args.lr if getattr(args, "lr", 0.0) else (3e-4 if with_cls else 1e-3)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    ds = ColumnDataset(inst, fg_only)
    ld = DataLoader(ds, args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                    pin_memory=True, persistent_workers=args.workers > 0,
                    prefetch_factor=4 if args.workers > 0 else None)
    net = UNet(out_ch=4 if with_cls else 3)
    if init:
        src = torch.load(init, map_location="cpu", weights_only=False)["model"]
        sd = net.state_dict()
        for k, v in src.items():
            if sd[k].shape == v.shape:
                sd[k] = v
            else:                                  # head 3 -> 4 channels: copy, new row zero
                sd[k][: v.shape[0]] = v
                sd[k][v.shape[0]:] = 0
        net.load_state_dict(sd)
    net.to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(stage=stage, steps=steps, lr=lr, seed=args.seed, n_inst=len(inst), n_fg_only=len(fg_only),
               n_val=len(val))
    tr = Tracker(out_dir, f"columns_{args.name}_s{stage}", MONITOR, GROUP, cfg, steps,
                 hero=("val_deploy/inst_iou", "val_deploy/count_mae"))

    def batches():
        while True:
            yield from ld
    it, run, status = batches(), {}, "failed"
    try:
        for step in range(1, steps + 1):
            net.train()
            x, fg, fl, valid, cls = (t.to(DEVICE, non_blocking=True) for t in next(it))
            out = net(x)
            bce, dice = dice_bce(out[:, :1], fg)
            w = fg * valid[:, None, None, None]
            mse = (((out[:, 1:3] - fl) ** 2) * w).sum() / (w.sum() * 2 + 1e-6)
            loss = bce + dice + 2.0 * mse
            parts = {"bce": bce, "dice": dice, "mse": mse}
            if with_cls:
                cl = (F.binary_cross_entropy_with_logits(out[:, 3:4], cls, reduction="none") * w).sum() / (w.sum() + 1e-6)
                loss = loss + cl
                parts["cls"] = cl
            parts["total"] = loss
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + float(v)
            if step % args.eval_every == 0 or step == steps:
                n = (step - 1) % args.eval_every + 1
                for k in run:
                    tr.add_scalar(f"loss/{k}", run[k] / n, step)
                run = {}
                tr.add_scalar("lr", sch.get_last_lr()[0], step)
                res = evaluate(net, val)
                for g, m in res.items():
                    for k in ("inst_iou", "count_acc", "count_mae", "merge_rate", "cls_acc"):
                        if k in m:
                            tr.add_scalar(f"val_{g}/{k}", m[k], step)
                tr.add_image("preview/val", preview(net, val), step)
                d = res.get("deploy", {})
                print(f"  S{stage} {step:5d}/{steps} loss {float(loss):.3f} | deploy instIoU {d.get('inst_iou', 0):.3f} "
                      f"countAcc {d.get('count_acc', 0):.2f} MAE {d.get('count_mae', 0):.2f}"
                      + (f" cls {d.get('cls_acc', 0):.3f}" if with_cls else "")
                      + " | " + " ".join(f"{g} {m['inst_iou']:.3f}" for g, m in sorted(res.items()) if g != "deploy"),
                      flush=True)
        last = os.path.join(out_dir, "last.pt")
        save_checkpoint(last, net, step=steps, recipe=cfg)
        res = evaluate(net, val)
        json.dump(res, open(os.path.join(out_dir, "audit.json"), "w"), indent=1)
        tr.log_audit({f"audit_{g}/{k}": v for g, m in res.items() for k, v in m.items() if k != "n"}, steps,
                     note="audit/* = LAST checkpoint on pinned val frames; no checkpoint selection.")
        status = "finished"
        return last, res
    finally:
        tr.close(status)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps1", type=int, default=12000)
    ap.add_argument("--steps2", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--no-legacy", action="store_true", help="drop the fg-only legacy frames (untested)")
    ap.add_argument("--init", default="", help="fine-tune from this 4-ch checkpoint: runs STAGE 2 ONLY "
                                               "(field adaptation -- see field/adapt.py)")
    ap.add_argument("--lr", type=float, default=0.0, help="override the stage LR; use ~5e-5 with --init")
    ap.add_argument("--boost-stems", default="", help="file of frame stems, one per line, to OVERSAMPLE "
                                                      "(3-4 new site frames are otherwise lost in the pool)")
    ap.add_argument("--boost-factor", type=int, default=12)
    ap.add_argument("--smoke", action="store_true", help="tiny run to check plumbing")
    args = ap.parse_args()
    if args.smoke:
        args.steps1, args.steps2, args.eval_every, args.workers = 60, 40, 30, 2

    items = discover()
    split = sync_split(items)
    val_set, exc = set(split["val"]), set(split["excluded"])
    inst = [it for it in items if it["stem"] not in val_set and it["stem"] not in exc]
    val = build_val([it for it in items if it["stem"] in val_set])
    fg_only = [] if args.no_legacy else legacy_items()
    groups = {}
    for it in val:
        groups[it["group"]] = groups.get(it["group"], 0) + 1
    print(f"columns/{args.name}: train {len(inst)} instance frames (x4) + {len(fg_only)} fg-only | "
          f"val {len(val)} {groups} | excluded {len(exc)}")
    if args.boost_stems:
        want = {l.strip() for l in open(args.boost_stems) if l.strip()}
        extra = [it for it in inst if it["stem"] in want] * (args.boost_factor - 1)
        missing = want - {it["stem"] for it in inst}
        if missing:
            print(f"WARNING: {len(missing)} boosted stems are not in TRAIN (val/excluded?): {sorted(missing)[:3]}")
        inst = inst + extra
        print(f"BOOST x{args.boost_factor} on {len(want) - len(missing)} frames -> {len(inst)} sampling slots")
    base = os.path.join(paths.RUNS, f"columns_{args.name}")
    if args.init:                       # field adaptation: stage 2 only, from the live model
        print(f"INIT from {args.init} -- stage 2 only, {args.steps2} steps, lr {args.lr or 3e-4}")
        last, res = run_stage(2, os.path.join(base, "stage2"), args, inst, fg_only, val, init=args.init)
    else:
        s1, _ = run_stage(1, os.path.join(base, "stage1"), args, inst, fg_only, val)
        last, res = run_stage(2, os.path.join(base, "stage2"), args, inst, fg_only, val, init=s1)
    print(f"\nSHIP: {last}")
    for g, m in sorted(res.items()):
        print(f"  {g:12s} n={m['n']:3d} instIoU {m['inst_iou']:.3f} countAcc {m['count_acc']:.2f} "
              f"MAE {m['count_mae']:.2f} cls {m.get('cls_acc', float('nan')):.3f}")


if __name__ == "__main__":
    main()
