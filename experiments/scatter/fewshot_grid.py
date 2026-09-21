"""Run the new-type grid with a concurrency limit.

    python -m experiments.scatter.fewshot_grid --approach prod [--types type10 type2 type3]
        [--ks 0 2 4 8 16 32] [--seeds 7] [--parallel 5] [--log-dir DIR]

Skips any (approach, type, k, seed) whose runs/fewshot_*/audit.json already exists,
so it can be re-invoked to fill gaps or add seeds.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time

import paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--approach", required=True, choices=["prod", "scatter"])
    ap.add_argument("--types", nargs="+", default=["type10", "type2", "type3"])
    ap.add_argument("--ks", nargs="+", type=int, default=[0, 2, 4, 8, 16, 32])
    ap.add_argument("--seeds", nargs="+", type=int, default=[7])
    ap.add_argument("--parallel", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-dir", default=paths.RUNS)
    args = ap.parse_args()

    jobs = []
    for s in args.seeds:
        for t in args.types:
            for k in args.ks:
                name = f"fewshot_{args.approach}_{t}_k{k:02d}_s{s}"
                if os.path.exists(os.path.join(paths.RUNS, name, "audit.json")):
                    continue
                jobs.append((name, t, k, s))
    print(f"{len(jobs)} jobs, {args.parallel} in parallel", flush=True)
    sem = threading.Semaphore(args.parallel)
    t0 = time.time()

    def run(name, t, k, s):
        with sem:
            log = os.path.join(args.log_dir, f"_{name}.log")
            cmd = [sys.executable, "-u", "-m", "experiments.scatter.fewshot", "--held-out", t,
                   "--k", str(k), "--approach", args.approach, "--seed", str(s),
                   "--workers", str(args.workers)]
            with open(log, "w") as fh:
                rc = subprocess.run(cmd, cwd=paths.ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
            tail = ""
            try:
                tail = [l for l in open(log).read().splitlines() if l.startswith("RESULT")][-1]
            except Exception:
                pass
            print(f"[{time.time() - t0:6.0f}s] rc={rc} {tail or name}", flush=True)

    threads = [threading.Thread(target=run, args=j) for j in jobs]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print("GRID DONE", flush=True)


if __name__ == "__main__":
    main()
