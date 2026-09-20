"""Compare edge runs by their audit.json (LAST checkpoint, pinned val), per input mode and type.

    python -m tools.compare_edge_runs edges_wcenter_s42 edges_wcenter_s7 ...

Runs sharing a name up to the trailing _s<seed> are grouped and shown as seed mean
plus the per-seed values, because single edge runs vary (same-seed replicates
have moved 18/18 exact to 15/18).
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

import paths

KEYS = (("count_mae", "band MAE", "{:.2f}"), ("count_exact", "exact", "{:.0%}"),
        ("count_within1", "<=1", "{:.0%}"), ("inst_count_mae", "inst MAE", "{:.2f}"), ("inst_iou", "instIoU", "{:.3f}"))


def main():
    runs = sys.argv[1:]
    groups: dict[str, list] = {}
    for r in runs:
        a = json.load(open(os.path.join(paths.RUNS, r, "audit.json")))
        if "center" not in a:                       # old single-mode audit
            continue
        groups.setdefault(re.sub(r"_s\d+$", "", r), []).append((r, a))
    for mode in ("center", "squash"):
        for typ in ("type2", "type3", "all"):
            print(f"\n=== input {mode} | {typ} ===")
            print(f"{'train mix':22s}" + "".join(f"{h:>22s}" for _, h, _ in KEYS))
            for g, rs in groups.items():
                cells = []
                for k, _, fmt in KEYS:
                    vals = [a[mode][typ][k] for _, a in rs if typ in a[mode]]
                    per = "/".join(fmt.format(v) for v in vals)
                    cells.append(f"{fmt.format(np.mean(vals)):>7s} ({per})")
                print(f"{g:22s}" + "".join(f"{c:>22s}" for c in cells))


if __name__ == "__main__":
    main()
