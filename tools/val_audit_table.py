"""Per-type exact counts on the pinned val crops, straight from each run's audit.json.

    python -m tools.val_audit_table <run> [<run> ...] [mode=center]
"""
import json
import os
import sys

import paths


def main():
    runs = [a for a in sys.argv[1:] if a not in ("center", "squash")]
    mode = "squash" if "squash" in sys.argv else "center"
    for r in runs:
        p = os.path.join(paths.RUNS, r, "audit.json")
        if not os.path.exists(p):
            print(f"{r:20s} no audit.json")
            continue
        a = json.load(open(p))
        a = a[mode] if mode in a else a
        cells = []
        for t in sorted(a):
            if t == "all":
                continue
            m = a[t]
            cells.append(f"{t} {round(m['count_exact'] * m['n']):>2d}/{m['n']}")
        allm = a.get("all", {})
        tot = f"all {round(allm.get('count_exact', 0) * allm.get('n', 0)):>3d}/{allm.get('n', 0)}" if allm else ""
        print(f"{r:20s} " + " | ".join(cells) + (f" | {tot}" if tot else ""))


if __name__ == "__main__":
    main()
