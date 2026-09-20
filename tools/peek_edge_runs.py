"""Latest per-type validation counts from edge runs still training (reads TB event files).

    python -m tools.peek_edge_runs [glob=edges_*]
"""
import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import paths


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "edges_*"
    for d in sorted(glob.glob(os.path.join(paths.RUNS, pat))):
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        tags = set(ea.Tags()["scalars"])
        cells, step = [], None
        for mode, pre in (("center", "val"), ("squash", "val_squash")):
            for typ in ("type2", "type3"):
                t_mae, t_ex = f"{pre}_{typ}/count_mae", f"{pre}_{typ}/count_exact"
                if t_mae in tags:
                    m, e = ea.Scalars(t_mae)[-1], ea.Scalars(t_ex)[-1]
                    step = m.step
                    cells.append(f"{mode}/{typ} MAE {m.value:5.2f} exact {e.value:4.0%}")
        print(f"{os.path.basename(d):20s} step {step}: " + " | ".join(cells))


if __name__ == "__main__":
    main()
