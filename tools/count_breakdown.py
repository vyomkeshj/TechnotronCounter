"""Exact-count and error breakdown from a count_vs_filename json (centre input by default).

    python -m tools.count_breakdown <json> [mode] [run ...]
"""
import collections
import json
import sys


def main():
    d = json.load(open(sys.argv[1]))
    mode = sys.argv[2] if len(sys.argv) > 2 else "center"
    runs = sys.argv[3:] or sorted({k.split("|")[0] for k in d})
    for run in runs:
        rows = d.get(f"{run}|{mode}", [])
        for bucket, sel in (("unseen", True), ("trained", False)):
            rs = [r for r in rows if r["val"] == sel]
            if not rs:
                continue
            parts = []
            for e in sorted({r["expected"] for r in rs}):
                q = [r for r in rs if r["expected"] == e]
                err = collections.Counter(r["band"] - e for r in q)
                errs = dict(sorted((k, v) for k, v in err.items() if k))
                parts.append(f"{e}-sheet {err[0]}/{len(q)} misses {errs}")
            exact = sum(r["band"] == r["expected"] for r in rs)
            print(f"{run:18s} {bucket:7s} exact {exact}/{len(rs)} | " + " | ".join(parts))


if __name__ == "__main__":
    main()
