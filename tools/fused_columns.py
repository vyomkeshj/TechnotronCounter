"""Find FUSED columns: a predicted column mask holding >= 25 % of each of 2+ labelled columns (all labelled frames).
    python -m tools.fused_columns   -> prints them, writes runs/_t1/fused/<name>.json
    python -m tools.fused_columns --models <dir> --name <tag>"""
import json, re, collections, cv2, numpy as np
import paths
from columns.dataset import discover
from common import labelme
from tools.evalkit import sheet_counter
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--models", default=None); ap.add_argument("--name", default="fused")
args = ap.parse_args()
sc = sheet_counter(args.models)
split = json.load(open(paths.COLUMNS_SPLIT)); val = set(split["val"]); excl = set(split["excluded"])
rows = []; tot = collections.Counter()
for it in discover():
    m = re.search(r"type\d+", it["stem"]); typ = m.group(0) if m else it["group"]
    img = cv2.imread(it["image"]); H, W = img.shape[:2]
    lab, lut = labelme.column_labels(it["json"], hw=(H, W))
    cols = sc.segment_columns(img, typ if typ.startswith("type") else "type3")
    part = "val" if it["stem"] in val else ("excl" if it["stem"] in excl else "train")
    tot[(part, "frames")] += 1; tot[(part, "cols")] += len(cols)
    for k, c in enumerate(cols):
        ids, cnt = np.unique(lab[c.mask], return_counts=True)
        hit = [(int(i), int(n)) for i, n in zip(ids, cnt) if i > 0 and n >= 0.25 * (lab == i).sum()]
        if len(hit) >= 2:
            tot[(part, "fused")] += 1
            tot[(part, "fused_COMPLETE")] += int(not c.cropped)
            rows.append((part, it["group"], it["stem"], k, [h[0] for h in hit], [round(float(lut[h[0]]), 0) for h in hit], c.cropped))
for r in rows: print(*r)
print(dict(tot))
import os; os.makedirs("runs/_t1/fused", exist_ok=True)
json.dump(rows, open(f"runs/_t1/fused/{args.name}.json", "w"))
