# Where the edge (sheet-counting) work stands — 2026-09-20

Read this first, then `EXPERIMENTS_DATA_SCALING.md` for the study in flight.
**Count is the only measure of success.** Masks exist to make the count legible:
one mask per sheet, lying on that sheet, is enough — a ~1 px offset does not matter,
so instance IoU is NOT a ranking metric.

## Champion and defaults

* Model: **`runs/edges_all3_s7/last.pt`** (types 2+3+10), ONNX `onnx/sheet_edges.onnx`.
  Recipe: `--squash-weight 0 --lighting v2`, seed 7, 8000 steps.
* Column model (separate): `runs/columns_v1/stage2/last.pt`, ONNX `onnx/column_flow.onnx`.
* Counter defaults (`edges/counter.py`): band `frac=0.50`, `thr=0.50`, `MIN_AREA=6`,
  **`REL_FLOOR=0.25`**, `read(..., masks="grow")`. Revert = `REL_FLOOR=0`, `masks="flow"`.
* Input: tight crop (pixels > 0, pad 6) -> centre 74 px of column width (86 px with the
  pads) -> pad to /16. Height is NEVER rescaled (the model is locked to sheet pitch).

## Scores (count exact)

| set | all3_s7 | note |
|---|---|---|
| val crops (type10 15 / type2 17 / type3 18) | **50/50** | type10 + type3 val are SAME-SESSION |
| all 80 labelled type10 crops | 80/80 | the type2/3-only model scored 2/80 |
| older type3 300 (same 2 sessions as training) | 297/300 | inflated: same sessions |
| **honest cross-session type3** (`edges_loso131612_s7`, that session excluded from training) | **173/181 = 95.6 %** | vs 180/181 when trained on it; second seed scored 170/181 WITH it, so the gap ~= seed noise |
| older type10 122 (colleague's cutter) | 67/122 | partly crop bounds, not counting |

Caveats to quote with any number: only type2 has a held-out SESSION; `REL_FLOOR` and
the champion seed were chosen on these same sets; older crops come from the other
cutter and can include material from a neighbouring column.

## THE GROW-FROM-CENTRE READOUT (`edges/grow.py`) — keep this

Replaces the flow readout for MASKS. Why: near the strip's edges neighbouring sheets
compress to a few px, the predicted flow field is smooth, and tail pixels drift into
the neighbour's basin — one sheet's tail takes the neighbour's identity. Measured on
val crops (`tools/measure_leak.py`): flow readout leaked 5-71 px per sheet and split
15-95 % of sheets across two ids; contact pixels 287-1241 per crop. The defect is
produced by the FIELD AT INFERENCE — training on verified-clean labels reproduced it,
so it cannot be relabelled away.

Algorithm, exactly:

1. **Seeds.** `fg = sigmoid(ch0) > 0.5`. Take the centre 50 % of the foreground's
   width (`counter.band`), connected components (8-connectivity), drop blobs under
   `MIN_AREA` and under `REL_FLOOR` x the crop's median blob area. Number them 1..N
   top-down. N IS the count — so the mask count can never disagree with the count.
2. **Grow right, then left.** Step one pixel-column at a time outward from the band.
   In each column take the contiguous vertical runs of foreground. A run inherits the
   id of the run it overlaps in the PREVIOUS column.
3. **Merged runs split at the midpoint.** If a run overlaps several ids, each pixel
   goes to the id whose overlap centre (in y) is nearest — neither sheet swallows the
   other.
4. **No orphans.** A run overlapping no previous id stays unassigned (a detached
   fragment that never reaches the band is not a sheet of this column).

Guarantees: the number of ids is fixed by the seeds, so no sheet can take a
neighbour's identity, none can be created or lost. Worst case is a boundary a few px
off. Cost ~1 ms; no retraining; cannot move the count.

Measured after adopting it (all3_s7, val crops): leak **0.00 px/inst on all three
types**, contact 0-0.7 px/crop, sheets split across ids 0 %, **1929/1930 labelled
sheets get exactly one mask**. Mask IoU vs human labels is only 0.60-0.76 because the
mask can sit ~1 px off the labelled band — irrelevant for counting, see the rule above.
This supersedes `hybrid_readout.py` (leak 2.3 px/inst, two forward passes).

## Tools (all `python -m tools.<name>`)

| tool | does |
|---|---|
| `report_counts` | THE report: val + all labelled + unseen older sets, per run -> csv |
| `measure_leak` | flow vs grow readout: leak px/inst, sheets split, contact, count |
| `mask_shape_stats` | mask thickness/length/IoU vs human labels |
| `show_masks` / `show_edge_leak` | renders: whole column, tip zooms, contact pixels |
| `render_count_failures` | renders crops whose count != expected, culprit marked |
| `count_vs_filename` | counts vs the filename piece count (type2 only; type3=50, type10=24 fixed) |
| `test_relative_blob_floor` | blob-floor sweep on labelled or globbed crops |
| `lighting_stress` | counts under lighting NOT used in training |
| `val_audit_table` | per-type val exact counts straight from each run's audit.json |
| `peek_edge_runs` | latest per-type val numbers for runs still training |
| `compare_edge_runs` | group runs by recipe, seed mean + per-seed |

## Data and splits

`data/edges/<type>/<frame>_column_<k>.png` + `.json` (LabelMe, one `edge` polygon per
sheet). type2 54 crops (50 or 30 sheets, thin-line polygons), type3 72 (50 sheets),
type10 80 (24 sheets, band polygons). Pinned split in `data/edges/splits.json`:
type2 val = sessions 132812 + 134020; type3 val = 2 research-pinned frames; type10 val
= frames 9-14 with 8/15 excluded (single session -> same-session score).
New labels: drop the jsons next to their PNGs and train; `common/splits.py` assigns
new frames (session-level, variant-stratified, never re-rolls).

## Negative results — do not re-run

* Label-guided lighting aug (`--lighting v1`): glare dimmed on unlabelled pixels paints
  the label boundary into the image; counts dropped. An augmentation must never depend
  on the label. The label-free `v2` is the one to use.
* Squash-only or mixed width training: each model is best at the width it trained on;
  centre-only wins for centre-crop deployment.
* Relabelling to fix the mask edge leak (the leak is made at inference).
* Full +-180 deg rotation aug; inference-time upscaling of sheet crops; TTA/SWA.

## FIELD ADAPTATION (`field/`) -- new site, a few labelled frames, gated retraining

THE LOOP (this is the whole customer workflow; a GUI wraps these three calls later):

```
python -m field.adapt --site acme --from <folder>                # stage 1 + 2, then stop
#   stage 1 columns : <folder>/frames/*.jpg+.json (good_column/cropped_column)
#                     -> fine-tune the COLUMN model from the live one, gate, promote
#   stage 2 crops   : run the improved column model over every frame
#                     -> <folder>/edges_to_label/type<N>/<frame>_column_<k>.png
#   ... the labeller draws one "edge" polygon per sheet on those PNGs ...
python -m field.adapt --site acme --from <folder> --stage edges  # stage 3
#   stage 3 edges   : ingest the labels, fine-tune the EDGE model, gate, promote

python -m field.adapt ... --dry-run          # ingest + convention check + baselines only
python -m field.consistency '<glob>'         # label-free: WHICH crops are failing
python -m field.registry                     # what is live
python -m field.registry rollback edges|columns
```

| file | does |
|---|---|
| `field/adapt.py` | the three stages end to end |
| `field/ingest.py` | copy into `data/`, reject duplicate stems, CHECK label convention |
| `field/crops.py` | stage 2: column model -> predicted LabelMe json -> `straighten_frame` |
| `field/gate.py` | edge gate (count on the large labelled sets) |
| `field/gate_columns.py` | column gate (instIoU + countAcc + cls on the pinned val) |
| `field/consistency.py` | label-free error finder on UNLABELLED frames |
| `field/registry.py` | `runs/_active.json`: live models, per-site overrides, rollback |

**Stage 2 reuses the human path exactly**: predicted instances are written as a LabelMe
json and handed to the SAME `common.straighten.straighten_frame`, so nothing about the
geometry is reimplemented. Verified: crops from predicted polygons count 9/9 exact at 24
on type10, same as crops from drawn polygons. They come out ~10 % narrower (75-83 px vs
87-88), i.e. the model's columns are tighter than the human ones.

**Gate tolerances are MEASURED, not chosen.** Edges, from seeds 7 vs 42: `all_type*` 1,
`old_t3` 10, `old_t10` NOT RANKABLE (spread 23/122), pinned val NOT USED (saturated).
Columns, from the COLQ replicates: instIoU 0.02, countAcc 0.05, cls 0.02.

**Why the gate exists** (EXPERIMENTS_DATA_SCALING.md): growing type10 from 0 to 59 crops
cost type2 3 crops of 54 monotonically, and a type2+type3 model scores 1/80 on type10
where type3-only scores 72/80. Adding data DOES break other types and **the pinned val
never shows it**. Outcome order: promote globally -> promote for this site only (same
checkpoint, gate scoped to the types that site runs) -> refuse and change nothing.

Traps already hit and guarded, do not re-introduce:
* an empty site scope protects NOTHING, so a scoped gate passes vacuously -- the site's
  groups come from the new frames' types (`splits.type_of`), and an unknown scope refuses;
* re-ingesting a frame whose stem already exists gives it two entries and two split
  decisions -- ingest skips duplicate stems;
* "already correct" is not improvement: if the live model counts every new crop right,
  retraining can only spend accuracy elsewhere, so it refuses and points at
  `field.consistency` to find the crops that are actually failing.

Trainer flags added for this: edges `--init/--lr/--boost-stems/--boost-factor`; columns
the same plus `--init` meaning STAGE 2 ONLY. A handful of new crops is <3 % of batches
without the boost and is simply never learnt.

### Site bundle (`python -m field.bundle --out <dir> --profile <p>`)

Self-contained folder: code, data, the two live checkpoints, SITE_README.md. All paths
are relative, so it runs anywhere. Measured 2026-09-20:

| profile | MB | drops | can promote globally? |
|---|---|---|---|
| `full` | 400 | nothing | yes -- exact office behaviour |
| `lean` | 283 | `legacy_fg` | yes, but the column mix differs (`--no-legacy` is untested) |
| `site` | **149** | `legacy_fg` + `closeup` | **NO -- site-scoped models only** |

`site` verified standalone: edge training and the ENTIRE edge gate are untouched
(150 train / 50 val crops; gate 80/80, 51/54, 70/72 -- identical to the office, because
`data/edges` is only 30 MB and is not worth trimming). Only the column side shrinks:
135 instance frames instead of 170, 0 fg-only instead of 200, val 40 instead of 50.
The missing val row is `closeup`, so **drift away from the close-up camera domain is
invisible in this profile** -- that is why it may only produce site-scoped models.
`runs/` carries ONLY the checkpoint each track fine-tunes from (stage-1 weights, event
files and previews are ~18 MB of history the site never reads).

The `old_t3` / `old_t10` sets are NOT in any bundle (they live outside the pipeline);
the gate skips them loudly. On this machine they still resolve via the default path, so
a bundle test here does not prove the skip path -- that is covered by
`TECHNOTRON_OLD_CROPS=/nonexistent`.
