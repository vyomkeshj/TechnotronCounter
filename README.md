# Technotron training pipeline

Two models, one method, and the data they learn from.

| pipeline | input | output | used for |
|---|---|---|---|
| **columns** | a rack-camera frame (letterboxed to 256 px) | one mask per column + good/cropped verdict | finding and classifying columns |
| **edges** | one straightened column crop (74 px wide, native height) | one mask per sheet + sheet count | counting sheets in a column |

Both are the same small UNet (base 16, ~1.9M params) predicting a **foreground
logit plus a flow field pointing at each instance's centre**. Instances come out
by advecting pixels along the flow and clustering where they land (Cellpose-style).

Everything here was distilled from the research campaign in
`experiments/column_unet/` (≈60 experiments, logged in the repo `CLAUDE.md`).
This folder keeps only what won, and the reasons it won.

**Going on site to fine-tune on a customer's images? Read `finetune_pipeline.md`.** It is the step-by-step field guide: what to
label, the three-stage loop, how to read the gate, and what each refusal means.

```
training_pipeline/
  README.md            this file
  paths.py             every location (relative to this folder)
  data/
    columns/           closeup/  operational/  quality/type{2,3,10}/  legacy_fg/  splits.json
    edges/             type2/  type3/  type10/  splits.json
  common/              unet, flows (targets + readout), labelme io, metrics, splits,
                       augmentation, tracking, straighten, export_onnx
  columns/             dataset, train, evaluate
  edges/               dataset, train, evaluate, counter, make_crops, predict
  tools/bootstrap_data.py   how data/ was assembled (provenance)
  runs/                training outputs (git-ignored)
```

## Setup

Python 3.13 with a CUDA GPU. Install PyTorch for your CUDA version from pytorch.org,
then `pip install -r requirements.txt`. Run every command **from this folder**.

Tracking: each run writes TensorBoard to `runs/<run>/` (`tensorboard --logdir runs`)
and, when ExternalSoul credentials are configured (`~/.config/esoul/credentials`),
mirrors to the monitors *Technotron Column Flow* / *Technotron Sheet Flow*. Never put a
token in this repo.

## Train

```bash
python -m columns.train --name v2          # ~45 min on an RTX 4090
python -m edges.train --name v2            # ~35 min
python -m columns.train --name t --smoke   # 2-minute plumbing check (same for edges)
```

Each run prints the split it is using, trains to the end, and writes

* `runs/columns_<name>/stage2/last.pt` or `runs/edges_<name>/last.pt` — **the model to ship**
* `audit.json` next to it — held-out scores of that exact checkpoint, per group/type

Evaluate or inspect any checkpoint:

```bash
python -m columns.evaluate runs/columns_v2/stage2/last.pt --render runs/columns_v2/renders
python -m edges.evaluate   runs/edges_v2/last.pt --render runs/edges_v2/renders
python -m edges.predict    runs/edges_v2/last.pt data/edges/type2 runs/edges_v2/type2_preview
python -m common.export_onnx runs/edges_v2/last.pt sheet_model.onnx
```

**Always look at the renders.** Two of the campaign's worst mistakes (a bogus giant
instance, a mislabelled ground truth) were invisible in the metrics and obvious in a
picture.

---

## Adding data

### New column frames

1. Label frames in LabelMe: one **polygon per column**, label `good_column` (fully
   visible) or `cropped_column` (cut by the frame edge, the rack aperture, or hidden
   behind something). Label every column, including tiny slivers — partial edge
   columns are real ground truth.
2. Put `frame.jpg` + `frame.json` into `data/columns/quality/<type>/` (rack camera,
   file names like `20260626_131412_type3_-2_60pieces_001`) or a new folder under
   `data/columns/` for a new source.
3. Train. The first lines of the run print how every new frame was split (see
   *Validation splits*). `data/columns/splits.json` is updated — commit it.

### New edge (sheet) labels

1. Make crops from column-labelled frames:
   ```bash
   python -m edges.make_crops data/columns/quality/type5
   ```
   Every `good_column` becomes `data/edges/<type>/<frame>_column_<k>.png`: straightened,
   native resolution, full width, 2050 px tall. Near-duplicate frames and frames that
   already have crops are skipped (logged in `data/edges/make_crops_log.csv`).
2. See how the current model does before spending labelling effort:
   `python -m edges.predict runs/edges_v2/last.pt data/edges/type5 runs/type5_preview`
3. Send the PNGs to the labeller. Convention (must match the existing type3 labels):
   * one polygon labelled **`edge`** per sheet, covering the sheet's band and leaving
     the dark groove between sheets unlabelled;
   * optional **`background`** polygons for grooves/shadows or slivers of a
     neighbouring column inside the crop;
   * every sheet in the column, top to bottom, including partially visible ones.
   Give them `data/edges/type3/20260626_131412_type3_-2_60pieces_001_column_0.png`
   + `.json` as the reference example.
4. Put the returned `<same name>.json` files next to their PNGs and train. Unlabelled
   PNGs are ignored.

Label **native crops only** (what `make_crops` produces). The model input — 74 px wide,
centre crop or squash — is derived at load time, so labels never need redoing when that
choice changes.

### Validation splits

`splits.json` pins which **source frames** are validation. Existing assignments are
never changed; new frames are assigned automatically (`common/splits.py`):

* split by **frame**, never by column/crop (one frame's columns share lighting — a
  crop-level split once leaked half an audit set);
* the rack camera is fixed, so neighbouring frames are near-identical: whole
  **capture sessions** (`YYYYMMDD_HHMMSS_typeN`) go to val (~1 in 5, at least one per
  new type with ≥2 sessions);
* a type with a **single session** gets a contiguous block of frames as val with the
  frames either side excluded — its score is *same-session* and must be reported as such;
* a new frame of an existing session inherits the split of its nearest frame.

To hold out something specific, edit `splits.json` by hand before training.

---

## Current reference results

Measured with this pipeline on the data in `data/` (val frames never trained on).

**columns v1** (`runs/columns_v1/stage2/last.pt`, 2026-09-16)

| val group | frames | instance IoU | exact count | count MAE | good/cropped acc |
|---|---|---|---|---|---|
| type2 | 20 | 0.893 | 85 % | 0.20 | 0.984 |
| type3 | 14 | 0.915 | 79 % | 0.21 | 0.930 |
| type10 (same-session) | 6 | 0.897 | 83 % | 0.17 | 0.919 |
| **deploy (types 2/3/10 pooled)** | 40 | **0.901** | **82 %** | **0.20** | **0.940** |
| closeup | 10 | 0.828 | 60 % | 0.80 | 0.843 |

Reproduces the research runs on the same split (0.903 / 0.896 instIoU, two seeds). The
previously deployed model scored 0.850 / 67.5 % / 0.43 / 0.946 on the same 40 frames.

**edges all3** (`runs/edges_all3_s7/last.pt`, 2026-09-20) — one model for types 2, 3 and 10

| set | exact counts |
|---|---|
| **held-out val crops (type10 15, type2 17, type3 18)** | **50 / 50** |
| all 80 labelled type10 crops | 80 / 80 |
| unseen older type3 crops | 296 / 300 |
| unseen older type10 crops (colleague's cutter) | 67 / 122 |

Recipe: centre-crop width (`--squash-weight 0`) + label-free lighting aug
(`--lighting v2`). Counting uses the centre band with the relative blob floor, and
masks come from `edges/grow.py` (grown outward from the counted blobs). Masks:
1929/1930 labelled sheets get exactly one mask, zero identity leak — the previous flow
readout leaked 5-71 px per sheet. A mask may sit ~1 px off the labelled band; that is
irrelevant to counting and is the reason instance IoU is only 0.60-0.76.
Second seed (`edges_all3_s42`) also scores 50/50 val, 80/80 type10, 286/300 older type3.

Single runs carry noise (±0.05 instIoU on these val sizes; one replicate of the edge
recipe went from 18/18 to 15/18 exact with the same seed). Compare models with
replicates or with several late checkpoints, never with one peak.

## Why the recipe is what it is

Kept (each measured on held-out data):

| decision | evidence |
|---|---|
| flow representation, iterative readout | instIoU 0.82 vs 0.43 for one-shot centre offsets; count error −79 % vs binary masks |
| two-stage readout guard (≥3 hits, re-split at 25 % of peak density) | removed "tiny bridge" fusions between neighbouring columns |
| columns: 12000 steps + 4000 class stage, ship LAST | 3000 steps under-trained; 4-fold CV curve flat ≥8000, so selection only picks noise |
| columns: legacy blob frames, foreground-only | helps via the fg channel; the valid flag keeps their fused masks out of the flow loss |
| columns: zoom jitter + strong photometric aug | largest single aug gain; light aug overfits |
| edges: anisotropic flow target | sheets are 3.2× wider than tall; count MAE 2.26 → 1.90 |
| edges: pitch jitter sy 0.5–3.5 | without it the model is pitch-locked (2× pitch → MAE 5.2) |
| edges: fragment aug | fixed a ±1 count flutter on relocated column pieces |
| edges: count = centre band + connected components | 18/18 exact on human labels, 2× wider threshold margin, robust to blur/gain/noise; the flow count is not |
| edges: never rescale crop height at inference | the model is locked to sheet pitch, which already matches across types |

Tried and rejected — do not re-run without new evidence: full ±180° rotation aug
(gravity prior is real), 384 px input, base 24, hard-example rebalancing, boundary
contrastive flow loss, filtering border fragments, TTA and SWA/weight averaging
(attractor fields do not average), inference-time upscaling of sheet crops, mixing the
old `Data_1` instance labels (hurts at every dose), relabelling to fix sheet "edge leak"
(the leak comes from the field at inference).

## Known limits

* **Edges** are validated on type3 only until type2/type10 labels arrive. Before them,
  the type3-trained model failed on the type2 `-1_30pieces` sessions (sheets at ~30°
  in the centre of the S-curve; it saw ±6°) and over-split a few type10 crops.
* **Columns**: type10 has a single capture session, so its val score is same-session.
  Remaining column count misses are sub-25-px slivers at the aperture edge — below the
  readout's minimum instance size at 256 px.
* Blurred frames (e.g. type3 `131412_013`) are hard for both models; consider dropping
  them from labelling.
