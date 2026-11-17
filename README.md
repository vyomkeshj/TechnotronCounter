# Sheet counter: column segmentation, straightening, edge segmentation, counting

Counts the sheets in every complete column of a photo of stacked sheets, taken by the fixed rack
camera (1920 x 1080). Built and validated for **column types 2, 3 and 10**.

```
photo ──► 1. column segmenter ──► 2. straightener ──► 3. edge segmenter ──► 4. counter
          (every column +          (each complete      (every sheet in      (sheets crossing the
           complete/cropped)        column -> upright   the strip; two       middle of the strip)
                                    strip)              models averaged)          + self-check
```

## Results (reproduce with `python -m tools.evaluate`)

40 held-out frames from capture sessions that **neither model ever trained on**, 169 labelled
complete columns:

| type | complete columns counted exactly | expected count |
|---|---|---|
| type3 | **126 / 126** | 50 |
| type10 | **23 / 23** | 24 |
| type2 | **20 / 20** | 50 or 30 (the `_NNpieces_` number in the file name) |

Score 1.000 single frame and with 5-frame voting (column model `columns_t8_s7`, 2026-11-02, which also
segments type8 columns; sheet models `edges_r3_a|b`, 2026-10-23 round 3). The previous column model
(`columns_final`) scored 0.994 single / 1.000 voted.

**Self-check:** every counted column is also counted by each edge model alone and from a
column outline one pixel wider; if any count differs the column is flagged **uncertain** (take
another photo). On the held-out frames, every column that is NOT flagged is correct:
type2 18/18, type3 126/126, type10 23/23.

Other checks (all reproducible, see "Reproduce the numbers"):
- **Unseen type2 frames** (108 unlabelled frames, `data/unlabelled`): ~70 % exact per frame,
  ~72 % with 5-frame voting (the previous column model: ~73 % / ~75 %; the type8 data costs ~3 points here). Weak spot: the steep S-curve session 133033 (~46 %); about half of its
  errors are column outlines (needs labelled S-curve column frames), half sheets off by one. Type2 on brand-new
  frames is the weak spot, mostly its steep S-curve variant (session 133033).
- **Broken sheet masks** (`tools/fragment_eval.py`, `tools/fragment_stress.py`; one labelled sheet
  predicted as several pieces): 1.3 % of 1930 held-out labelled sheets; with a surface-texture
  overlay on the photo 14.7 % (2026-09-30 pair: 1.6 % and 30 %).
- **Robustness** (held-out frames, perturbed): brightness x0.6/x1.4, noise, JPEG q30, slight blur,
  colour cast: 0.97-0.99 overall. Low contrast 0.970, 4 deg rotation 0.976, zoom-out 5 % 0.964,
  10 % 0.911. Keep the camera where it is.

## Quick start: inference

```bash
pip install onnxruntime opencv-python numpy          # inference needs nothing else (no PyTorch)
python inference/predict.py samples/input --out my_output --save-crops
```

Compare `my_output/` with `samples/output/` (same files, same counts). Options:
- `--type type2|type3|type10` (default: read `typeN` from each file name; otherwise type3 settings)
- `--vote 5` for a run of consecutive frames from the fixed camera: each column's most common
  count over 5 frames (removes one-frame glitches)
- `--save-crops` also writes every straightened column with its sheet masks

Outputs: `results.csv` (one row per column: class, count, uncertain + why, voted count, centre,
P(cropped)), `<frame>_overlay.jpg` (green = complete and counted, yellow `N?` = uncertain, red =
cropped), `<frame>_col<k>.png` (the strip, then its sheet masks, every 5th sheet numbered).

## The pipeline, stage by stage

All of it is in `inference/sheetcount/pipeline.py` (ONNX + OpenCV + NumPy). Each stage is callable:

```python
import cv2, sys
sys.path.insert(0, "inference")
from sheetcount.pipeline import SheetCounter, SHEET_THR

sc = SheetCounter("inference/models")
frame = cv2.imread("samples/input/20260626_131412_type3_-2_60pieces_004.jpg")

# 1. column masks: one Column per column, left to right
cols = sc.segment_columns(frame, "type3")            # .mask (bool, frame size), .polygon, .cropped
good = [c for c in cols if not c.cropped]

# 2. straighten: one upright strip per complete column
strips = sc.straighten_columns(frame, [c.polygon for c in good])

# 3. edge masks: sheet foreground of one strip (the two edge models averaged)
x = sc.edge_input(strips[0])                         # tight crop, centre 74 px of the width
fg = sc.sheet_probability(x) > SHEET_THR

# 4. count, and one mask per counted sheet
count, seeds, band = sc.count_sheets(fg)
sheet_masks = sc.sheet_masks(fg, seeds, band)        # int labels 1..count

# or everything at once, with the self-check:
for c in sc.process(frame, "type3"):
    print(c.cropped, c.count, c.extra.get("uncertain"), c.extra.get("why"))
```

**1. Column segmenter** (`models/column_flow.onnx`). The frame is letterboxed to 256 x 256 and a
UNet predicts 4 channels: foreground, a flow field pointing to each column's centre (dy, dx), and
P(cropped). Pixels are moved along the flow for 40 steps; pixels that land together form one
column (instances smaller than 25 px at 256 are dropped). Complete vs cropped:
- type2, type3: the mean P(cropped) over the column > 0.5 means cropped.
- type10: geometry instead: a column that reaches the top or bottom of the whole stack (within
  6 px) is cropped. This beats the model's own call on unseen type10 frames.
"Complete" means the column's whole sheet stack is visible end to end; touching the image edge
with a corner does not make a column cropped.

**2. Straightener** (`sheetcount/straighten.py`, the TosaI66 BAD_ELIMINATE geometry). One
rotation angle per frame from the complete columns' minimum-area rectangles; each column is cut
out at full height, everything outside its outline blacked out, rotated 180 deg and placed on a
2050-px-high canvas at native width (never resampled: sheet pitch stays ~10-14 px).

**3. Edge segmenter** (`models/sheet_edges_a.onnx`, `sheet_edges_b.onnx`). The strip is cropped
tight (6 px pad) and to the centre 74 px of its width, then padded to a multiple of 16. Two UNets
(same recipe, different seeds; `runs/edges_r3_a`, `edges_r3_b`) each predict a sheet-foreground probability; they are averaged
(this removes seed-to-seed noise) and thresholded at 0.5.

**4. Counter.** Take the centre 50 % of the strip's width; count connected components of the
foreground there. Ignore blobs under 6 px or under 25 % of the strip's median blob; the first and
last blob must reach 50 % of the median (drops a sliver of a neighbouring column that the outline
took in). Sheet masks are grown outward from those counted blobs one pixel column at a time, so
there is exactly one mask per counted sheet and no sheet can take a neighbour's pixels.

**Self-check.** Each complete column is re-counted by each edge model alone and from its outline
grown by 1 px. Any disagreement -> `uncertain` with the reason. Those are the columns whose count
depends on the last pixel at a column end, which is where the remaining errors come from.

**Voting** (`vote()`, `predict.py --vote N`). For consecutive frames of the fixed camera: match
each column to the nearest column (<= 80 px) in the N frames around it; report the most common count.

## Folder layout

```
inference/                  the deliverable: predict.py, sheetcount/ (pipeline, straighten, flows),
                            models/ (the three ONNX models)
samples/input, output       four held-out frames and what inference/predict.py writes for them
tools/                      evaluate.py (THE score), unseen_check.py, stress.py, report.py,
                            pseudo_edges.py (edge self-training data), evalkit.py (shared helpers)
columns/                    column model: dataset, training, in-training evaluation
edges/                      edge model: dataset, training, counter, grow readout, make_crops.py
common/                     UNet, flow targets/readout, image I/O, LabelMe I/O, splits, straightener,
                            ONNX export, experiment tracking (TensorBoard)
data/columns/               labelled frames (LabelMe good_column / cropped_column polygons):
                            quality/ (types 2/3/10, the operational camera), operational/,
                            closeup/, legacy_fg/ (foreground-only), splits.json,
                            relabel_type3_edge_columns.json
data/edges/                 human-labelled straightened crops (LabelMe "edge" polygon per sheet),
                            splits.json
data/edges_pseudo/          self-labelled crops (made by tools/pseudo_edges.py)
data/oct26/                 October 2026 field labels, kept separate and grouped as delivered (not by
                            type): frames/<folder>/ (15 frames, columns + every sheet edge on the
                            full frame) and edges/<folder>/ (16 straightened strips + labels, made
                            by edges/from_frames.py). See data/oct26/README.md
data/unlabelled/type2/      108 frames with no label of any kind (for tools/unseen_check.py)
runs/                       the checkpoints behind the shipped models (see below)
paths.py                    every location, relative to this folder
```

## Reproduce the numbers

Run everything from this folder with Python 3.10+ (developed on 3.13, Windows, RTX 4090).

```bash
pip install -r requirements.txt           # after installing torch for your CUDA from pytorch.org
python -m tools.evaluate --vote 5         # the table above; writes runs/_evaluate/columns.csv
python -m tools.unseen_check              # unseen type2 frames (~70 % / ~72 % voted)
python -m tools.fragment_eval runs/edges_r3_a/last.pt runs/edges_r3_b/last.pt   # broken sheet masks
python -m tools.fragment_stress runs/edges_r3_a/last.pt runs/edges_r3_b/last.pt --sets old_val
python -m tools.column_field_eval         # column stage on the October field frames
python -m tools.stress                    # robustness table (slow); --only "exposure_-2EV,zoom_0.75,rotate_+10,pitch_10%,..."
python -m tools.negative_check            # false columns on column-free images (shipped: 82.9 %)
python -m tools.type1_check               # type1 June frames: counts vs file name (mostly IN TRAINING since 2026-11-15)
python -m tools.column_heldout_check      # column stage per type on the held-out frames of ALL labelled types
                                          #   (incl. types 1/4/7); --turn 90|180|270 for a turned camera
bash tools/candidate_check.sh <name> <column .pt> [<edge_a .pt> <edge_b .pt>]   # ALL of the above for one candidate
python -m tools.report                    # one render per held-out frame -> runs/_report/
```

Expected `tools.evaluate` output with the shipped models (2026-11-05: aligned + negatives column model, four
averaged sheet models, `MASK_THR = 0.30`; the previous models are in git history, commit fcb5899):

```
type    complete  exact  wrong  cropped  missed  false   score  unflagged correct   voted
type2         20     20      0        0       0      0   1.000           16/16      1.000
type3        126    126      0        0       0      0   1.000          125/125     1.000
type10        23     22      1        0       0      0   0.957           21/21      1.000
all          169    168      1        0       0      0   0.994          162/162     1.000
```
`tools.unseen_check`: type2 84.2 % / 89.1 % voted. `tools.negative_check`: 17.3 % of column-free images.
(Previous models: 1.000 / 1.000, unseen 70 % / 72 %, negatives 82.9 %.)

## Reproduce the models

Seven training runs (~2.5 h on one RTX 4090), then an ONNX export. The held-out test sessions
(type2 132812 + 134020, type3 131412, type10 frames 9-14) are never trained on: the column split
holds them out, and `--exclude-session 131412` keeps type3's test session out of the edge model.

```bash
# column model: base recipe, then fine-tune with wider zoom-out augmentation
python -m columns.train --name base  --class-override data/columns/relabel_type3_edge_columns.json
python -m columns.train --name final --class-override data/columns/relabel_type3_edge_columns.json \
    --init runs/columns_base/stage2/last.pt --lr 5e-5 --scale-lo 0.65
# 2026-11-02 (SHIPPED column model): fine-tune with the October labels (human + model-free
# auto-labels) and the new type8 labels; type8 frames 018/021 held out for testing
python -m columns.train --name columns_t8_s7 --seed 7 --class-override data/columns/relabel_type3_edge_columns.json \
    --init runs/columns_final/stage2/last.pt --lr 5e-5 --scale-lo 0.65 \
    --extra-data data/oct26/frames:4 --extra-data data/oct26_autolabel/frames:4 \
    --extra-data data/type8_autolabel/frames:4 --extra-holdout 39pieces_018 --extra-holdout 39pieces_021
# 2026-11-04 CANDIDATE (not shipped): as columns_t8_s7 plus column-free images taught as "no column",
# 8000 steps. Needs the negatives (see Data). False columns 82.9 % -> 20.8 %, old test 0.982 / 0.982.
python -m columns.train --name neg_s7 --seed 7 --steps2 8000 --class-override data/columns/relabel_type3_edge_columns.json \
    --init runs/columns_final/stage2/last.pt --lr 5e-5 --scale-lo 0.65 \
    --extra-data data/oct26/frames:4 --extra-data data/oct26_autolabel/frames:4 \
    --extra-data data/type8_autolabel/frames:4 --extra-holdout 39pieces_018 --extra-holdout 39pieces_021 \
    --negatives ../datasets/negatives_web/train --neg-frac 0.18 --bg-frac 0.12

# 2026-11-15 PRODUCTION CANDIDATE (types 1/4/7 added): as alneg_s7, seed 42, AI-agent labels at 1x weight
python -m columns.train --name rep1_s42 --seed 42 --init runs/columns_columns_t8_s7/stage2/last.pt --lr 5e-5 --steps2 12000 \
    --class-override data/columns/relabel_type3_edge_columns.json --scale-lo 0.65 \
    --extra-data data/oct26/frames:4 --extra-data data/oct26_autolabel/frames:4 \
    --extra-data data/type8_autolabel/frames:4 --extra-holdout 39pieces_018 --extra-holdout 39pieces_021 \
    --label-resize exact --negatives ../datasets/negatives_web/train --neg-frac 0.25 --bg-frac 0.15 --agent-rep 1

# edge models: a seed model, self-labelled crops, then two self-trained models
python -m edges.train --name seed --seed 7 --exclude-session 131412
python -m tools.pseudo_edges --model edges_seed --out data/edges_pseudo
python -m edges.train --name final_a --seed 42 --exclude-session 131412 --extra-data data/edges_pseudo
python -m edges.train --name final_b --seed 7  --exclude-session 131412 --extra-data data/edges_pseudo

# 2026-10-23: fine-tune each with the October field labels (x4), a connectivity loss and texture aug
python -m edges.train --name oct_a --seed 42 --init runs/edges_final_a/last.pt --lr 2e-4 --steps 4000 \
    --exclude-session 131412 --extra-data data/edges_pseudo --extra-data data/oct26/edges:4 \
    --cldice 0.5 --texture 0.3 --texture-max 0.25
python -m edges.train --name oct_b --seed 7  --init runs/edges_final_b/last.pt --lr 2e-4 --steps 4000 \
    --exclude-session 131412 --extra-data data/edges_pseudo --extra-data data/oct26/edges:4 \
    --cldice 0.5 --texture 0.3 --texture-max 0.25

# 2026-10-23 round 3 (SHIPPED): as oct_*, but type2 strips exempt from clDice and texture, lr 1e-4
python -m edges.train --name r3_a --seed 42 --init runs/edges_final_a/last.pt --lr 1e-4 --steps 4000 \
    --exclude-session 131412 --extra-data data/edges_pseudo --extra-data data/oct26/edges:4 \
    --cldice 0.5 --texture 0.3 --texture-max 0.25 --texture-skip type2 --cldice-skip-exempt
python -m edges.train --name r3_b --seed 7  --init runs/edges_final_b/last.pt --lr 1e-4 --steps 4000 \
    --exclude-session 131412 --extra-data data/edges_pseudo --extra-data data/oct26/edges:4 \
    --cldice 0.5 --texture 0.3 --texture-max 0.25 --texture-skip type2 --cldice-skip-exempt

# 2026-11-05 (SHIPPED column model): fine-tune the t8 model with centre-aligned labels and column-free negatives
# (needs ../datasets/negatives_web, see Data)
python -m columns.train --name alneg_s7 --seed 7 --init runs/columns_columns_t8_s7/stage2/last.pt --lr 5e-5 --steps2 12000 \
    --class-override data/columns/relabel_type3_edge_columns.json --scale-lo 0.65 \
    --extra-data data/oct26/frames:4 --extra-data data/oct26_autolabel/frames:4 \
    --extra-data data/type8_autolabel/frames:4 --extra-holdout 39pieces_018 --extra-holdout 39pieces_021 \
    --label-resize exact --negatives ../datasets/negatives_web/train --neg-frac 0.25 --bg-frac 0.15
# 2026-11-05 (SHIPPED second sheet pair, averaged with r3_a/b): rotation, farther camera, exposure, anti-merge loss
python -m edges.train --name rfg_a --seed 42 --init runs/edges_final_a/last.pt --lr 1e-4 --steps 8000 \
    --exclude-session 131412 --extra-data data/edges_pseudo --extra-data data/oct26/edges:4 \
    --cldice 0.5 --texture 0.3 --texture-max 0.25 --texture-skip type2 --cldice-skip-exempt \
    --rot-max 12 --far-p 0.3 --exposure 0.3 --groove 1.0
python -m edges.train --name rfg_b --seed 7 --init runs/edges_final_b/last.pt --lr 1e-4 --steps 8000 \
    --exclude-session 131412 --extra-data data/edges_pseudo --extra-data data/oct26/edges:4 \
    --cldice 0.5 --texture 0.3 --texture-max 0.25 --texture-skip type2 --cldice-skip-exempt \
    --rot-max 12 --far-p 0.3 --exposure 0.3 --groove 1.0
# 2026-11-15 EDGE CANDIDATE (not installed): three base-32 sheet models from scratch, + types 1/4/7 strips + clutter;
# seeds 42 / 7 / 3 = f32c_a / f32c_b / f32c_c, averaged (runs/_candidates/edges_f32c_abc)
python -m edges.train --name f32c_a --seed 42 --lr 1e-3 --steps 12000 --base 32 --clutter 0.5 \
    --exclude-session 131412 --extra-data data/edges_pseudo --extra-data data/oct26/edges:4 \
    --cldice 0.5 --texture 0.3 --texture-max 0.25 --texture-skip type2 --cldice-skip-exempt \
    --rot-max 12 --far-p 0.3 --exposure 0.3 --groove 1.0

# export to the inference package (the pipeline averages every models/sheet_edges*.onnx), then score
python -m common.export_onnx runs/columns_alneg_s7/stage2/last.pt inference/models/column_flow.onnx
python -m common.export_onnx runs/edges_r3_a/last.pt inference/models/sheet_edges_a.onnx
python -m common.export_onnx runs/edges_r3_b/last.pt inference/models/sheet_edges_b.onnx
python -m common.export_onnx runs/edges_rfg_a/last.pt inference/models/sheet_edges_c.onnx
python -m common.export_onnx runs/edges_rfg_b/last.pt inference/models/sheet_edges_d.onnx
python -m tools.evaluate
```

The checkpoints this produced are in `runs/` (`columns_base`, `columns_final`, `columns_columns_t8_s7`,
`edges_seed`, `edges_final_a`, `edges_final_b`, `edges_oct_a`, `edges_oct_b`, `edges_r3_a`, `edges_r3_b`), so any step can start from the shipped
result. `tools/pseudo_edges.py` regenerates `data/edges_pseudo/` byte-identically from
`runs/edges_seed`. GPU training is not bit-deterministic (cuDNN, data-loader order): a retrained
model can differ from the shipped one by a column or two on the held-out set; judge with
`tools.evaluate`, and when comparing recipes train two seeds.

What the recipe steps are for:
- **`--class-override data/columns/relabel_type3_edge_columns.json`**: the two type3 capture
  sessions labelled one column position differently (the training session called every column
  touching the image edge "cropped", even complete ones; the test session called them
  "complete", and they do count exactly 50). This file lists 19 such columns in the training
  session; they are trained as complete. Without it the model calls 14 test columns cropped.
- **`--scale-lo 0.65`**: wider zoom-out augmentation (default 0.8). Makes the column outlines
  robust to the camera being a little further away (+1 to +3 points on every perturbation).
- **Edge self-training** (`tools/pseudo_edges.py`): the seed model labels every complete column of
  the column-training frames; a crop is kept only if its count equals the product count, which
  the model never sees. Type3 edge training data goes from 9 hand-labelled crops to 170.
- **Two edge models averaged**: removes seed-to-seed noise in the count.
- **October fine-tune** (field report: textured sheets gave broken, jagged masks): `--cldice 0.5`
  (soft-clDice connectivity loss; 1.0 was also good, 2.0 fattens sheets until neighbours merge),
  `--texture 0.3 --texture-max 0.25` (label-free surface grain/streaks; stronger texture halves the
  broken masks again but costs ~5 points on unseen type2), fine-tuning from the previous pair (from
  scratch gave a confident type10 +1). `data/oct26/edges:4` repeats the 16 new strips 4x.
- **Round 3** (`--texture-skip type2 --cldice-skip-exempt`, lr 1e-4): type2's thin grooves suffer from
  both extras (sheets widened by clDice / faint grooves washed out by grain merge two thin sheets).
  Exempting type2 fixed the 30-piece undercounts (session 133746 12 -> 15-16 of 18, reproduced with
  a second seed pair); the half learning rate keeps the old test at 0.994 (lr 2e-4: 0.982-0.988).

## Candidate log (column model, newest first)

Every candidate is scored with `bash tools/candidate_check.sh <name> runs/columns_<name>/stage2/last.pt`
(shipped sheet models) and compared with a SAME-RECIPE control, never only with the shipped model
(retrain noise alone costs ~1 borderline column end). Common flags of every run below:
`--class-override data/columns/relabel_type3_edge_columns.json --scale-lo 0.65 --extra-data data/oct26/frames:4
--extra-data data/oct26_autolabel/frames:4 --extra-data data/type8_autolabel/frames:4 --extra-holdout 39pieces_018
--extra-holdout 39pieces_021`; NEG = `--negatives ../datasets/negatives_web/train --neg-frac 0.18 --bg-frac 0.12`.

| run | recipe (+ common flags) | false cols | old test single/voted | unseen t2 complete, voted | Oct sharp MAE | notes |
|---|---|---|---|---|---|---|
| **`alneg_s7`** (best, 2026-11-05) | `--seed 7 --init runs/columns_columns_t8_s7/stage2/last.pt --lr 5e-5 --steps2 12000 --label-resize exact --neg-frac 0.25 --bg-frac 0.15` + NEG paths; sheet models `edges_rot12_a/b` | **17.3 %** | **0.994 / 1.000** | 101, **81.2 %** | **0.51** | October exact 42 %; mask offset 1.0-1.2x (shipped 3-14x); weaker: type10 at -2.5 EV 12/23, pitch -10 % 0.888 |
| `seam_s42` / `seam_s7` | rep1 recipe + `--seam-w 5 --paste 0.3` (seam-weighted fg/flow loss between touching columns + pasted cropped pieces beside / past complete columns; controls rep1_s42 / rep1_s7) | 24.6 / 23.2 % (ctl 18.8 / 21.2) | 0.988 / 0.994, 0.982 / 0.994 (ctl 0.994 / 1.000, 0.982 / 1.000) | 96, 89.6 % / 85.4 % (ctl 104, 81.7 % / 100, 83.0 %) | 0.86 / 0.86 | NEGATIVE for fusion: `tools.fused_columns` held-out fused 18 / 19 (ctl 19 / 20), fused-and-called-complete 6 / 6 (ctl 6 / 5) -- unchanged; only on TRAINING frames 19 / 19 (ctl 34 / 36). Costs: more false columns, voted test 1.000 -> 0.994 both seeds, rotate -20 type10 10-13/23 (ctl 14). The held-out set holds only 2 harmful fusions (type1 142016 strap; seam_s42 fixes _020, not _019), so it barely measures the problem |
| `hr512_s42` / `hr512_s7` (control `hr256ctl_s42`) | rep1_s42 / rep1_s7 continued 8000 steps at lr 1e-4, batch 16, COLUMN INPUT 512 px (`COLUMN_SIZE=512` to train, `SHEETCOUNT_COLUMN_SIZE=512` to run; decode cap 1280, flow readout scaled); control = the same continuation at 256 | 19.6 / 19.4 % (ctl 15.7) | 0.965 / 0.982, 0.936 / 0.971 (ctl 0.971 / 0.982; s7: type2 18/20 + 2 false) | 88 cols 85.2 / 89.8 %, 93 cols 86.0 / 91.4 % (ctl 106 cols 78.3 / 81.1 %: FEWER type2 columns called complete, exact 75 / 80 vs 83) | October 76 / 69 complete found, sharp exact 71 / 78 % (ctl 66, 72 %) | FUSION FIXED on both seeds: `tools.seam_check` merged pairs 7 / 8 of 501 (ctl 20), pairs losing a complete column 1 / 2 (ctl 9), type3 gap painted 11.6 / 12.5 % (ctl 21.6); held-out outline IoU 0.912 / 0.911 (ctl 0.897). Zoom 0.6 stress 0.917 / 0.893 (ctl 0.290), exposure better, type1 voted 92.4 / 93.9 % (ctl 81.7). WORSE: rotate -20 0.836 / 0.868 (ctl 0.934), false columns +4 pts, fewer type2 complete on unseen frames. CPU 65 ms vs 19 ms per frame. Needs the straightener fix (below) |
| **`rep1_s42` (PRODUCTION CANDIDATE 2026-11-15)** | alneg recipe + `--agent-rep 1` (data/columns/quality_agent: types 1/4/7, shrunk 3 px); `delivery/column_sheet_counter_2026-11-15` | 18.8 % | 0.994 / 1.000, 0 unflagged | 104, 81.7 % | 0.86 | held out type1 94.5 % / type4 95.7 % / type7 89.9 % correct (shipped 81.7 / 94.3 / 91.1); side view 0 complete; seed 7: old test 0.982 / 1.000 |
| `newt2_s7/s42` | as rep1 but agent frames x4 | 19.6 / 17.1 % | 0.947 / 0.941 | 88.9 / 90.2 % | | type10 15/23: the model extends type10 outlines into the column above (agent convention at band boundaries) |
| `newt_s7/s42` | as newt2, agent outlines NOT shrunk | 16.7 / 23.0 % | 0.953 / 0.964 | 85.4 / 87.8 % | | agent outlines are ~3 px loose (6 % area vs human) |
| `turn_s7/s42` | rep1 + `--turn 0.5` (quarter turns) | 26.2 / 25.0 % | 0.947 / 0.953 | 90.2 / 89.2 % | 0.46 / 0.55 | column stage at 90/180/270 deg: 91 / 96 / 93 % correct (rep1: 89 / 93 / 81 %), but upright type10 16/23 and type1 82-86 % |
| `alneg_s42` | as alneg_s7, `--seed 42` | 18.1 % | 0.988 / 1.000 | 102, 82.4 % | 0.47 | replicates alneg_s7 (two-seed rule met); same type10 014 +1 |
| **CANDIDATE (final 2026-11-05)** `alneg_s7` + 4 sheet models averaged (`edges_r3_a/b` + `edges_rfg_a/b`) | `delivery/_cand_alneg_s7_2026-11-05`; score with `bash tools/candidate_check.sh <name> keep` on a folder holding the 5 ONNX files | 17.3 % | **0.994 / 1.000**, 0 unflagged wrong | 101, **89.1 %** | **0.49** | October exact 46.9 %; pitch -10 % 0.970; zoom 0.75 0.870; S-curve 133746_032 30; -2.5 EV 0.935 (shipped 0.953); sheet masks grown at `MASK_THR = 0.30` in the package (counts unchanged, whole masks 39 -> 52 %) |
| `alneg_s7` + shipped sheet models | packaged as `delivery/_cand_alneg_s7_2026-11-05` | 17.3 % | 0.988 / 0.994 | 101, **83.2 %** | 0.51 | 1 unflagged miss (type10 140757_014, neighbour sliver); S-curve 133746_032 28 |
| `alnegocc_s7` | as alneg_s7 + `--occlude 0.3` | 30.2 % | 0.941 / 0.959 | 77, 80.5 % | 0.88 | NEGATIVE: over-calls cropped (type2 4 test columns, side view 0 complete) |
| **`align_s7`** | `--seed 7 --init runs/columns_columns_t8_s7/stage2/last.pt --lr 5e-5 --steps2 8000 --label-resize exact` | 81.7 % | 0.988 / 0.994 | 106, 78.3 % | 0.95 | labels centre-aligned (half-pixel fix); October exact 36.6 % (type3 sharp 52 %); type8 IoU 0.717; S-curve 133746_032 merges 2 sheets (strip angle) |
| `t8ft_s7` (control for align) | as align_s7 without `--label-resize exact` | 80.2 % | 0.982 / 0.988 | 106, 69.8 % | 1.38 | October exact 22.4 % |
| shipped `columns_t8_s7` | chain: base -> final -> t8 fine-tune (above) | 82.9 % | 1.000 / 1.000 | 101, 72.3 % | 1.19 | |
| `ctrl_s42` | `--seed 42 --init runs/columns_final/stage2/last.pt --lr 5e-5` (= shipped recipe, other seed) | 84.1 % | 0.988 / 0.994 | -, 71.8 % | 1.51 | retrain noise: type2 132812_027, type10 014 |
| **`neg_s7` (candidate)** | `--seed 7 --init runs/columns_final/stage2/last.pt --lr 5e-5 --steps2 8000` NEG | 20.8 % | 0.982 / 0.982 | 102, 71.6 % | 0.89 | `delivery/_cand_neg_s7_2026-11-04`; 1 unflagged miss (132812_028) |
| `neg_s42` | as neg_s7, `--seed 42` | 23.2 % | 0.971 / 0.976 | -, 69.9 % | 1.00 | |
| `rob_s7` / `rob_s42` | as ctrl, NEG at 0.15/0.10, `--exposure 0.4 --wide-p 0.35` (4000 steps) | 40.9 / 36.1 % | 0.976-0.982 | -, 71.6-74.0 % | 1.19 / 1.32 | little robustness gain |
| `scratch_ctrl_s7` | `--seed 7`, NO --init (stage 1 12k + stage 2 4k) | 87.5 % | 0.988 / 0.988 | 88, 69.3 % | 2.84 | one run < the 3-step chain |
| `scratch_rob_s7` | as scratch_ctrl + NEG `--exposure 0.4 --wide-p 0.35` | **7.9 %** | 0.988 / 0.994 | 81, 56.8 % | 3.40 | roll/tilt/zoom 0.75 +0.04..0.09 vs its control; type2 called cropped more; type1 56/72 exact |
| `ftA_s7` / `ftB_s7` | `--seed 7 --init runs/columns_scratch_rob_s7/stage2/last.pt --lr 5e-5 --steps2 8000` NEG `--wide-p 0.15` (B: `--oversample type2=2`) | | | | | running 2026-11-04 |

## Edge candidate log (2026-11-15 on: types 1/4/7 added; 13-run budget)

Single models, scored with `python -m tools.edge_heldout_check <ckpt>` on the held-out strips (type1 5, type2 17,
type3 18, type4 7, type7 6, type10 15). R3 = the r3 flags (`--lr 1e-4 --steps 4000 --exclude-session 131412
--extra-data data/edges_pseudo --extra-data data/oct26/edges:4 --cldice 0.5 --texture 0.3 --texture-max 0.25
--texture-skip type2 --cldice-skip-exempt`); RFG = R3 with `--steps 8000 --rot-max 12 --far-p 0.3 --exposure 0.3 --groove 1.0`.

| run | recipe | type4 exact (MAE) | type7 | type1 | all 68 exact (MAE) | note |
|---|---|---|---|---|---|---|
| shipped `r3_a` | (no types 1/4/7) | 5/7 (0.71) | 4/6 | 5/5 | 63 (0.12) | |
| shipped `rfg_a` | (no types 1/4/7) | 4/7 (2.43) | 4/6 | 5/5 | 61 (0.31) | |
| `x1_a/b` | R3 from edges_final, + new strips (groove-to-groove labels) | 2/7 (9.86) | 3/6 | 4/5 | 59 (1.07) | wide labels -> sheets merge (pair: type4 2/7, MAE 9.3) |
| `x2_a` | as x1, type4/7 labels THINNED to ~45 % | 4/7 (0.86) | 4/6 | 5/5 | 63 (0.12) | thinning fixes the collapse |
| **`x3_a`** | RFG from edges_final, + new strips (thinned) | **6/7 (0.14)** | 4/6 | 4/5 | **63 (0.07)** | best single so far |
| `x4_a` | x3 + `--repeat-type type1=3,type4=3,type7=3` | 6/7 | | | 62 (0.09) | no gain; type2 15/17 |
| `s16_a` | RFG from SCRATCH, base 16, `--lr 1e-3 --steps 12000` | 4/7 (0.57) | 4/6 | 5/5 | 63 (0.09) | type2 17/17; rot +8/-8: 44 / 35 |
| `s24_a` | as s16 with `--base 24` | 5/7 (0.29) | 4/6 | | 64 (0.06) | rot +8/-8: 47 / 36 |
| **`s32_a`** | as s16 with `--base 32` | **6/7 (0.14)** | **5/6** | | **66 (0.03)** | best; rot +8/-8: 45 / 38 (x3_a 45 / 39) |
| `s32_b` | s32 seed 7 (second seed) | 6/7 (0.14) | 4/6 | 5/5 | 63 (0.07) | type2 17/17, type3 16/18; rot +8/-8: 47 / 40 |
| `s32_a` + `s32_b` averaged | | 5/7 | 4/6 | 5/5 | 64 (0.06) | type1/2/3/10 all exact; misses = type7 green light x2 + type4 x2 |
| `x5_a` | x3 + `--clutter 0.5` (background past the strip ends, `edges/clutter.py`) | 4/7 (1.29) | **6/6** | 4/5 | 62 (0.18) | type7 green-light misses FIXED; type4 125748_002/_005 +4 (dense lower half breaks into streaks, the shipped models' type4 failure -- not clutter-shaped); rot +8/-8: 42 / 40 |
| `f32c_a` | FINAL 1/3: s32 + `--clutter 0.5`, seed 42 | 6/7 (0.14) | 5/6 | 4/5 | 65 (0.04) | type2/3/10 all exact |
| `f32c_b` | FINAL 2/3: as f32c_a, seed 7 | **7/7** | 5/6 | 5/5 | **67 (0.01)** | one miss: type7 143956_017 (-1) |
| **`f32c_a` + `f32c_b` averaged** | | 6/7 | 5/6 | 5/5 | **66 (0.03)** | misses type4 130138_002 (-1), type7 143956_017 (-1) |
| **`f32c_c`** | FINAL 3/3: as f32c_a, seed 3 | **7/7** | **6/6** | **5/5** | **68 (0.00)** | every held-out strip exact |
| `f32c_a+b+c` averaged | | 7/7 | 5/6 | 5/5 | 67 (0.01) | `f32c_b+c`: 67 (0.01) |

Whole pipeline, `bash tools/candidate_check.sh edges_f32c_ab shipped runs/edges_f32c_a/last.pt runs/edges_f32c_b/last.pt`
(column model = shipped rep1_s42; vs the shipped four sheet models, `cc_rep1_s42`):

| check | shipped 4 x base16 | f32c_a + f32c_b (2 x base32) |
|---|---|---|
| held-out test single / voted | 0.994 / 1.000 | 0.988 / 0.988 (type10 21/23: 140757 col x~985 counted 25 in frames 013+014, unflagged) |
| unseen type2 exact / voted | 77.9 / 81.7 % | **80.8 / 85.6 %** |
| type1 142210 (30 pcs) exact | 61.1 % | **83.3 %** (all type1 91.4 %, voted 94.3 % vs 80.0 / 84.3 %) |
| October pinned type3 exact (sharp) | 29.6 % (60.9 %) | 25.9 % (56.5 %) |
| stress exposure -2.5 / +2 EV | 0.941 / 0.941 | 0.959 / 0.941 |
| stress rotate -10/+10/-20/+20 | 0.968 / 0.938 / 0.928 / 0.899 | 0.987 / 0.956 / 0.954 / 0.939 |
| stress zoom 0.6 / 0.75 | 0.243 / 0.828 | 0.314 / 0.846 |

The type10 miss is the top end sheet under the bright cap: the shipped set skips it and flags the column UNCERTAIN;
the base-32 pair marks it (25) with confidence, so the uncertainty flag no longer fires. Its top blob is 0.69 of the
median blob (shipped 0.40), above `END_FLOOR` 0.50. Raising END_FLOOR does NOT pay (f32c_a+b+c, voted):
0.60/0.65 test 0.988 (no change), type1 voted 94.3 -> 90.0 %, October sharp 71.4 -> 65.7 %; 0.75 test voted 1.000
but one type3 end sheet lost, type1 voted 81.4 %. Keep 0.50 (`python tools/with_floor.py <value> tools.evaluate ...` runs any tool with another value).
Three models (a+b+c): test 0.988 / 0.988, unseen type2 85.6 %, type1 voted 94.3 %, October type3 27.8 % (sharp 60.9 %).
f32c_c ALONE (68/68 strips): test 0.976 / 0.976 (two type3 columns wrong), unseen type2 85.6 %, type1 voted 95.7 %,
October type3 29.6 % (sharp 60.9 %), stress about the same -- the best strip model is not the best pipeline model;
average the three. Shipped four on the same strips: 64/68 (type4 4/7, MAE 2.86).

**Edge candidate 2026-11-15: `f32c_a + f32c_b + f32c_c` (runs/_candidates/edges_f32c_abc), NOT installed.**
vs the shipped four: held-out strips 67 vs 64/68; unseen type2 voted 85.6 vs 81.7 %; type1 voted 94.3 vs 84.3 %
(30-piece 142210: 83 vs 61 %); exposure/rotation stress better; October type3 equal (sharp 60.9 %); held-out frame
test 0.988 / 0.988 vs 0.994 / 1.000 (the one type10 column end, 140757 x~985). CPU cost ~150 ms per strip vs ~110 ms
(three base-32 vs four base-16); f32c_a+b alone is ~100 ms and scores the same except October sharp 56.5 %.
Inputs/outputs unchanged (n x 3 x h x w -> n x 3 x h x w).

Type7 misses (all 143956): the model counts the green window light at the strip end as a sheet (24 vs 23 labelled).
The +-8 deg rotation stress (`--rotate`) is NOT valid for type3 (and partly type1/type7): it rotates the strip inside
its own ~73 px wide canvas, which clips a narrow strip into a thin wedge while the labels still count the sheets in
the slivers (131412_001_column_1: 9 counted, 21 labelled sheets left, most a few px wide). Its type3 collapse
(18 -> 0-4 for every model) is the test, not the model; rotate on a padded canvas before trusting it.
Width (CPU, one 80x2048 strip, busy machine): base 16 28 ms, 24 38 ms, 32 51 ms -- one base-32 model costs less
than the shipped four averaged base-16 models (~110 ms). Rotation margin (+-8 deg) does not grow with width; type7
drops to 0/6 at -8 deg for EVERY model (its sheets are steep, so a tilted straightening merges them).
x3_a's 5 misses: type1 142016_009 (-1), type2 134020_011 (+1), type4 125748_002 (+1), type7 143956_012/_017 (+1, the
green light). The shipped four-model set counts type4 125748_002 as 59: textured type4 sheets break into pieces along
their length; the new type4 strips fix that (x3_a 51). `--clutter` targets the green-light miss: training windows
only ever had black past the strip ends, while real strips run past the stack into lit background.

## Data

- **Expected counts** (the ground truth for counting): type3 = 50, type10 = 24, type2 = the
  `_NNpieces_` number in the file name (50 or 30). For other types that number is the batch size,
  not the sheet count.
- **Splits** (`data/columns/splits.json`, `data/edges/splits.json`) are pinned. Held-out frames are
  whole capture sessions, because the camera is fixed and neighbouring frames are near-identical;
  a frame-level split leaks. Type10 has only one capture session, so its held-out frames (9-14,
  with 8 and 15 as buffers) come from the session it trained on.
- **Labels**: LabelMe JSON next to each image. Columns: `good_column` (complete) and
  `cropped_column` polygons on the full frame. Sheet edges: one `edge` polygon per sheet on the
  straightened crop (`edges/make_crops.py` makes crops from column labels; `common/labelme.py`
  reads both). type2's 30-piece crops are labelled as thin lines, all others as bands.
- **Negatives** (column-free images, NOT in this repo: ~1 GB, in `../datasets/negatives_web/`). To rebuild:
  download `http://data.csail.mit.edu/places/places365/val_256.tar` and
  `filelist_places365-standard.tar` (same folder), extract to `negatives_web/places/` and
  `negatives_web/meta/`, then `python -m tools.negative_check --build`. Train set: 90 Places images per
  scene category (32,850) + photos/patterns; test set: 1 other image per category (365), crops of the
  HELD-OUT frames away from any column (98), photos/patterns (33). `--bg-frac` crops come from the
  training frames on the fly. The photo/pattern part is copied from `runs/_wexp/negatives/` if present
  (the exact set behind the numbers here, incl. ~27 local wallpaper photos); otherwise `--build`
  regenerates synthetic patterns + scikit-image samples, which changes only the small "misc" row.
- **AI-agent column labels, types 1/4/7** (`data/columns/quality_agent/`, 245 frames, received 2026-11-14 as
  agent_colum_labels.zip, original kept in `../datasets/agent_column_labels_20261114/`). Made by Sonnet agents, NOT
  humans. Before use they were (1) checked against the 6 human-labelled type4 frames (every class matched, IoU
  0.78-0.95; those 6 keep the human labels), (2) shrunk 3 px (the agent tool traces ~3 px outside the sheets; 6 %
  larger than human), (3) relabelled cropped in the type1 side view 134502 (user rule). Train them at
  `--agent-rep 1` (human frames x4): at x4 the model copies their band-boundary convention and type10 counts +1.
  Held out: type1 session 142016, type4 sessions 125748 + 130138, and for type7 (one session per variant) a frame
  block of each session with a buffer frame on each side (`tools.column_heldout_check` reports type7 SAME-SESSION).
- **AI-agent EDGE labels, types 1/4/7** (`data/edges/type1|type4|type7/`, 82 strips, 3312 sheets; received 2026-11-15 as
  "type147_edges (1).zip" -- the newer, complete copy -- original in `../datasets/type147_edges_20261115/`). Made by an
  automatic chain + Opus fixer + zoom check by a lead agent (4 columns with a human in the loop), on full frames
  (good_column + one "edge" polygon per sheet); converted to strips with the production straightener
  (`common.straighten.straighten_frame_pairs`), named `<frame>_column_<col>`. Only status done/gold; the type1 side
  view 134502 is left out (those columns are cropped). Conventions differ from the human type2/3/10 labels: type1 =
  bright front stripe (~half pitch, 42-56 % of the strip), type4/7 = groove to groove (70-76 %). Type7 counts are NOT
  the file name: 143643 = 39 sheets, 143956 = 23 (22 + a separate end sheet). Held out like the column split
  (type1 142016, type4 125748 + 130138, type7 frame blocks). Score with `tools.edge_heldout_check`.
- **October test frames** for `tools.oct_field_count --pin`: `data/oct26/pinned_test_frames.json` (58).
- **Type1 frames** (no labels yet): `../datasets/quality_dataset_20260916/quality_dataset/type1_quality/passed/`.
  Three kinds: side view 134502 (every column cut off: target is NO complete column), 50 pieces
  (141559, 141837, 141934), tilted 30 pieces (142210).

## Known limits

- **Type2 on new frames** is ~75 % (91.5 % of unflagged columns); errors are sheets gained or lost at
  a column's end, mostly in the steep S-curve variant. More type2 column labels from new capture
  sessions is the fix; nothing tried without new labels helped (below).
- **Type10** comes from one capture session.
- **Camera placement** (`tools.stress`, old test): zoom-out 0.75x 0.805, 0.6x 0.095 (type10 columns
  missed, type2/type3 sheets merged); roll +-20 deg 0.89; pitch 20 % 0.82; exposure -2.5 / +2 EV
  0.953 / 0.929. Keep the camera fixed.
- **Column-free images**: the shipped column model reports a column on 83 % of them (`tools.negative_check`).
- **Motion blur** can miscount; it is acceptable to fail on blurry frames (take another photo).
- **Other column types** (1, 4, 5, 7, 8, 9) have no labels; do not trust their counts.

## What was tried and did NOT help (do not repeat without new evidence)

- Growing, shrinking or smoothing the column outline before straightening; an end-sheet "width"
  rule. Growing type2 outlines by 1-2 px makes the 20-column type2 test perfect but loses 4-6
  columns on unseen type2 frames (it fixes clipped end sheets and adds neighbour slivers equally).
- A 384-px column model (worse on type10, fragments called complete, 2.25x compute).
- Column self-training (count-validated pseudo labels for the column model): worse on type2.
- Training on all labelled data including the test sessions: no better on unseen frames.
- Boundary-weighted column loss; oversampling type2 frames: no change.
- Full +-180 deg rotation augmentation (the gravity prior matters); inference-time upscaling of
  strips (the edge model is tuned to the native sheet pitch); a label-guided lighting augmentation
  (an augmentation must never use the label).
- Rotation augmentation +-20 deg for the steep S-curve (133033 +2, 133746 -3, old test 0.982); a pair
  mixing one round-3 and one oct model (the type2 gain needs both models); column fine-tunes on the
  9 October frames (any weight or LR: new-camera columns 4->5 of 6 but borderline old type2 columns
  flip to cropped, unseen type2 complete columns 104 -> 94-99). The column model needs more
  new-camera column labels.
- clDice weight 2.0 (sheets fuse; type2 17/20); strong texture augmentation (p 0.5, 40 %) without
  restriction (unseen type2 65 %).
- Ranking models by mask IoU: the count is the measure of success; IoU does not rank models.
- (2026-11-04) Robustness augmentations as a 4000-step low-LR fine-tune (`--exposure 0.4 --wide-p 0.35`
  for columns, `--exposure 0.4 --far-p 0.3 --rot-max 8` for edges, two seeds): distance 0.6x 0.10 -> 0.35,
  roll/tilt and exposure unchanged, old test 0.971-0.988. Negatives alone did better (above). Retry the
  wide/exposure/far augmentations only in a FROM-SCRATCH run.
- (2026-11-04) Hand-editing weights (closed-form head re-solve, image-level gates): worked on the
  test sets but is not repeatable; discarded for training-time fixes.
- (2026-11-04) Correcting the half-pixel label offset (labels are resized nearest = block top-left, images
  bilinear = block centre, so masks sit ~3.75 px down-right at 1920 px): outlines fit the labels better
  (columns cut > 5 %: 309 -> 208) but type10 counts +1 on 7-11 of 23: the outline now reaches across the
  gap into a sliver of the neighbouring column. Needs the end handling fixed first.
- (2026-11-04) Smooth column outline at inference (foreground from the bilinearly scaled-up probability instead
  of the 7.5 px staircase): old test 1.000 -> 0.947 (type3 118/126, one column no longer straightens),
  October exact 24 -> 18 %, unseen type2 72 -> 70 %. The downstream stages are tuned to the current outline.
- (2026-11-04) Measured: the column mask sits SHIFTED DOWN on every type (held-out columns spill 5-15x more
  past their lower side than their upper side, and miss the upper side): the half-pixel label offset above.
  The straightener turns columns 180 deg, so this is the slice of the next column at the TOP of the strips.
- (2026-11-05) In-frame occlusion augmentation for columns (`--occlude 0.3`: a band hides part of a complete
  column, which is relabelled cropped): type1 side view fixed, but the model over-calls cropped everywhere
  (old test 0.941, unseen type2 complete 101 -> 77, false columns 17 -> 30 %).
- (2026-11-05) Anti-merge loss for sheet models (`edges.train --groove 1.0`): merged masks -> 0 and better counts
  on rotated strips, but every sheet mask thins and breaks (whole masks 65 -> 15 %, S-curve 133746_032 counts 31).
  Untested at a lower weight (0.2-0.3) -- the next thing to try for merged sheets.
- (2026-11-05) Sheet-model pairs on top of the aligned column models (`bash tools/candidate_check.sh`):
  `edges_rot12_a/b` (r3 recipe, `--rot-max 12`): same as r3 within noise (old test 0.994/1.000 vs 0.988/0.994,
  unseen type2 81 vs 83 %). `edges_rfg_a/b` (+ `--far-p 0.3 --exposure 0.3 --groove 1.0`, 8000 steps): fixes
  camera tilt (pitch -10 % 0.876 -> 0.970) and helps distance (0.6x 0.10 -> 0.24) but unseen type2 83 -> 74 %
  and broken masks (groove term). NEXT TO TRY: the rfg recipe WITHOUT `--groove` (or 0.2).
- (2026-11-15) STRAIGHTENER FIX (`SheetCounter.straighten_columns`): with a finely traced outline (512-px column
  model) a small notch could pass for the opposite edge, so the column looked ~10 px wide and the inward cut emptied
  it ("could not straighten": hr512_s42 lost 8 type3 test columns this way). A column whose two edges are closer than
  half its own width is now retried with simplified outlines (approxPolyDP 3, then 6 px); if none passes the original
  result stands. Production rep1_s42 and the 256 control score exactly as before (0.994 / 1.000, 0.971 / 0.982).
- (2026-11-15) DENSE fusion check, `python -m tools.seam_check [--models <dir>]`: every pair of labelled columns that
  touch on a held-out frame is one trial (501 pairs on 99 frames). rep1_s42 / rep1_s7 merge 17 / 18 pairs (8 / 7 with a
  complete column), seam_s42 / seam_s7 17 / 18 (9 / 9); share of the labelled gap painted as column 26.3 / 27.1 % vs
  25.4 / 26.0 %. Merges sit in type7 (5-7 of 72 pairs) and closeup (6-8 of 58). A quarter of every gap is painted
  column at 256 px input -- the gap is ~1 px there; resolution, not the loss, is the limit.
- (2026-11-15) Seam-weighted column loss + pasted cropped pieces (`columns.train --seam-w 5 --paste 0.3`, seeds 42/7
  vs rep1_s42/s7): held-out fusions unchanged (fused-and-complete 6/6 vs 6/5); fewer only on frames it trained on
  (19 vs 34-36). Costs +3-5 pts false columns and the voted test 1.000 -> 0.994. See the column candidate log.
  The held-out frames contain only 2 harmful fusions -- label more stacked-piece / strap frames before retrying.
- (2026-11-15) SMOOTH column outlines (`python tools/with_smooth_columns.py <sigma> <tool> ...`: each column's 256-px
  mask blurred and upsampled bilinearly instead of the ~7.5 px nearest-neighbour staircase). Looks right (no teeth
  in the gaps) but does NOT fix overcounts -- they come from how far the outline reaches at a column END, and the
  rounded end lets in one more partial end sheet: with f32c_a+b+c, sigma 1 / 0 / 2 give test 0.976 / 0.970 / 0.982
  (jagged 0.988), type10 19-20/23 (jagged 21/23); with the shipped four, sigma 1: test 0.982 / 1.000, unseen type2
  85.6 % voted (jagged 81.7), type1 voted 77.1 % (jagged 84.3). Keep the staircase for counting.
- (2026-11-15) Fused columns (`python -m tools.fused_columns`: a predicted mask holding >= 25 % of 2+ labelled
  columns): 55 of ~2600 predicted columns on all labelled frames, most two CROPPED stacks called cropped (harmless).
  The harmful kind: a cropped piece directly above/below a complete column (type1 142210_025, type7 143956_003/027/028)
  or two pieces either side of a strap (type1 142016_019/020), merged and called COMPLETE. The seam is ~1 px at 256.
  The class head barely sees it: P(cropped) > 0.5 on > 20 % of the mask flags 4/14 fused vs 4 % of good columns.
- (2026-11-15) Zoomed SECOND PASS of the column model (re-run on a crop around each predicted complete column, 15 %
  margin) to split fused columns: on all 465 labelled frames it splits 89 / 1119 GOOD complete columns (8 %, mostly
  dense type3 131612) and separates only 6 / 14 fused ones -- the model never saw zoomed-in crops. Inference-only
  zoom is not free for the current column model (it was for the July model trained with ROI zoom).
- (2026-11-15) Motion-blur augmentation for the sheet models (`edges.train --motion 0.4`): camera shake 9 px 0.18 -> 0.45,
  15-25 px still ~0 (the grooves are gone from the image), unseen type2 89 -> 84 %. Dropped; the defence is the
  uncertainty flag / retake. `tools.stress` has motion rows.
- (2026-11-15) Orientation: turned frames fail in EVERY stage (old test turned 90/180/270: 0.40 / 0.57 / 0.34; at 180 the
  columns are right but the strips reach the sheet models upside down). The column model's own confidence picks the
  upright orientation on only 68 % of frames. Quarter-turn training (`--turn`) fixes the column stage but costs upright
  type10. Open: a geometric type10 rule along the stack direction + upright strips for turned frames.
- (2026-11-15) AI-agent column labels (types 1/4/7): fit for training after shrinking 3 px and weighting 1x; at 4x the
  model adopts their band-boundary convention and type10 counts +1 (15/23).
- (2026-11-05) Averaging SIX sheet models (r3 + rfg + rot12 pairs) instead of the installed four (r3 + rfg): old test
  0.988 / 0.994, unseen type2 83 % (vs 89 %), type10 at -2.5 EV unchanged (15/23). The four-model set stays.
- (2026-11-05) Automatic exposure correction at inference (`SHEETCOUNT_AUTO_EXPOSURE=1`, off by default: a frame whose
  95th-percentile brightness is < 0.45 is brightened in linear light to 0.70): old test at -2.5 EV 0.935 -> 0.994
  (type10 15 -> 22/23), clean unchanged, but the real October frames get worse (exact 46.9 -> 40.0 %, MAE 3.75 ->
  5.81: brightened dim/blurred columns are called complete and miscounted). Use only for a known-dark camera.
- (2026-11-04) Retrain noise: the shipped column recipe retrained with seed 42 scores 0.988 / 0.994
  (type2 132812_027 49, type10 014 25). That borderline type2 column end flips on every retrain.

## Practical notes

- Windows: any script that starts training must keep its work under `if __name__ == "__main__":`
  (data-loader workers re-import the script). `cv2.setNumThreads(0)` is set in every module used
  by data-loader workers.
- Training writes TensorBoard logs to `runs/<run>/` (`tensorboard --logdir runs`).
- Checkpoints: `common.unet.load_checkpoint(path, device)`; ONNX input is float32 N x 3 x H x W,
  RGB, `(x / 255 - 0.45) / 0.23`, H and W divisible by 16.
