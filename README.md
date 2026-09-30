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
| type2 | **19 / 20** | 50 or 30 (the `_NNpieces_` number in the file name) |

**Self-check:** every counted column is also counted by each edge model alone and from a
column outline one pixel wider; if any count differs the column is flagged **uncertain** (take
another photo). On the held-out frames, every column that is NOT flagged is correct:
type2 19/19, type3 126/126, type10 21/21 (3 flags in 169 columns, one of them the one error).

Other checks (all reproducible, see "Reproduce the numbers"):
- **Unseen type2 frames** (108 unlabelled frames, `data/unlabelled`): ~73 % exact per frame,
  ~75 % with 5-frame voting; of the columns NOT flagged, 91.5 % are exact. Type2 on brand-new
  frames is the weak spot, mostly its steep S-curve variant (session 133033).
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
(same recipe, different seeds) each predict a sheet-foreground probability; they are averaged
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
data/unlabelled/type2/      108 frames with no label of any kind (for tools/unseen_check.py)
runs/                       the five checkpoints behind the shipped models (see below)
paths.py                    every location, relative to this folder
```

## Reproduce the numbers

Run everything from this folder with Python 3.10+ (developed on 3.13, Windows, RTX 4090).

```bash
pip install -r requirements.txt           # after installing torch for your CUDA from pytorch.org
python -m tools.evaluate --vote 5         # the table above; writes runs/_evaluate/columns.csv
python -m tools.unseen_check              # unseen type2 frames (~73 % / ~75 % voted)
python -m tools.stress                    # robustness table (slow: ~12 perturbations x 40 frames)
python -m tools.report                    # one render per held-out frame -> runs/_report/
```

Expected `tools.evaluate` output with the shipped models:

```
type    complete  exact  wrong  cropped  missed  false   score  unflagged correct   voted
type2         20     19      1        0       0      0   0.950           19/19      0.950
type3        126    126      0        0       0      0   1.000          126/126     1.000
type10        23     23      0        0       0      0   1.000           21/21      1.000
all          169    168      1        0       0      0   0.994          166/166     0.994
```

## Reproduce the models

Five training runs (~2 h on one RTX 4090), then an ONNX export. The held-out test sessions
(type2 132812 + 134020, type3 131412, type10 frames 9-14) are never trained on: the column split
holds them out, and `--exclude-session 131412` keeps type3's test session out of the edge model.

```bash
# column model: base recipe, then fine-tune with wider zoom-out augmentation
python -m columns.train --name base  --class-override data/columns/relabel_type3_edge_columns.json
python -m columns.train --name final --class-override data/columns/relabel_type3_edge_columns.json \
    --init runs/columns_base/stage2/last.pt --lr 5e-5 --scale-lo 0.65

# edge models: a seed model, self-labelled crops, then two self-trained models
python -m edges.train --name seed --seed 7 --exclude-session 131412
python -m tools.pseudo_edges --model edges_seed --out data/edges_pseudo
python -m edges.train --name final_a --seed 42 --exclude-session 131412 --extra-data data/edges_pseudo
python -m edges.train --name final_b --seed 7  --exclude-session 131412 --extra-data data/edges_pseudo

# export to the inference package, then score
python -m common.export_onnx runs/columns_final/stage2/last.pt inference/models/column_flow.onnx
python -m common.export_onnx runs/edges_final_a/last.pt inference/models/sheet_edges_a.onnx
python -m common.export_onnx runs/edges_final_b/last.pt inference/models/sheet_edges_b.onnx
python -m tools.evaluate
```

The five checkpoints this produced are in `runs/` (`columns_base`, `columns_final`,
`edges_seed`, `edges_final_a`, `edges_final_b`), so any step can start from the shipped
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

## Known limits

- **Type2 on new frames** is ~75 % (91.5 % of unflagged columns); errors are sheets gained or lost at
  a column's end, mostly in the steep S-curve variant. More type2 column labels from new capture
  sessions is the fix; nothing tried without new labels helped (below).
- **Type10** comes from one capture session.
- **Camera distance**: a 10 % zoom-out still costs type2/type10 columns. Keep the camera fixed.
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
- Ranking models by mask IoU: the count is the measure of success; IoU does not rank models.

## Practical notes

- Windows: any script that starts training must keep its work under `if __name__ == "__main__":`
  (data-loader workers re-import the script). `cv2.setNumThreads(0)` is set in every module used
  by data-loader workers.
- Training writes TensorBoard logs to `runs/<run>/` (`tensorboard --logdir runs`).
- Checkpoints: `common.unet.load_checkpoint(path, device)`; ONNX input is float32 N x 3 x H x W,
  RGB, `(x / 255 - 0.45) / 0.23`, H and W divisible by 16.
