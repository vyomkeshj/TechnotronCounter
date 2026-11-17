# Model candidates (2026-11-17)

Not installed: `inference/models/` still holds the shipped set (column `rep1_s42` + four base-16 sheet models).
Each folder below is a complete or partial models folder for `inference/predict.py --models <dir>` and every
`tools.*` check. Full numbers, commands and controls are in the main README (Edge candidate log, Candidate log).

## `edges_f32c_abc/` -- best sheet (edge) models: RECOMMENDED next edge set

`column_flow.onnx` (= shipped rep1_s42, unchanged) + `sheet_edges_a/b/c.onnx` = `f32c_a/b/c`: three base-32 UNets
from scratch (seeds 42/7/3), trained with the new type1/4/7 strips and `--clutter 0.5`, averaged by the pipeline.
Checkpoints: `runs/edges_f32c_{a,b,c}/last.pt`. Same inputs/outputs as before (n x 3 x h x w -> n x 3 x h x w).

| check | shipped 4 x base-16 | f32c_a+b+c |
|---|---|---|
| held-out strips counted exactly | 64/68 | **67/68** (f32c_c alone 68/68) |
| held-out frame test single / voted | 0.994 / 1.000 | 0.988 / 0.988 (one type10 column end, 2 frames) |
| unseen type2 voted | 81.7 % | **85.6 %** |
| type1 voted (30-piece session) | 84.3 % (61 %) | **94.3 % (83 %)** |
| October type3 sharp | 60.9 % | 60.9 % |
| exposure / rotation stress | | better on most rows |

CPU ~150 ms per strip (three base-32) vs ~110 ms (four base-16).

    python inference/predict.py <images> --models candidates/edges_f32c_abc --vote 5

## `columns_hr512_s42/`, `columns_hr512_s7/` -- 512-px column models: EXPERIMENTAL (fixes column fusion)

`column_flow.onnx` only (pair it with any sheet models). rep1_s42 / rep1_s7 continued 8000 steps at 512-px input.
Checkpoints: `runs/columns_hr512_s{42,7}/stage2/last.pt`. MUST run at 512: `SHEETCOUNT_COLUMN_SIZE=512` (the ONNX
shape is unchanged, but a caller that letterboxes to 256 must letterbox to 512). Needs the straightener fix in
`inference/sheetcount/pipeline.py` on this branch.

| held-out | 256 control | 512 s42 | 512 s7 |
|---|---|---|---|
| touching column pairs merged (`tools.seam_check`, of 501) | 20 | **7** | **8** |
| merges that lose a complete column | 9 | **1** | **2** |
| test single / voted | 0.971 / 0.982 | 0.965 / 0.982 | 0.936 / 0.971 |
| zoomed-out frames (0.6x) | 0.290 | **0.917** | **0.893** |
| rotate -20 deg | 0.934 | 0.836 | 0.868 |
| false columns on column-free images | 15.7 % | 19.6 % | 19.4 % |

Not production yet: worse at +-20 deg rotation, +4 pts false columns, fewer type2 columns called complete on unseen
frames. Next: a longer 512 run with rotation / negatives retuned.

    SHEETCOUNT_COLUMN_SIZE=512 python inference/predict.py <images> --models <dir with this column_flow.onnx + sheet models>
