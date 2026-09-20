# ONNX exports of the best models (2026-09-17)

Both files: input `input` float32 `[N, 3, H, W]`, RGB, `(pixel / 255 - 0.45) / 0.23`,
H and W divisible by 16. Output `output` float32 `[N, C, H, W]` (logits / flows).
PyTorch vs ONNX Runtime parity: max |diff| <= 6e-5. Re-export with
`python -m common.export_onnx <checkpoint> <out.onnx>`.

## column_flow.onnx — column instances + good/cropped

Source: `runs/columns_v1/stage2/last.pt`.
Held-out, types 2/3/10 (40 frames): instance IoU 0.901, exact column count 82 %,
count MAE 0.20, good/cropped accuracy 0.940.

* Input: the whole frame LETTERBOXED to 256 x 256 (aspect kept, black padding).
* Output channels: 0 foreground logit, 1-2 flow (dy, dx) to the column centre,
  3 cropped-column logit.
* Readout: fg = sigmoid(ch0) > 0.5 -> `common.flows.follow_flows(fg, out[1:3])` gives
  one label per column; a column is CROPPED if mean sigmoid(ch3) over its pixels > 0.5.

## sheet_edges.onnx — sheet count per column

Source: `runs/edges_all3_s7/last.pt` (types 2 + 3 + 10, centre-crop width,
label-free glare aug). Counts, with the relative blob floor below:
held-out val 49/50 (type10 15/15, type3 18/18, type2 16/17); all 80 labelled
type10 crops 80/80; unseen older type3 296/300; unseen older type10 67/122.
`sheet_edges_prev_type23only.onnx` is the earlier type2/3-only model (type10 2/80).

* Input: one straightened column crop -> tight crop (pixels > 0, pad 6) -> keep the
  centre 74 px of width (+ the 6 px pads). Never rescale the height. Run at native
  size (pad bottom/right to a multiple of 16), not letterboxed.
* Output channels: 0 foreground logit, 1-2 flow (dy, dx) to the sheet centre.
* MASKS: use `edges.grow.grow_instances(fg)` — seeds from the counted centre blobs,
  grown outward one pixel-column at a time. Leak 0.00 px/instance on all three
  types vs 5-71 for the flow readout, and the instance count equals the count.
  The flow channels are then only needed if you want the field itself.
* COUNT: fg = sigmoid(ch0) > 0.5; keep the central 50 % of the foreground's width;
  count connected components (8-connectivity) that are >= 6 px AND >= 25 % of the
  median component area. The flow channels are not needed for the count.
  (`edges/counter.py` still uses only the 6 px floor; the 25 % relative floor is the
  measured improvement, pending adoption.)
