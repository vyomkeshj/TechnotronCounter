# CLAUDE.md

Sheet counter for stacked-sheet columns (types 2, 3, 10): column segmenter -> straightener ->
edge segmenter -> counter. **Read README.md first**; it has the pipeline, the exact commands to
reproduce the numbers and the models, and the list of things that did not work.

Rules that matter here:
- The measure of success is the **sheet count** per complete column (type3 50, type10 24, type2 the
  `_NNpieces_` number). Mask IoU does not rank models.
- Score every change with `python -m tools.evaluate` (held-out test, expected 0.994) AND
  `python -m tools.unseen_check` (unseen type2 frames, ~73 % / ~75 % voted). A change that only
  improves the 20-column type2 test set is usually a fluke: check the larger unseen set and
  **look at the masks** (`inference/predict.py --save-crops`, `tools/report.py`) before believing it.
- Never train on the held-out frames (`data/*/splits.json` "val"/"excluded"; keep
  `--exclude-session 131412` for edge models). Splits are by capture session; do not re-roll them.
- Small test sets: train two seeds before claiming a gain.
- Run from this folder; paths are relative (`paths.py`). Keep training entry points under
  `if __name__ == "__main__":` (Windows data-loader workers re-import scripts).
