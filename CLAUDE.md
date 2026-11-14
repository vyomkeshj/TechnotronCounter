# CLAUDE.md

Sheet counter for stacked-sheet columns (types 2, 3, 10): column segmenter -> straightener ->
edge segmenter -> counter. **Read README.md first**; it has the pipeline, the exact commands to
reproduce the numbers and the models, and the list of things that did not work.

Rules that matter here:
- The measure of success is the **sheet count** per complete column (type3 50, type10 24, type2 the
  `_NNpieces_` number). Mask IoU does not rank models.
- Score every change with `python -m tools.evaluate` (held-out test, expected 0.994 single / 1.000 voted) AND
  `python -m tools.unseen_check` (unseen type2 frames, ~84 % / ~89 % voted); for sheet-mask quality
  also `tools.fragment_eval` / `tools.fragment_stress`; for column-model changes also
  `tools.negative_check` (false columns on column-free images) and `tools.stress`. Retraining the
  shipped column recipe alone costs ~1 borderline type2/type10 column end (0.988), so compare a
  recipe against a same-seed control, not against the shipped model. A change that only
  improves the 20-column type2 test set is usually a fluke: check the larger unseen set and
  **look at the masks** (`inference/predict.py --save-crops`, `tools/report.py`) before believing it.
- Never train on the held-out frames (`data/*/splits.json` "val"/"excluded"; keep
  `--exclude-session 131412` for edge models). Splits are by capture session; do not re-roll them.
- Small test sets: train two seeds before claiming a gain.
- Run from this folder; paths are relative (`paths.py`). Keep training entry points under
  `if __name__ == "__main__":` (Windows data-loader workers re-import scripts).
- Labels made by AI agents (data/columns/quality_agent) are NOT ground truth: check new agent labels against human
  gold frames (class + IoU + area ratio) before training, and render them. They have run loose and carried their own
  boundary conventions (README, Data). Train them at lower weight than human labels and watch type10.
- Run at most TWO trainings at once (three plus scoring once stalled for 2 h with the GPU idle); after starting a run,
  confirm a progress line within a few minutes. If esoul says "Session has expired", tracking falls back to TensorBoard
  only -- training is fine, the user must renew the token in ~/.config/esoul/credentials.
- The README candidate log is the project memory: add every run (command, same-seed control, numbers) in the same
  step as reporting it, and keep anything a comparison depends on inside the repo (runs/ is git-ignored).
