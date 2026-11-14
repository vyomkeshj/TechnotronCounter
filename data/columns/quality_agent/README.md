Column labels for types 1, 4, 7 (quality_dataset frames), made by AI agents (Sonnet) on 14 Nov 2026,
received as agent_colum_labels.zip (original kept in ../datasets/agent_column_labels_20261114/).
Checked 2026-11-14: on the 6 type4 frames that also have human labels (data/columns/operational), every column
class matches and IoU is 0.78-0.95; those 6 are NOT copied here (the human labels are used).
Changed: type1 side-view session 134502 -- every column set to cropped_column (user rule 2026-11-03: the columns
run behind the beam; the agents had marked 2-3 per frame good).
Shrunk 2026-11-14: every agent polygon eroded by 3 px. The agent tool traces ~3 px OUTSIDE the sheet ends; on the 6
type4 frames with human labels the agent area was 1.062x the human one (6.6 % outside vs 0.4 %), 3 px balances it
(2.5 % / 2.1 %, column IoU 0.916 -> 0.943). Unshrunk, a model trained on them spilled into the gap between columns
and type10 counted +1 on 5-6 test columns (neighbour sliver).
