# How much data per type, and how much does one type help another?

Two questions, one runnable plan. **Count is the only measure of success** (exact
counts per type); masks only have to be one-per-sheet, offset does not matter.

Run everything from `training_pipeline/`. One run ~35 min on the RTX 4090; up to 6
run in parallel with `--workers 8` (~75 min per batch of 6).

## Fixed recipe for every run in this study

```
python -m edges.train --name <NAME> --seed 7 --workers 8 --squash-weight 0 --lighting v2 [study flags]
```
Only the data flags change: `--limit-per-type`, `--only-types`. Everything else
(centre-crop width, label-free glare aug, 8000 steps, aniso flow target, pitch
jitter, fragment aug) is the measured champion recipe.

Training pool (pinned split, `data/edges/splits.json`): type10 59, type3 54, type2 37
crops. Val: type10 15, type2 17, type3 18 (type10 and type3 val are SAME-SESSION;
only type2 has a held-out session — say so when quoting).

## A. Data scaling per type (how many crops are enough)

For each type T, shrink T's training crops while the other two stay full:

| run name | flag |
|---|---|
| `scale_t10_04` | `--limit-per-type type10=4` |
| `scale_t10_08` | `--limit-per-type type10=8` |
| `scale_t10_16` | `--limit-per-type type10=16` |
| `scale_t10_32` | `--limit-per-type type10=32` |
| `scale_t2_04/08/16` | `--limit-per-type type2=4` … (37 = full) |
| `scale_t3_04/08/16/32` | `--limit-per-type type3=4` … (54 = full) |
| (full reference) | `edges_all3_s7`, `edges_all3_s42` |

`subsample_per_type` spreads the budget ACROSS FRAMES (round-robin, one crop per
frame) so a small budget is never all from one photo — otherwise the curve measures
frame diversity, not crop count.

Also run `--limit-per-type type10=8` with `--seed 42` as a noise probe: single-run
differences under ~5 crops per 180 are seed noise (measured: two seeds of the same
recipe scored 180/181 and 170/181 on the same set).

## B. Cross-type transfer (does one type help another)

| run name | flag | question |
|---|---|---|
| `only_t3` | `--only-types type3` | what does a type3-only model do on type2/type10? |
| `only_t2` | `--only-types type2` | … |
| `only_t10` | `--only-types type10` | … |
| `no_t10` | `--only-types type2,type3` | already trained: `edges_wlight2_s7` (reuse) |
| `no_t2` | `--only-types type3,type10` | how much does type2 contribute to the others? |
| `no_t3` | `--only-types type2,type10` | … |

Transfer = (score on T when T is absent from training) vs (score on T when present).
`edges_wlight2_s7` already gives the headline: no type10 in training -> 2/80 on
labelled type10 crops; with it -> 80/80.

## Reporting

```
python -m tools.report_counts <run> [<run> ...]        # writes runs/_report_counts.csv
```
Columns per run: val exact per type; all labelled crops per type; unseen older sets
(type3 131412 / 131612 split by session, type10 122 older crops). Plot A as
crops-per-type vs exact-% for that type; read B off the same table.

## Known confounds to state with any number

* type3 and type10 val are same-session; only type2 has a held-out session.
* The 300 older type3 crops come from the SAME two sessions as training (the honest
  cross-session number is `edges_loso131612_s7`: 173/181 = 95.6 %, vs 180/181 when
  the session is in training).
* The relative blob floor (0.25) and the champion seed were chosen on these same
  sets, so they are mildly optimistic.
* Older type10 crops (122) come from the colleague's cutter and are only 67/122 —
  part of that is crop bounds, not counting.

---

# RESULTS — data scaling (2026-09-20)

Every run is the fixed recipe, seed 7, only the data flags differ. Scored with
`tools.report_counts` (shipped counter defaults). **Count only.**

## Seed noise first — which columns can rank anything

`scale_t10_08` vs `scale_t10_08_s42` (identical recipe, seeds 7 / 42), and
`edges_all3_s7` vs `_s42`, give the per-set noise band:

| set | n | seed spread | usable for ranking? |
|---|---|---|---|
| val (pinned) | 50 | 0 | **NO — saturated**, 9 of 12 study runs score 50/50 |
| all_type10 | 80 | 0 | yes |
| all_type3 | 72 | 0 | yes |
| all_type2 | 54 | 1 | yes |
| old_t3_131412 | 119 | 0 | yes |
| old_t3_131612 | 181 | 6-10 | only for gaps > 10 |
| old_t10 | 122 | **23** (33 vs 56; 39 vs 61) | **NO — pure noise** |

The 50-crop val is saturated and ranks nothing; the older type10 set swings 23/122
between seeds of one recipe. Both were quoted in earlier sessions — do not.

## A. How many crops of a type are needed

Type T shrunk, other two full. "old t3" = 131412 + 131612 pooled (300 crops).

| crops of T | type10 (all 80) | type2 (all 54) | type3 (all 72) | type3 (old 300) |
|---|---|---|---|---|
| 0 | 1 | - | - | - |
| 4 | 76 | 37 | 64 | 256 |
| 8 | 77 / 77 | 42 | 68 | 273 |
| 16 | 79 | 50 | 66 | 263 |
| 32 | 80 | - | 69 | 288 |
| full (59/37/54) | 80 / 80 | 51 / 51 | 70 / 70 | 297 / 284 |

* **type10 saturates at 32** and is already at 95 % with 4 crops -- but all 80 type10
  crops are ONE session, so this is memorising one stack's geometry. Lower bound only.
* **type2 saturates at 16-37.** Its 37 crops span 36 photos and 7 SESSIONS, so this is
  the only externally valid curve. Plan new types from this one.
* **type3 has NOT saturated at 54** (288 -> 297 from 32 to full). 8 and 16 are tied
  inside the noise band; the curve is 4 < {8,16} < 32 < 54.
* Crops per photo: type2 1.0, type10 3.5, type3 4.5. Budget ~16 crops spread over as
  many photos and SESSIONS as possible -- not 4.

## B. Side effect found: type10 data crowds out type2

type2 was at FULL strength (37 crops) in every one of these runs; only the type10
budget changed. all_type2 (n=54, noise +-1):

| type10 crops | 0 | 4 | 8 | 16 | 32 | 59 |
|---|---|---|---|---|---|---|
| all_type2 | 54 | 54 | 53/52 | 52 | 51 | 51/51 |

Monotone over six points, total -3/54, outside the +-1 noise band. type3 shows NO
such trend (69/67/70/69/68/70), so it is specific to type2.
MECHANISM UNKNOWN. The obvious hypothesis -- type2's 30-piece crops use THIN-LINE
polygons while type3/type10 use BANDS -- was MEASURED AND DOES NOT HOLD. Label
coverage (labelled px / lit px on the tight crop, which equals label thickness over
sheet pitch) and pitch, over every labelled crop:

| type | coverage | pitch | sheets |
|---|---|---|---|
| type2 | 30 % (30pc 30 %, 50pc 24-31 %) | 12.5 px | 50 / 30 |
| type3 | 25 % | 11.5 px | 50 |
| type10 | **54 %** | 12.5 px | 24 |

type2 and type3 label to the SAME thickness; type10 is the outlier. Pitch is identical
across all three. So neither convention nor pitch explains it -- do NOT relabel type2
on this theory. Next diagnostic: what the t2+t3 model's FOREGROUND does on a type10
crop (merged sheets vs missed sheets) via `tools/measure_leak.py` `split_sheets`.

## C. Cross-type transfer (2026-09-20)

Score on type T with T ABSENT from training, vs present. All seed 7, fixed recipe.

| held-out T | model | T absent | T present | verdict |
|---|---|---|---|---|
| type3 | `no_t3` (t2+t10) | 56/72 (78 %) | 70/72 | mostly free |
| type2 | `no_t2` (t3+t10) | 22/54 (41 %) | 51/54 | partial |
| type10 | `no_t10` = `edges_wlight2_s7` (t2+t3) | **1/80 (1 %)** | 80/80 | cold start |

Single-type models (what ONE type alone buys on the others):

| model | type2 (54) | type3 (72) | type10 (80) | old t3 (300) | 
|---|---|---|---|---|
| `only_t2` | 50 | 16 | 44 | 71 |
| `only_t3` | 17 | 70 | **72** | 292 |
| `only_t10` | 19 | 50 | 80 | 205 |
| all three (`edges_all3_s7`) | 51 | 70 | 80 | 297 |

**THE RESULT OF THE DAY: `only_t3` scores 72/80 on type10, but `no_t10` (= type2 +
type3) scores 1/80.** Adding type2 to a type3-only model DESTROYS zero-shot type10.
Same recipe, same architecture, one extra type in the pool. Mechanism unknown (see the
measured non-explanations above).

Consequences:
* "How many crops for a NEW type" has no single answer -- the cold start ranges from
  78 % (type3) to 1 % (type10). Quote a range, not a number.
* Adding data for one type can silently destroy a type the model handles today, and
  **the pinned val would never show it**: `no_t10` scores 17/17 type2 and 18/18 type3
  val while sitting at 1/80 on type10. Gate on the large labelled sets, never on val.
* This is the measured justification for `field/gate.py`.
