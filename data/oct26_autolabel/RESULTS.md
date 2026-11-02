# Training on the auto-labelled October frames: results (2026-10-26, 3 training runs)

Data added: 8 frames (type3 x4, type5 x3, type1 x1) = 16 complete-column strips, 448 sheets, 29 column polygons.
Runs (all fine-tunes of the shipped recipe, only change = the new data, x4 weight like the human October labels):
1. edges_al_s42  (sheet model, seed 42)  2. edges_al_s7 (sheet model, seed 7)  3. columns_al_s42 (column model)

## Per type: October field frames NO model trained on (69 frames), correct complete-column counts
| type (frames added) | shipped | new sheet models | new column model | both new |
|---|---|---|---|---|
| type1 (+1) | 10/10 | 10/10 | 9/13 | 10/13 |
| type3 (+4) | 7/35 (sharp 7/11) | 6/35 (6/11) | 8/64 (8/34) | 7/64 (7/34) |
| type5 (+3) | 7/9 | 6/9 | 7/8 | 7/8 |
| mainshots (+0) | 5/8 | 5/8 | 5/6 | 6/6 |
| total correct | 29 | 27 | 29 | 30 |
("8/64" = 64 columns reported complete, 8 counted exactly.)

## Regression checks
| | shipped | new sheets | new columns | both |
|---|---|---|---|---|
| old held-out test (single / voted) | 0.994 / 1.000 | 0.988 / 0.994 | 0.988 / 0.994 | 0.988 / 0.994 |
| unseen type2 exact (voted) | 76/104 (78) | 74/104 (76) | 75/106 (78) | 71/106 (73) |
| human October holdout: complete columns found | 4/6 | - | 6/6 | 6/6 |
| human October train frames: false complete | 3 | - | 0 | 0 |

## Verdict
- Sheet models: NO improvement (within seed noise, slightly worse). The new strips are type3/5/1 where the shipped
  pair already counts the SHARP frames well; the type3 failures are motion-blurred frames (24 of 35 type3 columns)
  that no label can fix.
- Column model: BETTER at finding complete columns on the new camera (human holdout 4/6 -> 6/6, false complete 3 -> 0,
  type3 complete bands found 35 -> 64) but those extra type3 bands are mostly blurred, so the number of exact counts
  did not rise; one type2 test column flips (0.994 -> 0.988, within the known +-1 column noise).
- Not shipped. Recommended next: the column model is the one that benefits; retrain it after the column-label fix
  (outer shadowed sheets + beam corners now added for type1/type5, IoU vs human 0.95-0.97) with 2 seeds, and add
  sharp-frame filtering (or blur augmentation) for type3.
Labels used by run 3 were the version BEFORE the user's column-edge fix (sidegrow); the dataset now holds the fixed one.

## Dataset after the training runs (final state of data/oct26_autolabel)
9 frames: type3 x4, type5 x3, type1 x1, mainshots x1 (2026-10-02_10-32-49). The mainshots frame and the column-edge
fix (shadowed outer sheets + beam corners, type1/type5) were added AFTER the 3 training runs, so the runs above used
8 frames and the pre-fix column outlines. The 3-run budget was spent; retraining on this final set is the next step.

## FINAL 3 runs (after labelling was finished: 9 frames, column-edge fix included) -- these are the goal's runs
Runs: edges_fin_s42, edges_fin_s7 (sheet pair), columns_fin_s42 (column model). Same recipes as above.
Eval set: 67 October frames no model trained on (excluding labelled frames + near-duplicates).

Exact complete-column counts per type (correct / reported complete):
| type (frames added) | shipped | new sheets | new column model | both new |
|---|---|---|---|---|
| type1 (+1) | 9/9 | 9/9 | 10/11 | 10/11 |
| type3 (+4) | 7/35 | 6/35 | 9/65 | 10/65 |
| type5 (+3) | 7/9 | 6/9 | 8/8 | 8/8 |
| mainshots (+1) | 4/7 | 4/7 | 4/4 | 4/4 |
| total correct | 27 | 25 | 31 | 32 |

Regression checks:
| | shipped | new sheets | new column | both |
|---|---|---|---|---|
| old held-out test single / voted | 0.994 / 1.000 | 0.982 / 0.988 | 0.994 / 0.994 | 0.982 / 0.988 |
| unseen type2 exact (voted) | 76 (78) / 104 | 74 (75) / 104 | 77 (78) / 106 | 74 (75) / 106 |
| human Oct holdout complete columns found | 4/6 | - | 6/6 | 6/6 |
| human Oct frames: false complete | 3 | - | 0 | 0 |

Verdict:
- COLUMN MODEL (columns_fin_s42): IMPROVED. +4 correct complete columns on unseen October frames (27 -> 31), every
  type better or equal (type1 9->10, type3 7->9, type5 7->8, mainshots 4->4 with 0 false completes instead of 3),
  no regression on the old test (0.994 single, voted 0.994 vs 1.000 = one type2 column) and unseen type2 +1 (76->77).
  Why: the auto labels teach it what a COMPLETE column looks like on the new camera (beam-cut bands = cropped,
  the full band = complete) -> it stops calling complete bands cropped (holdout 4/6 -> 6/6) and stops false completes.
- SHEET MODELS (edges_fin): WORSE (25 vs 27; old test 0.994 -> 0.982, type2 -2). The shipped pair already counts sharp
  frames of these types; the extra strips mostly shift capacity away from type2. Do not ship.
- Recommended ship candidate: new column model + shipped sheet pair. Confirm with a 2nd column seed before shipping.
- Remaining type3 errors are motion-blurred frames (sharpness < 0.35): 54 of the 65 reported type3 complete columns.

## Bulk round (2026-10-26): 9 -> 16 auto-labelled frames, column model only, 2 seeds
Fixed evaluation set `runs/_oct_field/pin_spree.json` (58 October frames, none labelled, none NCC>=0.97 to a labelled
frame). Exact = complete columns whose end-to-end count equals the product count (shipped sheet models throughout).

| column model | auto frames | mainshots | type1 | type3 (sharp) | type5 | total exact | old test (single / voted) | unseen type2 voted |
|---|---|---|---|---|---|---|---|---|
| shipped | 0 | 4/7 | 6/6 | 0/4 | 3/3 | 13 | 0.994 / 1.000 | ~75 % |
| columns_fin_s42 | 9 | 4/4 | 6/7 | 4/29 | 3/3 | 17 | 0.994 / 0.994 | 77 % |
| columns_spree_s42 | 16 | 4/4 | 7/7 | 5/27 | 3/3 | 19 | 0.994 / 0.994 | 76.7 % |
| columns_spree_s7 | 16 | 4/4 | 5/7 | 4/26 | 3/3 | 16 | 0.994 / 0.994 | 78.8 % |

Reading: the first 9 frames bought the big step (type3 complete columns found 4 -> ~27, mainshots false
completes 3 -> 0). Going 9 -> 16 is within seed noise (19 vs 16 between seeds; type1 7/7 vs 5/7). The type3 misses
are undercounts by 1-3 on columns the model now finds (mostly session 103619) -> the next lever is the column ends /
sheet model on type3, not more column frames.
