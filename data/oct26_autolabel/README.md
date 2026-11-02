# October 2026 AUTO-LABELLED frames (model-free labeller + Claude review), kept separate from human labels

Made 2026-10-26 with tools/label_sheets (bands -> envelope -> beamfill -> cropclean -> sheets run -> darktrim ->
splitwide -> hand fixes), validated first against a human-labelled frame (VALIDATION.md).
Same LabelMe format as data/oct26/frames: good_column / cropped_column / one `edge` polygon per sheet
(narrowed to the human 0.5 x pitch band convention).

| folder | frame | complete columns | sheets per complete column (labels = photo count) |
|---|---|---|---|
| type3 | 20261002_111820_type3_-4_001 / 011 / 014 / 016 / 018 / 019 | 2 each | 19 + 19 |
| type5 | 20261002_110718_type5_-4_003 / 004 / 005 / 006 / 007 / 008 | 2 each | 36 + 36 |
| type1 | 20261002_102909_type1_-3_001 / 004 / 006 | 2 each | 50 + 50 |
| mainshots | 2026-10-02_10-32-49_241_original | 1 | 50 |

Every sheet count was checked against an independent photo count (bright crests along 4 lines across the column).
`review/<folder>/<frame>.jpg` shows every label for inspection.

Rejected (not good enough to train on, NOT included): type1 _013 (identity jumps in the left sheets), type1 _005
(shadowed right end of the lower column fragmented into end pieces + one diagonal label), type3 _017 (motion blur,
columns could not be separated),
mainshots 09-27-12 (outermost sheet in shadow could not be separated cleanly), mainshots 09-51-57
(low-contrast touching sheets, photo counts inconsistent 37-43).

None of these frames is in a held-out set (tools/fragment_eval.HOLDOUT: mainshots 09-53/54/56, type3 103619,
type1 103338) or near-duplicate (NCC < 0.97) of a held-out frame.

## Bulk round (2026-10-26): +7 frames
Unattended pipeline (round-3 labeller) + review. Hand fixes: t3_018 missing band gap (next band's corner cut out as
its own cropped column), t5_006 rack rail + bracket removed from a cropped column, t3_019 gap angle forced to -35
(`BANDS_FIX_ANGLE=1`), type1 merged sheet pairs split (`splitwide 1.6 <ids>`: _004 2 pairs, _006 2 pairs), t1_006 two
sheet bottoms re-joined to their sheets (lane check: equal +20 px shift), t5_005 one side-face patch joined to sheet 1.
Every complete column = product count AND agrees with the photo count.
The remaining unlabelled October frames are reserved for evaluation: `runs/_oct_field/pin_spree.json` (58 frames,
none labelled, none a near-duplicate of a labelled frame) -> `python -m tools.oct_field_count --pin runs/_oct_field/pin_spree.json`.
