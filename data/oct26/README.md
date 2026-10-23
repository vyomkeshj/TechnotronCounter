# October 2026 field labels (kept separate from the original data)

Source: `October26-20261022T143556Z-1-001.zip` (Google Drive), photos taken 2026-10-02.
Only the 15 frames that carry labels are copied here; the 100 unlabelled frames stay in
`D:\Projects\technotron\datasets\october26_20261022\October26\`.

Grouped by the folders they were delivered in, NOT by column type:

| folder | frames | resolution | content |
|---|---|---|---|
| `mainshots` | 5 | 1920 x 1440 | no type in the name; 50 sheets per complete column |
| `type3` | 5 | 2592 x 1944 | sessions 103619 (50 sheets) and 111820 (19 sheets) |
| `goodshots` | 5 | 2592 x 1944 | type1 (50 sheets), type5 (36 sheets) |

Labels: LabelMe, `good_column` / `cropped_column` plus one `edge` polygon per sheet on the
full frame (highest-quality labels in the project).

- `frames/<folder>/`  the labelled frames as delivered (jpg + json)
- `edges/<folder>/`   straightened strips + sheet labels, made with
  `python -m edges.from_frames data\oct26\frames\<folder> --out data\oct26\edges\<folder> --type . --dup-ncc 1.01`
  (1 strip per labelled good_column; frame 2026-10-02_09-31-36 has no good_column -> no strip)
