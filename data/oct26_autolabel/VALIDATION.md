# Validation of the model-free labeller against human labels (before labelling any training frame)

Frame: data/oct26/frames/type3/20261002_111820_type3_-4_004 (human-labelled, NOT added to this dataset).
Labelled from scratch with tools.label_sheets (bands -> envelope -> sheets run -> darktrim -> splitwide), compared with
the human polygons:

| item | result |
|---|---|
| complete columns, IoU with human outline | 0.937 and 0.956 |
| cropped columns, IoU | 0.77 (small top piece), 0.89, 0.89 |
| sheets | 38 human / 38 mine, 0 missed, 0 merged, 0 split |
| sheets per complete column | 19 / 19 and 19 / 19 (human 19 / 19) |
| sheet IoU (after narrowing to the human 0.5 x pitch convention) | median 0.75, min 0.68 |

Fixes the validation forced (all now in the tools): zigzag bands need a gap-angle hint (the sheet grooves line up
across bands and are longer than the band gaps); gaps are fitted as parallel, regularly spaced lines; both halves of a
chevron band kept (pieces >= 3 %, not 15 %); column = envelope of its band (line closing along the band, then tip
notches re-opened); sheet labels on dark gaps dropped; labels 2x the normal width split along the faint groove.

## Round 2 (2026-10-26, after the user: "this clearly needs more effort")
Full audit against ALL 15 human October frames (runs/_label_live/validate_all.py: window + complete/cropped taken from
the human file, everything else automatic; tools/label_sheets audit in runs/_label_live/audit_vs_human.py).

Sheet labels match the human convention: band width ratio 0.86-1.07, centre offset < 1 px, same brightness.
Not the problem.

Fixes:
1. chainmerge (sheets): pieces of one sheet split end-to-end (light -> shadow) joined by STACK ORDER (never on the same
   cross-line + same neighbours). type1 raw labels 55 -> 50 (= human), 0 false merges on 11 correct frames.
2. beamfill (columns): the beam edge is TILTED (camera roll) and, in deep shadow, is a faint lit ridge BELOW the visible
   sheet ends -> traced per 40 px slice, fitted as a line. type1 bottom cropped column IoU 0.851 -> 0.948.
3. bands (gaps): gaps judged by RELATIVE depth (over-exposed frames) and the angle searched at the hint AND its mirror,
   keeping the one whose dark lines are far apart (band gaps) not one pitch apart (sheet grooves).

Unattended column IoU over the 15 human frames: complete median 0.905 -> 0.913, cropped median 0.768 -> 0.831.
Still failing unattended (need the human-eye gap check, as done for every training frame): over-exposed/blurred type5
(_001, _011), type3 session 103619 (_017), mainshots with the wire rack beside the stack (09-53, 09-51).
Training frames re-exported with fix 2 (cut columns now reach the tilted beam: 2.3k-17k px added per frame).

## Round 3: equal-band-width assumption removed (gap finder)
- Gaps are no longer generated at a regular spacing. Every line at the gap angle gets a darkness score (median along
  the line inside the window); each clear minimum is a gap if it has METAL ON BOTH SIDES (rejects beams and the
  stack's outer edge). Slope from the traced gaps; traced curvature kept where a trace follows the line.
- Angle choice: contrast measured in the middle of the window only (the beams at the window's top/bottom pulled every
  frame toward 0 deg); hint and mirror both searched; spacing is only a VETO (dark lines closer than 0.6 x band
  spacing = sheet grooves), otherwise the higher-contrast direction wins.
- Column snap: near-saturated pixels count as metal (over-exposed sheets lose their grooves).

Unattended over all 15 human frames (column IoU vs human):
| | start of round 2 | round 2 | round 3 |
|---|---|---|---|
| complete columns median / worst | 0.905 / 0.157 | 0.913 / 0.401 | **0.929 / 0.816** |
| cropped columns median | 0.768 | 0.831 | **0.886** |
Still weak: small cropped corner pieces (some 0.13-0.58), mainshots with the wire rack (09-53).
