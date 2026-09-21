# Fine-tuning on site — the field pipeline

**Who this is for:** you are at a customer, the counts or the masks are not as good as
they should be, and you want to fix it there and then by labelling a few images.

**What it does:** you label 3–5 frames, run one command, and get a better model —
**and it refuses to give you one that is worse anywhere else.** That refusal is the
whole point of this pipeline, not a side feature.

Everything runs from this folder. Every path is relative; nothing outside is needed
except the Python environment.

---

## 0. The loop in one picture

**One labelling pass, on the full frame, in one LabelMe session:** draw the
`good_column` / `cropped_column` polygons, then one `edge` polygon per sheet inside each
good column. Then one command:

```
   you                                    the pipeline
   ───                                    ────────────
   label 3-5 full frames:         ──►  stage 1  fine-tune the COLUMN model, gate it
   columns AND edges in one go          derive the straightened crops + their edge
                                        labels from the same frames (no round-trip)
                                   ──►  stage 3  fine-tune the EDGE model, gate it
                                                 -> promoted, or refused with a reason
```

The pipeline straightens each good column exactly as production does and carries your
edge polygons through the identical geometry, so the crop labels are your frame labels
seen through the straightener (verified to 0.15 px).

The older two-pass route still works if you only label columns: stage 2 then writes
the straightened columns to `<folder>/edges_to_label/` for you to label, and you re-run
with `--stage edges`.

---

## 1. Before you leave the office

```bash
pip install -r requirements.txt
```

Python 3.13, and install torch for your CUDA version from https://pytorch.org first.

**Check it runs — do this before you travel, not on site:**

```bash
python -m edges.train --name smoke --smoke --workers 2
```

```bash
python -m columns.train --name smoke --smoke --workers 2
```

Both should print a line like `train 150 crops ... val 50 crops` and finish in a minute
or two. Delete `runs/edges_smoke` and `runs/columns_smoke` afterwards.

**You need an NVIDIA GPU to train.** Inference works on CPU; training does not, in any
useful sense. If the site machine has no GPU, collect and label there, and run the
training when you get back — the labelled folder is only a few MB.

**Check what is live:**

```bash
python -m field.registry
```

---

## 2. On site: which images to label

This is the decision that matters most. Labelling images the model already gets right
buys nothing — the pipeline will tell you so and refuse to train.

**Find the failing ones without labelling anything.** The rack camera is fixed and the
stack does not move during a burst, so every frame of one session must give the same
count for the same column position. Any disagreement is an error:

```bash
python -m field.consistency 'path/to/crops/*.png'
```

```
   session  col  frames  count  unstable   range
    090000    0       7     24         0   24-24
    090000    1       7     23         3   22-24  <-- unstable
```

Label the columns that come back unstable. If you have no crops yet, run stage 1 and 2
first with whatever frames you have, then use this on the crops it produces.

**How many.** Measured on this dataset:

| what we measured | result |
|---|---|
| a type the model had never seen, given 4 crops | 1/80 → **76/80** |
| type2 given 4 / 16 / 37 crops | 37/54 → 50/54 → 51/54 |
| crops you get per photo | type2 ~1, type10 ~3.5, type3 ~4.5 |

So 3–5 frames genuinely moves the needle. **Spread them across different stacks and
different times of day rather than taking five frames of one bundle** — five photos of
one stack teach the model that stack. If you can only get one stack, that is still
worth doing, just expect less.

---

## 3. Folder layout

Make a folder for the visit. Put the frames you labelled in `frames/`:

```
site_acme_20260920/
    frames/
        20260920_101500_type10_-1_28pieces_001.jpg
        20260920_101500_type10_-1_28pieces_001.json     <- LabelMe, column polygons
        ...
```

Label with **LabelMe**, on the full frame. Three labels:

* `good_column` — a column fully visible in the frame
* `cropped_column` — a column cut off by the frame edge or an occlusion
* `edge` — one polygon per visible sheet edge, drawn INSIDE a good column. Draw every
  sheet of every good column you label; a good column with no edges gets a crop but no
  label. Match the existing band style for the type (§5).

Filenames matter. The pipeline reads the **session** (second `_`-separated field) and
the **type** from the name, and it uses the session to decide what is a near-duplicate
and what must be held out together. Keep the existing convention:

```
<date>_<sessiontime>_<type>_<variant>_<index>.jpg
20260920_101500_type10_-1_28pieces_001.jpg
```

If your camera software names files differently, rename them to this pattern before
ingesting. If the type is not in the name, pass `--type type10` on the command line.

---

## 4. Run it (combined labels: everything in one go)

**Always look before you leap:**

```bash
python -m field.adapt --site acme --from site_acme_20260920 --dry-run
```

This ingests, checks your labels, and prints the current model's scores. It trains
nothing. Read the output, then run for real:

```bash
python -m field.adapt --site acme --from site_acme_20260920
```

It will:

1. copy your frames into `data/columns/site_acme/`
2. derive the straightened crops AND their edge labels into `data/edges/<type>/`
   (skipped for frames without `edge` polygons -- those get PNGs to label later)
3. fine-tune the column model from the live one, gate it, promote or refuse (§6)
4. fine-tune the edge model on the derived crops, gate it, promote or refuse

If none of your frames carry `edge` polygons it stops after the column stage and writes
`edges_to_label/` for the two-pass route instead.

Expect roughly 10–20 minutes on a desktop GPU, longer on a laptop.

**Near-duplicate frames are skipped on purpose.** If you gave it three near-identical
frames from one burst you may get crops from only one or two of them — the camera is
fixed, so the others would just be the same column labelled twice.

---

## 5. Edge labelling style (and the two-pass route)

Whether you draw edges on the frame (combined) or on the PNGs in `edges_to_label/type10/`
(two-pass), draw **one polygon per visible sheet edge**, label `edge`. For the two-pass
route, leave the `.json` files next to the PNGs and re-run with `--stage edges`.

**Match the existing labelling style.** Draw the polygon over the same part of the
sheet as the existing labels for that type; do not switch between thin lines and thick
bands. The pipeline measures this and will stop you if your labels are much thinner or
thicker than the type's existing ones (§6). If you are unsure, open a few crops from
`data/edges/type10/` in LabelMe and copy that style.

Then:

```bash
python -m field.adapt --site acme --from site_acme_20260920 --stage edges
```

---

## 6. Reading the result — the three outcomes

Every stage ends in one of these. **Nothing is ever promoted silently.**

### `PROMOTED GLOBALLY`

The new model is better on your frames and no worse than the measured noise on every
set that was already working. It is now live. The previous one is kept.

### `PROMOTED FOR SITE acme ONLY`

The new model regressed something — but something this site does not run, so it cannot
hurt you. It is registered for this site only; the global model is untouched. Perfectly
normal outcome, not a failure.

### `REFUSED`

The new model regressed a type this site *does* run. Nothing changed; you still have
the old model. The table above the verdict says exactly what broke. Options: label a
few more frames and try again, or bring the folder back to the office.

### The gate table

```
set             n  before  after  delta   status
all_type10     76      76     76     +0   ok (tol 1)
all_type2      54      51     50     -1   ok (tol 1)
all_type3      72      70     70     +0   ok (tol 1)
old_t10       122      61     54     -7   not ranked (seed spread 23/122 -- cannot rank)
old_t3        300     297    297     +0   ok (tol 10)
```

`delta` is crops gained or lost. The tolerances are **measured**, not chosen: we trained
the identical recipe under two random seeds and recorded how much the score moved on its
own. A drop inside that band is indistinguishable from luck. `old_t10` swings 23 crops
of 122 between seeds, so it is reported and deliberately **not** used to decide.

The column gate works the same way on `inst_iou` (mask quality — it feeds the
straightener, so it matters here), `count_acc` and `cls_acc`.

---

## 7. Messages that are NOT errors

**`NOTHING TO DO: the live model already counts all N new crops correctly`**
You labelled crops that already worked. Nothing to learn, and retraining could only
spend accuracy elsewhere. Use `field.consistency` (§2) to find the ones that are
actually failing.

**`SKIPPING old_t3 / old_t10: no crops under ...`**
Two extra evaluation sets that live outside this folder. The gate runs fine without
them; you just have two fewer rows.

**`<stem>: near-duplicate of a kept frame in session ..., skipped`**
Working as intended (§4).

**`<stem>: already in data/columns, skipped`**
You are re-ingesting a frame that is already in the dataset. If it is genuinely a new
photo, rename it — a repeated name would give one frame two conflicting split
decisions.

**`CONVENTION MISMATCH ... coverage 12% vs type10 median 54%`**
Your edge polygons are much thinner or thicker than the existing labels for that type.
This is a real risk: labels in a different style can damage types that work today. Fix
the labels. Only use `--force` if you are certain the new style is the correct one.

---

## 8. Undo

```bash
python -m field.registry rollback edges
```

```bash
python -m field.registry rollback columns
```

Puts the previous model back immediately. Nothing is deleted — every candidate stays
under `runs/` so it can be inspected later.

---

## 9. What to bring back

The whole visit folder. It is small (a few MB) and it contains the labelled frames, the
generated crops, and the edge labels. Also bring:

* `runs/_active.json` — what ended up live
* `runs/_adapt_edges_*.json` — the full before/after for each attempt

At the office, the same labels can be run through a `full` bundle, which also gates the
close-up camera domain and can therefore promote globally.

---

## 10. Making a smaller folder to carry

If you do not want to copy the whole pipeline:

```bash
python -m field.bundle --out D:\site_bundle --profile site
```

| profile | MB | drops | can promote globally? |
|---|---|---|---|
| `full` | 400 | nothing | yes — identical to the office |
| `lean` | 283 | `legacy_fg` | yes, but the column training mix differs |
| `site` | **149** | `legacy_fg` + `closeup` | **no — site-scoped models only** |

The `site` profile keeps the **entire** edge side intact — same training pool, same
gate, same numbers. Only the column side shrinks, and what it loses is the close-up
camera domain: those frames are neither replayed nor scored, so drift away from that
domain would be invisible. That is why it is restricted to site-scoped models.

The bundle is self-contained and carries a `SITE_README.md`.

---

## 11. Command reference

| command | does |
|---|---|
| `python -m field.adapt --site S --from F` | stages 1 + 2 |
| `python -m field.adapt --site S --from F --stage edges` | stage 3 |
| `python -m field.adapt ... --dry-run` | ingest + checks + baselines, trains nothing |
| `python -m field.adapt ... --force` | proceed despite a convention mismatch |
| `python -m field.adapt ... --type type10` | force the type when the filename lacks it |
| `python -m field.consistency '<glob>'` | find failing crops without labels |
| `python -m field.crops --frames D --out D2` | just regenerate crops, no training |
| `python -m field.registry` | what is live |
| `python -m field.registry rollback edges\|columns` | undo |
| `python -m field.bundle --out D --profile site` | build a carryable folder |
| `python -m tools.report_counts <run>...` | full count report for any run |

Tuning knobs on `field.adapt`, rarely needed: `--steps` / `--lr` (edge fine-tune,
default 2500 / 1e-4), `--column-steps` / `--column-lr` (default 1500 / 5e-5),
`--boost-factor` (how hard your new frames are oversampled, default 12),
`--workers` (dataloader processes, default 8; lower it on a small machine).

---

## 12. Known limits — read this before promising anything

* **The improvement number on your own frames is not accuracy.** With 3–5 frames there
  is no honest held-out set, so those frames are training data. The *regression* side of
  the gate is honest; the improvement side is evidence that the adaptation took.
* **The gate cannot protect what it cannot see.** In the `site` profile that means the
  close-up camera domain. In any profile it means types with no labelled data here.
* **Adding data really does break other types.** Measured: growing one type from 0 to 59
  crops cost another 3 crops of 54, monotonically; and a model trained on two types
  scored 1/80 on a third where a model trained on only one of them scored 72/80. The
  mechanism is not understood. This is why the gate exists — treat a refusal as the
  system working.
* **The pinned validation set is saturated** and is deliberately not used for gating.
  Nine of twelve study runs scored a perfect 50/50 on it while differing a lot elsewhere.
* **Same seed does not mean same result** (cuDNN nondeterminism). Re-running an identical
  fine-tune can land a crop or two either way. That is exactly what the tolerances absorb.
* **Not yet exercised on a real failure.** Every rehearsal so far started from a model
  that already counted the new crops correctly, so the "genuine improvement" branch of
  stage 3 has been tested for plumbing but never on a real site regression. Expect to
  find rough edges the first time; keep the folder and the logs.

---

## 13. If something goes wrong

Logs for every training run started by `field.adapt` are written next to the runs:

```
runs/_adapt_columns_<timestamp>.log
runs/_adapt_edges_<timestamp>.log
runs/_adapt_edges_<timestamp>.json      <- the full before/after table
```

Nothing the pipeline does is destructive: it copies data in, writes new runs, and
changes one pointer file (`runs/_active.json`). If the state gets confusing, roll back
(§8), or delete `runs/_active.json` to return to the shipped defaults.

Deeper background — why the counter works the way it does, the grow-from-centre mask
readout, and every measured number quoted here — is in `STATE.md` and
`EXPERIMENTS_DATA_SCALING.md`.
