#!/bin/bash
# Score one model combination on EVERY check, the way every candidate since 2026-11-04 was compared:
#   bash tools/candidate_check.sh <name> <column .pt|shipped> [<edge_a .pt|shipped> <edge_b .pt|shipped>]
# Exports to runs/_candidates/<name>/ (a complete models folder) and prints: October pinned field frames,
# type8 holdout, held-out test (+ 5-frame vote), unseen type2, column-free images, stress rows, type1.
# Run from training_pipeline/ (python = the Windows interpreter with torch + onnxruntime).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python.exe}
N=$1; M=runs/_candidates/$N; mkdir -p $M
if [ "$2" = shipped ]; then cp inference/models/column_flow.onnx $M/; else $PY -m common.export_onnx $2 $M/column_flow.onnx >/dev/null; fi
for s in a b; do v=${3:-shipped}; [ $s = b ] && v=${4:-shipped}
  if [ "$v" = shipped ]; then cp inference/models/sheet_edges_$s.onnx $M/; else $PY -m common.export_onnx $v $M/sheet_edges_$s.onnx >/dev/null; fi
done
md5sum $M/*.onnx
set +e
echo "== $N : pinned October set (58 frames, never trained on)"
$PY -m tools.oct_field_count --models $M --name pin_$N --pin data/oct26/pinned_test_frames.json 2>&1 | grep -v Warn | tail -6
echo "== $N : type8 holdout (018, 021)"
$PY -m tools.column_field_eval --models $M --sets type8 --frames-dir data/type8_autolabel/frames --only 39pieces_018,39pieces_021 2>&1 | grep "^type8"
echo "== $N : held-out test (old) + 5-frame vote"
$PY -m tools.evaluate --models $M --vote 5 --out runs/_candidates/${N}_evaluate 2>&1 | grep -E "^(type|all)"
echo "== $N : unseen type2"
$PY -m tools.unseen_check --models $M --name $N 2>&1 | grep -E "^type2 +columns"
echo "== $N : column-free images"
$PY -m tools.negative_check --models $M 2>&1 | grep -v Warn
echo "== $N : stress"
$PY -m tools.stress --models $M --name $N --only "clean,exposure_-2.5EV,exposure_-2EV,exposure_+1.5EV,exposure_+2EV,darker_x0.6,brighter_x1.4,zoom_0.6,zoom_0.75,rotate_-10,rotate_+10,rotate_-20,rotate_+20,pitch_10%,pitch_-10%,pitch_20%,yaw_10%,yaw_20%,blur_k3,noise_s10" 2>&1 | grep -v Warn
echo "== $N : type1 (June, unlabelled, count vs file name)"
$PY -m tools.type1_check --models $M --name $N 2>&1 | grep -v Warn
