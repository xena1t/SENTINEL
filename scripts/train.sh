#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PYTHON=${PYTHON:-python3}
RUNS=${RUNS:-$ROOT/runs}
MODELS=${MODELS:-$ROOT/models/model-paths.json}
VERSION=${1:?Usage: bash scripts/train.sh v1 or v2}
case "$VERSION" in v1|v2) ;; *) echo 'Expected v1 or v2' >&2; exit 2 ;; esac
OUTPUT=$RUNS/$VERSION
COMMON=(--root "$ROOT" --features "$OUTPUT/features")
ALIGNER=train_alignment.py
RECIPE=(--freeze-fusion)
if [ "$VERSION" = v2 ]; then
    "$PYTHON" scripts/train/train_window_fusion.py "${COMMON[@]}" --output "$OUTPUT/fusion-overfit" --mode overfit --epochs 300 --dropout 0 --lr 1e-3
    "$PYTHON" scripts/train/train_window_fusion.py "${COMMON[@]}" --output "$OUTPUT/fusion" --mode full --dropout .5
    ALIGNER=train_window_alignment.py
    RECIPE=(--draws 2000 --composite-share .5)
else
    "$PYTHON" scripts/train/train_fusion.py "${COMMON[@]}" --output "$OUTPUT/fusion" --epochs 60 --dropout .5
fi
COMMON+=(--fusion-checkpoint "$OUTPUT/fusion/best.pt" --models "$MODELS")
"$PYTHON" "scripts/train/$ALIGNER" "${COMMON[@]}" "${RECIPE[@]}" --output "$OUTPUT/alignment-smoke" --mode smoke --epochs 1 --accumulation 1
"$PYTHON" "scripts/train/$ALIGNER" "${COMMON[@]}" "${RECIPE[@]}" --output "$OUTPUT/alignment-overfit" --mode overfit --epochs 80 --lr 1e-4 --accumulation 4 --warmup-fraction 0
for ARM in control dropout; do
    RATE=0
    if [ "$ARM" = dropout ]; then RATE=.5; fi
    "$PYTHON" "scripts/train/$ALIGNER" "${COMMON[@]}" "${RECIPE[@]}" --output "$OUTPUT/alignment-$ARM" --mode full --gate "$OUTPUT/alignment-overfit/result.json" \
        --epochs 20 --patience 8 --lr 3e-5 --accumulation 8 --warmup-fraction .1 --modality-dropout "$RATE"
done
