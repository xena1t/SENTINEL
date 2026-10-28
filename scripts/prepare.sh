#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PYTHON=${PYTHON:-python3}
RUNS=${RUNS:-$ROOT/runs}
MODELS=${MODELS:-$ROOT/models/model-paths.json}
VERSION=${1:?Usage: bash scripts/prepare.sh v1 or v2}
case "$VERSION" in v1|v2) ;; *) echo 'Expected v1 or v2' >&2; exit 2 ;; esac
mkdir -p "$RUNS/$VERSION"
if [ "$VERSION" = v2 ]; then
    "$PYTHON" scripts/windows/fetch_window_media.py --report "$RUNS/$VERSION/media.json"
    "$PYTHON" tests/check_composite_dataset.py "$ROOT"
    "$PYTHON" scripts/labels/build_composite_dataset.py "$ROOT"
    "$PYTHON" tests/check_composite_dataset.py "$ROOT"
    "$PYTHON" tests/check_window_composite_adapter.py "$ROOT"
else
    "$PYTHON" scripts/setup/download_clip_media.py --report "$RUNS/$VERSION/media.json"
fi
"$PYTHON" scripts/train/prepare_rf.py --root "$ROOT" --output "$RUNS/$VERSION/rf-cache"
"$PYTHON" tests/check_rf_pipeline.py "$ROOT" "$RUNS/$VERSION/rf-cache"
"$PYTHON" scripts/train/train_rf.py --root "$ROOT" --cache "$RUNS/$VERSION/rf-cache" --output "$RUNS/$VERSION/rf-overfit" --mode overfit --epochs 300
"$PYTHON" scripts/train/train_rf.py --root "$ROOT" --cache "$RUNS/$VERSION/rf-cache" --output "$RUNS/$VERSION/rf-full" --mode full --epochs 80 --gate "$RUNS/$VERSION/rf-overfit/result.json"
EXTRACTOR=extract_fusion_features.py
if [ "$VERSION" = v2 ]; then EXTRACTOR=extract_window_features.py; fi
"$PYTHON" "scripts/train/$EXTRACTOR" --root "$ROOT" --rf-cache "$RUNS/$VERSION/rf-cache" --rf-checkpoint "$RUNS/$VERSION/rf-full/best.pt" --models "$MODELS" --output "$RUNS/$VERSION/features"
if [ "$VERSION" = v1 ]; then
    "$PYTHON" tests/check_fusion_features.py "$ROOT" "$RUNS/$VERSION/features"
fi
