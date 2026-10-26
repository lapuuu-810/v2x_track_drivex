#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$SCRIPT_DIR")"
KIT="$PROJECT/competition/starting_kit"
CKPT="${1:?usage: $0 CHECKPOINT [DECLARATION]}"
DECLARATION="${2:-}"

conda run -n eccv-v2x-submission python "$PROJECT/src/predict_submission.py" \
  --config "$PROJECT/configs/second_urbantwin.yaml" \
  --checkpoint "$CKPT" \
  --frames "$KIT/detection_test_frames" \
  --frame-ids "$KIT/detection_test_frame_ids.json" \
  --output "$PROJECT/outputs/predictions.json"

DECL_ARGS=()
if [[ -n "$DECLARATION" ]]; then
  DECL_ARGS=(--declaration "$DECLARATION")
else
  DECL_ARGS=(--allow-unsigned-draft)
fi
conda run -n eccv-v2x-submission python "$PROJECT/src/build_submission.py" \
  --synthetic-points "$PROJECT/data/openpcdet_custom/points" \
  --predictions "$PROJECT/outputs/predictions.json" \
  --frame-ids "$KIT/detection_test_frame_ids.json" \
  --output "$PROJECT/submissions/submission_v1.zip" \
  "${DECL_ARGS[@]}"
