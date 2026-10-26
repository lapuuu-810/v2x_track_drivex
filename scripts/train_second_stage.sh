#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT/third_party/OpenPCDet/tools"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
  conda run -n eccv-v2x-submission --no-capture-output \
  python -m torch.distributed.run --nproc_per_node=8 train.py \
  --launcher pytorch \
  --cfg_file "$PROJECT/configs/second_urbantwin.yaml" \
  --batch_size 16 --epochs 56 --workers 8 --use_amp --fix_random_seed \
  --extra_tag sim_only_v1
