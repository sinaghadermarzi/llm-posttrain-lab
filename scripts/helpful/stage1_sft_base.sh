#!/usr/bin/env bash
# Stage 1: build pi_SFT (shared starting point + KL reference) on 4 GPUs.
set -euo pipefail
cd "$(dirname "$0")/../.."
accelerate launch --config_file configs/accelerate/zero2_4gpu.yaml \
  --module posttrain_lab.train.sft configs/helpful/stage1_sft_base.yaml "$@"
