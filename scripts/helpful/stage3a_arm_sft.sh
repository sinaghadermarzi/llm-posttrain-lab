#!/usr/bin/env bash
# Arm SFT: continued imitation on chosen responses, 4 GPUs, single run.
set -euo pipefail
cd "$(dirname "$0")/.."
accelerate launch --config_file configs/accelerate/zero2_4gpu.yaml \
  --module posttrain_lab.train.sft configs/helpful/arm_sft.yaml "$@"
