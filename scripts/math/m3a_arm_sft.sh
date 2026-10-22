#!/usr/bin/env bash
# Math arm SFT: continued imitation on gold solutions, 4 GPUs, single run.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/math/seed.sh
accelerate launch --config_file configs/accelerate/zero2_4gpu.yaml \
  --module posttrain_lab.train.sft configs/math/m3_arm_sft.yaml "seed=$SEED" "$@"
