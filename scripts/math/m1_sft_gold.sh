#!/usr/bin/env bash
# Math m1: build pi_SFT-math on GSM8K gold chain-of-thought, 4 GPUs.
set -euo pipefail
cd "$(dirname "$0")/../.."
accelerate launch --config_file configs/accelerate/zero2_4gpu.yaml \
  --module posttrain_lab.train.sft configs/math/m1_sft_gold.yaml "$@"
