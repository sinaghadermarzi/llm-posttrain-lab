#!/usr/bin/env bash
# Arm GRPO: sequential 4-GPU runs over the beta grid, vLLM colocated.
set -euo pipefail
cd "$(dirname "$0")/.."

while IFS= read -r ov; do
  echo "[grpo] running $ov"
  accelerate launch --config_file configs/accelerate/zero2_4gpu.yaml \
    --module posttrain_lab.train.grpo configs/arm_grpo.yaml "$ov" "$@"
done < <(python -m posttrain_lab.config grpo)
echo "[grpo] all runs finished"
