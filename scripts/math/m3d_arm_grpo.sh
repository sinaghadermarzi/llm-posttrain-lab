#!/usr/bin/env bash
# Math arm GRPO: sequential 4-GPU runs over the beta grid, verifier reward,
# vLLM colocated.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/math/seed.sh

while IFS= read -r ov; do
  echo "[math-grpo] running $ov (seed $SEED)"
  accelerate launch --config_file configs/accelerate/zero2_4gpu.yaml \
    --module posttrain_lab.train.grpo configs/math/m3_arm_grpo.yaml "$ov" "seed=$SEED" "$@"
done < <(python -m posttrain_lab.config math grpo)
echo "[math-grpo] all runs finished"
