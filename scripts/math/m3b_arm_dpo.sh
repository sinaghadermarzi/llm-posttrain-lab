#!/usr/bin/env bash
# Math arm DPO: one run per beta in configs/math/sweeps.yaml, parallel, 1 GPU each.
set -euo pipefail
cd "$(dirname "$0")/../.."
NGPU="${NGPU:-4}"

mapfile -t OVERRIDES < <(python -m posttrain_lab.config math dpo)
i=0
for ov in "${OVERRIDES[@]}"; do
  gpu=$((i % NGPU))
  echo "[math-dpo] launching $ov on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu accelerate launch \
    --config_file configs/accelerate/single_gpu.yaml \
    --module posttrain_lab.train.dpo configs/math/m3_arm_dpo.yaml "$ov" "$@" \
    > "logs_math_dpo_${ov//[.=\/]/_}.log" 2>&1 &
  i=$((i + 1))
  if (( i % NGPU == 0 )); then wait; fi
done
wait
echo "[math-dpo] all runs finished"
