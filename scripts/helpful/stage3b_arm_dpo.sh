#!/usr/bin/env bash
# Arm DPO: one run per beta in sweeps.yaml, launched in parallel, one GPU each.
set -euo pipefail
cd "$(dirname "$0")/../.."
NGPU="${NGPU:-4}"

mapfile -t OVERRIDES < <(python -m posttrain_lab.config helpful dpo)
i=0
for ov in "${OVERRIDES[@]}"; do
  gpu=$((i % NGPU))
  echo "[dpo] launching $ov on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu accelerate launch \
    --config_file configs/accelerate/single_gpu.yaml \
    --module posttrain_lab.train.dpo configs/helpful/arm_dpo.yaml "$ov" "$@" \
    > "logs_dpo_${ov//[.=\/]/_}.log" 2>&1 &
  i=$((i + 1))
  # if more sweep values than GPUs, wait for a slot
  if (( i % NGPU == 0 )); then wait; fi
done
wait
echo "[dpo] all runs finished"
