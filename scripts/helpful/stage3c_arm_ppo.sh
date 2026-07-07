#!/usr/bin/env bash
# Arm PPO: sequential 4-GPU runs over the kl_coef grid (4 models resident;
# a full node per run). Drop kl_coef=0.1 from sweeps.yaml first if over budget.
set -euo pipefail
cd "$(dirname "$0")/../.."

while IFS= read -r ov; do
  echo "[ppo] running $ov"
  accelerate launch --config_file configs/accelerate/zero2_4gpu.yaml \
    --module posttrain_lab.train.ppo configs/helpful/arm_ppo.yaml "$ov" "$@"
done < <(python -m posttrain_lab.config helpful ppo)
echo "[ppo] all runs finished"
