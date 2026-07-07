#!/usr/bin/env bash
# Stage 0: end-to-end smoke test of the WHOLE pipeline on Qwen2.5-0.5B with
# tiny slices (configs/smoke.yaml). One GPU (or CPU, slowly). Target < 30 min
# on a single A100. Run this before burning GPU-days — it exercises every
# stage interface against the pinned library versions.
set -euo pipefail
cd "$(dirname "$0")/.."
export SMOKE=1
export TRL_EXPERIMENTAL_SILENCE=1

echo "=== smoke 0/8: data loaders"
python -m posttrain_lab.data

echo "=== smoke 1/8: SFT base (pi_SFT)"
python -m posttrain_lab.train.sft configs/stage1_sft_base.yaml

echo "=== smoke 2/8: reward model"
python -m posttrain_lab.train.rm configs/stage2_rm.yaml

echo "=== smoke 3/8: SFT arm"
python -m posttrain_lab.train.sft configs/arm_sft.yaml

echo "=== smoke 4/8: DPO arm (one beta)"
python -m posttrain_lab.train.dpo configs/arm_dpo.yaml train.beta=0.1

echo "=== smoke 5/8: PPO arm (one kl_coef)"
python -m posttrain_lab.train.ppo configs/arm_ppo.yaml train.kl_coef=0.05

echo "=== smoke 6/8: GRPO arm (one beta)"
python -m posttrain_lab.train.grpo configs/arm_grpo.yaml train.beta=0.04

echo "=== smoke 7/8: eval all checkpoints"
python -m posttrain_lab.eval.run_eval

echo "=== smoke 8/8: aggregate + figures"
python analysis/aggregate.py results/smoke
python analysis/frontier_plot.py results/smoke
python analysis/diagnostics.py results/smoke

echo "SMOKE OK — figures under results/smoke/figures/"
