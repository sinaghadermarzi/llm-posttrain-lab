#!/usr/bin/env bash
# Stage 0: end-to-end smoke test of BOTH tracks on Qwen2.5-0.5B with tiny
# slices (configs/smoke.yaml). One GPU (or CPU, slowly). Target < 30 min on
# a single A100. Run this before burning GPU-days — it exercises every stage
# interface, both reward paths, and every eval metric group against the
# pinned library versions.
set -euo pipefail
cd "$(dirname "$0")/.."
export SMOKE=1
export TRL_EXPERIMENTAL_SILENCE=1

echo "=== smoke helpful 0/9: data loaders"
python -m posttrain_lab.data

echo "=== smoke helpful 1/9: SFT base (pi_SFT)"
python -m posttrain_lab.train.sft configs/helpful/stage1_sft_base.yaml

echo "=== smoke helpful 2/9: reward model"
python -m posttrain_lab.train.rm configs/helpful/stage2_rm.yaml

echo "=== smoke helpful 3/9: SFT arm"
python -m posttrain_lab.train.sft configs/helpful/arm_sft.yaml

echo "=== smoke helpful 4/9: DPO arm (one beta)"
python -m posttrain_lab.train.dpo configs/helpful/arm_dpo.yaml train.beta=0.1

echo "=== smoke helpful 5/9: PPO arm (one kl_coef)"
python -m posttrain_lab.train.ppo configs/helpful/arm_ppo.yaml train.kl_coef=0.05

echo "=== smoke helpful 6/9: GRPO arm (one beta)"
python -m posttrain_lab.train.grpo configs/helpful/arm_grpo.yaml train.beta=0.04

echo "=== smoke helpful 7/9: eval all checkpoints, all metric groups"
python -m posttrain_lab.eval.run_eval

echo "=== smoke helpful 8/9: eval re-run must skip (incremental-merge check)"
python -m posttrain_lab.eval.run_eval | tee /dev/stderr | grep -q "skipping"

echo "=== smoke helpful 9/9: aggregate + figures"
python analysis/aggregate.py results/smoke/helpful
python analysis/frontier_plot.py results/smoke/helpful
python analysis/diagnostics.py results/smoke/helpful
python analysis/diversity_plot.py results/smoke/helpful
python analysis/retention_plot.py results/smoke/helpful
python analysis/pref_logps_plot.py results/smoke/helpful

echo "=== smoke math 0/6: data loaders + verifier sanity"
python -m posttrain_lab.data task=math

echo "=== smoke math 1/6: SFT base on gold CoT (pi_SFT-math)"
python -m posttrain_lab.train.sft configs/math/m1_sft_gold.yaml

echo "=== smoke math 2/6: self-generated preference pairs"
python -m posttrain_lab.gen_pairs configs/math/m2_gen_pairs.yaml

echo "=== smoke math 3/6: DPO arm on self-gen pairs (one beta)"
python -m posttrain_lab.train.dpo configs/math/m3_arm_dpo.yaml train.beta=0.1

echo "=== smoke math 4/6: GRPO arm with verifier reward (one beta)"
python -m posttrain_lab.train.grpo configs/math/m3_arm_grpo.yaml train.beta=0.04

echo "=== smoke math 5/6: eval all checkpoints, all metric groups"
python -m posttrain_lab.eval.run_eval task=math

echo "=== smoke math 6/6: aggregate + figures"
python analysis/aggregate.py results/smoke/math
python analysis/frontier_math.py results/smoke/math
python analysis/diversity_plot.py results/smoke/math
python analysis/retention_plot.py results/smoke/math
python analysis/pref_logps_plot.py results/smoke/math

echo "SMOKE OK — figures under results/smoke/{helpful,math}/figures/"
