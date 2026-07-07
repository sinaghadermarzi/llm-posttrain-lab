#!/usr/bin/env bash
# Math/RLVR track, full pipeline in order. Idempotent like the helpful track.
# Expected wall-clock on 4x A100 80GB: ~12-15 h (GRPO sweep dominates).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== math m1: pi_SFT-math (gold CoT)"
bash scripts/math/m1_sft_gold.sh

echo "=== math m2: self-generated preference pairs"
bash scripts/math/m2_gen_pairs.sh

echo "=== math m3a: SFT arm"
bash scripts/math/m3a_arm_sft.sh

echo "=== math m3b: DPO sweep (parallel, 1 GPU each)"
bash scripts/math/m3b_arm_dpo.sh

echo "=== math m3d: GRPO sweep (sequential, full node, verifier reward)"
bash scripts/math/m3d_arm_grpo.sh

echo "=== math m4: evaluate all checkpoints"
bash scripts/math/m4_eval.sh

echo "=== math m5: aggregate + figures"
python analysis/aggregate.py results/math
python analysis/frontier_math.py results/math
python analysis/diversity_plot.py results/math
python analysis/retention_plot.py results/math
python analysis/pref_logps_plot.py results/math

echo "MATH TRACK DONE — see results/math/figures/"
