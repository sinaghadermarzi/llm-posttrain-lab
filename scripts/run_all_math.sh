#!/usr/bin/env bash
# Math/RLVR track, full pipeline in order. Idempotent like the helpful track.
# Expected wall-clock on 4x A100 80GB: ~12-15 h (GRPO sweep dominates).
# One training seed per call: SEED=18 bash scripts/run_all_math.sh (make
# math-all SEED=18) trains and evaluates seed 18 next to the other seeds' runs;
# the m5 figures aggregate every seed present (README "More seeds").
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/math/seed.sh

ROOT=$(python -m posttrain_lab.config --results-root math)  # RESULTS_DIR-aware

echo "=== math m1: pi_SFT-math (gold CoT) [seed $SEED]"
bash scripts/math/m1_sft_gold.sh
# stop before hours of sampling if generation would not stop at end of turn
python -m posttrain_lab.utils.chat "$ROOT/runs/sft-base_seed$SEED" --probe configs/math/m1_sft_gold.yaml

echo "=== math m2: self-generated preference pairs [seed $SEED]"
bash scripts/math/m2_gen_pairs.sh

echo "=== math m3a: SFT arm [seed $SEED]"
bash scripts/math/m3a_arm_sft.sh

echo "=== math m3b: DPO sweep (parallel, 1 GPU each) [seed $SEED]"
bash scripts/math/m3b_arm_dpo.sh

echo "=== math m3c: DPO+NLL sweep (parallel, 1 GPU each) [seed $SEED]"
bash scripts/math/m3c_arm_dpo_nll.sh

echo "=== math m3d: GRPO sweep (sequential, full node, verifier reward) [seed $SEED]"
bash scripts/math/m3d_arm_grpo.sh

echo "=== math m4: evaluate all checkpoints [seed $SEED]"
bash scripts/math/m4_eval.sh

echo "=== math m5: aggregate + figures [all seeds]"
python analysis/aggregate.py "$ROOT"
python analysis/frontier_math.py "$ROOT"
python analysis/diversity_plot.py "$ROOT"
python analysis/retention_plot.py "$ROOT"
python analysis/pref_logps_plot.py "$ROOT"
python analysis/diagnostics.py "$ROOT"     # dpo_logps.png: DPO vs DPO+NLL log-probs in training

echo "MATH TRACK DONE [seed $SEED] — see $ROOT/figures/"
