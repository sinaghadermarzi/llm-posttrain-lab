#!/usr/bin/env bash
# Full pipeline in order. Idempotent: every training run writes a DONE marker
# and skips itself on re-run; eval skips checkpoints that already have rows.
# Safe to re-run after a preemption or crash — it continues where it stopped.
#
# Expected wall-clock on 4x A100 80GB: ~2.5-3 days (PPO sweep dominates).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== stage 1: pi_SFT"
bash scripts/helpful/stage1_sft_base.sh

echo "=== stage 2: reward model (gated on pairwise accuracy)"
bash scripts/helpful/stage2_rm.sh

echo "=== stage 3a: SFT arm"
bash scripts/helpful/stage3a_arm_sft.sh

echo "=== stage 3b: DPO sweep (parallel, 1 GPU each)"
bash scripts/helpful/stage3b_arm_dpo.sh

echo "=== stage 3c: PPO sweep (sequential, full node each)"
bash scripts/helpful/stage3c_arm_ppo.sh

echo "=== stage 3d: GRPO sweep (sequential, full node each)"
bash scripts/helpful/stage3d_arm_grpo.sh

echo "=== stage 4: evaluate all checkpoints"
bash scripts/helpful/stage4_eval.sh

echo "=== stage 5: aggregate + figures"
python analysis/aggregate.py results/helpful
python analysis/frontier_plot.py results/helpful
python analysis/diagnostics.py results/helpful
python analysis/diversity_plot.py results/helpful
python analysis/retention_plot.py results/helpful
python analysis/pref_logps_plot.py results/helpful

echo "ALL DONE — see results/helpful/figures/frontier.png"
