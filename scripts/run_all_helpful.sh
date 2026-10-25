#!/usr/bin/env bash
# Full pipeline in order. Idempotent: every training run writes a DONE marker
# and skips itself on re-run; eval skips checkpoints that already have rows.
# Safe to re-run after a preemption or crash — it continues where it stopped.
#
# Expected wall-clock on 4x A100 80GB: ~2.5-3 days (PPO sweep dominates).
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT=$(python -m posttrain_lab.config --results-root helpful)  # RESULTS_DIR-aware
# PROBE=1 (scripts/run_probe.sh): 4 steps of pi_SFT can't learn to stop, so report the stop probe without gating
REPORT_ONLY=""
if [ "${PROBE:-}" = 1 ]; then REPORT_ONLY="--report-only"; fi

echo "=== stage 1: pi_SFT"
bash scripts/helpful/stage1_sft_base.sh
# stop before the RL arms if generation would not stop at end of turn
python -m posttrain_lab.utils.chat "$ROOT/runs/sft-base_seed17" --probe configs/helpful/stage1_sft_base.yaml $REPORT_ONLY

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
python analysis/aggregate.py "$ROOT"
python analysis/frontier_plot.py "$ROOT"
python analysis/diagnostics.py "$ROOT"
python analysis/diversity_plot.py "$ROOT"
python analysis/retention_plot.py "$ROOT"
python analysis/pref_logps_plot.py "$ROOT"

echo "ALL DONE — see $ROOT/figures/frontier.png"
