#!/usr/bin/env bash
# Arm DPO+NLL: the DPO sweep with the NLL term on chosen responses
# (configs/helpful/arm_dpo_nll.yaml), one run per beta, parallel, 1 GPU each.
# Logs: logs_dponll_<override>.log (not logs_dpo_*: reset --stage 3b globs those).
# Stage key 3e, not 3c: PPO and GRPO keep 3c/3d (status, reset --stage), though it runs right after DPO.
set -euo pipefail
CFG=configs/helpful/arm_dpo_nll.yaml SWEEP=dpo_nll TAG=dponll exec bash "$(dirname "$0")/stage3b_arm_dpo.sh" "$@"
