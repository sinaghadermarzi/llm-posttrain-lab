#!/usr/bin/env bash
# Math arm DPO+NLL: the DPO sweep with the NLL term on chosen answers
# (configs/math/m3_arm_dpo_nll.yaml), one run per beta, parallel, 1 GPU each.
# Logs: logs_math_dponll_<override>.log (not logs_math_dpo_*: reset --stage m3b
# globs those).
set -euo pipefail
CFG=configs/math/m3_arm_dpo_nll.yaml SWEEP=dpo_nll TAG=dponll exec bash "$(dirname "$0")/m3b_arm_dpo.sh" "$@"
