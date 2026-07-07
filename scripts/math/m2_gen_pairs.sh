#!/usr/bin/env bash
# Math m2: self-generated preference pairs from pi_SFT-math (vLLM, 1 GPU).
# Inspect the printed stats (and stats.json) before launching DPO.
set -euo pipefail
cd "$(dirname "$0")/../.."
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python -m posttrain_lab.gen_pairs configs/math/m2_gen_pairs.yaml "$@"
