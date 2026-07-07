#!/usr/bin/env bash
# Stage 2: train the reward model. Exits nonzero if pairwise accuracy is
# below the gate in stage2_rm.yaml — do not proceed to the RL arms past that.
set -euo pipefail
cd "$(dirname "$0")/.."
accelerate launch --config_file configs/accelerate/zero2_4gpu.yaml \
  --module posttrain_lab.train.rm configs/helpful/stage2_rm.yaml "$@"
