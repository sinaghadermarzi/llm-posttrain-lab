#!/usr/bin/env bash
# Math arm DPO: one run per beta in configs/math/sweeps.yaml, parallel, 1 GPU each.
# Single-process ZeRO-2 for fp32 master weights, same as the other arms
# (see scripts/helpful/stage3b_arm_dpo.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."
NGPU="${NGPU:-4}"
PORT_BASE="${PORT_BASE:-29500}"   # each run gets its own rendezvous port

mapfile -t OVERRIDES < <(python -m posttrain_lab.config math dpo)
pids=(); names=(); failed=()
reap() {  # wait on each launched run individually; a bare `wait` always returns 0
  for k in "${!pids[@]}"; do
    wait "${pids[$k]}" || failed+=("${names[$k]}")
  done
  pids=(); names=()
}

i=0
for ov in "${OVERRIDES[@]}"; do
  gpu=$((i % NGPU))
  log="logs_math_dpo_${ov//[.=\/]/_}.log"
  echo "[math-dpo] launching $ov on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu accelerate launch \
    --config_file configs/accelerate/zero2_1gpu.yaml \
    --main_process_port $((PORT_BASE + i)) \
    --module posttrain_lab.train.dpo configs/math/m3_arm_dpo.yaml "$ov" "$@" \
    > "$log" 2>&1 &
  pids+=("$!"); names+=("$ov ($log)")
  i=$((i + 1))
  if (( i % NGPU == 0 )); then reap; fi
done
reap

if (( ${#failed[@]} )); then
  printf '[math-dpo] FAILED: %s\n' "${failed[@]}" >&2
  exit 1
fi
echo "[math-dpo] all runs finished"
