#!/usr/bin/env bash
# Arm DPO: one run per beta in sweeps.yaml, launched in parallel, one GPU each.
# Runs under single-process ZeRO-2 (not plain Trainer) so DPO gets the same
# fp32 master weights/optimizer state as every other arm — weights are loaded
# in bf16, and plain-Trainer AdamW at lr 5e-7 would round most updates away.
set -euo pipefail
cd "$(dirname "$0")/../.."
NGPU="${NGPU:-4}"
PORT_BASE="${PORT_BASE:-29500}"   # each run gets its own rendezvous port

mapfile -t OVERRIDES < <(python -m posttrain_lab.config helpful dpo)
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
  log="logs_dpo_${ov//[.=\/]/_}.log"
  echo "[dpo] launching $ov on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu accelerate launch \
    --config_file configs/accelerate/zero2_1gpu.yaml \
    --main_process_port $((PORT_BASE + i)) \
    --module posttrain_lab.train.dpo configs/helpful/arm_dpo.yaml "$ov" "$@" \
    > "$log" 2>&1 &
  pids+=("$!"); names+=("$ov ($log)")
  i=$((i + 1))
  # if more sweep values than GPUs, wait for the batch to finish
  if (( i % NGPU == 0 )); then reap; fi
done
reap

if (( ${#failed[@]} )); then
  printf '[dpo] FAILED: %s\n' "${failed[@]}" >&2
  exit 1
fi
echo "[dpo] all runs finished"
