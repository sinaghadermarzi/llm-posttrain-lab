#!/usr/bin/env bash
# Arm DPO: one run per beta in sweeps.yaml, launched in parallel, one GPU each.
# Runs under single-process ZeRO-2 (not plain Trainer) so DPO gets the same
# fp32 master weights/optimizer state as every other arm — weights are loaded
# in bf16, and plain-Trainer AdamW at lr 5e-7 would round most updates away.
# stage3e_arm_dpo_nll.sh runs the DPO+NLL arm through this script: CFG / SWEEP /
# TAG pick its config, grid and log names.
CFG="${CFG:-configs/helpful/arm_dpo.yaml}"
SWEEP="${SWEEP:-dpo}"
TAG="${TAG:-dpo}"   # logs_${TAG}_<override>.log (posttrain_lab/status.py TRACKS)
set -euo pipefail
cd "$(dirname "$0")/../.."
NGPU="${NGPU:-4}"
PORT_BASE="${PORT_BASE:-29500}"   # each run gets its own rendezvous port

OVERRIDES=()  # (a read loop, not mapfile: also runs under macOS bash 3.2, e.g. the tests)
while IFS= read -r ov; do OVERRIDES+=("$ov"); done < <(python -m posttrain_lab.config helpful "$SWEEP")
(( ${#OVERRIDES[@]} )) || { echo "[$TAG] no sweep values for $SWEEP" >&2; exit 1; }
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
  log="logs_${TAG}_${ov//[.=\/]/_}.log"
  echo "[$TAG] launching $ov on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu accelerate launch \
    --config_file configs/accelerate/zero2_1gpu.yaml \
    --main_process_port $((PORT_BASE + i)) \
    --module posttrain_lab.train.dpo "$CFG" "$ov" "$@" \
    > "$log" 2>&1 &
  pids+=("$!"); names+=("$ov ($log)")
  i=$((i + 1))
  # if more sweep values than GPUs, wait for the batch to finish
  if (( i % NGPU == 0 )); then reap; fi
done
reap

if (( ${#failed[@]} )); then
  printf "[$TAG] FAILED: %s\n" "${failed[@]}" >&2
  exit 1
fi
echo "[$TAG] all runs finished"
