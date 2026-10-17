#!/usr/bin/env bash
# Stage 4: evaluate every checkpoint (generate -> KL -> dual reward scores),
# sharded across GPUs. Already-evaluated checkpoints are skipped, so this is
# safe to re-run any time new checkpoints appear.
set -euo pipefail
cd "$(dirname "$0")/../.."
NGPU="${NGPU:-4}"

pids=(); failed=()
for i in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES=$i python -m posttrain_lab.eval.run_eval task=helpful --shard "$i/$NGPU" "$@" \
    > "logs_eval_shard$i.log" 2>&1 &
  pids+=("$!")
done
# wait on each shard individually; a bare `wait` always returns 0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || failed+=("shard $i (logs_eval_shard$i.log)")
done

if (( ${#failed[@]} )); then
  printf '[eval] FAILED: %s\n' "${failed[@]}" >&2
  exit 1
fi
echo "[eval] all shards finished"
