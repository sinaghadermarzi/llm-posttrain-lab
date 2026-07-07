#!/usr/bin/env bash
# Math m4: evaluate every math checkpoint (generate -> KL -> verify -> metric
# groups), sharded across GPUs. Incremental: re-run any time; only missing
# metric groups are computed.
set -euo pipefail
cd "$(dirname "$0")/../.."
NGPU="${NGPU:-4}"

for i in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES=$i python -m posttrain_lab.eval.run_eval task=math --shard "$i/$NGPU" "$@" \
    > "logs_math_eval_shard$i.log" 2>&1 &
done
wait
echo "[math-eval] all shards finished"
