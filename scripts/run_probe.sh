#!/usr/bin/env bash
# Scale probe of the helpful track: the real pipeline (run_all_helpful.sh, every
# launcher, 4 GPUs) at full scale for a few steps per stage (configs/probe.yaml),
# under <RESULTS_DIR>/probe/helpful. It catches what the 0.5B smoke can't before
# GPU-days go in: OOMs at 1.5B with the real batch sizes and lengths (PPO holds
# four models), saves under ZeRO-2, the 8B judge next to vLLM. Under an hour.
#
#   make helpful-probe          (start a GPU logger first, make gpu-log, for peak memory)
#
# Starts from an empty probe tree every time. The stage scripts log into the repo
# root under the real run's log names, so the probe moves its logs into the probe
# tree when it ends (and refuses to start if the real run's logs are there).
set -euo pipefail
cd "$(dirname "$0")/.."
unset SMOKE
export PROBE=1

ROOT=$(python -m posttrain_lab.config --results-root helpful)
case "$ROOT" in
  */probe/helpful) ;;
  *) echo "[probe] unexpected results root $ROOT: not touching it" >&2; exit 1 ;;
esac
if pgrep -f "run_all_helpful.sh" >/dev/null 2>&1; then
  echo "[probe] the helpful track is running (run_all_helpful.sh): probe before launching it, not alongside" >&2
  exit 1
fi
for f in logs_dpo_*.log logs_eval_shard*.log; do
  if [ -e "$f" ]; then
    echo "[probe] $f exists: the real helpful run's logs would be overwritten (make reset TASK=helpful first)" >&2
    exit 1
  fi
done

if [ -e "$ROOT" ]; then
  echo "[probe] removing the previous probe tree $ROOT"
  rm -rf "$ROOT"
fi
park_logs() {
  mkdir -p "$ROOT/logs"
  for f in logs_dpo_*.log logs_eval_shard*.log; do
    if [ -e "$f" ]; then mv "$f" "$ROOT/logs/"; fi
  done
}
trap park_logs EXIT

START=$(date +%s)
echo "[probe] full-scale probe of the helpful track -> $ROOT"
bash scripts/run_all_helpful.sh
echo "=== probe summary ($(( ($(date +%s) - START) / 60 )) min)"
python -m posttrain_lab.probe_report "$ROOT" --since "$START"
