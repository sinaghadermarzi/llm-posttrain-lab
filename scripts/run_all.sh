#!/usr/bin/env bash
# Both tracks, in order. Each track is independently idempotent; run the
# per-track scripts directly if you only want one.
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/run_all_helpful.sh
bash scripts/run_all_math.sh
