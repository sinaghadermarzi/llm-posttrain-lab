# Sourced by every math stage script: which training seed to run.
#   make math-all SEED=18     (or SEED=18 bash scripts/run_all_math.sh)
# Each stage passes seed=$SEED to its config, and every seed-dependent path
# carries it (runs/<arm>..._seed<N>, data/self_pairs_seed<N>, pi_SFT), so a
# second seed trains next to the first and is never skipped as DONE because
# of the first one's outputs. DEFAULT_SEED must match configs/base.yaml
# (tests/test_seeds.py checks it).
DEFAULT_SEED=17
SEED="${SEED:-$DEFAULT_SEED}"
case "$SEED" in
  ''|*[!0-9]*) echo "SEED must be a non-negative integer, got '$SEED'" >&2; exit 2 ;;
esac
export SEED
# per-run log names: the default seed keeps the original names, other seeds
# add _seed<N> (posttrain_lab/status.py:log_name)
if [ "$SEED" = "$DEFAULT_SEED" ]; then SEED_LOG=""; else SEED_LOG="_seed$SEED"; fi
