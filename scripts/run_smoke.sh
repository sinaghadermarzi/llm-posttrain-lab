#!/usr/bin/env bash
# Stage 0: end-to-end smoke test of BOTH tracks on Qwen2.5-0.5B with tiny
# slices (configs/smoke.yaml). One GPU (or CPU, slowly). Target < 30 min on
# a single A100. Run this before burning GPU-days — it exercises every stage
# interface, both reward paths, and every eval metric group against the
# pinned library versions.
set -euo pipefail
cd "$(dirname "$0")/.."
export SMOKE=1
export TRL_EXPERIMENTAL_SILENCE=1

# One GPU only. These stages run as plain `python -m` (no accelerate), so with
# several GPUs visible the HF Trainer wraps the model in nn.DataParallel. TRL
# 1.7's SFTTrainer binds its own model.forward to the original model on cuda:0
# (the chunked-CE patch), so the replica on cuda:1 fails with "Expected all
# tensors to be on the same device". Keep the first visible GPU (default 0);
# an explicitly empty CUDA_VISIBLE_DEVICES still means CPU.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
echo "[smoke] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<empty: CPU>}"

echo "=== smoke helpful 0/9: data loaders"
python -m posttrain_lab.data

echo "=== smoke helpful 1/9: SFT base (pi_SFT)"
python -m posttrain_lab.train.sft configs/helpful/stage1_sft_base.yaml
# generation must stop at the chat template's end of turn (utils/chat.py)
python -m posttrain_lab.utils.chat "$(python -m posttrain_lab.config --results-root helpful)/runs/sft-base_seed17" \
  --probe configs/helpful/stage1_sft_base.yaml --report-only  # a few smoke steps cannot learn to stop; just exercise it

echo "=== smoke helpful 2/9: reward model"
python -m posttrain_lab.train.rm configs/helpful/stage2_rm.yaml

echo "=== smoke helpful 3/9: SFT arm"
python -m posttrain_lab.train.sft configs/helpful/arm_sft.yaml

echo "=== smoke helpful 4/9: DPO arm (one beta)"
python -m posttrain_lab.train.dpo configs/helpful/arm_dpo.yaml train.beta=0.1

echo "=== smoke helpful 5/9: PPO arm (one kl_coef)"
python -m posttrain_lab.train.ppo configs/helpful/arm_ppo.yaml train.kl_coef=0.05

echo "=== smoke helpful 6/9: GRPO arm (one beta)"
python -m posttrain_lab.train.grpo configs/helpful/arm_grpo.yaml train.beta=0.04

echo "=== smoke helpful 7/9: eval all checkpoints, all metric groups"
python -m posttrain_lab.eval.run_eval

echo "=== smoke helpful 8/9: eval re-run must skip (incremental-merge check)"
python -m posttrain_lab.eval.run_eval | tee /dev/stderr | grep -q "skipping"

echo "=== smoke helpful 9/9: aggregate + figures"
ROOT=$(python -m posttrain_lab.config --results-root helpful)  # SMOKE=1 -> <RESULTS_DIR>/smoke/helpful
python -m posttrain_lab.utils.chat "$ROOT"/runs/*/   # every saved run stops at end of turn
python analysis/aggregate.py "$ROOT"
python analysis/frontier_plot.py "$ROOT"
python analysis/diagnostics.py "$ROOT"
python analysis/diversity_plot.py "$ROOT"
python analysis/retention_plot.py "$ROOT"
python analysis/pref_logps_plot.py "$ROOT"

echo "=== smoke math 0/6: data loaders + verifier sanity"
python -m posttrain_lab.data task=math

echo "=== smoke math 1/6: SFT base on gold CoT (pi_SFT-math)"
python -m posttrain_lab.train.sft configs/math/m1_sft_gold.yaml
python -m posttrain_lab.utils.chat "$(python -m posttrain_lab.config --results-root math)/runs/sft-base_seed17" \
  --probe configs/math/m1_sft_gold.yaml --report-only  # a few smoke steps cannot learn to stop; just exercise it

echo "=== smoke math 2/6: self-generated preference pairs"
python -m posttrain_lab.gen_pairs configs/math/m2_gen_pairs.yaml

echo "=== smoke math 3/6: DPO and DPO+NLL arms on self-gen pairs (one beta each)"
python -m posttrain_lab.train.dpo configs/math/m3_arm_dpo.yaml train.beta=0.1
python -m posttrain_lab.train.dpo configs/math/m3_arm_dpo_nll.yaml train.beta=0.1

echo "=== smoke math 4/6: GRPO arm with verifier reward (one beta)"
python -m posttrain_lab.train.grpo configs/math/m3_arm_grpo.yaml train.beta=0.04

echo "=== smoke math 5/6: eval all checkpoints, all metric groups"
python -m posttrain_lab.eval.run_eval task=math

echo "=== smoke math 6/6: aggregate + figures"
ROOT=$(python -m posttrain_lab.config --results-root math)  # SMOKE=1 -> <RESULTS_DIR>/smoke/math
python -m posttrain_lab.utils.chat "$ROOT"/runs/*/
python analysis/aggregate.py "$ROOT"
python analysis/frontier_math.py "$ROOT"
python analysis/diversity_plot.py "$ROOT"
python analysis/retention_plot.py "$ROOT"
python analysis/pref_logps_plot.py "$ROOT"
python analysis/diagnostics.py "$ROOT"

echo "SMOKE OK — figures under $(dirname "$ROOT")/{helpful,math}/figures/"
