.PHONY: preflight prefetch smoke all \
	sft-base rm arm-sft arm-dpo arm-ppo arm-grpo eval plots helpful-all \
	math-sft-base math-pairs math-arm-sft math-arm-dpo math-arm-grpo math-eval math-plots math-all

# node readiness: GPUs, NCCL, libs, HF assets, MLflow (see notebooks/onboarding.ipynb)
preflight:
	python -m posttrain_lab.preflight

# on a machine WITH internet: download every HF asset into PREFETCH_DIR, then
# verify it loads offline; copy that dir to a PVC used as HF_HOME (see README)
PREFETCH_DIR ?= hf_cache
PREFETCH_ARGS ?=
prefetch:
	python scripts/prefetch_assets.py $(PREFETCH_DIR) $(PREFETCH_ARGS)

smoke:
	bash scripts/run_smoke.sh

all:
	bash scripts/run_all.sh

# results_root per task, resolved by the configs (RESULTS_DIR in the env or
# .env, default results/). Recursive '=' so Python only runs when plotting.
HELPFUL_RESULTS = $(shell python -m posttrain_lab.config --results-root helpful)
MATH_RESULTS = $(shell python -m posttrain_lab.config --results-root math)

# ---- helpful track (UltraFeedback, learned reward) ----

sft-base:
	bash scripts/helpful/stage1_sft_base.sh

rm:
	bash scripts/helpful/stage2_rm.sh

arm-sft:
	bash scripts/helpful/stage3a_arm_sft.sh

arm-dpo:
	bash scripts/helpful/stage3b_arm_dpo.sh

arm-ppo:
	bash scripts/helpful/stage3c_arm_ppo.sh

arm-grpo:
	bash scripts/helpful/stage3d_arm_grpo.sh

eval:
	bash scripts/helpful/stage4_eval.sh

plots:
	python analysis/aggregate.py "$(HELPFUL_RESULTS)"
	python analysis/frontier_plot.py "$(HELPFUL_RESULTS)"
	python analysis/diagnostics.py "$(HELPFUL_RESULTS)"
	python analysis/diversity_plot.py "$(HELPFUL_RESULTS)"
	python analysis/retention_plot.py "$(HELPFUL_RESULTS)"
	python analysis/pref_logps_plot.py "$(HELPFUL_RESULTS)"

helpful-all:
	bash scripts/run_all_helpful.sh

# ---- math track (GSM8K, verifiable reward) ----

math-sft-base:
	bash scripts/math/m1_sft_gold.sh

math-pairs:
	bash scripts/math/m2_gen_pairs.sh

math-arm-sft:
	bash scripts/math/m3a_arm_sft.sh

math-arm-dpo:
	bash scripts/math/m3b_arm_dpo.sh

math-arm-grpo:
	bash scripts/math/m3d_arm_grpo.sh

math-eval:
	bash scripts/math/m4_eval.sh

math-plots:
	python analysis/aggregate.py "$(MATH_RESULTS)"
	python analysis/frontier_math.py "$(MATH_RESULTS)"
	python analysis/diversity_plot.py "$(MATH_RESULTS)"
	python analysis/retention_plot.py "$(MATH_RESULTS)"
	python analysis/pref_logps_plot.py "$(MATH_RESULTS)"

math-all:
	bash scripts/run_all_math.sh
