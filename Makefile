.PHONY: smoke all \
	sft-base rm arm-sft arm-dpo arm-ppo arm-grpo eval plots helpful-all \
	math-sft-base math-pairs math-arm-sft math-arm-dpo math-arm-grpo math-eval math-plots math-all

smoke:
	bash scripts/run_smoke.sh

all:
	bash scripts/run_all.sh

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
	python analysis/aggregate.py results/helpful
	python analysis/frontier_plot.py results/helpful
	python analysis/diagnostics.py results/helpful
	python analysis/diversity_plot.py results/helpful
	python analysis/retention_plot.py results/helpful
	python analysis/pref_logps_plot.py results/helpful

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
	python analysis/aggregate.py results/math
	python analysis/frontier_math.py results/math
	python analysis/diversity_plot.py results/math
	python analysis/retention_plot.py results/math
	python analysis/pref_logps_plot.py results/math

math-all:
	bash scripts/run_all_math.sh
