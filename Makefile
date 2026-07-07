.PHONY: smoke sft-base rm arm-sft arm-dpo arm-ppo arm-grpo eval plots all

smoke:
	bash scripts/run_smoke.sh

sft-base:
	bash scripts/stage1_sft_base.sh

rm:
	bash scripts/stage2_rm.sh

arm-sft:
	bash scripts/stage3a_arm_sft.sh

arm-dpo:
	bash scripts/stage3b_arm_dpo.sh

arm-ppo:
	bash scripts/stage3c_arm_ppo.sh

arm-grpo:
	bash scripts/stage3d_arm_grpo.sh

eval:
	bash scripts/stage4_eval.sh

plots:
	python analysis/aggregate.py results
	python analysis/frontier_plot.py results
	python analysis/diagnostics.py results

all:
	bash scripts/run_all.sh
