"""GRPO arm — online RL from pi_SFT with the stage-2 reward model.

Group-relative advantages, no value model; beta is the KL coefficient to the
frozen reference (pi_SFT). reward_funcs takes the RM checkpoint path and TRL
loads it as a sequence-classification reward model.

Usage: python -m posttrain_lab.train.grpo [configs/arm_grpo.yaml] train.beta=0.04 ...
"""

from __future__ import annotations

from trl import GRPOConfig, GRPOTrainer

from posttrain_lab.config import load_config, train_kwargs
from posttrain_lab.data import prompt_dataset
from posttrain_lab.train.common import begin_run, init_path, load_tokenizer, parse_entry
from posttrain_lab.utils.checkpoints import is_done, mark_done


def main() -> None:
    stage_yaml, overrides = parse_entry("arm_grpo.yaml")
    cfg = load_config(stage_yaml, overrides)
    if is_done(cfg.output_dir):
        print(f"[grpo] {cfg.output_dir} already DONE, skipping")
        return
    begin_run(cfg)

    src = init_path(cfg)
    tokenizer = load_tokenizer(src)

    args = GRPOConfig(
        output_dir=str(cfg.output_dir),
        run_name=str(cfg.run_name),
        seed=int(cfg.seed),
        **train_kwargs(cfg, GRPOConfig),
    )
    trainer = GRPOTrainer(
        model=src,
        reward_funcs=str(cfg.reward_model_path),
        args=args,
        train_dataset=prompt_dataset(cfg),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))
    mark_done(cfg.output_dir)


if __name__ == "__main__":
    main()
