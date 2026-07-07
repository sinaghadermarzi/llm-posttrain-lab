"""PPO arm — online RL from pi_SFT with the stage-2 reward model.

Four models are resident (policy, frozen ref = pi_SFT, reward, value); the
value model is warm-started from the reward model, standard practice since
InstructGPT. PPOTrainer consumes a pre-tokenized prompt dataset (input_ids),
left-padded for generation.

PPOTrainer lives in trl.experimental.ppo as of TRL 1.x (it was flagged
experimental after the 0.x rewrite); import defensively so a future
re-promotion to trl top-level doesn't break this script.

Usage: python -m posttrain_lab.train.ppo [configs/arm_ppo.yaml] train.kl_coef=0.05 ...
"""

from __future__ import annotations

import os

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

try:
    from trl import PPOConfig, PPOTrainer
except ImportError:
    from trl.experimental.ppo import PPOConfig, PPOTrainer

from posttrain_lab.config import load_config, train_kwargs
from posttrain_lab.data import prompt_dataset
from posttrain_lab.train.common import begin_run, init_path, load_tokenizer, model_kwargs, parse_entry
from posttrain_lab.utils.checkpoints import is_done, mark_done


def main() -> None:
    stage_yaml, overrides = parse_entry("helpful/arm_ppo.yaml")
    cfg = load_config(stage_yaml, overrides)
    if is_done(cfg.output_dir):
        print(f"[ppo] {cfg.output_dir} already DONE, skipping")
        return
    begin_run(cfg)

    src = init_path(cfg)
    rm_path = str(cfg.reward_model_path)
    tokenizer = load_tokenizer(src)
    tokenizer.padding_side = "left"

    policy = AutoModelForCausalLM.from_pretrained(src, **model_kwargs(cfg))
    ref = AutoModelForCausalLM.from_pretrained(src, **model_kwargs(cfg))
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        rm_path, num_labels=1, **model_kwargs(cfg)
    )
    value_model = AutoModelForSequenceClassification.from_pretrained(
        rm_path, num_labels=1, **model_kwargs(cfg)
    )
    for m in (reward_model, value_model):
        m.config.pad_token_id = tokenizer.pad_token_id

    max_prompt_tokens = int(cfg.data.max_prompt_length)

    def tokenize(ex):
        ids = tokenizer.apply_chat_template(
            ex["prompt"], add_generation_prompt=True, truncation=True,
            max_length=max_prompt_tokens,
        )
        return {"input_ids": ids}

    dataset = prompt_dataset(cfg).map(tokenize, remove_columns=["prompt"])

    args = PPOConfig(
        output_dir=str(cfg.output_dir),
        run_name=str(cfg.run_name),
        seed=int(cfg.seed),
        stop_token="eos",
        **train_kwargs(cfg, PPOConfig),
    )
    trainer = PPOTrainer(
        args=args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref,
        reward_model=reward_model,
        train_dataset=dataset,
        value_model=value_model,
    )
    trainer.train()
    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))
    mark_done(cfg.output_dir)


if __name__ == "__main__":
    main()
