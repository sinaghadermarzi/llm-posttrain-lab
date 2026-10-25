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
from posttrain_lab.train.common import (align_eos, begin_run, init_path, load_tokenizer, model_kwargs, parse_entry,
                                         save_final_state)
from posttrain_lab.utils.checkpoints import is_done, mark_done


class DeepSpeedSafePPOTrainer(PPOTrainer):
    """PPOTrainer whose checkpoints save under DeepSpeed.

    TRL 1.7's PPOTrainer.save_model points self.deepspeed at the bare policy
    and then calls transformers' Trainer.save_model, which in transformers 5.13
    reads self.deepspeed.config as DeepSpeed's config dict: every PPO
    checkpoint under DeepSpeed crashed with "'Qwen2Config' object has no
    attribute 'get'" (make helpful-probe, 2026-10-25; smoke runs PPO without
    DeepSpeed, so it never got there). Under ZeRO-2, which every config uses,
    each rank holds the full weights, so the main process saves the policy's
    own state dict, which is what save_model meant to write.
    """

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        if not self.is_deepspeed_enabled:
            super().save_model(output_dir, _internal_call)
            return
        if self.accelerator.state.deepspeed_plugin.zero_stage == 3:
            raise NotImplementedError("PPO checkpoints under ZeRO-3 need the weights gathered first; "
                                      "configs/accelerate uses ZeRO-2")
        if not self.args.should_save:
            return
        policy = self.accelerator.unwrap_model(self.model).policy
        backup = self.model
        self.model = policy  # Trainer._save calls save_pretrained on self.model
        try:
            self._save(output_dir, state_dict=policy.state_dict())
        finally:
            self.model = backup


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
    align_eos(policy, tokenizer)
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
            max_length=max_prompt_tokens, return_dict=False,  # transformers 5 defaults to a dict
        )
        return {"input_ids": ids}

    dataset = prompt_dataset(cfg).map(tokenize, remove_columns=["prompt"])

    args = PPOConfig(
        output_dir=str(cfg.output_dir),
        run_name=str(cfg.run_name),
        seed=int(cfg.seed),
        stop_token="eos",  # the tokenizer's eos: the end-of-turn token (load_tokenizer)
        **train_kwargs(cfg, PPOConfig),
    )
    trainer = DeepSpeedSafePPOTrainer(
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
    save_final_state(trainer)
    tokenizer.save_pretrained(str(cfg.output_dir))
    mark_done(cfg.output_dir)


if __name__ == "__main__":
    main()
