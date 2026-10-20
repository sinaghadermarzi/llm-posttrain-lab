"""GRPO arm — online RL from pi_SFT, both tracks.

Group-relative advantages, no value model; beta is the KL coefficient to the
frozen reference (pi_SFT). The reward dispatches on config:
  reward_model_path set   -> RM checkpoint path, TRL loads it as a
                             sequence-classification reward model (helpful)
  reward: verifier        -> tasks/gsm8k.verifier_reward callable; the gold
                             answer reaches it via the dataset's extra
                             'answer' column (math / RLVR)

Usage: python -m posttrain_lab.train.grpo [configs/helpful/arm_grpo.yaml] train.beta=0.04 ...
"""

from __future__ import annotations

from trl import GRPOConfig, GRPOTrainer

from posttrain_lab.config import load_config, train_kwargs
from posttrain_lab.data import prompt_dataset
from posttrain_lab.train.common import begin_run, init_path, load_tokenizer, parse_entry
from posttrain_lab.utils.checkpoints import is_done, mark_done


def rollout_generation_kwargs(tokenizer, use_vllm: bool, extra: dict | None = None) -> dict:
    """Stop rollouts at the tokenizer's eos (the end-of-turn token, see
    load_tokenizer) and only there: TRL 1.7 masks completions at
    processing_class.eos_token_id, so a different stop id would leave its
    padded tail counted as completion tokens. vLLM gets it explicitly (it
    otherwise stops only at the checkpoint's own eos ids); TRL's HF path
    already passes eos_token_id=tokenizer.eos_token_id."""
    out = dict(extra or {})
    if use_vllm:
        out["stop_token_ids"] = [tokenizer.eos_token_id]
    return out


def main() -> None:
    stage_yaml, overrides = parse_entry("helpful/arm_grpo.yaml")
    cfg = load_config(stage_yaml, overrides)
    if is_done(cfg.output_dir):
        print(f"[grpo] {cfg.output_dir} already DONE, skipping")
        return
    begin_run(cfg)

    src = init_path(cfg)
    tokenizer = load_tokenizer(src)

    if cfg.get("reward") == "verifier":
        from posttrain_lab.tasks.gsm8k import verifier_reward

        reward_funcs = verifier_reward
    else:
        reward_funcs = str(cfg.reward_model_path)

    kwargs = train_kwargs(cfg, GRPOConfig)
    kwargs["generation_kwargs"] = rollout_generation_kwargs(tokenizer, bool(kwargs.get("use_vllm")),
                                                            kwargs.get("generation_kwargs"))
    args = GRPOConfig(
        output_dir=str(cfg.output_dir),
        run_name=str(cfg.run_name),
        seed=int(cfg.seed),
        **kwargs,
    )
    trainer = GRPOTrainer(
        model=src,
        reward_funcs=reward_funcs,
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
