"""SFT training — serves both stage 1 (build pi_SFT from the base model) and
the SFT arm (continued imitation on chosen responses), selected by config.

Usage:
    python -m posttrain_lab.train.sft configs/stage1_sft_base.yaml [k=v ...]
    python -m posttrain_lab.train.sft configs/arm_sft.yaml [k=v ...]
"""

from __future__ import annotations

from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from posttrain_lab.config import load_config, train_kwargs
from posttrain_lab.data import sft_dataset
from posttrain_lab.train.common import (
    align_eos,
    begin_run,
    init_path,
    init_untrained_chat_tokens,
    load_tokenizer,
    model_kwargs,
    parse_entry,
    save_final_state,
)
from posttrain_lab.utils.checkpoints import is_done, mark_done


def main() -> None:
    stage_yaml, overrides = parse_entry("helpful/stage1_sft_base.yaml")
    cfg = load_config(stage_yaml, overrides)
    if is_done(cfg.output_dir):
        print(f"[sft] {cfg.output_dir} already DONE, skipping")
        return
    begin_run(cfg)

    src = init_path(cfg)
    tokenizer = load_tokenizer(src)
    model = AutoModelForCausalLM.from_pretrained(src, **model_kwargs(cfg))
    # before align_eos: the model's own eos is the row the end-of-turn token copies
    eos = model.config.eos_token_id
    init_untrained_chat_tokens(model, tokenizer, eos[0] if isinstance(eos, list) else eos)
    align_eos(model, tokenizer)

    args = SFTConfig(
        output_dir=str(cfg.output_dir),
        run_name=str(cfg.run_name),
        seed=int(cfg.seed),
        **train_kwargs(cfg, SFTConfig),
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=sft_dataset(cfg),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(cfg.output_dir))
    save_final_state(trainer)
    tokenizer.save_pretrained(str(cfg.output_dir))
    mark_done(cfg.output_dir)


if __name__ == "__main__":
    main()
