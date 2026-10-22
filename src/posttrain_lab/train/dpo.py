"""DPO arm — offline preference optimization from pi_SFT.

ref_model=None makes DPOTrainer snapshot the initial policy as the frozen
reference, which IS pi_SFT here — exactly the reference the eval KL is
measured against. DPOTrainer logs logps/chosen and logps/rejected every
logging step; analysis/diagnostics.py plots their joint decline.

Usage: python -m posttrain_lab.train.dpo [configs/arm_dpo.yaml] train.beta=0.1 ...
"""

from __future__ import annotations

from transformers import AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

from posttrain_lab.config import load_config, train_kwargs
from posttrain_lab.data import pref_dataset
from posttrain_lab.train.common import (align_eos, begin_run, init_path, load_tokenizer, model_kwargs, parse_entry,
                                         save_final_state)
from posttrain_lab.utils.checkpoints import is_done, mark_done


def main() -> None:
    stage_yaml, overrides = parse_entry("helpful/arm_dpo.yaml")
    cfg = load_config(stage_yaml, overrides)
    if is_done(cfg.output_dir):
        print(f"[dpo] {cfg.output_dir} already DONE, skipping")
        return
    begin_run(cfg)

    src = init_path(cfg)
    tokenizer = load_tokenizer(src)
    model = AutoModelForCausalLM.from_pretrained(src, **model_kwargs(cfg))
    align_eos(model, tokenizer)

    args = DPOConfig(
        output_dir=str(cfg.output_dir),
        run_name=str(cfg.run_name),
        seed=int(cfg.seed),
        **train_kwargs(cfg, DPOConfig),
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # snapshot of initial policy = pi_SFT
        args=args,
        train_dataset=pref_dataset(cfg),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(cfg.output_dir))
    save_final_state(trainer)
    tokenizer.save_pretrained(str(cfg.output_dir))
    mark_done(cfg.output_dir)


if __name__ == "__main__":
    main()
