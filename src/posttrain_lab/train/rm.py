"""Stage 2: reward model — scalar head on pi_SFT, trained on preference pairs.

Aborts (nonzero exit) if pairwise accuracy on test_prefs is below
cfg.min_accuracy: a weak RM would make every downstream RL result
uninterpretable, so fail loudly here.

Usage: python -m posttrain_lab.train.rm [configs/stage2_rm.yaml] [k=v ...]
"""

from __future__ import annotations

import sys

from transformers import AutoModelForSequenceClassification
from trl import RewardConfig, RewardTrainer

from posttrain_lab.config import load_config, train_kwargs
from posttrain_lab.data import pref_dataset
from posttrain_lab.train.common import begin_run, init_path, load_tokenizer, model_kwargs, parse_entry, save_final_state
from posttrain_lab.utils.checkpoints import is_done, mark_done
from posttrain_lab.utils.logging import append_jsonl


def main() -> None:
    stage_yaml, overrides = parse_entry("helpful/stage2_rm.yaml")
    cfg = load_config(stage_yaml, overrides)
    if is_done(cfg.output_dir):
        print(f"[rm] {cfg.output_dir} already DONE, skipping")
        return
    begin_run(cfg)

    src = init_path(cfg)
    tokenizer = load_tokenizer(src)
    model = AutoModelForSequenceClassification.from_pretrained(
        src, num_labels=1, **model_kwargs(cfg)
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    args = RewardConfig(
        output_dir=str(cfg.output_dir),
        run_name=str(cfg.run_name),
        seed=int(cfg.seed),
        **train_kwargs(cfg, RewardConfig),
    )
    trainer = RewardTrainer(
        model=model,
        args=args,
        train_dataset=pref_dataset(cfg),
        eval_dataset=pref_dataset(cfg, split=str(cfg.data.eval_split)),
        processing_class=tokenizer,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(f"[rm] eval metrics: {metrics}")
    trainer.save_model(str(cfg.output_dir))
    save_final_state(trainer)
    tokenizer.save_pretrained(str(cfg.output_dir))
    append_jsonl(f"{cfg.output_dir}/rm_eval.jsonl", metrics)

    accuracy = metrics.get("eval_accuracy")
    if accuracy is not None and accuracy < float(cfg.min_accuracy):
        print(
            f"[rm] FATAL: pairwise accuracy {accuracy:.3f} < gate {cfg.min_accuracy} — "
            "RL results would be uninterpretable with this RM.",
            file=sys.stderr,
        )
        sys.exit(1)
    mark_done(cfg.output_dir)


if __name__ == "__main__":
    main()
