"""UltraFeedback loaders — the single source of truth for every stage's data.

HuggingFaceH4/ultrafeedback_binarized columns:
  prompt (str), prompt_id, messages (list[{role,content}]),
  chosen (list[{role,content}] — full [user, assistant] conversation),
  rejected (same shape), score_chosen, score_rejected

TRL 1.x consumes conversational formats directly, so loaders return message
lists and let each trainer apply the tokenizer's chat template. All
subsampling is deterministic in (dataset, split, seed, max_samples).
"""

from __future__ import annotations

from datasets import Dataset, load_dataset
from omegaconf import DictConfig


def _load_split(cfg: DictConfig, split: str, max_samples: int | None) -> Dataset:
    ds = load_dataset(cfg.dataset.id, split=split)
    if max_samples is not None and max_samples < len(ds):
        ds = ds.shuffle(seed=cfg.seed).select(range(max_samples))
    return ds


def sft_dataset(cfg: DictConfig) -> Dataset:
    """Conversational SFT data: a ``messages`` column.

    Stage 1 uses the train_sft split as-is. The SFT *arm* sets
    data.use_chosen_as_sft=true to imitate the chosen responses of the
    preference split — preference alignment by pure imitation.
    """
    ds = _load_split(cfg, cfg.data.split, cfg.data.get("max_samples"))
    if cfg.data.get("use_chosen_as_sft", False):
        ds = ds.map(lambda ex: {"messages": ex["chosen"]})
    return ds.select_columns(["messages"])


def pref_dataset(cfg: DictConfig, split: str | None = None) -> Dataset:
    """Explicit-prompt preference data for DPOTrainer / RewardTrainer:
    prompt = [user turn], chosen/rejected = [assistant turn]."""
    ds = _load_split(cfg, split or cfg.data.split, cfg.data.get("max_samples"))

    def _split_prompt(ex):
        return {
            "prompt": ex["chosen"][:-1],
            "chosen": ex["chosen"][-1:],
            "rejected": ex["rejected"][-1:],
        }

    return ds.map(_split_prompt).select_columns(["prompt", "chosen", "rejected"])


def prompt_dataset(cfg: DictConfig) -> Dataset:
    """Prompt-only data for online RL (PPO/GRPO): a conversational ``prompt``
    column. Prompts longer than data.max_prompt_length (in characters as a
    cheap proxy, 4 chars/token) are dropped to bound rollout length."""
    ds = _load_split(cfg, cfg.data.split, cfg.data.get("max_samples"))
    max_chars = int(cfg.data.get("max_prompt_length", 512)) * 4
    ds = ds.filter(lambda ex: len(ex["prompt"]) <= max_chars)
    ds = ds.map(lambda ex: {"prompt": [{"role": "user", "content": ex["prompt"]}]})
    return ds.select_columns(["prompt"])


def eval_prompts(cfg: DictConfig) -> list[dict]:
    """Held-out prompts for the frontier evaluation, as
    [{prompt_id, prompt, messages([user turn]))}, ...]."""
    ds = _load_split(cfg, cfg.eval.prompts_split, int(cfg.eval.num_prompts))
    return [
        {
            "prompt_id": ex["prompt_id"],
            "prompt": ex["prompt"],
            "messages": [{"role": "user", "content": ex["prompt"]}],
        }
        for ex in ds
    ]


def _smoke() -> None:
    """Load 100 rows in every format and print one example of each."""
    from posttrain_lab.config import load_config

    cfg = load_config("stage1_sft_base.yaml", ["--smoke", "data.max_samples=100"])
    for name, fn, kwargs in [
        ("sft(train_sft)", sft_dataset, {}),
        ("pref(train_prefs)", pref_dataset, {"split": "train_prefs"}),
    ]:
        ds = fn(cfg, **kwargs) if kwargs else fn(cfg)
        print(f"--- {name}: {len(ds)} rows, columns={ds.column_names}")
        print({k: str(v)[:200] for k, v in ds[0].items()})

    cfg.data.split = "train_gen"
    ds = prompt_dataset(cfg)
    print(f"--- prompt(train_gen): {len(ds)} rows, columns={ds.column_names}")
    print({k: str(v)[:200] for k, v in ds[0].items()})

    cfg_eval = load_config("eval.yaml", ["--smoke"])
    rows = eval_prompts(cfg_eval)
    print(f"--- eval_prompts(test_prefs): {len(rows)} rows")
    print({k: str(v)[:200] for k, v in rows[0].items()})
    print("data smoke OK")


if __name__ == "__main__":
    _smoke()
