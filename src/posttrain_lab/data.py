"""Dataset loaders — the single source of truth for every stage's data.

Loaders dispatch on ``cfg.dataset.format``:

  ultrafeedback  HuggingFaceH4/ultrafeedback_binarized (helpful track):
                 prompt (str), prompt_id, messages, chosen/rejected (full
                 [user, assistant] conversations), score_chosen/rejected
  gsm8k          openai/gsm8k (math track): question (str), answer (gold
                 chain-of-thought ending in '#### <number>')
  pairs_disk     self-generated preference pairs written by gen_pairs.py
                 (math DPO + math pref_logps eval), already in the explicit
                 {prompt, chosen, rejected} message schema

TRL 1.x consumes conversational formats directly, so loaders return message
lists and let each trainer apply the tokenizer's chat template. All
subsampling is deterministic in (dataset, split, seed, max_samples).

Every split is loaded into memory (keep_in_memory=True), so map/filter/
shuffle work in RAM instead of writing cache-*.arrow files next to the
prepared dataset in HF_HOME. Under accelerate all ranks run these same maps
at once and race on the same cache file (fatal on a CIFS/SMB PVC, where
renaming onto a file another rank just created fails); a read-only or
shared HF cache never gets written either. The splits are small enough
(GSM8K 7.5k rows, UltraFeedback ~60k) to hold once per rank.
"""

from __future__ import annotations

from datasets import Dataset, load_dataset, load_from_disk
from omegaconf import DictConfig

from posttrain_lab.tasks.gsm8k import extract_gold_answer


def _format(cfg: DictConfig) -> str:
    return str(cfg.dataset.get("format", "ultrafeedback"))


def _load_split(cfg: DictConfig, split: str, max_samples: int | None) -> Dataset:
    kwargs = {}
    if cfg.dataset.get("config"):
        kwargs["name"] = str(cfg.dataset.config)
    ds = load_dataset(cfg.dataset.id, split=split, keep_in_memory=True, **kwargs)
    return _subsample(cfg, ds, max_samples)


def _subsample(cfg: DictConfig, ds: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is not None and max_samples < len(ds):
        ds = ds.shuffle(seed=cfg.seed).select(range(max_samples))
    return ds


def sft_dataset(cfg: DictConfig) -> Dataset:
    """Conversational SFT data: a ``messages`` column.

    ultrafeedback: train_sft as-is, or the chosen responses of the preference
    split when data.use_chosen_as_sft=true (the imitation arm).
    gsm8k: question -> gold chain-of-thought (which instills the '#### N'
    answer format the verifier and the RL arms rely on).
    """
    ds = _load_split(cfg, cfg.data.split, cfg.data.get("max_samples"))
    if _format(cfg) == "gsm8k":
        ds = ds.map(
            lambda ex: {
                "messages": [
                    {"role": "user", "content": ex["question"]},
                    {"role": "assistant", "content": ex["answer"]},
                ]
            }
        )
    elif cfg.data.get("use_chosen_as_sft", False):
        ds = ds.map(lambda ex: {"messages": ex["chosen"]})
    return ds.select_columns(["messages"])


def pref_dataset(cfg: DictConfig, split: str | None = None) -> Dataset:
    """Explicit-prompt preference data for DPOTrainer / RewardTrainer /
    pref_logps eval: prompt = [user turn], chosen/rejected = [assistant turn].

    pairs_disk loads the self-generated pairs (already in this schema);
    gsm8k has no native preferences — run gen_pairs.py first.
    """
    if _format(cfg) == "pairs_disk" or (_format(cfg) == "gsm8k" and cfg.data.get("pairs_path")):
        ds = load_from_disk(str(cfg.data.pairs_path), keep_in_memory=True)[split or str(cfg.data.get("split", "train"))]
        return _subsample(cfg, ds, cfg.data.get("max_samples"))

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
    column. The gsm8k branch keeps an ``answer`` column (the extracted gold
    number) — GRPOTrainer forwards extra dataset columns as kwargs to
    reward_funcs, which is how the verifier receives the label."""
    ds = _load_split(cfg, cfg.data.split, cfg.data.get("max_samples"))
    if _format(cfg) == "gsm8k":
        ds = ds.map(
            lambda ex: {
                "prompt": [{"role": "user", "content": ex["question"]}],
                "answer": extract_gold_answer(ex["answer"]),
            }
        )
        return ds.select_columns(["prompt", "answer"])

    max_chars = int(cfg.data.get("max_prompt_length", 512)) * 4
    ds = ds.filter(lambda ex: len(ex["prompt"]) <= max_chars)
    ds = ds.map(lambda ex: {"prompt": [{"role": "user", "content": ex["prompt"]}]})
    return ds.select_columns(["prompt"])


def eval_prompts(cfg: DictConfig, split: str | None = None, num: int | None = None) -> list[dict]:
    """Held-out prompts for the frontier evaluation, as
    [{prompt_id, prompt, messages, (answer)}, ...]. ``num`` defaults to
    eval.num_prompts."""
    return prompt_rows(cfg, split or cfg.eval.prompts_split, num or int(cfg.eval.num_prompts))


def prompt_rows(cfg: DictConfig, split: str, max_samples: int | None) -> list[dict]:
    """Prompts of ``split`` in the eval_prompts row format; max_samples=None
    means the whole split (gen_pairs samples every train prompt)."""
    ds = _load_split(cfg, split, max_samples)
    if _format(cfg) == "gsm8k":
        return [
            {
                "prompt_id": f"gsm8k-{i}",
                "prompt": ex["question"],
                "messages": [{"role": "user", "content": ex["question"]}],
                "answer": extract_gold_answer(ex["answer"]),
            }
            for i, ex in enumerate(ds)
        ]
    return [
        {
            "prompt_id": ex["prompt_id"],
            "prompt": ex["prompt"],
            "messages": [{"role": "user", "content": ex["prompt"]}],
        }
        for ex in ds
    ]


def _smoke() -> None:
    """Load 100 rows in every format for the selected task and print examples."""
    import sys

    from posttrain_lab.config import load_config

    task = "math" if "task=math" in sys.argv else "helpful"
    if task == "math":
        cfg = load_config("math/m1_sft_gold.yaml", ["--smoke", "data.max_samples=100"])
        ds = sft_dataset(cfg)
        print(f"--- gsm8k sft(train): {len(ds)} rows")
        print({k: str(v)[:200] for k, v in ds[0].items()})
        cfg.data.split = "train"
        ds = prompt_dataset(cfg)
        print(f"--- gsm8k prompt(train): {len(ds)} rows, columns={ds.column_names}")
        print({k: str(v)[:200] for k, v in ds[0].items()})
        cfg_eval = load_config("eval.yaml", ["--smoke", "task=math"])
        rows = eval_prompts(cfg_eval)
        print(f"--- gsm8k eval_prompts(test): {len(rows)} rows")
        print({k: str(v)[:200] for k, v in rows[0].items()})
        # verifier sanity
        from posttrain_lab.tasks.gsm8k import is_correct

        assert is_correct("... so the total is 42. #### 42", "42")
        assert is_correct("the answer is $1,234.", "1234")
        assert not is_correct("no numbers here", "7")
        print("verifier sanity OK")
    else:
        cfg = load_config("helpful/stage1_sft_base.yaml", ["--smoke", "data.max_samples=100"])
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
    print(f"data smoke OK ({task})")


if __name__ == "__main__":
    _smoke()
