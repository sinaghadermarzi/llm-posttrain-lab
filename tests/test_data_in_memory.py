"""The data loaders never write map/filter/shuffle cache files into the HF
datasets cache: under accelerate every rank runs the same maps at once and
races on one cache-*.arrow next to the prepared dataset (on the pod's CIFS
PVC that killed math SFT with PermissionError / FileNotFoundError).

Runs every loader in both formats on small local datasets, with a throwaway
HF_DATASETS_CACHE, and checks nothing but the prepared dataset landed there."""

from __future__ import annotations

import json
from pathlib import Path

PROBE = """
from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf
from posttrain_lab import data

def gsm8k(split):
    return {{"dataset": {{"id": "{gsm8k}", "format": "gsm8k"}}, "seed": 17,
             "data": {{"split": split, "max_samples": 5}}, "eval": {{"prompts_split": "test", "num_prompts": 3}}}}

def uf(split, **extra):
    return {{"dataset": {{"id": "{uf}", "format": "ultrafeedback"}}, "seed": 17,
             "data": {{"split": split, "max_samples": 5, **extra}}, "eval": {{"prompts_split": "test", "num_prompts": 3}}}}

out = []
cfg = OmegaConf.create(gsm8k("train"))
out += [data.sft_dataset(cfg), data.prompt_dataset(cfg)]
data.eval_prompts(cfg)
cfg = OmegaConf.create(uf("train"))
out += [data.sft_dataset(cfg), data.pref_dataset(cfg), data.prompt_dataset(cfg)]
out += [data.sft_dataset(OmegaConf.create(uf("train", use_chosen_as_sft=True)))]
data.eval_prompts(cfg)

pairs = Dataset.from_list([{{"prompt": [{{"role": "user", "content": f"q{{i}}"}}],
                            "chosen": [{{"role": "assistant", "content": "a"}}],
                            "rejected": [{{"role": "assistant", "content": "b"}}]}} for i in range(8)])
DatasetDict(train=pairs).save_to_disk("{pairs}")
cfg = OmegaConf.create({{"dataset": {{"format": "pairs_disk"}}, "seed": 17,
                        "data": {{"pairs_path": "{pairs}", "split": "train", "max_samples": 5}}}})
out += [data.pref_dataset(cfg)]

assert all(len(ds) == 5 for ds in out), [len(ds) for ds in out]
assert all(ds.cache_files == [] for ds in out), [ds.cache_files for ds in out]
print("OK")
"""


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_loaders_write_no_cache_files(repo, tmp_path: Path) -> None:
    gsm8k, uf = tmp_path / "gsm8k", tmp_path / "uf"
    for split in ("train", "test"):
        _jsonl(gsm8k / f"{split}.jsonl",
               [{"question": f"What is {i}+1?", "answer": f"{i}+1={i + 1}\n#### {i + 1}"} for i in range(8)])
        _jsonl(uf / f"{split}.jsonl", [{
            "prompt": f"p{i}", "prompt_id": str(i),
            "messages": [{"role": "user", "content": f"p{i}"}, {"role": "assistant", "content": "good"}],
            "chosen": [{"role": "user", "content": f"p{i}"}, {"role": "assistant", "content": "good"}],
            "rejected": [{"role": "user", "content": f"p{i}"}, {"role": "assistant", "content": "bad"}],
        } for i in range(8)])
    cache = tmp_path / "datasets-cache"
    probe = PROBE.format(gsm8k=gsm8k, uf=uf, pairs=tmp_path / "pairs")
    run = repo.python("-c", probe, check=False, HF_DATASETS_CACHE=str(cache), HF_HUB_OFFLINE="1")
    assert run.returncode == 0, run.stderr[-3000:]
    assert run.stdout.strip().endswith("OK")
    written = sorted(p.name for p in cache.rglob("cache-*.arrow"))
    assert written == [], written
