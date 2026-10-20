"""gen_pairs samples every train prompt when data.max_samples is null. It used
to go through eval_prompts, whose fallback is eval.num_prompts (512 for math),
so the full run built pairs from 512 of GSM8K's 7473 train prompts.

Runs gen_pairs.main on the real m2 config against a local GSM8K-shaped
dataset, with sampling stubbed (one right and one wrong answer per prompt)."""

from __future__ import annotations

import json
from pathlib import Path

PROBE = """
import sys
from posttrain_lab import gen_pairs

def fake_samples(cfg, model, prompts):
    return [{{"prompt_id": p["prompt_id"], "prompt": p["prompt"], "response": f"#### {{p['answer']}}{{suffix}}"}}
            for p in prompts for suffix in ("", "0")]  # one correct, one wrong

gen_pairs.sample_responses = fake_samples
sys.argv = ["gen_pairs", "configs/math/m2_gen_pairs.yaml", "dataset.id={gsm8k}", "dataset.config=null", "train.report_to=none", *{extra!r}]
gen_pairs.main()
"""


def _run(repo, tmp_path: Path, *extra: str) -> dict:
    gsm8k = tmp_path / "gsm8k"
    gsm8k.mkdir()
    for split in ("train", "test"):
        (gsm8k / f"{split}.jsonl").write_text("".join(
            json.dumps({"question": f"What is {i}+1?", "answer": f"{i}+1={i + 1}\n#### {i + 1}"}) + "\n"
            for i in range(30)))
    results = tmp_path / "results"
    run = repo.python("-c", PROBE.format(gsm8k=gsm8k, extra=list(extra)), check=False,
                      RESULTS_DIR=str(results), HF_DATASETS_CACHE=str(tmp_path / "cache"), HF_HUB_OFFLINE="1")
    assert run.returncode == 0, run.stderr[-3000:]
    return json.loads((results / "math" / "data" / "self_pairs_seed17" / "stats.json").read_text())


def test_null_max_samples_means_every_train_prompt(repo, tmp_path: Path) -> None:
    stats = _run(repo, tmp_path, "eval.num_prompts=5")  # the eval cap must not apply
    assert stats["prompts"] == 30
    assert stats["pairs"] == 30


def test_max_samples_still_caps(repo, tmp_path: Path) -> None:
    stats = _run(repo, tmp_path, "data.max_samples=12")
    assert stats["prompts"] == 12
