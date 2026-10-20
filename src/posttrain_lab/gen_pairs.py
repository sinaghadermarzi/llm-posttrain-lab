"""Math stage m2: build self-generated preference pairs from pi_SFT-math.

Samples k solutions per GSM8K train prompt from pi_SFT-math (at the same
temperature the RL arms and the eval use), labels each by exact-answer
correctness, and pairs them up: correct = chosen, incorrect = rejected. This
is the standard way to get DPO preference data for a verifiable task — and
it bakes in a selection bias worth remembering when reading results: only
"contested" prompts (some samples right, some wrong) yield pairs.

Output: a DatasetDict {train, test} at cfg.data.pairs_path in the same
explicit {prompt, chosen, rejected} message schema data.pref_dataset emits,
plus stats.json (solve rate, pair yield, drop counts) — check it before
launching DPO; a near-0 or near-1 solve rate means few pairs.

Usage: python -m posttrain_lab.gen_pairs [configs/math/m2_gen_pairs.yaml] [k=v ...]
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict

from posttrain_lab.config import load_config
from posttrain_lab.data import prompt_rows
from posttrain_lab.eval.generate import sample_responses
from posttrain_lab.tasks.gsm8k import is_correct
from posttrain_lab.train.common import begin_run, init_path, parse_entry
from posttrain_lab.utils.checkpoints import is_done, mark_done


def build_pairs(samples: list[dict], gold_by_prompt: dict[str, str], max_per_prompt: int, seed: int) -> tuple[list[dict], dict]:
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        by_prompt[sample["prompt_id"]].append(sample)

    rng = random.Random(seed)
    rows, n_all_correct, n_all_wrong, n_unfinished_correct = [], 0, 0, 0
    for prompt_id, group in sorted(by_prompt.items()):
        gold = gold_by_prompt[prompt_id]
        correct = [s for s in group if is_correct(s["response"], gold)]
        wrong = [s for s in group if not is_correct(s["response"], gold)]
        if not wrong:
            n_all_correct += 1
            continue
        if not correct:
            n_all_wrong += 1
            continue
        # chosen = a complete correct answer: one cut off at max_new_tokens
        # would teach DPO to prefer not stopping
        correct = [s for s in correct if s.get("finished", True)]
        if not correct:
            n_unfinished_correct += 1
            continue
        rng.shuffle(correct)
        rng.shuffle(wrong)
        for chosen, rejected in list(zip(correct, wrong))[:max_per_prompt]:
            rows.append(
                {
                    "prompt": [{"role": "user", "content": chosen["prompt"]}],
                    "chosen": [{"role": "assistant", "content": chosen["response"]}],
                    "rejected": [{"role": "assistant", "content": rejected["response"]}],
                }
            )
    n_correct = sum(
        is_correct(s["response"], gold_by_prompt[s["prompt_id"]]) for s in samples
    )
    stats = {
        "prompts": len(by_prompt),
        "samples": len(samples),
        "sample_solve_rate": n_correct / max(len(samples), 1),
        "pairs": len(rows),
        "prompts_all_correct": n_all_correct,
        "prompts_all_wrong": n_all_wrong,
        "prompts_no_finished_correct": n_unfinished_correct,
        "samples_unfinished": sum(not s.get("finished", True) for s in samples),
    }
    return rows, stats


def main() -> None:
    stage_yaml, overrides = parse_entry("math/m2_gen_pairs.yaml")
    cfg = load_config(stage_yaml, overrides)
    out_dir = Path(cfg.data.pairs_path)
    if is_done(out_dir):
        print(f"[gen_pairs] {out_dir} already DONE, skipping")
        return
    begin_run(cfg)

    # data.max_samples=null -> every train prompt (not eval.num_prompts)
    prompts = prompt_rows(cfg, str(cfg.data.split), cfg.data.get("max_samples"))
    gold_by_prompt = {p["prompt_id"]: p["answer"] for p in prompts}
    print(f"[gen_pairs] sampling {cfg.eval.samples_per_prompt} solutions x {len(prompts)} prompts")
    samples = sample_responses(cfg, init_path(cfg), prompts)

    rows, stats = build_pairs(
        samples, gold_by_prompt, int(cfg.pairs.max_per_prompt), int(cfg.seed)
    )
    if not rows:
        raise SystemExit(f"[gen_pairs] FATAL: no pairs produced — stats: {stats}")

    rng = random.Random(int(cfg.seed))
    rng.shuffle(rows)
    n_test = max(1, int(len(rows) * float(cfg.pairs.test_fraction)))
    dsd = DatasetDict(
        {
            "train": Dataset.from_list(rows[n_test:]),
            "test": Dataset.from_list(rows[:n_test]),
        }
    )
    dsd.save_to_disk(str(out_dir))
    stats["train_pairs"] = len(dsd["train"])
    stats["test_pairs"] = len(dsd["test"])
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[gen_pairs] {stats}")
    mark_done(out_dir)


if __name__ == "__main__":
    main()
