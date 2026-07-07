"""Per-checkpoint evaluation driver: generate -> KL -> dual reward scores.

Each checkpoint becomes one summary row (a frontier point) in
results/eval/<run>/<ckpt>.jsonl plus a per-sample file <ckpt>.samples.jsonl
(responses with individual scores — needed later for z-scoring against the
pi_SFT baseline). Already-evaluated checkpoints are skipped, so re-running
after preemption or after new training runs only does new work.

Usage:
    python -m posttrain_lab.eval.run_eval [--smoke] [--shard i/n] [k=v ...] [ckpt_dir ...]

With no explicit checkpoint dirs, discovers every policy checkpoint under
<results_root>/runs (all arms + sft-base, excluding the RM). --shard i/n
keeps only every n-th checkpoint starting at i — used by stage4_eval.sh to
spread checkpoints across GPUs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from posttrain_lab.config import load_config
from posttrain_lab.data import eval_prompts
from posttrain_lab.eval.generate import sample_responses
from posttrain_lab.eval.kl import compute_kl
from posttrain_lab.eval.reward_score import score_samples
from posttrain_lab.utils.checkpoints import discover_all_checkpoints, run_arm, run_hyperparam
from posttrain_lab.utils.logging import append_jsonl


def _parse_args(argv: list[str]) -> tuple[list[str], tuple[int, int], list[str]]:
    overrides, shard, ckpts = [], (0, 1), []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--shard":
            idx, total = argv[i + 1].split("/")
            shard = (int(idx), int(total))
            i += 2
        elif arg == "--smoke" or "=" in arg:
            overrides.append(arg)
            i += 1
        else:
            ckpts.append(arg)
            i += 1
    return overrides, shard, ckpts


def _ckpt_step(ckpt: Path) -> int:
    m = re.search(r"checkpoint-(\d+)", ckpt.name)
    return int(m.group(1)) if m else -1  # -1 = final flat save


def evaluate_checkpoint(cfg, ckpt: Path, prompts: list[dict]) -> None:
    run_dir = ckpt.parent if ckpt.name.startswith("checkpoint-") else ckpt
    out_dir = Path(cfg.eval.out_dir) / run_dir.name
    summary_path = out_dir / f"{ckpt.name}.jsonl"
    if summary_path.exists():
        print(f"[eval] {summary_path} exists, skipping")
        return
    print(f"[eval] {ckpt}")

    samples = sample_responses(cfg, str(ckpt), prompts)
    kl_stats = compute_kl(cfg, str(ckpt), samples)
    rm_scores = score_samples(cfg, str(cfg.eval.train_rm), samples, int(cfg.eval.judge_batch_size))
    judge_scores = score_samples(cfg, str(cfg.eval.judge), samples, int(cfg.eval.judge_batch_size))

    samples_path = out_dir / f"{ckpt.name}.samples.jsonl"
    for sample, rm_score, judge_score in zip(samples, rm_scores, judge_scores):
        append_jsonl(samples_path, {**sample, "score_train_rm": rm_score, "score_judge": judge_score})

    n = max(len(samples), 1)
    append_jsonl(
        summary_path,
        {
            "run": run_dir.name,
            "arm": run_arm(run_dir),
            "hyperparam": run_hyperparam(run_dir),
            "checkpoint": str(ckpt),
            "step": _ckpt_step(ckpt),
            **kl_stats,
            "reward_train_rm": sum(rm_scores) / n,
            "reward_judge": sum(judge_scores) / n,
            "num_samples": len(samples),
        },
    )


def main() -> None:
    overrides, (shard_idx, shard_total), explicit = _parse_args(sys.argv[1:])
    cfg = load_config("eval.yaml", overrides)

    if explicit:
        ckpts = [Path(p) for p in explicit]
    else:
        ckpts = discover_all_checkpoints(Path(cfg.results_root) / "runs")
    ckpts = [c for i, c in enumerate(ckpts) if i % shard_total == shard_idx]
    if not ckpts:
        print("[eval] no checkpoints found — train something first")
        return
    print(f"[eval] {len(ckpts)} checkpoints (shard {shard_idx}/{shard_total})")

    prompts = eval_prompts(cfg)
    for ckpt in ckpts:
        evaluate_checkpoint(cfg, ckpt, prompts)


if __name__ == "__main__":
    main()
