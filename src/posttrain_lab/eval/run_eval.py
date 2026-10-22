"""Per-checkpoint evaluation driver — incremental metric groups.

Each checkpoint has ONE summary row (a frontier point) in
results/<task>/eval/<run>/<ckpt>.jsonl plus a per-sample file
<ckpt>.samples.jsonl. Metrics are organized into groups (cfg.eval.
metric_groups, set per task); each group declares sentinel keys, and a group
runs only when its sentinels are missing from the existing row. The row is
atomically rewritten after every group, so:

  - re-running after a crash/preemption resumes at the missing groups
  - adding a new group to the config later computes ONLY that group on
    already-evaluated checkpoints (e.g. backfilling retention/diversity/
    pref_logps onto the helpful track)

Usage:
    python -m posttrain_lab.eval.run_eval [task=math] [--smoke] [--shard i/n] [k=v ...] [ckpt_dir ...]

With no explicit checkpoint dirs, discovers every policy checkpoint of the
config's seed (seed=N, default configs/base.yaml) under <results_root>/runs
(all arms + sft-base, excluding the RM): the KL reference (pi_SFT), the
pref_logps pairs and the sampling seed all follow cfg.seed, so another seed's
runs are evaluated by a run with that seed (scripts/math/m4_eval.sh SEED=N). --shard i/n
keeps every n-th checkpoint starting at i (used by the eval scripts to
spread checkpoints across GPUs; safe because groups touch one row each).
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from posttrain_lab.config import load_config
from posttrain_lab.data import eval_prompts
from posttrain_lab.eval import diversity, pref_logps, retention
from posttrain_lab.eval.generate import sample_responses
from posttrain_lab.eval.groups import SENTINELS, group_keys, missing_groups, stale_groups
from posttrain_lab.eval.kl import compute_kl
from posttrain_lab.eval.passk import passk_metrics
from posttrain_lab.eval.reward_score import score_samples
from posttrain_lab.tasks.gsm8k import is_correct
from posttrain_lab.utils.checkpoints import (discover_all_checkpoints, list_checkpoints, run_arm, run_hyperparam,
                                            run_seed)
from posttrain_lab.utils.gpu import free_gpu_memory
from posttrain_lab.utils.logging import log_eval_metrics, tracking_enabled


def _core_helpful(cfg, ckpt, ctx) -> dict:
    """Generate -> KL vs pi_SFT -> dual reward scoring (training RM + judge)."""
    samples = sample_responses(cfg, str(ckpt), ctx["prompts"])
    kl_stats = compute_kl(cfg, str(ckpt), samples)
    rm_scores = score_samples(cfg, str(cfg.eval.train_rm), samples, int(cfg.eval.judge_batch_size))
    judge_scores = score_samples(cfg, str(cfg.eval.judge), samples, int(cfg.eval.judge_batch_size))
    for sample, rm_score, judge_score in zip(samples, rm_scores, judge_scores):
        sample["score_train_rm"] = rm_score
        sample["score_judge"] = judge_score
    ctx["write_samples"](samples)

    n = max(len(samples), 1)
    return {
        **kl_stats,
        "reward_train_rm": sum(rm_scores) / n,
        "reward_judge": sum(judge_scores) / n,
        "num_samples": len(samples),
        "eval_prompts": len(ctx["prompts"]),
    }


def _core_math(cfg, ckpt, ctx) -> dict:
    """Generate -> KL vs pi_SFT-math -> verify -> pass@1 / pass@k."""
    samples = sample_responses(cfg, str(ckpt), ctx["prompts"])
    kl_stats = compute_kl(cfg, str(ckpt), samples)
    gold = {p["prompt_id"]: p["answer"] for p in ctx["prompts"]}
    for sample in samples:
        sample["correct"] = is_correct(sample["response"], gold[sample["prompt_id"]])
    ctx["write_samples"](samples)
    return {
        **kl_stats,
        **passk_metrics(samples, int(cfg.eval.pass_k)),
        "num_samples": len(samples),
        "eval_prompts": len(ctx["prompts"]),
    }


# group name -> (sentinel keys, compute_fn(cfg, ckpt, ctx) -> partial row);
# the sentinels live in eval/groups.py so posttrain_lab.status can read them
_COMPUTE = {
    "core": _core_helpful,
    "core_math": _core_math,
    "retention_ppl": retention.ppl_group,
    "retention_harness": retention.harness_group,
    "diversity": diversity.compute_group,
    "pref_logps": pref_logps.compute_group,
}
GROUPS = {name: (SENTINELS[name], fn) for name, fn in _COMPUTE.items()}


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


def _read_row(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    return None


def _atomic_write_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(row) + "\n")
    os.replace(tmp, path)


_ROW_META = {"run", "arm", "hyperparam", "checkpoint", "step"}


def _sync_tracking(cfg, run_dir: Path, summary_path: Path, row: dict) -> None:
    """Send MLflow any row metrics it hasn't received yet. A sidecar
    (<ckpt>.mlflow.json, kept out of the row so aggregate.py never sees it)
    records what was logged, so a tracking outage is retried on the next eval
    run and rows evaluated before tracking existed are backfilled."""
    if not tracking_enabled(cfg):
        return
    marker = summary_path.with_suffix(".mlflow.json")
    logged = set(json.loads(marker.read_text())) if marker.exists() else set()
    pending = {k: v for k, v in row.items() if k not in logged and k not in _ROW_META}
    if pending and log_eval_metrics(cfg, run_dir, row["step"], pending):
        tmp = marker.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(logged | set(pending))))
        os.replace(tmp, marker)


def _forget_logged(summary_path: Path, keys: set[str]) -> None:
    """Drop keys from the MLflow sidecar so _sync_tracking logs them again."""
    marker = summary_path.with_suffix(".mlflow.json")
    if not marker.exists() or not keys:
        return
    logged = set(json.loads(marker.read_text())) - keys
    tmp = marker.with_suffix(".tmp")
    tmp.write_text(json.dumps(sorted(logged)))
    os.replace(tmp, marker)


def evaluate_checkpoint(cfg, ckpt: Path, prompts: list[dict]) -> None:
    run_dir = ckpt.parent if ckpt.name.startswith("checkpoint-") else ckpt
    out_dir = Path(cfg.eval.out_dir) / run_dir.name
    summary_path = out_dir / f"{ckpt.name}.jsonl"
    samples_path = out_dir / f"{ckpt.name}.samples.jsonl"

    row = _read_row(summary_path) or {
        "run": run_dir.name,
        "arm": run_arm(run_dir),
        "hyperparam": run_hyperparam(run_dir),
        "checkpoint": str(ckpt),
        "step": _ckpt_step(ckpt),
    }
    requested = [g for g in cfg.eval.metric_groups if g in GROUPS]
    spp = int(cfg.eval.samples_per_prompt)
    stale = stale_groups(row, requested, len(prompts), spp)
    if stale:
        # evaluated on another prompt set: forget everything those groups wrote
        # first (row and MLflow marker), so neither a crash mid-way nor
        # aggregate.py reading the row in between can mix old-prompt and
        # new-prompt numbers, and the recomputed values are sent to MLflow again
        print(f"[eval] {summary_path}: {stale} were computed on {row.get('eval_prompts') or 'another number of'} "
              f"prompts, now {len(prompts)}: recomputing")
        dropped = {k for group in stale for k in group_keys(row, group)}
        for key in dropped:
            row.pop(key, None)
        _atomic_write_row(summary_path, row)
        _forget_logged(summary_path, dropped)
    missing = missing_groups(row, requested, len(prompts), spp)
    if not missing:
        print(f"[eval] {summary_path} complete for {requested}, skipping")
        _sync_tracking(cfg, run_dir, summary_path, row)
        return
    print(f"[eval] {ckpt} -> groups {missing}")

    samples_cache: list[dict] = []

    def write_samples(samples: list[dict]) -> None:
        samples_cache.clear()
        samples_cache.extend(samples)
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = samples_path.with_suffix(".tmp")
        with tmp.open("w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        os.replace(tmp, samples_path)

    def load_samples() -> list[dict]:
        if samples_cache:
            return samples_cache
        with samples_path.open() as f:
            samples_cache.extend(json.loads(line) for line in f if line.strip())
        return samples_cache

    ctx = {
        "prompts": prompts,
        "write_samples": write_samples,
        "load_samples": load_samples,
        "is_final": ckpt == list_checkpoints(run_dir)[-1],
    }
    for group in missing:
        metrics = GROUPS[group][1](cfg, ckpt, ctx)
        row.update(metrics)
        _atomic_write_row(summary_path, row)  # progress survives a crash mid-checkpoint
        _sync_tracking(cfg, run_dir, summary_path, row)
        free_gpu_memory()  # each group loads its own models; the next one (or checkpoint) needs the GPU back


def main() -> None:
    overrides, (shard_idx, shard_total), explicit = _parse_args(sys.argv[1:])
    cfg = load_config("eval.yaml", overrides)

    if explicit:
        ckpts = [Path(p) for p in explicit]
        for ckpt in ckpts:
            seed = run_seed(ckpt.parent if ckpt.name.startswith("checkpoint-") else ckpt)
            if seed is not None and seed != int(cfg.seed):
                sys.exit(f"[eval] {ckpt} is a seed-{seed} run but the config's seed is {cfg.seed}: its KL "
                         f"reference and pref_logps pairs would be seed {cfg.seed}'s; pass seed={seed}")
    else:
        ckpts = discover_all_checkpoints(Path(cfg.results_root) / "runs", seeds=[int(cfg.seed)])
    ckpts = [c for i, c in enumerate(ckpts) if i % shard_total == shard_idx]
    if not ckpts:
        print("[eval] no checkpoints found — train something first")
        return
    print(f"[eval] task={cfg.task} seed={cfg.seed}: {len(ckpts)} checkpoints (shard {shard_idx}/{shard_total}), "
          f"groups={list(cfg.eval.metric_groups)}")

    prompts = eval_prompts(cfg)
    for ckpt in ckpts:
        evaluate_checkpoint(cfg, ckpt, prompts)


if __name__ == "__main__":
    main()
