"""Paired comparison of math checkpoints on the same eval prompts.

Every checkpoint is scored on the same GSM8K test prompts, so the noise that
matters for "is A better than B" is the per-prompt difference, not each
point's own standard error: a prompt that is hard for A is usually hard for B
too. This prints each checkpoint's pass@1 ± SE (over prompts) and, for every
checkpoint after the first, its paired difference to the first with a 95% CI.

Usage:
    python analysis/compare_math.py [results_root] RUN@STEP RUN@STEP [...]
    python analysis/compare_math.py dpo_beta0.3@116 grpo_beta0.01@374 grpo_beta0.0@374

RUN is a run dir name, or a unique prefix of one (dpo_beta0.3 -> dpo_beta0.3_seed17);
STEP is a checkpoint step or 'last'. Picking each run's best checkpoint on
these same prompts flatters it: compare checkpoints chosen some other way
(matched KL, a fixed step) where you can.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from paths import default_root


def resolve(eval_root: Path, spec: str) -> Path:
    if "@" not in spec:
        sys.exit(f"{spec!r}: expected RUN@STEP (e.g. grpo_beta0.01@374 or dpo_beta0.3@last)")
    run, step = spec.rsplit("@", 1)
    runs = [d for d in eval_root.iterdir() if d.is_dir() and (d.name == run or d.name.startswith(run + "_"))]
    exact = [d for d in runs if d.name == run]
    if len(exact) == 1:
        runs = exact
    if len(runs) != 1:
        found = ", ".join(sorted(d.name for d in runs)) or "none"
        sys.exit(f"{spec!r}: run {run!r} matches {found} under {eval_root}")
    files = {int(m[1]): p for p in runs[0].glob("checkpoint-*.samples.jsonl")
             if (m := re.fullmatch(r"checkpoint-(\d+)\.samples\.jsonl", p.name))}
    if not files:
        sys.exit(f"{spec!r}: no samples files in {runs[0]}")
    key = max(files) if step == "last" else int(step)
    if key not in files:
        sys.exit(f"{spec!r}: no checkpoint-{key} samples (have {sorted(files)})")
    return files[key]


def per_prompt(path: Path) -> dict[str, float]:
    by_prompt: dict[str, list[bool]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                by_prompt[s["prompt_id"]].append(bool(s["correct"]))
    return {pid: sum(v) / len(v) for pid, v in by_prompt.items()}


def mean_se(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    se = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1) / n) if n > 1 else float("nan")
    return mean, se


def kl_of(samples_path: Path) -> float | None:
    row_path = samples_path.with_name(samples_path.name.replace(".samples.jsonl", ".jsonl"))
    try:
        return json.loads(row_path.read_text().splitlines()[0]).get("kl_k1")
    except (OSError, IndexError, ValueError):
        return None


def main(argv: list[str]) -> None:
    if argv and Path(argv[0]).is_dir():
        root, specs = Path(argv[0]), argv[1:]
    else:
        root, specs = Path(default_root("math")), argv
    if len(specs) < 2:
        sys.exit(__doc__)
    eval_root = root / "eval"
    paths = [resolve(eval_root, s) for s in specs]
    scores = [per_prompt(p) for p in paths]

    print(f"{'checkpoint':<44} {'prompts':>7} {'pass@1':>7} {'± SE':>6} {'KL k1':>7}")
    for spec, path, sc in zip(specs, paths, scores):
        mean, se = mean_se(list(sc.values()))
        kl = kl_of(path)
        print(f"{spec:<44} {len(sc):>7} {mean:>7.3f} {se:>6.3f} {kl if kl is None else round(kl, 2)!s:>7}")

    base_spec, base = specs[0], scores[0]
    print(f"\npaired vs {base_spec} (same prompts; 95% CI = ± 1.96 SE)")
    for spec, sc in zip(specs[1:], scores[1:]):
        shared = sorted(set(base) & set(sc))
        if len(shared) < min(len(base), len(sc)):
            print(f"  note: {spec} shares only {len(shared)} prompts with {base_spec} (evaluated on other prompt sets?)")
        if len(shared) < 2:
            print(f"  {spec}: too few shared prompts")
            continue
        diff, se = mean_se([sc[p] - base[p] for p in shared])
        lo, hi = diff - 1.96 * se, diff + 1.96 * se
        verdict = "better" if lo > 0 else "worse" if hi < 0 else "no clear difference"
        print(f"  {spec:<42} {diff:+.3f} ± {se:.3f}  [{lo:+.3f}, {hi:+.3f}]  {verdict}  ({len(shared)} prompts)")


if __name__ == "__main__":
    main(sys.argv[1:])
