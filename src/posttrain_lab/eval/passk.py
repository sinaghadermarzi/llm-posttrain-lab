"""pass@1 / pass@k scoring for the math track (correctness already attached
to each sample by the core_math group).

pass@1 = mean correctness over ALL samples (the unbiased single-sample
success rate at the eval temperature). pass@k = fraction of prompts with at
least one correct among their k samples. Their gap is the diversity dividend:
imitation-style training tends to collapse it (pass@k falls toward pass@1)
while KL-constrained RL preserves it.

Standard errors are over prompts (the unit that varies: the 8 samples of one
prompt are correlated): pass1_se = sd of the per-prompt solve rates / sqrt(n).
Checkpoints are scored on the same prompts, so a difference between two of
them is better judged paired (analysis/compare_math.py) than from these.
"""

from __future__ import annotations

import math
from collections import defaultdict


def _se(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1) / n)


def passk_metrics(samples: list[dict], k: int) -> dict:
    by_prompt: dict[str, list[bool]] = defaultdict(list)
    for sample in samples:
        by_prompt[sample["prompt_id"]].append(bool(sample["correct"]))

    total = sum(len(v) for v in by_prompt.values())
    n_correct = sum(sum(v) for v in by_prompt.values())
    pass1 = n_correct / max(total, 1)
    passk = sum(any(v) for v in by_prompt.values()) / max(len(by_prompt), 1)
    return {
        "pass1": pass1,
        f"pass{k}": passk,
        "pass_gap": passk - pass1,
        "pass1_se": _se([sum(v) / len(v) for v in by_prompt.values()]),
        f"pass{k}_se": _se([float(any(v)) for v in by_prompt.values()]),
    }
