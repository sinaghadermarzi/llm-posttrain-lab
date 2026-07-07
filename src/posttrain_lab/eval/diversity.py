"""Diversity / mode-collapse metrics, computed from the samples the core
group already generated (never regenerates; CPU-only).

Within each prompt's group of N samples:
  distinct-n   unique n-grams / total n-grams pooled over the group
               (1.0 = every n-gram unique; falling distinct-n = repetition)
  self-BLEU    each sample scored as BLEU against the other N-1 samples as
               references (hand-rolled smoothed BLEU-4, no nltk dep);
               RISING self-BLEU = the samples are converging = mode collapse
Reported values are means over prompts (prompts need >= 2 samples).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def distinct_n(groups: list[list[list[str]]], n: int) -> float:
    """Mean over groups of (unique n-grams / total n-grams) pooled per group."""
    ratios = []
    for group in groups:
        pooled = Counter()
        for tokens in group:
            pooled.update(_ngrams(tokens, n))
        total = sum(pooled.values())
        if total:
            ratios.append(len(pooled) / total)
    return sum(ratios) / max(len(ratios), 1)


def _bleu(candidate: list[str], references: list[list[str]], max_n: int = 4) -> float:
    """Smoothed (add-1) BLEU-max_n with brevity penalty, multi-reference."""
    if not candidate:
        return 0.0
    log_precision = 0.0
    for n in range(1, max_n + 1):
        cand = _ngrams(candidate, n)
        if not cand:
            log_precision += math.log(1.0 / (len(candidate) + 1))
            continue
        merged = Counter()
        for ref in references:
            for gram, count in _ngrams(ref, n).items():
                merged[gram] = max(merged[gram], count)
        overlap = sum(min(count, merged[gram]) for gram, count in cand.items())
        log_precision += math.log((overlap + 1) / (sum(cand.values()) + 1))
    ref_len = min(references, key=lambda r: abs(len(r) - len(candidate)))
    bp = 1.0 if len(candidate) >= len(ref_len) else math.exp(1 - len(ref_len) / max(len(candidate), 1))
    return bp * math.exp(log_precision / max_n)


def self_bleu(groups: list[list[list[str]]]) -> float:
    scores = []
    for group in groups:
        if len(group) < 2:
            continue
        for i, candidate in enumerate(group):
            refs = group[:i] + group[i + 1 :]
            scores.append(_bleu(candidate, refs))
    return sum(scores) / max(len(scores), 1)


def compute_group(cfg, ckpt, ctx) -> dict:
    samples = ctx["load_samples"]()
    by_prompt: dict[str, list[list[str]]] = defaultdict(list)
    for sample in samples:
        by_prompt[sample["prompt_id"]].append(sample["response"].split())
    groups = [g for g in by_prompt.values() if len(g) >= 2]
    return {
        "distinct_1": distinct_n(groups, 1),
        "distinct_2": distinct_n(groups, 2),
        "distinct_3": distinct_n(groups, 3),
        "self_bleu": self_bleu(groups),
    }
