"""GSM8K answer extraction and verification — THE single verifier module.

Used by three consumers, so correctness rules live only here:
  - gen_pairs.py     labels sampled solutions to build preference pairs
  - train/grpo.py    verifier_reward as the TRL reward_funcs callable
  - eval/passk.py    pass@1 / pass@k scoring

GSM8K gold answers end with '#### <number>'. Generated text may or may not
follow that format (base-model checkpoints often won't), so prediction
extraction tries the '####' form first and falls back to the last number in
the text. Comparison is numeric (int/float tolerant), after stripping commas,
'$' and trailing periods.
"""

from __future__ import annotations

import re

_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
_HASHES = re.compile(r"####\s*(-?\$?\d[\d,]*(?:\.\d+)?)")


def _to_float(text: str) -> float | None:
    cleaned = text.replace(",", "").replace("$", "").rstrip(".").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_gold_answer(answer_text: str) -> str:
    """Final answer from a GSM8K gold solution (after '####')."""
    m = _HASHES.search(answer_text)
    if not m:
        raise ValueError(f"no '####' answer in gold solution: {answer_text[-80:]!r}")
    return m.group(1)


def extract_pred_answer(generated: str) -> str | None:
    """Final answer from generated text: '####' form first, else last number."""
    m = _HASHES.search(generated)
    if m:
        return m.group(1)
    numbers = _NUMBER.findall(generated)
    return numbers[-1] if numbers else None


def is_correct(generated: str, gold: str) -> bool:
    pred = extract_pred_answer(generated)
    if pred is None:
        return False
    pred_val, gold_val = _to_float(pred), _to_float(gold)
    return pred_val is not None and gold_val is not None and abs(pred_val - gold_val) < 1e-6


def _completion_text(completion) -> str:
    """TRL passes completions as strings or as [{role, content}] message lists."""
    if isinstance(completion, str):
        return completion
    return "".join(turn.get("content", "") for turn in completion)


def verifier_reward(completions, answer, **kwargs) -> list[float]:
    """TRL GRPOTrainer reward_funcs signature: batched completions plus the
    dataset's extra columns as keyword lists ('answer' carries the gold
    number, kept by data.prompt_dataset for the gsm8k format)."""
    return [
        1.0 if is_correct(_completion_text(completion), gold) else 0.0
        for completion, gold in zip(completions, answer)
    ]
