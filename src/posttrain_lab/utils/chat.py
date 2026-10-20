"""End-of-turn handling: generation must stop where the chat template ends an
assistant turn.

Qwen2.5 *base* models declare eos = <|endoftext|>, but the chat template every
stage trains and samples with closes each assistant turn with <|im_end|>. Left
alone, nothing stops at <|im_end|>: every sample runs to max_new_tokens and
the text after the answer is invented follow-up turns (the first full math
run: 100% of eval samples at the 512 cap, KL and diversity measured mostly on
that tail). So:

  - load_tokenizer (train/common.py) makes the end-of-turn token the
    tokenizer's eos, the way Qwen's -Instruct tokenizers are set up; every
    checkpoint a stage saves then carries it, and the HF Trainer aligns the
    model's config/generation_config to it (eos_token_id [<|im_end|>, <|endoftext|>]).
  - every sampler passes stop_token_ids(...) explicitly (eval/generate.py,
    GRPO's vLLM/HF generation), which also covers checkpoints saved before
    this fix.
  - `python -m posttrain_lab.utils.chat RUN_DIR...` checks saved runs and
    exits nonzero if one would not stop at end of turn (run_smoke.sh gates on it).

Kept free of torch so posttrain_lab.status can import it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_MARKER = "⁣END-OF-TURN-PROBE⁣"


def end_of_turn_token(tokenizer) -> str | None:
    """The special token the chat template closes an assistant turn with
    (<|im_end|> for Qwen), or None without a template / a special token there."""
    if not getattr(tokenizer, "chat_template", None):
        return None
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": _MARKER}], tokenize=False
        )
    except Exception:  # templates that reject this probe: leave eos alone
        return None
    if _MARKER not in text:
        return None
    after = text.split(_MARKER, 1)[1]
    specials = set(tokenizer.all_special_tokens) | set(tokenizer.get_added_vocab())
    for token in sorted(specials, key=len, reverse=True):
        if token and after.startswith(token):
            return token
    return None


def use_end_of_turn_eos(tokenizer):
    """Make the end-of-turn token the tokenizer's eos (pad stays as it was, so
    eos is never masked out as padding). Returns the tokenizer."""
    eot = end_of_turn_token(tokenizer)
    if eot and eot != tokenizer.eos_token:
        tokenizer.eos_token = eot
    return tokenizer


def stop_token_ids(tokenizer, checkpoint: str | Path | None = None) -> list[int]:
    """Every id generation should stop at: the end-of-turn token, the
    tokenizer's eos, and the checkpoint's generation_config eos ids."""
    ids: list[int] = []
    eot = end_of_turn_token(tokenizer)
    if eot is not None:
        ids.append(tokenizer.convert_tokens_to_ids(eot))
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    if checkpoint is not None:
        ids += generation_eos_ids(checkpoint)
    return [i for n, i in enumerate(ids) if i is not None and i not in ids[:n]]


def generation_eos_ids(checkpoint: str | Path) -> list[int]:
    path = Path(checkpoint) / "generation_config.json"
    try:
        eos = json.loads(path.read_text()).get("eos_token_id")
    except (OSError, ValueError):
        return []
    if eos is None:
        return []
    return [eos] if isinstance(eos, int) else [int(i) for i in eos]


def stop_problems(checkpoint: str | Path) -> list[str]:
    """Why a saved checkpoint would not stop at end of turn (empty if fine)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    eot = end_of_turn_token(tokenizer)
    if eot is None:
        return []
    problems = []
    if tokenizer.eos_token != eot:
        problems.append(f"tokenizer eos is {tokenizer.eos_token!r}, the chat template ends turns with {eot!r}")
    eot_id = tokenizer.convert_tokens_to_ids(eot)
    gen = generation_eos_ids(checkpoint)
    if (Path(checkpoint) / "generation_config.json").exists() and eot_id not in gen:
        problems.append(f"generation_config eos_token_id {gen} lacks {eot!r} ({eot_id})")
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id == eot_id:
        problems.append(f"pad token is {eot!r}: training would mask the end-of-turn token out as padding")
    return problems


def _cli(argv: list[str]) -> int:
    """python -m posttrain_lab.utils.chat RUN_DIR... : nonzero if any saved run
    (its tokenizer + generation_config) would not stop at end of turn."""
    bad = 0
    for arg in argv:
        path = Path(arg)
        if not (path / "tokenizer_config.json").exists():
            continue  # not a saved model dir (e.g. the pairs dataset)
        problems = stop_problems(path)
        print(f"[stops] {'FAIL' if problems else 'ok  '} {path}")
        for p in problems:
            print(f"        {p}")
        bad += bool(problems)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
