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

Pointing eos at <|im_end|> is not enough on its own: Qwen2.5 base never
trained the <|im_start|>/<|im_end|> embeddings, so pi_SFT still gave
<|im_end|> only 0.12 where a turn ends (the second full math run). SFT now
re-initializes them first (train/common.py:init_untrained_chat_tokens), and
`--probe STAGE_YAML` measures the probability on the training data after
base SFT (run_all_*.sh gate on it; status shows it).

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


def template_token_ids(tokenizer) -> list[int]:
    """The special/added tokens the chat template writes around a turn
    (<|im_start|>, <|im_end|> for Qwen)."""
    if not getattr(tokenizer, "chat_template", None):
        return []
    try:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], tokenize=True
        )
    except Exception:
        return []
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    specials = set(tokenizer.all_special_ids) | set(tokenizer.get_added_vocab().values())
    return sorted({int(i) for i in ids if int(i) in specials})


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


PROBE_FILE = "stop_probe.json"
PROBE_MIN_P = 0.8


def probe_min_p(stage_yaml: str) -> float:
    """The gate for a stage's stop probe: its config's stop_probe_min_p, else
    PROBE_MIN_P. Math answers end at one unambiguous point ('#### N'), so a
    model that has learned to stop puts ~0.98 there; open-ended chat answers
    have no single right place to stop (configs/helpful/stage1_sft_base.yaml)."""
    from posttrain_lab.config import load_config

    return float(load_config(stage_yaml, []).get("stop_probe_min_p", PROBE_MIN_P))


def probe_conversations(stage_yaml: str, n: int = 64) -> list[list[dict]]:
    """The first n training conversations of a stage (ending in an assistant
    turn): what pi_SFT was just trained to end."""
    from posttrain_lab.config import load_config
    from posttrain_lab.data import sft_dataset

    cfg = load_config(stage_yaml, [f"data.max_samples={n}"])
    rows = [list(m) for m in sft_dataset(cfg)["messages"]]
    return [m for m in rows if m and m[-1]["role"] == "assistant"][:n]


def end_of_turn_probe(checkpoint: str | Path, conversations: list[list[dict]], max_tokens: int = 2048) -> dict:
    """How likely the saved model is to emit the end-of-turn token right where
    each conversation's last assistant turn ends. A model that has learned to
    stop puts nearly all its mass there; the first full math run's pi_SFT put
    0.12 (untrained <|im_end|> embedding, train/common.py:untrained_rows), so
    most samples ran on into invented turns."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    eot = end_of_turn_token(tokenizer)
    if eot is None:
        return {}
    eot_id = tokenizer.convert_tokens_to_ids(eot)
    stops = stop_token_ids(tokenizer, checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint), dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device).eval()
    p_eot, p_stop = [], []
    for messages in conversations:
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        cut = text.rfind(eot)
        if cut < 0:
            continue
        ids = tokenizer(text[:cut], add_special_tokens=False, return_tensors="pt").input_ids[:, -max_tokens:]
        with torch.no_grad():
            probs = model(ids.to(device)).logits[0, -1].float().softmax(-1)
        p_eot.append(probs[eot_id].item())
        p_stop.append(probs[stops].sum().item())
    if not p_eot:
        return {}
    return {"token": eot, "n": len(p_eot), "mean": sum(p_eot) / len(p_eot), "min": min(p_eot),
            "max": max(p_eot), "mean_any_stop": sum(p_stop) / len(p_stop)}


def _cli(argv: list[str]) -> int:
    """python -m posttrain_lab.utils.chat RUN_DIR... [--probe STAGE_YAML] :
    nonzero if any saved run (its tokenizer + generation_config) would not
    stop at end of turn. --probe also measures it (end_of_turn_probe) on the
    stage's training conversations, writes RUN_DIR/stop_probe.json, and fails
    below --min-p (default: the stage's stop_probe_min_p, else PROBE_MIN_P) unless
    --report-only (smoke: a few steps cannot learn it)."""
    import argparse

    ap = argparse.ArgumentParser(prog="python -m posttrain_lab.utils.chat")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--probe", metavar="STAGE_YAML", help="measure P(end of turn) on this stage's training data")
    ap.add_argument("--min-p", type=float, default=None,
                    help=f"gate (default: the stage yaml's stop_probe_min_p, else {PROBE_MIN_P})")
    ap.add_argument("--report-only", action="store_true", help="print the probe, never fail on it")
    args = ap.parse_args(argv)
    if args.min_p is None:
        args.min_p = probe_min_p(args.probe) if args.probe else PROBE_MIN_P
    conversations = None
    bad = 0
    for arg in args.runs:
        path = Path(arg)
        if not (path / "tokenizer_config.json").exists():
            continue  # not a saved model dir (e.g. the pairs dataset)
        problems = stop_problems(path)
        print(f"[stops] {'FAIL' if problems else 'ok  '} {path}")
        for p in problems:
            print(f"        {p}")
        bad += bool(problems)
        if args.probe:
            if conversations is None:
                conversations = probe_conversations(args.probe)
            result = end_of_turn_probe(path, conversations)
            if not result:
                continue
            result["min_p"] = args.min_p
            (path / PROBE_FILE).write_text(json.dumps(result, indent=2) + "\n")
            low = result["mean"] < args.min_p
            tag = ("low " if args.report_only else "FAIL") if low else "ok  "
            print(f"[stops] {tag} probe: P({result['token']}) where "
                  f"{result['n']} training answers end: mean {result['mean']:.3f} (min {result['min']:.3f}, "
                  f"need >= {args.min_p}); any stop token: {result['mean_any_stop']:.3f}")
            if low:
                # gate on the end-of-turn token itself: GRPO rollouts stop and TRL masks only there
                # (train/grpo.py); eval and pair sampling also stop at the other ids (mean_any_stop)
                print(f"        {1 - result['mean']:.0%} of samples would not end the turn with {result['token']} "
                      f"there (GRPO's only clean stop); {1 - result['mean_any_stop']:.0%} would not stop at all "
                      "and run on into invented turns until max_new_tokens")
            bad += low and not args.report_only
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
