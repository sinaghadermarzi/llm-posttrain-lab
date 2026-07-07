"""Held-out chosen/rejected log-probabilities per checkpoint, for EVERY arm.

DPO's training logs already show its own logps drifting, but only an
eval-time pass on held-out pairs lets the arms be compared on equal footing:
does RL also move probability mass off the chosen responses, or is the
"squeezing effect" (logp of chosen AND rejected both falling while the margin
grows) specific to DPO's off-policy objective?

Pairs: helpful = test_prefs; math = the held-out split of the m2
self-generated pairs. Reuses kl.response_logprobs for exact token logprobs.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from posttrain_lab.data import pref_dataset
from posttrain_lab.eval.kl import response_logprobs


def _completion_logps(cfg, model, tokenizer, device, prompts, completions) -> tuple[list[float], list[float]]:
    """Sum and per-token-mean logp of each completion given its prompt."""
    sums, means = [], []
    batch_size = int(cfg.eval.logprob_batch_size)
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_completions = completions[start : start + batch_size]
        prompt_texts = [
            tokenizer.apply_chat_template(list(p), tokenize=False, add_generation_prompt=True)
            for p in batch_prompts
        ]
        full_texts = [pt + c[-1]["content"] for pt, c in zip(prompt_texts, batch_completions)]
        enc = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        prompt_lens = [len(tokenizer(pt)["input_ids"]) for pt in prompt_texts]

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        response_mask = attention_mask.clone()
        for i, plen in enumerate(prompt_lens):
            response_mask[i, :plen] = 0

        lp = response_logprobs(model, input_ids, attention_mask, response_mask)
        mask = response_mask[:, 1:].bool()
        n_tokens = mask.sum(dim=1).clamp(min=1)
        sums.extend(lp.sum(dim=1).tolist())
        means.extend((lp.sum(dim=1) / n_tokens).tolist())
    return sums, means


def compute_group(cfg, ckpt, ctx) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, str(cfg.model.dtype)) if device == "cuda" else torch.float32

    pairs = pref_dataset(cfg, split=str(cfg.eval.pref_logps.split))
    num = int(cfg.eval.pref_logps.num_pairs)
    if num < len(pairs):
        pairs = pairs.shuffle(seed=int(cfg.seed)).select(range(num))

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(str(ckpt), torch_dtype=dtype).to(device).eval()

    prompts = pairs["prompt"]
    chosen_sum, chosen_mean = _completion_logps(cfg, model, tokenizer, device, prompts, pairs["chosen"])
    rejected_sum, rejected_mean = _completion_logps(cfg, model, tokenizer, device, prompts, pairs["rejected"])

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    n = max(len(chosen_sum), 1)
    logp_chosen = sum(chosen_sum) / n
    logp_rejected = sum(rejected_sum) / n
    return {
        "logp_chosen": logp_chosen,
        "logp_rejected": logp_rejected,
        "logp_margin": logp_chosen - logp_rejected,
        "logp_chosen_per_token": sum(chosen_mean) / n,
        "logp_rejected_per_token": sum(rejected_mean) / n,
        "num_pref_pairs": len(chosen_sum),
    }
