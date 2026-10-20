"""Sequence-level KL(policy || reference) from sampled responses.

For each sampled response (drawn from the policy), response-token logprobs are
computed under both the policy and the reference (pi_SFT) with two exact,
batched HF forward passes — no approximations from the generation engine.

Per-token estimators of KL(pi || pi_ref), with r = pi_ref(t) / pi(t):
  k1 = -log r            (unbiased, high variance, can go negative)
  k3 = r - 1 - log r     (unbiased, low variance, always >= 0; Schulman 2020,
                          the same estimator GRPO uses for its KL term)
Sequence KL = sum of per-token values; we report the mean over sequences.
A response that stopped is scored with the token it stopped on
(row["stop_token_id"]): deciding to stop is part of the policy's distribution.

Also returns mean response length (tokens) and mean sampled-token negative
logprob under the policy (an MC estimate of token entropy) — the cheap
mode-collapse diagnostics.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from posttrain_lab.utils.chat import end_of_turn_token


def response_logprobs(model, input_ids, attention_mask, response_mask):
    """Logprob of each realized token, zeroed outside the response span."""
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    target = input_ids[:, 1:].unsqueeze(-1)
    token_logprobs = logprobs.gather(-1, target).squeeze(-1)
    return token_logprobs * response_mask[:, 1:]


def scored_response(sample: dict, tokenizer, eot: str) -> str:
    """The response text to score: with the token it actually stopped on, if it
    stopped. Rows from before stop_token_id was recorded fall back to the
    template's end-of-turn token."""
    if not sample.get("finished"):
        return sample["response"]
    if "stop_token_id" in sample:
        stop = sample["stop_token_id"]
        return sample["response"] + (tokenizer.convert_ids_to_tokens(stop) if stop is not None else "")
    return sample["response"] + eot


def compute_kl(cfg: DictConfig, policy_path: str, samples: list[dict]) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, str(cfg.model.dtype)) if device == "cuda" else torch.float32
    ref_path = str(cfg.eval.reference)

    tokenizer = AutoTokenizer.from_pretrained(policy_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    eot = end_of_turn_token(tokenizer) or ""

    policy = AutoModelForCausalLM.from_pretrained(policy_path, torch_dtype=dtype).to(device).eval()
    reference = AutoModelForCausalLM.from_pretrained(ref_path, torch_dtype=dtype).to(device).eval()

    seq_k1, seq_k3, lengths, entropies = [], [], [], []
    batch_size = int(cfg.eval.logprob_batch_size)
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        prompt_texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": s["prompt"]}], tokenize=False, add_generation_prompt=True
            )
            for s in batch
        ]
        full_texts = [p + scored_response(s, tokenizer, eot) for p, s in zip(prompt_texts, batch)]
        enc = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        prompt_lens = [len(tokenizer(p)["input_ids"]) for p in prompt_texts]

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        response_mask = attention_mask.clone()
        for i, plen in enumerate(prompt_lens):
            response_mask[i, :plen] = 0

        lp_pi = response_logprobs(policy, input_ids, attention_mask, response_mask)
        lp_ref = response_logprobs(reference, input_ids, attention_mask, response_mask)

        mask = response_mask[:, 1:].bool()
        log_r = (lp_ref - lp_pi) * mask  # log(pi_ref / pi) per response token
        k1 = (-log_r).sum(dim=1)
        k3 = ((torch.exp(log_r) - 1 - log_r) * mask).sum(dim=1)
        n_tokens = mask.sum(dim=1).clamp(min=1)

        seq_k1.extend(k1.tolist())
        seq_k3.extend(k3.tolist())
        lengths.extend(mask.sum(dim=1).tolist())
        entropies.extend(((-lp_pi * mask).sum(dim=1) / n_tokens).tolist())

    n = max(len(seq_k3), 1)
    return {
        "kl_k3": sum(seq_k3) / n,
        "kl_k1": sum(seq_k1) / n,
        "mean_response_tokens": sum(lengths) / n,
        "mean_token_nll": sum(entropies) / n,  # MC estimate of policy token entropy
    }
