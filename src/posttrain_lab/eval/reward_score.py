"""Score (prompt, response) pairs with a sequence-classification reward model.

Used for both graders of every frontier point:
  - the stage-2 training RM (the reward that PPO/GRPO optimized), and
  - the held-out judge (Skywork-Reward-V2-Llama-3.1-8B), which none of the
    methods ever saw.
A checkpoint that pleases its own RM but not the judge is reward hacking;
that gap is one of the deliverable diagnostics.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def score_samples(cfg: DictConfig, rm_path: str, samples: list[dict], batch_size: int) -> list[float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, str(cfg.model.dtype)) if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(rm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        rm_path, num_labels=1, torch_dtype=dtype
    ).to(device).eval()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    scores: list[float] = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": s["prompt"]},
                    {"role": "assistant", "content": s["response"]},
                ],
                tokenize=False,
            )
            for s in batch
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        scores.extend(logits.squeeze(-1).float().tolist())

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return scores
