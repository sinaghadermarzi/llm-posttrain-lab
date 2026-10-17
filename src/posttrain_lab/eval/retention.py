"""Capability retention — does preference/reward optimization erode what the
model knew? Two groups:

  retention_ppl      perplexity on wikitext-2-raw-v1 test (general English the
                     model was never post-trained on; eval.retention.dataset).
                     Cheap: runs on every checkpoint. Rising PPL = the base
                     distribution eroding.
  retention_harness  lm-eval benchmarks (hellaswag, arc_easy, arc_challenge by
                     default) — benchmark-grade evidence. eval.harness.which
                     selects coverage: 'all' checkpoints (default; user opted
                     for full curves) or 'final' only.
"""

from __future__ import annotations

import torch


def ppl_group(cfg, ckpt, ctx) -> dict:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, str(cfg.model.dtype)) if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForCausalLM.from_pretrained(str(ckpt), torch_dtype=dtype).to(device).eval()

    retention = cfg.eval.retention
    if retention.get("data_files"):
        rows = load_dataset("parquet", data_files=str(retention.data_files), split="train")
    else:
        rows = load_dataset(str(retention.dataset), str(retention.config), split=str(retention.split))
    text = "\n\n".join(row["text"] for row in rows)
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0][: int(cfg.eval.retention.max_tokens)]

    window = 1024
    nll_sum, n_tokens = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(ids) - 1, window):
            chunk = ids[start : start + window + 1].unsqueeze(0).to(device)
            if chunk.shape[1] < 2:
                break
            logits = model(input_ids=chunk[:, :-1]).logits.float()
            logprobs = torch.log_softmax(logits, dim=-1)
            token_lp = logprobs.gather(-1, chunk[:, 1:].unsqueeze(-1)).squeeze(-1)
            nll_sum += -token_lp.sum().item()
            n_tokens += chunk.shape[1] - 1

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"ppl_wikitext": float(torch.tensor(nll_sum / max(n_tokens, 1)).exp())}


def harness_group(cfg, ckpt, ctx) -> dict:
    if str(cfg.eval.harness.which) == "final" and not ctx["is_final"]:
        return {"harness": None}  # sentinel present -> group counts as done

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm = HFLM(
        pretrained=str(ckpt),
        batch_size=8,
        dtype=str(cfg.model.dtype) if device == "cuda" else "float32",
        device=device,
    )
    limit = cfg.eval.harness.limit
    result = simple_evaluate(
        model=lm,
        tasks=list(cfg.eval.harness.tasks),
        limit=int(limit) if limit is not None else None,
    )
    scores = {}
    for task, metrics in result["results"].items():
        acc = metrics.get("acc_norm,none", metrics.get("acc,none"))
        if acc is not None:
            scores[task] = float(acc)
    return {"harness": scores}
