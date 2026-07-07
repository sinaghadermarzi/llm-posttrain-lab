"""Sample N responses per eval prompt from one policy checkpoint.

Sampling is at the policy's own distribution (temperature/top_p from
eval.yaml, 1.0/1.0 by default) — a requirement for the Monte-Carlo KL
estimate to be unbiased, not a style choice.

Uses vLLM when available and enabled; falls back to batched HF generate
(the smoke path, and any box without vllm installed).
"""

from __future__ import annotations

from omegaconf import DictConfig


def sample_responses(cfg: DictConfig, checkpoint: str, prompts: list[dict]) -> list[dict]:
    """Returns rows: {prompt_id, prompt, sample_idx, response}."""
    n = int(cfg.eval.samples_per_prompt)
    if bool(cfg.eval.use_vllm):
        try:
            return _vllm_generate(cfg, checkpoint, prompts, n)
        except ImportError:
            print("[generate] vllm not installed, falling back to HF generate")
    return _hf_generate(cfg, checkpoint, prompts, n)


def _vllm_generate(cfg, checkpoint, prompts, n):
    from vllm import LLM, SamplingParams

    llm = LLM(model=checkpoint, dtype=str(cfg.model.dtype), gpu_memory_utilization=0.85)
    params = SamplingParams(
        n=n,
        temperature=float(cfg.eval.temperature),
        top_p=float(cfg.eval.top_p),
        max_tokens=int(cfg.eval.max_new_tokens),
        seed=int(cfg.seed),
    )
    outputs = llm.chat([p["messages"] for p in prompts], params)
    rows = []
    for prompt, out in zip(prompts, outputs):
        for i, completion in enumerate(out.outputs):
            rows.append(
                {
                    "prompt_id": prompt["prompt_id"],
                    "prompt": prompt["prompt"],
                    "sample_idx": i,
                    "response": completion.text,
                }
            )
    return rows


def _hf_generate(cfg, checkpoint, prompts, n):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=getattr(torch, str(cfg.model.dtype)) if device == "cuda" else torch.float32
    ).to(device).eval()

    rows = []
    batch_size = int(cfg.eval.get("logprob_batch_size", 8))
    texts = [
        tokenizer.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    torch.manual_seed(int(cfg.seed))
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        enc = tokenizer(texts[start : start + batch_size], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=float(cfg.eval.temperature),
                top_p=float(cfg.eval.top_p),
                max_new_tokens=int(cfg.eval.max_new_tokens),
                num_return_sequences=n,
                pad_token_id=tokenizer.pad_token_id,
            )
        completions = out[:, enc["input_ids"].shape[1] :]
        for j, prompt in enumerate(batch_prompts):
            for i in range(n):
                text = tokenizer.decode(completions[j * n + i], skip_special_tokens=True)
                rows.append(
                    {
                        "prompt_id": prompt["prompt_id"],
                        "prompt": prompt["prompt"],
                        "sample_idx": i,
                        "response": text,
                    }
                )
    return rows
