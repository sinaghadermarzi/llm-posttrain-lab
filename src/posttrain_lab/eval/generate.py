"""Sample N responses per eval prompt from one policy checkpoint.

Sampling is at the policy's own distribution (temperature/top_p from
eval.yaml, 1.0/1.0 by default) — a requirement for the Monte-Carlo KL
estimate to be unbiased, not a style choice.

Generation stops at the chat template's end-of-turn token (utils/chat.py),
passed explicitly so checkpoints saved with a base-model eos stop too. Each
row records `finished` (False: it hit max_new_tokens) and `stop_token_id`,
the id it actually stopped on, which KL scores (eval/kl.py).

Uses vLLM when available and enabled; falls back to batched HF generate
(the smoke path, and any box without vllm installed).
"""

from __future__ import annotations

from omegaconf import DictConfig

from posttrain_lab.utils.gpu import free_gpu_memory, shutdown_vllm


def sample_responses(cfg: DictConfig, checkpoint: str, prompts: list[dict]) -> list[dict]:
    """Returns rows: {prompt_id, prompt, sample_idx, response, finished, stop_token_id}."""
    n = int(cfg.eval.samples_per_prompt)
    stops = _stop_ids(checkpoint)
    if bool(cfg.eval.use_vllm):
        try:
            rows = _vllm_generate(cfg, checkpoint, prompts, n, stops)
        except ImportError:
            print("[generate] vllm not installed, falling back to HF generate")
        else:
            _report(rows, cfg)
            return rows
    rows = _hf_generate(cfg, checkpoint, prompts, n, stops)
    _report(rows, cfg)
    return rows


def _stop_ids(checkpoint: str) -> list[int]:
    from transformers import AutoTokenizer

    from posttrain_lab.utils.chat import stop_token_ids

    return stop_token_ids(AutoTokenizer.from_pretrained(checkpoint), checkpoint)


def vllm_stop_token(completion, stops: list[int]) -> int | None:
    """The id a vLLM completion stopped on: stop_reason for a stop_token_ids
    stop; an EOS stop leaves stop_reason None, so then the last generated id.
    None if it did not stop (or stopped on something else)."""
    if completion.finish_reason != "stop":
        return None
    if isinstance(completion.stop_reason, int):
        return completion.stop_reason
    ids = list(getattr(completion, "token_ids", None) or [])
    return ids[-1] if ids and ids[-1] in stops else None


def _report(rows: list[dict], cfg) -> None:
    unfinished = sum(not r["finished"] for r in rows)
    print(f"[generate] {len(rows)} samples, {unfinished} ({unfinished / max(len(rows), 1):.0%}) "
          f"hit max_new_tokens={int(cfg.eval.max_new_tokens)} without stopping")


def _vllm_generate(cfg, checkpoint, prompts, n, stops):
    from vllm import LLM, SamplingParams

    free_gpu_memory()  # the engine runs in its own process: it can't use what this one holds cached
    llm = LLM(model=checkpoint, dtype=str(cfg.model.dtype), gpu_memory_utilization=0.85)
    try:
        return _vllm_rows(llm, SamplingParams, cfg, prompts, n, stops)
    finally:
        shutdown_vllm(llm)  # free its share now, before the next checkpoint's engine starts
        del llm
        free_gpu_memory()


def _vllm_rows(llm, SamplingParams, cfg, prompts, n, stops):
    params = SamplingParams(
        n=n,
        temperature=float(cfg.eval.temperature),
        top_p=float(cfg.eval.top_p),
        max_tokens=int(cfg.eval.max_new_tokens),
        seed=int(cfg.seed),
        stop_token_ids=stops,
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
                    "finished": completion.finish_reason == "stop",
                    "stop_token_id": vllm_stop_token(completion, stops),
                }
            )
    return rows


def _hf_generate(cfg, checkpoint, prompts, n, stops):
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
                eos_token_id=stops,
            )
        completions = out[:, enc["input_ids"].shape[1] :]
        for j, prompt in enumerate(batch_prompts):
            for i in range(n):
                ids = completions[j * n + i].tolist()
                cut = next((k for k, t in enumerate(ids) if t in stops), None)
                text = tokenizer.decode(ids[:cut], skip_special_tokens=True)
                rows.append(
                    {
                        "prompt_id": prompt["prompt_id"],
                        "prompt": prompt["prompt"],
                        "sample_idx": i,
                        "response": text,
                        "finished": cut is not None,
                        "stop_token_id": ids[cut] if cut is not None else None,
                    }
                )
    return rows
