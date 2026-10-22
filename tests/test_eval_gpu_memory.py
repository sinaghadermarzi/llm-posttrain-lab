"""Eval gives the GPU back between checkpoints.

Each checkpoint starts its own vLLM engine (a separate process asking for 85%
of the GPU). On the pod, the second checkpoint's engine found 36.8 of 79 GiB
free (the first checkpoint's KL models, still cached by PyTorch in the eval
process) and refused to start, so every eval shard died after one checkpoint.
The engine is now shut down when sampling ends, and cached memory is freed
after KL, after every metric group and before every engine starts."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from test_end_of_turn import ENDOFTEXT, IM_END, save_checkpoint, serve_tokenizer


class _Engine:
    def __init__(self, events: list[str], fail: bool = False):
        self.events, self.fail = events, fail

    def shutdown(self) -> None:
        self.events.append("shutdown")
        if self.fail:
            raise RuntimeError("engine already gone")


def _fake_vllm(monkeypatch, events: list[str], chat_error: Exception | None = None, shutdown_fails: bool = False):
    class LLM:
        def __init__(self, model, dtype, gpu_memory_utilization):
            events.append("init")
            self.llm_engine = types.SimpleNamespace(engine_core=_Engine(events, shutdown_fails))

        def chat(self, messages, params):
            events.append("chat")
            if chat_error:
                raise chat_error
            done = types.SimpleNamespace(text="#### 1", finish_reason="stop", stop_reason=IM_END, token_ids=[IM_END])
            return [types.SimpleNamespace(outputs=[done] * params.n) for _ in messages]

    class SamplingParams:
        def __init__(self, n, **kw):
            self.n = n

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=LLM, SamplingParams=SamplingParams))


def _cfg():
    return OmegaConf.create({"seed": 1, "model": {"dtype": "bfloat16"},
                             "eval": {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 8}})


PROMPTS = [{"prompt_id": f"p{i}", "prompt": "q", "messages": [{"role": "user", "content": "q"}]} for i in range(3)]


def _generate(monkeypatch, events: list[str]):
    from posttrain_lab.eval import generate

    monkeypatch.setattr(generate, "free_gpu_memory", lambda: events.append("free"))
    return generate._vllm_generate(_cfg(), "ckpt", PROMPTS, 2, [IM_END, ENDOFTEXT])


def test_every_checkpoint_shuts_its_engine_down(monkeypatch) -> None:
    events: list[str] = []
    _fake_vllm(monkeypatch, events)
    rows = _generate(monkeypatch, events)
    assert len(rows) == 3 * 2 and all(r["finished"] and r["stop_token_id"] == IM_END for r in rows)
    # free what this process caches before the engine starts; stop it and free again once sampled
    assert events == ["free", "init", "chat", "shutdown", "free"]
    events.clear()
    _generate(monkeypatch, events)                        # the next checkpoint: the same cycle
    assert events == ["free", "init", "chat", "shutdown", "free"]


def test_a_failed_sampling_still_shuts_the_engine_down(monkeypatch) -> None:
    events: list[str] = []
    _fake_vllm(monkeypatch, events, chat_error=RuntimeError("CUDA error"))
    with pytest.raises(RuntimeError, match="CUDA error"):
        _generate(monkeypatch, events)
    assert events == ["free", "init", "chat", "shutdown", "free"]


def test_a_failed_shutdown_keeps_the_samples(monkeypatch, capsys) -> None:
    events: list[str] = []
    _fake_vllm(monkeypatch, events, shutdown_fails=True)
    assert len(_generate(monkeypatch, events)) == 6
    assert "vLLM engine shutdown failed" in capsys.readouterr().out


def test_shutdown_tolerates_engines_without_a_core() -> None:
    from posttrain_lab.utils.gpu import free_gpu_memory, shutdown_vllm

    shutdown_vllm(object())
    shutdown_vllm(types.SimpleNamespace(llm_engine=types.SimpleNamespace(engine_core=None)))
    free_gpu_memory()                                      # no CUDA here: collects and returns


def test_kl_frees_its_models(monkeypatch, tmp_path: Path) -> None:
    import gc

    import torch

    from posttrain_lab.eval import kl

    ckpt = save_checkpoint(tmp_path / "ckpt", [IM_END, ENDOFTEXT], fixed=True, with_model=True)
    serve_tokenizer(monkeypatch, fixed=True)
    live = []

    def free() -> None:  # by now the policy and reference must be unreachable
        gc.collect()
        live.append(sum(type(o).__name__.endswith("ForCausalLM") and isinstance(o, torch.nn.Module)
                        for o in gc.get_objects()))

    monkeypatch.setattr(kl, "free_gpu_memory", free)
    cfg = OmegaConf.create({"model": {"dtype": "float32"}, "eval": {"reference": str(ckpt), "logprob_batch_size": 4}})
    out = kl.compute_kl(cfg, str(ckpt), [{"prompt": "q", "response": "hi there", "finished": True}])
    assert out["kl_k3"] == pytest.approx(0.0, abs=1e-6)
    assert live == [0]


def test_memory_is_freed_after_every_metric_group(monkeypatch, tmp_path: Path) -> None:
    from posttrain_lab.eval import groups, run_eval

    calls: list[str] = []

    def fake(name):
        def compute(cfg, ckpt, ctx):
            calls.append(name)
            return {key: 1.0 for key in groups.SENTINELS[name]}
        return compute

    for name in ("retention_ppl", "pref_logps"):
        monkeypatch.setitem(run_eval.GROUPS, name, (groups.SENTINELS[name], fake(name)))
    monkeypatch.setattr(run_eval, "free_gpu_memory", lambda: calls.append("free"))
    ckpt = tmp_path / "runs" / "dpo_beta0.1_seed17" / "checkpoint-58"
    ckpt.mkdir(parents=True)
    cfg = OmegaConf.create({"eval": {"out_dir": str(tmp_path / "eval"), "metric_groups": ["retention_ppl", "pref_logps"],
                                     "samples_per_prompt": 8}, "task": "math", "train": {"report_to": "none"}})
    run_eval.evaluate_checkpoint(cfg, ckpt, [{"prompt_id": "p0"}])
    assert calls == ["retention_ppl", "free", "pref_logps", "free"]
    row = json.loads((tmp_path / "eval" / "dpo_beta0.1_seed17" / "checkpoint-58.jsonl").read_text())
    assert set(groups.SENTINELS["pref_logps"]) <= set(row)
