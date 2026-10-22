"""Give GPU memory back between eval steps.

Eval runs many checkpoints in one process, and each one starts its own vLLM
engine, which runs in a separate process and asks for a fixed share of the
GPU (gpu_memory_utilization). Memory PyTorch keeps cached in the eval process
(the KL step's policy + reference models and their activations) counts as used
there: the second checkpoint's engine then found 36.8 of 79 GiB free and
refused to start ("Free memory on device ... is less than desired GPU memory
utilization"), killing every eval shard after its first checkpoint.
"""

from __future__ import annotations

import gc


def free_gpu_memory() -> None:
    """Collect unreachable objects (models of the step just finished), then
    return PyTorch's cached CUDA blocks to the driver. A no-op without CUDA."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def shutdown_vllm(llm) -> None:
    """Stop a vLLM LLM's engine core now, so its GPU memory is free before the
    next engine starts, instead of whenever garbage collection reaches it."""
    core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
    if core is None or not hasattr(core, "shutdown"):
        return
    try:
        core.shutdown()
    except Exception as exc:  # never lose a finished checkpoint's samples over the cleanup
        print(f"[gpu] vLLM engine shutdown failed ({exc!r}); its memory is freed when it is collected")
