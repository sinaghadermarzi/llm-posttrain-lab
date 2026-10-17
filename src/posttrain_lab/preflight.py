"""Preflight checks for the training node — each piece can be run on its own.

    python -m posttrain_lab.preflight                  # every default check
    python -m posttrain_lab.preflight gpu mlflow       # just these
    python -m posttrain_lab.preflight --list           # names + descriptions

From a notebook (see notebooks/preflight.ipynb):

    from posttrain_lab import preflight
    preflight.run("gpu")

Levels: OK; INFO (expected / for the record); WARN (works, but look at it);
FAIL (a pipeline stage will break). Exit code is 1 if anything FAILs. A check
that crashes reports FAIL and the remaining checks still run. Heavy imports
(torch, vllm, ...) happen inside each check, so --list is instant.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from posttrain_lab.config import CONFIG_DIR, load_config

REPO_ROOT = CONFIG_DIR.parent
Result = list[tuple[str, str]]


# ---------------------------------------------------------------- packages


def check_packages() -> Result:
    """Installed versions vs requirements.txt pins (the pins are load-bearing)."""
    from importlib.metadata import PackageNotFoundError, version

    def installed(name: str) -> str | None:
        for candidate in {name, name.replace("-", "_"), name.replace("_", "-")}:
            try:
                return version(candidate)
            except PackageNotFoundError:
                continue
        return None

    out: Result = [("INFO", f"python {sys.version.split()[0]} ({sys.executable})")]
    for line in (REPO_ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, _, pin = line.partition("==")
        have = installed(name)
        if have is None and name == "mlflow-skinny":
            have = installed("mlflow")  # the full package provides the same client
        if have is None:
            out.append(("FAIL", f"{name}: not installed"))
        elif pin and have != pin:
            out.append(("WARN", f"{name}: {have} installed, requirements.txt pins {pin}"))
        else:
            out.append(("OK", f"{name} {have}"))
    return out


# ------------------------------------------------------------------ system


def check_system() -> Result:
    """CPU/RAM, /dev/shm (NCCL + dataloaders need it; pods default to 64MB),
    and free disk for results and the HF cache."""
    out: Result = [("INFO", f"host {socket.gethostname()}, {os.cpu_count()} CPUs")]
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        out.append(("INFO", f"RAM {ram:.0f} GB"))
    except (ValueError, OSError):
        pass

    if Path("/dev/shm").exists():
        shm = shutil.disk_usage("/dev/shm").total / 2**30
        level = "OK" if shm >= 8 else "WARN"
        hint = "" if level == "OK" else " — mount a larger emptyDir (medium: Memory) at /dev/shm"
        out.append((level, f"/dev/shm {shm:.1f} GB{hint}"))
    else:
        out.append(("INFO", "/dev/shm not present (not Linux?)"))

    # README budget: ~250 GB helpful + ~60 GB math results, plus ~20 GB of models
    cfg = load_config("helpful/stage1_sft_base.yaml", [])
    targets = {"results": REPO_ROOT / Path(cfg.results_root).parts[0], "HF cache": _hf_home()}
    for label, path in targets.items():
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        free = shutil.disk_usage(probe).free / 2**30
        level = "OK" if free >= 330 else "WARN"
        out.append((level, f"{label} ({path}): {free:.0f} GB free (budget ~330 GB for both tracks)"))
    return out


# --------------------------------------------------------------------- gpu


def check_gpu() -> Result:
    """CUDA devices, bf16, a matmul benchmark per GPU, P2P and an NCCL all-reduce."""
    import torch

    if not torch.cuda.is_available():
        return [("FAIL", f"torch {torch.__version__}: CUDA not available")]
    n = torch.cuda.device_count()
    nccl = ".".join(map(str, torch.cuda.nccl.version())) if hasattr(torch.cuda, "nccl") else "?"
    out: Result = [
        ("INFO", f"torch {torch.__version__}, CUDA {torch.version.cuda}, "
                 f"cuDNN {torch.backends.cudnn.version()}, NCCL {nccl}"),
        ("INFO", f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"),
        ("OK" if n >= 4 else "WARN", f"{n} GPU(s) visible (scripts assume 4)"),
    ]
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        torch.cuda.set_device(i)
        bf16 = torch.cuda.is_bf16_supported()
        out.append(("OK" if bf16 else "FAIL",
                    f"GPU {i}: {props.name}, sm_{props.major}{props.minor}, "
                    f"{free / 2**30:.0f}/{total / 2**30:.0f} GB free, bf16={'yes' if bf16 else 'NO'}"))
        tflops = _matmul_tflops(i)
        out.append(("OK" if tflops >= 100 else "WARN",
                    f"GPU {i}: bf16 matmul {tflops:.0f} TFLOPS (A100 typically 200-280)"))

    if n > 1:
        missing = [(i, j) for i in range(n) for j in range(n)
                   if i != j and not torch.cuda.can_device_access_peer(i, j)]
        out.append(("OK", "P2P access between all GPU pairs") if not missing
                   else ("WARN", f"no P2P for pairs {missing} — NCCL falls back to slower paths"))
        out.extend(_nccl_allreduce(n))
    return out


def _matmul_tflops(device: int, size: int = 8192, iters: int = 10) -> float:
    import torch

    a = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    b = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    a @ b  # warmup
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize(device)
    return 2 * size**3 * iters / (time.perf_counter() - start) / 1e12


def _allreduce_worker(rank: int, world: int, port: int, queue) -> None:
    import torch
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    x = torch.ones(64 * 2**20, device=rank, dtype=torch.float32)  # 256 MB
    dist.all_reduce(x)  # warmup + correctness
    correct = bool(torch.all(x == world).item())
    torch.cuda.synchronize()
    iters = 10
    start = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    if rank == 0:
        bus_bw = 2 * (world - 1) / world * x.numel() * 4 / elapsed / 1e9
        queue.put((correct, bus_bw))
    dist.destroy_process_group()


def _nccl_allreduce(world: int) -> Result:
    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    queue = mp.get_context("spawn").SimpleQueue()
    mp.spawn(_allreduce_worker, args=(world, port, queue), nprocs=world, join=True)
    correct, bus_bw = queue.get()
    if not correct:
        return [("FAIL", "NCCL all-reduce returned wrong values")]
    level = "OK" if bus_bw >= 50 else "WARN"
    return [(level, f"NCCL all-reduce across {world} GPUs OK, bus bandwidth {bus_bw:.0f} GB/s "
                    "(NVLink A100: ~150-250; PCIe: ~10-25)")]


# -------------------------------------------------- training/inference libs


def check_flash_attn() -> Result:
    """flash-attn import + a tiny kernel call (base.yaml asks for flash_attention_2)."""
    cfg = load_config("helpful/stage1_sft_base.yaml", [])
    wanted = str(cfg.model.attn_implementation) == "flash_attention_2"
    try:
        import flash_attn
        from flash_attn import flash_attn_func
    except ImportError as e:
        level = "FAIL" if wanted else "INFO"
        return [(level, f"flash-attn not importable ({e}); base.yaml attn_implementation="
                        f"{cfg.model.attn_implementation} — install it or set it to sdpa")]
    import torch

    out: Result = [("OK", f"flash-attn {flash_attn.__version__}")]
    if torch.cuda.is_available():
        q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
        flash_attn_func(q, q, q, causal=True)
        out.append(("OK", "flash_attn_func ran on GPU"))
    return out


def check_deepspeed() -> Result:
    """DeepSpeed import, accelerator, nvcc (needed if it JIT-builds ops)."""
    import deepspeed
    from deepspeed.accelerator import get_accelerator

    out: Result = [("OK", f"deepspeed {deepspeed.__version__}, accelerator={get_accelerator().device_name()}")]
    nvcc = shutil.which("nvcc")
    out.append(("OK", f"nvcc at {nvcc}") if nvcc
               else ("INFO", "nvcc not on PATH — fine unless DeepSpeed needs to JIT-build an op"))
    accel_cfg = REPO_ROOT / "configs/accelerate/zero2_4gpu.yaml"
    out.append(("OK" if accel_cfg.exists() else "FAIL", f"accelerate config {accel_cfg.relative_to(REPO_ROOT)}"))
    return out


def check_vllm() -> Result:
    """vLLM import (TRL 1.7 supports 0.13.0-0.23.0)."""
    import vllm

    return [("OK", f"vllm {vllm.__version__}")]


def check_vllm_generate() -> Result:
    """Load the smoke model in vLLM and generate a few tokens (slow: ~1 min)."""
    from vllm import LLM, SamplingParams

    model = str(load_config("helpful/stage1_sft_base.yaml", ["--smoke"]).model.base)
    llm = LLM(model=model, gpu_memory_utilization=0.3, max_model_len=512, enforce_eager=True)
    text = llm.generate(["2 + 2 ="], SamplingParams(max_tokens=8, temperature=0))[0].outputs[0].text
    return [("OK", f"vLLM generated with {model}: {text!r}")]


# --------------------------------------------------------- network / hub


def _probe(url: str, timeout: float = 5) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return True, f"HTTP {r.status}"
    except urllib.error.HTTPError as e:  # reached the server; it just said no
        return True, f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        reason = getattr(e, "reason", e)
        return False, f"{type(reason).__name__}: {reason}"


def check_internet() -> Result:
    """Outbound HTTPS to public hosts. Unreachable is expected on an offline cluster."""
    out: Result = []
    proxies = {k: v for k, v in os.environ.items() if k.lower() in ("http_proxy", "https_proxy", "no_proxy")}
    out.append(("INFO", f"proxy env: {proxies or 'none'}"))
    for url in ["https://pypi.org", "https://huggingface.co", "https://github.com", "https://download.pytorch.org"]:
        ok, detail = _probe(url)
        out.append(("OK" if ok else "INFO", f"{url}: {'reachable' if ok else 'unreachable'} ({detail})"))
    return out


def _hf_home() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HOME)


def _hub_assets() -> list[tuple[str, str, str]]:
    helpful = load_config("helpful/stage1_sft_base.yaml", [])
    math = load_config("math/m1_sft_gold.yaml", [])
    smoke = load_config("helpful/stage1_sft_base.yaml", ["--smoke"])
    return [
        ("model", str(helpful.model.base), "base model (both tracks)"),
        ("model", str(smoke.model.base), "smoke-test model"),
        ("model", str(helpful.eval.judge), "held-out judge (helpful eval)"),
        ("dataset", str(helpful.dataset.id), "helpful data"),
        ("dataset", str(math.dataset.id), "math data"),
        # hardcoded in eval/retention.py and lm-eval's task yamls
        ("dataset", "Salesforce/wikitext", "retention PPL (loaded as 'wikitext')"),
        ("dataset", "Rowan/hellaswag", "lm-eval hellaswag"),
        ("dataset", "allenai/ai2_arc", "lm-eval arc_easy / arc_challenge"),
    ]


def check_hf() -> Result:
    """Every Hub model/dataset the pipeline loads: in the local cache, or
    downloadable from HF_ENDPOINT? Missing from both = that stage fails offline."""
    from huggingface_hub import HfApi, constants, scan_cache_dir

    offline = constants.HF_HUB_OFFLINE
    out: Result = [
        ("INFO", f"HF_HOME={constants.HF_HOME}"),
        ("INFO", f"HF_ENDPOINT={constants.ENDPOINT}  HF_HUB_OFFLINE={int(offline)}  "
                 f"HF_DATASETS_OFFLINE={os.environ.get('HF_DATASETS_OFFLINE', '0')}  "
                 f"token={'set' if os.environ.get('HF_TOKEN') or Path(constants.HF_TOKEN_PATH).exists() else 'none'}"),
    ]
    try:
        cached = {(r.repo_type, r.repo_id) for r in scan_cache_dir().repos}
    except Exception:  # noqa: BLE001 — no cache dir yet
        cached = set()
    ds_cache = Path(os.environ.get("HF_DATASETS_CACHE", _hf_home() / "datasets"))

    reachable = False
    if not offline:
        reachable, detail = _probe(constants.ENDPOINT)
        out.append(("OK" if reachable else "INFO", f"{constants.ENDPOINT}: {'reachable' if reachable else 'unreachable'} ({detail})"))
    api = HfApi()
    for kind, repo, why in _hub_assets():
        names = {repo, repo.split("/")[-1]}  # 'wikitext' is cached under its short name
        in_cache = any((kind, name) in cached for name in names) or (
            kind == "dataset" and any(any(ds_cache.glob(name.replace("/", "___") + "*")) for name in names)
        )
        if in_cache:
            out.append(("OK", f"{kind} {repo}: in local cache — {why}"))
            continue
        if reachable:
            try:
                info = api.model_info if kind == "model" else api.dataset_info
                info(repo, timeout=10)
                out.append(("OK", f"{kind} {repo}: not cached, downloadable — {why}"))
                continue
            except Exception as e:  # noqa: BLE001
                out.append(("FAIL", f"{kind} {repo}: not cached and endpoint refused ({type(e).__name__}) — {why}"))
                continue
        out.append(("FAIL", f"{kind} {repo}: not cached and no reachable endpoint — {why}"))
    return out


# ------------------------------------------------------------------ mlflow


def check_mlflow() -> Result:
    """Resolve the tracking URI (.env / env / config), hit /health and /version,
    then write and read back a test run in experiment 'llm-posttrain-lab-preflight'."""
    import mlflow
    from mlflow.tracking import MlflowClient

    from posttrain_lab.utils.logging import _tracking_uri, tracking_enabled

    cfg = load_config("helpful/stage1_sft_base.yaml", [])
    out: Result = [("INFO", f"mlflow client {mlflow.__version__}; train.report_to={cfg.train.report_to}")]
    if not tracking_enabled(cfg):
        out.append(("WARN", "report_to does not include mlflow — training will not log"))
    uri = _tracking_uri(cfg)
    if not uri:
        return out + [("FAIL", f"no tracking URI: set MLFLOW_TRACKING_URI in {REPO_ROOT / '.env'} or the env")]
    out.append(("INFO", f"tracking URI: {uri}"))
    auth = [k for k in ("MLFLOW_TRACKING_TOKEN", "MLFLOW_TRACKING_USERNAME") if os.environ.get(k)]
    out.append(("INFO", f"auth env set: {auth or 'none'}"))

    if uri.startswith(("http://", "https://")):
        ok, detail = _probe(uri.rstrip("/") + "/health")
        if not ok:
            return out + [("FAIL", f"{uri}/health unreachable ({detail})")]
        out.append(("OK", f"/health {detail}"))
        try:
            with urllib.request.urlopen(uri.rstrip("/") + "/version", timeout=5) as r:
                server = r.read().decode().strip()
            same_major = server.split(".")[0] == mlflow.__version__.split(".")[0]
            out.append(("OK" if same_major else "WARN",
                        f"server {server} vs client {mlflow.__version__}"
                        + ("" if same_major else " — different major versions; pin the client to match")))
        except Exception as e:  # noqa: BLE001
            out.append(("INFO", f"/version not available ({type(e).__name__})"))

    client = MlflowClient(tracking_uri=uri)
    name = "llm-posttrain-lab-preflight"
    exp = client.get_experiment_by_name(name)
    exp_id = exp.experiment_id if exp else client.create_experiment(name)
    run = client.create_run(exp_id, run_name=f"preflight-{socket.gethostname()}", tags={"preflight": "1"})
    run_id = run.info.run_id
    client.log_param(run_id, "host", socket.gethostname())
    for step in range(3):
        client.log_metric(run_id, "preflight/value", float(step), step=step)
    client.set_terminated(run_id)
    history = [m.value for m in client.get_metric_history(run_id, "preflight/value")]
    if sorted(history) != [0.0, 1.0, 2.0]:
        return out + [("FAIL", f"wrote 3 metric points, read back {history}")]
    link = f"{uri.rstrip('/')}/#/experiments/{exp_id}/runs/{run_id}" if uri.startswith("http") else run_id
    out.append(("OK", f"wrote + read back a test run: {link}"))
    return out


# ------------------------------------------------------------------ runner

CHECKS: dict[str, Callable[[], Result]] = {
    "packages": check_packages,
    "system": check_system,
    "gpu": check_gpu,
    "flash_attn": check_flash_attn,
    "deepspeed": check_deepspeed,
    "vllm": check_vllm,
    "internet": check_internet,
    "hf": check_hf,
    "mlflow": check_mlflow,
}
EXTRA: dict[str, Callable[[], Result]] = {"vllm_generate": check_vllm_generate}  # opt-in: slow


def list_checks() -> None:
    for name, fn in {**CHECKS, **EXTRA}.items():
        tag = " (opt-in)" if name in EXTRA else ""
        summary = " ".join(fn.__doc__.split("\n\n")[0].split())
        print(f"  {name:14s}{tag:10s} {summary}")


def run(*names: str) -> bool:
    """Run the named checks (default: all of CHECKS). True if none FAILed."""
    names = names or tuple(CHECKS)
    unknown = [n for n in names if n not in CHECKS and n not in EXTRA]
    if unknown:
        raise ValueError(f"unknown check(s) {unknown}; choose from {list(CHECKS) + list(EXTRA)}")
    failed = []
    for name in names:
        print(f"[{name}]")
        start = time.perf_counter()
        try:
            results = {**CHECKS, **EXTRA}[name]()
        except Exception as e:  # noqa: BLE001
            tb = traceback.extract_tb(e.__traceback__)[-1]
            results = [("FAIL", f"{type(e).__name__}: {e}  ({Path(tb.filename).name}:{tb.lineno})")]
        for level, msg in results:
            print(f"  {level:5s} {msg}")
        if any(level == "FAIL" for level, _ in results):
            failed.append(name)
        print(f"  ({time.perf_counter() - start:.1f}s)\n")
    print("summary: " + (f"FAILED {failed}" if failed else "no failures"))
    return not failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checks", nargs="*", help="check names (default: all default checks)")
    parser.add_argument("--list", action="store_true", help="list checks and exit")
    args = parser.parse_args()
    if args.list:
        list_checks()
        return
    unknown = [n for n in args.checks if n not in CHECKS and n not in EXTRA]
    if unknown:
        parser.error(f"unknown check(s) {unknown}; see --list")
    sys.exit(0 if run(*args.checks) else 1)


if __name__ == "__main__":
    main()
