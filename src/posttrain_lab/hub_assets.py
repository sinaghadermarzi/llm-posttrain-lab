"""Every Hugging Face Hub model/dataset the pipeline loads, read from the
configs — the one list behind ``preflight hf`` / ``hf_load`` and
``scripts/prefetch_assets.py``.

Offline (HF_HUB_OFFLINE=1) each asset must be in HF_HOME:

  model    a complete snapshot in HF_HOME/hub (transformers / vLLM / TRL load
           it by repo id from there)
  dataset  the *prepared Arrow cache* of that config in HF_HOME/datasets.
           The raw parquet files in HF_HOME/hub are not enough: offline,
           datasets 5.x resolves a Hub id only through the Arrow cache.

lm-eval tasks are mapped to the dataset they load via lm-eval's own task
yamls, so changing eval.harness.tasks changes what gets prefetched.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from posttrain_lab.config import load_config


@dataclass(frozen=True)
class Asset:
    name: str                  # handle for prefetch_assets.py --only / --skip
    kind: str                  # "model" | "dataset"
    repo: str                  # Hub repo id
    config: str | None = None  # dataset config (None = the default config)
    why: str = ""
    smoke: bool = False        # needed by `make smoke`
    harness_task: str | None = None  # lm-eval task that loads this dataset


def hub_assets() -> list[Asset]:
    """Models + datasets, in pipeline order. Imports lm-eval (~10 s) to map
    the harness tasks to their datasets."""
    helpful = load_config("eval.yaml", [])
    math = load_config("eval.yaml", ["task=math"])
    smoke = load_config("eval.yaml", ["--smoke"])
    ret = helpful.eval.retention
    assets = [
        Asset("base", "model", str(helpful.model.base), why="base model (both tracks)"),
        Asset("smoke-model", "model", str(smoke.model.base), why="smoke-test model", smoke=True),
        Asset("judge", "model", str(helpful.eval.judge), why="held-out judge (helpful eval)"),
        Asset("helpful-data", "dataset", str(helpful.dataset.id), helpful.dataset.get("config"),
              "helpful data", smoke=True),
        Asset("math-data", "dataset", str(math.dataset.id), math.dataset.get("config"),
              "math data", smoke=True),
    ]
    if not ret.get("data_files"):
        assets.append(Asset("wikitext", "dataset", str(ret.dataset), str(ret.config),
                            "retention_ppl", smoke=True))
    smoke_tasks = set(smoke.eval.harness.tasks)
    for task, (path, name) in harness_datasets(list(helpful.eval.harness.tasks)).items():
        assets.append(Asset(task, "dataset", path, name, f"retention_harness (lm-eval {task})",
                            smoke=task in smoke_tasks, harness_task=task))
    return assets


def harness_datasets(tasks: list[str]) -> dict[str, tuple[str, str | None]]:
    """lm-eval task -> (dataset_path, dataset_name), from lm-eval's task index."""
    from lm_eval.tasks import TaskManager

    index = TaskManager().task_index
    out = {}
    for task in tasks:
        entry = index.get(task)
        if entry is None or not entry.cfg or "dataset_path" not in entry.cfg:
            raise ValueError(f"lm-eval task {task!r} not found or is a group; list leaf tasks in eval.harness.tasks")
        out[task] = (str(entry.cfg["dataset_path"]), entry.cfg.get("dataset_name"))
    return out


def model_snapshot(repo: str) -> tuple[Path | None, str]:
    """Local snapshot dir of a model and a problem description ('' = complete:
    config.json plus every weight shard the index lists). No network."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = Path(snapshot_download(repo, local_files_only=True))
    except LocalEntryNotFoundError:
        return None, "no snapshot in HF_HOME/hub"
    if not (path / "config.json").exists():
        return path, "snapshot has no config.json"
    for index in path.glob("*.index.json"):
        shards = set(json.loads(index.read_text())["weight_map"].values())
        missing = sorted(s for s in shards if not (path / s).exists())
        if missing:
            return path, f"{len(missing)}/{len(shards)} weight shards missing (e.g. {missing[0]})"
        return path, ""
    if not any(path.glob("*.safetensors")) and not any(path.glob("*.bin")):
        return path, "snapshot has no weight files"
    return path, ""


def datasets_cache() -> Path:
    from datasets import config

    return Path(config.HF_DATASETS_CACHE)


def dataset_prepared(repo: str, config: str | None) -> tuple[list[str], Path | None]:
    """Splits of the prepared Arrow cache for (repo, config) — exactly what
    load_dataset uses offline — and its directory; ([], None) if absent."""
    config_dir = datasets_cache() / repo.replace("/", "___") / (config or "default")
    infos = sorted(config_dir.glob("*/*/dataset_info.json"), key=os.path.getmtime)
    if not infos:
        return [], None
    info = json.loads(infos[-1].read_text())
    return sorted(info.get("splits", {})), infos[-1].parent
