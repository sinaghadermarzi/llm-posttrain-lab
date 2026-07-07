"""wandb-or-local logging setup and a jsonl metric sink."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig


def setup_wandb(cfg: DictConfig) -> None:
    """Honor cfg.wandb; offline by default so runs never block on a login."""
    os.environ.setdefault("WANDB_PROJECT", str(cfg.wandb.project))
    os.environ.setdefault("WANDB_MODE", str(cfg.wandb.mode))
    if cfg.train.get("report_to") in (None, "none"):
        os.environ["WANDB_MODE"] = "disabled"


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
