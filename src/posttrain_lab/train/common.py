"""Shared model/tokenizer loading for all training stages."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

from posttrain_lab.config import seed_everything
from posttrain_lab.utils.logging import setup_wandb


def parse_entry(default_yaml: str) -> tuple[str, list[str]]:
    """Entry convention: ``python -m posttrain_lab.train.X [stage.yaml] [k=v ...]``.
    A leading .yaml arg selects the stage config; everything else is a dotlist
    override (plus the --smoke flag handled by load_config)."""
    argv = sys.argv[1:]
    if argv and argv[0].endswith((".yaml", ".yml")):
        return argv[0], argv[1:]
    return default_yaml, argv


def init_path(cfg: DictConfig) -> str:
    """Where this stage's model weights start from: a prior run if configured
    (and already trained), else the hub base model."""
    path = cfg.get("init_from")
    if path and Path(path, "config.json").exists():
        return str(path)
    if path:
        print(f"[train] init_from={path} not found, falling back to {cfg.model.base}", file=sys.stderr)
    return str(cfg.model.base)


def load_tokenizer(path: str) -> "AutoTokenizer":
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def model_kwargs(cfg: DictConfig) -> dict:
    return {
        "torch_dtype": getattr(torch, str(cfg.model.dtype)),
        "attn_implementation": str(cfg.model.attn_implementation),
    }


def begin_run(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))
    setup_wandb(cfg)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"[train] stage={cfg.stage} run={cfg.run_name} out={cfg.output_dir}")
