"""Shared model/tokenizer loading for all training stages."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

from posttrain_lab.config import seed_everything
from posttrain_lab.utils.chat import use_end_of_turn_eos
from posttrain_lab.utils.logging import setup_tracking


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
    """The stage's tokenizer, with eos = the chat template's end-of-turn token
    (<|im_end|> for Qwen; utils/chat.py). Pad is set first, from the original
    eos, so it never equals the end-of-turn token."""
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return use_end_of_turn_eos(tok)


def align_eos(model, tokenizer) -> None:
    """Point the model's config and generation_config at the tokenizer's eos,
    keeping the original eos ids as extra stops, so the checkpoint it saves
    (and vLLM loading it) stops at end of turn. The HF Trainer does the same
    at init; this covers models built outside it."""
    eos = tokenizer.eos_token_id
    model.config.eos_token_id = eos
    gen = getattr(model, "generation_config", None)
    if gen is not None:
        old = gen.eos_token_id
        old = [] if old is None else [old] if isinstance(old, int) else list(old)
        gen.eos_token_id = [eos] + [i for i in old if i != eos]


def model_kwargs(cfg: DictConfig) -> dict:
    return {
        "torch_dtype": getattr(torch, str(cfg.model.dtype)),
        "attn_implementation": str(cfg.model.attn_implementation),
    }


def begin_run(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))
    setup_tracking(cfg)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"[train] stage={cfg.stage} run={cfg.run_name} out={cfg.output_dir}")
