"""Shared model/tokenizer loading for all training stages."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
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


CROWD_COS = 0.95  # Qwen2.5 base: ~2.8k rows within this of <|im_end|>; trained tokens: none


def untrained_rows(weight: torch.Tensor, ids: list[int], cos: float = CROWD_COS, min_twins: int = 8) -> list[int]:
    """The ids whose embedding row is one of a crowd of near-identical rows:
    tokens the base model never learned to tell apart. Qwen2.5 base leaves
    <|im_start|>/<|im_end|> in such a crowd (with ~2.8k rare Thai and other
    tokens, cosine ~0.99), so a trained push toward <|im_end|> lifts
    the whole crowd and sampling picks a lookalike instead."""
    w = torch.nn.functional.normalize(weight.detach().float(), dim=-1)
    out = []
    for i in ids:
        twins = int((w @ w[i] >= cos).sum()) - 1
        if twins >= min_twins:
            out.append(i)
    return out


def init_untrained_chat_tokens(model, tokenizer, original_eos: int | None) -> list[str]:
    """Give the chat template's special tokens a usable embedding before SFT,
    if the base model never trained them (untrained_rows): the end-of-turn
    token gets a copy of the original eos row (<|endoftext|>, what pretraining
    predicts at the end of a document), the others the mean of the rows outside
    the crowd. Trained rows (an -Instruct base, or a checkpoint this already
    ran on) are left alone. Deterministic, so every rank makes the same
    change. Returns the tokens re-initialized."""
    from posttrain_lab.utils.chat import end_of_turn_token, template_token_ids

    out_emb, in_emb = model.get_output_embeddings(), model.get_input_embeddings()
    if out_emb is None or in_emb is None:
        return []
    ids = [i for i in template_token_ids(tokenizer) if i < out_emb.weight.shape[0]]
    bad = untrained_rows(out_emb.weight, ids)
    if not bad:
        return []
    eot = end_of_turn_token(tokenizer)
    eot_id = tokenizer.convert_tokens_to_ids(eot) if eot else None
    matrices = [in_emb.weight] + ([out_emb.weight] if out_emb.weight is not in_emb.weight else [])
    source_ok = (original_eos is not None and original_eos != eot_id and original_eos not in bad
                 and not untrained_rows(out_emb.weight, [original_eos]))
    w = torch.nn.functional.normalize(out_emb.weight.detach().float(), dim=-1)
    crowd = (w @ w[bad].T >= CROWD_COS).any(1)
    with torch.no_grad():
        for weight in matrices:
            mean = weight[~crowd].float().mean(0).to(weight.dtype)
            for i in bad:
                weight[i] = weight[original_eos] if (i == eot_id and source_ok) else mean
    names = tokenizer.convert_ids_to_tokens(bad)
    print(f"[train] re-initialized untrained chat-template token embeddings: {', '.join(names)}"
          + (f" ({eot} <- {tokenizer.convert_ids_to_tokens(original_eos)})" if eot_id in bad and source_ok else ""))
    return names


def model_kwargs(cfg: DictConfig) -> dict:
    return {
        "torch_dtype": getattr(torch, str(cfg.model.dtype)),
        "attn_implementation": str(cfg.model.attn_implementation),
    }


RESOLVED_CONFIG = "posttrain_config.yaml"  # read by analysis/export_evidence.py (configs/as_run/)


def save_resolved_config(cfg: DictConfig) -> None:
    """The fully merged config this run trains with (base, task, stage, smoke,
    CLI overrides), next to its checkpoints, so evidence can show what ran.
    Every rank writes the same content; tmp + rename keeps the file whole."""
    path = Path(cfg.output_dir) / RESOLVED_CONFIG
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    OmegaConf.save(cfg, tmp, resolve=True)
    os.replace(tmp, path)


def save_final_state(trainer) -> None:
    """trainer_state.json next to the final model, with the whole log
    (train_runtime included): the last checkpoint's copy is written before the
    run ends, so it never has the run's totals. Main process only (HF Trainer)."""
    if hasattr(trainer, "save_state"):
        trainer.save_state()


def begin_run(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))
    setup_tracking(cfg)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg)
    print(f"[train] stage={cfg.stage} run={cfg.run_name} out={cfg.output_dir}")
