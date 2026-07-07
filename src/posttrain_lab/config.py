"""Config loading: base.yaml + stage yaml (+ smoke.yaml) + CLI dotlist overrides.

Every entry point calls::

    cfg = load_config("configs/arm_dpo.yaml", sys.argv[1:])

CLI args are OmegaConf dotlist overrides (``train.beta=0.05``). The special
flag ``--smoke`` (or env ``SMOKE=1``) additionally merges configs/smoke.yaml
on top, shrinking every stage to a minutes-scale run.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(stage_yaml: str | Path, argv: list[str] | None = None) -> DictConfig:
    argv = list(sys.argv[1:] if argv is None else argv)
    smoke = os.environ.get("SMOKE", "") == "1"
    if "--smoke" in argv:
        smoke = True
        argv.remove("--smoke")

    cfg = OmegaConf.load(CONFIG_DIR / "base.yaml")
    stage_path = Path(stage_yaml)
    if not stage_path.exists():
        stage_path = CONFIG_DIR / stage_yaml
    cfg = OmegaConf.merge(cfg, OmegaConf.load(stage_path))
    if smoke:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(CONFIG_DIR / "smoke.yaml"))
    if argv:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(argv))
    OmegaConf.resolve(cfg)
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def train_kwargs(cfg: DictConfig, config_cls) -> dict:
    """Filter cfg.train down to fields the given TRL/HF *Config actually accepts.

    TRL 1.x shifts config fields across minor versions; filtering (with a loud
    warning for anything dropped) keeps one YAML schema working across pins.
    """
    import dataclasses

    valid = {f.name for f in dataclasses.fields(config_cls)}
    out, dropped = {}, []
    for key, val in OmegaConf.to_container(cfg.train, resolve=True).items():
        if key in valid:
            out[key] = val
        else:
            dropped.append(key)
    if dropped:
        print(f"[config] {config_cls.__name__} does not accept, dropping: {dropped}", file=sys.stderr)

    # bf16 requires an accelerator; on a CPU-only box (smoke/dry-run) fall back
    # so the same YAMLs work everywhere. Real runs on the GPU node keep bf16.
    import torch

    if out.get("bf16") and not torch.cuda.is_available():
        print("[config] no CUDA available — disabling bf16 for this run", file=sys.stderr)
        out["bf16"] = False
        out["use_cpu"] = True
    return out


def _sweep_cli() -> None:
    """Print sweep override values for a stage, one per line (used by scripts/)."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["dpo", "ppo", "grpo"])
    args = parser.parse_args()
    sweeps = OmegaConf.load(CONFIG_DIR / "sweeps.yaml")[args.stage]
    for key, values in sweeps.items():
        for value in values:
            print(f"{key}={value}")


if __name__ == "__main__":
    _sweep_cli()
