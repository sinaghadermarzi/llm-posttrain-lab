"""Config loading with a task dimension. Resolution order::

    base.yaml -> configs/tasks/<task>.yaml -> <stage yaml> -> smoke.yaml (--smoke) | probe.yaml (PROBE=1)
              -> CLI dotlist

Every entry point calls::

    cfg = load_config("configs/helpful/arm_dpo.yaml", sys.argv[1:])

The task is read in a pre-pass from the stage yaml's ``task:`` key (a
``task=`` dotlist arg wins), then the full merge runs. CLI args are OmegaConf
dotlist overrides (``train.beta=0.05``). The special flag ``--smoke`` (or env
``SMOKE=1``) merges configs/smoke.yaml, shrinking every stage of either task
to a minutes-scale run. Env ``PROBE=1`` merges configs/probe.yaml instead: full
scale (model, batch sizes, lengths), a few steps per stage, under
<RESULTS_DIR>/probe/<task> (scripts/run_probe.sh).
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from posttrain_lab.utils.dotenv import results_dir

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

# ${results_dir:} in base.yaml / smoke.yaml: RESULTS_DIR (env, else .env), default results
OmegaConf.register_new_resolver("results_dir", results_dir, replace=True)


def load_config(stage_yaml: str | Path, argv: list[str] | None = None) -> DictConfig:
    argv = list(sys.argv[1:] if argv is None else argv)
    smoke = os.environ.get("SMOKE", "") == "1"
    if "--smoke" in argv:
        smoke = True
        argv.remove("--smoke")
    probe = is_probe()
    if smoke and probe:
        raise ValueError("SMOKE=1 and PROBE=1 together: smoke shrinks the model, the probe keeps it full size")

    stage_path = Path(stage_yaml)
    if not stage_path.exists():
        stage_path = CONFIG_DIR / stage_yaml
    stage_cfg = OmegaConf.load(stage_path)

    task = stage_cfg.get("task", "helpful")
    for arg in argv:
        if arg.startswith("task="):
            task = arg.split("=", 1)[1]

    cfg = OmegaConf.merge(
        OmegaConf.load(CONFIG_DIR / "base.yaml"),
        OmegaConf.load(CONFIG_DIR / "tasks" / f"{task}.yaml"),
        stage_cfg,
        OmegaConf.create({"task": str(task)}),
    )
    if smoke:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(CONFIG_DIR / "smoke.yaml"))
    if probe:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(CONFIG_DIR / "probe.yaml"))
    if argv:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(argv))
    OmegaConf.resolve(cfg)
    return cfg


def is_probe() -> bool:
    """PROBE=1: a scale probe (configs/probe.yaml, scripts/run_probe.sh)."""
    return os.environ.get("PROBE", "") == "1"


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


def results_root(task: str) -> str:
    """A task's resolved results_root (RESULTS_DIR, and smoke/ under SMOKE=1, probe/ under PROBE=1)."""
    return str(load_config("eval.yaml", [f"task={task}"]).results_root)


def _cli() -> None:
    """Used by scripts/ and the Makefile:

    python -m posttrain_lab.config helpful dpo          sweep override values, one per line
                                                        (PROBE=1: only the last, the most memory-hungry:
                                                        GRPO beta 0.0 loads no reference model)
    python -m posttrain_lab.config --results-root math  the task's results_root
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["helpful", "math"])
    parser.add_argument("stage", nargs="?", choices=["dpo", "dpo_nll", "ppo", "grpo"])
    parser.add_argument("--results-root", action="store_true", help="print the task's results_root and exit")
    args = parser.parse_args()
    if args.results_root:
        print(results_root(args.task))
        return
    if args.stage is None:
        parser.error("stage is required (or pass --results-root)")
    sweeps = OmegaConf.load(CONFIG_DIR / args.task / "sweeps.yaml")[args.stage]
    for key, values in sweeps.items():
        for value in (list(values)[-1:] if is_probe() else values):
            print(f"{key}={value}")


if __name__ == "__main__":
    _cli()
