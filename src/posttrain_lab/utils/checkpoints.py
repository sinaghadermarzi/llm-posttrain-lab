"""Run/checkpoint discovery and DONE markers for stage resumability."""

from __future__ import annotations

import re
from pathlib import Path

DONE_MARKER = "DONE"


def mark_done(run_dir: str | Path) -> None:
    Path(run_dir, DONE_MARKER).touch()


def is_done(run_dir: str | Path) -> bool:
    return Path(run_dir, DONE_MARKER).exists()


def run_arm(run_dir: str | Path) -> str:
    """Arm name from the run-dir convention {arm}[_{hp}{val}]_seed{seed}."""
    return Path(run_dir).name.split("_")[0].replace("-base", "-base")


def run_hyperparam(run_dir: str | Path) -> str | None:
    parts = Path(run_dir).name.split("_")
    middle = [p for p in parts[1:] if not p.startswith("seed")]
    return middle[0] if middle else None


def list_checkpoints(run_dir: str | Path) -> list[Path]:
    """checkpoint-* dirs sorted by step; falls back to the run dir itself if the
    final model was saved flat (trainer.save_model at the end)."""
    run_dir = Path(run_dir)
    ckpts = sorted(
        (p for p in run_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(re.search(r"checkpoint-(\d+)", p.name).group(1)),
    )
    if not ckpts and (run_dir / "config.json").exists():
        ckpts = [run_dir]
    return ckpts


def discover_all_checkpoints(runs_root: str | Path) -> list[Path]:
    """All evaluable checkpoints under results/runs/, excluding the RM (it is
    not a policy) but including sft-base (the KL=0 baseline point)."""
    out: list[Path] = []
    for run_dir in sorted(Path(runs_root).glob("*_seed*")):
        if run_dir.name.startswith("rm_"):
            continue
        out.extend(list_checkpoints(run_dir))
    return out
