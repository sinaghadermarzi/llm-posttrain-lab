"""The repo's .env (git-ignored; template in .env.example). Variables already
set in the process env always take precedence over the file."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DOTENV = REPO_ROOT / ".env"


def read_dotenv(path: Path = DOTENV) -> dict[str, str]:
    """Minimal KEY=VALUE parser (comments, blank lines, optional quotes)."""
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        out[key.strip().removeprefix("export ").strip()] = val.strip().strip("'\"")
    return out


def export_hf_env(path: Path = DOTENV) -> None:
    """Copy HF_* (and TRANSFORMERS_OFFLINE) from .env into os.environ unless
    already set. Runs on `import posttrain_lab`, i.e. before huggingface_hub /
    datasets / transformers read them at their own import time."""
    for key, val in read_dotenv(path).items():
        if (key.startswith("HF_") or key == "TRANSFORMERS_OFFLINE") and val:
            os.environ.setdefault(key, val)


def results_dir() -> str:
    """Where results/checkpoints go: RESULTS_DIR from the env, else .env, else
    'results'. A relative value is taken relative to the repo root; the
    default stays relative, as before (every launcher runs from the repo root)."""
    value = os.environ.get("RESULTS_DIR") or read_dotenv().get("RESULTS_DIR")
    if not value:
        return "results"
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else REPO_ROOT / path)


def results_dir_source() -> str:
    if os.environ.get("RESULTS_DIR"):
        return "env"
    return ".env" if read_dotenv().get("RESULTS_DIR") else "default"
