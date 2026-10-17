"""The repo's .env (git-ignored; template in .env.example). Variables already
set in the process env always take precedence over the file."""

from __future__ import annotations

import os
from pathlib import Path

DOTENV = Path(__file__).resolve().parents[3] / ".env"


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
