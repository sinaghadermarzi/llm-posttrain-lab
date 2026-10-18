"""Shared fixture: a throwaway copy of the repo with its own .env, so tests that
depend on .env never read or touch the real one."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
# anything that would leak the caller's setup into the copy
LEAKY_VARS = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE",
              "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "RESULTS_DIR", "SMOKE", "PYTHONPATH")


class RepoCopy:
    def __init__(self, root: Path):
        self.root = root

    def write_env(self, text: str) -> None:
        (self.root / ".env").write_text(text)

    def env(self, **extra: str) -> dict[str, str]:
        """Caller's env minus LEAKY_VARS; imports resolve to the copy's src/,
        and `python` (for make) is this interpreter."""
        env = {k: v for k, v in os.environ.items() if k not in LEAKY_VARS}
        env["PYTHONPATH"] = str(self.root / "src")
        env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
        return {**env, **extra}

    def run(self, *cmd: str, check: bool = True, **extra_env: str) -> subprocess.CompletedProcess:
        return subprocess.run(list(cmd), cwd=self.root, env=self.env(**extra_env),
                              capture_output=True, text=True, check=check)

    def python(self, *args: str, **kw) -> subprocess.CompletedProcess:
        return self.run(sys.executable, *args, **kw)


@pytest.fixture
def repo(tmp_path: Path) -> RepoCopy:
    root = tmp_path / "repo"
    for part in ("scripts", "src", "configs", "analysis"):
        shutil.copytree(REPO_ROOT / part, root / part, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    for part in ("Makefile", "requirements.txt"):
        shutil.copy(REPO_ROOT / part, root / part)
    return RepoCopy(root)
