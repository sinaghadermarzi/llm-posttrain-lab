"""scripts/prefetch_assets.py must download into the requested HF_HOME, online,
even when the repo's .env holds offline-cluster settings: `import posttrain_lab`
exports HF_* from .env into any key that is still unset.

Each test runs in a throwaway copy of the repo (scripts/, src/, configs/) with
its own .env, so the real .env is never read or touched.

    python -m pytest tests/
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_VARS = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE",
           "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Repo copy whose .env looks like the offline cluster's and points the
    caches somewhere else ('wrong')."""
    root = tmp_path / "repo"
    for part in ("scripts", "src", "configs"):
        shutil.copytree(REPO_ROOT / part, root / part, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    wrong = tmp_path / "wrong"
    (root / ".env").write_text(
        f"HF_HOME={wrong}\nHF_HUB_CACHE={wrong}/hub\nHF_DATASETS_CACHE={wrong}/datasets\n"
        "HF_HUB_OFFLINE=1\nHF_DATASETS_OFFLINE=1\n"
    )
    return root


def _env() -> dict[str, str]:
    """The test process env minus anything HF, so only .env can set it."""
    return {k: v for k, v in os.environ.items() if k not in HF_VARS}


def test_pinned_env_survives_dotenv(repo: Path, tmp_path: Path) -> None:
    target = tmp_path / "target"
    probe = f"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("prefetch", {str(repo / "scripts/prefetch_assets.py")!r})
prefetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prefetch)   # puts the copy's src/ first on sys.path
prefetch.pin_hf_env(prefetch.Path({str(target)!r}), offline=False)
import posttrain_lab                 # exports HF_* from the copy's .env
assert posttrain_lab.__file__.startswith({str(repo)!r}), posttrain_lab.__file__
from huggingface_hub import constants
from datasets import config
print(json.dumps({{"offline": constants.HF_HUB_OFFLINE, "datasets_offline": config.HF_HUB_OFFLINE,
                  "hf_home": constants.HF_HOME, "hub": constants.HF_HUB_CACHE,
                  "datasets": str(config.HF_DATASETS_CACHE)}}))
"""
    out = subprocess.run([sys.executable, "-c", probe], env=_env(), cwd=tmp_path,
                         capture_output=True, text=True, check=True)
    seen = json.loads(out.stdout.strip().splitlines()[-1])
    assert seen == {"offline": False, "datasets_offline": False, "hf_home": str(target),
                    "hub": str(target / "hub"), "datasets": str(target / "datasets")}


def _hub_reachable() -> bool:
    try:
        urllib.request.urlopen("https://huggingface.co/api/datasets/Salesforce/wikitext", timeout=10)
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _hub_reachable(), reason="needs huggingface.co")
def test_prefetch_downloads_into_target(repo: Path, tmp_path: Path) -> None:
    """End to end: fetch one small dataset (~6 MB) online, then pass the offline check."""
    target, wrong = tmp_path / "target", tmp_path / "wrong"
    run = subprocess.run([sys.executable, str(repo / "scripts/prefetch_assets.py"), str(target), "--only", "wikitext"],
                         env=_env(), cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    assert "all 1 selected assets load offline" in run.stdout
    assert any((target / "datasets" / "Salesforce___wikitext").glob("*/*/*/dataset_info.json"))
    assert not wrong.exists()
