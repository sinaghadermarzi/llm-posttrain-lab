"""scripts/prefetch_assets.py must download into the requested HF_HOME, online,
even when the repo's .env holds offline-cluster settings: `import posttrain_lab`
exports HF_* from .env into any key that is still unset.

    python -m pytest tests/
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest


@pytest.fixture
def offline_repo(repo, tmp_path: Path):
    """.env looks like the offline cluster's and points the caches at 'wrong'."""
    wrong = tmp_path / "wrong"
    repo.write_env(
        f"HF_HOME={wrong}\nHF_HUB_CACHE={wrong}/hub\nHF_DATASETS_CACHE={wrong}/datasets\n"
        "HF_HUB_OFFLINE=1\nHF_DATASETS_OFFLINE=1\n"
    )
    return repo


def test_pinned_env_survives_dotenv(offline_repo, tmp_path: Path) -> None:
    target = tmp_path / "target"
    probe = f"""
import importlib.util, json
spec = importlib.util.spec_from_file_location("prefetch", "scripts/prefetch_assets.py")
prefetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prefetch)
prefetch.pin_hf_env(prefetch.Path({str(target)!r}), offline=False)
import posttrain_lab                 # exports HF_* from the copy's .env
assert posttrain_lab.__file__.startswith({str(offline_repo.root)!r}), posttrain_lab.__file__
from huggingface_hub import constants
from datasets import config
print(json.dumps({{"offline": constants.HF_HUB_OFFLINE, "datasets_offline": config.HF_HUB_OFFLINE,
                  "hf_home": constants.HF_HOME, "hub": constants.HF_HUB_CACHE,
                  "datasets": str(config.HF_DATASETS_CACHE)}}))
"""
    seen = json.loads(offline_repo.python("-c", probe).stdout.strip().splitlines()[-1])
    assert seen == {"offline": False, "datasets_offline": False, "hf_home": str(target),
                    "hub": str(target / "hub"), "datasets": str(target / "datasets")}


def _hub_reachable() -> bool:
    try:
        urllib.request.urlopen("https://huggingface.co/api/datasets/Salesforce/wikitext", timeout=10)
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _hub_reachable(), reason="needs huggingface.co")
def test_prefetch_downloads_into_target(offline_repo, tmp_path: Path) -> None:
    """End to end: fetch one small dataset (~6 MB) online, then pass the offline check."""
    target, wrong = tmp_path / "target", tmp_path / "wrong"
    run = offline_repo.python("scripts/prefetch_assets.py", str(target), "--only", "wikitext", check=False)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    assert "all 1 selected assets load offline" in run.stdout
    assert any((target / "datasets" / "Salesforce___wikitext").glob("*/*/*/dataset_info.json"))
    assert not wrong.exists()
