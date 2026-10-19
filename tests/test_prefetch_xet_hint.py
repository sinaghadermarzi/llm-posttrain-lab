"""prefetch_assets.py turns the Artifactory/Nexus Xet 404 into a hint to set
HF_HUB_DISABLE_XET=1, and leaves every other error alone."""

from __future__ import annotations

from pathlib import Path

import pytest

ENDPOINT = "https://mirror.example/artifactory/api/huggingfaceml/hf-remote"
URL = f"{ENDPOINT}/api/models/Qwen/Qwen2.5-0.5B/xet-read-token/5e271e3a1061baa8e91fbc262e6611c479d7f6f9"

PROBE = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("prefetch", "scripts/prefetch_assets.py")
prefetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prefetch)

def fail(assets):
    kind = {kind!r}
    if kind == "xet":  # raised the way the real failure is: a 404 through hf_raise_for_status
        import httpx
        from huggingface_hub.utils import hf_raise_for_status
        hf_raise_for_status(httpx.Response(404, request=httpx.Request("GET", "{url}")))
    raise ValueError("something else broke")

prefetch.fetch = fail
sys.argv = ["scripts/prefetch_assets.py", {target!r}, "--only", "smoke-model"]
prefetch.main()
"""


def _run(repo, tmp_path: Path, kind: str, **env):
    probe = PROBE.format(kind=kind, url=URL, target=str(tmp_path / "hf"))
    return repo.python("-c", probe, check=False, HF_ENDPOINT=ENDPOINT, **env)


def test_xet_404_prints_hint(repo, tmp_path: Path) -> None:
    run = _run(repo, tmp_path, "xet")
    assert run.returncode == 1
    assert "error: " in run.stderr and "404" in run.stderr and "xet-read-token" in run.stderr
    assert "does not serve the Hub's Xet token API" in run.stderr
    assert f"HF_HUB_DISABLE_XET=1 python scripts/prefetch_assets.py {tmp_path / 'hf'} --only smoke-model" in run.stderr
    assert "Traceback" not in run.stderr


def test_no_hint_when_xet_already_disabled(repo, tmp_path: Path) -> None:
    run = _run(repo, tmp_path, "xet", HF_HUB_DISABLE_XET="1")
    assert run.returncode != 0
    assert "Traceback" in run.stderr and "HfHubHTTPError" in run.stderr
    assert "does not serve the Hub's Xet token API" not in run.stderr


@pytest.mark.parametrize("env", [{}, {"HF_HUB_DISABLE_XET": "1"}])
def test_other_errors_propagate(repo, tmp_path: Path, env) -> None:
    run = _run(repo, tmp_path, "other", **env)
    assert run.returncode != 0
    assert "ValueError: something else broke" in run.stderr
    assert "Xet token API" not in run.stderr
