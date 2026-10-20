"""scripts/run_smoke.sh runs every stage on one GPU: with several visible,
plain `python -m` gets nn.DataParallel, which breaks TRL's SFTTrainer.

The real script runs with a fake `python` on PATH that records the
CUDA_VISIBLE_DEVICES each stage sees (and prints "skipping" for the script's
incremental-eval check), so nothing trains."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

FAKE_PYTHON = """#!/bin/sh
echo "CVD=${CUDA_VISIBLE_DEVICES-<unset>} $*" >> "$SMOKE_CALLS"
echo skipping
"""


def _stage_gpus(repo, tmp_path: Path, cvd: str | None) -> set[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "python").write_text(FAKE_PYTHON)
    (fake_bin / "python").chmod(0o755)
    calls = tmp_path / "calls.log"
    env = repo.env(SMOKE_CALLS=str(calls))
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if cvd is not None:
        env["CUDA_VISIBLE_DEVICES"] = cvd
    run = subprocess.run(["bash", "scripts/run_smoke.sh"], cwd=repo.root, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    lines = calls.read_text().splitlines()
    assert len(lines) > 20, lines  # every stage, eval and plot call went through the fake
    return {line.split(" ", 1)[0].removeprefix("CVD=") for line in lines}


@pytest.mark.parametrize(("visible", "expected"), [
    (None, "0"),                 # unset -> GPU 0
    ("0,1,2,3", "0"),
    ("2,3", "2"),                # first of what the pod exposes
    ("1", "1"),
    ("GPU-aaaa,GPU-bbbb", "GPU-aaaa"),
    ("", ""),                    # explicitly empty -> CPU, left alone
])
def test_every_smoke_stage_sees_one_gpu(repo, tmp_path: Path, visible, expected) -> None:
    assert _stage_gpus(repo, tmp_path, visible) == {expected}
