"""posttrain_lab.reset: stops only the track's own jobs, removes only the
track's results and logs, asks first, and refuses an unexpected results root."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


def _tree(repo, tmp_path: Path) -> dict[str, str]:
    results = tmp_path / "results"
    for track in ("math", "helpful", "smoke/math", "smoke/helpful"):
        (results / track / "runs" / "r_seed17").mkdir(parents=True)
        (results / track / "runs" / "r_seed17" / "model.bin").write_bytes(b"x" * 100)
    for log in ("math-all.log", "logs_math_dpo_train_beta_0_1.log", "logs_math_eval_shard2.log",
                "helpful-all.log", "logs_dpo_train_beta_0_1.log", "smoke.log", "all.log"):
        (repo.root / log).write_text("log\n")
    cache = tmp_path / "hf" / "datasets" / "openai___gsm8k" / "main" / "0.0.0" / "abc"
    cache.mkdir(parents=True)
    (cache / "tmpdead").write_text("half-written")
    os.utime(cache / "tmpdead", (time.time() - 7200,) * 2)      # left by a crash hours ago
    (cache / "tmplive").write_text("being written right now")   # a live map, maybe another checkout's
    (cache / "gsm8k-train.arrow").write_text("keep me")
    return {"RESULTS_DIR": str(results), "HF_HOME": str(tmp_path / "hf")}


def test_dry_run_changes_nothing(repo, tmp_path: Path) -> None:
    env = _tree(repo, tmp_path)
    run = repo.python("-m", "posttrain_lab.reset", "math", "--dry-run", check=False, **env)
    assert run.returncode == 0, run.stderr[-2000:]
    assert f"{tmp_path / 'results' / 'math'}  (" in run.stdout
    assert "logs_math_dpo_train_beta_0_1.log" in run.stdout and "helpful-all.log" not in run.stdout
    assert "tmpdead" in run.stdout and "tmplive" not in run.stdout and "dry run: nothing changed" in run.stdout
    assert (tmp_path / "results" / "math").exists() and (repo.root / "math-all.log").exists()


def test_needs_confirmation(repo, tmp_path: Path) -> None:
    env = _tree(repo, tmp_path)
    run = repo.python("-m", "posttrain_lab.reset", "math", check=False, **env)  # stdin is not a terminal
    assert run.returncode == 1 and "re-run with --yes" in run.stdout
    assert (tmp_path / "results" / "math").exists()


def test_reset_math_removes_only_math(repo, tmp_path: Path) -> None:
    env = _tree(repo, tmp_path)
    run = repo.python("-m", "posttrain_lab.reset", "math", "--yes", check=False, **env)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    results = tmp_path / "results"
    assert not (results / "math").exists()
    assert (results / "helpful").exists() and (results / "smoke" / "math").exists()
    left = {p.name for p in repo.root.glob("*.log")}
    assert left == {"helpful-all.log", "logs_dpo_train_beta_0_1.log", "smoke.log", "all.log"}
    cache = tmp_path / "hf" / "datasets" / "openai___gsm8k" / "main" / "0.0.0" / "abc"
    assert not (cache / "tmpdead").exists() and (cache / "gsm8k-train.arrow").exists()
    assert (cache / "tmplive").exists()


def test_archive_and_smoke(repo, tmp_path: Path) -> None:
    env = _tree(repo, tmp_path)
    run = repo.python("-m", "posttrain_lab.reset", "all", "--smoke", "--archive", "--yes", check=False, **env)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    smoke = tmp_path / "results" / "smoke"
    names = sorted(p.name for p in smoke.iterdir())
    assert len(names) == 2 and all(n.startswith(("helpful.", "math.")) for n in names)
    assert (tmp_path / "results" / "math").exists()  # the full run is not smoke
    assert not (repo.root / "smoke.log").exists() and (repo.root / "math-all.log").exists()


def test_stops_only_the_tracks_own_jobs(repo, tmp_path: Path) -> None:
    """A math driver (and its child) and a math DPO job are stopped; a helpful
    DPO job keeps running."""
    env = _tree(repo, tmp_path)
    driver = tmp_path / "fake" / "scripts" / "run_all_math.sh"
    driver.parent.mkdir(parents=True)
    driver.write_text("sleep 120\n")
    job = [sys.executable, "-c", "import time; time.sleep(120)", "-m", "posttrain_lab.train.dpo"]
    procs = {
        "driver": subprocess.Popen(["bash", str(driver)], cwd=repo.root),
        "math": subprocess.Popen(job + ["configs/math/m3_arm_dpo.yaml", "train.beta=0.3"], cwd=repo.root,
                                 env=repo.env(**env)),
        "helpful": subprocess.Popen(job + ["configs/helpful/arm_dpo.yaml", "train.beta=0.1"], cwd=repo.root,
                                    env=repo.env(**env)),
    }
    try:
        time.sleep(1)  # let bash start its sleep child
        run = repo.python("-m", "posttrain_lab.reset", "math", "--yes", "--timeout", "10", check=False, **env)
        assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
        assert "driver run_all_math.sh" in run.stdout and "posttrain_lab.train.dpo" in run.stdout
        assert "child: sleep 120" in run.stdout
        assert procs["driver"].wait(timeout=15) is not None
        assert procs["math"].wait(timeout=15) is not None
        assert procs["helpful"].poll() is None
    finally:
        for p in procs.values():
            p.kill()
            p.wait()


def test_refuses_an_unexpected_root(monkeypatch, tmp_path: Path) -> None:
    from posttrain_lab import reset

    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    assert reset.safe_root("math", False) == (tmp_path / "results" / "math").resolve()
    monkeypatch.setattr(reset, "results_root", lambda task: str(tmp_path))  # e.g. a config pointing elsewhere
    with pytest.raises(SystemExit, match="refusing"):
        reset.safe_root("math", False)


def test_make_reset_needs_a_task(repo) -> None:
    run = repo.run("make", "reset", check=False)
    assert run.returncode != 0 and "make reset needs TASK=" in run.stdout


def test_full_reset_removes_the_combined_log(repo, tmp_path: Path) -> None:
    env = _tree(repo, tmp_path)
    run = repo.python("-m", "posttrain_lab.reset", "all", "--yes", check=False, **env)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    assert not (repo.root / "all.log").exists() and (repo.root / "smoke.log").exists()


def test_shared_driver_keeps_the_other_tracks_run(repo, tmp_path: Path) -> None:
    """`make all` in its helpful phase: reset math stops run_all.sh itself but
    not the helpful job it is running."""
    env = _tree(repo, tmp_path)
    driver = tmp_path / "fake" / "scripts" / "run_all.sh"
    driver.parent.mkdir(parents=True)
    driver.write_text(f"{sys.executable} -c 'import time; time.sleep(120)' -m posttrain_lab.train.dpo "
                      "configs/helpful/arm_dpo.yaml train.beta=0.1\n")
    proc = subprocess.Popen(["bash", str(driver)], cwd=repo.root, env=repo.env(**env), start_new_session=True)
    group = os.getpgid(proc.pid)  # the driver and its helpful job, for cleanup
    try:
        time.sleep(1.5)
        run = repo.python("-m", "posttrain_lab.reset", "math", "--yes", "--timeout", "10", check=False, **env)
        assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
        assert "driver run_all.sh (shared with the other track: this process only)" in run.stdout
        assert proc.wait(timeout=15) is not None   # the driver is gone ...
        helpful = repo.python("-m", "posttrain_lab.reset", "helpful", "--dry-run", check=False, **env)
        assert "posttrain_lab.train.dpo configs/helpful/arm_dpo.yaml" in helpful.stdout  # ... its helpful job is not
    finally:
        try:
            os.killpg(group, 9)
        except ProcessLookupError:
            pass
        proc.wait()


RESCAN = """
import subprocess, sys, time
from posttrain_lab import reset
pl = reset.plan("math", False)          # what the prompt showed: nothing running yet
job = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "-m", "posttrain_lab.train.dpo",
                        "configs/math/m3_arm_dpo.yaml", "train.beta=0.05"])   # started while the prompt waited
time.sleep(1)
ok = reset.execute(pl, archive=False, timeout=10)
print("OK" if ok else "ABORT", "job-exit", job.wait(timeout=15))
"""


def test_jobs_started_after_the_plan_are_stopped_too(repo, tmp_path: Path) -> None:
    env = _tree(repo, tmp_path)
    run = repo.python("-c", RESCAN, check=False, **env)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    assert "stopping" in run.stdout and "posttrain_lab.train.dpo configs/math/m3_arm_dpo.yaml" in run.stdout
    assert run.stdout.strip().splitlines()[-1].startswith("OK job-exit")
    assert not (tmp_path / "results" / "math").exists()
