"""RESULTS_DIR (env, else .env, default results/) moves every results path:
the configs (results_root, paths.*, output_dir, smoke), the Makefile plot
targets and preflight's system check."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

PROBE = """
import json
from posttrain_lab.config import load_config
helpful = load_config("helpful/stage1_sft_base.yaml", [])
smoke = load_config("helpful/stage1_sft_base.yaml", ["--smoke"])
math = load_config("math/m3_arm_dpo.yaml", [])
print(json.dumps({"root": helpful.results_root, "sft_base": helpful.paths.sft_base,
                  "output_dir": helpful.output_dir, "smoke": smoke.results_root,
                  "smoke_output_dir": smoke.output_dir, "math": math.results_root,
                  "math_pairs": math.data.pairs_path}))
"""


def _paths(repo, **env) -> dict[str, str]:
    return json.loads(repo.python("-c", PROBE, **env).stdout.strip().splitlines()[-1])


def test_dotenv_moves_every_config_path(repo, tmp_path: Path) -> None:
    big = tmp_path / "big volume"
    repo.write_env(f'RESULTS_DIR="{big}"\n')
    p = _paths(repo)
    assert p["root"] == f"{big}/helpful"
    assert p["smoke"] == f"{big}/smoke/helpful"
    assert p["math"] == f"{big}/math"
    for key in ("sft_base", "output_dir"):
        assert p[key].startswith(f"{big}/helpful/runs/"), (key, p[key])
    assert p["smoke_output_dir"].startswith(f"{big}/smoke/helpful/runs/")
    assert p["math_pairs"].startswith(f"{big}/math/data/")


def test_env_wins_over_dotenv(repo, tmp_path: Path) -> None:
    repo.write_env(f"RESULTS_DIR={tmp_path / 'from_dotenv'}\n")
    assert _paths(repo, RESULTS_DIR=str(tmp_path / "from_env"))["root"] == f"{tmp_path / 'from_env'}/helpful"


def test_relative_value_is_from_repo_root(repo) -> None:
    repo.write_env("RESULTS_DIR=data/res\n")
    assert _paths(repo)["root"] == f"{repo.root}/data/res/helpful"


def test_unset_default_unchanged(repo) -> None:
    repo.write_env("MLFLOW_TRACKING_URI=\nRESULTS_DIR=\n")
    p = _paths(repo)
    assert (p["root"], p["smoke"], p["math"]) == ("results/helpful", "results/smoke/helpful", "results/math")
    assert p["sft_base"] == "results/helpful/runs/sft-base_seed17"


def test_make_plots_follow_results_dir(repo, tmp_path: Path) -> None:
    big = tmp_path / "big volume"  # a space must not split the argument
    repo.write_env(f'RESULTS_DIR="{big}"\n')
    commands = [shlex.split(line) for line in repo.run("make", "-n", "plots", "math-plots").stdout.splitlines()]
    plots = [argv for argv in commands if argv[:1] == ["python"] and argv[1].startswith("analysis/")]
    assert len(plots) == 11
    for argv in plots:
        assert len(argv) == 3, argv
        assert argv[2] in (f"{big}/helpful", f"{big}/math"), argv
    assert ["python", "analysis/frontier_math.py", f"{big}/math"] in plots


def test_preflight_system_reports_results_dir(repo, tmp_path: Path) -> None:
    big = tmp_path / "big"
    repo.write_env(f"RESULTS_DIR={big}\nHF_HOME={tmp_path / 'hf'}\n")
    out = repo.python("-m", "posttrain_lab.preflight", "system", check=False).stdout
    assert f"RESULTS_DIR -> {big} (from .env" in out
    assert f"HF_HOME -> {tmp_path / 'hf'}" in out
    assert f"results ({big}):" in out
