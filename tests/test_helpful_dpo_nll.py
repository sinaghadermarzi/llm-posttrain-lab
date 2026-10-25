"""The helpful track's DPO+NLL arm (stage 3e), added before its first run.

On math, pure DPO at the helpful arm's budget collapsed at small beta through
likelihood displacement (FINDINGS M3), and the NLL term on the chosen answer
prevented it in every seed (M4). The helpful arm is the same: arm_dpo.yaml plus
TRL's two-term loss, its own runs ('dpo-nll'), launcher, logs, make target,
status stage and reset stage, without the DPO arm's reset or logs touching it.
Stage key 3e: PPO and GRPO keep 3c/3d, though DPO+NLL runs right after DPO."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_config_is_the_dpo_arm_plus_the_nll_term() -> None:
    from trl import DPOConfig

    from posttrain_lab.config import load_config, train_kwargs

    nll = load_config("helpful/arm_dpo_nll.yaml", ["train.beta=0.05"])
    dpo = load_config("helpful/arm_dpo.yaml", ["train.beta=0.05"])
    args = DPOConfig(output_dir="unused", **train_kwargs(nll, DPOConfig))
    assert args.loss_type == ["sigmoid", "sft"] and args.loss_weights == [1.0, 1.0] and args.beta == 0.05
    rest = {k: v for k, v in nll.train.items() if k not in ("loss_type", "loss_weights")}
    assert rest == dict(dpo.train) and "loss_type" not in dpo.train
    assert (nll.init_from, nll.data, nll.task) == (dpo.init_from, dpo.data, "helpful")


def test_runs_read_as_their_own_arm() -> None:
    from posttrain_lab.config import load_config
    from posttrain_lab.utils.checkpoints import run_arm, run_hyperparam

    out = load_config("helpful/arm_dpo_nll.yaml", ["train.beta=0.1"]).output_dir
    assert (run_arm(out), run_hyperparam(out)) == ("dpo-nll", "beta0.1")
    assert out.endswith("/helpful/runs/dpo-nll_beta0.1_seed17")


def test_sweep_grid_matches_dpo(repo) -> None:
    grids = [repo.python("-m", "posttrain_lab.config", "helpful", s).stdout.split() for s in ("dpo", "dpo_nll")]
    assert grids[0] == grids[1] == [f"train.beta={b}" for b in (0.01, 0.05, 0.1, 0.3)]


def test_every_table_knows_the_arm() -> None:
    from posttrain_lab import reset, status

    keys = [s.key for s in status.TRACKS["helpful"]["stages"]]
    assert keys == ["1", "2", "3a", "3b", "3e", "3c", "3d", "4", "5"]   # status follows the driver's order
    (stage,) = [s for s in status.TRACKS["helpful"]["stages"] if s.key == "3e"]
    assert (stage.yaml, stage.sweep, stage.gpus) == ("helpful/arm_dpo_nll.yaml", "dpo_nll", "1 each")
    assert status.log_name(stage.log, "train.beta=0.1") == "logs_dponll_train_beta_0_1.log"
    assert "3e" in reset.ARM_STAGES["helpful"] and "arm-dpo-nll" in reset.MAKE_TARGETS["helpful"]
    driver = (ROOT / "scripts" / "run_all_helpful.sh").read_text()
    assert driver.index("stage3b_arm_dpo.sh") < driver.index("=== stage 3e:") < driver.index("stage3e_arm_dpo_nll.sh") \
        < driver.index("=== stage 3c:")
    assert "arm-dpo-nll:\n\tbash scripts/helpful/stage3e_arm_dpo_nll.sh" in (ROOT / "Makefile").read_text()
    smoke = (ROOT / "scripts" / "run_smoke.sh").read_text()
    assert "python -m posttrain_lab.train.dpo configs/helpful/arm_dpo_nll.yaml train.beta=0.1" in smoke


def test_launcher_passes_the_nll_config_and_log_names(repo, tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "accelerate").write_text('#!/usr/bin/env bash\necho "ACCELERATE $*"\n')
    (bin_dir / "accelerate").chmod(0o755)
    env = {"PATH": f"{bin_dir}:{repo.env()['PATH']}", "RESULTS_DIR": str(tmp_path / "results")}
    run = repo.run("bash", "scripts/helpful/stage3e_arm_dpo_nll.sh", check=False, **env)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "[dponll] launching train.beta=0.3 on GPU 3" in run.stdout and "[dponll] all runs finished" in run.stdout
    logs = sorted(p.name for p in repo.root.glob("logs_*.log"))
    assert logs == [f"logs_dponll_train_beta_{b}.log" for b in ("0_01", "0_05", "0_1", "0_3")]
    assert "configs/helpful/arm_dpo_nll.yaml train.beta=0.1" in (repo.root / logs[2]).read_text()

    dpo = repo.run("bash", "scripts/helpful/stage3b_arm_dpo.sh", check=False, **env)
    assert dpo.returncode == 0 and "[dpo] launching train.beta=0.01 on GPU 0" in dpo.stdout
    assert "configs/helpful/arm_dpo.yaml train.beta=0.01" in (repo.root / "logs_dpo_train_beta_0_01.log").read_text()


def test_dpo_stage_reset_leaves_the_nll_arm_alone(repo, tmp_path: Path) -> None:
    helpful = tmp_path / "results" / "helpful"
    for run in ("dpo_beta0.1_seed17", "dpo-nll_beta0.1_seed17"):
        (helpful / "runs" / run / "checkpoint-5").mkdir(parents=True)
        (helpful / "eval" / run).mkdir(parents=True)
    for log in ("logs_dpo_train_beta_0_1.log", "logs_dponll_train_beta_0_1.log"):
        (repo.root / log).write_text("log\n")
    env = {"RESULTS_DIR": str(tmp_path / "results"), "HF_HOME": str(tmp_path / "hf")}
    run = repo.python("-m", "posttrain_lab.reset", "helpful", "--stage", "3b", "--yes", check=False, **env)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    assert [p.name for p in (helpful / "runs").iterdir()] == ["dpo-nll_beta0.1_seed17"]
    assert [p.name for p in (helpful / "eval").iterdir()] == ["dpo-nll_beta0.1_seed17"]
    assert [p.name for p in repo.root.glob("logs_*.log")] == ["logs_dponll_train_beta_0_1.log"]
    run = repo.python("-m", "posttrain_lab.reset", "helpful", "--stage", "3e", "--yes", check=False, **env)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    assert not list((helpful / "runs").iterdir()) and not list(repo.root.glob("logs_*.log"))


def test_status_names_the_reset_for_a_stale_nll_run(repo, tmp_path: Path) -> None:
    import json

    run_dir = tmp_path / "results" / "helpful" / "runs" / "dpo-nll_beta0.1_seed17"
    for k in (1, 2, 3):
        (run_dir / f"checkpoint-{10 * k}").mkdir(parents=True)
    (run_dir / "DONE").touch()
    out = repo.python("-m", "posttrain_lab.status", "helpful", "--json", check=False,
                      RESULTS_DIR=str(tmp_path / "results"), HF_HOME=str(tmp_path / "hf"))
    assert out.returncode == 0, out.stderr[-2000:]
    attention = "\n".join(json.loads(out.stdout)["attention"])
    assert 'make reset TASK=helpful RESET_ARGS="--stage 3e --dry-run"' in attention
