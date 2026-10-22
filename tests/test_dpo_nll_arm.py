"""The DPO+NLL arm (math m3c) and the KL axis of the figures.

m3c is the DPO arm plus TRL's NLL term on the chosen answer (IRPO/RPO): its
config must reach DPOConfig as a two-term loss, its runs must read as their
own arm ('dpo-nll', not 'dpo'), and every launcher, make target, status and
reset table must know it, without the DPO arm's reset or logs touching it."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_config_reaches_dpo_config_as_two_losses() -> None:
    from trl import DPOConfig

    from posttrain_lab.config import load_config, train_kwargs

    nll = load_config("math/m3_arm_dpo_nll.yaml", ["train.beta=0.05"])
    dpo = load_config("math/m3_arm_dpo.yaml", ["train.beta=0.05"])
    args = DPOConfig(output_dir="unused", **train_kwargs(nll, DPOConfig))
    assert args.loss_type == ["sigmoid", "sft"] and args.loss_weights == [1.0, 1.0] and args.beta == 0.05
    # the NLL term is the only difference from the DPO arm
    same = ("learning_rate", "num_train_epochs", "save_steps", "warmup_ratio", "lr_scheduler_type", "max_length",
            "per_device_train_batch_size", "gradient_accumulation_steps")
    assert {k: nll.train[k] for k in same} == {k: dpo.train[k] for k in same}
    assert (nll.init_from, nll.data) == (dpo.init_from, dpo.data)
    assert "loss_type" not in dpo.train


def test_runs_read_as_their_own_arm() -> None:
    from posttrain_lab.config import load_config
    from posttrain_lab.utils.checkpoints import run_arm, run_hyperparam

    out = load_config("math/m3_arm_dpo_nll.yaml", ["train.beta=0.1"]).output_dir
    assert (run_arm(out), run_hyperparam(out)) == ("dpo-nll", "beta0.1")
    assert run_arm(load_config("math/m3_arm_dpo.yaml", ["train.beta=0.1"]).output_dir) == "dpo"


def test_sweep_grid_matches_dpo(repo) -> None:
    grids = [repo.python("-m", "posttrain_lab.config", "math", stage).stdout.split() for stage in ("dpo", "dpo_nll")]
    assert grids[0] == grids[1] == [f"train.beta={b}" for b in (0.01, 0.05, 0.1, 0.3)]


def test_every_table_knows_the_arm() -> None:
    from posttrain_lab import reset, status

    (stage,) = [s for s in status.TRACKS["math"]["stages"] if s.key == "m3c"]
    assert (stage.yaml, stage.sweep, stage.gpus) == ("math/m3_arm_dpo_nll.yaml", "dpo_nll", "1 each")
    assert status.log_name(stage.log, "train.beta=0.1") == "logs_math_dponll_train_beta_0_1.log"
    assert "m3c" in reset.ARM_STAGES["math"] and "math-arm-dpo-nll" in reset.MAKE_TARGETS["math"]
    driver = (ROOT / "scripts" / "run_all_math.sh").read_text()
    assert driver.index("m3b_arm_dpo.sh") < driver.index("=== math m3c:") < driver.index("m3c_arm_dpo_nll.sh") \
        < driver.index("=== math m3d:")
    assert "math-arm-dpo-nll:\n\tbash scripts/math/m3c_arm_dpo_nll.sh" in (ROOT / "Makefile").read_text()


def test_launcher_passes_the_nll_config_and_log_names(repo, tmp_path: Path) -> None:
    """m3c runs the m3b launcher with its own config, grid and log prefix
    (accelerate stubbed: prints its argv)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "accelerate").write_text('#!/usr/bin/env bash\necho "ACCELERATE $*"\n')
    (bin_dir / "accelerate").chmod(0o755)
    path = f"{bin_dir}:{repo.env()['PATH']}"
    run = repo.run("bash", "scripts/math/m3c_arm_dpo_nll.sh", check=False, PATH=path,
                   RESULTS_DIR=str(tmp_path / "results"))
    assert run.returncode == 0, run.stdout + run.stderr
    assert "[math-dponll] launching train.beta=0.3 on GPU 3" in run.stdout
    logs = sorted(p.name for p in repo.root.glob("logs_math_*.log"))
    assert logs == [f"logs_math_dponll_train_beta_{b}.log" for b in ("0_01", "0_05", "0_1", "0_3")]
    assert "configs/math/m3_arm_dpo_nll.yaml train.beta=0.1" in (repo.root / logs[2]).read_text()

    dpo = repo.run("bash", "scripts/math/m3b_arm_dpo.sh", check=False, PATH=path, RESULTS_DIR=str(tmp_path / "results"))
    assert dpo.returncode == 0 and "[math-dpo] launching train.beta=0.01 on GPU 0" in dpo.stdout
    assert "configs/math/m3_arm_dpo.yaml train.beta=0.01" in (repo.root / "logs_math_dpo_train_beta_0_01.log").read_text()


def test_dpo_stage_reset_leaves_the_nll_arm_alone(repo, tmp_path: Path) -> None:
    math = tmp_path / "results" / "math"
    for run in ("dpo_beta0.1_seed17", "dpo-nll_beta0.1_seed17"):
        (math / "runs" / run / "checkpoint-5").mkdir(parents=True)
        (math / "eval" / run).mkdir(parents=True)
    for log in ("logs_math_dpo_train_beta_0_1.log", "logs_math_dponll_train_beta_0_1.log"):
        (repo.root / log).write_text("log\n")
    env = {"RESULTS_DIR": str(tmp_path / "results"), "HF_HOME": str(tmp_path / "hf")}
    run = repo.python("-m", "posttrain_lab.reset", "math", "--stage", "m3b", "--yes", check=False, **env)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    assert [p.name for p in (math / "runs").iterdir()] == ["dpo-nll_beta0.1_seed17"]
    assert [p.name for p in (math / "eval").iterdir()] == ["dpo-nll_beta0.1_seed17"]
    assert [p.name for p in repo.root.glob("logs_math_*.log")] == ["logs_math_dponll_train_beta_0_1.log"]
    run = repo.python("-m", "posttrain_lab.reset", "math", "--stage", "m3c", "--yes", check=False, **env)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    assert not list((math / "runs").iterdir()) and not list(repo.root.glob("logs_math_*.log"))


# --- figures -----------------------------------------------------------------


def _frontier(root: Path) -> None:
    rows = [{"run": "sft-base_seed17", "arm": "sft-base", "hyperparam": None, "step": 72, "kl_k1": 0.0, "kl_k3": 0.0,
             "pass1": 0.42, "pass8": 0.81}]
    for arm in ("dpo", "dpo-nll", "grpo"):
        for step, kl1, kl3 in ((29, 0.5, 0.6), (58, 3.0, 1e21)):   # k3 explodes far out, k1 does not
            rows.append({"run": f"{arm}_beta0.1_seed17", "arm": arm, "hyperparam": "beta0.1", "step": step,
                         "kl_k1": kl1, "kl_k3": kl3, "pass1": 0.5, "pass8": 0.85})
    pd.DataFrame(rows).to_csv(root / "frontier.csv", index=False)


def test_math_frontier_plots_k1_on_symlog_with_the_nll_arm_styled(tmp_path: Path, monkeypatch) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.syspath_prepend(str(ROOT / "analysis"))
    import frontier_math

    _frontier(tmp_path)
    monkeypatch.setattr(sys, "argv", ["frontier_math.py", str(tmp_path)])
    plt.close("all")
    frontier_math.main()
    assert (tmp_path / "figures" / "frontier_math.png").exists()
    ax = plt.gcf().axes[0]
    assert ax.get_xscale() == "symlog" and "k1" in ax.get_xlabel()
    assert max(max(line.get_xdata(), default=0) for line in ax.get_lines()) == pytest.approx(3.0)  # not 1e21
    nll = [line for line in ax.get_lines() if line.get_color() == "#e87ba4"]
    assert nll and all(line.get_linestyle() == "--" and line.get_marker() == "D" for line in nll)
    labels = [t.get_text() for t in plt.gcf().legends[0].get_texts()]
    assert "DPO + NLL" in labels and "DPO" in labels


def test_nll_arm_never_relies_on_color_alone(monkeypatch) -> None:
    """Magenta vs DPO's aqua is CVD ΔE 6.1 (all-pairs worst of the set): legal
    only with a second encoding, so the arm also gets its own line and marker."""
    monkeypatch.syspath_prepend(str(ROOT / "analysis"))
    from style import METHOD_COLORS, METHOD_LINESTYLES, METHOD_MARKERS

    assert METHOD_COLORS["dpo-nll"] == "#e87ba4"
    assert METHOD_LINESTYLES["dpo-nll"] != "-" and METHOD_MARKERS["dpo-nll"] not in ("o", "*")


def test_diagnostics_runs_on_math(tmp_path: Path, monkeypatch) -> None:
    """The math pipeline now calls diagnostics.py: one dpo_logps panel per DPO
    arm, and diagnostics.png without the helpful-only reward-gap panel."""
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.syspath_prepend(str(ROOT / "analysis"))
    import diagnostics

    _frontier(tmp_path)
    df = pd.read_csv(tmp_path / "frontier.csv")
    df["mean_response_tokens"], df["mean_token_nll"] = 130.0, 0.4   # math has these, not rm_gap_z
    df.to_csv(tmp_path / "frontier.csv", index=False)
    for arm in ("dpo", "dpo-nll"):
        d = tmp_path / "runs" / f"{arm}_beta0.1_seed17" / "checkpoint-10"
        d.mkdir(parents=True)
        (d / "trainer_state.json").write_text(json.dumps(
            {"log_history": [{"step": 10, "logps/chosen": -60.0, "logps/rejected": -80.0}]}))

    figs = []
    monkeypatch.setattr(plt.Figure, "savefig", lambda self, path, **kw: figs.append((Path(path).name, self)))
    monkeypatch.setattr(sys, "argv", ["diagnostics.py", str(tmp_path)])
    diagnostics.main()
    saved = dict(figs)
    assert set(saved) == {"dpo_logps.png", "diagnostics.png"}
    titles = [ax.get_title() for ax in saved["dpo_logps.png"].axes]
    assert len(titles) == 2 and titles[1].startswith("DPO + NLL")
    assert [ax.get_title() for ax in saved["diagnostics.png"].axes] == ["Length vs KL", "Entropy vs KL"]
