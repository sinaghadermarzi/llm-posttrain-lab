"""More training seeds on the math track.

SEED=N (make math-all SEED=18) must reach every math stage as seed=N, so a
second seed trains next to seed 17 without touching it or being skipped as
DONE because of its outputs; eval scores each seed against its own pi_SFT;
status and reset tell the seeds apart; and the analysis averages them."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
MATH_STAGES = ["math/m1_sft_gold.yaml", "math/m2_gen_pairs.yaml", "math/m3_arm_sft.yaml", "math/m3_arm_dpo.yaml",
               "math/m3_arm_dpo_nll.yaml", "math/m3_arm_grpo.yaml"]


# --- the seed reaches every stage ----------------------------------------------


def test_default_seed_is_the_configs() -> None:
    base = int(OmegaConf.load(ROOT / "configs" / "base.yaml").seed)
    assert f"DEFAULT_SEED={base}\n" in (ROOT / "scripts" / "math" / "seed.sh").read_text()
    assert f"SEED ?= {base}\n" in (ROOT / "Makefile").read_text()


@pytest.mark.parametrize("stage", MATH_STAGES)
def test_every_seed_dependent_path_follows_the_seed(stage: str) -> None:
    from posttrain_lab.config import load_config

    cfg = load_config(stage, ["seed=18"])
    paths = [str(cfg.output_dir), str(cfg.get("init_from") or ""), str(cfg.data.pairs_path), str(cfg.paths.sft_base)]
    assert all("seed17" not in p for p in paths), paths
    assert str(cfg.output_dir).endswith("_seed18")
    ev = load_config("eval.yaml", ["task=math", "seed=18"])
    assert str(ev.eval.reference).endswith("runs/sft-base_seed18") and str(ev.data.pairs_path).endswith("seed18")


def _stubs(repo, tmp_path: Path) -> dict[str, str]:
    """accelerate and the GPU python entry points print their argv instead of
    running; everything else (posttrain_lab.config) is the real python."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "accelerate").write_text('#!/usr/bin/env bash\necho "ACCELERATE $*"\n')
    (bin_dir / "python").write_text(
        "#!/usr/bin/env bash\n"
        'case "$*" in\n'
        '  *posttrain_lab.gen_pairs*|*posttrain_lab.eval.run_eval*|*posttrain_lab.utils.chat*|analysis/*)\n'
        '    echo "PYTHON $*"; exit 0 ;;\n'
        "esac\n"
        f'exec "{sys.executable}" "$@"\n')
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    return {"PATH": f"{bin_dir}:{repo.env()['PATH']}", "RESULTS_DIR": str(tmp_path / "results")}


def _launches(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.startswith(("ACCELERATE ", "PYTHON ")) and "posttrain_lab" in ln]


def _logs(repo) -> str:
    return "\n".join(p.read_text() for p in sorted(repo.root.glob("logs_math_*.log")))


def test_the_whole_pipeline_runs_seed_18_without_seed_17(repo, tmp_path: Path) -> None:
    env = _stubs(repo, tmp_path)
    run = repo.run("bash", "scripts/run_all_math.sh", check=False, SEED="18", **env)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    launches = _launches(run.stdout + "\n" + _logs(repo))
    # m1, probe, m2, m3a, 4 DPO, 4 DPO+NLL, 4 GRPO, 4 eval shards
    assert len(launches) == 1 + 1 + 1 + 1 + 4 + 4 + 4 + 4, launches
    gate = [ln for ln in launches if "utils.chat" in ln]
    assert gate == [f"PYTHON -m posttrain_lab.utils.chat {tmp_path}/results/math/runs/sft-base_seed18 "
                    "--probe configs/math/m1_sft_gold.yaml"]
    assert all("seed=18" in ln for ln in launches if ln not in gate), launches
    assert "seed17" not in run.stdout + _logs(repo) and "seed=17" not in run.stdout + _logs(repo)
    assert "=== math m3b: DPO sweep (parallel, 1 GPU each) [seed 18]" in run.stdout
    logs = sorted(p.name for p in repo.root.glob("logs_math_*.log"))
    assert "logs_math_dpo_train_beta_0_1_seed18.log" in logs and "logs_math_dponll_train_beta_0_3_seed18.log" in logs
    assert "logs_math_eval_shard3_seed18.log" in logs
    assert not [n for n in logs if not n.endswith("_seed18.log")]  # seed 17's logs keep their names, untouched


def test_the_default_seed_keeps_the_old_names(repo, tmp_path: Path) -> None:
    env = _stubs(repo, tmp_path)
    run = repo.run("bash", "scripts/math/m3b_arm_dpo.sh", check=False, **env)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "[math-dpo] launching train.beta=0.01 on GPU 0 (seed 17)" in run.stdout
    assert sorted(p.name for p in repo.root.glob("logs_math_*.log")) == [
        f"logs_math_dpo_train_beta_{b}.log" for b in ("0_01", "0_05", "0_1", "0_3")]
    assert "configs/math/m3_arm_dpo.yaml train.beta=0.1 seed=17" in (repo.root / "logs_math_dpo_train_beta_0_1.log").read_text()


def test_make_forwards_seed_and_runs_several(repo, tmp_path: Path) -> None:
    env = _stubs(repo, tmp_path)
    run = repo.run("make", "math-arm-sft", "SEED=19", check=False, **env)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "configs/math/m3_arm_sft.yaml seed=19" in run.stdout
    run = repo.run("make", "math-seeds", "SEEDS=18 19", check=False, **env)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    assert "MATH TRACK DONE [seed 18]" in run.stdout and "MATH TRACK DONE [seed 19]" in run.stdout
    assert run.stdout.index("[seed 18]") < run.stdout.index("[seed 19]")


def test_a_bad_seed_stops_the_script(repo, tmp_path: Path) -> None:
    env = _stubs(repo, tmp_path)
    run = repo.run("bash", "scripts/math/m3a_arm_sft.sh", check=False, SEED="1e3", **env)
    assert run.returncode == 2 and "SEED must be a non-negative integer" in run.stderr


# --- eval: one seed at a time --------------------------------------------------


def _ckpt(run: Path, step: int) -> Path:
    d = run / f"checkpoint-{step}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "trainer_state.json").write_text(json.dumps({"global_step": step, "max_steps": step, "log_history": []}))
    return d


def test_eval_discovers_only_its_seed(tmp_path: Path) -> None:
    from posttrain_lab.utils.checkpoints import discover_all_checkpoints, run_seed

    for name in ("sft-base_seed17", "sft-base_seed18", "dpo_beta0.1_seed17", "dpo_beta0.1_seed18", "rm_seed18"):
        _ckpt(tmp_path / name, 5)
    names = {c.parent.name for c in discover_all_checkpoints(tmp_path, seeds=[18])}
    assert names == {"sft-base_seed18", "dpo_beta0.1_seed18"}
    assert len(discover_all_checkpoints(tmp_path)) == 4
    assert run_seed("dpo-nll_beta0.05_seed19") == 19 and run_seed("sft-base") is None


def test_eval_refuses_another_seeds_checkpoint(repo, tmp_path: Path) -> None:
    ckpt = _ckpt(tmp_path / "results" / "math" / "runs" / "dpo_beta0.1_seed17", 5)
    run = repo.python("-m", "posttrain_lab.eval.run_eval", "task=math", "seed=18", str(ckpt), check=False,
                      RESULTS_DIR=str(tmp_path / "results"))
    assert run.returncode != 0 and "is a seed-17 run" in run.stderr and "pass seed=17" in run.stderr


# --- status ----------------------------------------------------------------------


def _two_seeds(repo, tmp_path: Path) -> dict[str, str]:
    math = tmp_path / "results" / "math"
    for seed in (17, 18):
        _ckpt(math / "runs" / f"sft-base_seed{seed}", 72)
    (math / "runs" / "sft-base_seed17" / "DONE").touch()            # seed 18's pi_SFT crashed
    _ckpt(math / "runs" / "dpo_beta0.1_seed17", 10)
    (math / "runs" / "dpo_beta0.1_seed17" / "DONE").touch()
    _ckpt(math / "runs" / "dpo_beta0.1_seed18", 4)                  # seed 18's DPO failed, own log
    (repo.root / "logs_math_dpo_train_beta_0_1_seed18.log").write_text("Traceback (most recent call last):\n"
                                                                        "ValueError: seed 18 boom\n")
    (repo.root / "logs_math_dpo_train_beta_0_1.log").write_text("all good\n")
    old = repo.root / "math-all.log"
    old.write_text("=== math m4: evaluate all checkpoints [seed 17]\n")
    os.utime(old, (time.time() - 3600,) * 2)
    (repo.root / "math-all-seed18.log").write_text(
        "=== math m1: pi_SFT-math (gold CoT) [seed 18]\n"
        "[rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n")
    return {"RESULTS_DIR": str(tmp_path / "results"), "HF_HOME": str(tmp_path / "hf")}


def _stages(report: dict) -> dict:
    (track,) = report["tracks"]
    return {s["key"]: s for s in track["stages"]}


def test_status_shows_one_run_per_seed(repo, tmp_path: Path) -> None:
    env = _two_seeds(repo, tmp_path)
    run = repo.python("-m", "posttrain_lab.status", "math", "--json", check=False, **env)
    assert run.returncode == 0, run.stderr[-3000:]
    report = json.loads(run.stdout)
    assert report["tracks"][0]["seeds"] == [17, 18]
    assert report["tracks"][0]["driver_log"].endswith("math-all-seed18.log")   # the newer log
    stages = _stages(report)
    m1 = {r["seed"]: r for r in stages["m1"]["runs"]}
    assert m1[17]["state"] == "done" and m1[18]["state"] == "failed"
    assert "OutOfMemoryError" in m1[18]["errors"]["last_exception"]
    dpo = {(r["seed"], r["override"]): r for r in stages["m3b"]["runs"]}
    assert len(dpo) == 8                                                       # 4 betas x 2 seeds
    assert dpo[17, "train.beta=0.1"]["state"] == "done"
    assert dpo[18, "train.beta=0.1"]["state"] == "failed"
    assert dpo[18, "train.beta=0.1"]["errors"]["last_exception"] == "ValueError: seed 18 boom"
    assert dpo[18, "train.beta=0.1"]["log"].endswith("logs_math_dpo_train_beta_0_1_seed18.log")
    ev = stages["m4"]["eval"]
    assert ev["per_seed"] == {"17": [0, 2], "18": [0, 2]}

    text = repo.python("-m", "posttrain_lab.status", "math", "--no-color", check=False, **env).stdout
    assert "seeds 17, 18" in text and "train.beta=0.1 [seed 18]" in text
    assert 'make math-seeds SEEDS="17 18"' in text


def test_status_seed_flag(repo, tmp_path: Path) -> None:
    env = _two_seeds(repo, tmp_path)
    run = repo.python("-m", "posttrain_lab.status", "math", "--seed", "18", "--json", check=False, **env)
    assert run.returncode == 0, run.stderr[-3000:]
    report = json.loads(run.stdout)
    stages = _stages(report)
    assert report["tracks"][0]["seeds"] == [18]
    assert {r["name"] for r in stages["m1"]["runs"]} == {"sft-base_seed18"}
    assert len(stages["m3b"]["runs"]) == 4 and stages["m4"]["eval"]["checkpoints"] == 2


def test_seed_flag_ignores_another_seeds_driver_log(repo, tmp_path: Path) -> None:
    """--seed 17 must not read the newer math-all-seed18.log (its stage, errors, resume log)."""
    env = _two_seeds(repo, tmp_path)
    (repo.root / "math-all.log").write_text("=== math m3a: SFT arm [seed 17]\n")
    os.utime(repo.root / "math-all.log", (time.time() - 3600,) * 2)
    run = repo.python("-m", "posttrain_lab.status", "math", "--seed", "17", "--json", check=False, **env)
    assert run.returncode == 0, run.stderr[-3000:]
    (track,) = json.loads(run.stdout)["tracks"]
    assert track["driver_log"].endswith("math-all.log") and track["log_stage"] == "=== math m3a: SFT arm [seed 17]"
    text = repo.python("-m", "posttrain_lab.status", "math", "--seed", "17", "--no-color", check=False, **env).stdout
    assert ">> math-all.log" in text and "seed18" not in text.split("resume", 1)[1].splitlines()[0]
    # an untagged header (m5 runs for every seed) still counts; --log is taken as given
    run = repo.python("-m", "posttrain_lab.status", "math", "--seed", "17", "--log",
                      str(repo.root / "math-all-seed18.log"), "--json", check=False, **env)
    assert json.loads(run.stdout)["tracks"][0]["driver_log"].endswith("math-all-seed18.log")


def test_seed_flag_ignores_another_seeds_eval_shards(monkeypatch, tmp_path: Path) -> None:
    from posttrain_lab import status
    from posttrain_lab.config import load_config

    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    _ckpt(tmp_path / "results" / "math" / "runs" / "sft-base_seed18", 72)
    cfg = load_config("eval.yaml", ["task=math"])
    stage = next(s for s in status.TRACKS["math"]["stages"] if s.key == "m4")
    jobs = [status.Job(module="eval.run_eval", args=["task=math", "seed=17", "--shard", "0/4"]),
            status.Job(module="eval.run_eval", args=["task=math", "--shard", "1/4"])]   # no seed=: the default, 17
    assert status.eval_state("math", cfg, jobs, stage, seeds=[18])["shards"] == []
    assert len(status.eval_state("math", cfg, jobs, stage, seeds=[17])["shards"]) == 2
    jobs.append(status.Job(module="eval.run_eval", args=["task=math", "seed=18", "--shard", "0/4"]))
    assert [sh["shard"] for sh in status.eval_state("math", cfg, jobs, stage, seeds=[18])["shards"]] == ["0/4"]


def test_resume_only_rebuilds_figures_when_every_run_is_done() -> None:
    from posttrain_lab import status

    def track(states: dict[int, str], evals: dict[str, list[int]]) -> dict:
        runs = [status.Run(name=f"r_seed{s}", override=None, output_dir="x", seed=s, state=st) for s, st in states.items()]
        return {"task": "math", "stages": [{"runs": runs}, {"runs": [], "eval": {"per_seed": evals}},
                                           {"runs": [], "figures": {}}]}

    assert status.resume_target(track({18: "done"}, {"18": [5, 5]})) == "make math-plots"   # not seed 17's math-all
    assert status.resume_target(track({18: "done"}, {"18": [4, 5]})) == "make math-all SEED=18"
    assert status.resume_target(track({17: "stopped", 18: "done"}, {"18": [5, 5]})) == "make math-all"
    assert status.resume_target(track({17: "pending", 18: "failed"}, {})) == 'make math-seeds SEEDS="17 18"'


def test_one_seed_reads_as_before(repo, tmp_path: Path) -> None:
    (tmp_path / "results" / "math" / "runs").mkdir(parents=True)
    run = repo.python("-m", "posttrain_lab.status", "math", "--json", check=False, RESULTS_DIR=str(tmp_path / "results"))
    report = json.loads(run.stdout)
    assert report["tracks"][0]["seeds"] == [17]
    assert [r["name"] for r in _stages(report)["m1"]["runs"]] == ["sft-base_seed17"]


def test_log_names_follow_the_launcher() -> None:
    from posttrain_lab import status

    assert status.log_name("logs_math_dpo_{ov}.log", "train.beta=0.1", seed=17) == "logs_math_dpo_train_beta_0_1.log"
    assert status.log_name("logs_math_dpo_{ov}.log", "train.beta=0.1", seed=18) == \
        "logs_math_dpo_train_beta_0_1_seed18.log"
    assert status.log_name("logs_math_eval_shard{i}.log", shard=3, seed=19) == "logs_math_eval_shard3_seed19.log"


# --- reset ------------------------------------------------------------------------


def _reset_tree(repo, tmp_path: Path) -> tuple[Path, dict[str, str]]:
    math = tmp_path / "results" / "math"
    for name in ("sft-base_seed17", "sft-base_seed18", "dpo_beta0.1_seed17", "dpo_beta0.3_seed17",
                 "dpo_beta0.1_seed18", "dpo_beta0.1_seed3", "grpo_beta0.0_seed18"):
        _ckpt(math / "runs" / name, 5)
        (math / "eval" / name).mkdir(parents=True)
    for log in ("logs_math_dpo_train_beta_0_1.log", "logs_math_dpo_train_beta_0_1_seed18.log", "math-all.log",
                "math-all-seed18.log"):
        (repo.root / log).write_text("log\n")
    return math, {"RESULTS_DIR": str(tmp_path / "results"), "HF_HOME": str(tmp_path / "hf")}


def test_stage_reset_of_one_seed(repo, tmp_path: Path) -> None:
    math, env = _reset_tree(repo, tmp_path)
    dry = repo.python("-m", "posttrain_lab.reset", "math", "--stage", "m3b", "--seed", "18", "--relaunch",
                      "--dry-run", check=False, **env)
    assert dry.returncode == 0, dry.stderr[-2000:]
    assert "math m3b seed 18 (DPO sweep only" in dry.stdout and "make math-all SEED=18" in dry.stdout
    run = repo.python("-m", "posttrain_lab.reset", "math", "--stage", "m3b", "--seed", "18", "--yes", check=False, **env)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    assert sorted(p.name for p in (math / "runs").iterdir()) == [
        "dpo_beta0.1_seed17", "dpo_beta0.1_seed3", "dpo_beta0.3_seed17", "grpo_beta0.0_seed18",
        "sft-base_seed17", "sft-base_seed18"]
    assert not (math / "eval" / "dpo_beta0.1_seed18").exists() and (math / "eval" / "dpo_beta0.1_seed17").exists()
    left = {p.name for p in repo.root.glob("*.log")}
    assert left == {"logs_math_dpo_train_beta_0_1.log", "math-all.log", "math-all-seed18.log"}
    assert "nohup make math-all SEED=18 > math-all-seed18.log" in run.stdout


def test_stage_reset_of_the_default_seed_keeps_other_seeds_logs(repo, tmp_path: Path) -> None:
    math, env = _reset_tree(repo, tmp_path)
    run = repo.python("-m", "posttrain_lab.reset", "math", "--stage", "m3b", "--seed", "17", "--yes", check=False, **env)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    assert not (math / "runs" / "dpo_beta0.3_seed17").exists() and (math / "runs" / "dpo_beta0.1_seed18").exists()
    assert {p.name for p in repo.root.glob("logs_math_*.log")} == {"logs_math_dpo_train_beta_0_1_seed18.log"}


def test_stage_reset_clears_every_seed_and_relaunches_those_with_a_pi_sft(repo, tmp_path: Path) -> None:
    math, env = _reset_tree(repo, tmp_path)
    dry = repo.python("-m", "posttrain_lab.reset", "math", "--stage", "m3b", "--relaunch", "--dry-run", check=False,
                      **env)
    assert dry.returncode == 0, dry.stderr[-2000:]
    assert "make math-seeds SEEDS=17 18" in dry.stdout                        # not seed 3: no pi_SFT
    run = repo.python("-m", "posttrain_lab.reset", "math", "--stage", "m3b", "--yes", check=False, **env)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    assert sorted(p.name for p in (math / "runs").iterdir()) == [
        "grpo_beta0.0_seed18", "sft-base_seed17", "sft-base_seed18"]
    assert not list(repo.root.glob("logs_math_*.log"))
    assert 'nohup make math-seeds SEEDS="17 18" > math-all.log' in run.stdout


def test_seed_needs_a_math_stage(repo, tmp_path: Path) -> None:
    _, env = _reset_tree(repo, tmp_path)
    for argv in (["math", "--seed", "18"], ["helpful", "--stage", "3b", "--seed", "18"]):
        run = repo.python("-m", "posttrain_lab.reset", *argv, "--dry-run", check=False, **env)
        assert run.returncode == 2 and "--seed needs --stage on the math track" in run.stderr


# --- analysis ---------------------------------------------------------------------


def _row(run: str, step: int, kl: float, pass1: float) -> dict:
    arm, *rest = run.split("_")
    hp = rest[0] if len(rest) > 1 else None
    return {"run": run, "arm": arm, "hyperparam": hp, "checkpoint": f"x/{run}/checkpoint-{step}", "step": step,
            "kl_k1": kl, "kl_k3": kl, "pass1": pass1, "pass1_se": 0.01, "pass8": pass1 + 0.3, "pass8_se": 0.01,
            "num_samples": 8, "eval_prompts": 1, "pass_gap": 0.3,
            # the other figures' metrics, each a function of kl so the seed mean is easy to check
            "distinct_2": 0.9 - kl / 10, "self_bleu": 0.1 + kl / 10, "mean_token_nll": 0.5 - kl / 10,
            "mean_response_tokens": 100 + 10 * kl, "ppl_wikitext": 10 + kl, "logp_chosen": -50 - kl,
            "logp_rejected": -60 - 2 * kl, "harness": {"arc_easy": 0.8 - kl / 100}}


def _eval_tree(root: Path, seeds=(17, 18)) -> None:
    rows = []
    for seed in seeds:
        rows.append(_row(f"sft-base_seed{seed}", 72, 0.0, 0.40))
        # DPO trains on each seed's own pairs: different step counts per seed
        steps = (10, 20) if seed == 17 else (11, 22)
        rows += [_row(f"dpo_beta0.1_seed{seed}", s, kl + (seed - 17) * 0.2, p + (seed - 17) * 0.02)
                 for s, kl, p in zip(steps, (1.0, 2.0), (0.50, 0.55))]
        rows += [_row(f"grpo_beta0.04_seed{seed}", s, kl, p) for s, kl, p in ((100, 1.0, 0.52), (200, 3.0, 0.6))]
    for r in rows:
        path = root / "eval" / r["run"] / f"checkpoint-{r['step']}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(r) + "\n")


def _aggregate(root: Path) -> subprocess.CompletedProcess:
    out = subprocess.run([sys.executable, str(ROOT / "analysis" / "aggregate.py"), str(root)], capture_output=True,
                         text=True, cwd=ROOT / "analysis")
    assert out.returncode == 0, out.stderr
    return out


def test_aggregate_averages_seeds_by_checkpoint_index(tmp_path: Path) -> None:
    _eval_tree(tmp_path)
    out = _aggregate(tmp_path)
    assert "seeds: [17, 18]" in out.stdout
    df = pd.read_csv(tmp_path / "frontier.csv")
    assert set(df["seed"]) == {17, 18} and "dpo_beta0.1_seed18" in set(df["run"])
    assert df.loc[df["run"] == "dpo_beta0.1_seed18", "ckpt_index"].tolist() == [1, 2]

    means = pd.read_csv(tmp_path / "frontier_seeds.csv")
    dpo1 = means[(means["arm"] == "dpo") & (means["ckpt_index"] == 1)].iloc[0]
    assert dpo1["n_seeds"] == 2 and dpo1["seeds"] == "17;18"
    assert (dpo1["step_min"], dpo1["step_max"]) == (10, 11)
    assert dpo1["pass1"] == pytest.approx(0.51) and dpo1["pass1_seed_se"] == pytest.approx(0.01)
    assert dpo1["kl_k1"] == pytest.approx(1.1) and dpo1["kl_k1_seed_se"] == pytest.approx(0.1)
    assert dpo1["pass1_prompt_se"] == pytest.approx((2 * 0.01 ** 2) ** 0.5 / 2)
    assert "num_samples" not in means.columns and "pass1_se" not in means.columns
    base = means[means["arm"] == "sft-base"]
    assert len(base) == 1 and base.iloc[0]["n_seeds"] == 2            # NaN hyperparam still groups
    assert len(means) == 1 + 2 + 2


def test_aggregate_with_one_seed(tmp_path: Path) -> None:
    _eval_tree(tmp_path, seeds=(17,))
    _aggregate(tmp_path)
    means = pd.read_csv(tmp_path / "frontier_seeds.csv")
    assert (means["n_seeds"] == 1).all() and means["pass1_seed_se"].isna().all()


def _plot(tmp_path: Path, monkeypatch, script: str = "frontier_math"):
    """Run analysis/<script>.py on tmp_path; the last figure it drew."""
    import importlib

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.syspath_prepend(str(ROOT / "analysis"))
    module = importlib.import_module(script)

    monkeypatch.setattr(sys, "argv", [f"{script}.py", str(tmp_path)])
    plt.close("all")
    module.main()
    return plt.gcf()


def test_frontier_plots_the_seed_mean_with_error_bars(tmp_path: Path, monkeypatch) -> None:
    _eval_tree(tmp_path)
    _aggregate(tmp_path)
    fig = _plot(tmp_path, monkeypatch)
    assert "mean ± SE over seeds 17, 18" in fig._suptitle.get_text()
    ax = fig.axes[0]
    dpo = [ln for ln in ax.get_lines() if ln.get_color() == "#1baf7a" and len(ln.get_xdata()) > 1]
    assert len(dpo) == 1                                                # one curve per beta, not per seed
    assert list(dpo[0].get_xdata()) == pytest.approx([1.1, 2.1])
    bars = [c for c in ax.containers if c.has_xerr and c.has_yerr]
    assert bars, "no across-seed error bars on both axes"


def test_frontier_with_one_seed_draws_every_run(tmp_path: Path, monkeypatch) -> None:
    _eval_tree(tmp_path, seeds=(17,))
    _aggregate(tmp_path)
    fig = _plot(tmp_path, monkeypatch)
    assert "seeds" not in fig._suptitle.get_text()
    ax = fig.axes[0]
    assert all(not c.has_xerr for c in ax.containers)                  # prompt SEs on y only, as before
    dpo = [ln for ln in ax.get_lines() if ln.get_color() == "#1baf7a" and len(ln.get_xdata()) > 1]
    assert list(dpo[0].get_xdata()) == pytest.approx([1.0, 2.0])


# the other figures: seed means and across-seed bars with several seeds, every run with one


# (script, a panel title, that panel's seed-mean DPO y values: self_bleu = 0.1 + kl/10, ppl = 10 + kl,
# length = 100 + 10 kl, averaged over seed 17's and seed 18's (kl + 0.2) checkpoints)
SCATTER_FIGURES = [("diversity_plot", "Sample convergence", [0.21, 0.31]), ("retention_plot", "LM erosion", [11.1, 12.1]),
                   ("diagnostics", "Length vs KL", [111.0, 121.0])]


def _scatter(ax, arm: str) -> list[tuple[float, float]]:
    return [tuple(xy) for c in ax.collections if c.get_label() == arm for xy in c.get_offsets()]


@pytest.mark.parametrize("script,panel,expected", SCATTER_FIGURES)
def test_scatter_figures_plot_the_seed_mean_with_error_bars(tmp_path: Path, monkeypatch, script: str, panel: str,
                                                             expected: list[float]) -> None:
    _eval_tree(tmp_path)
    _aggregate(tmp_path)
    fig = _plot(tmp_path, monkeypatch, script)
    assert fig._suptitle.get_text() == "mean ± SE over seeds 17, 18"
    for ax in fig.axes:
        dpo = sorted(_scatter(ax, "dpo"))
        assert len(dpo) == 2                                            # one point per checkpoint index, not per seed
        assert [x for x, _ in dpo] == pytest.approx([1.1, 2.1])
        assert len(_scatter(ax, "sft-base")) == 1
        bars = [c for c in ax.containers if c.has_xerr and c.has_yerr]
        assert bars, f"{script}: no across-seed error bars on both axes in {ax.get_title()!r}"
    ax = next(ax for ax in fig.axes if ax.get_title() == panel)
    assert [y for _, y in sorted(_scatter(ax, "dpo"))] == pytest.approx(expected)


@pytest.mark.parametrize("script", [s for s, _, _ in SCATTER_FIGURES])
def test_scatter_figures_with_one_seed_draw_every_checkpoint(tmp_path: Path, monkeypatch, script: str) -> None:
    _eval_tree(tmp_path, seeds=(17,))
    _aggregate(tmp_path)
    fig = _plot(tmp_path, monkeypatch, script)
    assert fig._suptitle is None
    for ax in fig.axes:
        assert not ax.containers                                        # no error bars, as before
        assert sorted(x for x, _ in _scatter(ax, "dpo")) == pytest.approx([1.0, 2.0])


@pytest.mark.parametrize("seeds", [(17,), (17, 18)])
def test_retention_has_one_panel_per_benchmark(tmp_path: Path, monkeypatch, seeds) -> None:
    """With several seeds the rows carry harness_<task>_se too: error bars, not panels of their own."""
    _eval_tree(tmp_path, seeds=seeds)
    _aggregate(tmp_path)
    fig = _plot(tmp_path, monkeypatch, "retention_plot")
    assert [ax.get_title() for ax in fig.axes] == ["LM erosion", "arc_easy"]
    arc = fig.axes[1]
    assert all(c.has_yerr for c in arc.containers) and bool(arc.containers) == (len(seeds) > 1)


def test_scatter_figures_with_two_seeds_and_no_seed_column(tmp_path: Path, monkeypatch) -> None:
    """A frontier.csv written before the seed columns existed: the run names still tell the seeds apart."""
    _eval_tree(tmp_path)
    _aggregate(tmp_path)
    df = pd.read_csv(tmp_path / "frontier.csv")
    df.drop(columns=["seed", "ckpt_index"]).to_csv(tmp_path / "frontier.csv", index=False)
    fig = _plot(tmp_path, monkeypatch, "diversity_plot")
    assert sorted(x for x, _ in _scatter(fig.axes[0], "dpo")) == pytest.approx([1.1, 2.1])


def _logp_lines(fig, arm_title: str):
    ax = next(ax for ax in fig.axes if ax.get_title() == arm_title)
    return ax, [ln for ln in ax.get_lines() if ln.get_linestyle() in ("-", "--") and len(ln.get_xdata()) > 1]


def test_pref_logps_plots_the_seed_mean_by_checkpoint_index(tmp_path: Path, monkeypatch) -> None:
    _eval_tree(tmp_path)
    _aggregate(tmp_path)
    fig = _plot(tmp_path, monkeypatch, "pref_logps_plot")
    assert "mean ± SE over seeds 17, 18" in fig._suptitle.get_text()
    ax, lines = _logp_lines(fig, "DPO")
    assert len(lines) == 2                                              # one chosen + one rejected, not per seed
    chosen = next(ln for ln in lines if ln.get_linestyle() == "-")
    assert list(chosen.get_xdata()) == pytest.approx([10.5, 21.0])     # steps 10/11 and 20/22: matched by index
    assert list(chosen.get_ydata()) == pytest.approx([-51.1, -52.1])
    assert ax.get_xlabel() == "training step (mean over seeds)"
    assert len([c for c in ax.containers if c.has_yerr]) == 2
    base = [ln for ln in ax.get_lines() if ln.get_linestyle() == ":"]
    assert base[0].get_ydata()[0] == pytest.approx(-50.0)              # pi_SFT: the mean of both seeds' (kl 0)


def test_pref_logps_with_one_seed_draws_every_run(tmp_path: Path, monkeypatch) -> None:
    _eval_tree(tmp_path, seeds=(17,))
    _aggregate(tmp_path)
    fig = _plot(tmp_path, monkeypatch, "pref_logps_plot")
    assert "seeds" not in fig._suptitle.get_text()
    ax, lines = _logp_lines(fig, "DPO")
    assert [list(ln.get_xdata()) for ln in lines] == [[10, 20], [10, 20]]
    assert ax.get_xlabel() == "training step" and not ax.containers


def _dpo_run(root: Path, name: str, steps: list[int]) -> None:
    d = root / "runs" / name / f"checkpoint-{steps[-1]}"
    d.mkdir(parents=True)
    history = [{"step": s, "logps/chosen": -100.0 - s, "logps/rejected": -90.0 - 2 * s} for s in steps]
    (d / "trainer_state.json").write_text(json.dumps({"log_history": history}))


def _dpo_logps(tmp_path: Path, monkeypatch, seeds) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.syspath_prepend(str(ROOT / "analysis"))
    import diagnostics

    for seed in seeds:
        steps = [10, 20] if seed == 17 else [10, 20, 30]                 # each seed's own pairs: its own length
        for beta in ("0.1", "0.3"):
            _dpo_run(tmp_path, f"dpo_beta{beta}_seed{seed}", steps)
    plt.close("all")
    (tmp_path / "figures").mkdir()
    diagnostics.dpo_logp_trajectories(tmp_path, tmp_path / "figures")
    return plt.gcf().axes[0]


def test_dpo_logps_keeps_a_line_per_seed_colored_by_beta(tmp_path: Path, monkeypatch) -> None:
    ax = _dpo_logps(tmp_path, monkeypatch, (17, 18))
    lines = ax.get_lines()
    assert len(lines) == 2 * 2 * 2                                      # beta x seed x (chosen, rejected)
    by_color: dict[str, set[int]] = {}
    for ln in lines:
        by_color.setdefault(ln.get_color(), set()).add(len(ln.get_xdata()))
    assert len(by_color) == 2 and all(lengths == {2, 3} for lengths in by_color.values())   # both seeds per beta
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["β=0.1 chosen", "β=0.3 chosen"]
    assert ax.get_title().endswith("one line per training seed")
    assert [t.get_text() for t in ax.texts] == ["β=0.1", "β=0.3"]


def test_dpo_logps_with_one_seed_is_unchanged(tmp_path: Path, monkeypatch) -> None:
    ax = _dpo_logps(tmp_path, monkeypatch, (17,))
    from style import BLUE_RAMP

    assert [ln.get_color() for ln in ax.get_lines()] == [BLUE_RAMP[0]] * 2 + [BLUE_RAMP[1]] * 2
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["β=0.1 chosen", "β=0.3 chosen"]
    assert ax.get_title() == "DPO: chosen (solid) and rejected (dashed) log-probs"


def _samples(path: Path, solved: dict[str, int], kl: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for pid, k in solved.items():
            for j in range(8):
                f.write(json.dumps({"prompt_id": pid, "correct": j < k}) + "\n")
    path.with_name(path.name.replace(".samples", "")).write_text(json.dumps({"kl_k1": kl}) + "\n")


def _compare(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "analysis" / "compare_math.py"), str(root), *args],
                          capture_output=True, text=True, cwd=ROOT / "analysis")


def _two_seed_samples(root: Path) -> None:
    prompts = [f"gsm8k-{i}" for i in range(100)]
    ev = root / "eval"
    # GRPO solves one more of 8 than DPO on every prompt in seed 17, two more in seed 18
    for seed, gain, dpo_step in ((17, 1, 116), (18, 2, 118)):
        _samples(ev / f"dpo_beta0.3_seed{seed}" / "checkpoint-58.samples.jsonl", {p: 1 for p in prompts})
        _samples(ev / f"dpo_beta0.3_seed{seed}" / f"checkpoint-{dpo_step}.samples.jsonl",
                 {p: i % 5 for i, p in enumerate(prompts)})
        _samples(ev / f"grpo_beta0.01_seed{seed}" / "checkpoint-374.samples.jsonl",
                 {p: i % 5 + gain for i, p in enumerate(prompts)})


def test_compare_needs_a_seed_when_a_prefix_matches_several(tmp_path: Path) -> None:
    _two_seed_samples(tmp_path)
    out = _compare(tmp_path, "dpo_beta0.3@last", "grpo_beta0.01@374")
    assert out.returncode != 0 and "seeds 17, 18: pass --seed N or --pool" in out.stderr
    out = _compare(tmp_path, "--seed", "18", "dpo_beta0.3@118", "grpo_beta0.01@374")
    assert out.returncode == 0, out.stderr
    line = next(ln for ln in out.stdout.split("paired vs", 1)[1].splitlines() if "grpo_beta0.01@374" in ln)
    assert "+0.250 ± 0.000" in line and "better" in line
    out = _compare(tmp_path, "dpo_beta0.3_seed17@ckpt2", "grpo_beta0.01_seed17@last")   # explicit names still work
    assert out.returncode == 0 and "+0.125 ± 0.000" in out.stdout


def test_compare_pools_seeds(tmp_path: Path) -> None:
    _two_seed_samples(tmp_path)
    out = _compare(tmp_path, "--pool", "dpo_beta0.3@ckpt2", "grpo_beta0.01@last")
    assert out.returncode == 0, out.stderr
    assert "pooled over seeds 17, 18" in out.stdout
    paired = out.stdout.split("paired vs", 1)[1]
    assert re.search(r"seed 17\s+\+0\.125", paired) and re.search(r"seed 18\s+\+0\.250", paired)
    seeds_line = next(ln for ln in paired.splitlines() if "seeds " in ln and "t1 = 12.71" in ln)
    assert "+0.188 ± 0.062" in seeds_line and "(2 seeds" in seeds_line
    out = _compare(tmp_path, "--pool", "dpo_beta0.3@116", "grpo_beta0.01@last")   # seed 18's DPO has no step 116
    assert out.returncode != 0 and "no checkpoint-116 samples in dpo_beta0.3_seed18" in out.stderr


def _kl_samples(root: Path) -> None:
    """Three seeds whose runs reach a given KL at different checkpoint positions (FINDINGS M9).
    Every prompt solves k of 8 at a run's k-th checkpoint (GRPO k + 1), so accuracy rises with
    position while the checkpoint nearest a KL target moves between seeds."""
    prompts = [f"gsm8k-{i}" for i in range(50)]
    ev = root / "eval"
    dpo = {17: ((29, 58, 87, 116), (1.0, 2.0, 3.0, 4.0)),
           18: ((30, 60, 90, 120), (2.1, 4.2, 6.3, 8.4)),
           19: ((29, 58, 87, 116), (1.5, 3.1, 4.6, 6.1))}
    grpo = {17: (1.2, 2.2, 2.9, 3.3, 3.5), 18: (1.0, 2.0, 2.4, 3.2, 3.6), 19: (1.4, 2.6, 3.0, 3.4, 3.9)}
    for seed, (steps, kls) in dpo.items():
        for k, (step, kl) in enumerate(zip(steps, kls), 1):
            _samples(ev / f"dpo_beta0.3_seed{seed}" / f"checkpoint-{step}.samples.jsonl", {p: k for p in prompts}, kl)
        for k, (step, kl) in enumerate(zip((75, 150, 225, 300, 374), grpo[seed]), 1):
            _samples(ev / f"grpo_beta0.01_seed{seed}" / f"checkpoint-{step}.samples.jsonl",
                     {p: k + 1 for p in prompts}, kl)


def _row_of(out: str, *words: str) -> str:
    return next(ln for ln in out.splitlines() if all(w in ln for w in words))


def test_compare_kl_target_picks_the_nearest_checkpoint_per_seed(tmp_path: Path) -> None:
    _kl_samples(tmp_path)
    out = _compare(tmp_path, "--seed", "19", "dpo_beta0.3@kl3", "grpo_beta0.01@kl3")
    assert out.returncode == 0, out.stderr
    # KL 3.1 is nearest 3, though ckpt 4 (6.1 nats) solves more: KL alone picks
    row = _row_of(out.stdout, "dpo_beta0.3@kl3", "3.1")
    assert "ckpt2 (step 58)" in row and "!" not in row and "0.250" in row
    assert "ckpt3 (step 225)" in _row_of(out.stdout, "grpo_beta0.01@kl3 ", "3.0")
    paired = out.stdout.split("paired vs", 1)[1]
    assert "+0.250 ± 0.000" in paired and "not matched" not in paired
    assert "nats from its target" not in out.stdout
    # seed 18's DPO reaches 3 nats nowhere near a checkpoint: 2.1 is nearest, flagged, and so is the pair
    out = _compare(tmp_path, "--seed", "18", "grpo_beta0.01@kl3", "dpo_beta0.3@kl3")
    assert out.returncode == 0, out.stderr
    assert "ckpt1 (step 30)  ! -0.90 nats from 3" in _row_of(out.stdout, "dpo_beta0.3@kl3", "2.1")
    assert "more than 0.3 nats from its target" in out.stdout
    assert "! KL -1.10 nats from the baseline: not matched" in out.stdout.split("paired vs", 1)[1]


def test_compare_kl_target_edges(tmp_path: Path) -> None:
    _kl_samples(tmp_path)
    # past the end of a run: its last checkpoint, flagged
    out = _compare(tmp_path, "--seed", "17", "dpo_beta0.3@kl2.5", "grpo_beta0.01@kl5")
    assert out.returncode == 0, out.stderr
    assert "ckpt5 (step 374)  ! -1.50 nats from 5" in _row_of(out.stdout, "grpo_beta0.01@kl5")
    # 2.0 and 3.0 are equally near 2.5: the earlier checkpoint
    assert "ckpt2 (step 58)" in _row_of(out.stdout, "dpo_beta0.3@kl2.5", "2.0")
    # step and position specs are unchanged and get no klX notes, but a KL gap between a pair is still flagged
    out = _compare(tmp_path, "--seed", "18", "dpo_beta0.3@ckpt1", "grpo_beta0.01@last")
    assert out.returncode == 0, out.stderr
    assert "(step " not in out.stdout and "nats from its target" not in out.stdout
    assert "! KL +1.50 nats from the baseline" in out.stdout
    for f in (tmp_path / "eval" / "dpo_beta0.3_seed17").glob("checkpoint-*.jsonl"):
        if not f.name.endswith(".samples.jsonl"):
            f.unlink()
    out = _compare(tmp_path, "--seed", "17", "dpo_beta0.3@kl3", "grpo_beta0.01@kl3")
    assert out.returncode != 0 and "no kl_k1 in the eval rows of dpo_beta0.3_seed17" in out.stderr


def test_compare_pools_kl_targets(tmp_path: Path) -> None:
    _kl_samples(tmp_path)
    out = _compare(tmp_path, "--pool", "grpo_beta0.01@kl3", "dpo_beta0.3@kl3")
    assert out.returncode == 0, out.stderr
    assert "pooled over seeds 17, 18, 19" in out.stdout
    # each seed's own nearest checkpoint: DPO ckpt 3 / 1 / 2, GRPO ckpt 3 / 4 / 3
    for seed, kl, ckpt in ((17, "3.0", "ckpt3 (step 87)"), (18, "2.1", "ckpt1 (step 30)"),
                           (19, "3.1", "ckpt2 (step 58)")):
        assert ckpt in _row_of(out.stdout, "dpo_beta0.3@kl3", f" {seed} ", kl)
    assert "! -0.90 nats from 3" in _row_of(out.stdout, "dpo_beta0.3@kl3", " 18 ")
    assert "ckpt4 (step 300)" in _row_of(out.stdout, "grpo_beta0.01@kl3", " 18 ", "3.2")
    paired = out.stdout.split("paired vs", 1)[1]
    assert re.search(r"seed 17\s+-0\.125", paired) and re.search(r"seed 18\s+-0\.500", paired)
    assert re.search(r"seed 19\s+-0\.250", paired)
    assert "not matched" in _row_of(paired, "seed 18") and "not matched" not in _row_of(paired, "seed 17")
    seeds_line = next(ln for ln in paired.splitlines() if "t2 = 4.30" in ln)
    assert "-0.292 ± 0.110" in seeds_line and "(3 seeds" in seeds_line
