"""The helpful track's scale probe (PROBE=1, scripts/run_probe.sh, make helpful-probe).

The helpful track had only run as the 0.5B smoke: PPO (four 1.5B models at
512-token responses), helpful GRPO with its reward model next to vLLM, DPO at
1536 tokens on one GPU and the 8B judge had never run at full scale, where an
OOM would surface hours into a ~3-day run. The probe runs the real pipeline
at full scale for a few steps per stage."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from posttrain_lab.config import load_config

STAGES = ["helpful/stage1_sft_base.yaml", "helpful/stage2_rm.yaml", "helpful/arm_sft.yaml", "helpful/arm_dpo.yaml",
          "helpful/arm_ppo.yaml", "helpful/arm_grpo.yaml"]
SCALE = ["per_device_train_batch_size", "gradient_accumulation_steps", "max_length", "max_completion_length",
         "response_length", "local_rollout_forward_batch_size", "num_generations", "gradient_checkpointing",
         "vllm_gpu_memory_utilization", "use_vllm"]


@pytest.mark.parametrize("stage", STAGES)
def test_the_probe_keeps_full_scale_and_takes_a_few_steps(monkeypatch, stage: str) -> None:
    real = load_config(stage, [])
    monkeypatch.setenv("PROBE", "1")
    probe = load_config(stage, [])
    assert str(probe.results_root).endswith("/probe/helpful")
    assert probe.model == real.model                                   # 1.5B, flash attention
    assert {k: probe.train.get(k) for k in SCALE} == {k: real.train.get(k) for k in SCALE}
    assert probe.train.max_steps == 4 and probe.train.save_steps == 2 and probe.train.report_to == "none"
    assert probe.data.max_samples == 512


def test_the_probe_evaluates_like_the_real_run(monkeypatch) -> None:
    real = load_config("eval.yaml", ["task=helpful"])
    monkeypatch.setenv("PROBE", "1")
    probe = load_config("eval.yaml", ["task=helpful"])
    for key in ("judge", "max_new_tokens", "use_vllm", "samples_per_prompt", "metric_groups"):
        assert probe.eval[key] == real.eval[key], key                  # the real 8B judge, full-length samples
    assert probe.eval.num_prompts == 16 and str(probe.eval.reference).startswith(str(probe.results_root))


def test_smoke_and_probe_together_are_refused(monkeypatch) -> None:
    monkeypatch.setenv("SMOKE", "1")
    monkeypatch.setenv("PROBE", "1")
    with pytest.raises(ValueError, match="SMOKE=1 and PROBE=1"):
        load_config("helpful/arm_dpo.yaml", [])


def test_each_sweep_runs_its_most_memory_hungry_value(repo) -> None:
    def sweep(stage: str, **env: str) -> list[str]:
        return repo.python("-m", "posttrain_lab.config", "helpful", stage, **env).stdout.split()

    assert sweep("grpo") == ["train.beta=0.0", "train.beta=0.01", "train.beta=0.04", "train.beta=0.1"]
    assert sweep("grpo", PROBE="1") == ["train.beta=0.1"]              # beta 0.0 loads no reference model
    assert sweep("dpo", PROBE="1") == ["train.beta=0.3"]
    assert sweep("ppo", PROBE="1") == ["train.kl_coef=0.1"]


def _stubs(repo, tmp_path: Path) -> dict[str, str]:
    """accelerate and the GPU python entry points print their argv (and PROBE)
    instead of running; posttrain_lab.config is the real python."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "accelerate").write_text('#!/usr/bin/env bash\necho "ACCELERATE PROBE=${PROBE:-} $*"\n')
    (bin_dir / "python").write_text(
        "#!/usr/bin/env bash\n"
        'case "$*" in\n'
        '  *posttrain_lab.eval.run_eval*|*posttrain_lab.utils.chat*|*posttrain_lab.probe_report*|analysis/*)\n'
        '    echo "PYTHON PROBE=${PROBE:-} $*"; exit 0 ;;\n'
        "esac\n"
        f'exec "{sys.executable}" "$@"\n')
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    return {"PATH": f"{bin_dir}:{repo.env()['PATH']}", "RESULTS_DIR": str(tmp_path / "results")}


def _launches(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.startswith(("ACCELERATE ", "PYTHON ")) and "posttrain_lab" in ln]


def test_the_probe_runs_the_real_pipeline_once_per_arm(repo, tmp_path: Path) -> None:
    env = _stubs(repo, tmp_path)
    old = tmp_path / "results" / "probe" / "helpful" / "runs" / "stale"
    old.mkdir(parents=True)
    run = repo.run("bash", "scripts/run_probe.sh", check=False, **env)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    root = tmp_path / "results" / "probe" / "helpful"
    logs = sorted((root / "logs").glob("*.log"))
    launches = _launches(run.stdout + "\n" + "\n".join(p.read_text() for p in logs))
    modules = [ln.split("--module ")[1].split()[0] if "--module" in ln else ln.split()[3] for ln in launches]
    # pi_SFT, its stop probe, RM, SFT arm, one DPO, one PPO, one GRPO, 4 eval shards, the report (DPO and the
    # eval shards launch into their own logs, so compare as a multiset)
    assert sorted(modules) == sorted(["posttrain_lab.train.sft", "posttrain_lab.utils.chat", "posttrain_lab.train.rm",
                                      "posttrain_lab.train.sft", "posttrain_lab.train.dpo", "posttrain_lab.train.ppo",
                                      "posttrain_lab.train.grpo"] + ["posttrain_lab.eval.run_eval"] * 4
                                     + ["posttrain_lab.probe_report"])
    assert all("PROBE=1 " in ln for ln in launches), launches
    gate = next(ln for ln in launches if "utils.chat" in ln)
    assert gate.endswith("--probe configs/helpful/stage1_sft_base.yaml --report-only")
    assert "train.beta=0.3" in next(ln for ln in launches if "train.dpo" in ln)
    assert not old.exists()                                            # every probe starts empty
    assert not list(repo.root.glob("logs_dpo_*.log")) and not list(repo.root.glob("logs_eval_shard*.log"))
    assert [p.name for p in logs] == ["logs_dpo_train_beta_0_3.log"] + [f"logs_eval_shard{i}.log" for i in range(4)]


def test_the_probe_wont_touch_the_real_runs_logs(repo, tmp_path: Path) -> None:
    (repo.root / "logs_dpo_train_beta_0_1.log").write_text("the real run\n")
    run = repo.run("bash", "scripts/run_probe.sh", check=False, **_stubs(repo, tmp_path))
    assert run.returncode == 1 and "would be overwritten" in run.stderr
    assert (repo.root / "logs_dpo_train_beta_0_1.log").read_text() == "the real run\n"
    assert not _launches(run.stdout)


def test_the_real_launcher_still_gates_and_sweeps_every_value(repo, tmp_path: Path) -> None:
    env = _stubs(repo, tmp_path)
    run = repo.run("bash", "scripts/run_all_helpful.sh", check=False, **env)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    logs = "\n".join(p.read_text() for p in sorted(repo.root.glob("logs_*.log")))
    launches = _launches(run.stdout + "\n" + logs)
    gate = next(ln for ln in launches if "utils.chat" in ln)
    assert gate.endswith("--probe configs/helpful/stage1_sft_base.yaml")   # gates: no --report-only
    assert sum("train.dpo" in ln for ln in launches) == 4 and sum("train.grpo" in ln for ln in launches) == 4
    assert sum("train.ppo" in ln for ln in launches) == 3


def _run(root: Path, name: str, steps: int, runtime: float | None, ckpts: int, done: bool = True) -> None:
    run = root / "runs" / name
    for k in range(1, ckpts + 1):
        (run / f"checkpoint-{2 * k}").mkdir(parents=True)
    run.mkdir(parents=True, exist_ok=True)
    history = [{"step": steps, "loss": 1.0}] + ([{"step": steps, "train_runtime": runtime}] if runtime else [])
    (run / "trainer_state.json").write_text(json.dumps({"global_step": steps, "log_history": history}))
    if done:
        (run / "DONE").touch()


def test_the_report_gives_seconds_per_step_and_peak_memory(tmp_path: Path, capsys) -> None:
    from posttrain_lab import probe_report

    root = tmp_path / "probe" / "helpful"
    _run(root, "ppo_kl0.1_seed17", 4, 200.0, 2)
    _run(root, "grpo_beta0.1_seed17", 4, 60.0, 2)
    (root / "eval" / "ppo_kl0.1_seed17").mkdir(parents=True)
    (root / "eval" / "ppo_kl0.1_seed17" / "checkpoint-2.jsonl").write_text("{}\n")
    (root / "eval" / "ppo_kl0.1_seed17" / "checkpoint-2.samples.jsonl").write_text("{}\n")
    gpu = tmp_path / "gpu_util.csv"
    gpu.write_text("timestamp, index, utilization.gpu [%], memory.used [MiB], power.draw [W]\n"
                   "2026/10/25 03:00:00.000, 0, 99 %, 79000 MiB, 300.00 W\n"      # before the probe
                   "2026/10/25 04:00:00.000, 0, 90 %, 61440 MiB, 250.00 W\n"
                   "2026/10/25 04:00:30.000, 0, 95 %, 71680 MiB, 260.00 W\n"
                   "2026/10/25 04:00:30.000, 1, 95 %, 40960 MiB, 260.00 W\n")
    from datetime import datetime
    since = datetime(2026, 10, 25, 3, 30).timestamp()
    assert probe_report.main([str(root), "--since", str(since), "--gpu-log", str(gpu)]) == 0
    out = capsys.readouterr().out
    assert "ppo_kl0.1_seed17                   yes       4    50.0     2" in out
    assert "grpo_beta0.1_seed17                yes       4    15.0     2" in out
    assert "eval rows written: 1" in out
    assert "GPU 0: 70.0 GiB  GPU 1: 40.0 GiB" in out


def test_the_report_fails_when_a_run_did_not_finish(tmp_path: Path, capsys) -> None:
    from posttrain_lab import probe_report

    root = tmp_path / "probe" / "helpful"
    _run(root, "sft-base_seed17", 4, 30.0, 2)
    _run(root, "ppo_kl0.1_seed17", 1, None, 0, done=False)
    assert probe_report.main([str(root), "--gpu-log", str(tmp_path / "none.csv")]) == 1
    out = capsys.readouterr().out
    assert "ppo_kl0.1_seed17                   NO        1       ?     0" in out
    assert "no logger rows" in out
