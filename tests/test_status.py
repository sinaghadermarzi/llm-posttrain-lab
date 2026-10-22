"""posttrain_lab.status: the parsers it reads logs/processes with, and the
whole report on a fake math run that stopped partway (base SFT and pairs done,
the SFT arm crashed with an OOM, one DPO run done and one failed)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from posttrain_lab import status

# ---------------------------------------------------------------------------
# parsers

TRAIN_LOG = (
    "Map: 100%|██████████| 7473/7473 [00:00<00:00, 40411.12 examples/s]\n"
    " 36%|███▋      | 26/72 [01:40<02:50,  3.93s/it]\r 37%|███▋      | 27/72 [01:44<02:47,  3.92s/it]\n"
    "{'loss': '0.8994', 'grad_norm': '7.873', 'learning_rate': '1.963e-05', 'mean_token_accuracy': '0.8477', 'epoch': '0.4167'}\n"
    "Writing model shards: 100%|█████| 1/1 [00:44<00:00, 44.43s/it]\n"
)


def test_last_progress_picks_the_trainer_bar() -> None:
    bar = status.last_progress(status.log_lines(TRAIN_LOG))
    assert (bar["n"], bar["total"], bar["elapsed_s"], bar["remaining_s"]) == (27, 72, 104, 167)
    assert bar["label"] == ""
    assert status.last_progress(status.log_lines(TRAIN_LOG), "Map")["total"] == 7473


def test_last_progress_vllm_bar() -> None:
    line = "Processed prompts:  40%|████      | 23904/59784 [04:01<06:02, 98.97it/s, est. speed input: 1.2 toks/s]"
    bar = status.last_progress(status.log_lines(line), "Processed prompts")
    assert (bar["n"], bar["total"]) == (23904, 59784)
    assert status.last_progress(status.log_lines(line)) is None  # labelled: not a trainer bar


def test_last_metrics_numeric_and_skips_summary() -> None:
    text = TRAIN_LOG + "{'train_runtime': '396.3', 'train_loss': '0.4106'}\n"
    m = status.last_metrics(status.log_lines(text))
    assert m["loss"] == pytest.approx(0.8994) and m["epoch"] == pytest.approx(0.4167)


def test_errors_ignore_known_warnings() -> None:
    lines = status.log_lines(
        "(EngineCore pid=1) WARNING 10-20 02:03:26 [import_utils.py:408] ImportError: libnvrtc.so.13: cannot open\n"
        "  warnings.warn(\n"
        "[rank2]: Traceback (most recent call last):\n"
        "[rank2]: PermissionError: [Errno 13] Permission denied: 'x' -> 'y'\n"
        "[rank2]: FileNotFoundError: [Errno 2] No such file or directory: 'y'\n"
    )
    errs = status.summarize_errors(lines)
    assert errs["count"] == 3
    assert errs["last_exception"].startswith("[rank2]: FileNotFoundError")
    assert not any("libnvrtc" in line for line in errs["last"])
    assert status.summarize_errors(status.log_lines("all good\n")) is None


def test_last_section() -> None:
    text = "=== math m1: pi_SFT-math\nskip\n=== math m2: self-generated pairs\nsampling\n"
    assert status.last_section(text, "=== math ") == ("=== math m2: self-generated pairs", "sampling\n")
    assert status.last_section("no headers") == (None, "")


def test_last_section_of_a_both_tracks_log() -> None:
    """run_all.sh: the helpful track's section ends where the math track starts."""
    text = "=== stage 5: aggregate + figures\nALL DONE\n=== math m1: pi_SFT-math\nTraceback (most recent call last):\n"
    assert status.last_section(text, "=== stage ") == ("=== stage 5: aggregate + figures", "ALL DONE")
    assert status.last_section(text, "=== math ")[0] == "=== math m1: pi_SFT-math"
    assert status.last_section("=== math m1: x\n", "=== stage ") == (None, "")


def test_header_names() -> None:
    math_plan, helpful_plan = status.launcher_plan("math", False), status.launcher_plan("helpful", False)
    stage = status.TRACKS["math"]["stages"][2]
    assert status.header_names("=== math m3a: SFT arm", stage, math_plan, False)
    assert not status.header_names("=== math m3b: DPO sweep", stage, math_plan, False)
    assert status.header_names("=== stage 3a: SFT arm", status.TRACKS["helpful"]["stages"][2], helpful_plan, False)
    smoke = status.launcher_plan("math", True)
    m1 = status.TRACKS["math"]["stages"][0]
    assert status.header_names("=== smoke math 1/6: SFT base on gold CoT (pi_SFT-math)", m1, smoke, True)
    assert not status.header_names("=== smoke math 2/6: self-generated preference pairs", m1, smoke, True)


def test_launcher_plans_follow_the_scripts() -> None:
    smoke = status.launcher_plan("math", True)
    assert [(yaml, ovs) for _, yaml, ovs in smoke["steps"]] == [
        ("math/m1_sft_gold.yaml", []), ("math/m2_gen_pairs.yaml", []),
        ("math/m3_arm_dpo.yaml", ["train.beta=0.1"]), ("math/m3_arm_dpo_nll.yaml", ["train.beta=0.1"]),
        ("math/m3_arm_grpo.yaml", ["train.beta=0.04"])]
    helpful = status.launcher_plan("helpful", True)
    assert [ovs for _, _, ovs in helpful["steps"]] == [[], [], [], ["train.beta=0.1"], ["train.kl_coef=0.05"],
                                                        ["train.beta=0.04"]]
    assert status.launcher_plan("math", False)["analysis"] == [
        "aggregate.py", "frontier_math.py", "diversity_plot.py", "retention_plot.py", "pref_logps_plot.py"]
    assert "diagnostics.py" in status.launcher_plan("helpful", False)["analysis"]
    assert set(status.launcher_plan("helpful", True)["analysis"]) == set(status.launcher_plan("helpful", False)["analysis"])


def test_log_name_matches_the_launcher() -> None:
    shell = subprocess.run(["bash", "-c", 'ov=train.beta=0.01; echo "logs_math_dpo_${ov//[.=\\/]/_}.log"'],
                           capture_output=True, text=True).stdout.strip()
    assert status.log_name("logs_math_dpo_{ov}.log", "train.beta=0.01") == shell


def test_parse_clock() -> None:
    assert status.parse_clock("05:10") == 310
    assert status.parse_clock("1:02:03") == 3723
    assert status.parse_clock("2-01:00:00") == 2 * 86400 + 3600
    assert status.parse_clock("?") is None


def test_find_jobs_groups_launcher_and_ranks(monkeypatch) -> None:
    tail = "configs/math/m3_arm_dpo.yaml train.beta=0.1"
    procs = [
        {"pid": 10, "ppid": 1, "elapsed_s": 300, "cpu": 1.0,
         "args": f"/v/bin/python /v/bin/accelerate launch --config_file configs/accelerate/zero2_1gpu.yaml "
                 f"--main_process_port 29502 --module posttrain_lab.train.dpo {tail}"},
        {"pid": 11, "ppid": 10, "elapsed_s": 290, "cpu": 99.0, "args": f"/v/bin/python3 -u -m posttrain_lab.train.dpo {tail}"},
        {"pid": 20, "ppid": 1, "elapsed_s": 50, "cpu": 50.0,
         "args": "/v/bin/python -m posttrain_lab.eval.run_eval task=math --shard 1/4"},
        {"pid": 30, "ppid": 1, "elapsed_s": 5, "cpu": 0.0, "args": "vim notes.txt"},
    ]
    monkeypatch.setattr(status, "in_this_repo", lambda pid: True)
    jobs = {j.module: j for j in status.find_jobs(procs)}
    assert set(jobs) == {"train.dpo", "eval.run_eval"}
    dpo = jobs["train.dpo"]
    assert (dpo.ranks, dpo.elapsed_s, dpo.yaml, dpo.overrides) == (1, 300, "configs/math/m3_arm_dpo.yaml", ["train.beta=0.1"])
    assert jobs["eval.run_eval"].shard == "1/4"


def test_job_output_dir_uses_the_jobs_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    monkeypatch.delenv("SMOKE", raising=False)
    job = status.Job("train.dpo", ["configs/math/m3_arm_dpo.yaml", "train.beta=0.3"])
    assert status.job_output_dir(job) == str(tmp_path / "math" / "runs" / "dpo_beta0.3_seed17")


def test_gpu_history(tmp_path: Path) -> None:
    csv_path = tmp_path / "gpu_util.csv"
    rows = ["timestamp, index, utilization.gpu [%], memory.used [MiB], power.draw [W]"]
    for minute in range(10):
        for gpu, util in ((0, 90), (1, 0)):
            rows.append(f"2026/10/20 02:{minute:02d}:00.000, {gpu}, {util} %, 15000 MiB, 300.00 W")
    csv_path.write_text("\n".join(rows) + "\n")
    hist = status.gpu_history(csv_path)
    assert hist["recent"]["per_gpu"] == {0: 90.0, 1: 0.0}
    assert hist["recent"]["idle_frac"] == pytest.approx(0.5)


def test_expected_checkpoints() -> None:
    from omegaconf import OmegaConf

    assert status.expected_checkpoints(OmegaConf.create({"train": {"save_strategy": "epoch", "num_train_epochs": 3}})) == 3
    assert status.expected_checkpoints(OmegaConf.create({"train": {"save_steps": 0.2}})) == 5
    assert status.expected_checkpoints(OmegaConf.create({"train": {"save_steps": 0.111}})) == 9
    assert status.expected_checkpoints(OmegaConf.create({"train": {"save_steps": 5, "max_steps": 10}})) == 2


# ---------------------------------------------------------------------------
# a whole report on a fake, stopped math run


def _ckpt(run: Path, step: int, max_steps: int, **logged) -> None:
    d = run / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"x" * 1000)
    history = [{"step": step, "epoch": step / max_steps, **logged}]
    (d / "trainer_state.json").write_text(json.dumps({"global_step": step, "max_steps": max_steps,
                                                       "epoch": step / max_steps, "log_history": history}))


def _fake_run(repo, tmp_path: Path) -> tuple[Path, dict]:
    results = tmp_path / "results"
    math = results / "math"
    base = math / "runs" / "sft-base_seed17"
    for step in (24, 48, 72):
        _ckpt(base, step, 72, loss=0.25, mean_token_accuracy=0.93)
    (base / "DONE").touch()

    pairs = math / "data" / "self_pairs_seed17"
    pairs.mkdir(parents=True)
    (pairs / "stats.json").write_text(json.dumps({
        "prompts": 512, "samples": 4096, "sample_solve_rate": 0.57, "pairs": 701,
        "prompts_all_correct": 72, "prompts_all_wrong": 36, "train_pairs": 666, "test_pairs": 35}))
    (pairs / "DONE").touch()

    _ckpt(math / "runs" / "sft_seed17", 39, 351, loss=0.31)          # crashed: no DONE

    good = math / "runs" / "dpo_beta0.1_seed17"
    _ckpt(good, 10, 10, loss=0.6, **{"rewards/accuracies": 0.7})
    (good / "DONE").touch()
    _ckpt(math / "runs" / "dpo_beta0.01_seed17", 2, 10, loss=0.69)   # failed, own log
    (repo.root / "logs_math_dpo_train_beta_0_01.log").write_text(
        " 20%|██        | 2/10 [00:30<02:00,  15.0s/it]\nTraceback (most recent call last):\nValueError: boom\n")

    (repo.root / "math-all.log").write_text(
        "=== math m1: pi_SFT-math (gold CoT)\n[sft] already DONE, skipping\n"
        "=== math m2: self-generated preference pairs\n[gen_pairs] already DONE, skipping\n"
        "=== math m3a: SFT arm\n"
        "(EngineCore pid=1) WARNING [import_utils.py:408] ImportError: libnvrtc.so.13: cannot open shared object file\n"
        " 11%|█         | 39/351 [01:40<13:20,  2.56s/it]\n"
        "{'loss': '0.31', 'grad_norm': '2.1', 'mean_token_accuracy': '0.92', 'epoch': '0.33'}\n"
        "[rank0]: Traceback (most recent call last):\n"
        "[rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n")

    evals = math / "eval" / "sft-base_seed17"
    evals.mkdir(parents=True)
    (evals / "checkpoint-72.jsonl").write_text(json.dumps({"kl_k3": 0.0, "pass1": 0.41, "ppl_wikitext": 9.1}) + "\n")

    info = tmp_path / "hf" / "datasets" / "openai___gsm8k" / "main" / "0.0.0" / "abc123"
    info.mkdir(parents=True)
    (info / "dataset_info.json").write_text(json.dumps({"splits": {"train": {"num_examples": 7473}}}))
    env = {"RESULTS_DIR": str(results), "HF_HOME": str(tmp_path / "hf")}
    return math, env


def test_report_on_a_stopped_math_run(repo, tmp_path: Path) -> None:
    _, env = _fake_run(repo, tmp_path)
    run = repo.python("-m", "posttrain_lab.status", "math", "--json", check=False, **env)
    assert run.returncode == 0, run.stderr[-3000:]
    report = json.loads(run.stdout)
    (track,) = report["tracks"]
    stages = {s["key"]: s for s in track["stages"]}
    states = {k: s["state"] for k, s in stages.items()}
    assert states == {"m1": "done", "m2": "done", "m3a": "failed", "m3b": "failed", "m3c": "pending",
                      "m3d": "pending", "m4": "stopped", "m5": "pending"}

    (base,) = stages["m1"]["runs"]
    assert (base["checkpoints"], base["expected_checkpoints"], base["step"]) == (3, 3, 72)
    assert base["metrics"]["loss"] == 0.25

    (arm,) = stages["m3a"]["runs"]
    assert "OutOfMemoryError" in arm["errors"]["last_exception"]
    assert arm["log"].endswith("math-all.log")

    dpo = {r["override"]: r for r in stages["m3b"]["runs"]}
    assert dpo["train.beta=0.1"]["state"] == "done"
    assert dpo["train.beta=0.01"]["state"] == "failed"
    assert dpo["train.beta=0.01"]["errors"]["last_exception"] == "ValueError: boom"
    assert dpo["train.beta=0.3"]["state"] == "pending"

    ev = stages["m4"]["eval"]
    assert ev["checkpoints"] == 3 + 1 + 1 + 1  # sft-base 3, sft arm 1, two dpo runs 1 each
    assert ev["per_group"]["core_math"] == 1 and ev["per_group"]["retention_ppl"] == 1

    attention = "\n".join(report["attention"])
    assert "sampled 512 of 7473 train prompts" in attention
    assert "only 666 DPO training pairs" in attention
    assert "m3a sft_seed17 FAILED" in attention and "OutOfMemoryError" in attention
    assert "pipeline is not running and not finished" in attention


def test_text_report(repo, tmp_path: Path) -> None:
    _, env = _fake_run(repo, tmp_path)
    run = repo.python("-m", "posttrain_lab.status", "math", "--no-color", check=False, **env)
    assert run.returncode == 0, run.stderr[-3000:]
    out = run.stdout
    assert "MATH track   pipeline NOT running" in out
    assert "✓ DONE     m1" in out and "✗ FAILED   m3a" in out
    assert "ckpts 3/3" in out and "loss=0.25" in out
    assert "ValueError: boom" in out
    assert "attention (" in out
    assert "\033[" not in out


def test_smoke_flag_reads_the_smoke_tree(repo, tmp_path: Path) -> None:
    run = repo.python("-m", "posttrain_lab.status", "--smoke", "--json", check=False, RESULTS_DIR=str(tmp_path))
    assert run.returncode == 0, run.stderr[-3000:]
    roots = [t["results_root"] for t in json.loads(run.stdout)["tracks"]]
    assert roots == [str(tmp_path / "smoke" / "math"), str(tmp_path / "smoke" / "helpful")]


def test_running_job_is_detected(repo, tmp_path: Path) -> None:
    """A live process with a DPO job's command line shows as RUNNING with the
    progress and metrics from its sweep log."""
    import sys

    _, env = _fake_run(repo, tmp_path)
    (repo.root / "logs_math_dpo_train_beta_0_3.log").write_text(
        " 40%|████      | 4/10 [01:00<01:30,  15.0s/it]\n"
        "{'loss': '0.65', 'rewards/accuracies': '0.62', 'rewards/margins': '0.11', 'epoch': '0.4'}\n")
    fake = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)",
                             "-m", "posttrain_lab.train.dpo", "configs/math/m3_arm_dpo.yaml", "train.beta=0.3"],
                            cwd=repo.root)  # launchers run jobs from the repo root
    try:
        run = repo.python("-m", "posttrain_lab.status", "math", "--json", check=False, **env)
    finally:
        fake.kill()
        fake.wait()
    assert run.returncode == 0, run.stderr[-3000:]
    stages = {s["key"]: s for s in json.loads(run.stdout)["tracks"][0]["stages"]}
    dpo = {r["override"]: r for r in stages["m3b"]["runs"]}["train.beta=0.3"]
    assert dpo["state"] == "running"
    assert (dpo["progress"]["n"], dpo["progress"]["total"]) == (4, 10)
    assert dpo["metrics"]["rewards/accuracies"] == pytest.approx(0.62)
    assert stages["m3b"]["state"] == "running"


def test_sentinels_are_run_evals() -> None:
    """status reads eval coverage with the same sentinel keys run_eval computes by."""
    from posttrain_lab.eval import run_eval
    from posttrain_lab.eval.groups import SENTINELS

    assert {name: keys for name, (keys, _) in run_eval.GROUPS.items()} == SENTINELS


def test_only_this_repos_processes_count(repo, tmp_path: Path) -> None:
    """A driver or job running from another checkout (or a command line that
    only mentions a script) is not this repo's pipeline."""
    import sys

    other = tmp_path / "elsewhere"
    other.mkdir()
    procs = [subprocess.Popen(["bash", "-c", "sleep 120", "scripts/run_all_math.sh"], cwd=repo.root),  # not `bash script`
             subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "-m", "posttrain_lab.train.dpo",
                               "configs/math/m3_arm_dpo.yaml", "train.beta=0.3"], cwd=other)]
    try:
        run = repo.python("-m", "posttrain_lab.status", "math", "--json", check=False, RESULTS_DIR=str(tmp_path / "r"))
    finally:
        for proc in procs:
            proc.kill()
            proc.wait()
    assert run.returncode == 0, run.stderr[-3000:]
    report = json.loads(run.stdout)
    assert report["tracks"][0]["driver"] is None
    assert report["other_jobs"] == []
    dpo = {r["override"]: r for s in report["tracks"][0]["stages"] if s["key"] == "m3b" for r in s["runs"]}
    assert dpo["train.beta=0.3"]["state"] == "pending"


def _touch(path: Path, mtime: float, text: str = "x") -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, (mtime, mtime))


MATH_ROW = {"kl_k3": 0.1, "pass1": 0.3, "ppl_wikitext": 9.0, "harness": {}, "distinct_2": 0.5, "self_bleu": 0.2,
            "logp_chosen": -50.0}
MATH_OUTPUTS = ["figures/frontier_math.png", "figures/frontier_math.pdf", "figures/diversity.png",
                "figures/retention.png", "figures/pref_logps.png"]


def test_finished_smoke_run_reads_done(repo, tmp_path: Path) -> None:
    """--smoke expects exactly what run_smoke.sh runs (one value per sweep, no
    math SFT arm), so a finished smoke run is all DONE with nothing flagged."""
    import time

    math = tmp_path / "results" / "smoke" / "math"
    t0 = time.time() - 100
    for run in ("sft-base_seed17", "dpo_beta0.1_seed17", "dpo-nll_beta0.1_seed17", "grpo_beta0.04_seed17"):
        for step in (5, 10):
            _ckpt(math / "runs" / run, step, 10, loss=0.5)
            _touch(math / "eval" / run / f"checkpoint-{step}.jsonl", t0, json.dumps(MATH_ROW) + "\n")
        (math / "runs" / run / "DONE").touch()
    pairs = math / "data" / "self_pairs_seed17"
    pairs.mkdir(parents=True)
    (pairs / "stats.json").write_text(json.dumps({"prompts": 100, "samples": 200, "sample_solve_rate": 0.065,
                                                  "pairs": 13, "train_pairs": 12, "test_pairs": 1}))
    (pairs / "DONE").touch()
    _touch(math / "frontier.csv", t0 + 10)
    for out in MATH_OUTPUTS:
        _touch(math / out, t0 + 20)

    run = repo.python("-m", "posttrain_lab.status", "math", "--smoke", "--json", check=False,
                      RESULTS_DIR=str(tmp_path / "results"))
    assert run.returncode == 0, run.stderr[-3000:]
    report = json.loads(run.stdout)
    stages = report["tracks"][0]["stages"]
    assert [s["key"] for s in stages] == ["m1", "m2", "m3b", "m3c", "m3d", "m4", "m5"]  # no m3a in smoke
    assert {s["key"]: [r["override"] for r in s["runs"]] for s in stages if s["runs"]} == {
        "m1": [None], "m2": [None], "m3b": ["train.beta=0.1"], "m3c": ["train.beta=0.1"], "m3d": ["train.beta=0.04"]}
    assert all(s["state"] == "done" for s in stages), [(s["key"], s["state"]) for s in stages]
    assert report["attention"] == []


def test_figures_need_every_output_newer_than_its_inputs(tmp_path: Path) -> None:
    import time

    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"results_root": str(tmp_path)})
    analysis = status.launcher_plan("math", False)["analysis"]
    t0 = time.time() - 100
    _touch(tmp_path / "eval" / "sft-base_seed17" / "checkpoint-72.jsonl", t0)
    _touch(tmp_path / "frontier.csv", t0 + 10)
    _touch(tmp_path / "figures" / "frontier_math.png", t0 + 20)  # one plot made, the rest failed
    fig = status.figures_state(cfg, analysis)
    assert "figures/retention.png" in fig["missing"] and fig["stale"] == []

    for out in MATH_OUTPUTS:
        _touch(tmp_path / out, t0 + 20)
    assert status.figures_state(cfg, analysis)["missing"] == []
    _touch(tmp_path / "figures" / "diversity.png", t0 + 5)  # left over from an earlier aggregate
    assert status.figures_state(cfg, analysis)["stale"] == ["figures/diversity.png"]
    _touch(tmp_path / "figures" / "diversity.png", t0 + 20)
    _touch(tmp_path / "eval" / "sft_seed17" / "checkpoint-9.jsonl", t0 + 30)  # evaluated after the aggregate
    assert status.figures_state(cfg, analysis)["stale"] == ["frontier.csv"]


def test_figures_stage_not_done_with_a_missing_plot(repo, tmp_path: Path) -> None:
    import time

    _, env = _fake_run(repo, tmp_path)
    t0 = time.time() + 10
    math = tmp_path / "results" / "math"
    _touch(math / "frontier.csv", t0)
    _touch(math / "figures" / "frontier_math.png", t0 + 10)
    run = repo.python("-m", "posttrain_lab.status", "math", "--json", check=False, **env)
    report = json.loads(run.stdout)
    m5 = [s for s in report["tracks"][0]["stages"] if s["key"] == "m5"][0]
    assert m5["state"] == "stopped"
    assert any("m5 figures incomplete: missing" in a and "retention.png" in a for a in report["attention"])


def test_log_flag_applies_with_all_tracks(repo, tmp_path: Path) -> None:
    """The default task is all; --log must still reach the track whose headers
    the log carries, and only that track."""
    _, env = _fake_run(repo, tmp_path)
    (repo.root / "math-all.log").rename(repo.root / "custom.log")
    run = repo.python("-m", "posttrain_lab.status", "--log", str(repo.root / "custom.log"), "--json", check=False, **env)
    assert run.returncode == 0, run.stderr[-3000:]
    math, helpful = json.loads(run.stdout)["tracks"]
    assert math["driver_log"].endswith("custom.log") and math["log_stage"] == "=== math m3a: SFT arm"
    arm = [s for s in math["stages"] if s["key"] == "m3a"][0]["runs"][0]
    assert arm["state"] == "failed" and "OutOfMemoryError" in arm["errors"]["last_exception"]
    assert helpful["driver_log"] is None


# ---------------------------------------------------------------------------
# generation that never stops (the first full math run)


def test_health_flags_generation_that_never_stops() -> None:
    run = status.Run(name="grpo_beta0.04_seed17", override=None, output_dir="x", step=934, max_steps=934,
                     metrics={"completions/clipped_ratio": 1.0, "completions/mean_length": 320.0})
    status.health(run, "grpo")
    assert run.warnings == ["100% of completions hit max_completion_length: generation is not stopping at end of "
                            "turn (check: python -m posttrain_lab.utils.chat RUN_DIR)"]
    run = status.Run(name="g", override=None, output_dir="x", metrics={"completions/clipped_ratio": 0.6})
    status.health(run, "grpo")
    assert run.warnings == ["60% of completions hit max_completion_length"]


def test_health_flags_a_finished_dpo_run_that_did_not_learn() -> None:
    """Finished runs have no progress bar; the check uses trainer_state's step."""
    run = status.Run(name="dpo_beta0.1_seed17", override=None, output_dir="x", step=147, max_steps=147,
                     metrics={"loss": 0.6936, "rewards/accuracies": 0.30})
    status.health(run, "dpo")
    text = " | ".join(run.warnings)
    assert "rewards/accuracies 0.3 at 100%" in text and "still ~ln 2" in text


def test_stop_config_problem(tmp_path: Path) -> None:
    template = "{% for m in messages %}<|im_start|>{{ m['content'] }}<|im_end|>{% endfor %}"
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|endoftext|>", "chat_template": template}))
    assert "never appears in its chat template" in status.stop_config_problem(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|im_end|>"}))
    (tmp_path / "chat_template.jinja").write_text(template)   # transformers 5 saves it next to the config
    assert status.stop_config_problem(tmp_path) is None


def test_unfinished_share(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint-5.samples.jsonl"
    path.write_text("".join(json.dumps({"response": "r", "finished": i < 1}) + "\n" for i in range(4)))
    assert status.unfinished_share(path) == pytest.approx(0.75)
    path.write_text(json.dumps({"response": "r"}) + "\n")   # written before generation recorded it
    assert status.unfinished_share(path) is None


def test_report_flags_runs_that_cannot_stop(repo, tmp_path: Path) -> None:
    _, env = _fake_run(repo, tmp_path)
    base = tmp_path / "results" / "math" / "runs" / "sft-base_seed17"
    (base / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|endoftext|>",
                                                            "chat_template": "<|im_start|>{{ x }}<|im_end|>"}))
    samples = tmp_path / "results" / "math" / "eval" / "sft-base_seed17" / "checkpoint-72.samples.jsonl"
    samples.write_text("".join(json.dumps({"response": "r", "finished": False}) + "\n" for _ in range(8)))
    run = repo.python("-m", "posttrain_lab.status", "math", "--json", check=False, **env)
    assert run.returncode == 0, run.stderr[-3000:]
    attention = "\n".join(json.loads(run.stdout)["attention"])
    assert "sft-base_seed17: saved tokenizer's eos '<|endoftext|>' never appears in its chat template" in attention
    assert "most samples of sft-base_seed17 hit max_new_tokens" in attention


def test_report_shows_the_stop_probe(repo, tmp_path: Path) -> None:
    """Base SFT without a stop probe, with a failing one, and with a passing one."""
    _, env = _fake_run(repo, tmp_path)
    base = tmp_path / "results" / "math" / "runs" / "sft-base_seed17"

    def report() -> tuple[str, str]:
        run = repo.python("-m", "posttrain_lab.status", "math", "--json", check=False, **env)
        assert run.returncode == 0, run.stderr[-3000:]
        rep = json.loads(run.stdout)
        (m1,) = [s for s in rep["tracks"][0]["stages"] if s["key"] == "m1"]
        return "\n".join(rep["attention"]), "\n".join(m1["runs"][0]["notes"])

    attention, _ = report()
    assert "sft-base_seed17: no stop_probe.json yet" in attention
    assert "--probe configs/math/m1_sft_gold.yaml" in attention

    probe = {"token": "<|im_end|>", "n": 64, "mean": 0.12, "min": 0.12, "max": 0.13, "mean_any_stop": 0.2,
             "min_p": 0.8}
    (base / "stop_probe.json").write_text(json.dumps(probe))
    attention, notes = report()
    assert "P(end of turn) is only 0.12 where answers end (GRPO's only clean stop); about 80% of samples" in attention
    assert "stops: P(<|im_end|>) where 64 training answers end: mean 0.120" in notes

    (base / "stop_probe.json").write_text(json.dumps({**probe, "mean": 0.97, "min": 0.9}))
    attention, notes = report()
    assert "sft-base_seed17: " not in attention and "mean 0.970" in notes
