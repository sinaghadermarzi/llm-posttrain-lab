"""The evidence layer: make evidence freezes a results tree into evidence/<task>/<date>_<name>/
(figures, tables, configs, pre-registered comparisons, examples, a manifest), never overwrites or
bloats a snapshot, and keeps the index in evidence/README.md; and the agent docs point at it."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
T0 = time.time() - 10_000


def _expected_outputs() -> list[str]:
    from posttrain_lab import status

    return [o for script in status.launcher_plan("math", False)["analysis"] for o in status.ANALYSIS_OUTPUTS[script]]


EXPECTED = _expected_outputs()
FIGURES = sorted(Path(o).name for o in EXPECTED if o.startswith("figures/"))
RUNS = {"sft-base": [24, 72], "dpo_beta0.1": [10, 20], "grpo_beta0.04": [100, 200]}
SPEC = """
compare:
  - --seed 17 dpo_beta0.1@ckpt2 grpo_beta0.04@last
  - --pool dpo_beta0.1@ckpt2 grpo_beta0.04@last
  - --seed 99 dpo_beta0.1@last grpo_beta0.04@last
examples:
  seed: 17
  prompts: 3
  checkpoints: [sft-base@last, dpo_beta0.1@last, grpo_beta0.04@last]
"""


def _tree(root: Path) -> None:
    """A two-seed math results tree: eval rows + samples, trainer states, pair stats, a probe, figures."""
    for seed in (17, 18):
        for run, steps in RUNS.items():
            name = f"{run}_seed{seed}"
            arm, hp = run.split("_")[0], (run.split("_")[1] if "_" in run else None)
            for i, step in enumerate(steps):
                step += seed - 17 if arm == "dpo" else 0           # DPO steps differ per seed
                ev = root / "eval" / name
                ev.mkdir(parents=True, exist_ok=True)
                kl = 0.0 if arm == "sft-base" else 1.0 + i
                row = {"run": name, "arm": arm, "hyperparam": hp, "checkpoint": f"x/{name}/checkpoint-{step}",
                       "step": step, "kl_k1": kl, "kl_k3": kl, "pass1": 0.4 + 0.1 * i, "pass1_se": 0.01,
                       "pass8": 0.8, "pass8_se": 0.01, "num_samples": 40, "eval_prompts": 20}
                (ev / f"checkpoint-{step}.jsonl").write_text(json.dumps(row) + "\n")
                with (ev / f"checkpoint-{step}.samples.jsonl").open("w") as f:
                    for p in range(20):
                        for j in range(2):
                            correct = (p + i + (arm == "grpo")) % 3 != 0
                            f.write(json.dumps({"prompt_id": f"gsm8k-{p}", "prompt": f"question {p}?", "sample_idx": j,
                                                "response": f"{arm} answer {p}.{j} #### {p}", "finished": True,
                                                "correct": correct}) + "\n")
                ck = root / "runs" / name / f"checkpoint-{step}"
                ck.mkdir(parents=True, exist_ok=True)
                history = [{"step": s, "epoch": s / steps[-1], "loss": 0.5 - 0.001 * s} for s in range(1, step + 1, 5)]
                state = {"global_step": step, "max_steps": steps[-1], "log_history": history}
                (ck / "trainer_state.json").write_text(json.dumps(state))
                t = T0 + 100 * i                                     # checkpoints written 100 s apart
                os.utime(ck / "trainer_state.json", (t, t))
            run_dir = root / "runs" / name
            if arm == "grpo":   # trained after save_final_state: the full log next to the model
                state["log_history"] = history + [{"step": state["global_step"], "train_runtime": 123.4 * seed}]
                (run_dir / "trainer_state.json").write_text(json.dumps(state))
                (run_dir / "posttrain_config.yaml").write_text(f"seed: {seed}\ntrain:\n  beta: 0.04\n")
            (run_dir / "DONE").touch()
        stats = root / "data" / f"self_pairs_seed{seed}"
        stats.mkdir(parents=True, exist_ok=True)
        (stats / "stats.json").write_text(json.dumps({"pairs": 9000 + seed}))
        (root / "runs" / f"sft-base_seed{seed}" / "stop_probe.json").write_text(json.dumps({"mean": 0.98}))
    out = subprocess.run([sys.executable, str(ROOT / "analysis" / "aggregate.py"), str(root)], capture_output=True,
                         text=True, cwd=ROOT / "analysis")
    assert out.returncode == 0, out.stderr
    for out in EXPECTED:                                    # every output the launcher's plots make
        path = root / out
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"fake " + out.encode())


@pytest.fixture
def results(repo, tmp_path: Path) -> Path:
    root = tmp_path / "results" / "math"
    _tree(root)
    (repo.root / "evidence" / "math").mkdir(parents=True)
    (repo.root / "evidence" / "math" / "snapshot.yaml").write_text(SPEC)
    return root


def _export(repo, root: Path, name: str = "seeds17-18", *args: str):
    return repo.python("analysis/export_evidence.py", "math", name, "--root", str(root), "--date", "2026-10-22", *args,
                       check=False, RESULTS_DIR=str(root.parent))


def test_snapshot_holds_figures_tables_configs_and_a_manifest(repo, results: Path) -> None:
    run = _export(repo, results, "seeds17-18", "--note", "two seeds")
    assert run.returncode == 0, run.stdout + run.stderr
    snap = repo.root / "evidence" / "math" / "2026-10-22_seeds17-18"
    assert sorted(p.name for p in (snap / "figures").iterdir()) == FIGURES
    tables = {p.name for p in (snap / "tables").iterdir()}
    assert {"frontier.csv", "frontier_seeds.csv", "training_curves.csv", "pairs_self_pairs_seed17.json",
            "stop_probe_sft-base_seed18.json", "compare_1.txt", "compare_2.txt", "compare_3.txt"} <= tables
    assert (snap / "configs" / "math" / "sweeps.yaml").exists() and (snap / "configs" / "base.yaml").exists()
    assert not list(snap.rglob("*.samples.jsonl")) and not list(snap.rglob("checkpoint-*"))

    m = json.loads((snap / "MANIFEST.json").read_text())
    assert m["snapshot"] == "math/2026-10-22_seeds17-18" and m["seeds"] == [17, 18] and m["note"] == "two seeds"
    assert m["runs"]["dpo_beta0.1_seed18"]["global_step"] == 21 and m["runs"]["dpo_beta0.1_seed18"]["done"]
    assert "trl" in m["packages"] and m["git"]["commit"] is not None
    # every file but the manifest itself, and the sizes are those on disk
    on_disk = {str(p.relative_to(snap)): p.stat().st_size for p in snap.rglob("*") if p.is_file()}
    assert "MANIFEST.json" not in m["files"] and m["files"] == {k: v for k, v in on_disk.items() if k != "MANIFEST.json"}
    assert m["total_bytes"] == sum(m["files"].values())

    curves = (snap / "tables" / "training_curves.csv").read_text().splitlines()
    assert curves[0] == "run,arm,seed,step,epoch,metric,value" and any(",loss," in ln for ln in curves[1:])


def test_eval_settings_come_from_the_rows_not_just_the_config(repo, results: Path) -> None:
    """The rows were sampled on 20 prompts x 2; the config now says 1319 x 8 (a smoke tree, an override,
    a later change): the manifest records both and says they differ."""
    run = _export(repo, results)
    assert run.returncode == 0, run.stdout + run.stderr
    m = json.loads((repo.root / "evidence" / "math" / "2026-10-22_seeds17-18" / "MANIFEST.json").read_text())
    assert m["eval"]["measured"] == {"num_prompts": [20], "samples_per_prompt": [2], "rows": 12}
    assert (m["eval"]["configured"]["num_prompts"], m["eval"]["configured"]["samples_per_prompt"]) == (1319, 8)
    assert m["eval"]["matches_config"] is False and "eval rows were sampled on [20] prompts" in run.stderr
    assert "| 17, 18 | 20 |" in (repo.root / "evidence" / "README.md").read_text()   # the index shows the measured


def test_as_run_configs_and_training_time(repo, results: Path) -> None:
    assert _export(repo, results).returncode == 0
    snap = repo.root / "evidence" / "math" / "2026-10-22_seeds17-18"
    m = json.loads((snap / "MANIFEST.json").read_text())
    assert m["configs"]["as_run"] == ["grpo_beta0.04_seed17", "grpo_beta0.04_seed18"]
    assert "beta: 0.04" in (snap / "configs" / "as_run" / "grpo_beta0.04_seed18.yaml").read_text()
    grpo = m["runs"]["grpo_beta0.04_seed18"]                 # saved by save_final_state: the trainer's own total
    assert (grpo["train_runtime_s"], grpo["runtime_source"]) == (pytest.approx(123.4 * 18), "trainer_state")
    dpo = m["runs"]["dpo_beta0.1_seed18"]                    # trained before: checkpoints 11 and 21, 100 s apart
    assert dpo["runtime_source"] == "estimate: checkpoint mtimes"
    assert dpo["train_runtime_s"] == pytest.approx(100 * 21 / (21 - 11), abs=0.2)

def test_comparisons_come_from_snapshot_yaml_and_failures_are_recorded(repo, results: Path) -> None:
    run = _export(repo, results)
    assert run.returncode == 0, run.stdout + run.stderr
    snap = repo.root / "evidence" / "math" / "2026-10-22_seeds17-18"
    m = json.loads((snap / "MANIFEST.json").read_text())
    assert [c["status"] for c in m["compare"]] == ["ok", "ok", "failed"]           # seed 99 isn't there
    one = (snap / "tables" / "compare_1.txt").read_text()
    assert one.startswith("$ python analysis/compare_math.py <root> --seed 17 dpo_beta0.1@ckpt2 grpo_beta0.04@last")
    assert "paired vs dpo_beta0.1@ckpt2" in one
    assert "pooled over seeds 17, 18" in (snap / "tables" / "compare_2.txt").read_text()
    assert "[not run]" in (snap / "tables" / "compare_3.txt").read_text() and "compare line 3 did not run" in run.stderr


def test_examples_show_the_same_questions_from_each_checkpoint(repo, results: Path) -> None:
    assert _export(repo, results).returncode == 0
    snap = repo.root / "evidence" / "math" / "2026-10-22_seeds17-18"
    md = (snap / "examples" / "examples.md").read_text()
    assert md.count("## Q") == 3 and "seed 17" in md
    for ck in ("sft-base@last", "dpo_beta0.1@last", "grpo_beta0.04@last"):
        assert md.count(f"**{ck}**") == 3
    rows = [json.loads(ln) for ln in (snap / "examples" / "examples.jsonl").read_text().splitlines()]
    assert len(rows) == 9 and len({r["question"] for r in rows}) == 3
    by_q = {}
    for r in rows:
        by_q.setdefault(r["question"], set()).add(r["correct"])
    assert all(len(v) == 2 for v in by_q.values())                                # disagreements first
    m = json.loads((snap / "MANIFEST.json").read_text())
    assert m["examples"] == {"seed": 17, "checkpoints": ["sft-base@last", "dpo_beta0.1@last", "grpo_beta0.04@last"],
                             "prompts": 3}


def test_snapshots_are_never_overwritten_and_the_index_marks_superseded(repo, results: Path) -> None:
    assert _export(repo, results, "first").returncode == 0
    again = _export(repo, results, "first")
    assert again.returncode != 0 and "never overwritten" in again.stderr
    bad = _export(repo, results, "second", "--supersedes", "math/nope")
    assert bad.returncode != 0 and "no such snapshot" in bad.stderr
    run = _export(repo, results, "second", "--supersedes", "math/2026-10-22_first")
    assert run.returncode == 0, run.stdout + run.stderr
    readme = (repo.root / "evidence" / "README.md").read_text()
    row = next(ln for ln in readme.splitlines() if ln.startswith("| [math/2026-10-22_first]"))
    assert row.rstrip().endswith("| math/2026-10-22_second |")
    assert "| [math/2026-10-22_second](math/2026-10-22_second/) | 2026-10-22 | 17, 18 | 20 |" in readme


def test_the_index_keeps_the_rest_of_the_readme(repo, results: Path) -> None:
    readme = repo.root / "evidence" / "README.md"
    readme.write_text("# Evidence\n\nRules here.\n\n## Snapshots\n\n<!-- snapshots:begin -->\nold\n"
                      "<!-- snapshots:end -->\n\nFooter.\n")
    assert _export(repo, results).returncode == 0
    text = readme.read_text()
    assert text.startswith("# Evidence\n\nRules here.") and text.endswith("<!-- snapshots:end -->\n\nFooter.\n")
    assert "old" not in text and "2026-10-22_seeds17-18" in text


def test_stale_figures_and_oversized_snapshots_are_refused(repo, results: Path) -> None:
    row = results / "eval" / "dpo_beta0.1_seed18" / "checkpoint-21.jsonl"
    future = time.time() + 60
    os.utime(row, (future, future))                                  # evaluated after the plots
    stale = _export(repo, results)
    assert stale.returncode != 0 and "figures are stale" in stale.stderr and "make math-plots" in stale.stderr
    assert "older than their inputs: frontier.csv" in stale.stderr
    big = _export(repo, results, "seeds17-18", "--allow-stale", "--max-mb", "0.001")
    assert big.returncode != 0 and "--max-mb" in big.stderr
    assert not list((repo.root / "evidence" / "math").glob("2026-10-22_*"))       # nothing half-written left
    ok = _export(repo, results, "seeds17-18", "--allow-stale")
    assert ok.returncode == 0, ok.stdout + ok.stderr
    m = json.loads((repo.root / "evidence" / "math" / "2026-10-22_seeds17-18" / "MANIFEST.json").read_text())
    assert "frontier.csv" in m["figures_stale"]


def test_a_missing_or_old_figure_is_refused_too(repo, results: Path) -> None:
    """frontier.csv can be fresh while a plot failed: every output the launcher makes is checked."""
    (results / "figures" / "retention.png").unlink()
    run = _export(repo, results)
    assert run.returncode != 0 and "missing figures/retention.png" in run.stderr
    (results / "figures" / "retention.png").write_bytes(b"fake")
    old = (results / "frontier.csv").stat().st_mtime - 60
    os.utime(results / "figures" / "diversity.png", (old, old))     # left over from an earlier plot run
    run = _export(repo, results)
    assert run.returncode != 0 and "older than their inputs: figures/diversity.png" in run.stderr


def test_make_evidence_needs_a_task_and_a_name(repo, results: Path) -> None:
    run = repo.run("make", "evidence", check=False, RESULTS_DIR=str(results.parent))
    assert run.returncode != 0 and "make evidence needs TASK=math|helpful and NAME" in run.stdout
    run = repo.run("make", "evidence", "TASK=math", "NAME=x", "EVIDENCE_ARGS=--dry-run", check=False,
                   RESULTS_DIR=str(results.parent))
    assert run.returncode == 0, run.stdout + run.stderr
    assert "evidence/math/" in run.stdout and "3 comparisons" in run.stdout
    assert not list((repo.root / "evidence" / "math").glob("*_x"))


# --- the repo's own evidence files and agent docs -------------------------------


def test_the_math_snapshot_spec_is_well_formed() -> None:
    spec = OmegaConf.to_container(OmegaConf.load(ROOT / "evidence" / "math" / "snapshot.yaml"))
    spec_re = re.compile(r"[\w.\-]+@(\d+|last|ckpt\d+|kl\d+(\.\d+)?)")
    for line in spec["compare"]:
        words = line.split()
        flags = [w for w in words if w.startswith("--")]
        assert flags in (["--pool"], ["--seed"]), line
        specs = [w for w in words if "@" in w]
        assert len(specs) >= 2 and all(spec_re.fullmatch(s) for s in specs), line
    assert all(spec_re.fullmatch(c) for c in spec["examples"]["checkpoints"])


def test_findings_claims_have_ids_and_statuses() -> None:
    text = (ROOT / "evidence" / "FINDINGS.md").read_text()
    claims = re.findall(r"^### (M\d+)\. ", text, re.M)
    assert claims == [f"M{i}" for i in range(1, len(claims) + 1)]                # numbered, none reused
    statuses = re.findall(r"^Status: (\w+)", text, re.M)
    assert len(statuses) == len(claims) and set(statuses) <= {"tentative", "supported", "refuted", "superseded"}


def test_runs_save_their_resolved_config_and_final_state(tmp_path: Path) -> None:
    from posttrain_lab.config import load_config
    from posttrain_lab.train import common

    cfg = load_config("math/m3_arm_dpo.yaml", ["train.beta=0.3", "seed=18", f"output_dir={tmp_path / 'run'}"])
    (tmp_path / "run").mkdir()
    common.save_resolved_config(cfg)
    saved = OmegaConf.load(tmp_path / "run" / common.RESOLVED_CONFIG)
    assert saved.train.beta == 0.3 and saved.seed == 18 and saved.init_from.endswith("sft-base_seed18")
    assert not list((tmp_path / "run").glob(".*.tmp"))

    calls = []
    common.save_final_state(type("T", (), {"save_state": lambda self: calls.append("save_state")})())
    common.save_final_state(object())                       # trainers without it: a no-op
    assert calls == ["save_state"]
    for name in ("sft", "dpo", "grpo", "ppo", "rm"):        # every wrapper, right after the final model
        src = (ROOT / "src" / "posttrain_lab" / "train" / f"{name}.py").read_text()
        assert "trainer.save_model(str(cfg.output_dir))\n    save_final_state(trainer)\n" in src, name


def test_agent_docs_and_skills() -> None:
    assert (ROOT / "CLAUDE.md").read_text().strip() == "@AGENTS.md"
    agents = (ROOT / "AGENTS.md").read_text()
    assert "evidence/README.md" in agents and "make evidence" in agents and "bash` 3.2" in agents
    for skill in ("evidence-snapshot", "update-findings"):
        text = (ROOT / ".claude" / "skills" / skill / "SKILL.md").read_text()
        front = text.split("---")[1]
        assert f"name: {skill}" in front and "description:" in front
