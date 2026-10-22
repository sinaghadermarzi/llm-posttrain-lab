"""Math eval on the full GSM8K test set without redoing everything.

Moving eval.num_prompts from 512 to 1319 must re-sample (core_math) and
recompute what derives from the samples (diversity) on already-evaluated
checkpoints, keep the groups that don't depend on the prompts (retention,
pref_logps), and never leave a row claiming old numbers for new samples."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from posttrain_lab.eval import groups

ROOT = Path(__file__).resolve().parents[1]
OLD_ROW = {"run": "dpo_beta0.1_seed17", "arm": "dpo", "hyperparam": "beta0.1", "checkpoint": "x", "step": 58,
           "kl_k3": 1.0, "kl_k1": 0.9, "pass1": 0.5, "pass8": 0.8, "num_samples": 512 * 8,
           "ppl_wikitext": 12.0, "harness": {"arc_easy": 0.76}, "distinct_2": 0.6, "self_bleu": 0.4,
           "logp_chosen": -33.0}
ALL = ["core_math", "retention_ppl", "retention_harness", "diversity", "pref_logps"]


def test_math_evaluates_the_whole_test_split() -> None:
    from posttrain_lab.config import load_config

    assert load_config("eval.yaml", ["task=math"]).eval.num_prompts == 1319


def test_stale_groups_follow_the_prompt_count() -> None:
    assert groups.row_prompts(OLD_ROW, 8) == 512                       # older rows: from num_samples
    assert groups.missing_groups(OLD_ROW, ALL, 1319, 8) == ["core_math", "diversity"]
    assert groups.missing_groups(OLD_ROW, ALL, 512, 8) == []           # same count: nothing to do
    assert groups.missing_groups({**OLD_ROW, "eval_prompts": 1319}, ALL, 1319, 8) == []
    assert groups.missing_groups(OLD_ROW, ALL, None, 8) == []          # no count configured: sentinels only
    helpful = {"kl_k3": 1.0, "reward_train_rm": 0.1, "reward_judge": 0.2, "num_samples": 2048}
    assert groups.missing_groups(helpful, ["core"], 512, 4) == []      # helpful rows stay put


def test_passk_standard_errors_are_over_prompts() -> None:
    from posttrain_lab.eval.passk import passk_metrics

    samples = [{"prompt_id": "a", "correct": c} for c in (1, 1, 1, 1)] + \
              [{"prompt_id": "b", "correct": c} for c in (0, 0, 0, 0)]
    out = passk_metrics(samples, 4)
    assert out["pass1"] == 0.5 and out["pass4"] == 0.5
    assert out["pass1_se"] == pytest.approx(0.5)   # sd of (1, 0) = 0.707, / sqrt(2)
    assert out["pass4_se"] == pytest.approx(0.5)


@pytest.fixture
def fake_groups(monkeypatch):
    """run_eval with stand-in metric groups: records what ran."""
    from posttrain_lab.eval import run_eval

    ran: list[str] = []

    def core(cfg, ckpt, ctx):
        ran.append("core_math")
        ctx["write_samples"]([{"prompt_id": p["prompt_id"], "response": "r", "correct": True} for p in ctx["prompts"]])
        return {"kl_k3": 2.0, "kl_k1": 1.9, "pass1": 0.6, "pass8": 0.9, "num_samples": 8 * len(ctx["prompts"]),
                "eval_prompts": len(ctx["prompts"])}

    def make(name, out):
        def fn(cfg, ckpt, ctx):
            ran.append(name)
            if name == "diversity":
                assert len(ctx["load_samples"]()) == len(ctx["prompts"])   # the new samples, not the old file
            return out
        return fn

    fake = {"core_math": core, "retention_ppl": make("retention_ppl", {"ppl_wikitext": 11.0}),
            "retention_harness": make("retention_harness", {"harness": {}}),
            "diversity": make("diversity", {"distinct_2": 0.5, "self_bleu": 0.5}),
            "pref_logps": make("pref_logps", {"logp_chosen": -40.0})}
    monkeypatch.setattr(run_eval, "GROUPS", {g: (groups.SENTINELS[g], fn) for g, fn in fake.items()})
    return run_eval, ran, fake


def _cfg(tmp_path: Path):
    return OmegaConf.create({"eval": {"out_dir": str(tmp_path / "eval"), "metric_groups": ALL,
                                      "samples_per_prompt": 8}, "task": "math", "train": {"report_to": "none"}})


def _ckpt(tmp_path: Path) -> Path:
    ck = tmp_path / "runs" / "dpo_beta0.1_seed17" / "checkpoint-58"
    ck.mkdir(parents=True)
    return ck


def test_old_rows_are_resampled_and_the_rest_is_kept(fake_groups, tmp_path: Path) -> None:
    run_eval, ran, _ = fake_groups
    ck = _ckpt(tmp_path)
    row_path = tmp_path / "eval" / "dpo_beta0.1_seed17" / "checkpoint-58.jsonl"
    row_path.parent.mkdir(parents=True)
    row_path.write_text(json.dumps(OLD_ROW) + "\n")
    prompts = [{"prompt_id": f"gsm8k-{i}"} for i in range(1319)]

    run_eval.evaluate_checkpoint(_cfg(tmp_path), ck, prompts)
    assert ran == ["core_math", "diversity"]
    row = json.loads(row_path.read_text())
    assert (row["eval_prompts"], row["num_samples"], row["pass1"], row["distinct_2"]) == (1319, 1319 * 8, 0.6, 0.5)
    assert (row["ppl_wikitext"], row["logp_chosen"]) == (12.0, -33.0)        # prompt-independent: untouched

    ran.clear()
    run_eval.evaluate_checkpoint(_cfg(tmp_path), ck, prompts)
    assert ran == []                                                         # now complete


def test_a_crash_after_resampling_never_keeps_old_diversity(fake_groups, tmp_path: Path) -> None:
    run_eval, ran, fake = fake_groups

    def boom(cfg, ckpt, ctx):
        raise RuntimeError("preempted")

    ck = _ckpt(tmp_path)
    row_path = tmp_path / "eval" / "dpo_beta0.1_seed17" / "checkpoint-58.jsonl"
    row_path.parent.mkdir(parents=True)
    row_path.write_text(json.dumps(OLD_ROW) + "\n")
    prompts = [{"prompt_id": f"gsm8k-{i}"} for i in range(1319)]
    run_eval.GROUPS["diversity"] = (groups.SENTINELS["diversity"], boom)
    with pytest.raises(RuntimeError):
        run_eval.evaluate_checkpoint(_cfg(tmp_path), ck, prompts)
    row = json.loads(row_path.read_text())
    assert row["eval_prompts"] == 1319 and "distinct_2" not in row and "self_bleu" not in row

    run_eval.GROUPS["diversity"] = (groups.SENTINELS["diversity"], fake["diversity"])
    ran.clear()
    run_eval.evaluate_checkpoint(_cfg(tmp_path), ck, prompts)
    assert ran == ["diversity"]                                              # resumes at the missing group only


def test_status_counts_old_rows_as_not_done(tmp_path: Path, monkeypatch) -> None:
    from posttrain_lab import status

    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    ck = tmp_path / "math" / "runs" / "dpo_beta0.1_seed17" / "checkpoint-58"
    ck.mkdir(parents=True)
    (ck / "config.json").write_text("{}")
    row = tmp_path / "math" / "eval" / "dpo_beta0.1_seed17" / "checkpoint-58.jsonl"
    row.parent.mkdir(parents=True)
    row.write_text(json.dumps(OLD_ROW) + "\n")
    eval_cfg = status.load_config("eval.yaml", ["task=math"])
    (stage,) = [s for s in status.TRACKS["math"]["stages"] if s.kind == "eval"]
    ev = status.eval_state("math", eval_cfg, [], stage)
    assert ev["per_group"]["core_math"] == 0 and ev["per_group"]["diversity"] == 0
    assert ev["per_group"]["retention_ppl"] == 1 and ev["per_group"]["pref_logps"] == 1
    assert ev["complete"] == 0


# --- analysis ----------------------------------------------------------------


def _samples(path: Path, solved: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for pid, k in solved.items():
            for j in range(8):
                f.write(json.dumps({"prompt_id": pid, "correct": j < k}) + "\n")
    path.with_name(path.name.replace(".samples", "")).write_text(json.dumps({"kl_k1": 2.5}) + "\n")


def test_compare_math_reports_paired_differences(tmp_path: Path) -> None:
    ev = tmp_path / "eval"
    prompts = [f"gsm8k-{i}" for i in range(200)]
    # B solves exactly one more of 8 samples than A on every prompt: tiny paired SE
    _samples(ev / "dpo_beta0.3_seed17" / "checkpoint-116.samples.jsonl", {p: i % 7 for i, p in enumerate(prompts)})
    _samples(ev / "grpo_beta0.01_seed17" / "checkpoint-374.samples.jsonl", {p: i % 7 + 1 for i, p in enumerate(prompts)})
    out = subprocess.run([sys.executable, str(ROOT / "analysis" / "compare_math.py"), str(tmp_path),
                          "dpo_beta0.3@116", "grpo_beta0.01@last"], capture_output=True, text=True, cwd=ROOT / "analysis")
    assert out.returncode == 0, out.stderr
    paired = out.stdout.split("paired vs", 1)[1]
    line = next(l for l in paired.splitlines() if l.strip().startswith("grpo_beta0.01@last"))
    assert "+0.125 ± 0.000" in line and "better" in line and "(200 prompts)" in line
    assert "dpo_beta0.3@116" in out.stdout and "2.5" in out.stdout


def test_frontier_draws_error_bars_when_ses_exist(tmp_path: Path, monkeypatch) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    monkeypatch.syspath_prepend(str(ROOT / "analysis"))
    from frontier_plot import plot_panel

    df = pd.DataFrame([{"run": "grpo_beta0.1_seed17", "arm": "grpo", "hyperparam": "beta0.1", "step": s,
                        "kl_k1": k, "pass1": p, "pass1_se": 0.01} for s, k, p in ((1, 0.5, 0.5), (2, 1.0, 0.55))])
    fig, ax = plt.subplots()
    plot_panel(ax, df, "pass1", "pass@1")
    assert ax.containers, "no error bars drawn"
    plt.close(fig)
