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
           "kl_k3": 1.0, "kl_k1": 0.9, "pass1": 0.5, "pass8": 0.8, "pass_gap": 0.3, "num_samples": 512 * 8,
           "mean_response_tokens": 130.0, "mean_token_nll": 0.4,
           "ppl_wikitext": 12.0, "harness": {"arc_easy": 0.76}, "distinct_1": 0.3, "distinct_2": 0.6,
           "distinct_3": 0.8, "self_bleu": 0.4, "logp_chosen": -33.0}
OLD_PROMPT_KEYS = {"kl_k3", "kl_k1", "pass1", "pass8", "pass_gap", "num_samples", "mean_response_tokens",
                   "mean_token_nll", "distinct_1", "distinct_2", "distinct_3", "self_bleu"}
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
    assert row["eval_prompts"] == 1319
    # nothing of the old 512-prompt evaluation survives, not just the sentinels
    assert not {"distinct_1", "distinct_2", "distinct_3", "self_bleu", "pass_gap", "mean_token_nll"} & set(row)
    assert (row["ppl_wikitext"], row["logp_chosen"]) == (12.0, -33.0)

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


def test_recomputed_values_are_sent_to_mlflow_again(fake_groups, tmp_path: Path) -> None:
    """The MLflow sidecar lists what was logged; the stale keys must leave it,
    or _sync_tracking would keep MLflow on the 512-prompt numbers."""
    run_eval, ran, _ = fake_groups
    ck = _ckpt(tmp_path)
    row_path = tmp_path / "eval" / "dpo_beta0.1_seed17" / "checkpoint-58.jsonl"
    row_path.parent.mkdir(parents=True)
    row_path.write_text(json.dumps(OLD_ROW) + "\n")
    marker = row_path.with_suffix(".mlflow.json")
    marker.write_text(json.dumps(sorted(set(OLD_ROW) - {"run", "arm", "hyperparam", "checkpoint", "step"})))
    run_eval.evaluate_checkpoint(_cfg(tmp_path), ck, [{"prompt_id": f"gsm8k-{i}"} for i in range(1319)])
    logged = set(json.loads(marker.read_text()))
    assert not OLD_PROMPT_KEYS & logged
    assert {"ppl_wikitext", "harness", "logp_chosen"} <= logged


def test_group_outputs_cover_what_the_groups_write() -> None:
    """GROUP_OUTPUTS must own every key core_math and diversity return."""
    import ast

    from posttrain_lab.eval import diversity, kl
    from posttrain_lab.eval.passk import passk_metrics

    tree = ast.parse(Path(kl.__file__).read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "compute_kl")
    ret = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict)][-1]
    kl_keys = {k.value for k in ret.value.keys if isinstance(k, ast.Constant)}
    samples = [{"prompt_id": f"p{i % 3}", "response": "a b c d", "correct": i % 2 == 0} for i in range(12)]
    passk_keys = set(passk_metrics(samples, 4))
    core = {**dict.fromkeys(kl_keys | passk_keys), "num_samples": 1, "eval_prompts": 1, "ppl_wikitext": 1}
    assert set(groups.group_keys(core, "core_math")) == kl_keys | passk_keys | {"num_samples", "eval_prompts"}
    div = diversity.compute_group(None, None, {"load_samples": lambda: samples})
    assert set(groups.group_keys({**div, "logp_chosen": 1}, "diversity")) == set(div)


def test_compare_math_pairs_by_question_not_position(tmp_path: Path) -> None:
    """gsm8k-<i> is a position: an old 512-prompt file and a new full-split file
    give the same ids to different questions. Pairing must use the question."""
    def write(path: Path, rows: list[tuple[str, str, int]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for pid, question, k in rows:
                for j in range(8):
                    f.write(json.dumps({"prompt_id": pid, "prompt": question, "correct": j < k}) + "\n")
    ev = tmp_path / "eval"
    write(ev / "dpo_beta0.3_seed17" / "checkpoint-116.samples.jsonl", [("gsm8k-0", "A", 2), ("gsm8k-1", "B", 4)])
    # the full split puts other questions at those positions
    write(ev / "grpo_beta0.01_seed17" / "checkpoint-374.samples.jsonl",
          [("gsm8k-0", "X", 8), ("gsm8k-1", "A", 3), ("gsm8k-2", "B", 5)])
    out = subprocess.run([sys.executable, str(ROOT / "analysis" / "compare_math.py"), str(tmp_path),
                          "dpo_beta0.3@116", "grpo_beta0.01@374"], capture_output=True, text=True, cwd=ROOT / "analysis")
    assert out.returncode == 0, out.stderr
    paired = out.stdout.split("paired vs", 1)[1]
    assert "share 2 questions" in paired                       # A and B, not positions 0 and 1
    assert "+0.125 ± 0.000" in paired and "(2 prompts)" in paired
