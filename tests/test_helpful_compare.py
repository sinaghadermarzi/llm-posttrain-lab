"""Matched-KL comparisons on the helpful track (judge reward), fixed before its first run's results.

compare_math.py scored pass@1 only, and evidence/helpful/snapshot.yaml did not
exist, so a helpful snapshot could hold no comparisons fixed in advance.
compare_math now scores helpful samples by the judge's reward (or the
training RM's, --metric train_rm), and klX can name an arm, picking the
checkpoint nearest X nats across all of the arm's runs by KL alone. The
math output is unchanged (test_full_test_eval.py, test_seeds.py)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import compare_math  # noqa: E402
import export_evidence  # noqa: E402

PROMPTS = [f"Write about topic {i}." for i in range(50)]


def _ckpt(root: Path, run: str, step: int, kl: float, judge: float, train_rm: float | None = None,
          slope: float = 0.01) -> None:
    """A helpful eval row and its samples: per prompt i, judge + i * slope (4 samples alike)."""
    d = root / "eval" / run
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"checkpoint-{step}.samples.jsonl").open("w") as f:
        for i, p in enumerate(PROMPTS):
            for j in range(4):
                f.write(json.dumps({"prompt_id": f"id{i}", "prompt": p, "sample_idx": j, "response": f"{run} {p}",
                                    "finished": True, "score_judge": judge + i * slope,
                                    "score_train_rm": (judge if train_rm is None else train_rm) + i * slope}) + "\n")
    (d / f"checkpoint-{step}.jsonl").write_text(json.dumps({"kl_k1": kl}) + "\n")


def _compare(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "analysis" / "compare_math.py"), str(root), *args],
                          capture_output=True, text=True, cwd=ROOT / "analysis")


def _row(text: str, *needles: str) -> str:
    return next(line for line in text.splitlines() if all(n in line for n in needles))


def test_helpful_samples_are_scored_by_the_judge(tmp_path: Path) -> None:
    _ckpt(tmp_path, "dpo_beta0.1_seed17", 30, 4.0, judge=1.0, train_rm=3.0)
    _ckpt(tmp_path, "ppo_kl0.05_seed17", 13, 4.1, judge=1.5, train_rm=2.0)
    out = _compare(tmp_path, "dpo_beta0.1@last", "ppo_kl0.05@last")
    assert out.returncode == 0, out.stderr
    assert "judge" in out.stdout.splitlines()[0] and "pass@1" not in out.stdout
    assert "1.245" in _row(out.stdout, "dpo_beta0.1@last ")                     # mean of 1.00..1.49
    paired = _row(out.stdout.split("paired vs", 1)[1], "ppo_kl0.05@last")
    assert "+0.500 ± 0.000" in paired and "better" in paired and "(50 prompts)" in paired

    rm = _compare(tmp_path, "--metric", "train_rm", "dpo_beta0.1@last", "ppo_kl0.05@last")
    assert rm.returncode == 0 and "train RM" in rm.stdout.splitlines()[0]
    assert "-1.000 ± 0.000" in _row(rm.stdout.split("paired vs", 1)[1], "ppo_kl0.05@last") and "worse" in rm.stdout


def test_a_math_metric_on_helpful_samples_says_so(tmp_path: Path) -> None:
    _ckpt(tmp_path, "dpo_beta0.1_seed17", 30, 4.0, judge=1.0)
    _ckpt(tmp_path, "ppo_kl0.05_seed17", 13, 4.1, judge=1.5)
    out = _compare(tmp_path, "--metric", "pass1", "dpo_beta0.1@last", "ppo_kl0.05@last")
    assert out.returncode != 0 and "no 'correct' (wrong --metric for this track?)" in out.stderr


def _arms(root: Path, seeds=(17,)) -> None:
    for seed in seeds:
        for run, steps in {"dpo_beta0.01": [(10, 3.0), (20, 9.0)], "dpo_beta0.3": [(10, 1.0), (20, 4.6)],
                           "dpo-nll_beta0.3": [(10, 5.0)], "ppo_kl0.05": [(13, 2.0), (26, 5.2)]}.items():
            for step, kl in steps:
                _ckpt(root, f"{run}_seed{seed}", step, kl + (seed - 17) / 10, judge=kl / 10)


def test_an_arm_at_a_kl_target_takes_the_nearest_checkpoint_of_any_of_its_runs(tmp_path: Path) -> None:
    _arms(tmp_path)
    out = _compare(tmp_path, "--seed", "17", "ppo@kl5", "dpo@kl5", "dpo-nll@kl5")
    assert out.returncode == 0, out.stderr
    assert "ppo_kl0.05_seed17 ckpt2 (step 26)" in _row(out.stdout, "ppo@kl5 ")
    # DPO's runs: 3.0 and 9.0 (beta 0.01), 1.0 and 4.6 (beta 0.3): the nearest is beta 0.3's step 20, not dpo-nll's
    assert "dpo_beta0.3_seed17 ckpt2 (step 20)" in _row(out.stdout, "dpo@kl5 ")
    assert "dpo-nll_beta0.3_seed17 ckpt1 (step 10)" in _row(out.stdout, "dpo-nll@kl5 ")


def test_arms_pool_over_seeds(tmp_path: Path) -> None:
    _arms(tmp_path, seeds=(17, 18))
    out = _compare(tmp_path, "--pool", "ppo@kl5", "dpo@kl5")
    assert out.returncode == 0, out.stderr
    assert "pooled over seeds 17, 18" in out.stdout
    assert "dpo_beta0.3_seed18 ckpt2 (step 20)" in _row(out.stdout, "dpo@kl5 ", " 18 ")
    assert "(2 seeds, t1 = 12.71)" in _row(out.stdout, "  dpo@kl5 ", "seeds   ")


def test_an_arm_needs_a_kl_target(tmp_path: Path) -> None:
    _arms(tmp_path)
    out = _compare(tmp_path, "--seed", "17", "ppo@last", "dpo@last")
    assert out.returncode != 0 and "use klX to pick by KL across them" in out.stderr


def test_one_run_keeps_its_old_note(tmp_path: Path) -> None:
    _arms(tmp_path)
    out = _compare(tmp_path, "--seed", "17", "ppo_kl0.05@kl5", "dpo_beta0.3@kl5")
    assert "  ckpt2 (step 26)" in _row(out.stdout, "ppo_kl0.05@kl5 ") and "_seed17 ckpt" not in out.stdout


def _helpful_tree(root: Path) -> None:
    runs = {"sft-base": [(1, 0.0)], "sft": [(40, 3.0), (80, 7.5)], "ppo_kl0.02": [(13, 2.1), (26, 6.0), (39, 11.0)],
            "ppo_kl0.1": [(13, 1.0), (26, 2.5)], "grpo_beta0.0": [(100, 2.0), (200, 5.0), (300, 12.0)],
            "grpo_beta0.1": [(100, 0.5), (200, 1.0)], "dpo_beta0.01": [(32, 4.0), (64, 15.0)],
            "dpo_beta0.1": [(32, 1.8), (64, 5.5)], "dpo-nll_beta0.01": [(32, 2.2), (64, 9.0)],
            "dpo-nll_beta0.1": [(32, 1.5), (64, 4.8)]}
    for run, steps in runs.items():
        for step, kl in steps:
            # the per-prompt slope grows with KL, so the checkpoints' judge spread is widest on the last prompt
            _ckpt(root, f"{run}_seed17", step, kl, judge=kl / 5, train_rm=kl / 2, slope=kl / 100)


def test_every_line_of_the_helpful_snapshot_spec_runs(tmp_path: Path) -> None:
    spec = OmegaConf.to_container(OmegaConf.load(ROOT / "evidence" / "helpful" / "snapshot.yaml"))
    root, out = tmp_path / "helpful", tmp_path / "snap"
    _helpful_tree(root)
    (out / "tables").mkdir(parents=True)
    done = export_evidence.run_compares(root, spec["compare"], out)
    failed = [(d["args"], (out / d["file"]).read_text()[-300:]) for d in done if d["status"] != "ok"]
    assert not failed, failed
    ten = (out / "tables" / "compare_3.txt").read_text()                  # ppo@kl10 ... sft@kl10
    assert "ppo_kl0.02_seed17 ckpt3 (step 39)" in ten and "judge" in ten
    assert "! -2.50 nats from 10" in _row(ten, "sft@kl10 ")                 # never reached: flagged, not dropped
    assert "train RM" in (out / "tables" / "compare_4.txt").read_text()

    examples = export_evidence.export_examples(root, spec["examples"], out)
    assert examples["checkpoints"] == spec["examples"]["checkpoints"] and examples["prompts"] == 6
    md = (out / "examples" / "examples.md").read_text()
    # prompt 49 on grpo_beta0.0's last checkpoint (12 nats): judge 12/5 + 49 * 0.12, training RM 12/2 + 49 * 0.12
    assert "widest spread of judge reward" in md and "**grpo_beta0.0@last**: judge +8.28, training RM +11.88" in md
    first = json.loads((out / "examples" / "examples.jsonl").read_text().splitlines()[0])
    assert first["question"] == PROMPTS[-1] and first["score_judge"] is not None   # the widest spread first


@pytest.mark.parametrize("task", ["math", "helpful"])
def test_snapshot_specs_only_use_known_flags(task: str) -> None:
    spec = OmegaConf.to_container(OmegaConf.load(ROOT / "evidence" / task / "snapshot.yaml"))
    for line in spec["compare"]:
        words = line.split()
        flags = [w for w in words if w.startswith("--")]
        assert set(flags) <= {"--pool", "--seed", "--metric"} and ("--pool" in flags) != ("--seed" in flags), line
        if "--metric" in flags:
            assert words[words.index("--metric") + 1] in compare_math.METRICS, line
