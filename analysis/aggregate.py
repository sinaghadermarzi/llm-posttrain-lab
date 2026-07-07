"""Collect per-checkpoint eval rows into results/frontier.csv.

Adds z-scored reward columns: both graders' raw scores live on arbitrary
scales, so each is standardized against the per-sample score distribution of
the pi_SFT baseline — "how many baseline standard deviations better than the
model every arm started from."

Usage: python analysis/aggregate.py [results_root]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pandas as pd


def load_rows(eval_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(eval_root.glob("*/*.jsonl")):
        if path.name.endswith(".samples.jsonl"):
            continue
        with path.open() as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    if not rows:
        sys.exit(f"no eval rows under {eval_root} — run stage 4 first")
    return pd.DataFrame(rows)


def baseline_stats(eval_root: Path) -> dict[str, tuple[float, float]]:
    """Per-sample judge/train-RM score mean+std of the sft-base run."""
    sample_files = sorted(eval_root.glob("sft-base_*/*.samples.jsonl"))
    if not sample_files:
        sys.exit("no sft-base samples found — evaluate the pi_SFT baseline first")
    scores: dict[str, list[float]] = {"score_judge": [], "score_train_rm": []}
    for path in sample_files:
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                for key in scores:
                    if key in row:
                        scores[key].append(row[key])
    return {
        key: (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 1.0)
        for key, vals in scores.items()
        if vals
    }


def main() -> None:
    results_root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    df = load_rows(results_root / "eval")

    # z-score the learned-reward columns against the pi_SFT baseline where they
    # exist (helpful track); math rows carry absolute pass@k metrics instead.
    if "reward_judge" in df.columns:
        stats = baseline_stats(results_root / "eval")
        judge_mean, judge_std = stats["score_judge"]
        rm_mean, rm_std = stats["score_train_rm"]
        df["reward_judge_z"] = (df["reward_judge"] - judge_mean) / (judge_std or 1.0)
        df["reward_train_rm_z"] = (df["reward_train_rm"] - rm_mean) / (rm_std or 1.0)
        df["rm_gap_z"] = df["reward_train_rm_z"] - df["reward_judge_z"]

    # flatten nested harness scores into harness_<task> columns for the CSV
    if "harness" in df.columns:
        flat = pd.json_normalize(df["harness"]).add_prefix("harness_")
        df = pd.concat(
            [df.drop(columns=["harness"]).reset_index(drop=True), flat.reset_index(drop=True)],
            axis=1,
        )

    df = df.sort_values(["arm", "hyperparam", "step"], na_position="first")
    out = results_root / "frontier.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} frontier points, "
          f"{df['run'].nunique()} runs, arms: {sorted(df['arm'].unique())})")


if __name__ == "__main__":
    main()
