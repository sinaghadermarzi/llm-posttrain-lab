"""Collect per-checkpoint eval rows into results/frontier.csv.

Adds z-scored reward columns: both graders' raw scores live on arbitrary
scales, so each is standardized against the per-sample score distribution of
the pi_SFT baseline — "how many baseline standard deviations better than the
model every arm started from."

Every row keeps its run name (..._seed<N>) and gets a `seed` and a
`ckpt_index` column (1 = the run's first checkpoint). frontier_seeds.csv
averages the seeds: one row per (arm, hyperparam, ckpt_index) with n_seeds,
each metric's mean across seeds, `<metric>_seed_se` (the SE of that mean
across seeds: sd / sqrt(n_seeds), empty with one seed) and, where the rows
carry a prompt SE (pass@k), `<metric>_prompt_se` (the part of the error that
eval sampling alone explains). Seeds are matched by checkpoint index, not
step: the DPO arms train on each seed's own self-generated pairs, so their
step counts differ between seeds.

Usage: python analysis/aggregate.py [results_root]
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path

import pandas as pd

from paths import default_root


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


# row columns that are not metrics to average across seeds
META_COLS = {"run", "arm", "hyperparam", "checkpoint", "step", "seed", "ckpt_index", "num_samples", "eval_prompts"}
GROUP_COLS = ["arm", "hyperparam", "ckpt_index"]


def add_seed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """seed (from the run name) and ckpt_index (1-based, by step within the
    run; a final flat save, step -1, comes last)."""
    df = df.copy()
    df["seed"] = df["run"].map(lambda r: int(m[1]) if (m := re.search(r"_seed(\d+)$", str(r))) else None).astype("Int64")
    order = df["step"].where(df["step"] >= 0, float("inf"))
    df["ckpt_index"] = order.groupby(df["run"]).rank(method="first").astype(int)
    return df


def seed_means(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and across-seed SE of every numeric metric per (arm, hyperparam,
    ckpt_index). df: frontier.csv rows (add_seed_columns applied)."""
    metrics = [c for c in df.columns if c not in META_COLS and not c.endswith("_se")
               and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    out = []
    for key, g in df.groupby(GROUP_COLS, dropna=False, sort=True):
        n = int(g["seed"].nunique(dropna=False))
        row = dict(zip(GROUP_COLS, key))
        row.update({"n_seeds": n, "seeds": ";".join(str(s) for s in sorted(g["seed"].dropna().unique())),
                    "step": g["step"].mean(), "step_min": g["step"].min(), "step_max": g["step"].max()})
        for col in metrics:
            vals = g[col].dropna()
            row[col] = vals.mean() if len(vals) else float("nan")
            row[f"{col}_seed_se"] = vals.std(ddof=1) / math.sqrt(len(vals)) if len(vals) > 1 else float("nan")
            if f"{col}_se" in g.columns:
                ses = g[f"{col}_se"].dropna()
                row[f"{col}_prompt_se"] = math.sqrt((ses ** 2).sum()) / len(ses) if len(ses) else float("nan")
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    results_root = Path(sys.argv[1] if len(sys.argv) > 1 else default_root())
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

    df = add_seed_columns(df)
    df = df.sort_values(["arm", "hyperparam", "seed", "step"], na_position="first")
    out = results_root / "frontier.csv"
    df.to_csv(out, index=False)
    seeds = sorted(df["seed"].dropna().unique())
    print(f"wrote {out} ({len(df)} frontier points, "
          f"{df['run'].nunique()} runs, arms: {sorted(df['arm'].unique())}, seeds: {[int(s) for s in seeds]})")
    means = seed_means(df)
    means.to_csv(results_root / "frontier_seeds.csv", index=False)
    short = means[means["n_seeds"] < len(seeds)]
    print(f"wrote {results_root / 'frontier_seeds.csv'} ({len(means)} points, mean over {len(seeds)} seed(s)"
          + (f"; {len(short)} with fewer seeds: not every seed is trained / evaluated there yet)" if len(short) else ")"))


if __name__ == "__main__":
    main()
