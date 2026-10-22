"""Math-track headline figure: accuracy vs KL(policy || pi_SFT-math).

Left: pass@1 (single-sample accuracy) vs sequence KL. Right: pass@k (any of
k samples correct). The RL story to look for: GRPO pushes pass@1 up at
bounded KL while pass@k holds (diversity preserved); imitation-style arms
buy pass@1 by collapsing onto fewer solution modes, so pass@k stalls or
falls even as pass@1 rises.

With one training seed each run is drawn as is (error bars: SE over prompts).
With several, each (arm, hyperparam) is one curve through the seed means at
each checkpoint index (aggregate.py:seed_means), with error bars = SE of the
mean across seeds on both pass@k and KL.

Usage: python analysis/frontier_math.py [results_root]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from aggregate import add_seed_columns, seed_means
from paths import default_root
from frontier_plot import plot_panel
from style import KL_COL
from style import METHOD_COLORS, METHOD_LABELS, METHOD_LINESTYLES, apply_style, marker_for


def seed_curves(df: pd.DataFrame, n_seeds: int) -> pd.DataFrame:
    """frontier.csv rows -> one pseudo-run per (arm, hyperparam): the seed
    means at each checkpoint index, <metric>_se = the across-seed SE (which
    already contains each seed's eval noise), in the shape plot_panel draws."""
    means = seed_means(add_seed_columns(df))
    short = means[means["n_seeds"] < n_seeds]
    if len(short):
        print(f"note: {len(short)} of {len(means)} points have fewer than {n_seeds} seeds (not trained / "
              "evaluated for every seed yet): their mean is over the seeds that have them")
    out = means[["arm", "hyperparam", "ckpt_index", "n_seeds"]].copy()
    out["run"] = means["arm"] + "_" + means["hyperparam"].fillna("").astype(str)
    out["step"] = means["ckpt_index"]
    for col in [c for c in means.columns if c.endswith("_seed_se")]:
        metric = col[: -len("_seed_se")]
        out[metric] = means[metric]
        out[f"{metric}_se"] = means[col]
    if KL_COL not in out.columns:
        out[KL_COL] = means[KL_COL]
    return out


def main() -> None:
    apply_style()
    results_root = Path(sys.argv[1] if len(sys.argv) > 1 else default_root("math"))
    df = pd.read_csv(results_root / "frontier.csv")
    seeds = sorted(add_seed_columns(df)["seed"].dropna().unique())
    if len(seeds) > 1:
        df = seed_curves(df, len(seeds))

    passk_col = next(
        (c for c in df.columns if re.fullmatch(r"pass\d+", c) and c != "pass1"), None
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    plot_panel(axes[0], df, "pass1", "pass@1 (test accuracy)")
    if passk_col:
        plot_panel(axes[1], df, passk_col, f"pass@{passk_col[4:]}")
    for ax in axes:
        ax.set_ylabel(ax.get_ylabel().replace("  [z vs π_SFT]", ""))

    handles = [
        plt.Line2D([], [], color=METHOD_COLORS[m], marker=marker_for(m),
                   linestyle="none" if m == "sft-base" else METHOD_LINESTYLES.get(m, "-"), label=METHOD_LABELS[m])
        for m in METHOD_COLORS
        if (df["arm"] == m).any()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.04))
    title = "GSM8K (verifiable rewards): accuracy vs distribution shift"
    if len(seeds) > 1:
        title += f"\nmean ± SE over seeds {', '.join(str(int(s)) for s in seeds)}"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    out_dir = results_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"frontier_math.{ext}")
    print(f"wrote {out_dir}/frontier_math.png|.pdf")


if __name__ == "__main__":
    main()
