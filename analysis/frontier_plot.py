"""THE figure: reward vs KL(policy || pi_SFT) frontier.

Two panels, one shared x (sequence KL in nats, k1): the held-out judge (the
honest scoreboard, left) and the training RM (the optimized objective,
right). Each run is one curve through its checkpoints in training order;
color = method, direct label = the run's KL-knob value at the curve end.

The story to look for: PPO/GRPO curves bending up-and-left of SFT/DPO —
more judge reward per nat of distribution shift — and the gap between the
two panels growing where reward hacking sets in.

Usage: python analysis/frontier_plot.py [results_root]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from paths import default_root
from style import INK, KL_COL, METHOD_COLORS, METHOD_LABELS, METHOD_LINESTYLES, apply_style, kl_axis, marker_for


def label_for(run_df: pd.DataFrame) -> str:
    hp = run_df["hyperparam"].iloc[0]
    if pd.isna(hp) or hp is None:
        return ""
    return str(hp).replace("beta", "β=").replace("kl", "c=")


def _errorbars(ax, run_df: pd.DataFrame, ycol: str, color: str) -> None:
    """Thin bars behind the curve: <ycol>_se (math pass@k: SE over prompts,
    eval/passk.py; or across seeds, frontier_math.py) and, where present,
    <KL_COL>_se on x (across seeds)."""
    yerr = run_df[f"{ycol}_se"] if f"{ycol}_se" in run_df.columns else None
    xerr = run_df[f"{KL_COL}_se"] if f"{KL_COL}_se" in run_df.columns else None
    if not any(e is not None and e.notna().any() for e in (yerr, xerr)):
        return
    ax.errorbar(run_df[KL_COL], run_df[ycol], yerr=yerr, xerr=xerr, fmt="none", ecolor=color,
                elinewidth=0.8, alpha=0.45, capsize=0, zorder=1)


def plot_panel(ax, df: pd.DataFrame, ycol: str, title: str) -> None:
    # stagger direct labels so runs that end nearby don't collide
    offsets = [(6, 4), (6, -12), (6, 14), (-2, -16), (6, 24)]
    for i, (run, run_df) in enumerate(df.groupby("run")):
        run_df = run_df.sort_values("step")
        arm = run_df["arm"].iloc[0]
        color = METHOD_COLORS.get(arm, INK)
        if arm == "sft-base":
            ax.plot(run_df[KL_COL], run_df[ycol], marker="*", markersize=14,
                    linestyle="none", color=color, zorder=5)
            _errorbars(ax, run_df, ycol, color)
            continue
        ax.plot(run_df[KL_COL], run_df[ycol], marker=marker_for(arm), color=color, alpha=0.85,
                linestyle=METHOD_LINESTYLES.get(arm, "-"))
        _errorbars(ax, run_df, ycol, color)
        last = run_df.iloc[-1]
        text = label_for(run_df)
        if text:
            ax.annotate(text, (last[KL_COL], last[ycol]), textcoords="offset points",
                        xytext=offsets[i % len(offsets)], fontsize=8, color=INK)
    kl_axis(ax)
    ax.set_ylabel(f"{title}  [z vs π_SFT]")
    ax.set_title(title, fontsize=11)


def main() -> None:
    apply_style()
    results_root = Path(sys.argv[1] if len(sys.argv) > 1 else default_root("helpful"))
    df = pd.read_csv(results_root / "frontier.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    plot_panel(axes[0], df, "reward_judge_z", "Held-out judge reward")
    plot_panel(axes[1], df, "reward_train_rm_z", "Training RM reward (optimized)")

    handles = [
        plt.Line2D([], [], color=METHOD_COLORS[m], marker=marker_for(m),
                   linestyle="none" if m == "sft-base" else METHOD_LINESTYLES.get(m, "-"), label=METHOD_LABELS[m])
        for m in METHOD_COLORS
        if (df["arm"] == m).any()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Preference alignment: reward vs distribution shift", fontsize=13)
    fig.tight_layout()

    out_dir = results_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"frontier.{ext}")
    print(f"wrote {out_dir}/frontier.png|.pdf")


if __name__ == "__main__":
    main()
