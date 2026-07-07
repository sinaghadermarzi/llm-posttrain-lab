"""The upgraded DPO-failure-mode figure: held-out chosen/rejected log-probs
per checkpoint, for EVERY arm on equal footing.

One panel per arm: mean logp of held-out chosen (solid) and rejected
(dashed) completions vs training step, one curve pair per run. The
"squeezing effect" reads as both lines falling while their gap widens —
the question this figure answers is whether that is DPO-specific or shared
by the RL arms. (Training-log-based DPO logps remain in diagnostics.py;
this eval-time version is the cross-method comparison.)

Usage: python analysis/pref_logps_plot.py [results_root]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from style import METHOD_COLORS, METHOD_LABELS, apply_style


def main() -> None:
    apply_style()
    results_root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    df = pd.read_csv(results_root / "frontier.csv")
    if "logp_chosen" not in df.columns:
        print("no pref_logps columns in frontier.csv — run eval with the pref_logps group first")
        return

    arms = [a for a in METHOD_COLORS if a != "sft-base" and (df["arm"] == a).any()]
    baseline = df[df["arm"] == "sft-base"]

    fig, axes = plt.subplots(1, len(arms), figsize=(4.3 * len(arms), 4), sharey=True)
    axes = [axes] if len(arms) == 1 else list(axes)
    for ax, arm in zip(axes, arms):
        color = METHOD_COLORS[arm]
        for run, run_df in df[df["arm"] == arm].groupby("run"):
            run_df = run_df.sort_values("step")
            ax.plot(run_df["step"], run_df["logp_chosen"], color=color, alpha=0.85)
            ax.plot(run_df["step"], run_df["logp_rejected"], color=color, alpha=0.6, linestyle="--")
        if not baseline.empty:
            ax.axhline(baseline["logp_chosen"].iloc[0], color=METHOD_COLORS["sft-base"],
                       linewidth=1, linestyle=":")
            ax.axhline(baseline["logp_rejected"].iloc[0], color=METHOD_COLORS["sft-base"],
                       linewidth=1, linestyle=":", alpha=0.6)
        ax.set_xlabel("training step")
        ax.set_title(METHOD_LABELS.get(arm, arm), fontsize=11)
    axes[0].set_ylabel("held-out mean log-prob\n(chosen solid, rejected dashed, π_SFT dotted)")
    fig.suptitle("Do methods move probability mass off the preferred responses?", fontsize=13)
    fig.tight_layout()

    out_dir = results_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "pref_logps.png")
    print(f"wrote {out_dir}/pref_logps.png")


if __name__ == "__main__":
    main()
