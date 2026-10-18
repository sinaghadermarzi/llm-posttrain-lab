"""Diversity / mode-collapse diagnostics vs distribution shift.

Panels vs KL: distinct-2 (falling = repetition), self-BLEU (rising = the N
samples per prompt converging on one mode), token NLL (falling = sharpening),
and — when present (math track) — the pass@k−pass@1 gap (shrinking gap while
pass@1 holds = the model trading solution diversity for single-shot accuracy).

Usage: python analysis/diversity_plot.py [results_root]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from paths import default_root
from style import INK, METHOD_COLORS, apply_style


def main() -> None:
    apply_style()
    results_root = Path(sys.argv[1] if len(sys.argv) > 1 else default_root())
    df = pd.read_csv(results_root / "frontier.csv")

    panels = [
        ("distinct_2", "distinct-2 (unique bigram ratio)", "Repetition"),
        ("self_bleu", "self-BLEU across samples", "Sample convergence"),
        ("mean_token_nll", "token NLL (entropy est., nats)", "Distribution sharpening"),
    ]
    if "pass_gap" in df.columns:
        panels.append(("pass_gap", "pass@k − pass@1", "Diversity dividend"))
    panels = [p for p in panels if p[0] in df.columns]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4))
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, (col, ylabel, title) in zip(axes, panels):
        for arm, arm_df in df.groupby("arm"):
            color = METHOD_COLORS.get(arm, INK)
            marker = "*" if arm == "sft-base" else "o"
            ax.scatter(arm_df["kl_k3"], arm_df[col], color=color, marker=marker,
                       s=90 if arm == "sft-base" else 30, label=arm, alpha=0.85)
        ax.set_xlabel("KL(policy ‖ π_SFT) [nats/seq]")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
    axes[0].legend(fontsize=8)
    fig.tight_layout()

    out_dir = results_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "diversity.png")
    print(f"wrote {out_dir}/diversity.png")


if __name__ == "__main__":
    main()
