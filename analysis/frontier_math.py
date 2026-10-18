"""Math-track headline figure: accuracy vs KL(policy || pi_SFT-math).

Left: pass@1 (single-sample accuracy) vs sequence KL. Right: pass@k (any of
k samples correct). The RL story to look for: GRPO pushes pass@1 up at
bounded KL while pass@k holds (diversity preserved); imitation-style arms
buy pass@1 by collapsing onto fewer solution modes, so pass@k stalls or
falls even as pass@1 rises.

Usage: python analysis/frontier_math.py [results_root]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from paths import default_root
from frontier_plot import plot_panel
from style import METHOD_COLORS, METHOD_LABELS, apply_style


def main() -> None:
    apply_style()
    results_root = Path(sys.argv[1] if len(sys.argv) > 1 else default_root("math"))
    df = pd.read_csv(results_root / "frontier.csv")

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
        plt.Line2D([], [], color=METHOD_COLORS[m], marker="*" if m == "sft-base" else "o",
                   linestyle="none" if m == "sft-base" else "-", label=METHOD_LABELS[m])
        for m in METHOD_COLORS
        if (df["arm"] == m).any()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("GSM8K (verifiable rewards): accuracy vs distribution shift", fontsize=13)
    fig.tight_layout()

    out_dir = results_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"frontier_math.{ext}")
    print(f"wrote {out_dir}/frontier_math.png|.pdf")


if __name__ == "__main__":
    main()
