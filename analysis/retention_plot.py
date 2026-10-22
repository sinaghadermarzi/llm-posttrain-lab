"""Capability retention vs distribution shift (the alignment tax, quantified).

Left panel: wikitext-2 perplexity vs KL — general-language modeling erosion.
Remaining panels: one per lm-eval benchmark (harness_* columns) vs KL.
Flat curves = preference gained without capability lost; curves bending up
(PPL) or down (accuracy) with KL show what unconstrained drift costs.

Usage: python analysis/retention_plot.py [results_root]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from paths import default_root
from style import INK, KL_COL, METHOD_COLORS, apply_style, kl_axis, marker_for


def main() -> None:
    apply_style()
    results_root = Path(sys.argv[1] if len(sys.argv) > 1 else default_root())
    df = pd.read_csv(results_root / "frontier.csv")

    harness_cols = sorted(c for c in df.columns if c.startswith("harness_"))
    panels = [("ppl_wikitext", "wikitext-2 perplexity ↓", "LM erosion")] + [
        (col, f"{col.removeprefix('harness_')} acc ↑", col.removeprefix("harness_"))
        for col in harness_cols
    ]
    panels = [p for p in panels if p[0] in df.columns and df[p[0]].notna().any()]
    if not panels:
        print("no retention columns in frontier.csv — run eval with retention groups first")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4))
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, (col, ylabel, title) in zip(axes, panels):
        sub = df[df[col].notna()]
        for arm, arm_df in sub.groupby("arm"):
            color = METHOD_COLORS.get(arm, INK)
            ax.scatter(arm_df[KL_COL], arm_df[col], color=color, marker=marker_for(arm),
                       s=90 if arm == "sft-base" else 30, label=arm, alpha=0.85)
        kl_axis(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
    axes[0].legend(fontsize=8)
    fig.tight_layout()

    out_dir = results_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "retention.png")
    print(f"wrote {out_dir}/retention.png")


if __name__ == "__main__":
    main()
