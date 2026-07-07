"""Shared figure style: validated categorical palette + recessive chrome.

Palette slots follow the fixed order blue/aqua/yellow/green (CVD-validated as a
set; aqua and yellow are sub-3:1 on white, so every figure direct-labels its
series rather than relying on color alone).
"""

import matplotlib as mpl

METHOD_COLORS = {
    "sft-base": "#52514e",  # reference point, not a method — secondary ink
    "sft": "#2a78d6",       # blue
    "dpo": "#1baf7a",       # aqua
    "ppo": "#eda100",       # yellow
    "grpo": "#008300",      # green
}
METHOD_LABELS = {
    "sft-base": r"$\pi_{SFT}$ (reference)",
    "sft": "SFT (imitation)",
    "dpo": "DPO",
    "ppo": "PPO",
    "grpo": "GRPO",
}
# ordinal blue ramp for within-method hyperparameter series (e.g. DPO betas)
BLUE_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "font.family": "sans-serif",
            "font.size": 10,
            "legend.frameon": False,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )
