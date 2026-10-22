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
    "dpo-nll": "#e87ba4",   # magenta; CVD ΔE 6.1 vs aqua, so it also gets a dashed line / diamond (below)
    "ppo": "#eda100",       # yellow
    "grpo": "#008300",      # green
}
METHOD_LABELS = {
    "sft-base": r"$\pi_{SFT}$ (reference)",
    "sft": "SFT (imitation)",
    "dpo": "DPO",
    "dpo-nll": "DPO + NLL",
    "ppo": "PPO",
    "grpo": "GRPO",
}
# secondary encoding where a hue alone sits in the CVD floor band (validate_palette:
# all-pairs worst dpo-nll vs dpo, deutan ΔE 6.1): never identify an arm by color only
METHOD_LINESTYLES = {"dpo-nll": "--"}
METHOD_MARKERS = {"sft-base": "*", "dpo-nll": "D"}


def marker_for(arm: str) -> str:
    return METHOD_MARKERS.get(arm, "o")


# x axis of every KL plot. kl_k1 (mean of log pi/pi_ref over the policy's own
# samples) stays a usable estimate far from pi_SFT; kl_k3 sums exp(log ratio)
# per token and explodes there (the 5e-6 DPO run: 1e21+ nats from a few tokens
# the reference likes and the policy almost never samples). symlog: GRPO's
# 0.5-6 nats and a collapsed run's hundreds on one readable axis.
KL_COL = "kl_k1"
KL_LABEL = "KL(policy ‖ π_SFT)  [nats/sequence, k1; symlog]"


def kl_axis(ax) -> None:
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel(KL_LABEL)


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
