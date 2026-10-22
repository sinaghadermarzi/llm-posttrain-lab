"""Secondary evidence, one figure each:

  dpo_logps.png    — DPO's chosen AND rejected log-probs during training (the
                     "squeezing effect": both often decline; DPO moves mass off
                     the whole preference distribution, unlike RL), one panel
                     per DPO-trained arm (DPO, DPO + NLL: the NLL term should
                     hold the chosen line up). Read from the
                     trainer_state.json log history in each run. With several
                     training seeds every seed keeps its own line (one color
                     and legend entry per β): each seed trains on its own
                     pairs, so its runs log at different steps and run for a
                     different number of them, and a mean would have to
                     interpolate between logged steps.
  diagnostics.png  — panels against KL: response length (length hacking),
                     policy token entropy (mode collapse), and (helpful: two
                     graders) the train-RM-minus-judge z-gap (reward hacking).
                     With several training seeds: seed means per (arm,
                     hyperparam, checkpoint index) with across-seed SE bars
                     on both axes (aggregate.py:seed_view).

Usage: python analysis/diagnostics.py [results_root]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from aggregate import seed_view, seeds_caption
from paths import default_root  # also puts src/ on sys.path
from posttrain_lab.utils.checkpoints import list_checkpoints
from style import BLUE_RAMP, INK, KL_COL, METHOD_COLORS, METHOD_LABELS, apply_style, errorbars, kl_axis, marker_for


DPO_ARMS = ["dpo", "dpo-nll"]  # arms trained by train/dpo.py (run dirs <arm>_beta*_seed*)


def dpo_logp_trajectories(results_root: Path, out_dir: Path) -> None:
    arms = [(arm, sorted(results_root.glob(f"runs/{arm}_*_seed*"))) for arm in DPO_ARMS]
    arms = [(arm, runs) for arm, runs in arms if runs]
    if not arms:
        print("no DPO runs found, skipping dpo_logps figure")
        return
    fig, axes = plt.subplots(1, len(arms), figsize=(6.5 * len(arms), 4.5), sharey=True, squeeze=False)
    for ax, (arm, runs) in zip(axes[0], arms):
        _dpo_logps_panel(ax, arm, runs)
    fig.tight_layout()
    if not any(ax.get_lines() for ax in axes[0]):
        print("no logps/chosen in any DPO run's trainer_state.json; dpo_logps figure is empty")
    fig.savefig(out_dir / "dpo_logps.png")
    print(f"wrote {out_dir}/dpo_logps.png")


def _dpo_logps_panel(ax, arm: str, runs: list[Path]) -> None:
    # color (and legend entry, label) per beta, so the seeds of one beta read as one series
    betas = list(dict.fromkeys(run.name.split("_")[1] for run in runs))
    labeled: set[str] = set()
    for run in runs:
        # numeric step order: as strings, checkpoint-800 sorts after checkpoint-1000
        states = [c / "trainer_state.json" for c in list_checkpoints(run) if (c / "trainer_state.json").exists()]
        if not states:
            continue
        history = json.loads(states[-1].read_text())["log_history"]  # latest = fullest history
        rows = [h for h in history if "logps/chosen" in h]
        if not rows:
            continue
        steps = [h["step"] for h in rows]
        key = run.name.split("_")[1]
        color = BLUE_RAMP[betas.index(key) % len(BLUE_RAMP)]
        beta = key.replace("beta", "β=")
        first = key not in labeled
        labeled.add(key)
        # markers so a run with a single logged step (e.g. smoke) still shows
        ax.plot(steps, [h["logps/chosen"] for h in rows], color=color, marker="o", markersize=3,
                label=f"{beta} chosen" if first else None)
        ax.plot(steps, [h["logps/rejected"] for h in rows], color=color, marker="o", markersize=3,
                linestyle="--", alpha=0.7)
        if first:
            ax.annotate(beta, (steps[-1], rows[-1]["logps/chosen"]),
                        textcoords="offset points", xytext=(5, 0), fontsize=8, color=INK)
    ax.set_xlabel("training step")
    ax.set_ylabel("mean log-prob of completion")
    title = f"{METHOD_LABELS.get(arm, arm)}: chosen (solid) and rejected (dashed) log-probs"
    if len({run.name.rsplit("_seed", 1)[-1] for run in runs}) > 1:
        title += "\none line per training seed"
    ax.set_title(title, fontsize=11)
    if ax.get_lines():
        ax.legend(fontsize=8, ncol=2)


def kl_diagnostics(results_root: Path, out_dir: Path) -> None:
    df, seeds = seed_view(pd.read_csv(results_root / "frontier.csv"))
    panels = [
        ("mean_response_tokens", "response length (tokens)", "Length vs KL"),
        ("mean_token_nll", "token NLL (entropy est., nats)", "Entropy vs KL"),
        ("rm_gap_z", "train-RM z − judge z", "Reward hacking gap vs KL"),  # helpful only (two graders)
    ]
    panels = [p for p in panels if p[0] in df.columns and df[p[0]].notna().any()]
    if not panels:
        print("no length/entropy/reward-gap columns in frontier.csv, skipping diagnostics figure")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4), squeeze=False)
    axes = list(axes[0])
    for ax, (col, ylabel, title) in zip(axes, panels):
        for arm, arm_df in df.groupby("arm"):
            color = METHOD_COLORS.get(arm, INK)
            ax.scatter(arm_df[KL_COL], arm_df[col], color=color, marker=marker_for(arm),
                       s=90 if arm == "sft-base" else 30, label=arm, alpha=0.85)
            if len(seeds) > 1:
                errorbars(ax, arm_df, col, color)
        kl_axis(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
    axes[0].legend(fontsize=8)
    if len(seeds) > 1:
        fig.suptitle(seeds_caption(seeds), fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostics.png")
    print(f"wrote {out_dir}/diagnostics.png")


def main() -> None:
    apply_style()
    results_root = Path(sys.argv[1] if len(sys.argv) > 1 else default_root("helpful"))
    out_dir = results_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    dpo_logp_trajectories(results_root, out_dir)
    if (results_root / "frontier.csv").exists():
        kl_diagnostics(results_root, out_dir)
    else:
        print("frontier.csv missing — run aggregate.py first")


if __name__ == "__main__":
    main()
