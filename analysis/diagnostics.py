"""Secondary evidence, one figure each:

  dpo_logps.png    — DPO's chosen AND rejected log-probs during training (the
                     "squeezing effect": both often decline; DPO moves mass off
                     the whole preference distribution, unlike RL). Read from
                     the trainer_state.json log history in each DPO run.
  diagnostics.png  — 3 panels against KL: response length (length hacking),
                     policy token entropy (mode collapse), and the
                     train-RM-minus-judge z-gap (reward hacking).

Usage: python analysis/diagnostics.py [results_root]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from paths import default_root
from style import BLUE_RAMP, INK, METHOD_COLORS, apply_style


def dpo_logp_trajectories(results_root: Path, out_dir: Path) -> None:
    runs = sorted(results_root.glob("runs/dpo_*_seed*"))
    if not runs:
        print("no DPO runs found, skipping dpo_logps figure")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, run in enumerate(runs):
        states = sorted(run.glob("checkpoint-*/trainer_state.json"))
        if not states:
            continue
        history = json.loads(states[-1].read_text())["log_history"]
        rows = [h for h in history if "logps/chosen" in h]
        if not rows:
            continue
        steps = [h["step"] for h in rows]
        color = BLUE_RAMP[i % len(BLUE_RAMP)]
        beta = run.name.split("_")[1].replace("beta", "β=")
        ax.plot(steps, [h["logps/chosen"] for h in rows], color=color, label=f"{beta} chosen")
        ax.plot(steps, [h["logps/rejected"] for h in rows], color=color, linestyle="--", alpha=0.7)
        ax.annotate(beta, (steps[-1], rows[-1]["logps/chosen"]),
                    textcoords="offset points", xytext=(5, 0), fontsize=8, color=INK)
    ax.set_xlabel("training step")
    ax.set_ylabel("mean log-prob of completion")
    ax.set_title("DPO: chosen (solid) and rejected (dashed) log-probs both drift down", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(out_dir / "dpo_logps.png")
    print(f"wrote {out_dir}/dpo_logps.png")


def kl_diagnostics(results_root: Path, out_dir: Path) -> None:
    df = pd.read_csv(results_root / "frontier.csv")
    panels = [
        ("mean_response_tokens", "response length (tokens)", "Length vs KL"),
        ("mean_token_nll", "token NLL (entropy est., nats)", "Entropy vs KL"),
        ("rm_gap_z", "train-RM z − judge z", "Reward hacking gap vs KL"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
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
