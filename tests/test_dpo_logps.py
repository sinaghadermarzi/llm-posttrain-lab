"""analysis/diagnostics.py's dpo_logps figure must read the latest checkpoint
by step number (as strings, checkpoint-800 sorts after checkpoint-1000) and
still draw a run that logged a single step (smoke: logging_steps=10, max_steps=10)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import diagnostics  # noqa: E402


def _write_run(root: Path, name: str, checkpoints: dict[int, list[int]]) -> None:
    """checkpoints: step -> the steps that checkpoint's log_history has logps for."""
    for ckpt, logged in checkpoints.items():
        d = root / "runs" / name / f"checkpoint-{ckpt}"
        d.mkdir(parents=True)
        history = [{"step": s, "logps/chosen": -100.0 - s / 100, "logps/rejected": -90.0 - s / 50} for s in logged]
        (d / "trainer_state.json").write_text(json.dumps({"log_history": history}))


def _plotted(tmp_path: Path) -> list[tuple[list[float], str]]:
    plt.close("all")
    out = tmp_path / "figures"
    out.mkdir()
    diagnostics.dpo_logp_trajectories(tmp_path, out)
    assert (out / "dpo_logps.png").exists()
    return [(list(line.get_xdata()), line.get_marker()) for line in plt.gcf().axes[0].get_lines()]


def test_latest_checkpoint_by_step_number(tmp_path: Path) -> None:
    every_100 = list(range(100, 1001, 100))
    _write_run(tmp_path, "dpo_beta0.1_seed17", {
        200: every_100[:2], 800: every_100[:8], 1000: every_100,  # "800" > "1000" as strings
    })
    lines = _plotted(tmp_path)
    assert len(lines) == 2  # chosen + rejected
    for xs, _ in lines:
        assert xs == every_100


def test_single_logged_step_is_drawn(tmp_path: Path) -> None:
    _write_run(tmp_path, "dpo_beta0.1_seed17", {5: [], 10: [10]})  # the smoke layout
    lines = _plotted(tmp_path)
    assert [xs for xs, _ in lines] == [[10], [10]]
    assert all(marker == "o" for _, marker in lines)  # a one-point line needs a marker to be visible
