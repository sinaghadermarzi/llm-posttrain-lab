"""What a scale probe (scripts/run_probe.sh) measured, for planning the real run.

    python -m posttrain_lab.probe_report <probe results root> [--since EPOCH] [--gpu-log gpu_util.csv]

Per training run: optimizer steps taken, seconds per step (the trainer's own
train_runtime, so model loading and vLLM start-up are left out), checkpoints
saved and whether it finished (DONE). Then the eval rows written, and the peak
GPU memory per GPU while the probe ran, from the nvidia-smi logger's CSV (make
gpu-log) when it is running: how close each stage came to an OOM. Read-only.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from posttrain_lab.status import checkpoints, read_json, trainer_state
from posttrain_lab.utils.checkpoints import DONE_MARKER
from posttrain_lab.utils.dotenv import REPO_ROOT


def run_rows(root: Path) -> list[dict]:
    rows = []
    runs = root / "runs"
    for run_dir in sorted(p for p in runs.iterdir() if p.is_dir()) if runs.is_dir() else []:
        state = read_json(run_dir / "trainer_state.json") or trainer_state(run_dir) or {}
        runtime = next((e["train_runtime"] for e in reversed(state.get("log_history") or []) if "train_runtime" in e),
                       None)
        steps = state.get("global_step")
        rows.append({
            "run": run_dir.name, "done": (run_dir / DONE_MARKER).exists(), "steps": steps,
            "s_per_step": runtime / steps if runtime and steps else None,
            "checkpoints": len(checkpoints(run_dir)),
        })
    return rows


def eval_rows(root: Path) -> int:
    ev = root / "eval"
    return sum(1 for p in ev.glob("*/*.jsonl") if not p.name.endswith(".samples.jsonl")) if ev.is_dir() else 0


def peak_memory(path: Path, since: float | None) -> dict[int, float]:
    """GPU index -> peak memory.used (MiB) in the logger's rows at or after `since`."""
    peaks: dict[int, float] = {}
    if not path.exists():
        return peaks
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader, [])
        col = next((i for i, h in enumerate(header) if "memory.used" in h), 3)
        for rec in reader:
            if len(rec) <= col or not rec[1].strip().isdigit():
                continue
            try:
                ts = datetime.strptime(rec[0].strip()[:19], "%Y/%m/%d %H:%M:%S").timestamp()
                mib = float(rec[col].strip().split()[0])
            except ValueError:
                continue
            if since is None or ts >= since:
                gpu = int(rec[1])
                peaks[gpu] = max(peaks.get(gpu, 0.0), mib)
    return peaks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m posttrain_lab.probe_report")
    ap.add_argument("root", type=Path)
    ap.add_argument("--since", type=float, help="probe start (epoch seconds) for the GPU-memory peak")
    ap.add_argument("--gpu-log", type=Path, default=REPO_ROOT / "gpu_util.csv")
    args = ap.parse_args(argv)

    rows = run_rows(args.root)
    print(f"{'run':34s} {'done':5s} {'steps':>5s} {'s/step':>7s} {'ckpts':>5s}")
    for r in rows:
        sps = f"{r['s_per_step']:.1f}" if r["s_per_step"] is not None else "?"
        print(f"{r['run']:34s} {'yes' if r['done'] else 'NO':5s} {r['steps'] if r['steps'] is not None else '?':>5} "
              f"{sps:>7s} {r['checkpoints']:>5d}")
    print(f"eval rows written: {eval_rows(args.root)}")
    peaks = peak_memory(args.gpu_log, args.since)
    if peaks:
        print("peak GPU memory during the probe: "
              + "  ".join(f"GPU {i}: {mib / 1024:.1f} GiB" for i, mib in sorted(peaks.items())))
    else:
        print(f"peak GPU memory: no logger rows in {args.gpu_log} for this probe (start one with make gpu-log)")
    return 0 if rows and all(r["done"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
