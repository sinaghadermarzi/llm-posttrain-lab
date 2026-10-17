#!/usr/bin/env python
"""Download every Hugging Face asset the pipeline loads into one HF_HOME
directory, on a machine WITH internet (or with HF_ENDPOINT pointing at a
mirror), then verify it offline. Copy the directory to a PVC and point the
cluster's HF_HOME at it (README: "Offline cluster").

    python scripts/prefetch_assets.py /data/hf_home              # everything
    python scripts/prefetch_assets.py /data/hf_home --smoke      # just what `make smoke` needs
    python scripts/prefetch_assets.py /data/hf_home --skip judge # all but the 8B judge (~16 GB)
    python scripts/prefetch_assets.py --list
    make prefetch PREFETCH_DIR=/data/hf_home PREFETCH_ARGS=--smoke

Models: full snapshot_download into HF_HOME/hub. Datasets: load_dataset of
the exact (repo, config) the code uses, which writes the prepared Arrow cache
(HF_HOME/datasets) that offline loading needs — every split of the config.
lm-eval tasks: built through lm-eval itself, so its dataset cache is warmed
the way simple_evaluate will read it.

Afterwards it re-runs itself with HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 and
the same selection: preflight's cache report plus a real load of each asset.
Needs the HF stack from requirements.txt (datasets, transformers, lm-eval,
omegaconf); no GPU. Re-running is cheap: finished downloads are skipped.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))  # works without `pip install -e .`

OFFLINE_FLAGS = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
LEGACY_CACHES = ("HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")  # superseded by HF_HUB_CACHE


def pin_hf_env(hf_home: Path, offline: bool) -> None:
    """Point every HF cache at hf_home and set the offline flags, overriding
    the env and .env alike. Values must be set, not just removed: `import
    posttrain_lab` fills unset HF_* keys from .env, e.g. HF_HUB_OFFLINE=1 from
    a cluster-style .env, which would make this run offline."""
    pinned = {"HF_HOME": str(hf_home), "HF_HUB_CACHE": str(hf_home / "hub"),
              "HF_DATASETS_CACHE": str(hf_home / "datasets"),
              **{flag: "1" if offline else "0" for flag in OFFLINE_FLAGS}}
    for var, val in pinned.items():
        if os.environ.get(var, val) != val and not offline:
            print(f"note: overriding {var}={os.environ[var]} with {val} for this run")
        os.environ[var] = val
    for var in LEGACY_CACHES:
        if os.environ.pop(var, None) is not None and not offline:
            print(f"note: ignoring {var} from the environment for this run")


def select(assets, args):
    names = {a.name for a in assets}
    wanted = set(args.only.split(",")) if args.only else names
    skipped = set(args.skip.split(",")) if args.skip else set()
    unknown = (wanted | skipped) - names
    if unknown:
        sys.exit(f"unknown asset(s) {sorted(unknown)}; choose from {sorted(names)} (see --list)")
    return [a for a in assets
            if a.name in wanted and a.name not in skipped and (a.smoke or not args.smoke)]


def fetch(assets) -> None:
    from datasets import load_dataset
    from huggingface_hub import constants, snapshot_download

    print(f"HF_HOME={constants.HF_HOME}  HF_ENDPOINT={constants.ENDPOINT}\n")
    task_manager = None
    for a in assets:
        label = f"{a.kind} {a.repo}" + (f" [{a.config}]" if a.config else "")
        print(f"--- {a.name}: {label} ({a.why})", flush=True)
        start = time.perf_counter()
        if Path(a.repo).exists():
            print("    local path, nothing to fetch")
            continue
        if a.kind == "model":
            print(f"    {snapshot_download(a.repo)}")
        elif a.harness_task:
            from lm_eval.tasks import TaskManager

            task_manager = task_manager or TaskManager()
            task = task_manager.load([a.harness_task])["tasks"][a.harness_task]
            print(f"    lm-eval {a.harness_task}: " + ", ".join(f"{k}={len(v)}" for k, v in task.dataset.items()))
        else:
            ds = load_dataset(a.repo, a.config)
            print("    " + ", ".join(f"{k}={len(v)}" for k, v in ds.items()))
        print(f"    ({time.perf_counter() - start:.0f}s)", flush=True)


def verify(assets) -> bool:
    from posttrain_lab.preflight import _hf_env, hf_cache_report, hf_load_report

    ok = True
    for title, results in [("cache", _hf_env() + hf_cache_report(assets)), ("load", hf_load_report(assets))]:
        print(f"[offline {title}]")
        for level, msg in results:
            print(f"  {level:5s} {msg}")
            ok &= level != "FAIL"
    return ok


def du(path: Path) -> float:
    return sum(p.lstat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink()) / 2**30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("hf_home", nargs="?", type=Path, help="target HF_HOME directory (created if missing)")
    parser.add_argument("--smoke", action="store_true", help="only the assets `make smoke` needs")
    parser.add_argument("--only", help="comma-separated asset names (see --list)")
    parser.add_argument("--skip", help="comma-separated asset names to leave out, e.g. judge")
    parser.add_argument("--list", action="store_true", help="list assets and exit")
    parser.add_argument("--no-verify", action="store_true", help="skip the offline verification pass")
    parser.add_argument("--verify-only", action="store_true", help=argparse.SUPPRESS)  # the offline re-run
    args = parser.parse_args()
    if not args.list and args.hf_home is None:
        parser.error("hf_home is required (unless --list)")

    if args.hf_home is not None:
        hf_home = args.hf_home.expanduser().resolve()
        hf_home.mkdir(parents=True, exist_ok=True)
        pin_hf_env(hf_home, offline=args.verify_only)

    from posttrain_lab.hub_assets import hub_assets  # after pinning: HF reads the env at import

    assets = select(hub_assets(), args)
    if args.list:
        for a in hub_assets():
            tags = ",".join(t for t, on in [("smoke", a.smoke), ("selected", a in assets)] if on)
            print(f"  {a.name:14s} {a.kind:8s} {a.repo + (f' [{a.config}]' if a.config else ''):52s} {tags}")
        return
    if args.verify_only:
        sys.exit(0 if verify(assets) else 1)

    fetch(assets)
    print(f"\n{hf_home}: {du(hf_home):.1f} GB")
    if args.no_verify:
        return
    print("\nverifying offline (HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1) ...\n", flush=True)
    env = {**os.environ, **{flag: "1" for flag in OFFLINE_FLAGS}}
    cmd = [sys.executable, __file__, str(hf_home), "--verify-only"] + (["--smoke"] if args.smoke else [])
    for opt in ("only", "skip"):
        if getattr(args, opt):
            cmd += [f"--{opt}", getattr(args, opt)]
    if subprocess.run(cmd, env=env).returncode != 0:
        sys.exit("\noffline verification FAILED (see above)")
    print(f"\nall {len(assets)} selected assets load offline. Copy {hf_home} to the PVC, keeping "
          "symlinks (rsync -a / tar), and set HF_HOME to it on the cluster — see README.")


if __name__ == "__main__":
    main()
