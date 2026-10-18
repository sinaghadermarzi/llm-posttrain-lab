"""Default results root for the analysis scripts when none is passed: the same
path the configs resolve (RESULTS_DIR in the env or .env, default results/)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without `pip install -e .`


def default_root(task: str | None = None) -> str:
    """<RESULTS_DIR>/<task> (smoke/<task> under SMOKE=1), or RESULTS_DIR itself."""
    if task is None:
        from posttrain_lab.utils.dotenv import results_dir

        return results_dir()
    from posttrain_lab.config import results_root

    return results_root(task)
