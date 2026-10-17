"""MLflow tracking setup and a jsonl metric sink.

Training metrics go through HF/TRL's built-in MLflowCallback (report_to:
mlflow), configured here via its env vars. Eval metrics are attached to the
training run that produced each checkpoint, so one MLflow run holds a run's
training curves and its frontier points side by side.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from omegaconf import DictConfig

from posttrain_lab.config import CONFIG_DIR
from posttrain_lab.utils.checkpoints import run_arm, run_hyperparam
from posttrain_lab.utils.dotenv import read_dotenv as _read_dotenv


def tracking_enabled(cfg: DictConfig) -> bool:
    report_to = cfg.train.get("report_to")
    if report_to in (None, "none"):
        return False
    return "mlflow" in ([report_to] if isinstance(report_to, str) else list(report_to))


def _tracking_uri(cfg: DictConfig) -> str | None:
    """First non-empty of cfg.mlflow.tracking_uri, the process env, the repo's
    .env. An MLFLOW_PORT from the same env/.env source is appended when the
    URI has no port of its own (a cfg URI is taken as complete)."""
    if cfg.mlflow.get("tracking_uri"):
        return str(cfg.mlflow.tracking_uri)
    for source in (os.environ, _read_dotenv(CONFIG_DIR.parent / ".env")):
        if source.get("MLFLOW_TRACKING_URI"):
            uri, port = source["MLFLOW_TRACKING_URI"], source.get("MLFLOW_PORT")
            break
    else:
        return None
    if port:
        parts = urlsplit(str(uri))
        if parts.scheme in ("http", "https") and parts.port is None:
            uri = urlunsplit(parts._replace(netloc=f"{parts.netloc}:{port}"))
    return uri


def run_tags(cfg: DictConfig) -> dict[str, str]:
    run_dir = Path(cfg.output_dir).name
    return {
        "task": str(cfg.task),
        "stage": str(cfg.stage),
        "run_dir": run_dir,  # eval looks the training run up by this
        "arm": run_arm(run_dir),
        "hyperparam": run_hyperparam(run_dir) or "",
        "seed": str(cfg.seed),
    }


def setup_tracking(cfg: DictConfig) -> None:
    """Point MLflowCallback at cfg.mlflow. Only rank 0 logs, so this is safe to
    call on every process."""
    if not tracking_enabled(cfg):
        return
    uri = _tracking_uri(cfg)
    if not uri:
        # fail before training starts rather than hours in, or silently
        # logging to a local store nobody looks at
        raise SystemExit(
            "[mlflow] no tracking URI: set MLFLOW_TRACKING_URI in .env or the env (or mlflow.tracking_uri=...), "
            "or disable tracking with train.report_to=none"
        )
    os.environ["MLFLOW_TRACKING_URI"] = str(uri)
    os.environ["MLFLOW_EXPERIMENT_NAME"] = str(cfg.mlflow.experiment)
    os.environ["MLFLOW_TAGS"] = json.dumps(run_tags(cfg))
    os.environ.setdefault("MLFLOW_FLATTEN_PARAMS", "TRUE")
    # metrics and params only: checkpoints are ~3GB each and stay on the volume
    os.environ.setdefault("HF_MLFLOW_LOG_ARTIFACTS", "FALSE")


_run_ids: dict[str, str | None] = {}


def _numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """eval/<key> -> float, one level of nesting flattened (harness/<task>);
    names sanitized to MLflow's charset (pass@1 -> pass_1)."""
    flat: dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            flat.update({f"{k}/{sub}": subv for sub, subv in v.items()})
        else:
            flat[k] = v
    return {
        "eval/" + re.sub(r"[^0-9A-Za-z_\-\. :/]", "_", k): float(v)
        for k, v in flat.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def log_eval_metrics(cfg: DictConfig, run_dir: Path, step: int, metrics: dict[str, Any]) -> bool:
    """Log numeric eval results as eval/<key> at the checkpoint's training step
    on the run's MLflow training run. Best effort: results are already on disk,
    so a tracking outage warns instead of failing a long eval. Returns True once
    MLflow has the metrics (or there was nothing to log), False to retry later."""
    if not tracking_enabled(cfg):
        return False
    numeric = _numeric_metrics(metrics)
    if not numeric:
        return True
    if step < 0:
        print(f"[mlflow] {run_dir.name}: no checkpoint step, not logging eval", file=sys.stderr)
        return False
    try:
        from mlflow.entities import Metric
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=_tracking_uri(cfg))
        if not _run_ids.get(run_dir.name):
            exp = client.get_experiment_by_name(str(cfg.mlflow.experiment))
            runs = client.search_runs(
                [exp.experiment_id],
                filter_string=f"tags.run_dir = '{run_dir.name}'",
                order_by=["attributes.start_time DESC"],
                max_results=1,
            ) if exp else []
            _run_ids[run_dir.name] = runs[0].info.run_id if runs else None
        run_id = _run_ids[run_dir.name]
        if run_id is None:
            print(f"[mlflow] no training run tagged run_dir={run_dir.name}, not logging eval", file=sys.stderr)
            return False
        now = int(time.time() * 1000)
        client.log_batch(run_id, metrics=[Metric(k, v, now, step) for k, v in numeric.items()])
        return True
    except Exception as e:  # noqa: BLE001 — tracking must not kill the eval
        print(f"[mlflow] eval logging failed for {run_dir.name} step {step}: {e}", file=sys.stderr)
        return False


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
