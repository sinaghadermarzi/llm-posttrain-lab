"""Pipeline status: where each track is, what is running, and what needs attention.

    python -m posttrain_lab.status [math|helpful|all] [--smoke] [--json]
                                   [--watch SEC] [--log PATH] [--no-color]
    make status [TASK=math]

For every stage of a track (in run_all_<task>.sh order) it shows whether the
run is DONE, RUNNING, STOPPED (output exists but no DONE marker and nothing
running: crashed or killed) or pending, and for each run:

  - checkpoints written vs expected, the latest trainer_state step/metrics
  - for running jobs: the live progress bar and last logged metrics from the
    job's log (read from the process's stdout, else the known log names),
    elapsed/remaining time, the GPU it is on
  - stage-specific checks: pair yield (m2), the RM accuracy gate (stage 2),
    GRPO/DPO health signals, eval coverage per metric group

plus the GPUs (now, and the history in gpu_util.csv if the logger is
running), disk space, errors found in the logs, and an "attention" list of
everything that looks wrong. Expected runs come from the stage configs and
sweeps.yaml, so this follows the configs when they change.

Read-only and quick: it never imports torch, datasets or vLLM. --json prints
the same information as one JSON document.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from posttrain_lab.config import CONFIG_DIR, load_config
from posttrain_lab.eval.groups import SENTINELS
from posttrain_lab.utils.checkpoints import DONE_MARKER, discover_all_checkpoints
from posttrain_lab.utils.dotenv import REPO_ROOT, results_dir, results_dir_source

# ---------------------------------------------------------------------------
# what each track runs, in run_all_<task>.sh order


@dataclass(frozen=True)
class Stage:
    key: str                  # m1, m3b, 2, ...
    title: str
    kind: str                 # train | pairs | rm | eval | figures
    yaml: str | None = None   # stage config, relative to configs/
    sweep: str | None = None  # sweeps.yaml section: one run per value
    log: str | None = None    # per-run/per-shard log name the launcher writes ({ov} / {i})
    gpus: str = "4"           # how the launcher places it: "4" (one job, all GPUs) | "1 each" | "1"


TRACKS: dict[str, dict] = {
    "math": {
        "driver": "run_all_math.sh",
        "logs": ["math-all.log", "all.log", "nohup.out"],
        "stages": [
            Stage("m1", "pi_SFT-math (gold CoT)", "train", "math/m1_sft_gold.yaml"),
            Stage("m2", "self-generated pairs", "pairs", "math/m2_gen_pairs.yaml", gpus="1"),
            Stage("m3a", "SFT arm", "train", "math/m3_arm_sft.yaml"),
            Stage("m3b", "DPO sweep", "train", "math/m3_arm_dpo.yaml", "dpo", "logs_math_dpo_{ov}.log", "1 each"),
            Stage("m3d", "GRPO sweep", "train", "math/m3_arm_grpo.yaml", "grpo"),
            Stage("m4", "eval", "eval", log="logs_math_eval_shard{i}.log", gpus="1 each"),
            Stage("m5", "aggregate + figures", "figures", gpus="0"),
        ],
    },
    "helpful": {
        "driver": "run_all_helpful.sh",
        "logs": ["helpful-all.log", "all.log", "nohup.out"],
        "stages": [
            Stage("1", "pi_SFT", "train", "helpful/stage1_sft_base.yaml"),
            Stage("2", "reward model (accuracy gate)", "rm", "helpful/stage2_rm.yaml"),
            Stage("3a", "SFT arm", "train", "helpful/arm_sft.yaml"),
            Stage("3b", "DPO sweep", "train", "helpful/arm_dpo.yaml", "dpo", "logs_dpo_{ov}.log", "1 each"),
            Stage("3c", "PPO sweep", "train", "helpful/arm_ppo.yaml", "ppo"),
            Stage("3d", "GRPO sweep", "train", "helpful/arm_grpo.yaml", "grpo"),
            Stage("4", "eval", "eval", log="logs_eval_shard{i}.log", gpus="1 each"),
            Stage("5", "aggregate + figures", "figures", gpus="0"),
        ],
    },
}

# training metrics worth showing per trainer, in display order (TRL 1.7 names)
METRICS = {
    "sft": ["loss", "mean_token_accuracy", "grad_norm", "learning_rate"],
    "dpo": ["loss", "rewards/accuracies", "rewards/margins", "rewards/chosen", "rewards/rejected"],
    "grpo": ["reward", "reward_std", "kl", "frac_reward_zero_std", "completions/clipped_ratio",
             "completions/mean_length"],
    "ppo": ["objective/rlhf_reward", "objective/scores", "objective/kl", "policy/approxkl_avg",
            "val/clipfrac_avg", "loss/policy_avg"],
    "rm": ["loss", "accuracy", "margin"],
}
# the stage config each entry point uses when none is given (parse_entry defaults)
ENTRY_DEFAULTS = {
    "train.sft": "helpful/stage1_sft_base.yaml", "train.rm": "helpful/stage2_rm.yaml",
    "train.dpo": "helpful/arm_dpo.yaml", "train.ppo": "helpful/arm_ppo.yaml",
    "train.grpo": "helpful/arm_grpo.yaml", "gen_pairs": "math/m2_gen_pairs.yaml",
}
JOB_RE = re.compile(r"(?:^|\s)(?:-m|--module)\s+posttrain_lab\.(train\.\w+|gen_pairs|eval\.run_eval)\b(.*)$")
LAUNCHER_HINTS = ("accelerate launch", "accelerate/commands", "torch.distributed.run", "torchrun")
QUIET_MINUTES = 30            # a running job whose log is quieter than this gets flagged
FREE_GB_WARN = 50

# ---------------------------------------------------------------------------
# small helpers


def _now() -> float:
    return time.time()


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "?"
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}d{h:02d}h"
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def fmt_time(ts: float | None) -> str:
    if ts is None:
        return "?"
    t = datetime.fromtimestamp(ts)
    return t.strftime("%H:%M") if t.date() == datetime.now().date() else t.strftime("%m-%d %H:%M")


def fmt_num(value) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if value != value:  # NaN
        return "nan"
    if isinstance(value, int) or float(value).is_integer() and abs(value) >= 1000:
        return f"{int(value):,}"
    if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
        return f"{value:.2e}"
    return f"{value:.4g}"


def fmt_bytes(n: float | None) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return "?"


def parse_clock(text: str) -> float | None:
    """tqdm / ps clock: [[dd-]hh:]mm:ss -> seconds."""
    text = text.strip()
    if not text or "?" in text:
        return None
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    try:
        parts = [int(p) for p in text.split(":")]
    except ValueError:
        return None
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + p
    return days * 86400 + seconds


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def last_jsonl(path: Path) -> dict | None:
    try:
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        return json.loads(lines[-1]) if lines else None
    except (OSError, ValueError):
        return None


def tail_text(path: Path, max_bytes: int = 4_000_000) -> str:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def ckpt_step(path: Path) -> int:
    m = re.search(r"checkpoint-(\d+)$", path.name)
    return int(m.group(1)) if m else -1


def checkpoints(run_dir: Path) -> list[Path]:
    return sorted((p for p in run_dir.glob("checkpoint-*") if p.is_dir()), key=ckpt_step)


def log_name(template: str, override: str = "", shard: int = 0) -> str:
    """The name scripts/*/m3b_*.sh gives a sweep run's log: ${ov//[.=\\/]/_}."""
    return template.format(ov=re.sub(r"[.=/]", "_", override), i=shard)


# ---------------------------------------------------------------------------
# log parsing (pure functions; tested in tests/test_status.py)

BAR_RE = re.compile(
    r"^(?P<label>[^|\n]*?)\s*(?P<pct>\d{1,3})%\|[^|]*\|\s*(?P<n>\d+)/(?P<total>\d+)\s*"
    r"\[(?P<elapsed>[\d:-]+)<(?P<remaining>[\d:?-]+)(?:,\s*(?P<rate>[^\]]*))?\]"
)
ERROR_RE = re.compile(
    r"Traceback \(most recent call last\)|\b[A-Z]\w*(?:Error|Exception)\b:|\bFAILED\b|FATAL|"
    r"CUDA out of memory|ChildFailedError|NCCL error|\bKilled\b|Segmentation fault|core dumped"
)
BENIGN_RE = re.compile(r"WARNING|warnings\.warn|UserWarning|FutureWarning|DeprecationWarning|deep_gemm|libnvrtc")


def log_lines(text: str) -> list[str]:
    """tqdm redraws with \\r; split on it too so each redraw is a line."""
    return [ln for ln in text.replace("\r", "\n").split("\n") if ln.strip()]


def last_section(text: str) -> tuple[str | None, str]:
    """(last '=== ...' stage header, the log after it) of a run_all_*.sh log."""
    lines = text.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("=== "):
            return lines[i].strip(), "\n".join(lines[i + 1:])
    return None, text


def last_progress(lines: list[str], label: str | None = "") -> dict | None:
    """Latest tqdm bar. label="" -> unlabelled bars only (the Trainer's), None ->
    any bar, else bars whose label contains it (e.g. 'Processed prompts')."""
    for line in reversed(lines):
        m = BAR_RE.search(line.strip())
        if not m:
            continue
        lab = m["label"].strip().rstrip(":").strip()
        if label == "" and lab:
            continue
        if label and label not in lab:
            continue
        n, total = int(m["n"]), int(m["total"])
        return {
            "label": lab, "n": n, "total": total, "pct": 100.0 * n / total if total else 0.0,
            "elapsed_s": parse_clock(m["elapsed"]), "remaining_s": parse_clock(m["remaining"]),
            "rate": (m["rate"] or "").strip() or None,
        }
    return None


def last_metrics(lines: list[str]) -> dict | None:
    """Latest Trainer log dict ({'loss': '0.25', ...}); values made numeric."""
    for line in reversed(lines):
        s = line.strip()
        i = s.find("{'")
        if i < 0 or not s.endswith("}"):
            continue
        try:
            raw = ast.literal_eval(s[i:])
        except (ValueError, SyntaxError):
            continue
        if not isinstance(raw, dict) or "train_runtime" in raw:
            continue
        out = {}
        for k, v in raw.items():
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
        return out
    return None


def find_errors(lines: list[str]) -> list[str]:
    hits = []
    for line in lines:
        if ERROR_RE.search(line) and not BENIGN_RE.search(line):
            hits.append(line.strip()[:240])
    return hits


def summarize_errors(lines: list[str]) -> dict | None:
    """Count + the most informative lines: the last exception message and the
    last few distinct error lines."""
    hits = find_errors(lines)
    if not hits:
        return None
    exc = [h for h in hits if re.search(r"\b[A-Z]\w*(?:Error|Exception)\b:", h)]
    tail: list[str] = []
    for h in reversed(hits):
        if h not in tail:
            tail.append(h)
        if len(tail) == 3:
            break
    return {"count": len(hits), "last_exception": exc[-1] if exc else None, "last": list(reversed(tail))}


# ---------------------------------------------------------------------------
# processes


@dataclass
class Job:
    module: str               # train.sft, gen_pairs, eval.run_eval
    args: list[str]
    pids: list[int] = field(default_factory=list)
    ranks: int = 0
    elapsed_s: float | None = None
    cpu: float = 0.0
    gpus: str | None = None   # CUDA_VISIBLE_DEVICES
    log: str | None = None    # stdout of the launcher / first worker
    smoke: bool = False

    @property
    def yaml(self) -> str | None:
        return next((a for a in self.args if a.endswith((".yaml", ".yml"))), None)

    @property
    def overrides(self) -> list[str]:
        return [a for a in self.args if "=" in a and not a.startswith("-")]

    @property
    def shard(self) -> str | None:
        if "--shard" in self.args:
            i = self.args.index("--shard")
            return self.args[i + 1] if i + 1 < len(self.args) else None
        return None


def _proc_environ(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return {}
    env = {}
    for item in raw.split(b"\0"):
        if b"=" in item:
            k, v = item.split(b"=", 1)
            env[k.decode(errors="replace")] = v.decode(errors="replace")
    return env


def _proc_stdout(pid: int) -> str | None:
    try:
        target = os.readlink(f"/proc/{pid}/fd/1")
    except OSError:
        return None
    return target if target.startswith("/") and not target.startswith("/dev/") else None


def list_processes() -> list[dict]:
    try:
        out = subprocess.run(["ps", "-eo", "pid=,ppid=,etime=,pcpu=,args="], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    procs = []
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        try:
            procs.append({"pid": int(parts[0]), "ppid": int(parts[1]), "elapsed_s": parse_clock(parts[2]),
                          "cpu": float(parts[3]), "args": parts[4]})
        except ValueError:
            continue
    return procs


def find_jobs(procs: list[dict]) -> list[Job]:
    """Group posttrain_lab processes into jobs: an accelerate launcher plus its
    ranks share the same module + args."""
    jobs: dict[tuple, Job] = {}
    for p in procs:
        m = JOB_RE.search(p["args"])
        if not m:
            continue
        args = m[2].split()
        key = (m[1], tuple(args))
        job = jobs.setdefault(key, Job(module=m[1], args=args))
        job.pids.append(p["pid"])
        job.cpu += p["cpu"]
        if p["elapsed_s"] is not None:
            job.elapsed_s = max(job.elapsed_s or 0, p["elapsed_s"])
        launcher = any(h in p["args"] for h in LAUNCHER_HINTS)
        if not launcher:
            job.ranks += 1
        env = _proc_environ(p["pid"])
        if env:
            job.gpus = job.gpus or env.get("CUDA_VISIBLE_DEVICES")
            job.smoke = job.smoke or env.get("SMOKE") == "1"
        if launcher or job.log is None:
            job.log = _proc_stdout(p["pid"]) or job.log
    return list(jobs.values())


def find_drivers(procs: list[dict]) -> list[dict]:
    out = []
    for p in procs:
        m = re.search(r"(run_all(?:_\w+)?\.sh|run_smoke\.sh)", p["args"])
        if m and "grep" not in p["args"]:
            out.append({**p, "script": m[1], "log": _proc_stdout(p["pid"])})
    return out


def job_output_dir(job: Job) -> str | None:
    """The run dir a training/gen_pairs job writes, from its own config + overrides."""
    yaml = job.yaml or ENTRY_DEFAULTS.get(job.module)
    if not yaml:
        return None
    yaml = yaml.split("configs/", 1)[-1]
    try:
        argv = job.overrides + (["--smoke"] if job.smoke else [])
        return str(load_config(yaml, argv).output_dir)
    except Exception:  # a config we can't resolve shouldn't break status
        return None


# ---------------------------------------------------------------------------
# GPUs


def gpu_now() -> list[dict] | None:
    if not shutil.which("nvidia-smi"):
        return None
    q = "index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 7:
            continue

        def num(x):
            try:
                return float(x)
            except ValueError:
                return None

        gpus.append({"index": int(f[0]), "name": f[1], "util": num(f[2]), "mem_used_mib": num(f[3]),
                     "mem_total_mib": num(f[4]), "power_w": num(f[5]), "temp_c": num(f[6])})
    return gpus


def gpu_history(path: Path, window_min: float = 60.0) -> dict | None:
    """Summary of the nvidia-smi -l logger's CSV (timestamp, index, util %, ...)."""
    if not path.exists():
        return None
    rows = []
    try:
        with path.open() as f:
            for rec in csv.reader(f):
                if len(rec) < 3 or not rec[1].strip().isdigit():
                    continue
                try:
                    ts = datetime.strptime(rec[0].strip()[:19], "%Y/%m/%d %H:%M:%S").timestamp()
                    util = float(rec[2].strip().split()[0])
                except ValueError:
                    continue
                rows.append((ts, int(rec[1]), util))
    except OSError:
        return None
    if not rows:
        return None
    last_ts = max(r[0] for r in rows)

    def summary(sel):
        by_gpu: dict[int, list[float]] = {}
        for _, i, u in sel:
            by_gpu.setdefault(i, []).append(u)
        utils = [u for _, _, u in sel]
        return {
            "mean_util": sum(utils) / len(utils) if utils else None,
            "idle_frac": sum(u < 10 for u in utils) / len(utils) if utils else None,
            "per_gpu": {i: sum(v) / len(v) for i, v in sorted(by_gpu.items())},
        }

    recent = [r for r in rows if r[0] >= last_ts - window_min * 60]
    return {
        "path": str(path), "first": min(r[0] for r in rows), "last": last_ts,
        "stale_s": _now() - last_ts, "window_min": window_min,
        "recent": summary(recent), "all": summary(rows),
    }


# ---------------------------------------------------------------------------
# per-run and per-stage state


@dataclass
class Run:
    name: str
    override: str | None
    output_dir: str
    state: str = "pending"            # done | running | stopped | failed | pending
    checkpoints: int = 0
    expected_checkpoints: int | None = None
    step: int | None = None
    max_steps: int | None = None
    epoch: float | None = None
    metrics: dict = field(default_factory=dict)
    progress: dict | None = None
    elapsed_s: float | None = None
    finished: float | None = None
    last_write: float | None = None
    size: int | None = None
    gpus: str | None = None
    ranks: int | None = None
    log: str | None = None
    errors: dict | None = None
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def expected_checkpoints(cfg) -> int | None:
    train = cfg.get("train") or {}
    strategy = str(train.get("save_strategy", "steps"))
    if strategy == "epoch":
        epochs = train.get("num_train_epochs")
        return int(epochs) if epochs else None
    save_steps = train.get("save_steps")
    if save_steps is None:
        return None
    save_steps = float(save_steps)
    if 0 < save_steps < 1:
        return max(1, round(1 / save_steps))
    max_steps = train.get("max_steps")
    if max_steps and int(max_steps) > 0:
        return int(max_steps) // int(save_steps)
    return None


def trainer_state(run_dir: Path) -> dict | None:
    for ckpt in reversed(checkpoints(run_dir)):
        state = read_json(ckpt / "trainer_state.json")
        if state:
            return state
    return None


def last_logged(state: dict) -> dict:
    for entry in reversed(state.get("log_history") or []):
        if "train_runtime" not in entry and len(entry) > 2:
            return entry
    return {}


def pick_metrics(trainer: str, values: dict) -> dict:
    keys = METRICS.get(trainer, ["loss"])
    picked = {k: values[k] for k in keys if k in values}
    if not picked and "loss" in values:
        picked["loss"] = values["loss"]
    return picked


def _latest_mtime(path: Path) -> float | None:
    latest = None
    for root, _, files in os.walk(path):
        for name in files:
            try:
                mt = os.lstat(os.path.join(root, name)).st_mtime
            except OSError:
                continue
            latest = mt if latest is None else max(latest, mt)
    return latest


def health(run: Run, trainer: str) -> None:
    m = run.metrics
    for k, v in m.items():
        if isinstance(v, float) and v != v:
            run.warnings.append(f"{k} is NaN")
    pct = run.progress["pct"] if run.progress else None
    if trainer == "grpo":
        if m.get("frac_reward_zero_std", 0) >= 0.8:
            run.warnings.append(f"frac_reward_zero_std {fmt_num(m['frac_reward_zero_std'])}: most groups get "
                                "identical rewards, little learning signal")
        if m.get("completions/clipped_ratio", 0) >= 0.5:
            run.warnings.append(f"completions/clipped_ratio {fmt_num(m['completions/clipped_ratio'])}: half the "
                                "completions hit max_completion_length")
    if trainer == "dpo" and pct is not None and pct > 30 and m.get("rewards/accuracies", 1) < 0.55:
        run.warnings.append(f"rewards/accuracies {fmt_num(m['rewards/accuracies'])} at {pct:.0f}%: DPO is barely "
                            "separating chosen from rejected")


def resolve_run(stage: Stage, cfg, override: str | None, jobs_by_dir: dict[str, Job],
                driver_section: list[str] | None, driver_log: Path | None) -> Run:
    out = Path(str(cfg.output_dir))
    run = Run(name=out.name, override=override, output_dir=str(out))
    trainer = str(cfg.get("stage", "sft"))
    run.expected_checkpoints = expected_checkpoints(cfg) if stage.kind in ("train", "rm") else None
    job = jobs_by_dir.get(str(out))

    if out.exists():
        ckpts = checkpoints(out)
        run.checkpoints = len(ckpts)
        state = trainer_state(out)
        if state:
            run.step = state.get("global_step")
            run.max_steps = state.get("max_steps")
            run.epoch = state.get("epoch")
            run.metrics = pick_metrics(trainer, last_logged(state))
        run.last_write = _latest_mtime(out)
        run.size = dir_size(out)

    done = (out / DONE_MARKER).exists()
    if done:
        run.state = "done"
        run.finished = (out / DONE_MARKER).stat().st_mtime
    elif job is not None:
        run.state = "running"
    elif out.exists() and any(out.iterdir()):
        run.state = "stopped"

    # the log this run writes to: its own (sweep runs), the job's stdout, or the driver's section
    log_path = None
    if stage.log and override is not None:
        candidate = REPO_ROOT / log_name(stage.log, override)
        if candidate.exists():
            log_path = candidate
    if job is not None and job.log:
        log_path = Path(job.log)
    lines: list[str] | None = None
    if log_path is not None and log_path != driver_log:
        lines = log_lines(tail_text(log_path))
    elif driver_section is not None and (job is not None or run.state == "stopped"):
        # sequential stages log into the run_all log; driver_section is only
        # passed for the stage its last '=== ' header names
        log_path, lines = driver_log, driver_section
    if log_path is not None:
        run.log = str(log_path)

    if job is not None:
        run.elapsed_s = job.elapsed_s
        run.gpus = job.gpus
        run.ranks = job.ranks
    if lines:
        bar = last_progress(lines, "" if stage.kind in ("train", "rm") else "Processed prompts")
        if run.state == "running":
            run.progress = bar
            live = last_metrics(lines)
            if live:
                run.metrics = pick_metrics(trainer, live) or run.metrics
        if run.state in ("running", "stopped"):
            run.errors = summarize_errors(lines)
            if run.state == "stopped" and run.errors:
                run.state = "failed"

    if run.state == "running" and run.log:
        try:
            quiet = _now() - Path(run.log).stat().st_mtime
        except OSError:
            quiet = None
        if quiet is not None and quiet > QUIET_MINUTES * 60:
            run.warnings.append(f"log quiet for {fmt_duration(quiet)} (stuck? check the GPUs below)")
    health(run, trainer)
    return run


def pairs_checks(run: Run, cfg, eval_cfg) -> dict | None:
    stats = read_json(Path(run.output_dir) / "stats.json")
    if not stats:
        return None
    run.notes.append(
        f"{fmt_num(stats.get('prompts'))} prompts x {cfg.eval.samples_per_prompt} samples, solve rate "
        f"{fmt_num(stats.get('sample_solve_rate'))} -> {fmt_num(stats.get('pairs'))} pairs "
        f"({fmt_num(stats.get('train_pairs'))} train / {fmt_num(stats.get('test_pairs'))} test); "
        f"{fmt_num(stats.get('prompts_all_correct'))} prompts all-correct, "
        f"{fmt_num(stats.get('prompts_all_wrong'))} all-wrong (no pair)"
    )
    expected = split_size(str(cfg.dataset.id), cfg.dataset.get("config"), str(cfg.data.split))
    cap = cfg.data.get("max_samples")
    if cap is not None and expected is not None:
        expected = min(expected, int(cap))
    if expected and stats.get("prompts", 0) < expected:
        run.warnings.append(f"sampled {stats.get('prompts')} of {expected} {cfg.data.split} prompts")
    if stats.get("train_pairs", 0) < 2000:
        run.warnings.append(f"only {stats.get('train_pairs')} DPO training pairs: the DPO arm will be weak")
    want = int(eval_cfg.eval.get("pref_logps", {}).get("num_pairs", 0) or 0)
    if want and stats.get("test_pairs", 0) < want:
        run.warnings.append(f"{stats.get('test_pairs')} held-out pairs < pref_logps.num_pairs={want}")
    rate = stats.get("sample_solve_rate")
    if rate is not None and not 0.15 <= rate <= 0.9:
        run.warnings.append(f"solve rate {fmt_num(rate)} is extreme: few prompts are contested")
    return stats


def rm_checks(run: Run, cfg) -> None:
    row = last_jsonl(Path(run.output_dir) / "rm_eval.jsonl")
    if not row:
        return
    acc = row.get("eval_accuracy")
    gate = float(cfg.get("min_accuracy", 0))
    if acc is not None:
        run.notes.append(f"pairwise accuracy {fmt_num(acc)} (gate {fmt_num(gate)})")
        if acc < gate:
            run.state = "failed"
            run.warnings.append(f"RM accuracy {fmt_num(acc)} below the gate {fmt_num(gate)}: the pipeline stops here")


def split_size(repo: str, config: str | None, split: str) -> int | None:
    """num_examples of a prepared split in the HF datasets cache, no datasets import."""
    root = os.environ.get("HF_DATASETS_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "datasets")
    name = repo.replace("/", "___")
    for info in sorted(Path(root).glob(f"{name}/{config or 'default'}/*/*/dataset_info.json")):
        data = read_json(info) or {}
        n = (data.get("splits") or {}).get(split, {}).get("num_examples")
        if n:
            return int(n)
    return None


def eval_state(task: str, eval_cfg, jobs: list[Job], stage: Stage) -> dict:
    root = Path(str(eval_cfg.results_root))
    out_dir = Path(str(eval_cfg.eval.out_dir))
    groups = [g for g in eval_cfg.eval.metric_groups if g in SENTINELS]
    ckpts = discover_all_checkpoints(root / "runs") if (root / "runs").exists() else []
    per_group = {g: 0 for g in groups}
    per_run: dict[str, list[int]] = {}
    complete = 0
    for ck in ckpts:
        run_dir = ck.parent if ck.name.startswith("checkpoint-") else ck
        row = last_jsonl(out_dir / run_dir.name / f"{ck.name}.jsonl") or {}
        done_groups = [g for g in groups if all(k in row for k in SENTINELS[g])]
        for g in done_groups:
            per_group[g] += 1
        ok = len(done_groups) == len(groups)
        complete += ok
        tally = per_run.setdefault(run_dir.name, [0, 0])
        tally[0] += ok
        tally[1] += 1

    shards = []
    for job in jobs:
        if job.module != "eval.run_eval":
            continue
        job_task = next((a.split("=", 1)[1] for a in job.args if a.startswith("task=")), "helpful")
        if job_task != task:
            continue
        shard = job.shard or "0/1"
        log = Path(job.log) if job.log else REPO_ROOT / log_name(stage.log or "", shard=int(shard.split("/")[0]))
        lines = log_lines(tail_text(log)) if log.exists() else []
        current = None
        for line in reversed(lines):
            m = re.search(r"\[eval\] (\S+) -> groups (\[.*\])", line)
            if m:
                current = {"checkpoint": m[1], "groups": m[2]}
                break
        shards.append({"shard": shard, "gpus": job.gpus, "elapsed_s": job.elapsed_s, "log": str(log),
                       "current": current, "progress": last_progress(lines, None), "errors": summarize_errors(lines)})
    # shard logs of a finished/failed eval
    stopped = []
    if stage.log and not shards:
        for i in range(8):
            log = REPO_ROOT / log_name(stage.log, shard=i)
            if log.exists():
                errs = summarize_errors(log_lines(tail_text(log)))
                if errs:
                    stopped.append({"log": str(log), "errors": errs})
    return {"checkpoints": len(ckpts), "complete": complete, "groups": groups, "per_group": per_group,
            "per_run": per_run, "shards": shards, "failed_shards": stopped}


def figures_state(eval_cfg) -> dict:
    root = Path(str(eval_cfg.results_root))
    figs = sorted(p.name for p in (root / "figures").glob("*") if p.suffix in (".png", ".pdf")) \
        if (root / "figures").exists() else []
    csv_path = root / "frontier.csv"
    return {"figures": figs, "frontier_csv": str(csv_path) if csv_path.exists() else None,
            "frontier_mtime": csv_path.stat().st_mtime if csv_path.exists() else None}


def track_state(task: str, procs: list[dict], jobs: list[Job], jobs_by_dir: dict[str, Job],
                log_override: str | None) -> dict:
    spec = TRACKS[task]
    eval_cfg = load_config("eval.yaml", [f"task={task}"])
    root = Path(str(eval_cfg.results_root))

    drivers = [d for d in find_drivers(procs) if d["script"] in (spec["driver"], "run_all.sh", "run_smoke.sh")]
    driver = drivers[0] if drivers else None
    driver_log = None
    for candidate in [log_override, driver and driver.get("log")] + [str(REPO_ROOT / n) for n in spec["logs"]]:
        if candidate and Path(candidate).exists():
            driver_log = Path(candidate)
            break
    header, section_lines, driver_errors = None, None, None
    if driver_log:
        text = tail_text(driver_log)
        header, section = last_section(text)
        section_lines = log_lines(section)
        driver_errors = summarize_errors(section_lines)

    stages = []
    for stage in spec["stages"]:
        entry: dict = {"key": stage.key, "title": stage.title, "kind": stage.kind, "gpus": stage.gpus, "runs": []}
        if stage.kind in ("train", "pairs", "rm"):
            values = [None]
            if stage.sweep:
                sweeps = OmegaConf.load(CONFIG_DIR / task / "sweeps.yaml")[stage.sweep]
                values = [f"{k}={v}" for k, vs in sweeps.items() for v in vs]
            for ov in values:
                cfg = load_config(stage.yaml, [ov] if ov else [])
                section = section_lines if header_names(header, stage) else None
                run = resolve_run(stage, cfg, ov, jobs_by_dir, section, driver_log)
                if stage.kind == "pairs":
                    entry["pairs"] = pairs_checks(run, cfg, eval_cfg)
                if stage.kind == "rm":
                    rm_checks(run, cfg)
                entry["runs"].append(run)
            states = [r.state for r in entry["runs"]]
            entry["state"] = _rollup(states)
        elif stage.kind == "eval":
            ev = eval_state(task, eval_cfg, jobs, stage)
            entry["eval"] = ev
            if ev["shards"]:
                entry["state"] = "running"
            elif ev["failed_shards"]:
                entry["state"] = "failed"
            elif ev["checkpoints"] and ev["complete"] == ev["checkpoints"]:
                entry["state"] = "done"
            elif any(ev["per_group"].values()):
                entry["state"] = "stopped"
            else:
                entry["state"] = "pending"
        else:
            fig = figures_state(eval_cfg)
            entry["figures"] = fig
            entry["state"] = "done" if fig["figures"] else "pending"
        stages.append(entry)

    current = next((s for s in stages if s["state"] == "running"), None)
    return {
        "task": task, "results_root": str(root), "exists": root.exists(),
        "driver": driver, "driver_log": str(driver_log) if driver_log else None,
        "log_stage": header, "driver_errors": driver_errors, "stages": stages,
        "current": current["key"] if current else None,
    }


def header_names(header: str | None, stage: Stage) -> bool:
    """Does a run_all '=== math m3a: ...' / '=== stage 3a: ...' header name this stage?"""
    return bool(header) and re.search(rf"(?:math|stage)\s+{re.escape(stage.key)}\b", header) is not None


def _rollup(states: list[str]) -> str:
    if all(s == "done" for s in states):
        return "done"
    for s in ("running", "failed", "stopped"):
        if s in states:
            return s
    return "partial" if "done" in states else "pending"


# ---------------------------------------------------------------------------
# the whole picture + attention list


def collect(tasks: list[str], log_override: str | None = None) -> dict:
    procs = list_processes()
    jobs = find_jobs(procs)
    jobs_by_dir = {}
    for job in jobs:
        out = job_output_dir(job)
        if out:
            jobs_by_dir[out] = job
    tracks = [track_state(t, procs, jobs, jobs_by_dir, log_override if len(tasks) == 1 else None) for t in tasks]
    res = results_dir()
    try:
        usage = shutil.disk_usage(res if Path(res).exists() else REPO_ROOT)
        disk = {"free": usage.free, "total": usage.total}
    except OSError:
        disk = None
    gpus = gpu_now()
    hist = gpu_history(REPO_ROOT / "gpu_util.csv")
    try:
        commit = subprocess.run(["git", "-C", str(REPO_ROOT), "log", "-1", "--format=%h %s"],
                                capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "-uno"],
                                    capture_output=True, text=True, timeout=5).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "", False
    expected = {r.output_dir for t in tracks for s in t["stages"] for r in s["runs"]}
    other_jobs = [j for out, j in jobs_by_dir.items() if out not in expected]
    other_jobs += [j for j in jobs if j.module != "eval.run_eval" and j not in jobs_by_dir.values()]
    report = {
        "time": _now(), "host": socket.gethostname(), "commit": commit, "dirty": dirty,
        "smoke": os.environ.get("SMOKE") == "1",
        "results_dir": res, "results_dir_source": results_dir_source(), "disk": disk,
        "hf_home": os.environ.get("HF_HOME"), "offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "tracks": tracks, "gpus": gpus, "gpu_history": hist,
        "other_jobs": [asdict(j) for j in other_jobs],
    }
    report["attention"] = attention(report, jobs)
    return report


def attention(report: dict, jobs: list[Job]) -> list[str]:
    items = []
    disk = report.get("disk")
    if disk and disk["free"] < FREE_GB_WARN * 1024**3:
        items.append(f"only {fmt_bytes(disk['free'])} free under {report['results_dir']}")
    for track in report["tracks"]:
        t = track["task"]
        for stage in track["stages"]:
            for run in stage["runs"]:
                for w in run.warnings:
                    items.append(f"{t} {stage['key']} {run.name}: {w}")
                if run.state == "failed":
                    msg = (run.errors or {}).get("last_exception") or "see its log"
                    items.append(f"{t} {stage['key']} {run.name} FAILED: {msg}")
                elif run.state == "stopped" and not track["driver"]:
                    items.append(f"{t} {stage['key']} {run.name} stopped mid-run (no DONE marker); re-running the "
                                 "track restarts it")
            ev = stage.get("eval")
            if ev:
                for sh in ev["failed_shards"]:
                    items.append(f"{t} eval: errors in {Path(sh['log']).name}: {sh['errors']['last_exception'] or sh['errors']['last'][-1]}")
        started = any(s["state"] != "pending" for s in track["stages"])
        finished = all(s["state"] == "done" for s in track["stages"])
        if started and not finished and not track["driver"] and not any(s["state"] == "running" for s in track["stages"]):
            last = track.get("driver_errors") or {}
            why = last.get("last_exception") or (last.get("last") or [None])[-1]
            items.append(f"{t}: pipeline is not running and not finished"
                         + (f"; last error in {Path(track['driver_log']).name}: {why}" if why else ""))
    gpus = report.get("gpus") or []
    running = [r for tr in report["tracks"] for s in tr["stages"] for r in s["runs"] if r.state == "running"]
    if gpus and running and all((g["util"] or 0) < 5 for g in gpus):
        long_running = [r for r in running if (r.elapsed_s or 0) > 600]
        if long_running:
            items.append("every GPU is idle while " + ", ".join(r.name for r in long_running)
                         + " has been running >10 min (loading? stuck?)")
    hist = report.get("gpu_history")
    if hist and hist["stale_s"] > 300 and (running or any(tr["driver"] for tr in report["tracks"])):
        items.append(f"gpu_util.csv last written {fmt_duration(hist['stale_s'])} ago: the GPU logger has stopped")
    return items


# ---------------------------------------------------------------------------
# rendering

ICONS = {"done": "✓", "running": "▶", "failed": "✗", "stopped": "■", "partial": "◐", "pending": "·"}
COLORS = {"done": "32", "running": "33;1", "failed": "31;1", "stopped": "35", "partial": "36", "pending": "2"}


class Painter:
    def __init__(self, color: bool):
        self.color = color

    def __call__(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color and code else text

    def state(self, state: str) -> str:
        return self(f"{ICONS.get(state, '?')} {state.upper():<8}", COLORS.get(state, ""))


def _metrics_line(metrics: dict) -> str:
    return "  ".join(f"{k.split('/')[-1] if k.count('/') else k}={fmt_num(v)}" for k, v in metrics.items())


def render_run(run: Run, p: Painter, indent: str, label: str | None = None) -> list[str]:
    out = []
    bits = []
    if run.state != "pending" and (run.expected_checkpoints or run.checkpoints):
        exp = f"/{run.expected_checkpoints}" if run.expected_checkpoints else ""
        bits.append(f"ckpts {run.checkpoints}{exp}")
    if run.progress:
        pr = run.progress
        bits.append(f"step {pr['n']}/{pr['total']} ({pr['pct']:.0f}%)")
        left = pr["remaining_s"]
        eta = f" (~{fmt_time(_now() + left)})" if left is not None else ""
        bits.append(f"{fmt_duration(pr['elapsed_s'])} in, ~{fmt_duration(left)} left{eta}")
    elif run.step is not None:
        mx = f"/{run.max_steps}" if run.max_steps else ""
        bits.append(f"step {run.step}{mx}")
    if run.state == "running" and run.gpus is not None:
        bits.append(f"GPU {run.gpus}")
    elif run.state == "running" and run.ranks:
        bits.append(f"{run.ranks} ranks")
    if run.finished:
        bits.append(f"finished {fmt_time(run.finished)}")
    elif run.last_write and run.state in ("stopped", "failed"):
        bits.append(f"last write {fmt_time(run.last_write)}")
    if run.size:
        bits.append(fmt_bytes(run.size))
    name = label or run.name
    out.append(f"{indent}{p.state(run.state)} {name:<26} {' · '.join(bits)}".rstrip())
    if run.metrics:
        out.append(f"{indent}    {p(_metrics_line(run.metrics), '2')}")
    for note in run.notes:
        out.append(f"{indent}    {note}")
    for w in run.warnings:
        out.append(f"{indent}    {p('⚠ ' + w, '33')}")
    if run.errors and run.state in ("failed", "stopped", "running"):
        e = run.errors
        where = Path(run.log).name if run.log else "log"
        out.append(f"{indent}    " + p(f"{e['count']} error line(s) in {where}", "31"))
        for line in e["last"]:
            out.append(f"{indent}      {p(line, '31')}")
    if run.log and run.state in ("running", "failed", "stopped"):
        out.append(f"{indent}    {p('log: ' + run.log, '2')}")
    return out


def render(report: dict, color: bool) -> str:
    p = Painter(color)
    L: list[str] = []
    stamp = datetime.fromtimestamp(report["time"]).strftime("%Y-%m-%d %H:%M:%S")
    L.append(p(f"posttrain-lab status  {stamp}  {report['host']}", "1"))
    commit = report["commit"] + (" (uncommitted changes)" if report["dirty"] else "")
    L.append(f"  code      {commit or '?'}")
    disk = report["disk"]
    free = f"  ·  {fmt_bytes(disk['free'])} free of {fmt_bytes(disk['total'])}" if disk else ""
    L.append(f"  results   {report['results_dir']} (from {report['results_dir_source']}){free}")
    hf = report["hf_home"] or "~/.cache/huggingface"
    L.append(f"  HF cache  {hf}{'  ·  offline' if report['offline'] else ''}"
             f"{'  ·  SMOKE=1 (showing smoke results)' if report['smoke'] else ''}")

    for track in report["tracks"]:
        L.append("")
        drv = track["driver"]
        if drv:
            how = p(f"pipeline running: {drv['script']} pid {drv['pid']}, up {fmt_duration(drv['elapsed_s'])}", "33")
        elif all(s["state"] == "done" for s in track["stages"]):
            how = p("pipeline finished", "32")
        elif any(s["state"] != "pending" for s in track["stages"]):
            how = p("pipeline NOT running", "31;1")
        else:
            how = p("not started", "2")
        L.append(p(f"{track['task'].upper()} track", "1") + f"   {how}")
        L.append(f"  root {track['results_root']}")
        if not drv and all(s["state"] == "pending" for s in track["stages"]):
            continue
        if track["driver_log"]:
            stage_hdr = f"  ·  last stage header: {track['log_stage']}" if track["log_stage"] else ""
            L.append(f"  log  {track['driver_log']}{stage_hdr}")
        if not drv and any(s["state"] != "done" for s in track["stages"]):
            log = Path(track["driver_log"]).name if track["driver_log"] else f"{track['task']}-all.log"
            L.append(p(f"  resume   nohup make {track['task']}-all >> {log} 2>&1 &   (finished runs are skipped)", "2"))
        done = sum(s["state"] == "done" for s in track["stages"])
        L.append(f"  stages done {done}/{len(track['stages'])}"
                 + (f"  ·  now: {track['current']}" if track["current"] else ""))
        for stage in track["stages"]:
            head = f"  {p.state(stage['state'])} {stage['key']:<4} {stage['title']}"
            gpu_note = {"4": "4 GPUs", "1 each": "1 GPU per shard" if stage["kind"] == "eval" else "1 GPU per run",
                        "1": "1 GPU", "0": "CPU"}.get(stage["gpus"], "")
            L.append(head + p(f"   [{gpu_note}]", "2"))
            for run in stage["runs"]:
                L.extend(render_run(run, p, "        ", label=run.override or run.name))
            if stage.get("eval"):
                L.extend(render_eval(stage["eval"], p))
            if stage.get("figures"):
                fig = stage["figures"]
                if fig["figures"]:
                    L.append(f"        {len(fig['figures'])} figures: {', '.join(fig['figures'])}")
                if fig["frontier_csv"]:
                    L.append(f"        frontier.csv written {fmt_time(fig['frontier_mtime'])}")
        reported = any(r.errors for s in track["stages"] for r in s["runs"])
        if track["driver_errors"] and not track["driver"] and not reported:
            e = track["driver_errors"]
            L.append(p(f"  errors after the last stage header in {Path(track['driver_log']).name}:", "31"))
            for line in e["last"]:
                L.append(p(f"    {line}", "31"))

    if report["other_jobs"]:
        L.append("")
        L.append(p("other posttrain_lab jobs", "1"))
        for j in report["other_jobs"]:
            L.append(f"  {j['module']} {' '.join(j['args'])}  ·  up {fmt_duration(j['elapsed_s'])}"
                     f"{'  ·  GPU ' + j['gpus'] if j['gpus'] else ''}{'  ·  SMOKE' if j['smoke'] else ''}")

    L.append("")
    L.append(p("GPUs", "1"))
    gpus = report["gpus"]
    if gpus is None:
        L.append("  nvidia-smi not available")
    else:
        owner = _gpu_owners(report)
        for g in gpus:
            mem = f"{(g['mem_used_mib'] or 0) / 1024:5.1f}/{(g['mem_total_mib'] or 0) / 1024:.0f} GB"
            util = g["util"] if g["util"] is not None else 0
            bar = "█" * int(util // 10) + "░" * (10 - int(util // 10))
            L.append(f"  {g['index']}  {bar} {util:3.0f}%  {mem}  {g['power_w'] or 0:4.0f} W  "
                     f"{g['temp_c'] or 0:3.0f}°C  {owner.get(g['index'], '')}")
    hist = report["gpu_history"]
    if hist:
        rec, al = hist["recent"], hist["all"]
        per = "  ".join(f"{i}:{u:.0f}%" for i, u in rec["per_gpu"].items())
        L.append(f"  gpu_util.csv: last {hist['window_min']:.0f} min mean {rec['mean_util']:.0f}% ({per}), "
                 f"idle {100 * rec['idle_frac']:.0f}% of samples; since {fmt_time(hist['first'])} mean "
                 f"{al['mean_util']:.0f}%, idle {100 * al['idle_frac']:.0f}%")
    else:
        L.append(p("  no gpu_util.csv (start the logger to get utilization history)", "2"))

    L.append("")
    items = report["attention"]
    if items:
        L.append(p(f"attention ({len(items)})", "31;1"))
        for item in items:
            L.append(p(f"  ⚠ {item}", "33"))
    else:
        L.append(p("attention: nothing", "32"))
    return "\n".join(L)


def render_eval(ev: dict, p: Painter) -> list[str]:
    out = []
    if not ev["checkpoints"]:
        out.append(p("        no checkpoints to evaluate yet", "2"))
        return out
    out.append(f"        {ev['complete']}/{ev['checkpoints']} checkpoints fully evaluated")
    out.append("        groups: " + "  ".join(f"{g} {n}/{ev['checkpoints']}" for g, n in ev["per_group"].items()))
    runs = "  ".join(f"{name} {a}/{b}" for name, (a, b) in ev["per_run"].items())
    out.append(f"        per run: {runs}")
    for sh in ev["shards"]:
        cur = sh["current"]["checkpoint"].split("/runs/")[-1] if sh["current"] else "starting"
        groups = f" {sh['current']['groups']}" if sh["current"] else ""
        bar = ""
        if sh["progress"]:
            pr = sh["progress"]
            bar = f" · {pr['label'] or 'progress'} {pr['n']}/{pr['total']} ({pr['pct']:.0f}%)"
        gpu = f"GPU {sh['gpus']} · " if sh["gpus"] else ""
        out.append(f"        ▶ shard {sh['shard']}: {gpu}{cur}{groups}{bar} · up {fmt_duration(sh['elapsed_s'])}")
        if sh["errors"]:
            out.append(p(f"          {sh['errors']['count']} error line(s): {sh['errors']['last'][-1]}", "31"))
    return out


def _gpu_owners(report: dict) -> dict[int, str]:
    owners: dict[int, list[str]] = {}
    gpus = [g["index"] for g in report["gpus"] or []]
    for track in report["tracks"]:
        for stage in track["stages"]:
            for run in stage["runs"]:
                if run.state != "running":
                    continue
                label = f"{track['task']} {stage['key']} {run.override or run.name}"
                ids = _gpu_ids(run.gpus, gpus)
                for i in ids:
                    owners.setdefault(i, []).append(label)
            for sh in (stage.get("eval") or {}).get("shards", []):
                for i in _gpu_ids(sh["gpus"], gpus):
                    owners.setdefault(i, []).append(f"{track['task']} eval shard {sh['shard']}")
    return {i: "← " + ", ".join(v) for i, v in owners.items()}


def _gpu_ids(cvd: str | None, all_ids: list[int]) -> list[int]:
    if cvd is None:
        return all_ids
    ids = []
    for part in cvd.split(","):
        part = part.strip()
        if part.isdigit():
            ids.append(int(part))
    return ids


def to_json(report: dict) -> str:
    def default(o):
        if isinstance(o, Run):
            return asdict(o)
        return str(o)

    return json.dumps(report, default=default, indent=2)


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task", nargs="?", default="all", choices=["all", *TRACKS])
    parser.add_argument("--smoke", action="store_true", help="show the smoke results tree (SMOKE=1)")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--watch", type=float, metavar="SEC", help="refresh every SEC seconds")
    parser.add_argument("--log", help="the run_all log, if it is not math-all.log / helpful-all.log / all.log")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        os.environ["SMOKE"] = "1"
    tasks = list(TRACKS) if args.task == "all" else [args.task]
    color = not args.no_color and not args.json and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    while True:
        report = collect(tasks, args.log)
        text = to_json(report) if args.json else render(report, color)
        if args.watch:
            sys.stdout.write("\033[2J\033[H")
        print(text, flush=True)
        if not args.watch:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
