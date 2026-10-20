"""Start a track over: stop its jobs, clear its results and logs, optionally relaunch.

    python -m posttrain_lab.reset {math|helpful|all} [--smoke] [--archive]
                                  [--relaunch] [--dry-run] [--yes]
    make reset TASK=math [RESET_ARGS="--relaunch"]     make reset-smoke

In order:

  1. stop this repo's jobs for the track: the run_all / run_smoke driver, the
     make target and stage scripts driving it, every training / pair /
     eval job writing into the track (accelerate launchers, ranks, vLLM
     engines: the whole process tree). Jobs of the other track, and of
     other checkouts, are left alone. SIGTERM, then SIGKILL after --timeout.
  2. delete <RESULTS_DIR>/<task> (or <RESULTS_DIR>/smoke/<task> with --smoke),
     or rename it to <task>.<timestamp> with --archive
  3. remove the track's logs in the repo (math-all.log, per-run DPO and eval
     shard logs, smoke.log) and the half-written tmp* files a crashed
     dataset map can leave in the HF datasets cache
  4. with --relaunch: start the GPU logger fresh (gpu_util.csv) and
     `nohup make <task>-all > <task>-all.log` (or `make smoke`), detached

It prints the whole plan first and asks you to type the track name before
touching anything (--yes skips that, --dry-run only prints the plan). It
never touches the HF model/dataset cache, .env, the other track's results,
smoke results (unless --smoke), or MLflow (delete old runs there in the UI).
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from posttrain_lab import status
from posttrain_lab.config import results_root
from posttrain_lab.utils.dotenv import REPO_ROOT, results_dir

# make targets (Makefile) that drive each track
MAKE_TARGETS = {
    "math": {"math-sft-base", "math-pairs", "math-arm-sft", "math-arm-dpo", "math-arm-grpo", "math-eval",
             "math-plots", "math-all", "all"},
    "helpful": {"sft-base", "rm", "arm-sft", "arm-dpo", "arm-ppo", "arm-grpo", "eval", "plots", "helpful-all", "all"},
    "smoke": {"smoke"},
}
GPU_LOGGER = "nvidia-smi --query-gpu=timestamp"


def _tasks(task: str) -> list[str]:
    return list(status.TRACKS) if task == "all" else [task]


# ---------------------------------------------------------------------------
# 1. which processes


def _descendants(roots: set[int], procs: list[dict]) -> set[int]:
    children: dict[int, list[int]] = {}
    for p in procs:
        children.setdefault(p["ppid"], []).append(p["pid"])
    out, todo = set(), list(roots)
    while todo:
        pid = todo.pop()
        if pid in out:
            continue
        out.add(pid)
        todo.extend(children.get(pid, []))
    return out


def _ancestors(procs: list[dict]) -> set[int]:
    """This process and its parents (make reset, the shell): never ours to stop."""
    parent = {p["pid"]: p["ppid"] for p in procs}
    out, pid = set(), os.getpid()
    while pid and pid not in out:
        out.add(pid)
        pid = parent.get(pid, 0)
    return out


def track_processes(tasks: list[str], smoke: bool, procs: list[dict] | None = None) -> dict[int, str]:
    """pid -> why, for every process of this repo driving or running the tracks.

    A driver shared with a track that is not being reset (run_all.sh / make
    all, or run_smoke.sh / make smoke for one smoke track) is stopped on its
    own, without its process tree: its children may be the other track's run,
    which keeps going (and nothing starts this track's stages after it)."""
    procs = status.list_processes() if procs is None else procs
    roots: dict[int, str] = {}
    partial = set(tasks) != set(status.TRACKS)
    shared_scripts = {"run_smoke.sh"} if smoke else {"run_all.sh"}
    shared_targets = {"smoke"} if smoke else {"all"}
    shared: set[int] = set()

    scripts = {"run_smoke.sh"} if smoke else {status.TRACKS[t]["driver"] for t in tasks} | {"run_all.sh"}
    for d in status.find_drivers(procs):
        if d["script"] in scripts:
            roots[d["pid"]] = f"driver {d['script']}"
            if partial and d["script"] in shared_scripts:
                shared.add(d["pid"])

    targets = MAKE_TARGETS["smoke"] if smoke else set().union(*(MAKE_TARGETS[t] for t in tasks))
    stage_dirs = [] if smoke else [f"scripts/{t}/" for t in tasks]
    for p in procs:
        words = p["args"].split()
        if not words:
            continue
        exe = Path(words[0]).name
        why = None
        if exe in ("make", "gmake") and targets & set(words[1:]):
            why = f"make {' '.join(w for w in words[1:] if w in targets)}"
        elif exe in ("bash", "sh") and any(d in p["args"] for d in stage_dirs):
            why = "stage script " + next(w for w in words[1:] if any(d in w for d in stage_dirs))
        # the cwd check last: it is an lsof call per process off Linux
        if why and status.in_this_repo(p["pid"]):
            roots[p["pid"]] = why
            if partial and exe in ("make", "gmake") and not (targets - shared_targets) & set(words[1:]):
                shared.add(p["pid"])

    track_roots = {str(safe_root(t, smoke)) for t in tasks}
    for job in status.find_jobs(procs):
        if job.smoke != smoke:
            continue
        if job.module == "eval.run_eval":
            job_task = next((a.split("=", 1)[1] for a in job.args if a.startswith("task=")), "helpful")
            mine = job_task in tasks
        else:
            out = status.job_output_dir(job)
            if out is not None and not Path(out).is_absolute():
                out = str(REPO_ROOT / out)
            mine = out is not None and any(str(Path(out).resolve()).startswith(r + os.sep) for r in track_roots)
        if mine:
            for pid in job.pids:
                roots[pid] = f"posttrain_lab.{job.module} {' '.join(job.args)}"

    # everything they started: torchrun ranks, vLLM engine cores, shells
    keep_out = _ancestors(procs)
    out: dict[int, str] = {}
    for pid in _descendants(set(roots) - shared, procs) - keep_out:
        out[pid] = roots.get(pid) or f"child: {next(p['args'] for p in procs if p['pid'] == pid)[:100]}"
    for pid in shared - keep_out:
        out[pid] = f"{roots[pid]} (shared with the other track: this process only)"
    return out


def gpu_logger_pids(procs: list[dict] | None = None) -> list[int]:
    procs = status.list_processes() if procs is None else procs
    return [p["pid"] for p in procs if GPU_LOGGER in p["args"] and Path(p["args"].split()[0]).name == "nvidia-smi"
            and status.in_this_repo(p["pid"])]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie is gone for our purposes
    try:
        st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    except OSError:
        return True
    return bool(st) and not st.startswith("Z")


def stop(pids: list[int], timeout: float) -> list[int]:
    """SIGTERM, wait up to timeout, SIGKILL what is left; returns survivors."""
    for sig, wait in ((signal.SIGTERM, timeout), (signal.SIGKILL, 10.0)):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.time() + wait
        while time.time() < deadline and any(_alive(p) for p in pids):
            time.sleep(0.5)
        pids = [p for p in pids if _alive(p)]
        if not pids:
            return []
        if sig == signal.SIGTERM:
            print(f"  {len(pids)} still running after {timeout:.0f}s, sending SIGKILL")
    return pids


# ---------------------------------------------------------------------------
# 2./3. what to remove


def _root(task: str, smoke: bool) -> str:
    prev = os.environ.get("SMOKE")
    os.environ["SMOKE"] = "1" if smoke else "0"
    try:
        return results_root(task)
    finally:
        if prev is None:
            os.environ.pop("SMOKE", None)
        else:
            os.environ["SMOKE"] = prev


def safe_root(task: str, smoke: bool) -> Path:
    """The track's results root, refusing anything that is not
    <RESULTS_DIR>/<task> or <RESULTS_DIR>/smoke/<task>."""
    def anchored(path: str) -> Path:  # the default 'results' is relative to the repo root
        p = Path(path)
        return (p if p.is_absolute() else REPO_ROOT / p).resolve()

    root, base = anchored(_root(task, smoke)), anchored(results_dir())
    expected = (base / "smoke" / task) if smoke else (base / task)
    if root != expected or root == base:
        raise SystemExit(f"[reset] refusing: results root {root} is not {expected}")
    return root


def track_logs(tasks: list[str], smoke: bool) -> list[Path]:
    if smoke:
        names = ["smoke.log"]
    else:
        # all.log (make all) belongs to both tracks: only a full reset removes it
        shared = {"nohup.out"} | (set() if set(tasks) == set(status.TRACKS) else {"all.log"})
        names = list(dict.fromkeys(n for t in tasks for n in status.TRACKS[t]["logs"] if n not in shared))
        names += [s.log.replace("{ov}", "*").replace("{i}", "*") for t in tasks
                  for s in status.TRACKS[t]["stages"] if s.log]
    found: list[Path] = []
    for pattern in names:
        found += sorted(p for p in REPO_ROOT.glob(pattern) if p.is_file())
    return found


STALE_TMP_S = 3600


def hf_tmp_files(min_age_s: float = STALE_TMP_S) -> list[Path]:
    """tmp* files a crashed dataset map left next to prepared datasets. The
    cache may be shared (the other track, preflight, other checkouts), so only
    files nothing has written to for min_age_s: a live map keeps its tmp file
    fresh."""
    root = Path(os.environ.get("HF_DATASETS_CACHE") or Path(
        os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "datasets")
    if not root.exists():
        return []
    cutoff = time.time() - min_age_s
    return sorted(p for p in root.glob("*/*/*/*/tmp*") if p.is_file() and p.stat().st_mtime < cutoff)


# ---------------------------------------------------------------------------
# 4. relaunch


def relaunch(task: str, smoke: bool) -> list[str]:
    for pid in gpu_logger_pids():
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    started = []
    if shutil.which("nvidia-smi"):
        with (REPO_ROOT / "gpu_util.csv").open("w") as out:
            p = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw",
                                  "--format=csv", "-l", "30"], cwd=REPO_ROOT, stdout=out, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        started.append(f"GPU logger pid {p.pid} -> gpu_util.csv")
    target, log = ("smoke", "smoke.log") if smoke else (f"{task}-all", f"{task}-all.log")
    if task == "all" and not smoke:
        target, log = "all", "all.log"
    env = {k: v for k, v in os.environ.items() if k != "SMOKE"}
    with (REPO_ROOT / log).open("w") as out:
        p = subprocess.Popen(["make", target], cwd=REPO_ROOT, stdout=out, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True, env=env)
    started.append(f"make {target} pid {p.pid} -> {log}")
    return started


# ---------------------------------------------------------------------------


def plan(task: str, smoke: bool) -> dict:
    tasks = _tasks(task)
    procs = status.list_processes()
    pids = track_processes(tasks, smoke, procs)
    roots = [safe_root(t, smoke) for t in tasks]
    others = {pid for pid in track_processes([t for t in status.TRACKS if t not in tasks] or [], smoke, procs)} \
        if task != "all" else set()
    return {
        "tasks": tasks, "smoke": smoke, "processes": pids,
        "roots": [(r, status.dir_size(r) if r.exists() else None) for r in roots],
        "logs": track_logs(tasks, smoke), "hf_tmp": hf_tmp_files(), "other_running": sorted(others),
    }


def show(pl: dict, archive: bool, do_relaunch: bool) -> None:
    label = " + ".join(pl["tasks"]) + (" (smoke)" if pl["smoke"] else "")
    print(f"reset {label}   code {_commit()}")
    print("\n1. stop jobs")
    if pl["processes"]:
        for pid, why in sorted(pl["processes"].items()):
            print(f"   {pid:>7}  {why}")
    else:
        print("   nothing running for this track")
    print(f"\n2. {'archive (rename)' if archive else 'DELETE'} results")
    for root, size in pl["roots"]:
        if size is None:
            print(f"   {root}  (does not exist)")
        else:
            dest = f" -> {root.name}.<timestamp>" if archive else ""
            print(f"   {root}  ({status.fmt_bytes(size)}){dest}")
    print(f"\n3. remove logs and dataset-cache leftovers (tmp* untouched for {STALE_TMP_S // 60} min)")
    for path in pl["logs"] + pl["hf_tmp"]:
        print(f"   {path}")
    if not pl["logs"] and not pl["hf_tmp"]:
        print("   none")
    if do_relaunch:
        print("\n4. relaunch: fresh GPU logger + detached make "
              + ("smoke" if pl["smoke"] else "all" if len(pl["tasks"]) > 1 else f"{pl['tasks'][0]}-all"))
        if pl["other_running"]:
            print(f"   note: {len(pl['other_running'])} process(es) of the other track are still running and will "
                  "share the GPUs")
    print("\nnot touched: HF model/dataset cache, .env, other tracks' results, MLflow runs")


def _commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "log", "-1", "--format=%h %s"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or "?"
    except (OSError, subprocess.SubprocessError):
        return "?"


def confirm(pl: dict) -> bool:
    word = "smoke" if pl["smoke"] else pl["tasks"][0] if len(pl["tasks"]) == 1 else "all"
    if not sys.stdin.isatty():
        print("\n[reset] not a terminal: re-run with --yes to confirm (or --dry-run to only look)")
        return False
    answer = input(f"\nType '{word}' to go ahead: ").strip()
    return answer == word


def stop_until_quiet(tasks: list[str], smoke: bool, timeout: float, rounds: int = 5) -> list[int]:
    """Re-scan the track's processes and stop them until a scan finds none:
    the plan was printed before the confirmation prompt, and a driver may
    have started a new job since. Returns what is still running (empty: quiet)."""
    stopped = 0
    for _ in range(rounds):
        found = track_processes(tasks, smoke)
        if not found:
            print(f"   stopped {stopped} process(es)" if stopped else "   nothing to stop")
            return []
        for pid, why in sorted(found.items()):
            print(f"   stopping {pid}  {why}")
        survivors = stop(sorted(found), timeout)
        if survivors:
            return survivors
        stopped += len(found)
        time.sleep(1.0)  # let a dying driver's last spawn show up in the next scan
    return sorted(track_processes(tasks, smoke))


def execute(pl: dict, archive: bool, timeout: float) -> bool:
    print("\n== stopping jobs")
    survivors = stop_until_quiet(pl["tasks"], pl["smoke"], timeout)
    if survivors:
        print(f"[reset] ABORT: still running: {survivors}; nothing deleted")
        return False

    print("== results")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for root, _ in pl["roots"]:
        if not root.exists():
            print(f"   {root} does not exist")
        elif archive:
            dest = root.with_name(f"{root.name}.{stamp}")
            root.rename(dest)
            print(f"   archived {root} -> {dest}")
        else:
            print(f"   deleting {root} ...", flush=True)
            shutil.rmtree(root)
            print("   deleted")
    print("== logs and leftovers")
    for path in pl["logs"] + pl["hf_tmp"]:
        path.unlink(missing_ok=True)
        print(f"   removed {path}")
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task", choices=["all", *status.TRACKS])
    parser.add_argument("--smoke", action="store_true", help="reset the smoke tree (<RESULTS_DIR>/smoke/<task>)")
    parser.add_argument("--archive", action="store_true", help="rename the results dir instead of deleting it")
    parser.add_argument("--relaunch", action="store_true", help="start the GPU logger and the track again, detached")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds to wait after SIGTERM (default 60)")
    args = parser.parse_args(argv)

    pl = plan(args.task, args.smoke)
    show(pl, args.archive, args.relaunch)
    if args.dry_run:
        print("\n[reset] dry run: nothing changed")
        return
    if not args.yes and not confirm(pl):
        print("[reset] cancelled: nothing changed")
        sys.exit(1)
    if not execute(pl, args.archive, args.timeout):
        sys.exit(1)
    if args.relaunch:
        print("== relaunch")
        for line in relaunch(args.task, args.smoke):
            print(f"   {line}")
        print("\nwatch it with: make status" + ("" if args.smoke else f" TASK={args.task}" if args.task != "all" else ""))
    else:
        print("\n[reset] done. Start again with: "
              + ("make smoke" if args.smoke else f"nohup make {args.task}-all > {args.task}-all.log 2>&1 &"
                 if args.task != "all" else "nohup make all > all.log 2>&1 &"))


if __name__ == "__main__":
    main()
