"""Freeze a results tree's evidence into git: evidence/<task>/<date>_<name>/.

A snapshot is what was measured at one point, exported by this script and
never edited by hand (evidence/README.md). It holds small things only:

  MANIFEST.json   git commit, date, seeds, eval settings, sweeps, package versions,
                  host, each run's training time, every file and its size
  configs/        base/eval/task/stage yamls and sweeps.yaml as run
  figures/        <results_root>/figures/*.png|pdf
  tables/         frontier.csv, frontier_seeds.csv, pairs stats, stop probes,
                  training_curves.csv (trainer_state log_history, long format),
                  compare_<i>.txt (analysis/compare_math.py, lines from snapshot.yaml)
  examples/       a few responses to the same questions from chosen checkpoints

What to compare and which examples to show come from evidence/<task>/snapshot.yaml,
reviewed in git, so the comparisons are fixed before the numbers are seen.
Checkpoints and *.samples.jsonl never go in (the snapshot must stay under --max-mb).

Usage:
    python analysis/export_evidence.py TASK NAME [--root RESULTS_ROOT] [--date YYYY-MM-DD]
        [--supersedes math/<snapshot> ...] [--note TEXT] [--max-mb 20] [--allow-stale] [--dry-run]
    make evidence TASK=math NAME=seeds17-19_full-test [EVIDENCE_ARGS="--supersedes math/..."]

It refuses figures older than the eval rows (run make math-plots first, or
--allow-stale) and a snapshot dir that already exists; it rewrites the
snapshot index in evidence/README.md from every MANIFEST.json.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import platform
import random
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from paths import default_root

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = REPO_ROOT / "evidence"
PACKAGES = ["torch", "transformers", "trl", "vllm", "datasets", "accelerate", "lm-eval", "omegaconf", "deepspeed",
            "flash-attn", "pandas", "matplotlib"]
TRAINER = {"sft-base": "sft", "sft": "sft", "dpo": "dpo", "dpo-nll": "dpo", "grpo": "grpo", "ppo": "ppo", "rm": "rm"}
INDEX_BEGIN, INDEX_END = "<!-- snapshots:begin -->", "<!-- snapshots:end -->"
MAX_RESPONSE_CHARS = 1500


def run_seed(name: str) -> int | None:
    m = re.search(r"_seed(\d+)$", name)
    return int(m[1]) if m else None


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def stale_figures(root: Path) -> str | None:
    """Why the figures don't reflect the eval rows, or None."""
    csv_path = root / "frontier.csv"
    if not csv_path.exists():
        return "no frontier.csv (run the track's plots first)"
    rows = [p for p in (root / "eval").glob("*/*.jsonl") if not p.name.endswith(".samples.jsonl")]
    newer = [p for p in rows if p.stat().st_mtime > csv_path.stat().st_mtime]
    if newer:
        return f"{len(newer)} eval rows are newer than frontier.csv (e.g. {newer[0].parent.name}/{newer[0].name})"
    return None


def copy_configs(task: str, out: Path) -> None:
    configs = REPO_ROOT / "configs"
    for path in [configs / "base.yaml", configs / "eval.yaml", configs / "tasks" / f"{task}.yaml",
                 *sorted((configs / task).glob("*.yaml"))]:
        if path.exists():
            dest = out / "configs" / path.relative_to(configs)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)


def copy_figures_and_tables(root: Path, out: Path) -> None:
    (out / "figures").mkdir(parents=True)
    (out / "tables").mkdir(parents=True)
    for fig in sorted((root / "figures").glob("*")):
        if fig.suffix in (".png", ".pdf", ".svg"):
            shutil.copy2(fig, out / "figures" / fig.name)
    for name in ("frontier.csv", "frontier_seeds.csv"):
        if (root / name).exists():
            shutil.copy2(root / name, out / "tables" / name)
    for stats in sorted((root / "data").glob("*/stats.json")):
        shutil.copy2(stats, out / "tables" / f"pairs_{stats.parent.name}.json")
    for probe in sorted((root / "runs").glob("*/stop_probe.json")):
        shutil.copy2(probe, out / "tables" / f"stop_probe_{probe.parent.name}.json")


def latest_trainer_state(run_dir: Path) -> dict | None:
    ckpts = sorted((p for p in run_dir.glob("checkpoint-*") if p.is_dir()),
                   key=lambda p: int(re.search(r"(\d+)$", p.name)[1]))
    for ckpt in reversed(ckpts):
        try:
            return json.loads((ckpt / "trainer_state.json").read_text())
        except (OSError, ValueError):
            continue
    return None


def training_curves(root: Path, out: Path) -> dict:
    """tables/training_curves.csv (run, arm, seed, step, epoch, metric, value) from
    each run's latest trainer_state.json; returns per-run training time / steps."""
    from posttrain_lab.status import METRICS

    runs: dict[str, dict] = {}
    with (out / "tables" / "training_curves.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "arm", "seed", "step", "epoch", "metric", "value"])
        for run_dir in sorted((root / "runs").glob("*_seed*")):
            state = latest_trainer_state(run_dir)
            if state is None:
                continue
            arm = run_dir.name.split("_")[0]
            keys = METRICS.get(TRAINER.get(arm, ""), ["loss"])
            info = {"global_step": state.get("global_step"), "max_steps": state.get("max_steps"),
                    "checkpoints": len(list(run_dir.glob("checkpoint-*"))), "done": (run_dir / "DONE").exists()}
            for entry in state.get("log_history") or []:
                if "train_runtime" in entry:
                    info["train_runtime_s"] = entry["train_runtime"]
                    continue
                for key in keys:
                    if isinstance(entry.get(key), (int, float)):
                        writer.writerow([run_dir.name, arm, run_seed(run_dir.name), entry.get("step"),
                                         entry.get("epoch"), key, entry[key]])
            runs[run_dir.name] = info
    return runs


def run_compares(root: Path, lines: list[str], out: Path) -> list[dict]:
    """Each compare_math.py argument line -> tables/compare_<i>.txt (its command and
    output; a line that cannot run, e.g. a seed not trained yet, records why)."""
    import compare_math

    done = []
    for i, line in enumerate(lines, 1):
        buf = io.StringIO()
        status = "ok"
        with contextlib.redirect_stdout(buf):
            try:
                compare_math.main([str(root), *shlex.split(line)])
            except SystemExit as exc:  # compare_math reports bad specs with sys.exit(message)
                status = "failed"
                print(f"\n[not run] {exc.code}")
        path = out / "tables" / f"compare_{i}.txt"
        path.write_text(f"$ python analysis/compare_math.py <root> {line}\n\n{buf.getvalue()}")
        if status != "ok":
            print(f"[evidence] compare line {i} did not run: {line}", file=sys.stderr)
        done.append({"args": line, "file": str(path.relative_to(out)), "status": status})
    return done


def export_examples(root: Path, spec: dict, out: Path) -> dict | None:
    """examples/examples.md + .jsonl: sample 0 of each checkpoint on the same
    questions, the ones the checkpoints disagree on first (fixed seed)."""
    import compare_math

    checkpoints = list(spec.get("checkpoints") or [])
    if not checkpoints:
        return None
    seed = spec.get("seed")
    n = int(spec.get("prompts", 6))
    rows: dict[str, dict[str, dict]] = {}
    for ck in checkpoints:
        try:
            path = compare_math.resolve(root / "eval", ck, None if seed is None else int(seed))
        except SystemExit as exc:
            print(f"[evidence] examples: skipping {ck}: {exc.code}", file=sys.stderr)
            continue
        with path.open() as f:
            for line in f:
                s = json.loads(line) if line.strip() else None
                if s and s.get("sample_idx", 0) == 0:
                    rows.setdefault(s["prompt"], {})[ck] = s
    names = [ck for ck in checkpoints if any(ck in r for r in rows.values())]
    shared = sorted(q for q, r in rows.items() if all(ck in r for ck in names))
    if not names or not shared:
        return None
    rng = random.Random(0)
    disagree = [q for q in shared if len({bool(rows[q][ck].get("correct")) for ck in names}) > 1]
    rest = [q for q in shared if q not in disagree]
    rng.shuffle(disagree)
    rng.shuffle(rest)
    picked = (disagree + rest)[:n]

    (out / "examples").mkdir()
    lines = [f"# Example responses (sample 0 of each checkpoint{'' if seed is None else f', seed {seed}'})", "",
             "Questions the checkpoints disagree on come first. Exported by analysis/export_evidence.py.", ""]
    with (out / "examples" / "examples.jsonl").open("w") as f:
        for k, q in enumerate(picked, 1):
            lines += [f"## Q{k}", "", q.strip(), ""]
            for ck in names:
                s = rows[q][ck]
                mark = "correct" if s.get("correct") else "wrong"
                stop = "" if s.get("finished", True) else ", hit max_new_tokens"
                text = s["response"]
                if len(text) > MAX_RESPONSE_CHARS:
                    text = text[:MAX_RESPONSE_CHARS] + f" [... {len(s['response']) - MAX_RESPONSE_CHARS} more chars]"
                lines += [f"**{ck}**: {mark}{stop}", "", "```text", text, "```", ""]
                f.write(json.dumps({"question": q, "checkpoint": ck, "prompt_id": s.get("prompt_id"),
                                    "correct": s.get("correct"), "finished": s.get("finished"),
                                    "response": s["response"]}) + "\n")
    (out / "examples" / "examples.md").write_text("\n".join(lines))
    return {"seed": seed, "checkpoints": names, "prompts": len(picked)}


def package_versions() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def eval_settings(task: str) -> dict:
    from omegaconf import OmegaConf

    from posttrain_lab.config import load_config

    cfg = load_config("eval.yaml", [f"task={task}"])
    ev = OmegaConf.to_container(cfg.eval, resolve=True)
    keep = ("num_prompts", "samples_per_prompt", "pass_k", "metric_groups", "temperature", "top_p", "max_new_tokens",
            "kl_estimator", "harness")
    return {k: ev[k] for k in keep if k in ev}


def file_sizes(out: Path) -> dict[str, int]:
    return {str(p.relative_to(out)): p.stat().st_size for p in sorted(out.rglob("*")) if p.is_file()}


def write_index() -> None:
    """The snapshot table in evidence/README.md, rebuilt from every MANIFEST.json."""
    manifests = []
    for path in sorted(EVIDENCE.glob("*/*/MANIFEST.json")):
        try:
            manifests.append((path.parent.relative_to(EVIDENCE).as_posix(), json.loads(path.read_text())))
        except ValueError:
            continue
    superseded = {old: snap for snap, m in manifests for old in m.get("supersedes", [])}
    rows = ["| snapshot | date | seeds | eval prompts | commit | note | superseded by |", "|---|---|---|---|---|---|---|"]
    for snap, m in manifests:
        seeds = ", ".join(map(str, m.get("seeds", []))) or "?"
        rows.append(f"| [{snap}]({snap}/) | {m.get('date', '?')} | {seeds} | "
                    f"{(m.get('eval') or {}).get('num_prompts', '?')} | `{(m.get('git') or {}).get('commit', '?')}` | "
                    f"{m.get('note') or ''} | {superseded.get(snap, '')} |")
    table = "\n".join(rows) if manifests else "_No snapshots yet: `make evidence TASK=math NAME=<name>`._"
    readme = EVIDENCE / "README.md"
    text = readme.read_text() if readme.exists() else f"# Evidence\n\n## Snapshots\n\n{INDEX_BEGIN}\n{INDEX_END}\n"
    if INDEX_BEGIN not in text:
        text += f"\n## Snapshots\n\n{INDEX_BEGIN}\n{INDEX_END}\n"
    head, rest = text.split(INDEX_BEGIN, 1)
    tail = rest.split(INDEX_END, 1)[1] if INDEX_END in rest else "\n"
    readme.write_text(f"{head}{INDEX_BEGIN}\n{table}\n{INDEX_END}{tail}")


def load_spec(task: str) -> dict:
    path = EVIDENCE / task / "snapshot.yaml"
    if not path.exists():
        return {}
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("task", choices=["math", "helpful"])
    ap.add_argument("name", help="short snapshot name, e.g. seeds17-19_full-test")
    ap.add_argument("--root", help="results root (default: the task's, from RESULTS_DIR)")
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--supersedes", nargs="*", default=[], metavar="SNAPSHOT", help="e.g. math/2026-10-22_seed17")
    ap.add_argument("--note", default="", help="one line for the index")
    ap.add_argument("--max-mb", type=float, default=20.0)
    ap.add_argument("--allow-stale", action="store_true", help="export even if figures are older than eval rows")
    ap.add_argument("--dry-run", action="store_true", help="check and print the plan only")
    args = ap.parse_args(argv)

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.name):
        ap.error("NAME: letters, digits, '.', '_' and '-' only")
    root = Path(args.root or default_root(args.task))
    if not (root / "eval").is_dir():
        sys.exit(f"[evidence] no eval results under {root}")
    stale = stale_figures(root)
    if stale and not args.allow_stale:
        sys.exit(f"[evidence] figures are stale: {stale}. Run `make {'math-' if args.task == 'math' else ''}plots` "
                 "first (or pass --allow-stale)")
    for old in args.supersedes:
        if not (EVIDENCE / old / "MANIFEST.json").exists():
            sys.exit(f"[evidence] --supersedes {old}: no such snapshot under {EVIDENCE}")
    snap = f"{args.date}_{args.name}"
    out = EVIDENCE / args.task / snap
    if out.exists():
        sys.exit(f"[evidence] {out} exists: snapshots are never overwritten; pick another NAME")
    spec = load_spec(args.task)
    print(f"[evidence] {root} -> {out.relative_to(REPO_ROOT)}"
          f" ({len(spec.get('compare') or [])} comparisons, examples: {bool((spec.get('examples') or {}).get('checkpoints'))})")
    if args.dry_run:
        return

    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        copy_configs(args.task, tmp)
        copy_figures_and_tables(root, tmp)
        runs = training_curves(root, tmp)
        compares = run_compares(root, list(spec.get("compare") or []), tmp)
        examples = export_examples(root, spec.get("examples") or {}, tmp)
        seeds = sorted({s for name in runs if (s := run_seed(name)) is not None})
        manifest = {
            "snapshot": f"{args.task}/{snap}", "task": args.task, "date": args.date, "note": args.note,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git": {"commit": git("rev-parse", "--short", "HEAD"), "subject": git("log", "-1", "--format=%s"),
                    "dirty": bool(git("status", "--porcelain", "-uno"))},
            "results_root": str(root), "host": platform.node(), "seeds": seeds,
            "eval": eval_settings(args.task), "packages": package_versions(), "python": platform.python_version(),
            "runs": runs, "compare": compares, "examples": examples, "supersedes": args.supersedes,
            "figures_stale": stale,
        }
        (tmp / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        sizes = file_sizes(tmp)
        total = sum(sizes.values())
        if total > args.max_mb * 1024 ** 2:
            big = sorted(sizes.items(), key=lambda kv: -kv[1])[:5]
            sys.exit(f"[evidence] snapshot is {total / 1024 ** 2:.1f} MB > --max-mb {args.max_mb}; largest: "
                     + ", ".join(f"{p} ({s / 1024 ** 2:.1f} MB)" for p, s in big))
        manifest["files"], manifest["total_bytes"] = sizes, total
        (tmp / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        tmp.rename(out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    write_index()
    print(f"[evidence] wrote {out.relative_to(REPO_ROOT)} ({total / 1024 ** 2:.1f} MB, {len(sizes)} files); "
          f"index: evidence/README.md. Next: update evidence/FINDINGS.md and LOG.md, then commit.")


if __name__ == "__main__":
    main()
