"""Paired comparison of checkpoints on the same eval prompts (math or helpful).

Every checkpoint is scored on the same eval prompts, so the noise that
matters for "is A better than B" is the per-prompt difference, not each
point's own standard error: a prompt that is hard for A is usually hard for B
too. This prints each checkpoint's score ± SE (over prompts) and, for every
checkpoint after the first, its paired difference to the first with a 95% CI.

The score per prompt (--metric, default auto from the samples): pass1, the
solve rate (math); judge, the held-out Skywork judge's mean reward (helpful's
headline); train_rm, the training reward model's (the reward the RL arms
optimized: its gap to the judge shows reward over-optimization).

Usage:
    python analysis/compare_math.py [results_root] [--seed N | --pool] [--metric M] RUN@STEP RUN@STEP [...]
    python analysis/compare_math.py dpo_beta0.3@116 grpo_beta0.01@374 grpo_beta0.0@374
    python analysis/compare_math.py --pool dpo_beta0.3@ckpt10 grpo_beta0.01@last
    python analysis/compare_math.py --pool grpo_beta0.01@kl3 dpo_beta0.3@kl3 dpo-nll_beta0.3@kl3
    python analysis/compare_math.py --task helpful --pool ppo@kl5 grpo@kl5 dpo@kl5 dpo-nll@kl5

RUN is a run dir name, or a unique prefix of one (dpo_beta0.3 -> dpo_beta0.3_seed17);
STEP is a checkpoint step, 'last', ckptK (the run's K-th checkpoint), or klX:
the run's checkpoint whose kl_k1 (from its eval row) is nearest X nats, picked
per seed. klX looks at KL only, never at accuracy. The output names the
checkpoint it picked and marks with ! any whose KL is more than 0.3 nats from X
(the run has no nearer one), and every paired line whose two checkpoints are
more than 0.3 nats apart.
With klX, RUN may also name an arm, e.g. dpo (matches dpo_beta*, not
dpo-nll_*): the checkpoint nearest X nats across all of that arm's runs of the
seed, again by KL alone, so a comparison can be fixed before knowing how far
each beta moves; the output names the run it picked.
Picking each run's best checkpoint on these same prompts flatters it: compare
checkpoints chosen some other way (matched KL, a fixed step) where you can.

Several training seeds: a prefix that matches more than one seed's run needs
  --seed N   compare seed N's runs (the same as RUN_seedN), or
  --pool     compare every seed that has all the checkpoints: the paired
             difference per seed, then their mean with a 95% CI from the
             spread ACROSS seeds (t with n_seeds - 1 df). That CI is the one
             that covers training randomness; the per-seed and prompt-pooled
             CIs only cover eval sampling for these particular trained models.
             The DPO arms' step counts differ between seeds (each trains on
             its own seed's pairs), and so does how far each step moves them:
             name their checkpoints by klX for matched KL, else ckptK or last.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from paths import default_root

# two-sided 95% t quantiles by degrees of freedom (no scipy needed)
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
       15: 2.131, 20: 2.086, 30: 2.042}
KL_TOL = 0.3  # nats: further apart than this is not "matched KL" (AGENTS.md)
# --metric -> (column label, the per-sample field averaged per prompt)
METRICS = {"pass1": ("pass@1", "correct"), "judge": ("judge", "score_judge"), "train_rm": ("train RM", "score_train_rm")}


def t95(df: int) -> float:
    if df in T95:
        return T95[df]
    smaller = [k for k in T95 if k < df]
    return T95[max(smaller)] if df <= 30 else 1.96


def run_seed(name: str) -> int | None:
    m = re.search(r"_seed(\d+)$", name)
    return int(m[1]) if m else None


def matching_runs(eval_root: Path, run: str) -> list[Path]:
    runs = [d for d in eval_root.iterdir() if d.is_dir() and (d.name == run or d.name.startswith(run + "_"))]
    exact = [d for d in runs if d.name == run]
    return exact if len(exact) == 1 else runs


def sample_files(run_dir: Path) -> dict[int, Path]:
    return {int(m[1]): p for p in run_dir.glob("checkpoint-*.samples.jsonl")
            if (m := re.fullmatch(r"checkpoint-(\d+)\.samples\.jsonl", p.name))}


def kl_target(step: str) -> float | None:
    """X for a klX step, else None."""
    m = re.fullmatch(r"kl(\d+(?:\.\d*)?|\.\d+)", step)
    return float(m[1]) if m else None


def checkpoint_file(run_dir: Path, spec: str, step: str) -> Path:
    files = sample_files(run_dir)
    if not files:
        sys.exit(f"{spec!r}: no samples files in {run_dir}")
    if (target := kl_target(step)) is not None:
        kls = {s: kl for s, p in files.items() if (kl := kl_of(p)) is not None}
        if not kls:
            sys.exit(f"{spec!r}: no kl_k1 in the eval rows of {run_dir.name}")
        # nearest by KL alone, never by accuracy; a tie goes to the earlier checkpoint
        return files[min(sorted(kls), key=lambda s: abs(kls[s] - target))]
    if step == "last":
        return files[max(files)]
    if m := re.fullmatch(r"ckpt(\d+)", step):
        k, steps = int(m[1]), sorted(files)
        if not 1 <= k <= len(steps):
            sys.exit(f"{spec!r}: {run_dir.name} has {len(steps)} evaluated checkpoints, no ckpt{k}")
        return files[steps[k - 1]]
    if int(step) not in files:
        sys.exit(f"{spec!r}: no checkpoint-{step} samples in {run_dir.name} (have {sorted(files)})")
    return files[int(step)]


def checkpoint_across(run_dirs: list[Path], spec: str, step: str) -> Path:
    """The checkpoint a spec names among several runs of one seed: only klX can
    choose across runs (nearest X nats over every checkpoint of every run)."""
    target = kl_target(step)
    if target is None:
        names = ", ".join(sorted(d.name for d in run_dirs))
        sys.exit(f"{spec!r}: matches {names}; name one run, or use klX to pick by KL across them")
    candidates = []
    for d in sorted(run_dirs, key=lambda d: d.name):
        for s, p in sorted(sample_files(d).items()):
            if (kl := kl_of(p)) is not None:
                candidates.append((abs(kl - target), d.name, s, p))
    if not candidates:
        sys.exit(f"{spec!r}: no kl_k1 in the eval rows of {', '.join(sorted(d.name for d in run_dirs))}")
    return min(candidates, key=lambda c: c[:3])[3]  # nearest; a tie goes to the first run, then the earlier step


def split_spec(spec: str) -> tuple[str, str]:
    if "@" not in spec:
        sys.exit(f"{spec!r}: expected RUN@STEP (e.g. grpo_beta0.01@374, dpo_beta0.3@last, dpo_beta0.3@ckpt10 "
                 "or dpo_beta0.3@kl3)")
    return spec.rsplit("@", 1)


def resolve(eval_root: Path, spec: str, seed: int | None = None) -> Path:
    """The samples file a RUN@STEP spec names (seed: that seed's run)."""
    run, step = split_spec(spec)
    runs = matching_runs(eval_root, run)
    if seed is not None and run_seed(run) is None:
        runs = [d for d in runs if run_seed(d.name) == seed]
        run = f"{run} (seed {seed})"
    seeds = sorted({s for d in runs if (s := run_seed(d.name)) is not None})
    if len(runs) > 1 and len(seeds) == 1:
        return checkpoint_across(runs, spec, step)  # an arm (klX only: anything else says so), one seed
    if len(runs) != 1:
        found = ", ".join(sorted(d.name for d in runs)) or "none"
        hint = f" (seeds {', '.join(map(str, seeds))}: pass --seed N or --pool)" if len(seeds) > 1 else ""
        sys.exit(f"{spec!r}: run {run!r} matches {found} under {eval_root}{hint}")
    return checkpoint_file(runs[0], spec, step)


def resolve_seeds(eval_root: Path, spec: str) -> dict[int, Path]:
    """--pool: seed -> samples file for every seed whose run the spec matches."""
    run, step = split_spec(spec)
    by_seed: dict[int, list[Path]] = defaultdict(list)
    for d in matching_runs(eval_root, run):
        if (s := run_seed(d.name)) is not None:
            by_seed[s].append(d)
    for s, dirs in by_seed.items():
        if len(dirs) > 1 and kl_target(step) is None:
            sys.exit(f"{spec!r}: run {run!r} matches {', '.join(sorted(d.name for d in dirs))} for seed {s}")
    if not by_seed:
        sys.exit(f"{spec!r}: run {run!r} matches no seeded run under {eval_root}")
    return {s: checkpoint_file(dirs[0], spec, step) if len(dirs) == 1 else checkpoint_across(dirs, spec, step)
            for s, dirs in sorted(by_seed.items())}


def per_prompt(path: Path, field: str = "correct") -> dict[str, float]:
    """The mean of a per-sample field per question (math: correct -> the solve
    rate; helpful: a reward), keyed by the question text: prompt ids are
    positions (gsm8k-<i>) and point at different questions in a 512-prompt
    subsample and the full test split, so they must not be used to pair."""
    by_prompt: dict[str, list[float]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                if field not in s:
                    sys.exit(f"{path}: samples have no {field!r} (wrong --metric for this track?)")
                by_prompt[s.get("prompt") or s["prompt_id"]].append(float(s[field]))
    return {q: sum(v) / len(v) for q, v in by_prompt.items()}


def detect_metric(path: Path) -> str:
    """--metric auto: pass1 for math samples (correct), judge for helpful ones (score_judge)."""
    with path.open() as f:
        first = next((json.loads(line) for line in f if line.strip()), {})
    for metric in ("pass1", "judge"):
        if METRICS[metric][1] in first:
            return metric
    sys.exit(f"{path}: samples have neither 'correct' nor 'score_judge': pass --metric")


def mean_se(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    se = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1) / n) if n > 1 else float("nan")
    return mean, se


def kl_of(samples_path: Path) -> float | None:
    row_path = samples_path.with_name(samples_path.name.replace(".samples.jsonl", ".jsonl"))
    try:
        return json.loads(row_path.read_text().splitlines()[0]).get("kl_k1")
    except (OSError, IndexError, ValueError):
        return None


def fmt_kl(kl: float | None) -> str:
    return str(kl if kl is None else round(kl, 2))


def picked(spec: str, path: Path) -> str:
    """For a RUN@klX spec, the checkpoint it resolved to and, past KL_TOL, how far its KL is from X."""
    run, step_spec = split_spec(spec)
    target = kl_target(step_spec)
    if target is None:
        return ""
    step = int(re.fullmatch(r"checkpoint-(\d+)\.samples\.jsonl", path.name)[1])
    picked_run = re.sub(r"_seed\d+$", "", path.parent.name)
    arm_level = picked_run != re.sub(r"_seed\d+$", "", run)  # an arm-level spec: say which run it picked
    note = f"  {path.parent.name if arm_level else ''}{' ' if arm_level else ''}" \
           f"ckpt{sorted(sample_files(path.parent)).index(step) + 1} (step {step})"
    kl = kl_of(path)
    if kl is not None and abs(kl - target) > KL_TOL:
        note += f"  ! {kl - target:+.2f} nats from {target:g}"
    return note


def kl_apart(base: Path, path: Path) -> str:
    """A note when two compared checkpoints are further apart in KL than KL_TOL."""
    a, b = kl_of(base), kl_of(path)
    if a is None or b is None or abs(b - a) <= KL_TOL:
        return ""
    return f"  ! KL {b - a:+.2f} nats from the baseline: not matched"


def unmatched_note(notes: list[str]) -> None:
    if any("!" in n for n in notes):
        print(f"! klX checkpoint more than {KL_TOL} nats from its target: the run has no nearer one, so it is not "
              "matched KL")


def verdict(lo: float, hi: float) -> str:
    return "better" if lo > 0 else "worse" if hi < 0 else "no clear difference"


def paired(base: dict[str, float], sc: dict[str, float]) -> tuple[float, float, int] | None:
    shared = sorted(set(base) & set(sc))
    if len(shared) < 2:
        return None
    diff, se = mean_se([sc[p] - base[p] for p in shared])
    return diff, se, len(shared)


def compare_one(specs: list[str], paths: list[Path], metric: str = "pass1") -> None:
    label, field = METRICS[metric]
    scores = [per_prompt(p, field) for p in paths]

    print(f"{'checkpoint':<44} {'prompts':>7} {label:>7} {'± SE':>6} {'KL k1':>7}")
    notes = []
    for spec, path, sc in zip(specs, paths, scores):
        mean, se = mean_se(list(sc.values()))
        notes.append(picked(spec, path))
        print(f"{spec:<44} {len(sc):>7} {mean:>7.3f} {se:>6.3f} {fmt_kl(kl_of(path)):>7}{notes[-1]}")
    unmatched_note(notes)

    base_spec, base = specs[0], scores[0]
    print(f"\npaired vs {base_spec} (same prompts; 95% CI = ± 1.96 SE)")
    for spec, path, sc in zip(specs[1:], paths[1:], scores[1:]):
        shared = set(base) & set(sc)
        if len(shared) < max(len(base), len(sc)):
            print(f"  note: {spec} ({len(sc)} prompts) and {base_spec} ({len(base)}) share {len(shared)} questions: "
                  "different eval prompt sets (re-run the track's eval to put every checkpoint on the same one)")
        res = paired(base, sc)
        if res is None:
            print(f"  {spec}: too few shared prompts")
            continue
        diff, se, n = res
        lo, hi = diff - 1.96 * se, diff + 1.96 * se
        print(f"  {spec:<42} {diff:+.3f} ± {se:.3f}  [{lo:+.3f}, {hi:+.3f}]  {verdict(lo, hi)}  ({n} prompts)"
              f"{kl_apart(paths[0], path)}")


def compare_pooled(specs: list[str], files: list[dict[int, Path]], metric: str = "pass1") -> None:
    label, field = METRICS[metric]
    seeds = sorted(set.intersection(*(set(f) for f in files)))
    every = sorted(set.union(*(set(f) for f in files)))
    if not seeds:
        sys.exit("--pool: no seed has every checkpoint named (" +
                 "; ".join(f"{s}: seeds {sorted(f)}" for s, f in zip(specs, files)) + ")")
    if seeds != every:
        print(f"note: seeds {sorted(set(every) - set(seeds))} lack some of these checkpoints and are left out")
    scores = [{s: per_prompt(f[s], field) for s in seeds} for f in files]
    print(f"pooled over seeds {', '.join(map(str, seeds))}\n")

    print(f"{'checkpoint':<36} {'seed':>5} {'prompts':>7} {label:>7} {'± SE':>6} {'KL k1':>7}")
    notes = []
    for spec, f, sc in zip(specs, files, scores):
        means = []
        for s in seeds:
            mean, se = mean_se(list(sc[s].values()))
            means.append(mean)
            notes.append(picked(spec, f[s]))
            print(f"{spec:<36} {s:>5} {len(sc[s]):>7} {mean:>7.3f} {se:>6.3f} {fmt_kl(kl_of(f[s])):>7}{notes[-1]}")
        m, se = mean_se(means)
        print(f"{spec:<36} {'mean':>5} {'':>7} {m:>7.3f} {se:>6.3f}   (± SE across seeds)")
    unmatched_note(notes)

    base_spec, base = specs[0], scores[0]
    n = len(seeds)
    print(f"\npaired vs {base_spec}, per seed (same prompts; ± 1.96 SE over prompts) and across seeds "
          f"(mean of the per-seed differences; ± t{n - 1} SE across seeds)")
    for spec, f, sc in zip(specs[1:], files[1:], scores[1:]):
        diffs = []
        for s in seeds:
            res = paired(base[s], sc[s])
            if res is None:
                print(f"  {spec} seed {s}: too few shared prompts")
                continue
            diff, se, k = res
            diffs.append(diff)
            lo, hi = diff - 1.96 * se, diff + 1.96 * se
            print(f"  {spec:<34} seed {s:<4} {diff:+.3f} ± {se:.3f}  [{lo:+.3f}, {hi:+.3f}]  {verdict(lo, hi)}  "
                  f"({k} prompts){kl_apart(files[0][s], f[s])}")
        # every seed's per-prompt difference, averaged over seeds: eval noise of the seed-mean only
        shared = sorted(set.intersection(*(set(base[s]) & set(sc[s]) for s in seeds)))
        if len(shared) > 1:
            pooled = [sum(sc[s][p] - base[s][p] for s in seeds) / n for p in shared]
            diff, se = mean_se(pooled)
            print(f"  {spec:<34} {'prompts':<9} {diff:+.3f} ± {se:.3f}  [{diff - 1.96 * se:+.3f}, "
                  f"{diff + 1.96 * se:+.3f}]  (seed-mean per prompt; eval noise only)")
        if len(diffs) < 2:
            print(f"  {spec:<34} {'seeds':<9} one seed: no across-seed CI (train another: README 'More seeds')")
            continue
        diff, se = mean_se(diffs)
        t = t95(len(diffs) - 1)
        lo, hi = diff - t * se, diff + t * se
        print(f"  {spec:<34} {'seeds':<9} {diff:+.3f} ± {se:.3f}  [{lo:+.3f}, {hi:+.3f}]  {verdict(lo, hi)}  "
              f"({len(diffs)} seeds, t{len(diffs) - 1} = {t:.2f})")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("specs", nargs="*", help="[results_root] RUN@STEP RUN@STEP [...]")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seed", type=int, help="compare this training seed's runs")
    group.add_argument("--pool", action="store_true", help="every seed: per-seed and across-seed differences")
    parser.add_argument("--metric", choices=["auto", *METRICS], default="auto",
                        help="the score per prompt (auto: pass1 for math samples, judge for helpful ones)")
    parser.add_argument("--task", choices=["math", "helpful"], default="math",
                        help="whose results root to use when none is given")
    args = parser.parse_intermixed_args(argv)  # [results_root] --pool RUN@STEP ...
    specs = args.specs
    if specs and Path(specs[0]).is_dir():
        root, specs = Path(specs[0]), specs[1:]
    else:
        root = Path(default_root(args.task))
    if len(specs) < 2:
        sys.exit(__doc__)
    eval_root = root / "eval"
    if args.pool:
        files = [resolve_seeds(eval_root, s) for s in specs]
        metric = detect_metric(next(iter(files[0].values()))) if args.metric == "auto" else args.metric
        compare_pooled(specs, files, metric)
    else:
        paths = [resolve(eval_root, s, args.seed) for s in specs]
        metric = detect_metric(paths[0]) if args.metric == "auto" else args.metric
        compare_one(specs, paths, metric)


if __name__ == "__main__":
    main(sys.argv[1:])
