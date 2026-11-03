"""Build a self-contained HTML viewer of a snapshot: evidence/viewer/<task>.html.

The snapshots under evidence/<task>/ hold the tables, figures, pre-registered
comparisons and example responses (evidence/README.md), but reading them means
opening a dozen CSV and text files. This renders one page from them so the
accuracy-vs-KL frontier, the per-seed numbers, every compare_<i>.txt and the
example responses can be browsed in a browser with nothing installed:

  - an interactive frontier (inline SVG, no JavaScript libraries): one point
    per checkpoint and seed, seed-mean lines per run, hover for the numbers,
    checkboxes to show or hide arms;
  - the seed-mean table (frontier_seeds.csv), one row per checkpoint position;
  - each pre-registered comparison from snapshot.yaml, as the exporter saved it;
  - the example responses, grouped by question.

It reads a snapshot and never writes into it (snapshots are never edited).
The page embeds no external resource, so it works offline and from GitHub's
raw view. By default it renders the task's latest snapshot: the one no other
snapshot supersedes (MANIFEST.json "supersedes").

Usage:
    python analysis/build_viewer.py math [helpful ...]
    python analysis/build_viewer.py math --snapshot math/2026-10-25_seeds17-19_dense-grpo --out /tmp/x.html
    make viewer
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = REPO_ROOT / "evidence"

# display order, label and color per arm (frontier_math.py's palette, roughly)
ARMS = {
    "sft-base": ("π_SFT (reference)", "#444444"),
    "sft": ("SFT (imitation)", "#1f77b4"),
    "dpo": ("DPO", "#2ca08c"),
    "dpo-nll": ("DPO + NLL", "#e377a8"),
    "grpo": ("GRPO", "#2a8f2a"),
    "ppo": ("PPO", "#d62728"),
}
# (metric column, axis label) per task; the first is the y axis of the chart
METRIC = {"math": ("pass1", "pass@1 (test accuracy)"), "helpful": ("reward_judge", "judge reward")}
SEED_TABLE_COLUMNS = ["arm", "hyperparam", "ckpt_index", "n_seeds", "step", "kl_k1", "kl_k1_seed_se"]


def latest_snapshot(task: str) -> Path | None:
    """The task's snapshot that no other snapshot supersedes (newest by name if several)."""
    manifests = {p.parent: json.loads(p.read_text()) for p in sorted((EVIDENCE / task).glob("*/MANIFEST.json"))}
    superseded = {old for m in manifests.values() for old in m.get("supersedes", [])}
    live = [d for d in manifests if f"{task}/{d.name}" not in superseded]
    return max(live) if live else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fnum(value: str, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return value or ""


def symlog(x: float) -> float:
    """The frontier figures' x scale: linear near 0, log10 above 1 nat."""
    return x if x < 1 else 1 + math.log10(x)


def run_label(row: dict[str, str]) -> str:
    return row["arm"] + (f" {row['hyperparam']}" if row.get("hyperparam") else "")


def frontier_svg(rows: list[dict[str, str]], metric: str, ylabel: str) -> str:
    """Inline SVG: per-seed points, seed-mean lines per run, <title> tooltips, one <g> per arm."""
    pts = [r for r in rows if r.get("kl_k1") not in (None, "") and r.get(metric) not in (None, "")]
    if not pts:
        return "<p>No rows with both kl_k1 and the metric.</p>"
    w, h, left, right, top, bottom = 960, 520, 70, 20, 20, 60
    xs = [symlog(float(r["kl_k1"])) for r in pts]
    ys = [float(r[metric]) for r in pts]
    x0, x1 = 0.0, max(xs) * 1.05 or 1.0
    y0, y1 = min(ys) - 0.02, max(ys) + 0.02

    def sx(x: float) -> float:
        return left + (x - x0) / (x1 - x0) * (w - left - right)

    def sy(y: float) -> float:
        return top + (y1 - y) / (y1 - y0) * (h - top - bottom)

    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px;font:12px sans-serif" '
           'xmlns="http://www.w3.org/2000/svg">']
    # axes and grid
    for tick in (0, 1, 3, 10, 30, 100):
        if symlog(tick) > x1:
            break
        gx = sx(symlog(tick))
        out.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{h - bottom}" stroke="#ddd"/>')
        out.append(f'<text x="{gx:.1f}" y="{h - bottom + 16}" text-anchor="middle">{tick}</text>')
    step = 0.05 if y1 - y0 < 0.6 else 0.1
    tick = math.ceil(y0 / step) * step
    while tick <= y1:
        gy = sy(tick)
        out.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{w - right}" y2="{gy:.1f}" stroke="#ddd"/>')
        out.append(f'<text x="{left - 6}" y="{gy + 4:.1f}" text-anchor="end">{tick:.2f}</text>')
        tick += step
    out.append(f'<text x="{(left + w - right) / 2:.0f}" y="{h - 12}" text-anchor="middle">'
               'KL(policy ‖ π_SFT), nats per sequence (k1; symlog)</text>')
    out.append(f'<text transform="translate(16,{(top + h - bottom) / 2:.0f}) rotate(-90)" text-anchor="middle">'
               f'{html.escape(ylabel)}</text>')
    # one group per arm, so the checkboxes can hide it
    for arm, (label, color) in ARMS.items():
        arm_rows = [r for r in pts if r["arm"] == arm]
        if not arm_rows:
            continue
        out.append(f'<g class="arm" data-arm="{arm}">')
        by_run: dict[str, list[dict[str, str]]] = {}
        for r in arm_rows:
            by_run.setdefault(run_label(r), []).append(r)
        for run, rrows in by_run.items():
            # seed-mean line per checkpoint position
            means: dict[str, list[dict[str, str]]] = {}
            for r in rrows:
                means.setdefault(r.get("ckpt_index", r.get("step", "")), []).append(r)
            line = sorted(((sum(float(x["kl_k1"]) for x in g) / len(g), sum(float(x[metric]) for x in g) / len(g))
                           for g in means.values()))
            if len(line) > 1:
                d = " ".join(f"{sx(symlog(x)):.1f},{sy(y):.1f}" for x, y in line)
                out.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="1.5" opacity="0.7"/>')
            for r in rrows:
                cx, cy = sx(symlog(float(r["kl_k1"]))), sy(float(r[metric]))
                tip = (f"{run} seed {r.get('seed', '')} ckpt {r.get('ckpt_index', '')} step {r.get('step', '')}\n"
                       f"KL {fnum(r['kl_k1'], 2)} nats, {metric} {fnum(r[metric])}")
                shape = (f'<rect x="{cx - 4:.1f}" y="{cy - 4:.1f}" width="8" height="8" fill="{color}"/>'
                         if arm == "sft-base" else f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{color}"/>')
                out.append(f'<g>{shape}<title>{html.escape(tip)}</title></g>')
        out.append("</g>")
    out.append("</svg>")
    boxes = " ".join(f'<label><input type="checkbox" checked data-arm="{arm}" onchange="toggleArm(this)"> '
                     f'<span style="color:{color}">■</span> {html.escape(label)}</label>'
                     for arm, (label, color) in ARMS.items() if any(r["arm"] == arm for r in pts))
    return f'<div class="legend">{boxes}</div>' + "".join(out)


def table(rows: list[dict[str, str]], columns: list[str]) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(fnum(r.get(c, '')))}</td>" for c in columns) + "</tr>"
                   for r in rows)
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def examples_html(snapshot: Path) -> str:
    path = snapshot / "examples" / "examples.jsonl"
    if not path.exists():
        return "<p>No examples in this snapshot.</p>"
    by_q: dict[str, list[dict]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            ex = json.loads(line)
            by_q.setdefault(ex["question"], []).append(ex)
    out = []
    for i, (q, exs) in enumerate(by_q.items(), 1):
        items = []
        for ex in exs:
            verdict = ex.get("correct")
            mark = "" if verdict in (None, "") else (" ✓" if str(verdict) == "True" else " ✗")
            items.append(f'<div class="ex"><b>{html.escape(str(ex["checkpoint"]))}</b>{mark}'
                         f'<pre>{html.escape(str(ex["response"]))}</pre></div>')
        out.append(f'<details><summary>Q{i}. {html.escape(q[:160])}</summary>'
                   f'<p class="q">{html.escape(q)}</p>{"".join(items)}</details>')
    return "".join(out)


def build(task: str, snapshot: Path, out: Path) -> Path:
    manifest = json.loads((snapshot / "MANIFEST.json").read_text())
    metric, ylabel = METRIC.get(task, METRIC["math"])
    tables = snapshot / "tables"
    frontier = read_csv(tables / "frontier.csv") if (tables / "frontier.csv").exists() else []
    seeds = read_csv(tables / "frontier_seeds.csv") if (tables / "frontier_seeds.csv").exists() else []
    if metric not in (frontier[0] if frontier else {}):   # helpful's column, when it exists, may be named differently
        metric = next((c for c in (frontier[0] if frontier else {}) if c.startswith(metric)), metric)
    compares = sorted(tables.glob("compare_*.txt"), key=lambda p: int(re.search(r"(\d+)", p.stem)[1]))
    seed_cols = SEED_TABLE_COLUMNS + [c for c in (metric, f"{metric}_seed_se", "pass8", "pass8_seed_se")
                                      if seeds and c in seeds[0]]
    rel = snapshot.relative_to(EVIDENCE).as_posix()
    git = manifest.get("git", {})
    figures = sorted(p.name for p in (snapshot / "figures").glob("*.png"))
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>llm-posttrain-lab: {html.escape(task)} snapshot {html.escape(snapshot.name)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{{font:15px/1.45 system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#222}}
table{{border-collapse:collapse;font-size:13px;margin:1em 0}} th,td{{border:1px solid #ccc;padding:2px 6px;text-align:right}}
th{{background:#f3f3f3}} td:first-child,td:nth-child(2){{text-align:left}}
pre{{background:#f7f7f7;padding:.6em;overflow-x:auto;font-size:12.5px;white-space:pre-wrap}}
.legend label{{margin-right:1.2em}} details{{margin:.4em 0}} summary{{cursor:pointer}}
.ex{{margin:.5em 0 .5em 1em}} .q{{font-style:italic}} .meta{{color:#555;font-size:13px}}
</style></head><body>
<h1>llm-posttrain-lab: {html.escape(task)} track</h1>
<p class="meta">Snapshot <code>{html.escape(rel)}</code>, taken {html.escape(str(manifest.get("date", "")))}
 at commit <code>{html.escape(str(git.get("commit", "")))}</code>; seeds {html.escape(", ".join(map(str, manifest.get("seeds", []))))};
 eval rows {manifest.get("eval", {}).get("measured", {}).get("rows", "")}. {html.escape(manifest.get("note") or "")}
 Generated by <code>analysis/build_viewer.py</code>; the claims these support are in <code>evidence/FINDINGS.md</code>.</p>

<h2>Frontier: {html.escape(ylabel)} vs KL from π_SFT</h2>
<p class="meta">One point per checkpoint and seed; lines join the seed means of each run's checkpoint positions. Hover a point for its numbers.</p>
{frontier_svg(frontier, metric, ylabel)}

<h2>Seed means per checkpoint position</h2>
<p class="meta"><code>tables/frontier_seeds.csv</code>: {len(seeds)} rows. <code>_seed_se</code> is the SE across seeds.</p>
<details><summary>Show table</summary>{table(seeds, seed_cols)}</details>

<h2>Pre-registered comparisons</h2>
<p class="meta">Each is one line of <code>evidence/{html.escape(task)}/snapshot.yaml</code>, fixed before the numbers were seen, as <code>analysis/compare_math.py</code> printed it.</p>
{"".join(f'<details><summary>{html.escape(p.name)}: {html.escape(p.read_text().splitlines()[0][:150])}</summary><pre>{html.escape(p.read_text())}</pre></details>' for p in compares)}

<h2>Example responses</h2>
<p class="meta"><code>examples/examples.jsonl</code>: sample 0 of each chosen checkpoint on the same questions, the disagreements first. ✓ and ✗ are the verifier's verdicts.</p>
{examples_html(snapshot)}

<h2>Figures in the snapshot</h2>
<p class="meta">Not embedded (the page stays small): {html.escape(", ".join(figures))} under <code>{html.escape(rel)}/figures/</code>.</p>

<script>
function toggleArm(box) {{
  document.querySelectorAll('g.arm[data-arm="' + box.dataset.arm + '"]').forEach(g => g.style.display = box.checked ? '' : 'none');
}}
</script>
</body></html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="+", help="math, helpful")
    ap.add_argument("--snapshot", help="evidence-relative snapshot dir (default: the task's latest)")
    ap.add_argument("--out", help="output file (default: evidence/viewer/<task>.html)")
    args = ap.parse_args(argv)
    if (args.snapshot or args.out) and len(args.tasks) != 1:
        ap.error("--snapshot and --out take exactly one task")
    for task in args.tasks:
        snapshot = EVIDENCE / args.snapshot if args.snapshot else latest_snapshot(task)
        if snapshot is None or not (snapshot / "MANIFEST.json").exists():
            print(f"{task}: no snapshot under evidence/{task}/, nothing to render", file=sys.stderr)
            continue
        out = build(task, snapshot, Path(args.out) if args.out else EVIDENCE / "viewer" / f"{task}.html")
        print(f"{task}: {out} from {snapshot.relative_to(EVIDENCE)} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
