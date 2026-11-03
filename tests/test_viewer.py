"""analysis/build_viewer.py renders a snapshot into one self-contained HTML page (evidence/viewer/<task>.html):
every arm in the frontier, the seed-mean table, each pre-registered comparison, the examples; it picks the
snapshot no other supersedes by default, never writes into the snapshot, and embeds nothing external."""

from __future__ import annotations

import html
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = sorted(p.parent for p in (ROOT / "evidence" / "math").glob("*/MANIFEST.json"))


def _viewer():
    spec = importlib.util.spec_from_file_location("build_viewer", ROOT / "analysis" / "build_viewer.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_viewer"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def viewer():
    return _viewer()


@pytest.mark.skipif(not SNAPSHOTS, reason="no math snapshot in evidence/")
def test_renders_the_latest_math_snapshot_with_every_arm_comparison_and_example(viewer, tmp_path: Path) -> None:
    snapshot = viewer.latest_snapshot("math")
    assert snapshot is not None
    out = viewer.build("math", snapshot, tmp_path / "math.html")
    page = out.read_text()
    arms = {line.split(",")[1] for line in (snapshot / "tables" / "frontier.csv").read_text().splitlines()[1:]}
    for arm in arms:
        assert f'<g class="arm" data-arm="{arm}">' in page, arm
    for compare in (snapshot / "tables").glob("compare_*.txt"):
        assert compare.name in page and html.escape(compare.read_text().splitlines()[0][:60]) in page
    examples = [json.loads(line) for line in (snapshot / "examples" / "examples.jsonl").read_text().splitlines() if line]
    for ex in examples:
        assert ex["checkpoint"] in page
    assert len(re.findall(r"<details><summary>Q\d+\.", page)) == len({ex["question"] for ex in examples})
    assert snapshot.name in page and "<svg" in page and "<title>" in page   # hover tooltips on the points


@pytest.mark.skipif(not SNAPSHOTS, reason="no math snapshot in evidence/")
def test_the_page_is_self_contained_and_the_snapshot_is_untouched(viewer, tmp_path: Path) -> None:
    snapshot = viewer.latest_snapshot("math")
    before = sorted((p.relative_to(snapshot).as_posix(), p.stat().st_mtime_ns) for p in snapshot.rglob("*") if p.is_file())
    page = viewer.build("math", snapshot, tmp_path / "math.html").read_text()
    after = sorted((p.relative_to(snapshot).as_posix(), p.stat().st_mtime_ns) for p in snapshot.rglob("*") if p.is_file())
    assert before == after
    assert not re.search(r'(src|href)="https?://', page), "the page must not load anything from the network"
    assert "<img" not in page   # figures are referenced by path, not embedded or linked


def test_the_default_snapshot_is_the_one_nothing_supersedes(viewer, tmp_path: Path, monkeypatch) -> None:
    ev = tmp_path / "evidence"
    for name, supersedes in (("2026-01-01_a", []), ("2026-01-02_b", ["math/2026-01-01_a"]), ("2026-01-03_c", [])):
        d = ev / "math" / name
        d.mkdir(parents=True)
        (d / "MANIFEST.json").write_text(json.dumps({"supersedes": supersedes}))
    monkeypatch.setattr(viewer, "EVIDENCE", ev)
    assert viewer.latest_snapshot("math").name == "2026-01-03_c"
    (ev / "math" / "2026-01-04_d").mkdir()
    (ev / "math" / "2026-01-04_d" / "MANIFEST.json").write_text(json.dumps({"supersedes": ["math/2026-01-03_c"]}))
    assert viewer.latest_snapshot("math").name == "2026-01-04_d"
    assert viewer.latest_snapshot("helpful") is None


def test_a_task_without_a_snapshot_is_skipped_not_an_error(viewer, tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(viewer, "EVIDENCE", tmp_path)
    assert viewer.main(["helpful"]) == 0
    assert "no snapshot" in capsys.readouterr().err


def test_frontier_lines_follow_training_order_not_kl_order(viewer, tmp_path: Path) -> None:
    """A run whose KL falls between two checkpoints (DPO+NLL beta 0.1 in the math snapshot) is drawn in
    checkpoint order, as frontier_plot.py draws it, not sorted by KL (Codex on PR #38)."""
    rows = [{"arm": "dpo-nll", "hyperparam": "beta0.1", "seed": "17", "ckpt_index": str(i), "step": str(29 * i),
             "kl_k1": str(kl), "pass1": str(p)} for i, (kl, p) in enumerate([(1.0, 0.45), (9.5, 0.52), (8.9, 0.54),
                                                                              (10.2, 0.53)], start=1)]
    svg = viewer.frontier_svg(rows, "pass1", "pass@1")
    points = re.search(r'<polyline points="([^"]+)"', svg)[1].split()
    xs = [float(pt.split(",")[0]) for pt in points]
    order = [sorted(xs).index(x) for x in xs]
    assert order == [0, 2, 1, 3], "the line must go 1.0 -> 9.5 -> 8.9 -> 10.2 nats, i.e. back and forth"
