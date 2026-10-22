# Evidence

This folder is the record of what the experiments found, kept in git: small figures, tables and numbers, plus the
documents built on them. The repo's README covers how to run things; this folder covers what came out. The write-up
or blog post draws on it.

| File | What it is | Edited by |
|---|---|---|
| [`FINDINGS.md`](FINDINGS.md) | The claims, one per ID (M1, M2, …), each with a status, a seed count, numbers with CIs and links to snapshot files | hand, after each snapshot |
| [`LOG.md`](LOG.md) | Lab notebook, newest first: runs, timings, incidents, decisions | hand, as things happen |
| `<task>/<date>_<name>/` | A **snapshot**: what one results tree held at one point | `make evidence` only |
| `<task>/snapshot.yaml` | The comparisons and example checkpoints a snapshot exports, fixed before the numbers are seen | hand, in a reviewed commit |

## Rules

1. **Snapshots are made by `make evidence`, never by hand, and never edited.** If one is wrong or out of date, take
   a new one and pass `--supersedes <old>`. The old one stays, marked as superseded in the index below. A changed
   conclusion is evidence too.
2. **Small things only.** That means figures, CSVs, configs, `MANIFEST.json` and a few example responses.
   Checkpoints, `*.samples.jsonl` and logs never go in; the export refuses anything over `--max-mb` (20 MB).
3. **Every claim cites a snapshot file**, not `$RESULTS_DIR`, which isn't in git and changes as runs continue. A
   number without a snapshot behind it is marked as pending.
4. **Comparisons are fixed in advance.** `snapshot.yaml` lists the matched-KL checkpoint pairs to compare. Change it
   in its own commit before the snapshot that uses it, so no checkpoint is chosen for how good it looks on the same
   test prompts.

## Taking a snapshot (on the pod, after a run's figures are rebuilt)

```bash
make status TASK=math                                   # every stage DONE, figures not stale
make evidence TASK=math NAME=seeds17-19_full-test EVIDENCE_ARGS="--dry-run"
make evidence TASK=math NAME=seeds17-19_full-test EVIDENCE_ARGS="--note '3 seeds, 1319 prompts'"
```

The export writes `evidence/math/<date>_seeds17-19_full-test/`:

| Path | Contents |
|---|---|
| `MANIFEST.json` | git commit, date, seeds, eval settings, package versions, host, each run's training time and steps, every file and its size |
| `configs/` | base, eval, task and stage yamls and `sweeps.yaml`, as run |
| `figures/` | every figure the track's plots wrote |
| `tables/` | `frontier.csv`, `frontier_seeds.csv`, DPO pair stats, stop probes, `training_curves.csv` (from `trainer_state.json`), and `compare_<i>.txt`, one per `snapshot.yaml` line |
| `examples/` | `examples.md`: sample 0 of each chosen checkpoint on the same questions |

The export refuses if the figures are older than the eval rows (run `make math-plots` first), or if the snapshot dir
already exists. It rebuilds the index below from every `MANIFEST.json`. Then update `FINDINGS.md` and `LOG.md`, and
commit the snapshot with them.

## Snapshots

<!-- snapshots:begin -->
_No snapshots yet: `make evidence TASK=math NAME=<name>`._
<!-- snapshots:end -->
