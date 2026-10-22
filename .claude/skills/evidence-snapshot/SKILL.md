---
name: evidence-snapshot
description: Freeze a finished run's figures and tables into evidence/ with make evidence, then log it. Use after a math or helpful run (or a new seed) finishes on the pod, or when the user asks to snapshot, record or save results as evidence.
---

# Take an evidence snapshot

A snapshot freezes what one results tree holds into `evidence/<task>/<date>_<name>/`. The rules are in
`evidence/README.md` and `AGENTS.md`. This skill is the procedure. The export runs on the pod, where `$RESULTS_DIR`
is. From a laptop session, give the user the commands and ask them to paste the output.

## 1. Check the run is finished

```bash
make status TASK=math
```

Every stage the snapshot is about must be DONE.
- The figures stage must not be stale. If it is, `make math-plots` first.
- If a seed is only partly evaluated, either wait or say so in `--note`: a partial seed drags the seed means.
- Note which seeds are complete.

## 2. Check the comparisons are fixed

Read `evidence/<task>/snapshot.yaml`. If the user wants different comparisons or example checkpoints, change it in
its own commit before exporting, and say why in `evidence/LOG.md`. Never change it after seeing the numbers it
would produce.

## 3. Dry run, then export

Pick a NAME saying what's in it: `seeds17-19_full-test`, `seed17_full-test`, `dpo-lr2e-6`.

```bash
make evidence TASK=math NAME=<name> EVIDENCE_ARGS="--dry-run"
make evidence TASK=math NAME=<name> EVIDENCE_ARGS="--note '<one line>'"
```

If this snapshot replaces an older one (same question, better data), add `--supersedes math/<old snapshot>`.
Never delete or edit the old one.

The export refuses when:
- the figures are stale;
- the snapshot dir exists;
- it would be over `--max-mb`.

Fix the cause; don't work around it. Every `tables/compare_<i>.txt` should say `ok` in `MANIFEST.json` →
`compare[].status`. A failed line usually means a seed or checkpoint isn't there yet: report it, don't drop the line.

## 4. Look at it

- Read `MANIFEST.json`: seeds, `eval.num_prompts`, git commit, training times.
- Read `tables/compare_*.txt` and `examples/examples.md`.
- Glance at the figures.
- Check the snapshot is small (`total_bytes`) and holds no checkpoints or samples files.

## 5. Log it, then update the findings

- Add a dated entry at the top of `evidence/LOG.md`:
  - what ran (seeds, prompts);
  - wall-clock per stage, from `MANIFEST.json` → `runs[].train_runtime_s` and from `make status`;
  - anything that went wrong, and the snapshot path.
- Then follow the `update-findings` skill, so `evidence/FINDINGS.md` cites this snapshot.
- Commit the snapshot dir, `evidence/README.md` (whose index the export rebuilt), `LOG.md` and `FINDINGS.md`
  together. On the pod's checkout or a PR, whichever the user uses.
