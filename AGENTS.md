# Agent guide: llm-posttrain-lab

This is a post-training study on Qwen2.5-1.5B. It compares SFT, DPO, DPO+NLL and GRPO (plus PPO on the helpful
track) by accuracy vs KL from π_SFT. There are two tracks:
- **math:** GSM8K with a verifier reward;
- **helpful:** UltraFeedback with a learned reward model.

Real runs happen on a 4×A100 pod. Laptops only run tests and smoke-sized checks. The README covers how to run
things. `evidence/` holds what the runs found.

## Layout

- `configs/`: OmegaConf yamls, merged in this order:
  `base.yaml` → `tasks/<task>.yaml` → stage yaml → `smoke.yaml` (when `--smoke`) → CLI dotlist (`train.beta=0.1`).
  The sweep grids are in `configs/<task>/sweeps.yaml`.
- `scripts/`: stage launchers. `run_all_math.sh` runs `m1`–`m5`. Math stages take `SEED` (`scripts/math/seed.sh`,
  default 17).
- `src/posttrain_lab/`: training, eval (`eval/run_eval.py`, incremental metric groups), `status`, `reset`, config.
- `analysis/`: aggregate → `frontier.csv` / `frontier_seeds.csv`, the figure scripts, `compare_math.py` and
  `export_evidence.py`.
- `evidence/`: git-tracked results. Read `evidence/README.md` before touching it.
- Results live in `$RESULTS_DIR/<task>` (from `.env`; `results/` by default), which is not in git.

## Tests

Run them in a venv with exactly these packages:

```
trl==1.7.0 transformers==5.13.0 datasets==5.0.0 accelerate==1.14.0 omegaconf==2.3.1 lm-eval==0.4.12 torch pandas matplotlib
```

Install the repo editable, then run `python -m pytest tests/` (about 4–5 min, 170+ tests). Tests use the `repo`
fixture (`tests/conftest.py`), which works on a throwaway copy of the repo. They stub `accelerate`, vLLM and the GPU
entry points: GPU behaviour, such as vLLM memory, is only exercised for real on the pod. Every fix gets a test that
fails without it.

## Code conventions

- **Shell scripts must run under macOS `/bin/bash` 3.2:** no `mapfile`, no associative arrays.
- **Wait on each background job individually,** so a launcher exits nonzero if any job failed. A bare `wait` always
  returns 0.
- **Seeds:**
  - run dirs, pairs and π_SFT paths all end in `_seed<N>`;
  - eval scores each seed's runs against its own π_SFT, and `run_eval` refuses a checkpoint of another seed;
  - seed 17's log names have no suffix; other seeds' logs end in `_seed<N>`.
- **Don't change a config, sweep or checkpoint cadence for one seed only.** Seeds must stay comparable.
- **Keep the existing names:** don't rename run dirs, log names or metric keys. `status`, `reset`, `aggregate` and
  the figures parse them.
- **Before a track's first full run, probe it at full scale:** `make helpful-probe` (`PROBE=1`,
  `configs/probe.yaml`) runs the real launchers for a few steps per stage. The smoke test runs 0.5B on one GPU and
  can't catch an OOM or a multi-GPU failure.
- `make status TASK=math` is read-only and safe any time. `make reset` deletes results: dry-run it first
  (`RESET_ARGS="--dry-run"`) and never run it without the user asking.

## Analysis conventions

- **KL:** use `kl_k1` (nats per sequence). `kl_k3` explodes far from π_SFT.
- **Ties:** pass@1 differences of 0.02 or less count as ties unless a paired CI says otherwise.
- **Compare at matched KL,** within about 0.3 nats, using `analysis/compare_math.py`.
  - Name checkpoints by KL target, `@klX`: each seed's checkpoint nearest X nats by `kl_k1`. DPO's step counts and KL
    per step differ between seeds (each seed trains on its own pairs), so the same `@ckptK` can sit at very different
    KL. A checkpoint flagged `!` is more than 0.3 nats off: not matched.
  - Only `--pool`'s across-seed line covers training randomness; per-seed CIs cover eval sampling only.
- **Picking each arm's best checkpoint on the test prompts flatters it.** Prefer the comparisons fixed in advance in
  `evidence/<task>/snapshot.yaml`, and say so when a comparison isn't one of them.

## Evidence rules (`evidence/`)

- **Snapshots** (`evidence/<task>/<date>_<name>/`) come only from `make evidence`.
  - Never edit one. To replace one, take a new snapshot with `--supersedes`.
  - Never put checkpoints, `*.samples.jsonl` or logs in git.
- **Every claim in `evidence/FINDINGS.md`:**
  - has an ID (M1…);
  - has a status (`tentative` / `supported` / `refuted` / `superseded`) and a seed count;
  - gives numbers with CIs, linked to files inside a snapshot (not `$RESULTS_DIR`);
  - keeps a History line when it changes.
- **Add an entry to `evidence/LOG.md`** (newest first) for each run, incident or decision, with the numbers that
  mattered and the PR.
- **Change `snapshot.yaml`** (comparisons, example checkpoints) in its own commit, before the snapshot that uses it.

## PRs

- Branch from `main`.
- The description has three parts: why (what was observed), changes, and tests (with the count).
- Merge with a regular merge commit, not a squash. Stacked PRs rely on it.
- Codex reviews PRs automatically: answer each finding with a fix and a test, or with the reason not to.
