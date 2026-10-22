# llm-posttrain-lab

Experiments demonstrating **why preference alignment in LLM post-training uses
reinforcement learning (PPO, GRPO) rather than plain SFT or DPO.**

## Hypothesis

SFT on preferred answers is imitation learning: next-token-prediction loss pulls
the whole output distribution toward the demonstrations, with nothing anchoring
it to what the model already knows. RL methods instead *optimize a reward under
an explicit KL-divergence penalty to a reference policy* — they induce a
preference over answers while provably staying close to the reference
distribution. DPO sits in between: its loss is derived from the same
KL-regularized objective, but it optimizes a proxy on a fixed offline dataset,
and in practice drifts in ways the derivation doesn't promise (famously, the
log-probability of the *chosen* responses often falls during DPO training).

## Two task tracks, one question

Both tracks share the model (`Qwen/Qwen2.5-1.5B`, full fine-tuning), the
training stack (TRL 1.7 + accelerate/DeepSpeed), the pipeline shape, and the
evaluation machinery — they differ only in what "reward" means:

| | **helpful** | **math** |
|---|---|---|
| Dataset | [`HuggingFaceH4/ultrafeedback_binarized`](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) |
| Reward | learned preference RM (+ held-out Skywork judge for eval) | **verifiable**: exact answer match (`src/posttrain_lab/tasks/gsm8k.py`) |
| Arms | SFT / DPO / PPO / GRPO | SFT / DPO (self-generated pairs) / GRPO |
| Headline metric | judge reward vs KL | pass@1 vs KL (+ pass@8) |
| Preference pairs | human-annotated | **self-generated**: sample 8 solutions/prompt from π_SFT-math, correct=chosen vs incorrect=rejected |

PPO is deliberately absent from the math track: TRL's PPOTrainer requires an
`nn.Module` reward model, while a verifier is a function; GRPO natively accepts
callable rewards and is the community-standard RLVR setup. The PPO-vs-GRPO
evidence lives on the helpful track. (Escape hatch: train an RM on the
self-generated pairs and reuse the PPO wrapper unchanged.)

### Pipelines

```
helpful:  base → SFT(train_sft) → π_SFT → RM (acc-gated) → arms:
            SFT(chosen) | DPO β∈{.01,.05,.1,.3} | PPO c∈{.02,.05,.1} | GRPO β∈{0,.01,.04,.1}
math:     base → SFT(gold CoT) → π_SFT-math → self-gen pairs → arms:
            SFT(gold) | DPO β∈{.01,.05,.1,.3} | DPO+NLL β∈{.01,.05,.1,.3} | GRPO(verifier) β∈{0,.01,.04,.1}
```

Each task's π_SFT is both the shared starting point of its arms and the KL
reference for its eval; the β=0 GRPO runs trace what *unconstrained* reward
optimization does. Math's DPO+NLL arm is the DPO arm plus the NLL of the
chosen answer (IRPO / RPO; TRL `loss_type: [sigmoid, sft]`), at the same
budget and β grid, so the two show what that term changes.

## Evaluation — every checkpoint becomes one frontier point

Per checkpoint: sample N responses per held-out prompt at temperature 1.0
(unbiased MC), then compute **metric groups** (per-task list in
`configs/tasks/*.yaml`):

| Group | Evidence | Key outputs |
|---|---|---|
| `core` / `core_math` | the frontier | k3/k1 sequence-KL vs π_SFT (figures plot k1: k3 explodes far from π_SFT); dual reward scores (helpful) or pass@1/pass@8 ± SE over prompts (math; all 1319 GSM8K test prompts) |
| `retention_ppl` | capability retention | wikitext-2 perplexity |
| `retention_harness` | capability retention | lm-eval HellaSwag/ARC accuracy (all checkpoints by default — `eval.harness.which: final` to economize) |
| `diversity` | mode collapse | distinct-1/2/3, self-BLEU across the N samples (reused, not regenerated) |
| `pref_logps` | DPO failure modes | held-out logp(chosen), logp(rejected), margin — for EVERY arm |

**Metric groups are incremental**: each checkpoint has one summary row; a group
runs only if its keys are missing (atomic read-merge-replace). Adding a group
to the config later and re-running `make eval` computes only the new metrics —
nothing is regenerated. (Corollary: don't tail the summary files; they are
rewritten, not appended.) Each row also records how many prompts it was
sampled on, so changing `eval.num_prompts` re-samples just the prompt-dependent
groups (`core`/`core_math`, `diversity`) on the next `make eval` /
`make math-eval`. Retention and `pref_logps` are kept.

**Comparing two math checkpoints.** All checkpoints are scored on the same
prompts, so judge a difference paired, per prompt. That is much tighter than
each point's own ± SE. Pick checkpoints by matched KL or a fixed step: picking
each run's best on these same prompts flatters it.

```bash
python analysis/compare_math.py "$RESULTS_DIR/math" dpo_beta0.3@116 grpo_beta0.01@374 grpo_beta0.0@374
```

These CIs cover eval sampling only, for these particular trained models. How
much a rerun with another training seed moves the numbers needs more seeds
(see [More seeds](#more-seeds-math-track)).

## Quickstart

New node? Walk through [`notebooks/onboarding.ipynb`](notebooks/onboarding.ipynb)
first. It sets up `.env` (MLflow endpoint) and checks packages, `/dev/shm`, GPUs + NCCL,
flash-attn/DeepSpeed/vLLM, Hugging Face assets (offline cluster) and MLflow, one
section each. The same checks run headless with `make preflight` or
`python -m posttrain_lab.preflight gpu mlflow ...`.

Tracking is MLflow (`report_to: mlflow`). Set `MLFLOW_TRACKING_URI` in `.env` or the pod
env, plus `MLFLOW_PORT` only if the server's non-standard port isn't already in the URI. Runs land in experiment
`llm-posttrain-lab-<task>`, tagged with `arm`, `hyperparam`, `seed` and `run_dir`; eval
metrics are attached to each training run as `eval/*` at the checkpoint's step.

Storage: results and checkpoints need ~310 GB for both tracks and the HF cache
~22 GB, often more than the pod's home volume. Put both on a bigger volume in
`.env` (or the pod env, which wins), using absolute paths:

```bash
RESULTS_DIR=/mnt/data/results   # default: results/ in the repo; a relative value is taken from the repo root
HF_HOME=/mnt/data/hf            # default: ~/.cache/huggingface
```

Every stage, `make plots`/`make math-plots`, the `run_all_*`/`run_smoke` launchers
and the analysis scripts' defaults follow `RESULTS_DIR` (runs land in
`$RESULTS_DIR/<task>`, smoke in `$RESULTS_DIR/smoke/<task>`). `make preflight`'s
`system` check prints both resolved paths and their free space.

```bash
pip install -r requirements.txt && pip install -e .

make preflight    # node readiness (GPUs, NCCL, HF assets, MLflow)
make smoke        # REQUIRED first: both tracks end-to-end on Qwen2.5-0.5B,
                  # tiny slices, 1 GPU, <30 min. Gates all API/version drift,
                  # both reward paths, and every metric group.

make helpful-all  # ~2.5-3 days on 4x A100 80GB (+~15-20h for harness-everywhere)
make math-all     # ~12-15 h on 4x A100 80GB (one seed; SEED=18 for another: "More seeds")
```

Everything is idempotent (DONE markers, incremental eval): re-run any script
after a crash and it continues. Per-stage targets:

| helpful | math | |
|---|---|---|
| `make sft-base` | `make math-sft-base` | π_SFT (2–3 h / ~30 min) |
| `make rm` | `make math-pairs` | RM (2–3 h) / self-gen pairs (~1 h) |
| `make arm-sft arm-dpo` | `make math-arm-sft math-arm-dpo math-arm-dpo-nll` | imitation + DPO (+NLL) sweeps |
| `make arm-ppo arm-grpo` | `make math-arm-grpo` | RL sweeps (the long pole) |
| `make eval` | `make math-eval` | sharded, incremental |
| `make plots` | `make math-plots` | figures |

Overrides are OmegaConf dotlist args
(`python -m posttrain_lab.train.dpo configs/math/m3_arm_dpo.yaml train.beta=0.2`);
grids live in `configs/{helpful,math}/sweeps.yaml`. Disk: ~250 GB helpful,
~60 GB math, under `RESULTS_DIR` (see Storage above).

## Monitoring a run

Launch a track detached, with the GPU logger next to it, then check on it any time:

```bash
nohup make math-all > math-all.log 2>&1 &   # helpful: > helpful-all.log
make gpu-log                                 # GPU utilization every 30 s -> gpu_util.csv
make status                                  # or: make status TASK=math STATUS_ARGS="--watch 60"
```

`make status` (`python -m posttrain_lab.status`) reads the stage configs, the
run dirs, the logs and the process table, and prints for each track:

- every stage in `run_all_<task>.sh` order, and every sweep run, as
  `DONE` / `RUNNING` / `FAILED` / `STOPPED` (output but no DONE marker and
  nothing running) / `PENDING`, with checkpoints written vs expected and the
  latest step and training metrics
- for running jobs: the live progress bar from the job's log (step, elapsed,
  remaining, expected finish), the GPU it is on, its latest logged metrics
- stage checks: m2 pair yield (prompts sampled vs the split, train/test
  pairs), the RM accuracy gate, GRPO zero-variance groups and clipped
  completions, DPO preference accuracy, eval coverage per metric group and
  what each eval shard is on; figures count as done only when every plot the
  launcher makes exists and is newer than `frontier.csv` (itself newer than
  every eval row)
- GPUs now, plus the `gpu_util.csv` history (mean utilization, idle share)
- errors found in the logs, a resume command when a track has stopped, and
  an **attention** list of everything that looks wrong

It is read-only and quick (no torch import). `--json` prints the same report
as JSON, `--smoke` shows the smoke tree against exactly what `run_smoke.sh`
runs, `--log PATH` points at a launcher log with a different name (each track
reads only its own `=== ` headers from it). Re-running `make <task>-all` after a stop skips
finished runs; the report's `resume` line gives the exact command. With more
than one training seed, every stage lists one run per sweep value and seed,
and eval coverage is counted per seed; `--seed N` (repeatable) shows only those seeds.

### Starting over

To throw away a track's run and start it clean (a fix landed, a config
changed), use `make reset` rather than deleting by hand:

```bash
make reset TASK=math RESET_ARGS="--dry-run"    # look first: what would stop / be removed
make reset TASK=math RESET_ARGS="--relaunch"   # do it, then start math-all again, detached
make reset-smoke                               # the smoke tree (before re-running make smoke)
```

`python -m posttrain_lab.reset {math|helpful|all}` stops this repo's jobs for
that track (driver, make target, stage scripts, every training/pairs/eval job
writing into it and their child processes; the other track and other
checkouts are left alone), deletes `$RESULTS_DIR/<task>` (`--archive` renames
it to `<task>.<timestamp>` instead), removes the track's logs and any
half-written `tmp*` files in the HF datasets cache, and with `--relaunch`
restarts the GPU logger and `make <task>-all` in the background. It prints
the plan and asks you to type the track name first (`--yes` skips that). The
HF cache, `.env`, other tracks' and smoke results, and MLflow runs are never
touched.

To redo a single arm after changing its config (e.g. the DPO learning rate),
keep everything upstream and reset just that stage:

```bash
make reset TASK=math RESET_ARGS="--stage m3b --archive --relaunch"
```

`--stage` takes an arm (math `m3a`–`m3d`, helpful `3a`–`3d`). It removes
only that arm's run dirs, their eval results and their per-run logs. That
includes old runs of the arm the config no longer names, such as a previous
sweep grid or seed. **By default it clears the arm for every seed**: once an
arm's config changes, no seed's old runs are comparable any more. `--relaunch`
then retrains it for each removed seed that still has its π_SFT, with
`make math-seeds SEEDS="17 18"`. Add `--seed N` (math only) to clear and
relaunch just seed N's runs of the arm (`make math-all SEED=N`); the other
seeds' runs, eval results and logs stay. It stops only that arm's own jobs, and refuses while
anything else in the track is running (the driver, another arm, eval), so
nothing else gets cut short.
`--archive` moves them to `<root>/archive/<timestamp>/`, where eval and the
figures don't see them, which is handy to compare old and new. The relaunched
`make <task>-all` skips every stage that is still DONE, trains the arm,
evaluates just its checkpoints and rebuilds the figures. You type the stage
key to confirm.

## More seeds (math track)

Seed 17 is one training run per arm and β. The pass@1 SEs cover eval
sampling, not how much a rerun with another seed would move a curve. The math
stages take a `SEED` (default 17, from `configs/base.yaml`). It reaches every
stage as `seed=N`, and every seed-dependent path carries it: π_SFT
(`runs/sft-base_seedN`), the self-generated pairs (`data/self_pairs_seedN`)
and every arm's run dir. So a second seed retrains the whole chain from π_SFT
next to seed 17's runs. It never touches or reuses them, and it is never
skipped as DONE because of them.

```bash
nohup make math-all SEED=18 > math-all-seed18.log 2>&1 &     # one more seed
nohup make math-seeds SEEDS="18 19" > math-all.log 2>&1 &     # or several, one after the other
make status TASK=math                                         # every seed; STATUS_ARGS="--seed 18" for one
```

**Cost per extra seed** on 4× A100 80GB: about 12–15 h and about 60 GB of disk.
- **Training, about 6 h.** π_SFT about 30 min, pairs about 1 h, SFT arm, the
  DPO and DPO+NLL sweeps (parallel, 1 GPU each), then the 4 GRPO runs one
  after another at about 1 h each.
- **Eval, about 6–9 h.** That covers the seed's 112 checkpoints (π_SFT 3,
  SFT 9, DPO 40, DPO+NLL 40, GRPO 20) on all 1,319 test prompts.
  `eval.harness.which: final` in `configs/eval.yaml` runs the lm-eval harness
  on each run's last checkpoint only, which cuts a good part of that.

Eval (`m4`, `make math-eval SEED=N`) scores only seed N's runs. Each is
measured against its own seed's π_SFT (KL) and held-out pairs (`pref_logps`).
`run_eval` refuses a checkpoint of another seed, so the reference can't be
mixed up. Per-run logs of a seed other than 17 end in `_seedN`
(`logs_math_dpo_train_beta_0_1_seed18.log`). Seed 17's names are unchanged.

**Analysis across seeds.**
- **`frontier.csv`** keeps one row per checkpoint. Run names include the seed,
  and each row has `seed` and `ckpt_index` columns (1 = the run's first
  checkpoint).
- **`frontier_seeds.csv`** has one row per (arm, hyperparam, `ckpt_index`).
  Columns: `n_seeds`; each metric's mean over seeds; `<metric>_seed_se`, the SE
  of that mean across seeds; and for pass@k, `<metric>_prompt_se`, the part of
  the error eval sampling alone would explain. Seeds are matched by checkpoint
  index, not step: each seed's DPO arms train on that seed's own pairs, so
  their step counts differ.
- **`figures/frontier_math.png`.** With one seed it is unchanged. With more
  than one, it draws one curve per (arm, β) through the seed means, with
  across-seed error bars on both pass@k and KL.
- **The other figures** (`diversity.png`, `retention.png`, `pref_logps.png`,
  `diagnostics.png`) work the same way. With one seed they are unchanged.
  With more than one, each point is the seed mean of an (arm, β) at one
  checkpoint index, with across-seed error bars. The KL scatters get bars on
  both axes. `pref_logps.png` draws one chosen/rejected curve pair per
  (arm, β) at the mean of the seeds' steps, with bars on the log-probs.
- **`figures/dpo_logps.png`** reads each run's training log, and it keeps one
  line per seed. Each seed's DPO runs log at their own steps and train for a
  different number of them, so a mean would have to interpolate. Each β gets
  one color and one legend entry, so the spread between a β's lines is the
  seed-to-seed variation.
- **`compare_math.py`.** Pick one seed with `--seed 18` (or name the run in
  full, `dpo_beta0.3_seed18@last`). Or pool them with `--pool`: that prints the
  paired difference per seed, then their mean with a 95% CI from the spread
  across seeds (t with n_seeds − 1 df). Only that last CI covers training
  randomness. With 2 seeds, t₁ = 12.7 makes it very wide; 3 seeds (t₂ = 4.3)
  is the least that says much. Name DPO checkpoints by position (`@ckpt10`) or
  `@last`, since their steps differ between seeds.

```bash
python analysis/compare_math.py "$RESULTS_DIR/math" --pool dpo_beta0.3@ckpt10 grpo_beta0.01@last
```

## Offline cluster (Hugging Face assets)

The pipeline loads these from the Hub (`python scripts/prefetch_assets.py --list`):
the models `Qwen/Qwen2.5-1.5B` (base), `Qwen/Qwen2.5-0.5B` (smoke) and
`Skywork/Skywork-Reward-V2-Llama-3.1-8B` (judge, helpful eval), and the datasets
`HuggingFaceH4/ultrafeedback_binarized`, `openai/gsm8k` [main],
`Salesforce/wikitext` [wikitext-2-raw-v1] (retention PPL) plus lm-eval's
`Rowan/hellaswag` and `allenai/ai2_arc` [ARC-Easy, ARC-Challenge]. Without
internet there are two ways to provide them. Set the variables in the pod spec
or in `.env` (template in `.env.example`; `HF_*` lines there are exported on
`import posttrain_lab`, and the pod env wins).

**A. Internal mirror / proxy.** Point every HF client at it; nothing else changes:

```bash
HF_ENDPOINT=https://hf-mirror.internal.example   # + HF_TOKEN if it needs one
HF_HUB_DISABLE_XET=1                             # Artifactory / Nexus HF remotes (see below)
HF_HOME=/mnt/hf                                  # optional: cache on a PVC, survives pod restarts
```

Hub repos store large files in Xet, and huggingface_hub downloads those
through a Hub-only token API (`/api/models/<repo>/xet-read-token/...`) that
Artifactory and Nexus Hugging Face remotes don't implement. They answer 404,
and the download dies. `HF_HUB_DISABLE_XET=1` makes the client fetch the same
files over plain HTTP through the mirror. `prefetch_assets.py` prints this hint
when it hits that 404.

**B. Pre-populated cache on a PVC.** On a machine *with* internet (or with
`HF_ENDPOINT` set to a mirror), download everything into one directory; the
script then re-runs itself offline to prove the copy is self-sufficient:

```bash
python scripts/prefetch_assets.py /data/hf_home              # all, ~22 GB (judge ~16 GB)
python scripts/prefetch_assets.py /data/hf_home --smoke      # only what `make smoke` needs, ~2.5 GB
python scripts/prefetch_assets.py /data/hf_home --skip judge # e.g. math track only
make prefetch PREFETCH_DIR=/data/hf_home PREFETCH_ARGS=--smoke
```

Copy the directory to the PVC with symlinks intact (`rsync -a`, or
`tar -C /data/hf_home -cf - . | kubectl exec -i <pod> -- tar -C /mnt/hf -xf -`),
then on the cluster:

```bash
HF_HOME=/mnt/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
```

Both `hub/` (model snapshots, raw dataset files) and `datasets/` (prepared
Arrow tables) must come along: offline, `datasets` resolves a Hub id only
through the Arrow cache, so leave `HF_DATASETS_CACHE`/`HF_HUB_CACHE` unset (or
pointing inside the copy). Models are loaded by repo id from the cache, so no
config changes are needed; `model.base`, `eval.judge` and `dataset.id` also
accept local paths. The retention PPL set can instead come from a local
parquet file: `eval.retention.data_files=/mnt/data/wikitext-2-raw-v1-test.parquet`.

Check either option with `make preflight` (or `python -m posttrain_lab.preflight hf`:
cached / downloadable / missing per asset), and `python -m posttrain_lab.preflight
hf_load` to actually load each one (tokenizer + config per model, every dataset
split, lm-eval's task build) with the current env.

## Reading the figures (`results/<task>/figures/`)

- **`frontier.png`** (helpful): judge reward (left) vs training-RM reward
  (right) vs KL. Up-and-left wins; a right panel soaring while the left sags
  is reward hacking, quantified in `diagnostics.png`'s RM-gap panel.
- **`frontier_math.png`**: pass@1 vs KL and pass@8 vs KL. Watch whether a
  method buys pass@1 by collapsing pass@8 (diversity spent) or keeps both
  (diversity preserved).
- **`retention.png`**: wikitext PPL + benchmark accuracy vs KL — the alignment
  tax. Flat = preference gained without capability lost.
- **`diversity.png`**: distinct-2 / self-BLEU / token-NLL vs KL (+ pass-gap
  panel on math) — the mode-collapse signature.
- **`pref_logps.png`**: held-out chosen (solid) / rejected (dashed) logp per
  arm — is the "squeezing effect" (both fall, margin grows) DPO-specific?
- **`dpo_logps.png`**: the same phenomenon in DPO's own training logs.

## Repo map

```
configs/
  base.yaml               task-agnostic defaults (merge order documented inside)
  tasks/{helpful,math}.yaml   per-task datasets, graders, metric groups
  helpful/  math/         per-task stage configs + sweeps.yaml
  eval.yaml smoke.yaml    global eval knobs; global tiny-mode override
src/posttrain_lab/
  data.py                 loaders dispatching on dataset.format (ultrafeedback | gsm8k | pairs_disk)
  tasks/gsm8k.py          THE verifier (extraction, correctness, TRL reward_funcs)
  gen_pairs.py            math m2: self-generated preference pairs + stats.json
  status.py               make status: per-stage progress, GPUs, errors, attention list
  reset.py                make reset: stop a track's jobs, clear its results/logs, relaunch
  train/                  thin TRL wrappers (the swappable layer for future verl/OpenRLHF)
  eval/                   run_eval group engine; kl, reward_score, passk, retention, diversity, pref_logps
analysis/                 aggregate → frontier.csv + frontier_seeds.csv (seed means); frontier, frontier_math, retention, diversity, pref_logps, diagnostics
scripts/{helpful,math}/   per-stage launchers; run_all_{helpful,math}.sh; run_smoke.sh
scripts/prefetch_assets.py  fill an HF_HOME for the offline cluster
tests/                    python -m pytest tests/ (CPU; the download test skips without internet)
```

## Known risks / gotchas

- **End of turn vs eos** (base models): Qwen2.5 base declares eos =
  `<|endoftext|>`, but the chat template ends assistant turns with
  `<|im_end|>`. Unhandled, nothing stops at `<|im_end|>`: every sample runs to
  the token cap with invented follow-up turns, which corrupts KL, diversity,
  the self-generated pairs and GRPO rollouts (the first full math run: 100% of
  eval samples at 512 tokens). `load_tokenizer` therefore makes the
  end-of-turn token the eos of every saved checkpoint, every sampler passes it
  as a stop id, eval rows record `finished`, and `run_all_*.sh` / `make smoke`
  stop right after π_SFT if `python -m posttrain_lab.utils.chat RUN_DIR`
  finds a checkpoint that would not stop. Runs from before this fix need
  retraining; `make status` flags them.
- **Untrained chat-token embeddings** (base models): Qwen2.5 base never
  trained `<|im_start|>` / `<|im_end|>`. Their (tied) embedding rows are
  near-copies of ~2.8k other untrained rows (rare Thai and other tokens,
  cosine ~0.99), so SFT pushing toward `<|im_end|>` lifts the whole crowd.
  The second full math run's π_SFT gave `<|im_end|>` 0.12 where an answer ends,
  and sampled a lookalike token (`#### 135รุ่ง`) the rest of the time. Those
  samples then invent new questions until the cap (39% of pair samples, 41% of
  rejected answers). SFT now re-initializes such rows first
  (`train/common.py:init_untrained_chat_tokens`: `<|im_end|>` ← `<|endoftext|>`,
  the rest ← mean row; a no-op on trained rows). `run_all_*.sh` then stop
  after π_SFT unless `python -m posttrain_lab.utils.chat RUN_DIR --probe
  STAGE_YAML` measures P(`<|im_end|>`) ≥ 0.8 where the training answers end
  (written to `RUN_DIR/stop_probe.json`; `make status` shows it).
- **DPO budget and likelihood displacement** (math): DPO on the self-generated
  pairs either barely moves or collapses.
  - At lr 5e-7 × 1 epoch: KL 0.16, loss stuck near ln 2.
  - At 5e-6 × 2 epochs: pass@1 peaked at 0.57 by step 58, then fell to
    0.07–0.37 while training accuracy reached 0.99. The held-out
    log-probability of the *correct* answers fell along with the wrong ones'.
  - The answer format stays intact (`#### N`); the reasoning comes apart.
  - The DPO arm now runs at 2e-6 × 2 epochs with 10 checkpoints per run, and
    the DPO+NLL arm adds the term meant to prevent this.
  - `kl_k3` is unusable that far out (up to 1e23), so the figures plot `kl_k1`
    on a symlog axis.
- **TRL API drift**: PPOTrainer lives in `trl.experimental.ppo` (defensive
  import); pins in `requirements.txt` are load-bearing — bump trl+vllm as a
  pair and re-run `make smoke`.
- **PPO memory** (helpful): 4 models resident; lower
  `train.local_rollout_forward_batch_size` first on OOM.
- **vLLM colocate** (GRPO): `vllm_gpu_memory_utilization: 0.25` is the start point.
- **Self-gen pair yield** (math): prompts where all samples are correct or all
  wrong yield no pairs — check `stats.json` after `make math-pairs`; knobs are
  k, temperature, `pairs.max_per_prompt`. The pairs carry a selection bias
  (only contested prompts) — remember it when reading DPO-math results.
- **GRPO zero-variance groups** (math): all-correct/all-wrong rollout groups
  contribute zero advantage — expected RLVR behavior, visible as wasted rollouts
  at very high/low accuracy.
- **Reward hacking / length bias** at weak KL penalties is expected — it is
  part of the story the diagnostics make visible, not a bug to silence.
- **flash-attn** optional; without it set `model.attn_implementation: sdpa`.
