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
            SFT(gold) | DPO β∈{.01,.05,.1,.3} | GRPO(verifier) β∈{0,.01,.04,.1}
```

Each task's π_SFT is both the shared starting point of its arms and the KL
reference for its eval; the β=0 GRPO runs trace what *unconstrained* reward
optimization does.

## Evaluation — every checkpoint becomes one frontier point

Per checkpoint: sample N responses per held-out prompt at temperature 1.0
(unbiased MC), then compute **metric groups** (per-task list in
`configs/tasks/*.yaml`):

| Group | Evidence | Key outputs |
|---|---|---|
| `core` / `core_math` | the frontier | k3/k1 sequence-KL vs π_SFT; dual reward scores (helpful) or pass@1/pass@8 (math) |
| `retention_ppl` | capability retention | wikitext-2 perplexity |
| `retention_harness` | capability retention | lm-eval HellaSwag/ARC accuracy (all checkpoints by default — `eval.harness.which: final` to economize) |
| `diversity` | mode collapse | distinct-1/2/3, self-BLEU across the N samples (reused, not regenerated) |
| `pref_logps` | DPO failure modes | held-out logp(chosen), logp(rejected), margin — for EVERY arm |

**Metric groups are incremental**: each checkpoint has one summary row; a group
runs only if its keys are missing (atomic read-merge-replace). Adding a group
to the config later and re-running `make eval` computes only the new metrics —
nothing is regenerated. (Corollary: don't tail the summary files; they are
rewritten, not appended.)

## Quickstart

```bash
pip install -r requirements.txt && pip install -e .

make smoke        # REQUIRED first: both tracks end-to-end on Qwen2.5-0.5B,
                  # tiny slices, 1 GPU, <30 min. Gates all API/version drift,
                  # both reward paths, and every metric group.

make helpful-all  # ~2.5-3 days on 4x A100 80GB (+~15-20h for harness-everywhere)
make math-all     # ~12-15 h on 4x A100 80GB
```

Everything is idempotent (DONE markers, incremental eval): re-run any script
after a crash and it continues. Per-stage targets:

| helpful | math | |
|---|---|---|
| `make sft-base` | `make math-sft-base` | π_SFT (2–3 h / ~30 min) |
| `make rm` | `make math-pairs` | RM (2–3 h) / self-gen pairs (~1 h) |
| `make arm-sft arm-dpo` | `make math-arm-sft math-arm-dpo` | imitation + DPO sweep |
| `make arm-ppo arm-grpo` | `make math-arm-grpo` | RL sweeps (the long pole) |
| `make eval` | `make math-eval` | sharded, incremental |
| `make plots` | `make math-plots` | figures |

Overrides are OmegaConf dotlist args
(`python -m posttrain_lab.train.dpo configs/math/m3_arm_dpo.yaml train.beta=0.2`);
grids live in `configs/{helpful,math}/sweeps.yaml`. Disk: ~250 GB helpful,
~60 GB math.

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
  train/                  thin TRL wrappers (the swappable layer for future verl/OpenRLHF)
  eval/                   run_eval group engine; kl, reward_score, passk, retention, diversity, pref_logps
analysis/                 aggregate → frontier.csv; frontier, frontier_math, retention, diversity, pref_logps, diagnostics
scripts/{helpful,math}/   per-stage launchers; run_all_{helpful,math}.sh; run_smoke.sh
```

## Known risks / gotchas

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
