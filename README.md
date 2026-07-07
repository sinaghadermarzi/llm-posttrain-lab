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

The experiment makes this concrete with one plot: **reward achieved vs
KL(policy ‖ reference)** — how much preference-reward each method buys per nat
of distribution shift. Expected picture: PPO/GRPO trace efficient frontiers
(reward up, KL bounded by the coefficient); SFT walks rightward with no control
over KL; DPO lands between, with its β sweep as the knob; and the β=0 GRPO run
shows what unconstrained reward optimization does (reward hacking, visible as
train-RM/judge divergence).

## Setup

- **Task**: helpfulness alignment on [`HuggingFaceH4/ultrafeedback_binarized`](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
- **Model**: [`Qwen/Qwen2.5-1.5B`](https://huggingface.co/Qwen/Qwen2.5-1.5B) (base), full fine-tuning
- **Framework**: [TRL](https://github.com/huggingface/trl) 1.7 + accelerate/DeepSpeed
- **Hardware target**: one node with 4× A100 80GB, ~2.5–3 days total

### Pipeline

```
Qwen2.5-1.5B (base)
   │  stage 1: SFT on train_sft (~61k conversations)
   ▼
π_SFT  ── shared starting point of every arm AND the KL reference for eval
   │  stage 2: RewardTrainer on train_prefs → RM (gated: pairwise acc ≥ 0.65)
   ▼
stage 3 arms (all start from π_SFT):
   SFT   continued SFT on chosen responses      no KL knob (that's the point)
   DPO   train_prefs pairs                      β ∈ {0.01, 0.05, 0.1, 0.3}
   PPO   train_gen prompts + RM                 kl_coef ∈ {0.02, 0.05, 0.1}
   GRPO  train_gen prompts + RM (vLLM colocate) β ∈ {0.0, 0.01, 0.04, 0.1}
   │  stage 4: per-checkpoint eval → frontier points
   ▼
stage 5: results/figures/frontier.png
```

Why start from the **base** model instead of Qwen2.5-1.5B-Instruct: the Instruct
model already embeds Qwen's own RLHF, which would muddy both the reference
distribution and the reward headroom. If π_SFT chat quality disappoints, set
`model.base: Qwen/Qwen2.5-1.5B-Instruct` in `configs/base.yaml` and skip stage 1.

### Evaluation (stage 4) — every checkpoint becomes one frontier point

- 512 held-out `test_prefs` prompts × 4 samples at temperature 1.0 / top-p 1.0
  (sampling from the true policy distribution keeps the KL estimate unbiased).
- **KL(π ‖ π_SFT)**: exact response-token logprobs from two batched HF forward
  passes; sequence KL via the low-variance k3 estimator (`r − 1 − log r`,
  Schulman), k1 logged as cross-check.
- **Two rewards per point**: the stage-2 training RM (the objective PPO/GRPO
  optimized) and a held-out judge, `Skywork/Skywork-Reward-V2-Llama-3.1-8B`,
  which no method ever saw. Both z-scored against π_SFT's score distribution.
  A widening train-RM/judge gap = reward hacking, and is itself a deliverable.
- Diagnostics logged for free: response length, token entropy (mode-collapse
  indicators), and DPO's chosen/rejected logprob trajectories.

## Quickstart

```bash
pip install -e .                    # analysis/data work
pip install -r requirements.txt    # exact pinned training stack (GPU node)

make smoke      # REQUIRED first: full 6-stage pipeline on Qwen2.5-0.5B,
                # tiny slices, 1 GPU, <30 min. Gates all API/version drift.

./scripts/run_all.sh   # the real thing: ~2.5-3 days on 4x A100 80GB
```

Every stage is idempotent (DONE markers, per-checkpoint eval skip), so
`run_all.sh` can be re-run after any crash/preemption and continues where it
stopped. Stages can also be run individually:

| Stage | Command | Est. wall-clock (4×A100) |
|---|---|---|
| 0 smoke | `make smoke` | ~30 min (1 GPU) |
| 1 π_SFT | `make sft-base` | 2–3 h |
| 2 reward model | `make rm` | 2–3 h |
| 3a SFT arm | `make arm-sft` | ~2 h |
| 3b DPO sweep ×4 | `make arm-dpo` | 4–6 h (parallel, 1 GPU each) |
| 3c PPO sweep ×3 | `make arm-ppo` | 15–21 h (sequential, full node) |
| 3d GRPO sweep ×4 | `make arm-grpo` | 9–14 h (sequential, full node) |
| 4 eval all ckpts | `make eval` | ~3 h (sharded across GPUs) |
| 5 figures | `make plots` | minutes |

Overrides are OmegaConf dotlist args, e.g.
`python -m posttrain_lab.train.dpo configs/arm_dpo.yaml train.beta=0.2`.
Sweep grids live in `configs/sweeps.yaml`. Disk: budget ~250 GB for
checkpoints + HF cache (`results_root` is configurable).

## Reading the figures

- **`frontier.png`** (the headline): judge reward (left) and train-RM reward
  (right) vs sequence KL from π_SFT. Up-and-left is good. Each curve is one
  run's checkpoints in training order, labeled with its KL-knob value; the
  star is π_SFT. Look for: RL curves dominating at fixed KL; the SFT curve
  marching right without a knob to stop it; β=0 GRPO soaring on the right
  panel while sagging on the left (pure reward hacking).
- **`dpo_logps.png`**: DPO's chosen (solid) and rejected (dashed) logprobs —
  the "squeezing" failure mode where both decline: DPO increases the *margin*
  while moving mass off the preference data entirely.
- **`diagnostics.png`**: length-vs-KL (length hacking), entropy-vs-KL (mode
  collapse), and the train-RM−judge gap vs KL (reward hacking, quantified).

## Repo map

```
configs/            all hyperparameters; base.yaml + per-stage + sweeps.yaml + smoke.yaml
src/posttrain_lab/
  data.py           UltraFeedback loaders (sft / preference / prompt-only / eval prompts)
  train/            thin TRL wrappers: sft, rm, dpo, ppo, grpo
  eval/             generate → kl (k3/k1) → dual reward scoring → per-ckpt driver
  utils/            checkpoint discovery, DONE markers, offline-first logging
analysis/           aggregate.py → frontier.csv, frontier_plot.py, diagnostics.py
scripts/            stage launch scripts + run_all.sh orchestration
```

## Known risks / gotchas

- **TRL API drift**: PPOTrainer currently lives in `trl.experimental.ppo`
  (imported defensively in `train/ppo.py`); GRPO config fields have shifted
  across minors. The pins in `requirements.txt` are load-bearing — bump trl
  and vllm *as a pair* (TRL 1.7 supports vllm 0.13–0.23) and re-run `make smoke`.
- **PPO memory**: four 1.5B models resident (policy/ref/RM/value). Fine under
  ZeRO-2 on 80GB; if OOM, lower `train.local_rollout_forward_batch_size` first.
- **GRPO vLLM colocate**: training and vLLM share GPUs;
  `vllm_gpu_memory_utilization: 0.25` is the starting point — too high OOMs
  the trainer, too low OOMs the KV cache.
- **Reward hacking / length bias** at weak KL penalties is expected — it is
  part of the story the diagnostics make visible, not a bug to silence.
- **flash-attn** is commented out in requirements; install it on the node if
  you want it, or set `model.attn_implementation: sdpa`.
