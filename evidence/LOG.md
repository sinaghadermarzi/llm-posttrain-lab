# Lab log

Newest first. One entry per run, incident or decision, with the numbers that mattered at the time and links to the
PR or snapshot. Times are the pod's clock. This is the "what happened, including what went wrong" record; the claims
it leads to live in `FINDINGS.md`.

## 2026-10-23

**First snapshot: [`math/2026-10-23_seeds17-19_full-test`](math/2026-10-23_seeds17-19_full-test/)** (PR #25).
- **What it holds:** 3 seeds × 112 checkpoints on 1,319 prompts × 8 samples. All 7 `snapshot.yaml` comparisons ran
  `ok`. 2.8 MB, 50 files.
- **Exported on the pod** with `make evidence TASK=math NAME=seeds17-19_full-test`.
- **Its manifest commit, `3f812b9`, isn't on GitHub.** It's the pod checkout's commit in the ADO repo
  `Personal-Sina-Ghadermarzi`, which carries this repo as a squashed subtree. Its `llm-posttrain-lab/` tree is
  identical to GitHub `e996453` (tree `365c388`).
- **Getting it to GitHub:** the pod reaches only that ADO repo. The snapshot was committed there (`6aa0816`) and
  replayed here unchanged.
  - The laptop now syncs both directions with `sync_llm_posttrain_lab.sh` in the ADO repo: `status`, `export` to a
    GitHub branch, and `pull`.

**FINDINGS updated against it.**
- `supported`: M3, M4, M5, M6, M10.
- `tentative`: M2, M8, M9.
- `superseded`: M1 (by M8) and M7 (by M10).

What changed:
- **The shared curve below ~3 nats (M1) holds only in seed 17.** In seeds 18–19 GRPO is ahead at equal or lower KL.
- **Pure DPO at β=0.1 also collapses in one seed:** seed 18 ends at 0.166.
- **DPO+NLL moves up to ~15 nats, not ≈2–8,** and still never collapses.
- **Continued SFT gains a small amount:** +0.031 [+0.006, +0.056].

**Problem with the fixed comparisons.**
- `snapshot.yaml` names checkpoints by position, chosen to match KL in seed 17.
- In seeds 18 and 19 the DPO arms move 1.5–4× further per step. For example, DPO β=0.05 ckpt 2 is at 2.13 / 8.14 /
  3.05 nats in seeds 17 / 18 / 19 (FINDINGS M9).
- So the pooled lines aren't matched-KL tests.
- **Next:** KL-targeted comparisons in `snapshot.yaml`. The KL targets are to be fixed from the design (e.g. 2, 3, 5
  nats), in their own commit, before a new snapshot. This snapshot stays as exported.
  - We have seen this snapshot's numbers, so the reason for the change is the KL mismatch above, not the results.

**Open question: why DPO's KL per step varies so much between seeds.**
- Each seed self-generates its own pairs: 9,598 / 9,764 / 9,663, with similar solve rates. Data order differs too.
- **A test:** train seeds 18 and 19 on seed 17's pairs. That separates pair-sampling randomness from optimization
  randomness.

**Seed 19 run** (after `make math-seeds SEEDS="18 19"` resumed seed 18's eval).
- **Training:**

  | Stage | Finished | Detail |
  |---|---|---|
  | π_SFT | 01:57 | stop probe 0.985 |
  | Pairs | 02:07 | 9,663, solve rate 0.604 |
  | SFT arm | 02:23 | |
  | DPO | 03:02 | 288 steps |
  | DPO+NLL | 03:43 | |
  | GRPO | 04:43–07:50 | the 4 runs, one after another |

- **Eval and figures** took 07:50–14:12 (about 6.3 h). The whole seed took about 12.4 h.
- **Training times from `MANIFEST.json`.** Seed 19's are the trainer's own totals; seeds 17–18 are estimated from
  checkpoint times.
  - π_SFT ≈ 7 min.
  - SFT arm 14–16 min.
  - Each DPO or DPO+NLL run 35–39 min (4 run in parallel).
  - Each GRPO run 57–61 min.
- **Rambling at the last checkpoint** (samples hitting the 512-token cap, seeds 17 / 18 / 19):
  - pure DPO β=0.01: 3% / 20% / 11%;
  - DPO+NLL: 0–4%;
  - GRPO: 0%.

**Server shutdown during seed 18's eval** (between 10-22 16:16 and 10-23 01:18).
- Seed 18 was at 107/112 checkpoints, with one SFT checkpoint halfway through its metric groups.
- Nothing was lost, because rows are written after every group. `make math-seeds SEEDS="18 19"` finished seed 18's
  last 5 checkpoints, then ran seed 19.

## 2026-10-22

**Seed 18 and 19 running** (`make math-seeds SEEDS="17 18 19"`, started about 04:49).
- **Seed 17:** training skipped, since every stage was DONE. Its re-score on the full test set (core_math +
  diversity on 108 checkpoints) finished at about 08:00, roughly 3 h.
- **Seed 18 training:** 08:02–14:05 (6 h).
  - π_SFT stop probe 0.984 (seed 17: 0.985).
  - 9,764 DPO pairs (seed 17: 9,598); solve rate 0.597 (0.600).
  - DPO runs are 290 steps (seed 17: 286), because each seed generates its own pairs. That's why seed means are
    matched by checkpoint index, not step.
- **Seed 18 eval:** started 14:09, at about 14 min per checkpoint per shard, since a fresh checkpoint runs all five
  metric groups. Expected done about 20:45. Seed 19 then needs about 13 h more.
- GPUs sat idle in about 46% of samples during eval: every metric group reloads the checkpoint from the network
  drive. That's a possible speed-up later.

**Eval crash: vLLM engine couldn't start at the second checkpoint** (fixed in PR #23).
- The full-test-set re-score from PR #19 died on all 4 shards after exactly one checkpoint each (core_math 4/112):
  "Free memory on device cuda:0 (36.79/79.25 GiB) on startup is less than desired GPU memory utilization (0.85,
  67.36 GiB)".
- Cause: each checkpoint starts a new vLLM engine in its own process, while the eval process still held the KL
  step's two models in PyTorch's cache, and the previous engine was never shut down.
- It hadn't shown before because the slow retention groups ran between checkpoints. The re-score ran only
  core_math and diversity, back to back.
- Fix: free GPU memory after KL, after each metric group and before each engine starts, and shut each engine down
  explicitly.

**Seed 17 on the full test set** (first look, before any snapshot).
- Matched KL (`frontier.csv`):
  - at 2.13 nats, GRPO β=0.01 0.559 vs DPO β=0.05 0.540;
  - at ~2.66 nats, DPO β=0.3 0.576 equals GRPO interpolated;
  - at ~3.05–3.16 nats, GRPO 0.588 vs DPO and DPO+NLL 0.558–0.575.
- Paired vs DPO β=0.3 ckpt 4, not at matched KL: GRPO β=0.01 last +0.029 [+0.019, +0.039]; GRPO β=0.0 last +0.039
  [+0.028, +0.050].
- Each checkpoint's own SE is ±0.010, the paired SE ±0.005. These are FINDINGS M1 and M2.

**Decided: an evidence layer in git** (this folder): snapshots made by `make evidence`, claims with IDs and status,
and this log. The matched-KL comparisons are fixed in `math/snapshot.yaml` before the multi-seed numbers are in.

**Multi-seed support merged** (PRs #20–#22).
- `make math-all SEED=N` and `make math-seeds SEEDS="…"`.
- Eval scores each seed's runs against that seed's own π_SFT.
- `status` and `reset` know about seeds.
- `frontier_seeds.csv`, and figures with across-seed error bars.
- `compare_math.py --pool`, whose 95% CI uses t with n_seeds − 1 degrees of freedom. With 2 seeds t₁ = 12.7, which
  is useless; 3 seeds (t₂ = 4.3) is the practical minimum.

**Eval moved to the full GSM8K test set** (PR #19): 1,319 prompts × 8 samples instead of 512. At 512 prompts the
pass@1 SE was about 0.015–0.02, the same size as the gaps being judged. Rows record their prompt count, and a changed
count re-samples only the prompt-dependent groups. Added `compare_math.py` for paired per-question comparisons.

**Third DPO budget + the DPO+NLL arm** (PR #18): lr 2e-6 × 2 epochs, 10 checkpoints per run, at 512 prompts.
- Pure DPO peaks at about 0.58–0.59 around steps 87–174.
- β 0.01 and 0.05 then collapse, to 0.24 and 0.37 pass@1, at 79 and 21 nats, with logp_chosen −194 and −68.
- β 0.1 and 0.3 hold at about 0.55.
- DPO+NLL (TRL 1.7 `loss_type: [sigmoid, sft]`, weights 1, 1) never collapses: final 0.53–0.56, KL ≤ 8, peaks
  1–2 points below DPO's.
- The plots moved to `kl_k1`, because `kl_k3` explodes far from π_SFT (up to 1e21+).

## 2026-10-21

**Second DPO budget overshot** (PR #17): lr 5e-6 × 2 epochs (286 steps).
- pass@1 peaked at step 58 (β=0.1: 0.570; β=0.3: 0.554, with pass@8 0.869).
- By step 116 it had collapsed to 0.07–0.37. Answers got longer (≈150–175 tokens), and held-out logp_chosen kept
  falling while training preference accuracy reached 0.94–0.99: likelihood displacement.
- Harness scores stayed flat. The run is archived under `math/archive/` on the pod.
- Added `make reset --stage` to redo one arm.

**First valid math run** (seed 17, 512 test prompts).
- π_SFT pass@1 0.418, pass@8 0.814.
- GRPO β=0 reached 0.616 at 6.2 nats; β=0.1 0.523 at 0.8 nats.
- SFT arm 0.451 at 8.6 nats, the worst curve.
- DPO at lr 5e-7 × 1 epoch (143 steps) barely moved: KL 0.16, loss ≈ ln 2, every β on the same curve. Adam
  normalizes β out while the loss is still in its linear regime.

**Untrained chat-token embeddings** (PR #16).
- After the end-of-turn fix, 39% of the pair samples still hit the 512-token cap. The model finished its answer,
  emitted a junk token where `<|im_end|>` belongs, then invented a new question.
- Cause: Qwen2.5 *base* never trained `<|im_start|>` / `<|im_end|>`. Their embedding rows sit in a crowd of ~2,800
  near-identical untrained rows, so π_SFT gave `<|im_end|>` only 0.12 probability where answers end.
- Fix: before SFT, re-initialize those rows (end of turn ← the original eos row), and add a stop probe that gates the
  pipeline right after π_SFT (≥ 0.8; now 0.985).

## 2026-10-20

**First full math run was invalid: generation never stopped** (PR #14).
- Qwen2.5 base declares eos `<|endoftext|>`, but the chat template ends assistant turns with `<|im_end|>`.
- 100% of eval samples ran to the token cap, with invented follow-up turns, and GRPO `clipped_ratio` was 1.
- The DPO pairs were mostly that tail: loss ≈ ln 2, preference accuracy 0.30–0.36.
- KL and diversity were measured on garbage. pass@k mostly survived, because the verifier reads the first `####`.
- Fix: the end-of-turn token becomes the eos everywhere, stop ids are passed to every generator, and status checks
  for non-stopping generation.

Also: `make status`, the pipeline monitor (PR #13); gen_pairs samples every train prompt (PR #12); dataset splits
load in memory so ranks don't race (PR #11).

## 2026-10-17 – 2026-10-19

Infrastructure for the pod:
- MLflow tracking, preflight checks and an onboarding notebook (PR #4);
- DPO under single-GPU ZeRO-2 (PR #3);
- every Hugging Face asset usable offline, with a configurable `RESULTS_DIR` (PRs #5, #7, #8);
- a working `make smoke` (PRs #6, #9).

## 2026-07-07

Repo created (PR #1): SFT vs DPO vs PPO/GRPO on a helpfulness track with a learned reward. Then the math/RLVR track
(GSM8K, verifier reward) and the evidence metrics: capability retention, diversity, DPO failure modes (PR #2).
