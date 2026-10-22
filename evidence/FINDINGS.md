# Findings

The claims the experiments support so far, one per ID. A write-up cites the IDs.

**Status:** `tentative` (one seed, or evidence not yet in a snapshot) · `supported` (holds across seeds, with an
across-seed CI) · `refuted` · `superseded` (replaced by a sharper claim; say which).

**Every entry gives:**
- the claim in one sentence;
- status, number of seeds, and the snapshot it was last checked against;
- evidence: snapshot files, with numbers and their CIs;
- caveats;
- history: how the claim changed, and why.

**Conventions:**
- KL is `kl_k1` in nats per sequence (`kl_k3` explodes far from π_SFT).
- pass@1 differences of 0.02 or less count as ties unless a paired CI says otherwise.
- "Matched KL" means checkpoints within about 0.3 nats.
- Picking each arm's best checkpoint on the test prompts flatters it, so prefer the pre-registered comparisons in
  `math/snapshot.yaml`.

> **Snapshot pending.** The numbers below are seed 17 on the full GSM8K test set (1,319 prompts × 8 samples). They
> come from `frontier.csv` and `compare_math.py` output on the pod on 2026-10-22, before the first snapshot existed.
> Exact values are from the tables. Values marked ≈ were read off the figures and need replacing with table values.
> The first `make evidence` snapshot should replace each "pending" with a file link.

---

## Math track (GSM8K, verifier reward, Qwen2.5-1.5B)

### M1. Up to ~3 nats, DPO, DPO+NLL and GRPO share one accuracy-vs-KL curve

Status: tentative (1 seed). Last checked: seed 17, full test, 2026-10-22 (snapshot pending).

Evidence:
- `figures/frontier_math.png`: from π_SFT's 0.43 the three arms climb together to about 0.55–0.58 by ~3 nats.
- At 2.13 nats, almost exactly matched: GRPO β=0.01 ckpt 2 is 0.559 (2.134 nats) vs DPO β=0.05 ckpt 2 at 0.540
  (2.128 nats), a gap of +0.019. Its paired CI is pending (`snapshot.yaml` line 1).
- At ~2.66 nats: DPO β=0.3 ckpt 4 reaches 0.576. GRPO has no checkpoint there; interpolating GRPO β=0.01's
  ckpt 2 (2.13 nats, 0.559) and ckpt 3 (3.05 nats, 0.588) gives ~0.576 at 2.66, so it's a tie. The paired
  bracket is pending (line 2).

Caveats:
- GRPO saves only 5 checkpoints, so it has none between 2.26 and 3.05 nats.
- DPO β=0.3 ckpt 4 is DPO's best-looking point.
- The +0.019 at 2.13 nats may not be a tie: wait for its paired CI.

History: the first claim from the 512-prompt eval (2026-10-22, SE ~0.015–0.02) was "DPO β=0.3 ties or edges GRPO at
~2–2.7 nats". It survives the full test set for seed 17.

### M2. Above ~3 nats GRPO keeps gaining while DPO and DPO+NLL level off

Status: tentative (1 seed). Last checked: seed 17, full test, 2026-10-22 (snapshot pending).

Evidence:
- ~3.0–3.2 nats: GRPO β=0.01 ckpt 3 is 0.588 (3.05 nats). In the same band: DPO+NLL β=0.3 ckpt 5 is 0.558
  (2.99), DPO β=0.3 ckpt 5 is 0.570 (3.16), DPO+NLL β=0.1 ckpt 3 is 0.575 (3.15). Paired CIs pending (line 3).
- GRPO goes on to 0.605 (β=0.01, 3.26 nats) and 0.615 (β=0.0, 4.95 nats). DPO and DPO+NLL stay around
  0.56–0.58 out to ~10 nats (≈, `frontier_math.png`).
- Paired against DPO β=0.3 ckpt 4 (0.576 at 2.66 nats), not at matched KL:
  - GRPO β=0.01 last: +0.029 ± 0.005, 95% CI [+0.019, +0.039];
  - GRPO β=0.0 last: +0.039 ± 0.005, 95% CI [+0.028, +0.050].

  These show the paired SE is half each point's own ±0.010.

Caveats:
- The not-matched comparisons include KL: GRPO buys part of that gap by moving further from π_SFT.
- One seed: these CIs cover eval sampling, not training randomness.

History: first seen on the 512-prompt eval (GRPO 0.596 at 3.3 nats, 0.616 at 5.1 nats).

### M3. Pure DPO with small β collapses, through likelihood displacement

Status: tentative (1 seed). Last checked: seed 17, full test, 2026-10-22 (snapshot pending).

Evidence:
- `frontier_math.png`: β=0.05 drifts to ≈20 nats and pass@1 falls to ≈0.38. β=0.01 drifts to ≈80 nats and falls to
  ≈0.22, and its pass@8 to ≈0.55, against π_SFT's 0.43 and 0.82.
- `dpo_logps.png` (training): at β=0.01, chosen answers fall from ≈−35 to ≈−180 nats and rejected ones to ≈−335. The
  margin grows while the correct answers become far less likely.
- `pref_logps.png` (held-out pairs) shows the same: chosen falls to ≈−195 at β=0.01.
- β=0.1 and β=0.3 keep chosen at ≈−37 to −45 and don't collapse.
- `diagnostics.png`: collapsed runs write longer answers (≈185–195 tokens, against ≈130) at very low entropy (≈0.09).

Caveats: this is at learning rate 2e-6 × 2 epochs. The collapse depends on the training budget as much as on β.

History:
- lr 5e-7 × 1 epoch barely moved: KL 0.16, loss ≈ ln 2, every β alike.
- lr 5e-6 × 2 epochs peaked at step 58 (pass@1 0.57) and had collapsed by step 116 (0.07–0.37) for every β (PR #17).
- 2e-6 × 2 epochs with 10 checkpoints (PR #18) traces the rise, the peak and the collapse; only small β collapses.

### M4. Adding an NLL term on the chosen answers prevents the collapse

Status: tentative (1 seed). Last checked: seed 17, full test, 2026-10-22 (snapshot pending).

Evidence:
- `dpo_logps.png` and `pref_logps.png`: with the NLL term, chosen log-probs stay at ≈−35 for every β, and rejected
  ones fall only to ≈−60 to −85.
- `frontier_math.png`: every DPO+NLL β stays within ≈2–8 nats, with pass@1 0.52–0.58. Table points: β=0.3 ckpt 4 is
  0.569 (2.47 nats); β=0.1 ckpt 3 is 0.575 (3.15 nats).

Caveats:
- DPO+NLL's best points are slightly below pure DPO's best (≈0.57 vs 0.576 on the full test set). The 512-prompt
  eval put the gap at 1–2 points.
- The NLL weight is 1.0 and wasn't swept.

History: the arm was added in PR #18 because of M3.

### M5. The accuracy gains come from sharpening, not from solving new problems

Status: tentative (1 seed). Last checked: seed 17, full test, 2026-10-22 (snapshot pending).

Evidence:
- `frontier_math.png`: pass@1 rises from 0.43 to 0.615, while pass@8 goes from 0.82 to at most ≈0.90.
- `diversity.png`, as KL grows, for every arm:
  - distinct-2: ≈0.62 → 0.45;
  - self-BLEU across samples: ≈0.35 → 0.55;
  - token entropy (GRPO): ≈0.35 → 0.21;
  - pass@8 − pass@1: ≈0.39 → 0.27–0.30.
- GRPO shrinks pass@8 − pass@1 the fastest. Collapsed DPO's value goes back up (≈0.33) only because its pass@1
  falls faster than its pass@8.

Caveats: pass@8 uses the same 8 samples as pass@1, so this doesn't show whether larger k would still find new
solutions.

### M6. General ability is kept, except in the collapsed runs

Status: tentative (1 seed). Last checked: seed 17, full test, 2026-10-22 (snapshot pending).

Evidence (`retention.png`):
- WikiText-2 perplexity stays at ≈11.75–12.3 (π_SFT 11.8) up to ≈10 nats. Only the collapsed DPO runs reach
  ≈13.2–13.4.
- HellaSwag (≈10k items, SE ≈0.005) drops from ≈0.682 to ≈0.671 only in the collapsed runs.
- ARC-Easy (2,376 items, SE ≈0.009) drops to ≈0.753, also only in the collapsed runs.
- ARC-Challenge (1,172 items, SE ≈0.015) varies 0.460–0.476, all within noise.

Caveats: the harness scores are unpaired here. Paired per-item differences would be tighter.

### M7. Continued SFT on gold answers moves the model without improving it

Status: tentative (1 seed). Last checked: seed 17, full test, 2026-10-22 (snapshot pending).

Evidence:
- `frontier_math.png`: the SFT arm moves 2.6–8 nats while its pass@1 stays at 0.40–0.47, around π_SFT's 0.43. Table
  points: ckpt 1 is 0.406 (2.60 nats), ckpt 3 is 0.422 (2.84 nats).
- `diversity.png`: it keeps π_SFT-like diversity (pass@8 − pass@1 ≈0.36–0.41).

Caveats: π_SFT already trained 3 epochs on the same gold answers, so this arm mostly over-fits them further.

History: at 512 prompts it was 0.451 at 8.6 nats, the worst curve.

---

## Helpful track (UltraFeedback, learned reward)

Not run yet.
