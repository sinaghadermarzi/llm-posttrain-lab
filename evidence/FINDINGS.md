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

> **Last snapshot:** [`math/2026-10-25_seeds17-19_dense-grpo`](math/2026-10-25_seeds17-19_dense-grpo/) (below: `S/`).
> It supersedes [`math/2026-10-23_seeds17-19_kl-matched`](math/2026-10-23_seeds17-19_kl-matched/) (below: `K/`). GRPO was retrained in every seed with 10 checkpoints per run instead of
> 5 (PR #29), and the GRPO positions in `snapshot.yaml` were doubled to match (PR #30). The SFT, DPO and DPO+NLL rows
> are identical to `K/`'s.
> - A History line's `S/` means the snapshot that was current on its date: before 2026-10-25, `K/` or its
>   predecessor `math/2026-10-23_seeds17-19_full-test` (same data).
> It covers seeds 17, 18 and 19 on the full GSM8K test set (1,319 prompts × 8 samples), and all 396 checkpoints.
> - **Per-seed numbers** are read from `S/tables/frontier.csv`, and are listed as seed 17 / 18 / 19.
> - **Across-seed values** are mean ± SE with a 95% CI from t with 2 df (t₂ = 4.30).
> - **"Derived"** marks a number computed from those tables. It is not one of the comparisons fixed in
>   `snapshot.yaml`, so it is descriptive.
> - **π_SFT** scores pass@1 0.426 / 0.427 / 0.437 and pass@8 0.821 / 0.811 / 0.813.

---

## Math track (GSM8K, verifier reward, Qwen2.5-1.5B)

### M1. Up to ~3 nats, DPO, DPO+NLL and GRPO share one accuracy-vs-KL curve

Status: superseded (by M8, now by M11). Last checked: `K/`, 3 seeds, 2026-10-23.

The claim holds only in seed 17. M11 states what the three seeds show below ~3 nats.

Evidence (seed 17, in `K/`, with the 5-checkpoint GRPO runs):
- [`K/tables/compare_1.txt`](math/2026-10-23_seeds17-19_kl-matched/tables/compare_1.txt): at 2.13 nats, matched within
  0.006 nats, GRPO β=0.01 ckpt 2 beats DPO β=0.05 ckpt 2 by +0.019 ± 0.005, CI [+0.009, +0.029]. So even in seed 17
  this is not a tie.
- [`K/tables/compare_2.txt`](math/2026-10-23_seeds17-19_kl-matched/tables/compare_2.txt): DPO β=0.3 ckpt 4 (0.576 at
  2.66 nats) lies between two GRPO points:
  - GRPO at 2.26 nats is behind it, −0.013 [−0.023, −0.003];
  - GRPO at 3.05 nats is ahead of it, +0.012 [+0.002, +0.022].

  So seed 17 is consistent with a tie at ~2.7 nats.

History:
- 2026-10-22, 512-prompt eval: "DPO β=0.3 ties or edges GRPO at ~2–2.7 nats".
- 2026-10-22, full test set, seed 17: the claim survived.
- 2026-10-23, `S/`, 3 seeds: superseded by M8. In seeds 18 and 19, GRPO is more accurate at equal or lower KL. Even
  in seed 17, the matched-KL gap at 2.13 nats is +0.019 with a CI that excludes 0.
- 2026-10-25, `S/`: M8, which replaced it, is refuted. M11 now covers the arms below ~3 nats: in seed 17 they are
  again within about 0.015 of each other there, with leads in both directions.

### M2. From ~3 nats up GRPO is ahead of DPO and DPO+NLL at matched KL, and the gap grows as the DPO arms level off

Status: supported (3 seeds). Last checked: `S/`, 2026-10-25.

**Every matched pair at ~3 and ~5 nats has GRPO ahead,** by 0.022–0.069, each with a paired CI excluding 0. The 7
pairs cover all three seeds:

| Target | Seed | DPO arm | KL, arm vs GRPO (nats) | DPO arm − GRPO (95% CI) |
|---|---|---|---|---|
| ~3 nats | 17 | DPO β=0.3 | 3.16 vs 3.01 | −0.022 [−0.032, −0.011] |
| ~3 nats | 17 | DPO+NLL β=0.3 | 2.99 vs 3.01 | −0.034 [−0.044, −0.023] |
| ~3 nats | 19 | DPO β=0.3 | 2.97 vs 2.94 | −0.046 [−0.057, −0.036] |
| ~3 nats | 19 | DPO+NLL β=0.3 | 2.68 vs 2.94 | −0.049 [−0.059, −0.038] |
| ~5 nats | 17 | DPO β=0.3 | 4.99 vs 4.79 | −0.045 [−0.057, −0.033] |
| ~5 nats | 18 | DPO+NLL β=0.3 | 5.08 vs 5.12 | −0.069 [−0.080, −0.057] |
| ~5 nats | 19 | DPO β=0.3 | 5.39 vs 5.16 | −0.043 [−0.054, −0.032] |

Evidence:
- **KL-targeted comparisons,** fixed in `snapshot.yaml` before the snapshot. Each uses every seed's checkpoint nearest
  the target. GRPO's picks are now within 0.21 nats of the target at both targets in every seed. Differences are the
  DPO arm minus GRPO.
  - **~3 nats,** [`S/tables/compare_9.txt`](math/2026-10-25_seeds17-19_dense-grpo/tables/compare_9.txt).
    - GRPO β=0.01 scores 0.589 ± 0.004 (2.93–3.01 nats).
    - DPO β=0.3: −0.047 ± 0.015, CI [−0.110, +0.016]. Per seed −0.022 / −0.072 / −0.046.
    - DPO+NLL β=0.3: −0.050 ± 0.010, CI [−0.094, −0.007]. Per seed −0.034 / −0.068 / −0.049.
    - Seed 18 isn't matched for either arm. Its DPO-arm checkpoints sit 1.13 and 0.46 nats further out than GRPO's.
  - **~5 nats,** [`S/tables/compare_10.txt`](math/2026-10-25_seeds17-19_dense-grpo/tables/compare_10.txt).
    - GRPO β=0.0 scores 0.606 ± 0.002 (4.79–5.16 nats).
    - DPO β=0.3: −0.057 ± 0.013, CI [−0.111, −0.002]. Per seed −0.045 / −0.082 / −0.043.
    - DPO+NLL β=0.3: −0.062 ± 0.011, CI [−0.110, −0.013]. Per seed −0.039 / −0.069 / −0.076.
    - Unmatched: seed 18's DPO pair (0.35 nats apart), and DPO+NLL in seeds 17 (1.36) and 19 (0.33).
- **Seed 17's position-matched comparison at ~3 nats agrees,** [`S/tables/compare_3.txt`](math/2026-10-25_seeds17-19_dense-grpo/tables/compare_3.txt). GRPO β=0.01 ckpt 6 (2.95 nats) is ahead of all
  three DPO-arm checkpoints at 2.99–3.16 nats, by 0.011–0.028, each CI excluding 0.
- **Levelling off,** seed means at the two targets:

  | Arm | ~3 nats | ~5 nats |
  |---|---|---|
  | GRPO | 0.589 | 0.606 |
  | DPO β=0.3 | 0.542 | 0.550 |
  | DPO+NLL β=0.3 | 0.538 | 0.545 |

- **Most accurate points,** [`S/tables/compare_7.txt`](math/2026-10-25_seeds17-19_dense-grpo/tables/compare_7.txt). Not matched KL: GRPO β=0.0 last scores 0.633 ± 0.004 (5.67–7.64 nats), against
  DPO β=0.3 ckpt 4's 0.551 ± 0.013. That's +0.082, CI [+0.008, +0.155].
- **Derived from `S/tables/frontier.csv`.** Best checkpoint at any KL: GRPO − best DPO or DPO+NLL is +0.067 ± 0.021,
  CI [−0.025, +0.158].

Caveats:
- **Seed 18 has one matched pair** (DPO+NLL at ~5 nats). Its DPO β=0.3 jumps from 1.48 to 4.06 nats, and from 4.16 to
  5.47, between consecutive checkpoints. Its unmatched pairs (−0.072, −0.068, −0.082) point the same way but don't
  count.
- **DPO's across-seed CI at ~3 nats includes 0.** The other three across-seed CIs exclude it, but each includes at
  least one unmatched pair. The status rests on the matched pairs above.
- **"Level off" is descriptive:** seed means at two targets, with no test of the slope.
- **This is 3 seeds,** so t₂ = 4.30.

History:
- First seen on the 512-prompt eval: GRPO 0.596 at 3.3 nats, 0.616 at 5.1.
- 2026-10-22, seed 17, full test set: GRPO +0.029 and +0.039 over DPO β=0.3 ckpt 4.
- 2026-10-23, `math/2026-10-23_seeds17-19_full-test`, 3 seeds: the gap is larger in seeds 18–19 (up to +0.108). It
  stayed tentative: the position-matched comparisons weren't KL-matched, and their CIs mostly include 0.
- 2026-10-23, `S/` (KL-targeted comparisons): stays tentative, narrowed to the matched pairs.
  - Every matched pair favors GRPO: seeds 17 and 19 at ~3 nats, and seed 17 at ~5 nats.
  - A first draft of this update (PR #28) called it supported, on the strength of the across-seed CIs at ~5 nats.
    Review pointed out that those mix in unmatched pairs.
- 2026-10-25, `S/` (GRPO with 10 checkpoints per run): supported.
  - With GRPO's picks on target, 7 matched pairs at ~3 and ~5 nats cover all three seeds, and every one favors GRPO
    by more than 0.02 with a CI excluding 0.
  - Reworded from "Above ~3 nats GRPO keeps gaining while DPO and DPO+NLL level off": the matched-KL gap is what the
    comparisons test, and the levelling is descriptive.

### M3. Pure DPO with small β collapses through likelihood displacement: at β=0.01 in every seed, at β=0.05 in two of three, at β=0.1 in one

Status: supported (3 seeds) for β=0.01. Last checked: `S/`, 2026-10-25.

Evidence: last checkpoints, from `S/tables/frontier.csv`.

| Pure DPO | pass@1 (17 / 18 / 19) | KL, nats | Held-out logp(chosen) |
|---|---|---|---|
| β=0.01 | 0.246 / 0.172 / 0.399 | 77.8 / 121.9 / 80.9 | −193.8 / −251.8 / −214.7 |
| β=0.05 | 0.379 / 0.325 / 0.507 | 21.0 / 44.0 / 33.0 | −68.3 / −82.6 / −77.6 |
| β=0.1 | 0.557 / **0.166** / 0.525 | 10.3 / 28.3 / 21.8 | −48.1 / −57.3 / −55.7 |
| β=0.3 | 0.567 / 0.536 / 0.561 | 4.7 / 7.4 / 6.2 | −36.9 / −38.1 / −38.9 |

- **β=0.01 falls far below its own best checkpoint.** The drop is 0.333 / 0.277 / 0.147: +0.252 ± 0.055, CI
  [+0.015, +0.490].
  - Its last checkpoint ends below π_SFT in every seed. The across-seed CI of that gap includes 0, because seed 19
    falls only to 0.399.
  - Its pass@8 falls to 0.589 / 0.466 / 0.752.
- **The correct answers become far less likely.** logp(chosen) is −188 ± 17 nats below π_SFT's (≈ −32), CI
  [−260, −115].
  - `S/figures/dpo_logps.png` shows the training log-probs, one line per seed. At β=0.01, chosen answers fall to
    ≈ −180 to −245 nats and rejected ones to ≈ −330 to −430.
- **Collapsed runs ramble.** Answers grow to 186 / 267 / 208 tokens (π_SFT ≈ 130), with token NLL 0.090–0.100.
- **β=0.05 falls from its best** by 0.217 / 0.160 / 0.034: +0.137 ± 0.054, CI [−0.096, +0.370]. Only seeds 17 and 18
  end below π_SFT.
- **β=0.1 collapses only in seed 18,** to 0.166, with a minimum of 0.157.
- **β=0.3 holds** in every seed.

Caveats:
- This is at learning rate 2e-6 × 2 epochs. The collapse depends on the training budget as much as on β.
- Which β collapses depends on the seed (M9).

History:
- lr 5e-7 × 1 epoch barely moved: KL 0.16, loss ≈ ln 2, every β alike.
- lr 5e-6 × 2 epochs peaked at step 58 (pass@1 0.57) and had collapsed by step 116 (0.07–0.37) for every β (PR #17).
- 2e-6 × 2 epochs with 10 checkpoints (PR #18) traces the rise, the peak and the collapse; only small β collapses.
- 2026-10-23, `S/`, 3 seeds: supported for β=0.01. "β=0.1 and β=0.3 don't collapse" was wrong for β=0.1, which
  collapses in seed 18. The ≈ values from seed 17's figures were replaced by table values.
- 2026-10-25, `S/`: unchanged. The DPO runs and their rows are identical to `K/`'s.

### M4. Adding an NLL term on the chosen answers prevents the collapse in every seed and at every β, at no cost to DPO's best accuracy

Status: supported (3 seeds). Last checked: `S/`, 2026-10-25.

Evidence: all 12 DPO+NLL runs (4 β × 3 seeds), from `S/tables/frontier.csv`.
- **No run collapses.**
  - Final pass@1 is 0.521–0.560, and no checkpoint scores below 0.434.
  - The drop from each run's best to its last checkpoint is at most 0.055. Per β, the across-seed means are
    0.005–0.022, and every CI includes 0.
  - Pure DPO β=0.01 falls by +0.252 (M3).
- **The chosen answers stay as likely as under π_SFT.**
  - Final logp(chosen) is −34.9 to −45.2 nats (π_SFT ≈ −32).
  - At β=0.01 that's +181 ± 16 nats above pure DPO, CI [+112, +251].
  - In `S/figures/dpo_logps.png`, the right panel stays flat in every seed.
- **Final pass@1 against pure DPO at β=0.01:** +0.309 / +0.352 / +0.136. That's +0.266 ± 0.066, CI [−0.018, +0.549].
  It's borderline only because seed 19's pure DPO collapses the least.
- **Each arm's best checkpoint is the same** (derived): DPO+NLL − DPO is −0.011 / +0.002 / +0.004. That's −0.002 ±
  0.005, CI [−0.021, +0.018], a tie.

Caveats:
- **DPO+NLL moves further than first reported:** up to 3.4–14.8 nats, not ≈2–8.
- **It isn't always monotonic.** Seed 18 β=0.1 dips to 0.460 at 9.25 nats (ckpt 3), then recovers to 0.52.
- **The NLL weight is 1.0 and wasn't swept.**
- **It trails GRPO from ~3 nats up** (M2). Below that, the arms are within about 0.02 (M11).

History:
- The arm was added in PR #18 because of M3.
- 2026-10-23, `S/`, 3 seeds: supported.
  - The KL range was corrected.
  - The old caveat "DPO+NLL's best is slightly below DPO's" no longer holds: across seeds, the two arms' best
    checkpoints tie.
- 2026-10-25, `S/`: unchanged (identical DPO and DPO+NLL rows). The comparison with GRPO now points to M2 and M11.

### M5. The accuracy gains come from sharpening, not from solving new problems

Status: supported (3 seeds). Last checked: `S/`, 2026-10-25.

Evidence: GRPO β=0.0's last checkpoint against π_SFT (`S/tables/frontier.csv`; per-seed values in the table's rows).

| Metric | Change from π_SFT (95% CI) | Values |
|---|---|---|
| pass@1 | +0.203 ± 0.005 [+0.182, +0.224] | |
| pass@8 | +0.085 ± 0.005 [+0.065, +0.105] | 0.891–0.905 |
| pass@8 − pass@1 | −0.118 ± 0.001 [−0.122, −0.114] | 0.377–0.395 → 0.258–0.279 |
| distinct-2 | −0.180 ± 0.002 [−0.191, −0.169] | |
| self-BLEU across samples | +0.237 ± 0.006 [+0.212, +0.262] | |
| token NLL | | 0.37–0.39 → 0.18–0.21 |

- In `S/figures/diversity.png`, every arm except continued SFT sharpens as KL grows.

Caveats: pass@8 uses the same 8 samples as pass@1, so this doesn't show whether larger k would still find new
solutions.

History:
- 2026-10-23, `S/`, 3 seeds: supported. Table values replaced seed 17's ≈ values.
- 2026-10-25, `S/`: GRPO retrained with 10 checkpoints per run. Numbers updated; the conclusion is unchanged.

### M6. General ability is largely kept: only collapsed DPO loses clearly, while GRPO and SFT erode a little

Status: supported (3 seeds). Last checked: `S/`, 2026-10-25.

Evidence: last checkpoints against π_SFT (`S/tables/frontier.csv`, `S/figures/retention.png`).

| Run | WikiText-2 ppl | HellaSwag | ARC-Easy |
|---|---|---|---|
| π_SFT | 11.81 / 12.00 / 11.84 | 0.682 in every seed | 0.763 / 0.757 / 0.761 |
| Collapsed DPO β=0.01 | +1.73 ± 0.10 [+1.31, +2.14] | −0.010 ± 0.000 [−0.011, −0.009] | 0.734–0.755 |
| GRPO β=0.0 | +0.33 ± 0.05 [+0.10, +0.56] | −0.004 ± 0.001 [−0.006, −0.002] | −0.004 ± 0.001 [−0.007, −0.002] |
| Continued SFT | +0.62 ± 0.05 [+0.40, +0.85] | | 0.745–0.758 |

- The collapsed runs' perplexity reaches 13.3–13.8. HellaSwag's drop is about 2 SE at its ≈10k items.
- ARC-Challenge (1,172 items, SE ≈ 0.015) varies 0.450–0.477 across every checkpoint: within noise.

Caveats:
- The harness scores are unpaired. Paired per-item differences would be tighter.
- GRPO's HellaSwag change is consistent across seeds but tiny, about 1 SE. Its ARC-Easy change (≈ 2,400 items,
  SE ≈ 0.009) is under half an SE.

History:
- 2026-10-23, `S/`, 3 seeds. The claim was "kept, except in the collapsed runs". It now also states GRPO's and SFT's
  small but consistent erosion.
- 2026-10-25, `S/`: GRPO's numbers updated for the retrained runs, with its ARC-Easy change added.

### M7. Continued SFT on gold answers moves the model without improving it

Status: superseded (by M10). Last checked: `K/`, 3 seeds, 2026-10-23.

Across 3 seeds, continued SFT does improve, by a small but non-zero amount (M10).

History:
- At 512 prompts it was 0.451 at 8.6 nats, the worst curve.
- 2026-10-22, seed 17, full test set: pass@1 0.40–0.47, around π_SFT's 0.43.
- 2026-10-23, `S/`: superseded by M10.

### M8. At matched KL, GRPO is never behind DPO or DPO+NLL; the only tie is seed 17's DPO at ~2 nats

Status: refuted (3 seeds). Last checked: `S/`, 2026-10-25. M11 states what the matched pairs below ~3 nats show, and
M2 what they show from ~3 nats up.

With GRPO's denser checkpoints, two matched pairs in seed 17 have DPO β=0.3 ahead, each with a paired CI excluding 0:
- [`S/tables/compare_8.txt`](math/2026-10-25_seeds17-19_dense-grpo/tables/compare_8.txt): at ~2 nats (1.91 vs 1.95), DPO +0.011, CI [+0.001, +0.021].
- [`S/tables/compare_2.txt`](math/2026-10-25_seeds17-19_dense-grpo/tables/compare_2.txt): at 2.66 vs 2.62 nats, DPO +0.015 over GRPO β=0.0 ckpt 4, CI [+0.005, +0.025].

Both gaps are under 0.02. But under the tie convention's paired-CI exception they count as DPO leads.

The evidence it rested on is in `K/`
([`compare_8.txt`](math/2026-10-23_seeds17-19_kl-matched/tables/compare_8.txt), [`compare_9.txt`](math/2026-10-23_seeds17-19_kl-matched/tables/compare_9.txt)).

History:
- 2026-10-23, `math/2026-10-23_seeds17-19_full-test`: new, replacing M1. Its wording was "Up to ~3 nats GRPO is at
  or above DPO and DPO+NLL in every seed; DPO comes within about 0.02 only in seed 17".
- 2026-10-23, `S/` (KL-targeted comparisons): stated at matched KL. It stays tentative.
  - The only matched tie is still seed 17's DPO at ~2 nats. Seed 19's matched DPO+NLL pair (−0.018) has a paired CI
    that excludes 0, so it's a GRPO lead.
  - A first draft of this update (PR #28) counted that pair as a second tie. Review pointed out that the tie
    convention's paired-CI exception applies.
- 2026-10-25, `S/`: refuted. In `K/`, seed 17's GRPO checkpoint nearest DPO's 1.91 nats was at 2.13 nats (a tie,
  0.000). The retrained runs have one at 1.95 nats, and DPO is ahead of it.

### M9. Training randomness dominates DPO's results but not GRPO's: across seeds DPO's pass@1 varies by about 7× its eval SE, GRPO's by about half

Status: tentative (3 seeds). Last checked: `S/`, 2026-10-25.

Evidence: medians over each arm's checkpoints, from
[`S/tables/frontier_seeds.csv`](math/2026-10-25_seeds17-19_dense-grpo/tables/frontier_seeds.csv).

| Arm | `pass1_seed_se` / `pass1_prompt_se` | Ratio | `kl_k1_seed_se` |
|---|---|---|---|
| DPO | 0.041 / 0.0056 | 7.4× | 4.9 nats |
| DPO+NLL | 0.010 / 0.0057 | 1.7× | 0.64 nats |
| GRPO | 0.003 / 0.0055 | 0.57× | 0.05 nats |
| Continued SFT | 0.0037 / 0.0053 | 0.69× | |

- **The same checkpoint position lands at very different KL for DPO.** DPO β=0.05 ckpt 2 is at 2.13 / 8.14 / 3.05
  nats (`compare_4.txt`). GRPO β=0.01 ckpt 6 is at 2.95 / 2.93 / 2.66 (`compare_5.txt`).
- **Seed 17 was DPO's best seed,** so single-seed DPO conclusions were overconfident.
- **Checkpoints don't always land near a KL target.** In the KL-targeted comparisons
  ([`S/tables/compare_8.txt`](math/2026-10-25_seeds17-19_dense-grpo/tables/compare_8.txt) to `compare_10.txt`), 9 of the 18 DPO-arm picks are more
  than 0.3 nats from their target. Only 1 of the 9 GRPO picks is (seed 18 at ~2 nats, 0.31 off), against 3 with 5
  checkpoints per run.
  - Seed 18's DPO β=0.3 moves from 1.48 to 4.06 nats between two consecutive checkpoints (29 steps).
- **Rerunning GRPO with the same seed moves it about as much as changing the seed** (derived, against
  `K/tables/frontier.csv`).
  - Retraining for PR #29 changed only `save_steps`. It moved the final checkpoints' pass@1 by −0.006 to +0.011:
    in seed 17 by +0.011 / −0.002 / +0.009 / −0.006 (β=0.0 / 0.01 / 0.04 / 0.1), and in seeds 18–19 by at most
    0.006, two runs unchanged.
  - Seed 17 β=0.0's final KL moved from 4.95 to 5.67 nats. The other runs' final KL moved by 0.14 nats or less.
  - GRPO's across-seed SE is 0.003, so its seed spread is at the level of run-to-run nondeterminism.

Caveats:
- This is 3 seeds, and much of DPO's spread comes from seed 18.
- Each seed generates its own DPO pairs, so this mixes data randomness with optimization randomness. Telling them
  apart is an open question in `LOG.md`.

History:
- 2026-10-23, `math/2026-10-23_seeds17-19_full-test`: new.
- 2026-10-23, `S/`: added how far the KL-targeted picks miss their targets.
- 2026-10-25, `S/`: GRPO retrained with 10 checkpoints per run. GRPO's ratio is 0.57×, 1 of its 9 KL picks misses,
  and the same-seed rerun gives a direct measure of its run-to-run noise.

### M10. Continued SFT on the gold answers buys only about +0.03 pass@1 for ~7.6 nats of KL, and dips below π_SFT on the way

Status: supported (3 seeds). Last checked: `S/`, 2026-10-25.

Evidence, from `S/tables/frontier.csv`:

| Measure | Seed 17 / 18 / 19 | Across seeds (95% CI) |
|---|---|---|
| Last checkpoint − π_SFT, pass@1 | +0.036 / +0.037 / +0.020 | +0.031 ± 0.006 [+0.006, +0.056] |
| KL of the last checkpoint | 7.69 / 7.47 / 7.77 nats | |
| Lowest checkpoint − π_SFT | | −0.027 ± 0.003 [−0.040, −0.014] |

- For comparison, GRPO β=0.0 gains +0.203 by 5.7–7.6 nats (M5).
- SFT keeps π_SFT-like diversity, with pass@8 − pass@1 at 0.366–0.375. It costs WikiText perplexity +0.62 (M6).

Caveats: π_SFT already trained 3 epochs on the same gold answers, so this arm mostly over-fits them further.

History:
- 2026-10-23, `S/`: new, replacing M7. Across 3 seeds the small gain is non-zero.
- 2026-10-25, `S/`: unchanged (identical SFT rows); GRPO's comparison number updated.
### M11. Below ~3 nats, at matched KL, GRPO and the DPO arms are within about 0.02 pass@1, and which leads depends on the seed and the GRPO run

Status: tentative (3 seeds, matched pairs in 2). Last checked: `S/`, 2026-10-25.

Evidence: every pair below ~3 nats matched within 0.3 nats in the comparisons fixed in `snapshot.yaml`. Differences
are the DPO arm minus GRPO.

| File | Seed | DPO arm vs GRPO | KL, arm vs GRPO (nats) | DPO arm − GRPO (95% CI) |
|---|---|---|---|---|
| [`compare_1.txt`](math/2026-10-25_seeds17-19_dense-grpo/tables/compare_1.txt) | 17 | DPO β=0.05 ckpt 2 vs β=0.01 ckpt 4 | 2.13 vs 2.24 | −0.015 [−0.025, −0.005] |
| [`compare_2.txt`](math/2026-10-25_seeds17-19_dense-grpo/tables/compare_2.txt) | 17 | DPO β=0.3 ckpt 4 vs β=0.0 ckpt 4 | 2.66 vs 2.62 | +0.015 [+0.005, +0.025] |
| [`compare_2.txt`](math/2026-10-25_seeds17-19_dense-grpo/tables/compare_2.txt) | 17 | DPO β=0.3 ckpt 4 vs β=0.01 ckpt 6 | 2.66 vs 2.95 | −0.011 [−0.021, −0.001] |
| [`compare_8.txt`](math/2026-10-25_seeds17-19_dense-grpo/tables/compare_8.txt) | 17 | DPO β=0.3 vs β=0.01, @kl2 | 1.91 vs 1.95 | +0.011 [+0.001, +0.021] |
| [`compare_8.txt`](math/2026-10-25_seeds17-19_dense-grpo/tables/compare_8.txt) | 17 | DPO+NLL β=0.3 vs β=0.01, @kl2 | 1.79 vs 1.95 | +0.007 [−0.003, +0.017]: a tie |
| [`compare_8.txt`](math/2026-10-25_seeds17-19_dense-grpo/tables/compare_8.txt) | 19 | DPO+NLL β=0.3 vs β=0.01, @kl2 | 2.12 vs 2.02 | −0.020 [−0.029, −0.011] |

- **In seed 17,** DPO leads in two pairs, GRPO in two, and one is a tie. Every gap is 0.015 or less.
- **The same DPO checkpoint** (β=0.3 ckpt 4, 2.66 nats) is behind GRPO β=0.01 but ahead of GRPO β=0.0, both matched.
- **Seed 19's one matched pair** has GRPO ahead by 0.020.
- **Across seeds,** at ~2 nats DPO is −0.015 ± 0.014, CI [−0.077, +0.046], and DPO+NLL −0.018 ± 0.014, CI [−0.077,
  +0.041] ([`compare_8.txt`](math/2026-10-25_seeds17-19_dense-grpo/tables/compare_8.txt)). Both include unmatched pairs.
- **Best pass@1 at ≤ 3 nats** (derived from `S/tables/frontier.csv`):

  | Arm | Seed 17 / 18 / 19 |
  |---|---|
  | GRPO | 0.587 / 0.581 / 0.594 |
  | DPO | 0.576 / 0.502 / 0.547 |
  | DPO+NLL | 0.569 / 0.503 / 0.544 |

Caveats:
- **Seed 18 has no matched pair below 3 nats.** Its DPO arms jump past this range (DPO β=0.3 from 1.48 to 4.06 nats),
  and seed 19 has one pair. So "depends on the seed" rests mostly on seed 17 against one seed-19 pair.
- **The best-at-≤-3-nats numbers pick checkpoints by accuracy.** In seeds 18–19 they favor GRPO by 0.05–0.08, but
  seed 18's DPO arms have no checkpoint between 1.5 and 4 nats to compete with.
- **Gaps this small are within training noise:** rerunning GRPO with the same seed moved its pass@1 by up to 0.011
  (M9).

History:
- 2026-10-25, `S/`: new, replacing M8 (refuted). It is close to M1's original reading ("share one curve"), but only
  seed 17 has enough matched pairs to show it.

---

## Helpful track (UltraFeedback, learned reward)

Not run yet.
