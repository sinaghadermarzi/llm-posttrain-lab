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

> **Last snapshot:** [`math/2026-10-23_seeds17-19_kl-matched`](math/2026-10-23_seeds17-19_kl-matched/) (below: `S/`).
> It supersedes `math/2026-10-23_seeds17-19_full-test`. The data and the seven position-matched comparisons are
> identical, now with KL-gap flags. It adds the three KL-targeted comparisons, `compare_8`–`10`.
> - Where a History line says `S/` on its own, it was checked against the earlier snapshot.
> It covers seeds 17, 18 and 19 on the full GSM8K test set (1,319 prompts × 8 samples), and all 336 checkpoints.
> - **Per-seed numbers** are read from `S/tables/frontier.csv`, and are listed as seed 17 / 18 / 19.
> - **Across-seed values** are mean ± SE with a 95% CI from t with 2 df (t₂ = 4.30).
> - **"Derived"** marks a number computed from those tables. It is not one of the comparisons fixed in
>   `snapshot.yaml`, so it is descriptive.
> - **π_SFT** scores pass@1 0.426 / 0.427 / 0.437 and pass@8 0.821 / 0.811 / 0.813.

---

## Math track (GSM8K, verifier reward, Qwen2.5-1.5B)

### M1. Up to ~3 nats, DPO, DPO+NLL and GRPO share one accuracy-vs-KL curve

Status: superseded (by M8). Last checked: `S/`, 3 seeds, 2026-10-23.

The claim holds only in seed 17. M8 states what the three seeds show.

Evidence (seed 17):
- [`S/tables/compare_1.txt`](math/2026-10-23_seeds17-19_kl-matched/tables/compare_1.txt): at 2.13 nats, matched within
  0.006 nats, GRPO β=0.01 ckpt 2 beats DPO β=0.05 ckpt 2 by +0.019 ± 0.005, CI [+0.009, +0.029]. So even in seed 17
  this is not a tie.
- [`S/tables/compare_2.txt`](math/2026-10-23_seeds17-19_kl-matched/tables/compare_2.txt): DPO β=0.3 ckpt 4 (0.576 at
  2.66 nats) lies between two GRPO points:
  - GRPO at 2.26 nats is behind it, −0.013 [−0.023, −0.003];
  - GRPO at 3.05 nats is ahead of it, +0.012 [+0.002, +0.022].

  So seed 17 is consistent with a tie at ~2.7 nats.

History:
- 2026-10-22, 512-prompt eval: "DPO β=0.3 ties or edges GRPO at ~2–2.7 nats".
- 2026-10-22, full test set, seed 17: the claim survived.
- 2026-10-23, `S/`, 3 seeds: superseded by M8. In seeds 18 and 19, GRPO is more accurate at equal or lower KL. Even
  in seed 17, the matched-KL gap at 2.13 nats is +0.019 with a CI that excludes 0.

### M2. Above ~3 nats GRPO keeps gaining while DPO and DPO+NLL level off

Status: tentative (3 seeds). Last checked: `S/`, 2026-10-23.

**What the matched pairs establish:** every pair matched within 0.3 nats has GRPO ahead.
- At ~3 nats that's seeds 17 and 19: DPO β=0.3 −0.018 and −0.047, DPO+NLL β=0.3 −0.029 and −0.050, all with CIs
  excluding 0.
- At ~5 nats only seed 17's DPO pair is matched: −0.050.

That falls short of `supported`, which needs matched-KL comparisons across all three seeds. Seed 18's pairs and most
~5-nat pairs aren't matched, so their across-seed CIs mix accuracy with extra KL.

Evidence:
- **KL-targeted comparisons.** These were fixed in `snapshot.yaml` before this snapshot. Each uses every seed's
  checkpoint nearest the target. Differences are the DPO arm minus GRPO, across seeds.
  - **~3 nats,** [`S/tables/compare_9.txt`](math/2026-10-23_seeds17-19_kl-matched/tables/compare_9.txt).
    - GRPO β=0.01 scores 0.587 ± 0.004, on target in every seed (2.93–3.05 nats).
    - DPO β=0.3: −0.045 ± 0.016, CI [−0.113, +0.022]. Per seed −0.018 / −0.072 / −0.047.
    - DPO+NLL β=0.3: −0.049 ± 0.011, CI [−0.096, −0.001]. Per seed −0.029 / −0.068 / −0.050.
    - Every per-seed CI excludes 0. The pairs are matched in seeds 17 and 19.
  - **~5 nats,** [`S/tables/compare_10.txt`](math/2026-10-23_seeds17-19_kl-matched/tables/compare_10.txt).
    - GRPO β=0.0 scores 0.617 ± 0.002.
    - DPO β=0.3: −0.068 ± 0.015, CI [−0.131, −0.005].
    - DPO+NLL β=0.3: −0.073 ± 0.014, CI [−0.134, −0.011].
- **Levelling off,** seed means at the two targets:

  | Arm | ~3 nats | ~5 nats |
  |---|---|---|
  | GRPO | 0.587 | 0.617 |
  | DPO β=0.3 | 0.542 | 0.550 |
  | DPO+NLL β=0.3 | 0.538 | 0.545 |

- **Most accurate points,** [`S/tables/compare_7.txt`](math/2026-10-23_seeds17-19_kl-matched/tables/compare_7.txt). Not matched KL: GRPO β=0.0 last
  scores 0.630 ± 0.007 (4.95–7.65 nats), against DPO β=0.3 ckpt 4's 0.551 ± 0.013.
- **Derived from `S/tables/frontier.csv`.** Best checkpoint at any KL: GRPO − best DPO or DPO+NLL is +0.062 ± 0.024,
  CI [−0.040, +0.164].

Caveats:
- **The ~5-nat matching is imperfect.** GRPO β=0.0 has no checkpoint near 5 nats in seeds 18–19 (6.07 and 6.06), so
  it sits 0.6–1.5 nats further out than the DPO arms.
  - Only seed 17's DPO pair is matched. The across-seed CIs above therefore include the effect of GRPO's extra KL, and
    they can't establish the matched-KL result.
  - Against that: in seeds 18–19, GRPO β=0.0's earlier checkpoint at 3.52–3.57 nats already scores 0.576 / 0.582
    ([`compare_5.txt`](math/2026-10-23_seeds17-19_kl-matched/tables/compare_5.txt)).
  - DPO β=0.3 at 5.47 / 5.39 nats scores 0.520 / 0.564 (`compare_10.txt`). So GRPO is more accurate with less KL.
  - That reading takes numbers from two different files, unpaired, in two seeds. It's suggestive, not a test.
- **At ~3 nats DPO's CI includes 0.** DPO+NLL's only just excludes it. Seed 18's DPO-arm checkpoints there sit 1.13
  and 0.46 nats further out than GRPO's.
- **This is 3 seeds,** so t₂ = 4.30.
- **GRPO saves 5 checkpoints per run and DPO 10.** Denser GRPO checkpoints would allow tighter matching.

History:
- First seen on the 512-prompt eval: GRPO 0.596 at 3.3 nats, 0.616 at 5.1.
- 2026-10-22, seed 17, full test set: GRPO +0.029 and +0.039 over DPO β=0.3 ckpt 4.
- 2026-10-23, `math/2026-10-23_seeds17-19_full-test`, 3 seeds: the gap is larger in seeds 18–19 (up to +0.108). It
  stayed tentative: the position-matched comparisons weren't KL-matched, and their CIs mostly include 0.
- 2026-10-23, `S/` (KL-targeted comparisons): stays tentative, narrowed to the matched pairs.
  - Every matched pair favors GRPO: seeds 17 and 19 at ~3 nats, and seed 17 at ~5 nats.
  - A first draft of this update (PR #28) called it supported, on the strength of the across-seed CIs at ~5 nats.
    Review pointed out that those mix in unmatched pairs.

### M3. Pure DPO with small β collapses through likelihood displacement: at β=0.01 in every seed, at β=0.05 in two of three, at β=0.1 in one

Status: supported (3 seeds) for β=0.01. Last checked: `S/`, 2026-10-23.

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

### M4. Adding an NLL term on the chosen answers prevents the collapse in every seed and at every β, at no cost to DPO's best accuracy

Status: supported (3 seeds). Last checked: `S/`, 2026-10-23.

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
- **It still trails GRPO** (M2, M8).

History:
- The arm was added in PR #18 because of M3.
- 2026-10-23, `S/`, 3 seeds: supported.
  - The KL range was corrected.
  - The old caveat "DPO+NLL's best is slightly below DPO's" no longer holds: across seeds, the two arms' best
    checkpoints tie.

### M5. The accuracy gains come from sharpening, not from solving new problems

Status: supported (3 seeds). Last checked: `S/`, 2026-10-23.

Evidence: GRPO β=0.0's last checkpoint against π_SFT (`S/tables/frontier.csv`; per-seed values in the table's rows).

| Metric | Change from π_SFT (95% CI) | Values |
|---|---|---|
| pass@1 | +0.200 ± 0.006 [+0.172, +0.228] | |
| pass@8 | +0.078 ± 0.003 [+0.065, +0.092] | 0.891–0.895 |
| pass@8 − pass@1 | −0.122 ± 0.003 [−0.136, −0.107] | 0.377–0.395 → 0.254–0.280 |
| distinct-2 | −0.175 ± 0.008 [−0.210, −0.141] | |
| self-BLEU across samples | +0.230 ± 0.012 [+0.180, +0.280] | |
| token NLL | | 0.37–0.39 → 0.18–0.21 |

- In `S/figures/diversity.png`, every arm except continued SFT sharpens as KL grows.

Caveats: pass@8 uses the same 8 samples as pass@1, so this doesn't show whether larger k would still find new
solutions.

History: 2026-10-23, `S/`, 3 seeds: supported. Table values replaced seed 17's ≈ values.

### M6. General ability is largely kept: only collapsed DPO loses clearly, while GRPO and SFT erode a little

Status: supported (3 seeds). Last checked: `S/`, 2026-10-23.

Evidence: last checkpoints against π_SFT (`S/tables/frontier.csv`, `S/figures/retention.png`).

| Run | WikiText-2 ppl | HellaSwag | ARC-Easy |
|---|---|---|---|
| π_SFT | 11.81 / 12.00 / 11.84 | 0.682 in every seed | 0.763 / 0.757 / 0.761 |
| Collapsed DPO β=0.01 | +1.73 ± 0.10 [+1.31, +2.14] | −0.010 ± 0.000 [−0.011, −0.009] | 0.734–0.755 |
| GRPO β=0.0 | +0.32 ± 0.07 [+0.03, +0.61] | −0.004 ± 0.001 [−0.006, −0.002] | |
| Continued SFT | +0.62 ± 0.05 [+0.40, +0.85] | | 0.745–0.758 |

- The collapsed runs' perplexity reaches 13.3–13.8. HellaSwag's drop is about 2 SE at its ≈10k items.
- ARC-Challenge (1,172 items, SE ≈ 0.015) varies 0.453–0.474 across every run: within noise.

Caveats:
- The harness scores are unpaired. Paired per-item differences would be tighter.
- GRPO's HellaSwag change is consistent across seeds but tiny, about 1 SE.

History: 2026-10-23, `S/`, 3 seeds. The claim was "kept, except in the collapsed runs". It now also states GRPO's and
SFT's small but consistent erosion.

### M7. Continued SFT on gold answers moves the model without improving it

Status: superseded (by M10). Last checked: `S/`, 3 seeds, 2026-10-23.

Across 3 seeds, continued SFT does improve, by a small but non-zero amount (M10).

History:
- At 512 prompts it was 0.451 at 8.6 nats, the worst curve.
- 2026-10-22, seed 17, full test set: pass@1 0.40–0.47, around π_SFT's 0.43.
- 2026-10-23, `S/`: superseded by M10.

### M8. At matched KL, GRPO is never behind DPO or DPO+NLL; the only tie is seed 17's DPO at ~2 nats

Status: tentative (3 seeds). Last checked: `S/`, 2026-10-23.

Evidence:
- **~2 nats,** [`S/tables/compare_8.txt`](math/2026-10-23_seeds17-19_kl-matched/tables/compare_8.txt). GRPO β=0.01 scores 0.554 ± 0.003. Across
  seeds:
  - DPO β=0.3: −0.022 ± 0.015, CI [−0.084, +0.041];
  - DPO+NLL β=0.3: −0.024 ± 0.014, CI [−0.085, +0.037].

  | Pair | Seed | KL (arm vs GRPO) | Difference |
  |---|---|---|---|
  | DPO β=0.3, matched | 17 | 1.91 vs 2.13 nats | 0.000, CI [−0.010, +0.010]: a tie |
  | DPO+NLL β=0.3, matched | 19 | 2.12 vs 2.01 nats | −0.018, CI [−0.027, −0.008]: GRPO ahead |
  | DPO β=0.3, not matched | 18 | 1.48 vs 2.42 nats | −0.049 |

  - The seed-19 gap is under 0.02. But its paired CI excludes 0, so under the tie convention's exception it counts as
    a GRPO lead, not a tie.
  - Seed 18 isn't matched: its DPO checkpoints jump from 1.48 to 4.06 nats.
- **~3 nats,** [`S/tables/compare_9.txt`](math/2026-10-23_seeds17-19_kl-matched/tables/compare_9.txt): GRPO is ahead in every seed. See M2.
- **Seed 17's position-matched comparisons:**
  - [`compare_1.txt`](math/2026-10-23_seeds17-19_kl-matched/tables/compare_1.txt): GRPO +0.019, CI [+0.009, +0.029], at 2.13 nats.
  - [`compare_2.txt`](math/2026-10-23_seeds17-19_kl-matched/tables/compare_2.txt): DPO β=0.3 at 2.66 nats lies between GRPO at 2.26 and 3.05 nats.
- **No comparison has a DPO arm ahead of GRPO at matched KL.** Of all 10 comparisons, the only per-seed difference
  in DPO's favor is seed 17 in `compare_2.txt`. There DPO sits 0.39 nats further out than GRPO's 2.26-nat point.
- **Best pass@1 at ≤ 3 nats** (derived from `S/tables/frontier.csv`):

  | Arm | Seed 17 / 18 / 19 |
  |---|---|
  | GRPO | 0.563 / 0.580 / 0.594 |
  | DPO | 0.576 / 0.502 / 0.547 |
  | DPO+NLL | 0.569 / 0.503 / 0.544 |

Caveats:
- **The ~2-nat across-seed CIs include 0.** The one matched tie (seed 17) is a single pair.
- **Checkpoint spacing leaves gaps,** with GRPO's 5 checkpoints per run and seed 18's DPO jumps (M9).
- **The best-at-≤-3-nats numbers pick checkpoints by accuracy.** That favors the arm with more checkpoints (DPO), so
  it works against GRPO here.

History:
- 2026-10-23, `math/2026-10-23_seeds17-19_full-test`: new, replacing M1. Its wording was "Up to ~3 nats GRPO is at
  or above DPO and DPO+NLL in every seed; DPO comes within about 0.02 only in seed 17".
- 2026-10-23, `S/` (KL-targeted comparisons): stated at matched KL. It stays tentative.
  - The only matched tie is still seed 17's DPO at ~2 nats. Seed 19's matched DPO+NLL pair (−0.018) has a paired CI
    that excludes 0, so it's a GRPO lead.
  - A first draft of this update (PR #28) counted that pair as a second tie. Review pointed out that the tie
    convention's paired-CI exception applies.

### M9. Training randomness dominates DPO's results but not GRPO's: across seeds DPO's pass@1 varies by about 7× its eval SE, GRPO's by about half

Status: tentative (3 seeds). Last checked: `S/`, 2026-10-23.

Evidence: medians over each arm's checkpoints, from
[`S/tables/frontier_seeds.csv`](math/2026-10-23_seeds17-19_kl-matched/tables/frontier_seeds.csv).

| Arm | `pass1_seed_se` / `pass1_prompt_se` | Ratio | `kl_k1_seed_se` |
|---|---|---|---|
| DPO | 0.041 / 0.0056 | 7.4× | 4.9 nats |
| DPO+NLL | 0.010 / 0.0057 | 1.7× | 0.64 nats |
| GRPO | 0.003 / 0.0056 | 0.54× | 0.05 nats |
| Continued SFT | 0.0037 / 0.0053 | 0.69× | |

- **The same checkpoint position lands at very different KL for DPO.** DPO β=0.05 ckpt 2 is at 2.13 / 8.14 / 3.05
  nats (`compare_4.txt`). GRPO β=0.01 ckpt 3 is at 3.05 / 2.93 / 2.65.
- **Seed 17 was DPO's best seed,** so single-seed DPO conclusions were overconfident.
- **Checkpoints don't always land near a KL target.** In the KL-targeted comparisons
  ([`S/tables/compare_8.txt`](math/2026-10-23_seeds17-19_kl-matched/tables/compare_8.txt) to `compare_10.txt`), 9 of
  the 18 DPO-arm picks and 3 of the 9 GRPO picks are more than 0.3 nats from their target.
  - Seed 18's DPO β=0.3 moves from 1.48 to 4.06 nats between two consecutive checkpoints (29 steps).

Caveats:
- This is 3 seeds, and much of DPO's spread comes from seed 18.
- Each seed generates its own DPO pairs, so this mixes data randomness with optimization randomness. Telling them
  apart is an open question in `LOG.md`.

History:
- 2026-10-23, `math/2026-10-23_seeds17-19_full-test`: new.
- 2026-10-23, `S/`: added how far the KL-targeted picks miss their targets.

### M10. Continued SFT on the gold answers buys only about +0.03 pass@1 for ~7.6 nats of KL, and dips below π_SFT on the way

Status: supported (3 seeds). Last checked: `S/`, 2026-10-23.

Evidence, from `S/tables/frontier.csv`:

| Measure | Seed 17 / 18 / 19 | Across seeds (95% CI) |
|---|---|---|
| Last checkpoint − π_SFT, pass@1 | +0.036 / +0.037 / +0.020 | +0.031 ± 0.006 [+0.006, +0.056] |
| KL of the last checkpoint | 7.69 / 7.47 / 7.77 nats | |
| Lowest checkpoint − π_SFT | | −0.027 ± 0.003 [−0.040, −0.014] |

- For comparison, GRPO β=0.0 gains +0.200 by 5–7.7 nats (M5).
- SFT keeps π_SFT-like diversity, with pass@8 − pass@1 at 0.366–0.375. It costs WikiText perplexity +0.62 (M6).

Caveats: π_SFT already trained 3 epochs on the same gold answers, so this arm mostly over-fits them further.

History: 2026-10-23, `S/`: new, replacing M7. Across 3 seeds the small gain is non-zero.

---

## Helpful track (UltraFeedback, learned reward)

Not run yet.
