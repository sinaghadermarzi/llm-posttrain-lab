---
name: update-findings
description: Re-check every claim in evidence/FINDINGS.md against a new evidence snapshot - update numbers, CIs, status, seed count, evidence links and history. Use right after a snapshot is taken, or when the user asks whether a finding still holds or what changed with more seeds or data.
---

# Update the findings against a snapshot

`evidence/FINDINGS.md` holds the claims (M1, M2, …). After a new snapshot, every claim gets re-checked against it,
so no claim cites numbers the snapshot doesn't hold. Conventions are in `AGENTS.md` (analysis and evidence
sections).

## 1. Read the snapshot, not the live results

Use only files inside `evidence/<task>/<snapshot>/`:
- `tables/frontier.csv` and `tables/frontier_seeds.csv` (`n_seeds`, the means, `<metric>_seed_se`,
  `<metric>_prompt_se`);
- `tables/compare_*.txt`;
- the figures;
- `MANIFEST.json`, for the seeds and `eval.num_prompts`.

If a number you need isn't in the snapshot, the claim can't cite it. Mark it "pending" and say what export it needs.

## 2. For each claim, in order

1. **Find its evidence in the snapshot.** Replace ≈ values read off figures with table values wherever a table has
   them, and point each number at a file: `math/<snapshot>/tables/compare_3.txt`.
2. **Recompute what it rests on.**
   - With several seeds, use the `--pool` lines' across-seed CI (`seeds` row, t with n − 1 df), and
     `frontier_seeds.csv`'s `_seed_se`.
   - Compare `pass1_seed_se` with `pass1_prompt_se`. If the seed SE is much larger, training randomness dominates,
     and single-seed conclusions were overconfident: say so.
3. **Set the status:**
   - `supported`: the claim holds in the across-seed CI (≥ 3 seeds).
   - `tentative`: one seed, or an across-seed CI too wide to tell (2 seeds: t₁ = 12.7).
   - `refuted`: the CI goes the other way, or the effect vanishes at matched KL.
   - `superseded`: rewritten as a sharper claim. Keep the old entry, and point at the new ID.
4. **Update "Last checked"** (the snapshot and date) and the seed count.
5. **Add a History line whenever the claim's wording, numbers or status change.** Say what changed and why: more
   seeds, more prompts, a bug fix. Never silently overwrite an earlier number.

## 3. New claims

A pattern visible in the snapshot that no claim covers gets a new ID (next free number; IDs are never reused), as
`tentative`. Keep it to one sentence that a figure or table can show. A claim the data can't test isn't a claim yet:
put it in `LOG.md` as an open question instead.

## 4. Report to the user

List each claim whose status or numbers changed, with old → new and the reason. List new claims. Flag anything that
now looks like a bug rather than a finding, such as a seed far off the others, or `n_seeds` lower than expected.
Commit FINDINGS.md with the snapshot.

## Don'ts

- Don't pick checkpoints to make a claim look better. Use the `snapshot.yaml` comparisons. A different comparison
  needs a `snapshot.yaml` change and a new snapshot.
- Don't cite `$RESULTS_DIR` paths or numbers pasted from a terminal as final evidence; they're "pending" until a
  snapshot holds them.
- Don't drop caveats when a claim becomes `supported`. Best-checkpoint selection and the KL mismatch still apply.
