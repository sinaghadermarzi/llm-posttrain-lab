"""Metric-group sentinel keys: a group is complete for a checkpoint once every
one of its keys is in that checkpoint's eval row. Shared by eval/run_eval.py
(what is left to compute) and posttrain_lab.status (what is done); kept
import-free so the status CLI never loads torch or datasets."""

SENTINELS: dict[str, list[str]] = {
    "core": ["kl_k3", "reward_train_rm", "reward_judge"],
    "core_math": ["kl_k3", "pass1"],
    "retention_ppl": ["ppl_wikitext"],
    "retention_harness": ["harness"],
    "diversity": ["distinct_2", "self_bleu"],
    "pref_logps": ["logp_chosen"],
}

# groups computed from this checkpoint's samples on the eval prompts: they
# are stale when the row was evaluated on a different number of prompts
# (e.g. math moved from a 512-prompt subsample to the full 1319-prompt test set)
PROMPT_GROUPS = {"core", "core_math", "diversity"}


def row_prompts(row: dict, samples_per_prompt: int | None) -> int | None:
    """How many eval prompts the row's samples came from: recorded by the core
    groups (eval_prompts), else inferred from num_samples (older rows)."""
    if "eval_prompts" in row:
        return int(row["eval_prompts"])
    if "num_samples" in row and samples_per_prompt:
        return int(row["num_samples"]) // int(samples_per_prompt)
    return None


def stale_groups(row: dict, groups: list[str], num_prompts: int | None, samples_per_prompt: int | None) -> list[str]:
    """Prompt groups the row computed on another prompt count than num_prompts."""
    have = row_prompts(row, samples_per_prompt)
    if num_prompts is None or have is None or have == int(num_prompts):
        return []
    return [g for g in groups if g in PROMPT_GROUPS]


def missing_groups(row: dict, groups: list[str], num_prompts: int | None = None,
                   samples_per_prompt: int | None = None) -> list[str]:
    """Groups still to compute for a checkpoint: sentinels absent, or stale."""
    stale = set(stale_groups(row, groups, num_prompts, samples_per_prompt))
    return [g for g in groups if g in stale or any(k not in row for k in SENTINELS[g])]
