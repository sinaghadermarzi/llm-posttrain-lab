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
