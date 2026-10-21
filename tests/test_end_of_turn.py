"""Generation stops at the chat template's end-of-turn token.

The first full math run sampled every eval/pair/GRPO completion to the token
cap: Qwen2.5 base declares eos = <|endoftext|> while its chat template ends
assistant turns with <|im_end|>. These tests rebuild that setup offline with
a tiny Qwen-style tokenizer (and a tiny random Qwen2 model where needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from posttrain_lab.utils import chat

TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
WORDS = ["[UNK]", "q", "a", "user", "assistant", "system", "hi", "there", "x"]
ENDOFTEXT, IM_START, IM_END = 9, 10, 11
HI, THERE = 6, 7


def qwen_like_tokenizer(template: str | None = TEMPLATE):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]")
    fast.add_tokens(["<|endoftext|>", "<|im_start|>", "<|im_end|>"], special_tokens=True)
    fast.eos_token = "<|endoftext|>"   # what Qwen2.5 base declares
    fast.pad_token = "<|endoftext|>"
    if template:
        fast.chat_template = template
    return fast


def save_checkpoint(path: Path, eos_ids, fixed: bool, with_model: bool = False) -> Path:
    tok = qwen_like_tokenizer()
    if fixed:
        chat.use_end_of_turn_eos(tok)
    tok.save_pretrained(path)
    if with_model:
        tiny_model().save_pretrained(path)
    (path / "generation_config.json").write_text(json.dumps({"eos_token_id": eos_ids}))
    return path


def serve_tokenizer(monkeypatch, fixed: bool) -> None:
    """AutoTokenizer.from_pretrained -> the in-memory tokenizer: next to a
    qwen2 config.json, transformers 5 reloads it as Qwen2Tokenizer, which
    re-tokenizes this toy vocab differently."""
    import transformers

    def load(*args, **kwargs):
        tok = qwen_like_tokenizer()
        return chat.use_end_of_turn_eos(tok) if fixed else tok

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load)


def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(vocab_size=len(WORDS) + 3, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                      eos_token_id=ENDOFTEXT, pad_token_id=ENDOFTEXT)
    return Qwen2ForCausalLM(cfg).eval()


# ---------------------------------------------------------------------------


def test_end_of_turn_token_from_the_template() -> None:
    assert chat.end_of_turn_token(qwen_like_tokenizer()) == "<|im_end|>"
    assert chat.end_of_turn_token(qwen_like_tokenizer(template=None)) is None


def test_load_tokenizer_uses_end_of_turn_eos(tmp_path: Path) -> None:
    from posttrain_lab.train.common import load_tokenizer

    qwen_like_tokenizer().save_pretrained(tmp_path)
    tok = load_tokenizer(str(tmp_path))
    assert (tok.eos_token, tok.eos_token_id) == ("<|im_end|>", IM_END)
    assert tok.pad_token == "<|endoftext|>"  # pad never equals eos: eos stays a training target


def test_stop_ids_cover_old_checkpoints(tmp_path: Path) -> None:
    old = save_checkpoint(tmp_path / "old", [ENDOFTEXT], fixed=False)
    from transformers import AutoTokenizer

    assert chat.stop_token_ids(AutoTokenizer.from_pretrained(old), old) == [IM_END, ENDOFTEXT]


def test_align_eos_keeps_the_old_eos_as_a_stop() -> None:
    from posttrain_lab.train.common import align_eos

    model, tok = tiny_model(), chat.use_end_of_turn_eos(qwen_like_tokenizer())
    model.generation_config.eos_token_id = ENDOFTEXT
    align_eos(model, tok)
    assert model.config.eos_token_id == IM_END
    assert model.generation_config.eos_token_id == [IM_END, ENDOFTEXT]


def test_checkpoint_gate(repo, tmp_path: Path) -> None:
    """python -m posttrain_lab.utils.chat fails on a pre-fix checkpoint and
    passes on a fixed one (run_smoke.sh and run_all_*.sh gate on it)."""
    old = save_checkpoint(tmp_path / "old", [ENDOFTEXT], fixed=False)
    new = save_checkpoint(tmp_path / "new", [IM_END, ENDOFTEXT], fixed=True)
    bad = repo.python("-m", "posttrain_lab.utils.chat", str(old), check=False)
    assert bad.returncode == 1
    assert "tokenizer eos is '<|endoftext|>'" in bad.stdout and "lacks '<|im_end|>'" in bad.stdout
    good = repo.python("-m", "posttrain_lab.utils.chat", str(new), str(tmp_path / "not-a-model"), check=False)
    assert good.returncode == 0, good.stdout + good.stderr
    assert "ok" in good.stdout and "FAIL" not in good.stdout


def test_hf_generate_stops_and_records_finished(monkeypatch, tmp_path: Path) -> None:
    """The HF fallback passes every stop id to generate, cuts the text at the
    first one, and marks rows that never stopped."""
    import torch
    import transformers

    from posttrain_lab.eval import generate

    ckpt = save_checkpoint(tmp_path / "ckpt", [ENDOFTEXT], fixed=False, with_model=True)  # a pre-fix checkpoint
    serve_tokenizer(monkeypatch, fixed=False)
    seen = {}

    def fake_generate(self, input_ids=None, attention_mask=None, num_return_sequences=1, eos_token_id=None, **kw):
        seen["eos_token_id"] = eos_token_id
        tails = torch.tensor([[HI, IM_END, THERE, THERE, THERE], [HI, THERE, HI, THERE, HI],
                              [THERE, ENDOFTEXT, ENDOFTEXT, ENDOFTEXT, ENDOFTEXT]])
        return torch.cat([input_ids.repeat_interleave(num_return_sequences, 0), tails.repeat(len(input_ids), 1)], 1)

    monkeypatch.setattr(transformers.GenerationMixin, "generate", fake_generate)
    cfg = OmegaConf.create({"seed": 1, "model": {"dtype": "float32"},
                            "eval": {"samples_per_prompt": 3, "temperature": 1.0, "top_p": 1.0, "max_new_tokens": 5,
                                     "use_vllm": False, "logprob_batch_size": 4}})
    rows = generate.sample_responses(cfg, str(ckpt), [{"prompt_id": "p0", "prompt": "q",
                                                        "messages": [{"role": "user", "content": "q"}]}])
    assert seen["eos_token_id"] == [IM_END, ENDOFTEXT]
    assert [(r["response"], r["finished"], r["stop_token_id"]) for r in rows] == [
        ("hi", True, IM_END), ("hi there hi there hi", False, None), ("there", True, ENDOFTEXT)]


def test_vllm_stop_token() -> None:
    from types import SimpleNamespace as C

    from posttrain_lab.eval.generate import vllm_stop_token

    stops = [IM_END, ENDOFTEXT]
    assert vllm_stop_token(C(finish_reason="stop", stop_reason=ENDOFTEXT, token_ids=[HI, ENDOFTEXT]), stops) == ENDOFTEXT
    assert vllm_stop_token(C(finish_reason="stop", stop_reason=None, token_ids=[HI, IM_END]), stops) == IM_END  # EOS stop
    assert vllm_stop_token(C(finish_reason="length", stop_reason=None, token_ids=[HI, HI]), stops) is None


def test_kl_scores_the_token_the_sample_stopped_on() -> None:
    from posttrain_lab.eval.kl import scored_response

    tok = chat.use_end_of_turn_eos(qwen_like_tokenizer())
    assert scored_response({"response": "hi", "finished": True, "stop_token_id": ENDOFTEXT}, tok, "<|im_end|>") \
        == "hi<|endoftext|>"
    assert scored_response({"response": "hi", "finished": True, "stop_token_id": IM_END}, tok, "<|im_end|>") \
        == "hi<|im_end|>"
    assert scored_response({"response": "hi", "finished": True, "stop_token_id": None}, tok, "<|im_end|>") == "hi"
    assert scored_response({"response": "hi", "finished": True}, tok, "<|im_end|>") == "hi<|im_end|>"  # older rows
    assert scored_response({"response": "hi", "finished": False, "stop_token_id": None}, tok, "<|im_end|>") == "hi"


def test_grpo_rollouts_stop_where_trl_masks() -> None:
    """TRL 1.7 masks GRPO completions at the tokenizer's eos only, so rollouts
    must stop there and nowhere else."""
    from posttrain_lab.train.grpo import rollout_generation_kwargs

    tok = chat.use_end_of_turn_eos(qwen_like_tokenizer())
    assert rollout_generation_kwargs(tok, use_vllm=True) == {"stop_token_ids": [IM_END]}
    assert rollout_generation_kwargs(tok, use_vllm=False) == {}  # TRL passes eos_token_id=tokenizer.eos_token_id
    assert rollout_generation_kwargs(tok, True, {"top_k": 5}) == {"top_k": 5, "stop_token_ids": [IM_END]}


def test_kl_scores_the_end_of_turn_token(monkeypatch, tmp_path: Path) -> None:
    from posttrain_lab.eval.kl import compute_kl

    ckpt = save_checkpoint(tmp_path / "ckpt", [IM_END, ENDOFTEXT], fixed=True, with_model=True)
    serve_tokenizer(monkeypatch, fixed=True)
    cfg = OmegaConf.create({"model": {"dtype": "float32"}, "eval": {"reference": str(ckpt), "logprob_batch_size": 4}})
    samples = [{"prompt": "q", "response": "hi there", "finished": True},
               {"prompt": "q", "response": "hi there", "finished": False}]
    out = compute_kl(cfg, str(ckpt), samples)
    assert out["mean_response_tokens"] == pytest.approx((3 + 2) / 2)  # +1: the stop token of the finished one
    assert out["kl_k3"] == pytest.approx(0.0, abs=1e-6)               # policy == reference


def test_pairs_take_only_finished_correct_answers() -> None:
    from posttrain_lab.gen_pairs import build_pairs

    samples = [
        {"prompt_id": "a", "prompt": "q1", "response": "#### 5", "finished": False},   # right but cut off
        {"prompt_id": "a", "prompt": "q1", "response": "#### 4", "finished": True},
        {"prompt_id": "b", "prompt": "q2", "response": "#### 7", "finished": True},
        {"prompt_id": "b", "prompt": "q2", "response": "#### 1", "finished": False},
    ]
    rows, stats = build_pairs(samples, {"a": "5", "b": "7"}, max_per_prompt=2, seed=0)
    assert [r["chosen"][0]["content"] for r in rows] == ["#### 7"]
    assert stats["prompts_no_finished_correct"] == 1 and stats["samples_unfinished"] == 2


@pytest.mark.parametrize("script", ["run_all_math.sh", "run_all_helpful.sh"])
def test_launchers_gate_on_the_base_sft(script: str) -> None:
    text = (Path(__file__).resolve().parents[1] / "scripts" / script).read_text()
    sft = text.index("_sft_base.sh" if "helpful" in script else "m1_sft_gold.sh")
    gate = text.index('python -m posttrain_lab.utils.chat "$ROOT/runs/sft-base_seed17"')
    assert sft < gate < text.index("=== ", sft)  # right after pi_SFT, before the next stage
    yaml = "configs/helpful/stage1_sft_base.yaml" if "helpful" in script else "configs/math/m1_sft_gold.yaml"
    assert text[gate:].splitlines()[0].endswith(f"--probe {yaml}")  # and it measures, not only reads config



# --- untrained chat-token embeddings (the second full math run) -------------


def crowded_model(tied: bool):
    """Qwen2.5 base's state: <|im_start|>/<|im_end|> sit in a crowd of
    near-identical untrained rows (here: every row past the vocab)."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                      eos_token_id=ENDOFTEXT, pad_token_id=ENDOFTEXT, tie_word_embeddings=tied)
    model = Qwen2ForCausalLM(cfg).eval()
    crowd = torch.randn(16)
    with torch.no_grad():
        for emb in {id(e): e for e in (model.get_input_embeddings(), model.get_output_embeddings())}.values():
            for i in [IM_START, IM_END] + list(range(len(WORDS) + 3, 64)):
                emb.weight[i] = crowd + 0.01 * torch.randn(16)
    return model


def test_template_token_ids() -> None:
    assert chat.template_token_ids(qwen_like_tokenizer()) == [IM_START, IM_END]
    assert chat.template_token_ids(qwen_like_tokenizer(template=None)) == []


@pytest.mark.parametrize("tied", [True, False])
def test_untrained_chat_tokens_are_reinitialized(tied: bool) -> None:
    import torch

    from posttrain_lab.train.common import init_untrained_chat_tokens, untrained_rows

    model, tok = crowded_model(tied), chat.use_end_of_turn_eos(qwen_like_tokenizer())
    out = model.get_output_embeddings().weight
    before = out.detach().clone()
    assert untrained_rows(out, [IM_START, IM_END, ENDOFTEXT, HI]) == [IM_START, IM_END]

    assert init_untrained_chat_tokens(model, tok, ENDOFTEXT) == ["<|im_start|>", "<|im_end|>"]
    for emb in (model.get_input_embeddings().weight, out):
        assert torch.equal(emb[IM_END], emb[ENDOFTEXT])      # end of turn <- end of document
    assert torch.allclose(out[IM_START], before[:IM_START].mean(0))  # the rest <- mean of the trained rows
    keep = [i for i in range(64) if i not in (IM_START, IM_END)]
    assert torch.equal(out[keep], before[keep])                # nothing else moves
    assert untrained_rows(out, [IM_START, IM_END]) == []
    assert init_untrained_chat_tokens(model, tok, ENDOFTEXT) == []  # trained now: a no-op


def test_trained_chat_tokens_are_left_alone() -> None:
    import torch

    from posttrain_lab.train.common import init_untrained_chat_tokens

    model = tiny_model()  # random rows: no crowd (an -Instruct base, or pi_SFT)
    before = model.get_output_embeddings().weight.detach().clone()
    assert init_untrained_chat_tokens(model, chat.use_end_of_turn_eos(qwen_like_tokenizer()), ENDOFTEXT) == []
    assert torch.equal(model.get_output_embeddings().weight, before)


class EndsAfterThere:
    """A stand-in model: puts p_end on <|im_end|> right after the token
    'there' and almost nothing anywhere else, so the probe must read exactly
    the position where the answer ends."""

    def __init__(self, p_end: float):
        self.p_end = p_end

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, ids):
        import math
        from types import SimpleNamespace

        import torch

        logits = torch.zeros(1, ids.shape[1], 64)
        rest = math.log((1 - self.p_end) / 63)
        logits[0, :, :] = rest
        logits[0, :, IM_END] = math.log(1e-6)
        logits[0, ids[0] == THERE, IM_END] = math.log(self.p_end)
        return SimpleNamespace(logits=logits)


def serve_model(monkeypatch, p_end: float) -> None:
    import transformers

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        lambda *a, **k: EndsAfterThere(p_end))


CONVERSATIONS = [[{"role": "user", "content": "q"}, {"role": "assistant", "content": "hi there"}],
                 [{"role": "user", "content": "q"}, {"role": "assistant", "content": "x"},
                  {"role": "user", "content": "q"}, {"role": "assistant", "content": "a hi there"}]]


def test_probe_reads_where_the_answer_ends(monkeypatch, tmp_path: Path) -> None:
    ckpt = save_checkpoint(tmp_path / "ckpt", [IM_END, ENDOFTEXT], fixed=True)
    serve_tokenizer(monkeypatch, fixed=True)
    serve_model(monkeypatch, 0.9)
    out = chat.end_of_turn_probe(ckpt, CONVERSATIONS)
    assert out["token"] == "<|im_end|>" and out["n"] == 2
    assert out["mean"] == pytest.approx(0.9, abs=1e-4) and out["min"] == pytest.approx(0.9, abs=1e-4)


@pytest.mark.parametrize("p_end, flags, code", [(0.95, [], 0), (0.12, [], 1), (0.12, ["--report-only"], 0)])
def test_probe_gate(monkeypatch, tmp_path: Path, capsys, p_end: float, flags: list[str], code: int) -> None:
    ckpt = save_checkpoint(tmp_path / "ckpt", [IM_END, ENDOFTEXT], fixed=True)
    serve_tokenizer(monkeypatch, fixed=True)
    serve_model(monkeypatch, p_end)
    monkeypatch.setattr(chat, "probe_conversations", lambda stage_yaml, n=64: CONVERSATIONS)
    assert chat._cli([str(ckpt), "--probe", "math/m1_sft_gold.yaml", *flags]) == code
    saved = json.loads((ckpt / chat.PROBE_FILE).read_text())
    assert saved["mean"] == pytest.approx(p_end, abs=1e-4) and saved["min_p"] == chat.PROBE_MIN_P
    printed = capsys.readouterr().out
    assert ("run on into invented turns" in printed) == (p_end < chat.PROBE_MIN_P)
    assert "probe:" in printed and ("ok   probe:" in printed) == (p_end >= chat.PROBE_MIN_P)
    assert "any stop token:" in printed


def test_probe_counts_the_other_stop_ids_separately(monkeypatch, tmp_path: Path) -> None:
    """<|endoftext|> also stops eval/pair sampling, not GRPO: the gate reads
    P(<|im_end|>), the run-on estimate P(any stop)."""
    import math

    import transformers

    ckpt = save_checkpoint(tmp_path / "ckpt", [IM_END, ENDOFTEXT], fixed=True)
    serve_tokenizer(monkeypatch, fixed=True)
    base = EndsAfterThere(0.5)

    def call(ids):
        out = base(ids)
        out.logits[0, ids[0] == THERE, ENDOFTEXT] = math.log(0.3)
        return out

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: SimpleModel(call))
    out = chat.end_of_turn_probe(ckpt, CONVERSATIONS)
    assert out["mean_any_stop"] / out["mean"] == pytest.approx((0.5 + 0.3) / 0.5, rel=1e-3)


class SimpleModel:
    def __init__(self, fn):
        self.fn = fn

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, ids):
        return self.fn(ids)


def test_sft_reinitializes_before_aligning_eos() -> None:
    text = (Path(__file__).resolve().parents[1] / "src" / "posttrain_lab" / "train" / "sft.py").read_text()
    assert text.index("init_untrained_chat_tokens(model") < text.index("align_eos(model")
