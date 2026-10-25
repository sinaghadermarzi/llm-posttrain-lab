"""PPO checkpoints save under DeepSpeed (make helpful-probe, 2026-10-25).

TRL 1.7's PPOTrainer.save_model points self.deepspeed at the bare policy and
calls transformers' Trainer.save_model, which in transformers 5.13 reads
self.deepspeed.config as DeepSpeed's config dict. The probe's first PPO
checkpoint crashed with "'Qwen2Config' object has no attribute 'get'", about
8 h into what the real run would have been. Smoke runs PPO without DeepSpeed,
so it never reached that branch. DeepSpeedSafePPOTrainer saves the policy
itself. DeepSpeed is not installed here; the trainer is built without
__init__ around a tiny real policy and value model."""

from __future__ import annotations

import types
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM, Qwen2ForSequenceClassification

from posttrain_lab.train import ppo


def _tiny(cls, **kw):
    torch.manual_seed(0)
    return cls(Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                           num_attention_heads=2, num_key_value_heads=1, **kw))


def _trainer(cls, tmp_path: Path, zero_stage: int = 2, should_save: bool = True):
    from trl.experimental.ppo.ppo_trainer import PolicyAndValueWrapper

    trainer = object.__new__(cls)
    wrapper = PolicyAndValueWrapper(_tiny(Qwen2ForCausalLM), _tiny(Qwen2ForSequenceClassification, num_labels=1))
    trainer.model = wrapper                      # what accelerator.prepare wraps (the DeepSpeed engine's module)
    trainer.deepspeed = trainer.model_wrapped = types.SimpleNamespace(
        save_checkpoint=lambda output_dir, exclude_frozen_parameters=False: None)
    trainer.is_deepspeed_enabled, trainer.is_fsdp_enabled = True, False
    trainer.accelerator = types.SimpleNamespace(
        state=types.SimpleNamespace(deepspeed_plugin=types.SimpleNamespace(zero_stage=zero_stage)),
        unwrap_model=lambda m, **kw: m)
    trainer.args = types.SimpleNamespace(should_save=should_save, output_dir=str(tmp_path), push_to_hub=False)
    trainer.processing_class = trainer.data_collator = None
    return trainer, wrapper


def test_trl_and_transformers_as_pinned_still_crash_here(tmp_path: Path) -> None:
    """The bug itself: if a TRL or transformers upgrade fixes it, this fails and the subclass can go."""
    trainer, _ = _trainer(ppo.PPOTrainer, tmp_path)
    with pytest.raises(AttributeError, match="has no attribute 'get'"):
        ppo.PPOTrainer.save_model(trainer, str(tmp_path / "checkpoint-2"), _internal_call=True)


def test_the_policy_alone_is_saved_and_reloads(tmp_path: Path) -> None:
    trainer, wrapper = _trainer(ppo.DeepSpeedSafePPOTrainer, tmp_path)
    out = tmp_path / "checkpoint-2"
    trainer.save_model(str(out), _internal_call=True)
    assert trainer.model is wrapper                                   # restored for training to go on
    reloaded = AutoModelForCausalLM.from_pretrained(out)
    assert type(reloaded).__name__ == "Qwen2ForCausalLM"              # the policy, not the value model
    for name, param in wrapper.policy.state_dict().items():
        assert torch.equal(reloaded.state_dict()[name], param), name
    assert (out / "training_args.bin").exists()


def test_other_ranks_write_nothing(tmp_path: Path) -> None:
    trainer, _ = _trainer(ppo.DeepSpeedSafePPOTrainer, tmp_path, should_save=False)
    trainer.save_model(str(tmp_path / "checkpoint-2"), _internal_call=True)
    assert not (tmp_path / "checkpoint-2").exists()


def test_zero3_is_refused_not_saved_partial(tmp_path: Path) -> None:
    trainer, _ = _trainer(ppo.DeepSpeedSafePPOTrainer, tmp_path, zero_stage=3)
    with pytest.raises(NotImplementedError, match="ZeRO-3"):
        trainer.save_model(str(tmp_path / "checkpoint-2"))


def test_without_deepspeed_trl_saves_as_before(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(ppo.PPOTrainer, "save_model", lambda self, output_dir=None, _internal_call=False:
                        calls.append((output_dir, _internal_call)))
    trainer, _ = _trainer(ppo.DeepSpeedSafePPOTrainer, tmp_path)
    trainer.is_deepspeed_enabled = False
    trainer.save_model("out", _internal_call=True)
    assert calls == [("out", True)]


def test_the_arm_uses_it() -> None:
    src = Path(ppo.__file__).read_text()
    assert "trainer = DeepSpeedSafePPOTrainer(" in src and "trainer = PPOTrainer(" not in src
