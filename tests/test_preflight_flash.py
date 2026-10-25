"""preflight's flash_attn check knows transformers' kernels fallback.

On the pod flash-attn isn't installed, and preflight reported FAIL, yet every
math run worked with attn_implementation=flash_attention_2: transformers 5
loads kernels-community/flash-attn2 through the `kernels` package instead
(confirmed on the pod: a forward pass ran with it). preflight now tries that
fallback on a tiny model and fails only if it fails too."""

from __future__ import annotations

import sys
import types

import pytest
from omegaconf import OmegaConf

from posttrain_lab import preflight


@pytest.fixture
def no_flash_attn(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn", None)    # import flash_attn -> ImportError


def test_a_working_kernels_fallback_is_ok(monkeypatch, no_flash_attn) -> None:
    monkeypatch.setattr(preflight, "flash_kernel_fallback", lambda: ("kernels-community/flash-attn2", ""))
    [(level, msg)] = preflight.check_flash_attn()
    assert level == "OK" and "kernels-community/flash-attn2 from the kernels library" in msg


def test_no_flash_attn_and_no_fallback_fails_with_the_reason(monkeypatch, no_flash_attn) -> None:
    monkeypatch.setattr(preflight, "flash_kernel_fallback", lambda: (None, "the kernels package isn't installed either"))
    [(level, msg)] = preflight.check_flash_attn()
    assert level == "FAIL" and "the kernels package isn't installed either" in msg and "set it to sdpa" in msg


def test_sdpa_needs_neither(monkeypatch, no_flash_attn) -> None:
    monkeypatch.setattr(preflight, "load_config", lambda *a: OmegaConf.create({"model": {"attn_implementation": "sdpa"}}))
    monkeypatch.setattr(preflight, "flash_kernel_fallback", lambda: pytest.fail("not needed with sdpa"))
    [(level, msg)] = preflight.check_flash_attn()
    assert level == "INFO" and "isn't needed" in msg


def test_the_fallback_needs_the_kernels_package(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "kernels", None)
    assert preflight.flash_kernel_fallback() == (None, "the kernels package isn't installed either")


def test_the_fallback_needs_a_gpu(monkeypatch) -> None:
    import torch

    monkeypatch.setitem(sys.modules, "kernels", types.ModuleType("kernels"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert preflight.flash_kernel_fallback() == (None, "no GPU to try the kernels fallback on")
