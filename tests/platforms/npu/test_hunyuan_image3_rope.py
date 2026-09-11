# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _load_npu_rope_module(monkeypatch: pytest.MonkeyPatch, **torch_npu_attrs):
    """Load the NPU helper with a small fake torch_npu module."""
    monkeypatch.setitem(sys.modules, "torch_npu", types.SimpleNamespace(**torch_npu_attrs))
    path = Path(__file__).parents[3] / "vllm_omni" / "platforms" / "npu" / "models" / "hunyuan_image3.py"
    module_name = f"vllm_omni_test_hunyuan_image3_rope_{id(torch_npu_attrs)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def test_hunyuan_image3_npu_rope_fuses_qk_with_bsnd_half_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_HUNYUAN_IMAGE3_FUSED_ROPE", raising=False)
    calls = []

    def npu_apply_rotary_pos_emb(query, key, cos, sin, *, layout, rotary_mode):
        calls.append((query, key, cos, sin, layout, rotary_mode))
        query.copy_(query * cos + _rotate_half(query) * sin)
        key.copy_(key * cos + _rotate_half(key) * sin)
        return query, key

    module = _load_npu_rope_module(
        monkeypatch,
        npu_apply_rotary_pos_emb=npu_apply_rotary_pos_emb,
    )

    torch.manual_seed(7)
    batch, seq_len, q_heads, kv_heads, head_dim = 2, 17, 8, 2, 128
    query = torch.randn(batch, seq_len, q_heads, head_dim, dtype=torch.float32)
    key = torch.randn(batch, seq_len, kv_heads, head_dim, dtype=torch.float32)
    angles = torch.randn(batch, seq_len, head_dim // 2, dtype=torch.float32)
    cos, sin = torch.cos(angles), torch.sin(angles)

    query_ref = query.clone()
    key_ref = key.clone()
    cos_full = torch.cat((cos, cos), dim=-1).unsqueeze(2)
    sin_full = torch.cat((sin, sin), dim=-1).unsqueeze(2)
    expected_query = query_ref * cos_full + _rotate_half(query_ref) * sin_full
    expected_key = key_ref * cos_full + _rotate_half(key_ref) * sin_full

    def unexpected_fallback(*_args):
        raise AssertionError("single-input RoPE fallback must not run when the fused API is available")

    actual_query, actual_key = module.apply_hunyuan_image3_rope_npu(
        unexpected_fallback,
        query,
        key,
        cos,
        sin,
    )

    assert len(calls) == 1
    fused_query, fused_key, fused_cos, fused_sin, layout, rotary_mode = calls[0]
    assert fused_query.shape == (batch, seq_len, q_heads, head_dim)
    assert fused_key.shape == (batch, seq_len, kv_heads, head_dim)
    assert fused_cos.shape == (batch, seq_len, 1, head_dim)
    assert fused_sin.shape == fused_cos.shape
    assert layout == "BSND"
    assert rotary_mode == "half"
    torch.testing.assert_close(fused_cos[..., : head_dim // 2], cos.unsqueeze(2))
    torch.testing.assert_close(fused_cos[..., head_dim // 2 :], cos.unsqueeze(2))
    torch.testing.assert_close(fused_sin[..., : head_dim // 2], sin.unsqueeze(2))
    torch.testing.assert_close(fused_sin[..., head_dim // 2 :], sin.unsqueeze(2))
    torch.testing.assert_close(actual_query, expected_query)
    torch.testing.assert_close(actual_key, expected_key)


def test_hunyuan_image3_npu_rope_falls_back_before_mutation_without_fused_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_npu_rope_module(monkeypatch)
    calls = []

    def single_input_rope(x, cos, sin):
        calls.append((x, cos, sin))
        return x + len(calls)

    query = torch.zeros(1, 3, 2, 8)
    key = torch.full((1, 3, 1, 8), 10.0)
    cos = torch.ones(1, 3, 4)
    sin = torch.zeros_like(cos)

    actual_query, actual_key = module.apply_hunyuan_image3_rope_npu(
        single_input_rope,
        query,
        key,
        cos,
        sin,
    )

    assert len(calls) == 2
    torch.testing.assert_close(actual_query, query + 1)
    torch.testing.assert_close(actual_key, key + 2)


@pytest.mark.parametrize("disabled_value", ["0", "false", "no", "off", "disable"])
def test_hunyuan_image3_fused_rope_switch_restores_original_path(
    monkeypatch: pytest.MonkeyPatch,
    disabled_value: str,
) -> None:
    """The switch must bypass all fused preparation and call Q/K RoPE separately."""
    monkeypatch.setenv("VLLM_OMNI_HUNYUAN_IMAGE3_FUSED_ROPE", disabled_value)

    def unexpected_fused_rope(*_args, **_kwargs):
        raise AssertionError("fused RoPE must not run while disabled")

    module = _load_npu_rope_module(
        monkeypatch,
        npu_apply_rotary_pos_emb=unexpected_fused_rope,
    )
    calls = []

    def single_input_rope(x, cos, sin):
        calls.append((x, cos, sin))
        return x + len(calls)

    def unexpected_frequency_preparation(*_args, **_kwargs):
        raise AssertionError("fused frequency preparation must not run while disabled")

    monkeypatch.setattr(
        module,
        "prepare_hunyuan_image3_rope_frequencies_npu",
        unexpected_frequency_preparation,
    )
    query = torch.zeros(1, 3, 2, 8)
    key = torch.full((1, 3, 1, 8), 10.0)
    cos = torch.ones(1, 3, 4)
    sin = torch.zeros_like(cos)

    actual_query, actual_key = module.apply_hunyuan_image3_rope_npu(
        single_input_rope,
        query,
        key,
        cos,
        sin,
    )

    assert not module.is_hunyuan_image3_fused_rope_available()
    assert len(calls) == 2
    assert calls[0][0] is query
    assert calls[0][1] is cos
    assert calls[0][2] is sin
    assert calls[1][0] is key
    assert calls[1][1] is cos
    assert calls[1][2] is sin
    torch.testing.assert_close(actual_query, query + 1)
    torch.testing.assert_close(actual_key, key + 2)


def test_hunyuan_image3_npu_rope_reuses_prepared_full_width_frequencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-layer calls must not concatenate frequencies prepared by the model."""
    monkeypatch.delenv("VLLM_OMNI_HUNYUAN_IMAGE3_FUSED_ROPE", raising=False)
    calls = []

    def npu_apply_rotary_pos_emb(query, key, cos, sin, *, layout, rotary_mode):
        calls.append((cos, sin, layout, rotary_mode))
        return query, key

    module = _load_npu_rope_module(
        monkeypatch,
        npu_apply_rotary_pos_emb=npu_apply_rotary_pos_emb,
    )
    query = torch.randn(1, 17, 8, 128)
    key = torch.randn(1, 17, 2, 128)
    cos = torch.randn(1, 17, 1, 128)
    sin = torch.randn_like(cos)

    def unexpected_cat(*_args, **_kwargs):
        raise AssertionError("prepared full-width frequencies must not be concatenated again")

    monkeypatch.setattr(torch, "cat", unexpected_cat)
    module.apply_hunyuan_image3_rope_npu(lambda *_args: None, query, key, cos, sin)

    fused_cos, fused_sin, layout, rotary_mode = calls[0]
    assert fused_cos is cos
    assert fused_sin is sin
    assert layout == "BSND"
    assert rotary_mode == "half"


def test_prepared_frequencies_do_not_reuse_another_request_cache() -> None:
    """Full-width frequencies always belong to the current model forward."""
    from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
        HunYuanRotary2DEmbedder,
    )

    embedder = HunYuanRotary2DEmbedder.__new__(HunYuanRotary2DEmbedder)
    embedder.custom_pos_emb = (
        torch.full((1, 3, 1, 8), -1.0),
        torch.full((1, 3, 1, 8), -1.0),
    )
    current_cos = torch.randn(1, 3, 1, 8)
    current_sin = torch.randn_like(current_cos)

    actual_cos, actual_sin = embedder._prepare_cos_sin(
        (current_cos, current_sin),
        first_step=False,
        device=current_cos.device,
    )

    assert actual_cos is current_cos
    assert actual_sin is current_sin
