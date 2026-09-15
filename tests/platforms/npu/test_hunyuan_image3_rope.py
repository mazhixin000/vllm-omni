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

    assert not module.is_hunyuan_image3_fused_rope_enabled()
    assert len(calls) == 2
    assert calls[0][0] is query
    assert calls[0][1] is cos
    assert calls[0][2] is sin
    assert calls[1][0] is key
    assert calls[1][1] is cos
    assert calls[1][2] is sin
    torch.testing.assert_close(actual_query, query + 1)
    torch.testing.assert_close(actual_key, key + 2)


def test_hunyuan_image3_rope_optimizations_are_enabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both model-local RoPE optimizations default to enabled."""
    monkeypatch.delenv("VLLM_OMNI_HUNYUAN_IMAGE3_FUSED_ROPE", raising=False)
    monkeypatch.delenv("VLLM_OMNI_HUNYUAN_IMAGE3_ROPE_PREEXPAND", raising=False)
    module = _load_npu_rope_module(
        monkeypatch,
        npu_apply_rotary_pos_emb=lambda *args, **kwargs: (args[0], args[1]),
    )

    assert module.is_hunyuan_image3_fused_rope_enabled()
    assert module.is_hunyuan_image3_rope_preexpand_enabled()
    assert module.can_preexpand_hunyuan_image3_rope()


@pytest.mark.parametrize("disabled_value", ["0", "false", "no", "off", "disable"])
def test_hunyuan_image3_rope_preexpand_switch_keeps_fused_qk_enabled(
    monkeypatch: pytest.MonkeyPatch,
    disabled_value: str,
) -> None:
    """The new kill switch must not disable the established fused Q/K op."""
    monkeypatch.delenv("VLLM_OMNI_HUNYUAN_IMAGE3_FUSED_ROPE", raising=False)
    monkeypatch.setenv("VLLM_OMNI_HUNYUAN_IMAGE3_ROPE_PREEXPAND", disabled_value)
    fused_calls = []

    def fused_rope(query, key, cos, sin, **kwargs):
        fused_calls.append((query, key, cos, sin, kwargs))
        return query, key

    module = _load_npu_rope_module(
        monkeypatch,
        npu_apply_rotary_pos_emb=fused_rope,
    )
    query = torch.zeros(1, 3, 2, 8)
    key = torch.zeros(1, 3, 1, 8)
    cos = torch.ones(1, 3, 4)
    sin = torch.zeros_like(cos)

    module.apply_hunyuan_image3_rope_npu(lambda *_args: None, query, key, cos, sin)

    assert module.is_hunyuan_image3_fused_rope_enabled()
    assert not module.is_hunyuan_image3_rope_preexpand_enabled()
    assert not module.can_preexpand_hunyuan_image3_rope()
    assert len(fused_calls) == 1


@pytest.mark.parametrize("batch_size", [1, 2])
def test_prepare_hunyuan_image3_rope_frequencies_accepts_half_and_full_width(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
) -> None:
    """Half-width input expands once and the resulting full-width input is idempotent."""
    module = _load_npu_rope_module(monkeypatch)
    seq_len, head_dim = 5, 8
    angles = torch.randn(1, seq_len, head_dim // 2)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    cos_full, sin_full = module.prepare_hunyuan_image3_rope_frequencies_npu(
        cos,
        sin,
        batch_size=batch_size,
        seq_len=seq_len,
        head_dim=head_dim,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert cos_full.shape == (batch_size, seq_len, 1, head_dim)
    assert sin_full.shape == cos_full.shape
    torch.testing.assert_close(cos_full[..., : head_dim // 2], cos.expand(batch_size, -1, -1).unsqueeze(2))
    torch.testing.assert_close(sin_full[..., : head_dim // 2], sin.expand(batch_size, -1, -1).unsqueeze(2))

    def unexpected_cat(*_args, **_kwargs):
        raise AssertionError("full-width frequencies must not be expanded again")

    monkeypatch.setattr(module.torch, "cat", unexpected_cat)
    reused_cos, reused_sin = module.prepare_hunyuan_image3_rope_frequencies_npu(
        cos_full,
        sin_full,
        batch_size=batch_size,
        seq_len=seq_len,
        head_dim=head_dim,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert reused_cos is cos_full
    assert reused_sin is sin_full


def test_preexpanded_frequencies_reach_fused_rope_without_cat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every layer may reuse the model-level full-width pair without materialization."""
    fused_calls = []

    def fused_rope(query, key, cos, sin, **kwargs):
        fused_calls.append((cos, sin, kwargs))
        return query, key

    module = _load_npu_rope_module(
        monkeypatch,
        npu_apply_rotary_pos_emb=fused_rope,
    )
    query = torch.zeros(1, 3, 2, 8)
    key = torch.zeros(1, 3, 1, 8)
    cos_full = torch.ones(1, 3, 1, 8)
    sin_full = torch.zeros_like(cos_full)

    def unexpected_cat(*_args, **_kwargs):
        raise AssertionError("a decoder layer must not expand pre-expanded frequencies")

    monkeypatch.setattr(module.torch, "cat", unexpected_cat)
    module.apply_hunyuan_image3_rope_npu(
        lambda *_args: None,
        query,
        key,
        cos_full,
        sin_full,
    )

    assert len(fused_calls) == 1
    assert fused_calls[0][0] is cos_full
    assert fused_calls[0][1] is sin_full


def test_fused_rope_materializes_noncontiguous_packed_qk_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-place fused op receives independent contiguous Q/K tensors."""
    fused_inputs = []

    def fused_rope(query, key, cos, sin, **kwargs):
        fused_inputs.append((query, key, cos, sin, kwargs))
        return query, key

    module = _load_npu_rope_module(
        monkeypatch,
        npu_apply_rotary_pos_emb=fused_rope,
    )
    packed_qkv = torch.randn(1, 3, 32)
    query, key, _ = packed_qkv.split([16, 8, 8], dim=-1)
    query = query.reshape(1, 3, 2, 8)
    key = key.reshape(1, 3, 1, 8)
    assert not query.is_contiguous()
    assert not key.is_contiguous()
    cos = torch.ones(1, 3, 4)
    sin = torch.zeros_like(cos)

    actual_query, actual_key = module.apply_hunyuan_image3_rope_npu(
        lambda *_args: None,
        query,
        key,
        cos,
        sin,
    )

    fused_query, fused_key, _, _, _ = fused_inputs[0]
    assert fused_query.is_contiguous()
    assert fused_key.is_contiguous()
    assert fused_query.untyped_storage().data_ptr() != fused_key.untyped_storage().data_ptr()
    assert actual_query is fused_query
    assert actual_key is fused_key


@pytest.mark.parametrize(
    ("cos_shape", "sin_shape", "error"),
    [
        ((1, 3, 4), (1, 4, 4), "identical shapes"),
        ((1, 3, 5), (1, 3, 5), "half of head_dim"),
        ((1, 3, 2, 8), (1, 3, 2, 8), "Full-width"),
        ((1, 4, 4), (1, 4, 4), "sequence length mismatch"),
    ],
)
def test_prepare_hunyuan_image3_rope_frequencies_rejects_invalid_shapes(
    monkeypatch: pytest.MonkeyPatch,
    cos_shape: tuple[int, ...],
    sin_shape: tuple[int, ...],
    error: str,
) -> None:
    module = _load_npu_rope_module(monkeypatch)
    cos = torch.ones(cos_shape)
    sin = torch.zeros(sin_shape)

    with pytest.raises(ValueError, match=error):
        module.prepare_hunyuan_image3_rope_frequencies_npu(
            cos,
            sin,
            batch_size=1,
            seq_len=3,
            head_dim=8,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
