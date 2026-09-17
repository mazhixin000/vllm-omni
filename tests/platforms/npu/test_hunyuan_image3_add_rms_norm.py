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


def _load_npu_module(monkeypatch: pytest.MonkeyPatch, **torch_npu_attrs):
    """Load the HunyuanImage3 NPU helper with a small fake torch_npu module."""
    monkeypatch.setitem(sys.modules, "torch_npu", types.SimpleNamespace(**torch_npu_attrs))
    path = Path(__file__).parents[3] / "vllm_omni" / "platforms" / "npu" / "models" / "hunyuan_image3.py"
    module_name = f"vllm_omni_test_hunyuan_image3_add_rms_norm_{id(torch_npu_attrs)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    return (x_float * torch.rsqrt(variance + epsilon) * weight.float()).to(x.dtype)


@pytest.mark.parametrize("shape", [(3, 16), (2, 5, 16)])
def test_hunyuan_image3_npu_add_rms_norm_uses_fused_operator(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, ...],
) -> None:
    monkeypatch.delenv("VLLM_OMNI_HUNYUAN_IMAGE3_FUSED_ADD_RMS_NORM", raising=False)
    calls = []

    def npu_add_rms_norm(hidden_states, residual, weight, epsilon):
        calls.append((hidden_states, residual, weight, epsilon))
        added = hidden_states + residual
        normalized = _rms_norm(added, weight, epsilon)
        rstd = torch.rsqrt(added.float().pow(2).mean(dim=-1, keepdim=True) + epsilon)
        return normalized, rstd, added

    module = _load_npu_module(monkeypatch, npu_add_rms_norm=npu_add_rms_norm)
    torch.manual_seed(7)
    hidden_states = torch.randn(shape)
    residual = torch.randn(shape)
    weight = torch.randn(shape[-1])
    epsilon = 1e-6

    def unexpected_fallback(_x):
        raise AssertionError("RMSNorm fallback must not run when the fused API is available")

    normalized, added = module.apply_hunyuan_image3_add_rms_norm_npu(
        hidden_states,
        residual,
        weight,
        epsilon,
        unexpected_fallback,
    )

    assert module.is_hunyuan_image3_fused_add_rms_norm_enabled()
    assert module.is_hunyuan_image3_fused_add_rms_norm_available()
    assert calls == [(hidden_states, residual, weight, epsilon)]
    expected_added = hidden_states + residual
    torch.testing.assert_close(added, expected_added)
    torch.testing.assert_close(normalized, _rms_norm(expected_added, weight, epsilon))


def test_hunyuan_image3_npu_add_rms_norm_falls_back_when_api_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_npu_module(monkeypatch)
    hidden_states = torch.randn(2, 4, 8)
    residual = torch.randn_like(hidden_states)
    fallback_inputs = []

    def fallback(x):
        fallback_inputs.append(x)
        return x * 2

    normalized, added = module.apply_hunyuan_image3_add_rms_norm_npu(
        hidden_states,
        residual,
        torch.ones(8),
        1e-6,
        fallback,
    )

    expected_added = hidden_states + residual
    assert not module.is_hunyuan_image3_fused_add_rms_norm_available()
    assert len(fallback_inputs) == 1
    torch.testing.assert_close(fallback_inputs[0], expected_added)
    torch.testing.assert_close(added, expected_added)
    torch.testing.assert_close(normalized, expected_added * 2)


@pytest.mark.parametrize("disabled_value", ["0", "false", "no", "off", "disabled", "disable"])
def test_hunyuan_image3_fused_add_rms_norm_switch_restores_original_path(
    monkeypatch: pytest.MonkeyPatch,
    disabled_value: str,
) -> None:
    monkeypatch.setenv("VLLM_OMNI_HUNYUAN_IMAGE3_FUSED_ADD_RMS_NORM", disabled_value)

    def unexpected_fused_operator(*_args):
        raise AssertionError("fused Add+RMSNorm must not run while disabled")

    module = _load_npu_module(monkeypatch, npu_add_rms_norm=unexpected_fused_operator)
    hidden_states = torch.randn(2, 8)
    residual = torch.randn_like(hidden_states)
    fallback_inputs = []

    def fallback(x):
        fallback_inputs.append(x)
        return x

    normalized, added = module.apply_hunyuan_image3_add_rms_norm_npu(
        hidden_states,
        residual,
        torch.ones(8),
        1e-6,
        fallback,
    )

    expected_added = hidden_states + residual
    assert not module.is_hunyuan_image3_fused_add_rms_norm_enabled()
    assert not module.is_hunyuan_image3_fused_add_rms_norm_available()
    assert len(fallback_inputs) == 1
    torch.testing.assert_close(added, expected_added)
    torch.testing.assert_close(normalized, expected_added)


def test_hunyuan_image3_npu_add_rms_norm_does_not_hide_runtime_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_fused_operator(*_args):
        raise RuntimeError("operator failed")

    module = _load_npu_module(monkeypatch, npu_add_rms_norm=broken_fused_operator)

    with pytest.raises(RuntimeError, match="operator failed"):
        module.apply_hunyuan_image3_add_rms_norm_npu(
            torch.ones(2, 8),
            torch.ones(2, 8),
            torch.ones(8),
            1e-6,
            lambda x: x,
        )
