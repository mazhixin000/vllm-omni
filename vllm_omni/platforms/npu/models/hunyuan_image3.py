# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ascend-specific RoPE implementation for HunyuanImage3."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import cast

import torch
import torch_npu
from vllm.logger import init_logger

logger = init_logger(__name__)

_FUSED_ROPE_ENV = "VLLM_OMNI_HUNYUAN_IMAGE3_FUSED_ROPE"
_ROPE_PREEXPAND_ENV = "VLLM_OMNI_HUNYUAN_IMAGE3_ROPE_PREEXPAND"
_DISABLED_VALUES = frozenset({"0", "false", "no", "off", "disabled", "disable"})
_missing_fused_rope_logged = False


def _is_optimization_enabled(env_name: str) -> bool:
    """Return ``False`` only when an optimization is explicitly disabled."""
    value = os.environ.get(env_name, "").strip().lower()
    return value not in _DISABLED_VALUES


def is_hunyuan_image3_fused_rope_enabled() -> bool:
    """Return whether HunyuanImage3 Q/K fused RoPE is enabled.

    The optimization is enabled by default. Set
    ``VLLM_OMNI_HUNYUAN_IMAGE3_FUSED_ROPE=0`` before starting the process to
    restore the original two single-input RoPE calls.
    """
    return _is_optimization_enabled(_FUSED_ROPE_ENV)


def is_hunyuan_image3_fused_rope_available() -> bool:
    """Return whether the enabled HunyuanImage3 fused RoPE API is callable."""
    return is_hunyuan_image3_fused_rope_enabled() and callable(getattr(torch_npu, "npu_apply_rotary_pos_emb", None))


def is_hunyuan_image3_rope_preexpand_enabled() -> bool:
    """Return whether model-level cos/sin expansion is enabled.

    Pre-expansion is enabled by default and has an independent kill switch so
    it can be disabled without losing the existing fused Q/K RoPE optimization.
    """
    return _is_optimization_enabled(_ROPE_PREEXPAND_ENV)


def can_preexpand_hunyuan_image3_rope() -> bool:
    """Return whether full-width frequencies may safely enter the fused path."""
    return is_hunyuan_image3_rope_preexpand_enabled() and is_hunyuan_image3_fused_rope_available()


def prepare_hunyuan_image3_rope_frequencies_npu(
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare HunyuanImage3 frequencies for fused BSND RoPE.

    Hunyuan produces half-width ``[B, S, D/2]`` frequencies, while the Ascend
    fused operator consumes ``[B, S, 1, D]``. A full-width input is accepted as
    an idempotent fast path, allowing one model-level expansion to be shared by
    every local decoder layer.
    """
    if cos.shape != sin.shape:
        raise ValueError(f"cos and sin must have identical shapes, got {cos.shape} and {sin.shape}")
    if cos.dtype != sin.dtype or cos.device != sin.device:
        raise ValueError("cos and sin must have the same dtype and device")

    if cos.ndim == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    if cos.ndim == 3:
        if cos.shape[-1] * 2 != head_dim:
            raise ValueError(f"RoPE frequency width must be half of head_dim={head_dim}, got {cos.shape[-1]}")
        cos = torch.cat((cos, cos), dim=-1).unsqueeze(2)
        sin = torch.cat((sin, sin), dim=-1).unsqueeze(2)
    elif cos.ndim == 4:
        if cos.shape[2] != 1 or cos.shape[-1] != head_dim:
            raise ValueError(f"Full-width RoPE frequencies must have shape [B, S, 1, {head_dim}], got {cos.shape}")
    else:
        raise ValueError(f"Hunyuan RoPE frequencies must be [S, D/2], [B, S, D/2], or [B, S, 1, D], got {cos.shape}")

    if cos.shape[1] != seq_len:
        raise ValueError(f"RoPE sequence length mismatch: expected {seq_len}, got {cos.shape[1]}")

    freq_batch = cos.shape[0]
    if freq_batch == 1 and batch_size != 1:
        cos = cos.expand(batch_size, -1, -1, -1)
        sin = sin.expand(batch_size, -1, -1, -1)
    elif freq_batch != batch_size:
        raise ValueError(f"RoPE batch size must be 1 or {batch_size}, got {freq_batch}")

    if cos.device != device or cos.dtype != dtype:
        cos = cos.to(device=device, dtype=dtype)
        sin = sin.to(device=device, dtype=dtype)
    if not cos.is_contiguous():
        cos = cos.contiguous()
    if not sin.is_contiguous():
        sin = sin.contiguous()
    return cos, sin


def apply_hunyuan_image3_rope_npu(
    rope: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply HunyuanImage3 NeoX RoPE to Q/K with the Ascend fused operator.

    ``npu_apply_rotary_pos_emb`` updates query and key in place and returns both
    tensors. Capability is checked before any input is modified; older torch-npu
    packages therefore retain the established pair of single-input RoPE calls.
    Runtime errors from the fused operator are intentionally not caught because
    retrying with already-mutated inputs could apply RoPE twice.
    """
    global _missing_fused_rope_logged

    fused_rope_enabled = is_hunyuan_image3_fused_rope_enabled()
    fused_rope = cast(
        Callable[..., tuple[torch.Tensor, torch.Tensor]] | None,
        getattr(torch_npu, "npu_apply_rotary_pos_emb", None) if fused_rope_enabled else None,
    )
    if not callable(fused_rope):
        if not _missing_fused_rope_logged:
            reason = f"disabled by {_FUSED_ROPE_ENV}" if not fused_rope_enabled else "unavailable in torch-npu"
            logger.debug(
                "HunyuanImage3 fused Q/K RoPE is %s; using the existing two-call single-input RoPE path",
                reason,
            )
            _missing_fused_rope_logged = True
        return rope(query, cos, sin), rope(key, cos, sin)

    if query.ndim != 4 or key.ndim != 4:
        raise ValueError(f"query and key must use BSND layout, got {query.shape} and {key.shape}")
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError(f"query and key BSND shapes are incompatible: {query.shape} and {key.shape}")
    if query.dtype != key.dtype or query.device != key.device:
        raise ValueError("query and key must have the same dtype and device")

    batch_size, seq_len, _, head_dim = query.shape
    # Q/K normally become contiguous when BF16 activations are converted to
    # FP32. Keep a guarded copy for FP32 or wrapper inputs that remain strided
    # split views; the in-place NPU operator must not receive aliased views of
    # the original packed QKV tensor.
    if not query.is_contiguous():
        query = query.contiguous()
    if not key.is_contiguous():
        key = key.contiguous()

    cos, sin = prepare_hunyuan_image3_rope_frequencies_npu(
        cos,
        sin,
        batch_size=batch_size,
        seq_len=seq_len,
        head_dim=head_dim,
        dtype=query.dtype,
        device=query.device,
    )

    return fused_rope(
        query,
        key,
        cos,
        sin,
        layout="BSND",
        rotary_mode="half",
    )


__all__ = [
    "apply_hunyuan_image3_rope_npu",
    "can_preexpand_hunyuan_image3_rope",
    "is_hunyuan_image3_fused_rope_available",
    "is_hunyuan_image3_fused_rope_enabled",
    "is_hunyuan_image3_rope_preexpand_enabled",
    "prepare_hunyuan_image3_rope_frequencies_npu",
]
