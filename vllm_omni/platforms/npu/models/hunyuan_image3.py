# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ascend-specific RoPE implementation for HunyuanImage3."""

from __future__ import annotations

import os
from collections.abc import Callable

import torch
import torch_npu
from vllm.logger import init_logger

logger = init_logger(__name__)

_FUSED_ROPE_ENV = "VLLM_OMNI_HUNYUAN_IMAGE3_FUSED_ROPE"
_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})
_DISABLED_VALUES = frozenset({"0", "false", "no", "off", "disabled", "disable"})
_missing_fused_rope_logged = False


def is_hunyuan_image3_fused_rope_enabled() -> bool:
    """Return whether HunyuanImage3 Q/K fused RoPE is enabled.

    The optimization remains enabled by default. Set
    ``VLLM_OMNI_HUNYUAN_IMAGE3_FUSED_ROPE=0`` before starting the process to
    restore the original two single-input RoPE calls.
    """
    value = os.environ.get(_FUSED_ROPE_ENV)
    if value is None or value == "":
        return True
    value = value.lower()
    if value in _DISABLED_VALUES:
        return False
    return value in _ENABLED_VALUES


def _prepare_half_rope_frequencies(
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert Hunyuan half-width frequencies to ApplyRotaryPosEmb BSND inputs.

    Hunyuan stores one cosine/sine value for every NeoX pair, so the source
    tensors end in ``head_dim // 2``. ApplyRotaryPosEmb consumes full-width
    frequencies with a singleton head dimension. Duplicating the complete
    half-width vector, rather than repeating each element, preserves the pairs
    ``(x[i], x[i + head_dim // 2])`` used by ``rotary_mode="half"``.
    """
    if cos.shape != sin.shape:
        raise ValueError(f"cos and sin must have identical shapes, got {cos.shape} and {sin.shape}")
    if cos.ndim == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    if cos.ndim != 3:
        raise ValueError(f"Hunyuan RoPE frequencies must be [B, S, D/2] or [S, D/2], got {cos.shape}")
    if cos.shape[1] != seq_len:
        raise ValueError(f"RoPE sequence length mismatch: expected {seq_len}, got {cos.shape[1]}")
    if cos.shape[-1] * 2 != head_dim:
        raise ValueError(f"RoPE frequency width must be half of head_dim={head_dim}, got {cos.shape[-1]}")

    freq_batch = cos.shape[0]
    if freq_batch == 1 and batch_size != 1:
        cos = cos.expand(batch_size, -1, -1)
        sin = sin.expand(batch_size, -1, -1)
    elif freq_batch != batch_size:
        raise ValueError(f"RoPE batch size must be 1 or {batch_size}, got {freq_batch}")

    cos = torch.cat((cos, cos), dim=-1).unsqueeze(2).to(device=device, dtype=dtype).contiguous()
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(2).to(device=device, dtype=dtype).contiguous()
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
    fused_rope = (
        getattr(torch_npu, "npu_apply_rotary_pos_emb", None)
        if fused_rope_enabled
        else None
    )
    if fused_rope is None:
        if not _missing_fused_rope_logged:
            reason = (
                f"disabled by {_FUSED_ROPE_ENV}"
                if not fused_rope_enabled
                else "unavailable in torch-npu"
            )
            logger.debug(
                "HunyuanImage3 fused Q/K RoPE is %s; using the existing "
                "two-call single-input RoPE path",
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
    # query = query.contiguous()
    # key = key.contiguous()
    cos, sin = _prepare_half_rope_frequencies(
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


__all__ = ["apply_hunyuan_image3_rope_npu"]
