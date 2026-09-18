# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ascend-specific operator optimizations for HunyuanImage3."""

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
_FUSED_ADD_RMS_NORM_ENV = "VLLM_OMNI_HUNYUAN_IMAGE3_FUSED_ADD_RMS_NORM"
_FUSED_SWIGLU_ENV = "DIT_FUSE_SWIGLU"
_FUSED_SWIGLU_LOG_ENV = "DIT_FUSE_LOG"
_DISABLED_VALUES = frozenset({"0", "false", "no", "off", "disabled", "disable"})
_missing_fused_rope_logged = False
_missing_fused_add_rms_norm_logged = False
_fused_swiglu_patched = False


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


def is_hunyuan_image3_fused_add_rms_norm_enabled() -> bool:
    """Return whether HunyuanImage3 fused Add+RMSNorm is enabled.

    The optimization is enabled by default. Set
    ``VLLM_OMNI_HUNYUAN_IMAGE3_FUSED_ADD_RMS_NORM=0`` before starting the
    process to restore the separate residual add and RMSNorm calls.
    """
    return _is_optimization_enabled(_FUSED_ADD_RMS_NORM_ENV)


def is_hunyuan_image3_fused_add_rms_norm_available() -> bool:
    """Return whether the enabled fused Add+RMSNorm API is callable."""
    return is_hunyuan_image3_fused_add_rms_norm_enabled() and callable(getattr(torch_npu, "npu_add_rms_norm", None))


def apply_hunyuan_image3_add_rms_norm_npu(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    fallback: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse the post-attention residual add and RMSNorm when available.

    The second returned tensor is the sum before normalization and remains the
    residual for the following MLP branch. Runtime errors from the fused operator
    are not caught because the operator may have already modified its inputs.
    """
    global _missing_fused_add_rms_norm_logged

    fused_enabled = is_hunyuan_image3_fused_add_rms_norm_enabled()
    fused_add_rms_norm = cast(
        Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None,
        getattr(torch_npu, "npu_add_rms_norm", None) if fused_enabled else None,
    )
    if not callable(fused_add_rms_norm):
        if not _missing_fused_add_rms_norm_logged:
            reason = f"disabled by {_FUSED_ADD_RMS_NORM_ENV}" if not fused_enabled else "unavailable in torch-npu"
            logger.debug(
                "HunyuanImage3 fused Add+RMSNorm is %s; using separate residual add and RMSNorm calls",
                reason,
            )
            _missing_fused_add_rms_norm_logged = True
        added = residual + hidden_states
        return fallback(added), added

    normalized, _, added = fused_add_rms_norm(hidden_states, residual, weight, epsilon)
    return normalized, added


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
    """Convert half-width HunyuanImage3 frequencies to fused BSND layout.

    The model calls this once after sequence-parallel sharding and shares the
    resulting ``[B, S, 1, D]`` tensors across all local decoder layers. A 4-D
    input is accepted so each layer can reuse that prepared pair without copying.
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


# ==================================================================
# SwiGLU 融合 (SiluAndMul -> npu_swiglu)
# ==================================================================
# 背景：vLLM 上游 ``vllm.model_executor.layers.activation.SiluAndMul`` 只实现
# 了 forward_cuda/native/xpu/cpu，没有 forward_npu。``CustomOp.dispatch_forward``
# 在 NPU 平台走 ``is_out_of_tree() -> forward_oot``，而 CustomOp 默认的
# ``forward_oot`` 会 fallback 到 ``forward_native``（Slice + Silu + Mul 三个
# 独立 kernel）。这里通过一次 monkey-patch 把 ``SiluAndMul.forward_oot`` 替
# 换成 ``torch_npu.npu_swiglu``，使 NPU 上所有 ``SiluAndMul`` 实例（包括
# HunyuanImage3 的普通 MLP 与 shared expert）复用同一条融合实现。


def _fused_swiglu_log_enabled() -> bool:
    """SwiGLU patch 生效日志是否开启（默认开）。"""
    return _is_optimization_enabled(_FUSED_SWIGLU_LOG_ENV)


def _log_swiglu(msg: str) -> None:
    if _fused_swiglu_log_enabled():
        print(f"[hunyuan_image3_fusion] {msg}", flush=True)


def is_hunyuan_image3_fused_swiglu_enabled() -> bool:
    """Return whether HunyuanImage3 fused SwiGLU (SiluAndMul->npu_swiglu) is enabled.

    The optimization is enabled by default. Set ``DIT_FUSE_SWIGLU=0`` before
    starting the process to fall back to the native Slice + Silu + Mul path.
    """
    return _is_optimization_enabled(_FUSED_SWIGLU_ENV)


def is_hunyuan_image3_fused_swiglu_available() -> bool:
    """Return whether the enabled fused SwiGLU API is callable."""
    return is_hunyuan_image3_fused_swiglu_enabled() and callable(getattr(torch_npu, "npu_swiglu", None))


def apply_hunyuan_image3_fused_swiglu_patch() -> bool:
    """Install ``torch_npu.npu_swiglu`` as ``SiluAndMul`` forward path.
    Idempotent: safe to call multiple times. Returns ``True`` if the patch has
    been (or was already) applied, ``False`` otherwise.
    """
    global _fused_swiglu_patched
    if _fused_swiglu_patched:
        return True
    if not is_hunyuan_image3_fused_swiglu_enabled():
        return False

    try:
        from vllm.model_executor.layers.activation import SiluAndMul
    except Exception as e:  # pragma: no cover - depends on vLLM install
        _log_swiglu(f"cannot import SiluAndMul: {e!r}")
        return False

    # Someone else (e.g. a future vllm-ascend version) may already provide the
    # NPU forward; do not stack patches on top of it.
    if getattr(SiluAndMul, "_omni_swiglu_patched", False):
        _log_swiglu("SiluAndMul already patched by another module; skip")
        _fused_swiglu_patched = True
        return True

    if not callable(getattr(torch_npu, "npu_swiglu", None)):
        _log_swiglu("torch_npu.npu_swiglu unavailable; skip SwiGLU fusion")
        return False

    warn_holder = {"warned": False}

    def _fused_swiglu_impl(x):
        try:
            return torch_npu.npu_swiglu(x)
        except Exception as e:  # pragma: no cover - runtime fallback
            if not warn_holder["warned"]:
                _log_swiglu(f"npu_swiglu failed, fallback to native: {e!r}")
                warn_holder["warned"] = True
            d = x.shape[-1] // 2
            import torch.nn.functional as F

            return F.silu(x[..., :d]) * x[..., d:]

    def forward_oot(self, x):
        return _fused_swiglu_impl(x)

    def forward(self, x):  # override CustomOp.forward for SiluAndMul
        return _fused_swiglu_impl(x)

    SiluAndMul._orig_forward_oot = SiluAndMul.forward_oot
    SiluAndMul._orig_forward = SiluAndMul.forward
    SiluAndMul.forward_oot = forward_oot
    SiluAndMul.forward = forward  # <-- critical: 让已存在实例也立刻生效

    # 覆盖当前进程里所有已存在的 SiluAndMul 实例的 _forward_method 快照
    rebound = 0
    try:
        import gc

        for obj in gc.get_objects():
            if isinstance(obj, SiluAndMul):
                # 绑定新的 forward_oot 到实例上，替换 __init__ 里存下的旧引用
                try:
                    obj._forward_method = forward_oot.__get__(obj, SiluAndMul)
                    rebound += 1
                except Exception:  # pragma: no cover - defensive
                    pass
    except Exception as e:  # pragma: no cover - defensive
        _log_swiglu(f"failed to rebind existing SiluAndMul instances: {e!r}")

    SiluAndMul._omni_swiglu_patched = True
    _fused_swiglu_patched = True
    _log_swiglu(
        "patched SiluAndMul.forward/forward_oot -> npu_swiglu"
        f" (rebound {rebound} existing instance(s))"
    )
    return True


# Apply the patch at import time so it takes effect exactly like the previous
# ``diffusion/patches/hunyuan_image3_fusion.py`` behavior.
apply_hunyuan_image3_fused_swiglu_patch()


__all__ = [
    "apply_hunyuan_image3_add_rms_norm_npu",
    "apply_hunyuan_image3_fused_swiglu_patch",
    "apply_hunyuan_image3_rope_npu",
    "can_preexpand_hunyuan_image3_rope",
    "is_hunyuan_image3_fused_add_rms_norm_available",
    "is_hunyuan_image3_fused_add_rms_norm_enabled",
    "is_hunyuan_image3_fused_rope_available",
    "is_hunyuan_image3_fused_rope_enabled",
    "is_hunyuan_image3_fused_swiglu_available",
    "is_hunyuan_image3_fused_swiglu_enabled",
    "is_hunyuan_image3_rope_preexpand_enabled",
    "prepare_hunyuan_image3_rope_frequencies_npu",
]
