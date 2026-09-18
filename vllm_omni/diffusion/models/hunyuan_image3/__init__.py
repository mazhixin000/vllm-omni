# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hunyuan Image 3 diffusion model components."""

from . import hunyuan_image3_transformer as _transformer_module
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    HunyuanImage3Model,
    HunyuanImage3Text2ImagePipeline,
)
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    HunyuanImage3Pipeline,
)

# MegaMoE 是唯一需要在模型类定义完成后安装的可选 HunyuanImage3 补丁。
# RoPE、cos/sin、AddRMSNorm 和压缩 KV 均由模型及 NPU 平台代码直接实现。
try:
    from vllm_omni.diffusion.patches.hunyuan_image3_fusion import apply_hunyuan_image3_patches

    apply_hunyuan_image3_patches(_transformer_module)
except Exception as _fusion_exc:  # noqa: BLE001
    import logging as _logging

    _logging.getLogger(__name__).warning("HunyuanImage3 optional patch skipped: %s", _fusion_exc)

__all__ = ["HunyuanImage3Pipeline", "HunyuanImage3Model", "HunyuanImage3Text2ImagePipeline"]
