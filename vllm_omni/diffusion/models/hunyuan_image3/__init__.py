# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hunyuan Image 3 diffusion model components."""

from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    HunyuanImage3Model,
    HunyuanImage3Text2ImagePipeline,
)
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    HunyuanImage3Pipeline,
)

# 融合算子 monkey-patch：hunyuan_image3_transformer 已完整加载后立即打补丁。
# 关闭方式：设置 DIT_FUSE_ROPE_QK=0 / DIT_FUSE_ADD_RMSNORM=0 / DIT_FUSE_SWIGLU=0。
# 加载失败不影响主模型（内部 try/except 已兜底）。
try:
    from vllm_omni.diffusion.patches import hunyuan_image3_fusion as _hunyuan_image3_fusion  # noqa: F401
except Exception as _fusion_exc:  # noqa: BLE001
    import logging as _logging

    _logging.getLogger(__name__).warning("hunyuan_image3 fusion patch skipped: %s", _fusion_exc)

__all__ = ["HunyuanImage3Pipeline", "HunyuanImage3Model", "HunyuanImage3Text2ImagePipeline"]
