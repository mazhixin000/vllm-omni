# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.hunyuan_image3 import siglip2

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_hunyuan_image3_vit_fusion_is_enabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_HUNYUAN_IMAGE3_FUSED_VIT", raising=False)

    def fused_attention(*args, **kwargs):
        del args, kwargs

    def fused_add_layer_norm(*args, **kwargs):
        del args, kwargs

    monkeypatch.setattr(
        siglip2,
        "torch_npu",
        SimpleNamespace(
            npu_fused_infer_attention_score=fused_attention,
            npu_add_layer_norm=fused_add_layer_norm,
        ),
    )

    assert siglip2.is_hunyuan_image3_fused_vit_enabled()
    assert siglip2._get_hunyuan_image3_vit_npu_op("npu_fused_infer_attention_score") is fused_attention
    assert siglip2._get_hunyuan_image3_vit_npu_op("npu_add_layer_norm") is fused_add_layer_norm


@pytest.mark.parametrize("disabled_value", ["0", "false", "no", "off", "disabled", "disable"])
def test_hunyuan_image3_vit_fusion_switch_disables_both_operators(
    monkeypatch: pytest.MonkeyPatch,
    disabled_value: str,
) -> None:
    monkeypatch.setenv("VLLM_OMNI_HUNYUAN_IMAGE3_FUSED_VIT", disabled_value)
    monkeypatch.setattr(
        siglip2,
        "torch_npu",
        SimpleNamespace(
            npu_fused_infer_attention_score=lambda *args, **kwargs: None,
            npu_add_layer_norm=lambda *args, **kwargs: None,
        ),
    )

    assert not siglip2.is_hunyuan_image3_fused_vit_enabled()
    assert siglip2._get_hunyuan_image3_vit_npu_op("npu_fused_infer_attention_score") is None
    assert siglip2._get_hunyuan_image3_vit_npu_op("npu_add_layer_norm") is None


def test_hunyuan_image3_vit_fusion_falls_back_when_operator_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_OMNI_HUNYUAN_IMAGE3_FUSED_VIT", raising=False)
    monkeypatch.setattr(siglip2, "torch_npu", SimpleNamespace())

    assert siglip2.is_hunyuan_image3_fused_vit_enabled()
    assert siglip2._get_hunyuan_image3_vit_npu_op("npu_fused_infer_attention_score") is None
    assert siglip2._get_hunyuan_image3_vit_npu_op("npu_add_layer_norm") is None
