# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
import io

import pytest
import torch

from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    HunyuanImage3Config,
    HunyuanImage3ImageProcessor,
    ImageInfo,
    JointImageInfo,
)
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    HunyuanImage3Pipeline,
    _decode_external_conditions,
    _image_info_from_payload,
    _image_info_to_payload,
    _should_return_postprocess_meta,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def _encode_tensor(tensor: torch.Tensor) -> str:
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_assistant_prompt_fallback_and_ar_priority() -> None:
    pipeline = object.__new__(HunyuanImage3Pipeline)
    prompts = [
        {"prompt": "first", "extra": {"ar_generated_text": "ar cot"}},
        {"prompt": "second", "extra": {}},
    ]

    _, cot_text, _, _, _ = pipeline._extract_prompt_inputs(
        prompts,
        {"assistant_prompt": ["ignored", "external cot"]},
        request_id="req-1",
        allow_cond_image=False,
    )

    assert cot_text == ["ar cot", "external cot"]


def test_external_conditions_support_aggregated_latents_and_timestep_broadcast() -> None:
    batched_latents = torch.randn(2, 32, 8, 8)
    timestep = torch.tensor([0.75])

    latents, timesteps = _decode_external_conditions(
        [_encode_tensor(batched_latents)],
        [_encode_tensor(timestep)],
        expected_num=2,
    )

    assert len(latents) == 2
    torch.testing.assert_close(latents[0], batched_latents[0])
    torch.testing.assert_close(latents[1], batched_latents[1])
    torch.testing.assert_close(timesteps, torch.tensor([0.75, 0.75]))


def test_external_condition_fields_must_be_paired_and_require_images() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        HunyuanImage3Pipeline._extract_external_condition_inputs(
            {"cond_vae_images": [["latent"]]},
            [[object()]],  # type: ignore[list-item]
        )

    with pytest.raises(ValueError, match="require real condition image"):
        HunyuanImage3Pipeline._extract_external_condition_inputs(
            {"cond_vae_images": [["latent"]], "cond_timesteps": ["timestep"]},
            None,
        )


def test_postprocess_meta_is_returned_by_default_and_can_be_disabled() -> None:
    assert _should_return_postprocess_meta({}) is True
    assert _should_return_postprocess_meta({"return_postprocess_meta": None}) is True
    assert _should_return_postprocess_meta({"return_postprocess_meta": True}) is True
    assert _should_return_postprocess_meta({"return_postprocess_meta": False}) is False


def test_postprocess_meta_preserves_condition_original_aspect_ratio() -> None:
    processor = HunyuanImage3ImageProcessor(HunyuanImage3Config(image_base_size=1024))
    ratio_index = processor.reso_group.get_base_size_and_ratio_index(720, 1280)[1]
    output_res = processor.reso_group[ratio_index]
    generated = ImageInfo(
        image_type="gen_image",
        image_width=output_res.width,
        image_height=output_res.height,
        token_width=1,
        token_height=1,
        base_size=1024,
        ratio_index=ratio_index,
    )
    cond_vae = ImageInfo(
        image_type="vae",
        image_width=output_res.width,
        image_height=output_res.height,
        ori_image_width=571,
        ori_image_height=1000,
        token_width=1,
        token_height=1,
        base_size=1024,
        ratio_index=ratio_index,
    )
    cond_vit = ImageInfo(
        image_type="siglip2",
        image_width=1,
        image_height=1,
        token_width=1,
        token_height=1,
    )
    cond = JointImageInfo(cond_vae, cond_vit)

    [aligned] = processor.compute_postprocess_meta(
        [generated],
        [[cond]],
        infer_align_image_size=True,
    )
    [bucket] = processor.compute_postprocess_meta(
        [generated],
        [[cond]],
        infer_align_image_size=False,
    )

    assert aligned == {"w": 774, "h": 1355}
    assert bucket == {"w": output_res.width, "h": output_res.height}


def test_image_info_payload_preserves_original_size() -> None:
    original = ImageInfo(
        image_type="vae",
        image_width=720,
        image_height=1280,
        ori_image_width=571,
        ori_image_height=1000,
        token_width=1,
        token_height=1,
        base_size=1024,
        ratio_index=0,
    )

    restored = _image_info_from_payload(_image_info_to_payload(original))

    assert restored.ori_image_width == 571
    assert restored.ori_image_height == 1000


def test_external_conditions_reject_count_mismatch() -> None:
    with pytest.raises(ValueError, match="resolves to 1 tensors"):
        _decode_external_conditions(
            [_encode_tensor(torch.randn(32, 8, 8))],
            [_encode_tensor(torch.tensor([0.1, 0.2]))],
            expected_num=2,
        )

    with pytest.raises(ValueError, match="resolves to 3 tensors"):
        _decode_external_conditions(
            [
                _encode_tensor(torch.randn(32, 8, 8)),
                _encode_tensor(torch.randn(32, 8, 8)),
                _encode_tensor(torch.randn(32, 8, 8)),
            ],
            [_encode_tensor(torch.tensor([0.1, 0.2]))],
            expected_num=2,
        )
