# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import defaultdict

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_tokenizer import TokenizerWrapper
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    HunyuanImage3Config,
    HunyuanImage3Text2ImagePipeline,
    ImageInfo,
)
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    HunyuanImage3Pipeline,
    _hunyuan_cfg_factor,
    _image_info_from_payload,
    _image_info_to_payload,
    _meanflow_timestep_r,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


class _ScalarEmbedder(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.stack((values, values + 1), dim=-1)


def test_checkpoint_variant_config_flags_are_preserved() -> None:
    config = HunyuanImage3Config(cfg_distilled=True, use_meanflow=True)

    assert config.cfg_distilled is True
    assert config.use_meanflow is True


def test_cfg_distilled_always_uses_one_model_row() -> None:
    assert _hunyuan_cfg_factor(cfg_distilled=True, guidance_scale=1.0) == 1
    assert _hunyuan_cfg_factor(cfg_distilled=True, guidance_scale=7.5) == 1
    assert _hunyuan_cfg_factor(cfg_distilled=False, guidance_scale=1.0) == 1
    assert _hunyuan_cfg_factor(cfg_distilled=False, guidance_scale=7.5) == 2


def test_meanflow_uses_next_timestep_and_zero_for_last_step() -> None:
    timesteps = torch.tensor([1000.0, 500.0, 125.0])

    torch.testing.assert_close(_meanflow_timestep_r(timesteps, 0), torch.tensor(500.0))
    torch.testing.assert_close(_meanflow_timestep_r(timesteps, 1), torch.tensor(125.0))
    torch.testing.assert_close(_meanflow_timestep_r(timesteps, 2), torch.tensor(0.0))

    with pytest.raises(IndexError, match="outside"):
        _meanflow_timestep_r(timesteps, 3)


def test_image_info_payload_preserves_variant_tokens() -> None:
    image_info = ImageInfo(
        image_type="gen_image",
        image_width=1024,
        image_height=768,
        token_width=32,
        token_height=24,
        base_size=1024,
        ratio_index=1,
        add_guidance_token=True,
        add_timestep_r_token=True,
    )

    restored = _image_info_from_payload(_image_info_to_payload(image_info))

    assert restored.add_guidance_token is True
    assert restored.add_timestep_r_token is True
    assert restored.meta_info["add_guidance_token"] is True
    assert restored.meta_info["add_timestep_r_token"] is True


def test_tokenizer_emits_dynamic_tokens_in_reference_order() -> None:
    wrapper = object.__new__(TokenizerWrapper)
    wrapper.special_token_map = {
        "<img_size_1024>": 10,
        "<img_ratio_1>": 11,
        "<timestep>": 12,
        "<guidance>": 13,
        "<timestep_r>": 14,
    }
    token_seq: list[int] = []
    positions: defaultdict[str, list[int]] = defaultdict(list)

    token_count = wrapper._add_image_meta_info_token(
        token_seq,
        token_count=0,
        extra_token_pos=positions,
        add_timestep_token=True,
        add_guidance_token=True,
        add_timestep_r_token=True,
        add_image_shape_token=True,
        base_size=1024,
        ratio_idx=1,
        image_type="gen_image",
    )

    assert token_seq == [10, 11, 12, 13, 14]
    assert positions["gen_timestep"] == [2]
    assert positions["guidance"] == [3]
    assert positions["timestep_r"] == [4]
    assert token_count == 5


def test_scalar_tokens_expand_per_batch_row() -> None:
    hidden = torch.zeros(2, 4, 2)
    values = torch.tensor([3.0, 7.0])
    scatter_index = torch.tensor([[1], [2]])

    output = HunyuanImage3Pipeline._instantiate_scalar_tokens(
        hidden,
        values,
        scatter_index,
        _ScalarEmbedder(),
        "guidance",
    )

    torch.testing.assert_close(output[0, 1], torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(output[1, 2], torch.tensor([7.0, 8.0]))
    assert torch.count_nonzero(output).item() == 4


def test_ar_kv_prefix_truncation_shifts_all_dynamic_token_indexes() -> None:
    pipeline = object.__new__(HunyuanImage3Text2ImagePipeline)
    input_ids = torch.arange(10).reshape(1, 10)
    model_kwargs = {
        "query_lens": [10],
        "attention_mask": torch.ones(1, 1, 10, 10, dtype=torch.bool),
        "position_ids": torch.arange(10).reshape(1, 10),
        "image_mask": torch.zeros(1, 10, dtype=torch.bool),
        "gen_timestep_scatter_index": torch.tensor([[6]]),
        "guidance_scatter_index": torch.tensor([[7]]),
        "timestep_r_scatter_index": torch.tensor([[8]]),
    }

    truncated = pipeline._truncate_reused_prefix(input_ids, model_kwargs, positive_reuse_len=5)

    assert truncated.tolist() == [[5, 6, 7, 8, 9]]
    assert model_kwargs["gen_timestep_scatter_index"].tolist() == [[1]]
    assert model_kwargs["guidance_scatter_index"].tolist() == [[2]]
    assert model_kwargs["timestep_r_scatter_index"].tolist() == [[3]]
