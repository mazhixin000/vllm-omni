# HunyuanImage3 COT、外部条件 VAE 与独立 VAE Decode 升级说明

## 1. 背景与范围

本次升级在 `internal/v0.26.0` 分支恢复旧版 `0.21.0rc2` 中以下生产能力：

1. standalone DiT 使用外部 `assistant_prompt` 作为 COT；
2. 图片编辑请求传入预编码的 `cond_vae_images` 和 `cond_timesteps`；
3. DiT 以 latent-only 模式独立部署，VAE decode 在外部服务完成；
4. 保持 Chat Completions 和 Images API 的旧 latent 响应协议。

本次不包含 MLU platform、worker 或算子迁移。

## 2. 总体方案

### 2.1 COT

HunyuanImage3 的 COT 来源优先级为：

1. AR stage 生成的 `extra.ar_generated_text`；
2. 客户端传入的 `assistant_prompt`；
3. 无 COT。

因此完整 AR + DiT 部署不会被请求中的 `assistant_prompt` 覆盖；`assistant_prompt` 主要用于 standalone DiT。

### 2.2 外部条件 VAE

请求仍需同时传入真实条件图片。服务端继续使用真实图片完成：

- ViT encode；
- image mask；
- RoPE；
- conditional section 构造。

仅跳过条件图片的本地 VAE encode，改用客户端传入的 latent 和 timestep。

字段协议保持为：

- `cond_vae_images`: `list[list[str]]`，兼容单 batch 的 `list[str]`；
- `cond_timesteps`: `list[str]`；
- 每个字符串为 `base64(torch.save(tensor))`。

两个字段必须同时提供，并且只允许 standalone DiT 请求使用。反序列化使用 `torch.load(..., weights_only=True)`，单个序列化 tensor 最大为 64 MiB。

### 2.3 独立 VAE decode

DiT stage 配置 `output_type: latent` 后：

- 自动将 stage 的 `final_output_type` 映射为 `latents`；
- 不构造 Hunyuan VAE；
- 不加载 `vae.*` 权重；
- 不注册 VAE profiler hook；
- warmup 不注入需要本地 VAE encode 的 dummy image；
- 返回未经 inverse scaling、shift 和 VAE decode 的 model-space latent。

外部 VAE decoder 必须根据响应中的 `postprocess_meta` 完成最终尺寸后处理。`postprocess_meta` 使用模型 resolution group 对齐后的真实宽高，而不是未经对齐的原始请求尺寸。

## 3. 使用方式

### 3.1 启动 latent-only DiT

使用部署文件：

```bash
vllm serve <MODEL_PATH> --omni --deploy-config vllm_omni/deploy/hunyuan_image3_dit_latent.yaml
```

关键配置为：

```yaml
pipeline: hunyuan_image3_dit
stages:
  - stage_id: 0
    output_type: latent
```

latent-only 模式仅支持单输出，即 `n=1` 或 `num_outputs_per_prompt=1`。

### 3.2 Chat Completions：外部 COT

```json
{
  "model": "<MODEL_NAME>",
  "messages": [
    {
      "role": "user",
      "content": "生成一张图片"
    }
  ],
  "modalities": ["image"],
  "stream": false,
  "extra_body": {
    "assistant_prompt": "<think>...外部 COT...</think>"
  }
}
```

也兼容将 `assistant_prompt` 放在请求顶层或 `extra_args` 中。latent 输出不支持 `stream=true`。

### 3.3 Images generation：外部 COT

```json
{
  "model": "<MODEL_NAME>",
  "prompt": "生成一张图片",
  "assistant_prompt": "<think>...外部 COT...</think>",
  "n": 1,
  "response_format": "b64_json"
}
```

### 3.4 Images edit：外部条件 VAE

`/v1/images/edits` 使用 multipart form：

- `image`: 真实条件图片；
- `assistant_prompt`: 可选外部 COT；
- `cond_vae_images`: JSON 编码的 `list[list[str]]`；
- `cond_timesteps`: JSON 编码的 `list[str]`。

示例序列化函数：

```python
import base64
import io
import json
import torch


def encode_tensor(tensor: torch.Tensor) -> str:
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


form = {
    "assistant_prompt": "<think>...</think>",
    "cond_vae_images": json.dumps([[encode_tensor(cond_latent)]]),
    "cond_timesteps": json.dumps([encode_tensor(cond_timestep)]),
}
```

Chat JSON 请求直接传数组，不需要外层 `json.dumps`。

## 4. Latent 响应协议

### 4.1 Chat Completions

Chat 响应保持旧协议：

```json
{
  "image": "data:application/x-torch-tensor;base64,<BASE64>",
  "postprocess_meta": {
    "w": 1024,
    "h": 768
  }
}
```

解码方式：

```python
import base64
import io
import torch

prefix = "data:application/x-torch-tensor;base64,"
raw = base64.b64decode(response["image"][len(prefix):])
latent = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
```

### 4.2 Images API

Images generation/edit 响应保持旧协议：

```json
{
  "data": [
    {
      "b64_json": "<BASE64_TORCH_TENSOR>"
    }
  ],
  "output_format": "pt",
  "postprocess_meta": {
    "w": 1024,
    "h": 768
  }
}
```

虽然沿用 OpenAI Images 的 `b64_json` 字段，但当 `output_format` 为 `pt` 时，其中是 `torch.save` tensor，而不是 PNG/JPEG/WebP。latent-only 模式只支持 `response_format=b64_json`，不支持 `response_format=file`。

## 5. 独立 VAE Decoder 合约

DiT 返回的是 model-space latent。独立 VAE 服务负责执行：

1. 按 Hunyuan VAE 配置完成 inverse scaling/shift；
2. 补齐 VAE 所需维度；
3. 调用相同 checkpoint 的 `vae.decode()`；
4. 执行 image processor 后处理；
5. 根据 `postprocess_meta.w/h` 输出最终图片尺寸。

不要在 DiT 服务返回前执行以上步骤，否则会破坏解耦部署语义。

## 6. 校验与限制

- `assistant_prompt` 必须是字符串或字符串数组；数组长度必须与 diffusion batch 一致；
- `cond_vae_images` 和 `cond_timesteps` 必须同时提供；
- 外部条件 VAE 必须同时提供真实条件图片；
- 外部条件 VAE 只支持 standalone DiT，不支持 AR -> DiT 跨 stage 透传；
- 当前外部条件 VAE 一次只支持一个 diffusion request，但支持该请求包含多张条件图；
- latent/timestep 数量必须与真实条件图数量匹配；
- 单 timestep 支持广播到多张条件图；
- latent-only 图片编辑必须提供外部条件 VAE，因为该部署不会创建本地 VAE；
- latent-only 输出只支持非流式、`n=1`。

## 7. 修改点

### 请求协议与参数注册

- `vllm_omni/model_extras/hunyuan_image3.py`
  - 注册 `assistant_prompt`、`cond_vae_images`、`cond_timesteps`。
- `vllm_omni/model_extras/registry.py`
  - 为 HunyuanImage3 pipeline/architecture 注册扩展请求字段。
- `vllm_omni/entrypoints/openai/protocol/images.py`
  - Images generation 显式声明 `assistant_prompt`；
  - Images response 增加 `postprocess_meta`；
  - 文档化 `pt` tensor payload。
- `vllm_omni/entrypoints/openai/protocol/chat_completion.py`
  - Chat response 恢复顶层 `image` 和 `postprocess_meta`。

### OpenAI 服务层

- `vllm_omni/entrypoints/openai/api_server.py`
  - Images generation/edit 透传外部 COT；
  - edit multipart 解析并前置校验条件 VAE JSON；
  - latent 输出编码为 `base64(torch.save(tensor))`；
  - 返回 `output_format=pt` 和 `postprocess_meta`；
  - 前置拒绝 latent 的流式、`n>1` 和 file 响应。
- `vllm_omni/entrypoints/openai/serving_chat.py`
  - Chat 请求前置校验新增字段；
  - 通用多阶段和纯 Diffusion 两条路径均支持 latent data URI；
  - latent 请求前置拒绝 `stream=true`。

### Hunyuan pipeline

- `vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py`
  - AR COT 优先、`assistant_prompt` fallback；
  - 安全反序列化外部 tensor；
  - 支持聚合 latent、数量校验和 timestep 广播；
  - 外部 latent 仅替代 VAE encode，保留 ViT/mask/RoPE；
  - latent-only 时跳过 VAE 构造、权重和 profiler；
  - step/request 两条执行路径均返回 canonical latent envelope；
  - 返回模型实际对齐后的 `postprocess_meta`。
- `vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py`
  - legacy/request generation 在 `output_type=latent` 时直接返回 model-space latent。

### 编排、输出与 warmup

- `vllm_omni/config/stage_config.py`
  - `output_type=latent(s)` 自动映射为 `final_output_type=latents`。
- `vllm_omni/diffusion/output_formatter.py`
  - 将最终 latent 写入 `OmniRequestOutput.latents`，不复用 `trajectory_latents`。
- `vllm_omni/outputs/__init__.py`
  - 支持最终 latent 类型并将其识别为 diffusion 输出。
- `vllm_omni/outputs/output_metadata.py`
  - 将 `latents` 纳入最终输出模态。
- `vllm_omni/diffusion/diffusion_engine.py`
  - latent-only warmup 不注入 dummy image。
- `vllm_omni/deploy/hunyuan_image3_dit_latent.yaml`
  - 新增 CUDA/NPU/XPU 的 standalone latent-only DiT 配置；不包含 MLU。

## 8. 测试

新增或扩展以下 CPU 测试：

- `tests/diffusion/models/hunyuan_image3/test_external_cond_vae_inputs.py`
  - AR COT 与 `assistant_prompt` 优先级；
  - 聚合 latent 展开；
  - timestep 广播；
  - 字段成对性、真实图片依赖和数量错误。
- `tests/diffusion/test_diffusion_output_formatter.py`
  - 最终 latent 写入 `latents`；
  - 不污染 `trajectory_latents`；
  - 保留 `postprocess_meta`。

当前开发环境已通过 `compileall` 和 `git diff --check`。该环境未安装 `pytest` 和 `ruff`，因此未在本机执行 pytest/ruff；需要在项目标准测试环境继续执行：

```bash
python -m pytest -q tests/diffusion/models/hunyuan_image3/test_external_cond_vae_inputs.py
python -m pytest -q tests/diffusion/test_diffusion_output_formatter.py
python -m ruff check vllm_omni tests/diffusion/models/hunyuan_image3/test_external_cond_vae_inputs.py
```

还需在可加载 HunyuanImage3 checkpoint 的环境执行四条 E2E：

1. standalone DiT + `assistant_prompt`；
2. 图片编辑 + 外部 `cond_vae_images/cond_timesteps`；
3. Images API latent -> 独立 VAE decode；
4. Chat API latent -> 独立 VAE decode。
