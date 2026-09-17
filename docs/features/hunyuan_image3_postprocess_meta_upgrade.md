# HunyuanImage3 `postprocess_meta` P0/P1 升级说明

## 1. 问题

HunyuanImage3 standalone DiT 在 latent-only 模式中已经在 pipeline 内部生成了 `postprocess_meta`，但 OpenAI Chat/Images 响应中该字段可能为空。

根因有两层：

1. 多阶段 orchestrator 将 diffusion `OmniRequestOutput` 再包装时，没有复制最终 `latents` 和 `_multimodal_output`；
2. 0.26 输出协议将 metadata 移入 canonical envelope，部分 serving 代码仍只读取 outer output 或旧 flat 字段。

此外，当前实现只返回生成 bucket 尺寸，没有恢复 0.21 版本的 `infer_align_image_size` 语义：根据匹配条件图的原始宽高比计算独立 VAE decode 的最终尺寸。

## 2. P0：输出链路修复

### Orchestrator 转发

`vllm_omni/entrypoints/omni_base.py` 包装最终 stage 时新增转发：

- `latents`
- `_multimodal_output`
- 递归 `custom_output`

因此外层和最内层 `OmniRequestOutput` 都可以直接读取最终 latent 与 metadata。

### 统一读取接口

`OmniRequestOutput.postprocess_meta` 按以下顺序兼容读取：

1. `multimodal_output.metadata.image.postprocess_meta`
2. `multimodal_output.metadata.postprocess_meta`
3. `custom_output.postprocess_meta`

Images API 的 latent 提取增加 `unwrap()` fallback；Chat 的多阶段和纯 diffusion 两条路径统一使用该属性。

### 新旧协议并存

新实现使用 canonical 结构：

```json
{
  "metadata": {
    "image": {
      "postprocess_meta": {
        "w": 774,
        "h": 1355
      }
    }
  }
}
```

formatter 同时镜像旧结构：

```json
{
  "custom_output": {
    "postprocess_meta": {
      "w": 774,
      "h": 1355
    }
  }
}
```

这样旧客户端和 0.26 canonical consumer 均可读取。

## 3. P1：恢复原始尺寸对齐

### 请求参数

HunyuanImage3 新增/恢复两个 model-specific 参数：

- `infer_align_image_size: bool = false`
- `return_postprocess_meta: bool | null`

支持入口：

- Chat Completions `extra_body`
- Images generation JSON
- Images edit multipart form
- offline `extra_args`

普通 image 和 latent-only 输出中，`return_postprocess_meta` 均默认是 `true`；只有显式传 `false` 才关闭。

### 尺寸计算

当 `infer_align_image_size=true` 时：

1. 查找 `ratio_index` 与生成目标一致的条件图；
2. 读取条件图 resize/crop 前的原始 `width/height`；
3. 如果原始宽高比与 ratio bucket 的比例差异不小于 `0.01`，按目标面积重新缩放：

```text
target_area = image_base_size²
scale = sqrt(target_area / (original_width * original_height))
output_width = round(original_width * scale)
output_height = round(original_height * scale)
```

否则使用生成 bucket 尺寸。

例如条件图原始尺寸为 `571×1000`，匹配 `720×1280` ratio bucket，base size 为 1024 时，返回：

```json
{
  "w": 774,
  "h": 1355
}
```

### 原始尺寸传输

以下数据结构增加原始条件图尺寸：

- `ImageInfo.ori_image_width`
- `ImageInfo.ori_image_height`
- diffusion preprocess 的 JointImageInfo payload
- vLLM multimodal processor 的 `ori_image_size`

旧 payload 不包含原始尺寸时，自动回退到 bucket `image_width/image_height`。

### Step/request 两条路径

- step-based：在 `prepare_encode()` 中计算并保存 `postprocess_meta`，在 `post_decode()` 中写入 canonical metadata；
- request-based：生成结束后调用同一计算函数；
- latent-only：返回 model-space latent 和目标后处理尺寸；
- 服务内 image decode：`infer_align_image_size=true` 时同步 resize 最终 PIL 图片。

## 4. 使用方式

### Chat Completions

非流式 `/v1/chat/completions` 在普通 image 输出和 latent-only 输出中都会返回顶层 `postprocess_meta`。普通图片仍位于 `choices[].message.content[].image_url.url`；顶层 `image` 字段只用于 latent tensor data URI，因此 image 输出时 `image: null` 是正常的。

普通 image 和 latent-only 输出都会默认返回 metadata，无需传 `return_postprocess_meta`；如需关闭，可显式设置 `return_postprocess_meta: false`。

```json
{
  "model": "<MODEL_NAME>",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "编辑这张图片"},
        {"type": "image_url", "image_url": {"url": "<IMAGE_URL>"}}
      ]
    }
  ],
  "modalities": ["image"],
  "stream": false,
  "extra_body": {
    "infer_align_image_size": true
  }
}
```

### Images generation

```json
{
  "model": "<MODEL_NAME>",
  "prompt": "生成图片",
  "infer_align_image_size": true,
  "return_postprocess_meta": true,
  "response_format": "b64_json"
}
```

纯 T2I 没有条件图时，`infer_align_image_size` 会回退到 bucket 尺寸。

### Images edit

Multipart form 新增：

```text
infer_align_image_size=true
return_postprocess_meta=true
```

### Latent 响应

Chat：

```json
{
  "image": "data:application/x-torch-tensor;base64,<BASE64>",
  "postprocess_meta": {"w": 774, "h": 1355}
}
```

Images：

```json
{
  "data": [{"b64_json": "<BASE64_TORCH_TENSOR>"}],
  "output_format": "pt",
  "postprocess_meta": {"w": 774, "h": 1355}
}
```

独立 VAE decoder 使用 `postprocess_meta.w/h` 作为最终输出尺寸。

## 5. 修改文件

- `vllm_omni/entrypoints/omni_base.py`
  - 转发 latent、canonical metadata 和 custom output。
- `vllm_omni/outputs/__init__.py`
  - 递归 custom output；增加统一 `postprocess_meta` 属性。
- `vllm_omni/diffusion/output_formatter.py`
  - canonical metadata 镜像到旧 `custom_output`。
- `vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py`
  - 原始尺寸字段和旧版尺寸对齐算法。
- `vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py`
  - 原始尺寸 payload、step/request metadata、服务内 resize。
- `vllm_omni/model_executor/models/hunyuan_image3/hunyuan_image3.py`
  - vLLM multimodal 原始尺寸字段。
- `vllm_omni/model_extras/hunyuan_image3.py`
  - 注册两个请求参数。
- `vllm_omni/entrypoints/openai/protocol/images.py`
  - Images generation 请求 schema。
- `vllm_omni/entrypoints/openai/api_server.py`
  - Images generation/edit 参数透传、latent unwrap，以及多阶段 image 响应中的 metadata 转发。
- `vllm_omni/entrypoints/openai/serving_chat.py`
  - 共用图片生成方法向 Chat/Images 调用方返回 postprocess metadata，并兼容 canonical/legacy metadata 读取。

## 6. 测试

新增或更新：

- `tests/diffusion/test_diffusion_output_formatter.py`
  - canonical metadata、legacy flat metadata、custom output 镜像。
- `tests/diffusion/models/hunyuan_image3/test_external_cond_vae_inputs.py`
  - `571×1000 -> 774×1355` 对齐算法；原始尺寸 payload。
- `tests/outputs/test_postprocess_meta_passthrough.py`
  - nested orchestrator 输出和 legacy custom output fallback。

验证命令：

```bash
python -m pytest -q \
  tests/diffusion/test_diffusion_output_formatter.py \
  tests/diffusion/models/hunyuan_image3/test_external_cond_vae_inputs.py \
  tests/outputs/test_postprocess_meta_passthrough.py
```

还需在加载真实 HunyuanImage3 checkpoint 的环境验证：

1. standalone DiT latent-only Images API；
2. standalone DiT latent-only Chat API；
3. AR + DiT 多阶段 latent 输出；
4. 多条件图中匹配 ratio 条件图的尺寸选择；
5. latent -> 独立 VAE decode 的最终图片尺寸。
