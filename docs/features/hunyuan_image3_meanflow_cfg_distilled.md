# HunyuanImage3 MeanFlow 与 CFG-Distilled 支持

## 1. 目标

本次升级参考上游 `hunyuan_image_3_pipeline.py`、`modeling_hunyuan_image_3.py` 和 tokenizer 实现，在 vLLM-Omni 的 HunyuanImage3 diffusion pipeline 中支持：

- MeanFlow 时间区间条件；
- CFG-distilled 单分支 guidance；
- request-based 与 step-based 两条执行路径；
- grouped step batching；
- AR KV reuse；
- 条件图编辑；
- latent-only DiT 与独立 VAE decode。

功能完全由 checkpoint 的 `config.json` 自动启用，不增加请求级开关：

```json
{
  "cfg_distilled": true,
  "use_meanflow": true
}
```

普通 HunyuanImage3 checkpoint 保持原有行为。

## 2. CFG-Distilled

### 2.1 算法

普通 CFG 使用 conditional/unconditional 两个模型分支：

```text
pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
```

CFG-distilled checkpoint 已将 guidance 能力蒸馏进单个 conditional 分支。因此推理时：

- 不构造 unconditional prompt；
- 不复制 latent；
- 不执行 cond/uncond 合并；
- 将 guidance 值写入 `<guidance>` token。

传给 embedding 的值与参考实现一致：

```text
guidance = 1000 * guidance_scale
```

例如 `guidance_scale=5.0` 时，模型收到 `5000.0`。

### 2.2 限制

- `cfg_distilled=true` 要求 `cfg_parallel_size=1`；
- distilled 模式不支持 `guidance_rescale > 0`，请求会明确失败；
- checkpoint 必须包含 `guidance_emb.*` 权重；
- tokenizer 必须包含 `<guidance>` special token。

普通非 distilled checkpoint 仍支持传统 CFG 和 `guidance_rescale`。

## 3. MeanFlow

MeanFlow 除当前时间条件 \(t\) 外，还向模型提供区间右端点 \(r\)：

```text
r_i = timestep[i + 1]
r_last = 0
```

`r` 通过 `<timestep_r>` token 和 `timestep_r_emb` 注入模型。scheduler 的 Euler step 公式不变。

示例：

```text
timesteps = [1000, 500, 125]
step 0: t=1000, r=500
step 1: t=500,  r=125
step 2: t=125,  r=0
```

checkpoint 必须包含 `timestep_r_emb.*` 权重，tokenizer 必须包含 `<timestep_r>`。

## 4. 动态 Token 布局

生成图片 section 的动态 token 顺序与参考实现一致：

| 模型变体 | 动态 token 顺序 |
|---|---|
| 普通 | `<timestep>`, image tokens |
| CFG-distilled | `<timestep>`, `<guidance>`, image tokens |
| MeanFlow | `<timestep>`, `<timestep_r>`, image tokens |
| 两者同时启用 | `<timestep>`, `<guidance>`, `<timestep_r>`, image tokens |

首个 denoise step 在完整 prompt embedding 中按 scatter index 替换这些 token。后续 step 只构造动态 token 与 image token，并同步更新：

- `position_ids`；
- `attention_mask`；
- image query token 数；
- final layer 前的动态 token 裁剪数量；
- AR KV reuse 前缀截断后的 scatter index。

## 5. 两条执行路径

### 5.1 Request-based

HunyuanImage3Text2ImagePipeline 的 denoise loop 会：

1. 根据 checkpoint 计算 CFG batch factor；
2. 每步生成 `timestep_r`；
3. 按实际 batch 生成 `1000 * guidance_scale`；
4. 将二者传给 Hunyuan model；
5. distilled 模式跳过传统 CFG 合并；
6. 普通 CFG 按需执行 `guidance_rescale`。

### 5.2 Step-based

每个 `StepRequestState` 保存自身的：

- timestep 序列和当前 step index；
- guidance scale；
- guidance rescale；
- CFG factor。

因此 grouped batching 中不同请求即使处于不同 denoise step，也会使用自己的下一 timestep；最后一步使用零。distilled 模式每个请求只占一个模型 row。

不同条件图结构和 latent shape 会被拆入不同 step group，避免异构条件输入被静默丢弃或错误 padding。

## 6. 与已有功能组合

### 外部 COT

`assistant_prompt` 和 AR 生成的 COT 逻辑不变。AR COT 仍具有更高优先级。

### 外部条件 VAE

`cond_vae_images` / `cond_timesteps` 继续支持。distilled 模式的 CFG factor 为 1，因此条件 latent 和 ViT 条件不会生成 unconditional 副本。

### AR KV reuse

AR KV 前缀截断时会同步平移：

- `gen_timestep_scatter_index`；
- `guidance_scatter_index`；
- `timestep_r_scatter_index`。

Distilled checkpoint 不会执行 negative CFG prefill。

### Latent-only 与独立 Decode

`output_type: latent` 行为不变。MeanFlow/CFG-distilled 只影响 DiT denoise，不改变返回的 model-space latent 协议，也不改变独立 VAE decoder 合约。

## 7. 部署

无需增加模型变体专用 YAML。以下现有配置会根据 checkpoint 自动启用：

- `vllm_omni/deploy/hunyuan_image3_dit.yaml`
- `vllm_omni/deploy/hunyuan_image3_dit_latent.yaml`
- `vllm_omni/deploy/hunyuan_image_3_moe.yaml`

启动示例：

```bash
vllm serve <MODEL_PATH> \
  --omni \
  --deploy-config vllm_omni/deploy/hunyuan_image3_dit.yaml
```

对于 distilled checkpoint，配置中的 `cfg_parallel_size` 必须保持为 `1`。

客户端继续使用普通 `guidance_scale` 参数，不应传 `cfg_distilled` 或 `use_meanflow`：

```json
{
  "prompt": "a cat sitting beside a window",
  "guidance_scale": 5.0,
  "num_inference_steps": 50
}
```

服务会从 checkpoint 配置自动决定该值用于传统 CFG，还是编码为 distilled guidance token。

## 8. 修改点

### `hunyuan_image3_transformer.py`

- `ImageInfo` 增加 `add_timestep_r_token`；
- `HunyuanImage3Config` 增加 `cfg_distilled`、`use_meanflow`；
- `build_image_info()` 根据 checkpoint 自动设置特殊 token；
- request-based pipeline 增加单分支 distilled CFG 和 MeanFlow `r`；
- AR KV reuse 同步平移新增 scatter index；
- distilled 模式跳过 negative CFG prefill；
- 普通 CFG 补齐 guidance rescale。

### `hunyuan_image3_tokenizer.py`

- tokenizer 输出增加 `timestep_r_scatter_index`；
- 支持 `<timestep_r>` token；
- 将 token 计入 section 长度、位置索引与 batch stacking。

### `pipeline_hunyuan_image3.py`

- 自动创建并严格加载 `guidance_emb` / `timestep_r_emb`；
- 校验 checkpoint 所需 special token；
- 首步 scatter 和后续 step 拼接动态 embedding；
- 更新动态 token 的 position/attention/query length；
- step-based 和 grouped batching 支持 MeanFlow/distilled CFG；
- AR KV reuse、条件图分组和 request-local guidance rescale 适配。

### 部署配置

在三份 HunyuanImage3 YAML 中补充自动检测与 `cfg_parallel_size=1` 说明。

## 9. 测试

新增：

- `tests/diffusion/models/hunyuan_image3/test_meanflow_cfg_distilled.py`

覆盖：

- checkpoint 配置字段；
- distilled CFG factor；
- MeanFlow 下一 timestep 和末步零值；
- ImageInfo payload；
- 动态 token 顺序；
- batch scalar embedding；
- AR KV reuse scatter index 平移。

已有 `test_hunyuan_image3_step_execution.py` fixture 已补充普通 checkpoint 默认标志。

当前环境已通过：

```bash
python -m compileall -q vllm_omni tests/diffusion/models/hunyuan_image3

git diff --check
```

当前环境未安装 `pytest` 和 `ruff`。在标准环境执行：

```bash
python -m pytest -q \
  tests/diffusion/models/hunyuan_image3/test_meanflow_cfg_distilled.py \
  tests/diffusion/models/hunyuan_image3/test_hunyuan_image3_step_execution.py

python -m ruff check \
  vllm_omni/diffusion/models/hunyuan_image3 \
  tests/diffusion/models/hunyuan_image3/test_meanflow_cfg_distilled.py
```

还需要在实际 checkpoint 环境验证：

1. `cfg_distilled=true, use_meanflow=true` 的 image 输出；
2. 同一 checkpoint 的 latent-only 输出和独立 VAE decode；
3. distilled + 外部条件 VAE 图片编辑；
4. AR + DiT KV reuse；
5. 普通非蒸馏 checkpoint 回归。
