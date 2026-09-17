# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage3 DiT 运行时融合补丁（monkey-patch，无侵入源码）。

不修改任何模型源码，只在 vllm_omni 被 import 时安装 import-hook；
一旦 `vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer`
被首次 import，立即替换 3 处目标函数为 Ascend 融合算子。

融合项：
  1) HunYuanRotary2DEmbedder.__call__
     - 原：q = rope(q, cos, sin); k = rope(k, cos, sin) （2 次 RotaryPositionEmbedding kernel）
     - 新：torch_npu.npu_apply_rotary_pos_emb(q, k, cos, sin) （1 次 kernel 完成 Q+K）
  2) HunyuanImage3DecoderLayer.forward
     - 原：residual + hidden_states → RMSNorm(...) （Add + RmsNorm 两次 kernel）
     - 新：torch_npu.npu_add_rms_norm(x1, x2, gamma, eps) → (y, rstd, x1+x2)
     * 只融合本层内 (attn_residual + attn_out) → post_attention_layernorm 这 1 处；
       另一处 (mlp_residual + mlp_out) → 下一层 input_layernorm 是跨 DecoderLayer 的
       Add+Norm，为避免破坏 DecoderLayer 对外接口这里不做跨层融合。
  3) HunYuanMLP.forward
     - 原：SiluAndMul()(gate_up) （拆成 Swish + Mul 两次 kernel）
     - 新：torch_npu.npu_swiglu(gate_up)
  4) cos/sin 跨层共享
     - 原：`HunYuanRotary2DEmbedder._prepare_cos_sin` 里 32 层各自做
       `to(device) + repeat([...,2])`，各自缓存一份 tensor。
     - 新：把 `build_batch_2d_rope` wrap 一层，函数出口一次性完成 to+repeat；
       `_prepare_cos_sin` 加 fast path 识别已预处理，直接返回，32 层共享
       同一份 tensor。cos/sin 与 layer/step/t/guidance 无关，跨层复用完全等价。
  5) 跳过 repeat_kv（GQA head 展开）
     - 原：HunYuanAttention 里 `repeat_kv` 把 KV heads 从 8 展开到 32（与 Q 对齐）。
     - 新：直接 no-op，让下游 GQA-capable kernel 自己处理广播。
       当前部署走 mindiesd → aclnnFlashAttentionScoreV4，原生支持 GQA。
     - 有风险，默认关闭。开启需要 DIT_FUSE_SKIP_REPEAT_KV=1。

环境变量开关：
  DIT_FUSE_ROPE_QK          默认 1（开）
  DIT_FUSE_ADD_RMSNORM      默认 1（开）
  DIT_FUSE_SWIGLU           默认 1（开）
  DIT_FUSE_COS_SIN_SHARE    默认 1（开）
  DIT_FUSE_SKIP_REPEAT_KV   默认 0（关，需显式开启；见 (5) 里的风险说明）
  DIT_FUSE_LOG              默认 1（是否打印生效日志）

生效方式：
  本模块由 `vllm_omni/patch.py` 尾部一次性 import，随 vllm_omni 首次
  加载自动生效，不需要修改任何启动脚本或 PYTHONPATH。
"""

from __future__ import annotations

import builtins
import os
import sys

# -------------------- 开关 --------------------
def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip() not in ("0", "", "false", "False")


FUSE_ROPE_QK = _flag("DIT_FUSE_ROPE_QK")
FUSE_ADD_RMSNORM = _flag("DIT_FUSE_ADD_RMSNORM")
FUSE_SWIGLU = _flag("DIT_FUSE_SWIGLU")
# cos/sin 跨层共享（把 `to(device) + repeat` 从每层每步一次提前到 pipeline 一次）
FUSE_COS_SIN_SHARE = _flag("DIT_FUSE_COS_SIN_SHARE")
# 跳过 repeat_kv（GQA head 展开）：默认关闭，需要显式 opt-in。
# 前提：attention backend 是 mindiesd（走 aclnnFlashAttentionScoreV4），原生支持 GQA。
FUSE_SKIP_REPEAT_KV = _flag("DIT_FUSE_SKIP_REPEAT_KV", default="0")
# CANN MegaMoE：把 MoE 的 dispatch / grouped-matmul / routing / combine 一次融合。
# 触发条件（我们在下面 monkey-patch use_cann_megamoe，全部满足才生效）：
#   - VLLM_ASCEND_ENABLE_FUSED_MC2=1 且 additional_config.enable_fused_mc2=1
#   - EP 打开 且 1 < ep_world_size <= 64
#   - is_moe_model 判定为真（DiT 里 HunYuanSparseMoeBlock 满足）
#   - is_megamoe_supported_by_config 通过（H∈[1024,8192]%512、N∈[1024,3072]%512）
#   - _CANN_OPS_TRANSFORMER_AVAILABLE（cann_ops_transformer 已安装）
# 默认关闭，需要显式 DIT_ENABLE_MEGA_MOE=1 才 patch。
FUSE_MEGA_MOE = _flag("DIT_ENABLE_MEGA_MOE", default="0")
LOG_ENABLED = _flag("DIT_FUSE_LOG")


def _log(msg: str) -> None:
    if LOG_ENABLED:
        print(f"[hunyuan_image3_fusion] {msg}", flush=True)


# -------------------- torch_npu 探测 --------------------
try:
    import torch  # noqa: F401
    import torch_npu  # noqa: F401

    HAS_NPU = True
except Exception as _e:  # pragma: no cover
    HAS_NPU = False
    _log(f"torch_npu unavailable ({_e!r}); patches will not take effect.")


# ==================================================================
# 1) RoPE Q/K 融合
# ==================================================================
def _patch_rope(mod_transformer) -> None:
    if not FUSE_ROPE_QK:
        _log("skip RoPE Q/K fusion (DIT_FUSE_ROPE_QK=0)")
        return
    if not HAS_NPU:
        return
    cls = getattr(mod_transformer, "HunYuanRotary2DEmbedder", None)
    if cls is None:
        _log("HunYuanRotary2DEmbedder not found; skip RoPE fusion")
        return

    import torch
    import torch_npu

    def fused_call(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        hidden_states: torch.Tensor,
        custom_pos_emb,
        **kwargs,
    ):
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states_shape[-1])
        if kwargs.get("mode", "gen_text") != "gen_image":
            return q, k

        first_step = kwargs.get("first_step", False)
        device = q.device
        cos, sin = self._prepare_cos_sin(custom_pos_emb, first_step, device)

        query_lens = kwargs.get("query_lens")
        bs = len(query_lens)
        q_len = query_lens[0]
        assert hidden_states.shape[0] == bs * q_len, f"{hidden_states.shape[0]} != {bs * q_len}"

        # Reshape 到 [B, S, H, D]
        q_ = q.reshape(bs, q_len, self.num_heads, self.head_dim).to(torch.float32)
        k_ = k.reshape(bs, q_len, self.num_kv_heads, self.head_dim).to(torch.float32)

        # 让 cos/sin 与 q/k 广播兼容
        # 原实现里 cos/sin 是 [B, S, D] 或 [S, D]
        # npu_apply_rotary_pos_emb 要求 cos/sin shape 能与 q/k 广播: 一般是 [B, S, 1, D]
        cos_b = cos
        sin_b = sin
        if cos_b.dim() == 2:  # [S, D] -> [1, S, 1, D]
            cos_b = cos_b.unsqueeze(0).unsqueeze(2)
            sin_b = sin_b.unsqueeze(0).unsqueeze(2)
        elif cos_b.dim() == 3:  # [B, S, D] -> [B, S, 1, D]
            cos_b = cos_b.unsqueeze(2)
            sin_b = sin_b.unsqueeze(2)
        cos_b = cos_b.to(q_.dtype)
        sin_b = sin_b.to(q_.dtype)

        try:
            q_out, k_out = torch_npu.npu_apply_rotary_pos_emb(q_, k_, cos_b, sin_b)
        except Exception as e:
            # 若融合 API 不可用或形状不兼容，退回到两次 rope。
            if not getattr(self, "_fuse_rope_warned", False):
                _log(f"npu_apply_rotary_pos_emb failed, fallback to per-tensor rope: {e!r}")
                self._fuse_rope_warned = True
            q_out = self.rope(q_, cos, sin)
            k_out = self.rope(k_, cos, sin)

        q_out = q_out.reshape(hidden_states.shape[0], self.num_heads * self.head_dim).to(torch.bfloat16)
        k_out = k_out.reshape(hidden_states.shape[0], self.num_kv_heads * self.head_dim).to(torch.bfloat16)
        return q_out, k_out

    cls._orig_call = cls.__call__
    cls.__call__ = fused_call
    _log("patched HunYuanRotary2DEmbedder.__call__  (RoPE Q/K fused)")


# ==================================================================
# 2) Add + RmsNorm 融合
# ==================================================================
def _patch_decoder_layer(mod_transformer) -> None:
    if not FUSE_ADD_RMSNORM:
        _log("skip Add+RmsNorm fusion (DIT_FUSE_ADD_RMSNORM=0)")
        return
    if not HAS_NPU:
        return
    cls = getattr(mod_transformer, "HunyuanImage3DecoderLayer", None)
    if cls is None:
        _log("HunyuanImage3DecoderLayer not found; skip AddRmsNorm fusion")
        return

    import torch
    import torch_npu

    warn_holder = {"warned": False}

    def _add_rms(x1: torch.Tensor, x2: torch.Tensor, norm_mod) -> tuple[torch.Tensor, torch.Tensor]:
        """(y_norm, x_sum) = AddRmsNorm(x1, x2, gamma). 失败回退到 Add + RmsNorm。"""
        try:
            gamma = norm_mod.weight
            eps = norm_mod.variance_epsilon
            if x1.dtype != x2.dtype:
                x2 = x2.to(x1.dtype)
            gamma_used = gamma if gamma.dtype == x1.dtype else gamma.to(x1.dtype)
            y, _rstd, x_sum = torch_npu.npu_add_rms_norm(x1, x2, gamma_used, epsilon=eps)
            return y, x_sum
        except Exception as e:
            if not warn_holder["warned"]:
                _log(f"npu_add_rms_norm failed, fallback to Add+RmsNorm: {e!r}")
                warn_holder["warned"] = True
            x_sum = x1 + x2
            y = norm_mod(x_sum)
            return y, x_sum

    # ---------- 层内 & 跨层 AddRmsNorm 融合 ----------
    # 语义原始版：
    #   residual = h
    #   h = input_layernorm(h)                 # ① pre-attn RmsNorm，第 0 层前面没有 Add
    #   h, ... = self_attn(h, ...)
    #   h = residual + h                        # ② attn 后残差 Add
    #   residual = h
    #   h = post_attention_layernorm(h)         # ③ pre-mlp RmsNorm
    #   h = mlp(h)
    #   h = residual + h                        # ④ mlp 后残差 Add；接下一层 ①
    #
    # 融合后：
    #   - 层内 ②+③   -> 1 次 AddRmsNorm（本层固有）
    #   - ④@layer_i + ①@layer_{i+1}  -> 1 次 AddRmsNorm（跨层）
    #     具体做法：本层结尾直接调 `AddRmsNorm(residual, mlp_out, next_layer.input_layernorm.weight)`，
    #     一次返回：
    #       * `next_input_normed`  = RmsNorm(residual+mlp_out) with 下一层的 gamma
    #       * `sum` (即残差和)      = 用作本层对外返回的 hidden_states
    #     然后把 next_input_normed 挂到 next_layer._omni_input_normed；
    #     下一层入口检测到该字段就直接用（跳过自己的 input_layernorm），
    #     并把传入的 hidden_states 作为 residual。
    #   - 第 0 层：无 _omni_input_normed 字段，走原生 input_layernorm（不融合）。
    #   - 最后一层：无 next_layer 可用其 gamma，退化为普通 `residual + mlp_out`（不融合）。
    def fused_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        custom_pos_emb=None,
        **kwargs,
    ):
        # ---- ① pre-attn：优先复用上一层预算的 input_layernorm 结果 ----
        precomputed = getattr(self, "_omni_input_normed", None)
        if precomputed is not None:
            self._omni_input_normed = None  # 一次性消费
            residual = hidden_states        # 传入的 hidden_states 就是 sum，用作残差
            hidden_states = precomputed     # 直接用预算好的 input_layernorm 结果
        else:
            # 第 0 层，或链上未提前算：走原生 input_layernorm
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            custom_pos_emb=custom_pos_emb,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        # ---- 层内 ②+③ 融合：AddRmsNorm(residual, attn_out, post_norm) ----
        mlp_input, mlp_residual = _add_rms(
            residual, hidden_states, self.post_attention_layernorm
        )
        mlp_out = self.mlp(mlp_input)

        # ---- ④ + 下一层 ① 融合 ----
        next_layer = getattr(self, "_omni_next_layer", None)
        if next_layer is not None:
            # 一次 AddRmsNorm 同时得到"下一层的 input_layernorm 结果" + "残差和"
            next_input_normed, sum_ = _add_rms(
                mlp_residual, mlp_out, next_layer.input_layernorm
            )
            next_layer._omni_input_normed = next_input_normed
            hidden_states = sum_  # 对外仍返回真实 sum，接口不变
        else:
            # 最后一层：没有 next_layer 提供 gamma，无法融合。
            hidden_states = mlp_residual + mlp_out

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

    cls._orig_forward = cls.forward
    cls.forward = fused_forward
    _log(
        "patched HunyuanImage3DecoderLayer.forward "
        "(intra-layer + cross-layer AddRmsNorm; first layer & last layer non-fused as expected)"
    )

    # ---------- 建立 layer 链，需要感知 model.layers 顺序 ----------
    # 通过 patch HunyuanImage3Model.forward 在每次前向开始时惰性建立 next 指针。
    # 只做一次；用 _omni_layer_chain_ready 打标幂等。
    model_cls = getattr(mod_transformer, "HunyuanImage3Model", None)
    if model_cls is None:
        _log("HunyuanImage3Model not found; cross-layer AddRmsNorm chain skipped")
        return

    orig_model_forward = model_cls.forward

    def _ensure_layer_chain(model):
        if getattr(model, "_omni_layer_chain_ready", False):
            return
        layers = getattr(model, "layers", None)
        if layers is None:
            return
        n = len(layers)
        for i, layer in enumerate(layers):
            layer._omni_next_layer = layers[i + 1] if i + 1 < n else None
            layer._omni_input_normed = None  # side channel 初始值
        model._omni_layer_chain_ready = True
        _log(
            f"linked {n} decoder layers for cross-layer AddRmsNorm fusion "
            f"(layer 0 uses native input_layernorm; layer {n - 1} defers no output)"
        )

    def model_forward_wrapper(self, *args, **kwargs):
        _ensure_layer_chain(self)
        return orig_model_forward(self, *args, **kwargs)

    model_cls._orig_forward = orig_model_forward
    model_cls.forward = model_forward_wrapper


# ==================================================================
# 3) SwiGLU 融合 (SiluAndMul -> npu_swiglu)
# ==================================================================
# 背景：vLLM 上游 `vllm.model_executor.layers.activation.SiluAndMul`
# 只实现了 forward_cuda/native/xpu/cpu 四种，**没有 forward_npu**。
# `CustomOp.dispatch_forward` 在 NPU 平台走 `is_out_of_tree() -> forward_oot`，
# 而 CustomOp 默认的 `forward_oot` 直接 fallback 到 `forward_native`：
#     d = x.shape[-1] // 2
#     return F.silu(x[..., :d]) * x[..., d:]
# 展开成 `Slice + Silu(Swish) + Mul` 三个独立 kernel（正是 profile 里
# shared_experts 分支看到的序列）。
#
# 我们改成：直接把 SiluAndMul 类的 forward_oot 替换成 `torch_npu.npu_swiglu`，
# 这样 **所有** NPU 上创建的 `SiluAndMul()` 实例（无论何时创建、是否被
# `torch.compile` 编译入图，只要还没进 dispatcher 的阶段就取到 forward_oot）
# 都会得到融合。对 shared_experts 分支尤其关键（HunYuanMLP.forward 那个
# monkey-patch 可能已经被更早的 dynamo 编译产物锁死，改类属性无效）。
def _patch_silu_and_mul_forward_oot() -> None:
    """给 vLLM 上游 SiluAndMul 补一个 forward_oot(npu) 实现。"""
    if not FUSE_SWIGLU:
        return
    if not HAS_NPU:
        return
    try:
        from vllm.model_executor.layers.activation import SiluAndMul
    except Exception as e:
        _log(f"cannot import SiluAndMul: {e!r}")
        return

    import torch_npu

    warn_holder = {"warned": False}

    # 如果已被别人（e.g. 未来版本 vllm-ascend）改过，就不要再叠加了。
    if getattr(SiluAndMul, "_omni_swiglu_patched", False):
        _log("SiluAndMul.forward_oot already patched by another module; skip")
        return

    def forward_oot(self, x):
        try:
            return torch_npu.npu_swiglu(x)
        except Exception as e:
            if not warn_holder["warned"]:
                _log(f"npu_swiglu failed, fallback to native: {e!r}")
                warn_holder["warned"] = True
            # native fallback：等价于原 forward_native
            d = x.shape[-1] // 2
            import torch.nn.functional as F
            return F.silu(x[..., :d]) * x[..., d:]

    SiluAndMul._orig_forward_oot = SiluAndMul.forward_oot
    SiluAndMul.forward_oot = forward_oot
    SiluAndMul._omni_swiglu_patched = True
    _log("patched SiluAndMul.forward_oot  (native Slice+Silu+Mul -> npu_swiglu)")


def _patch_mlp(mod_transformer) -> None:
    """SwiGLU 融合入口。

    这里做两件事：
    1) 主打：`_patch_silu_and_mul_forward_oot()` —— 修补 vLLM 上游 SiluAndMul，
       让 NPU 平台上所有 `SiluAndMul()` 实例真正跑 `npu_swiglu`。这一步对
       `shared_experts` 分支的融合是决定性的。
    2) 备用：把 `HunYuanMLP.forward` 也替换掉，直接调用 `npu_swiglu`。这条
       路径只在没有 `torch.compile` 提前编译时才会实际生效。有了第 1 步之后
       第 2 步已经不必要，但保留一层保险不影响正确性。
    """
    _patch_silu_and_mul_forward_oot()

    if not FUSE_SWIGLU:
        _log("skip HunYuanMLP.forward SwiGLU fusion (DIT_FUSE_SWIGLU=0)")
        return
    if not HAS_NPU:
        return
    cls = getattr(mod_transformer, "HunYuanMLP", None)
    if cls is None:
        _log("HunYuanMLP not found; skip SwiGLU fusion")
        return

    import torch_npu

    warn_holder = {"warned": False}

    def fused_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        try:
            act_out = torch_npu.npu_swiglu(gate_up)
        except Exception as e:
            if not warn_holder["warned"]:
                _log(f"npu_swiglu failed in HunYuanMLP.forward, fallback: {e!r}")
                warn_holder["warned"] = True
            act_out = self.act_fn(gate_up)
        out, _ = self.down_proj(act_out)
        return out

    cls._orig_forward = cls.forward
    cls.forward = fused_forward
    _log("patched HunYuanMLP.forward (backup path; primary fusion is via SiluAndMul.forward_oot)")


# ==================================================================
# 4) cos/sin 跨层共享（一次预处理，32 层复用）
# ==================================================================
# 原实现的问题：
#   `HunYuanRotary2DEmbedder._prepare_cos_sin` 在每个 attention 层里都会做
#     cos = cos_input.to(device); cos = cos.repeat([..., 2])
#   32 层里各自的 embedder 实例各做一次 to/repeat，然后每层各自缓存一份
#   （first_step=False 之后走 self.custom_pos_emb 缓存，第 2 步起才复用）。
#   这意味着：
#     - 首步会有 32×2 = 64 个 to + repeat 小算子（Q/K 共用 cos/sin 但每层单独算）
#     - 32 份重复缓存 tensor（每份约 10527×head_dim×2B 级别，累计 O(10) MB）
#
# 优化：cos/sin 只依赖 token 布局，与 layer / step / t / guidance 全部无关。
#   1) 把 `build_batch_2d_rope` wrap 一层：在函数返回处一次性做
#      `to(device)` + `repeat([..., 2])`，last-dim 从 head_dim/2 铺到 head_dim。
#   2) 给 `_prepare_cos_sin` 加 fast path：检测 `cos.shape[-1] == self.head_dim`
#      则识别为"上游已预处理"，直接返回，不再 to/repeat/缓存。
#      32 层共享同一份 tensor，跨层零重复。
#
# 兼容性：`_prepare_cos_sin` 的 slow path 完整保留，若 cos/sin 未预处理（例如
# 未来新加的调用路径绕过 pipeline 直接构造）自动走老逻辑。
def _patch_cos_sin_share(mod_transformer) -> None:
    if not FUSE_COS_SIN_SHARE:
        _log("skip cos/sin cross-layer share (DIT_FUSE_COS_SIN_SHARE=0)")
        return
    embedder_cls = getattr(mod_transformer, "HunYuanRotary2DEmbedder", None)
    orig_build = getattr(mod_transformer, "build_batch_2d_rope", None)
    if embedder_cls is None or orig_build is None:
        _log("HunYuanRotary2DEmbedder/build_batch_2d_rope not found; skip cos/sin share")
        return

    import torch

    # ---- (1) wrap build_batch_2d_rope：函数出口一次性做 to(device) + repeat ----
    warn_holder = {"warned": False}

    def build_batch_2d_rope_shared(image_infos, seq_len, n_elem, device, base=10000):
        cos, sin = orig_build(
            image_infos=image_infos,
            seq_len=seq_len,
            n_elem=n_elem,
            device=device,
            base=base,
        )
        try:
            # 1) 确保 cos/sin 已在目标 device（原实现里通常已经在 device 上，此处 no-op）
            if cos.device != device:
                cos = cos.to(device)
                sin = sin.to(device)
            # 2) 若 last-dim 还是 half（== n_elem // 2），铺到 full（== n_elem）
            if cos.shape[-1] == n_elem // 2:
                repeat_sizes = [1] * (cos.dim() - 1) + [2]
                cos = cos.repeat(*repeat_sizes)
                sin = sin.repeat(*repeat_sizes)
            # 3) 打标记，便于下游 _prepare_cos_sin 快速识别（shape 判断已足够，
            #    但打标记可以在极端情况下 shape 意外相等时避免误判——此处属于
            #    "锦上添花"的信号）
            cos._omni_cos_sin_prepared = True  # type: ignore[attr-defined]
            sin._omni_cos_sin_prepared = True  # type: ignore[attr-defined]
        except Exception as e:  # pragma: no cover
            if not warn_holder["warned"]:
                _log(f"build_batch_2d_rope wrap failed, fallback to original: {e!r}")
                warn_holder["warned"] = True
        return cos, sin

    # 保留原函数以便回退
    mod_transformer._orig_build_batch_2d_rope = orig_build
    mod_transformer.build_batch_2d_rope = build_batch_2d_rope_shared

    # 同步 patch pipeline 模块里已经 import 好的引用（`from ... import build_batch_2d_rope`
    # 会把函数拷到 pipeline 模块名字空间，改 hunyuan_image3_transformer.build_...
    # 单独不够，pipeline 里仍会用旧函数）。
    _pipeline_mod = sys.modules.get(
        "vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3"
    )
    if _pipeline_mod is not None and hasattr(_pipeline_mod, "build_batch_2d_rope"):
        _pipeline_mod._orig_build_batch_2d_rope = _pipeline_mod.build_batch_2d_rope
        _pipeline_mod.build_batch_2d_rope = build_batch_2d_rope_shared
        _log("patched pipeline_hunyuan_image3.build_batch_2d_rope reference")
    else:
        # 若 pipeline 尚未 import，尝试用 __import__ 触发一次（正常场景下父包
        # __init__.py 已经保证 pipeline 与 transformer 同时加载）
        try:
            _pipeline_mod = __import__(
                "vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3",
                fromlist=["build_batch_2d_rope"],
            )
            if hasattr(_pipeline_mod, "build_batch_2d_rope"):
                _pipeline_mod._orig_build_batch_2d_rope = _pipeline_mod.build_batch_2d_rope
                _pipeline_mod.build_batch_2d_rope = build_batch_2d_rope_shared
                _log("patched pipeline_hunyuan_image3.build_batch_2d_rope after lazy import")
        except Exception as e:  # pragma: no cover
            _log(f"cannot patch pipeline_hunyuan_image3.build_batch_2d_rope: {e!r}")

    # ---- (2) 替换 HunYuanRotary2DEmbedder._prepare_cos_sin 加 fast path ----
    orig_prepare = embedder_cls._prepare_cos_sin

    def _prepare_cos_sin_shared(self, custom_pos_emb, first_step, device):
        cos_input, sin_input = custom_pos_emb
        # Fast path：上游已预处理（last-dim == head_dim，且在正确 device 上）
        already_prepared = (
            getattr(cos_input, "_omni_cos_sin_prepared", False)
            or (
                cos_input.dim() >= 1
                and cos_input.shape[-1] == self.head_dim
                and cos_input.device == device
            )
        )
        if already_prepared:
            # 32 层共享同一份 tensor，不再占用 self.custom_pos_emb 缓存槽。
            self.custom_pos_emb = None
            return cos_input, sin_input
        # Slow path：兼容未预处理的调用（走原实现，保留缓存策略）
        return orig_prepare(self, custom_pos_emb, first_step, device)

    embedder_cls._orig_prepare_cos_sin = orig_prepare
    embedder_cls._prepare_cos_sin = _prepare_cos_sin_shared
    _log(
        "patched build_batch_2d_rope + HunYuanRotary2DEmbedder._prepare_cos_sin "
        "(cos/sin now shared across all 32 layers; per-layer to/repeat/cache eliminated)"
    )


# ==================================================================
# 触发时机
# ==================================================================
# 本模块推荐的触发点是 `vllm_omni/diffusion/models/hunyuan_image3/__init__.py`
# 尾部（此时 `hunyuan_image3_transformer` 已经完整加载，3 个目标类都已定义）。
# 若被更早的地方（如 `vllm_omni/patch.py`）先 import，则安装 import hook 等
# 目标模块加载完成再 apply。
_TARGET_MOD = "vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer"
_TARGET_CLASSES = ("HunYuanRotary2DEmbedder", "HunyuanImage3DecoderLayer", "HunYuanMLP")
_PATCH_DONE = False


def _target_ready(mod) -> bool:
    """判断目标模块是否已完成 class 定义。

    Python 在文件开始执行前就会把 module 对象放进 `sys.modules`（避免循环 import），
    所以仅凭 `name in sys.modules` 判断"是否可 patch"会在类定义未完成时误触发。
    要求 3 个目标类全部已存在，才认为模块已经 import 完毕、可以安全打补丁。
    """
    return all(hasattr(mod, c) for c in _TARGET_CLASSES)


# ==================================================================
# 5) 跳过 repeat_kv（GQA 展开）
# ==================================================================
# 背景：HunYuanAttention 里的 `repeat_kv` 把 KV heads 从 8 展开到 32（与 Q 对齐），
# 是给"不支持 GQA 的 attention backend"准备的。当前部署实际走的是 mindiesd
# 的 `attention_forward(op_type="fused_attn_score")` → `aclnnFlashAttentionScoreV4`，
# 该 kernel 原生支持 GQA（q_head_num % kv_head_num == 0 即可，Hunyuan 32/8=4 ✅），
# 无需外部展开。
#
# `repeat_kv` 的开销：
#   - 每次调用 = 1 次 `.expand(...)` （BroadcastTo）+ 1 次 `.reshape(...)`
#   - Distil：32 层 × 每层 2 次（key + value）× N 步 = 64N 次调用，共 128N 个小算子
#   - 更实在的收益：KV tensor 从 `[B, S, 32, D]` 缩到 `[B, S, 8, D]`，attention
#     kernel input 内存流量降到约 1/2（Q + K/4 + V/4）
#
# 风险：如果某个未预期的分支不走 mindiesd（例如 SDPA fallback / joint text branch
# 的某个 attention），会得到 head 数不匹配的 tensor 传入 attention，报 shape 错。
# 因此本 patch 默认关闭，需要显式 `DIT_FUSE_SKIP_REPEAT_KV=1` 开启。
def _patch_skip_repeat_kv() -> None:
    if not FUSE_SKIP_REPEAT_KV:
        _log("skip repeat_kv fusion (DIT_FUSE_SKIP_REPEAT_KV=0)")
        return

    try:
        from vllm_omni.diffusion.utils import kv_utils
    except Exception as e:
        _log(f"cannot import kv_utils: {e!r}; skip repeat_kv fusion")
        return

    orig_repeat_kv = getattr(kv_utils, "repeat_kv", None)
    if orig_repeat_kv is None:
        _log("repeat_kv not found in kv_utils; skip")
        return

    # 幂等：避免多次 patch 叠加
    if getattr(orig_repeat_kv, "_omni_repeat_kv_skip", False):
        _log("repeat_kv already patched; skip")
        return

    warn_holder = {"warned": False}

    def repeat_kv_skip(hidden_states, n_rep):
        """No-op：直接返回原 tensor，让下游 GQA-capable kernel 自己处理广播。"""
        # 只在首次调用时输出一条 diag log，便于确认真的走到这里。
        if not warn_holder["warned"]:
            _log(
                f"repeat_kv called with n_rep={n_rep}, skipped (input kv_shape={tuple(hidden_states.shape)}). "
                "Downstream attention kernel must support GQA."
            )
            warn_holder["warned"] = True
        return hidden_states

    repeat_kv_skip._omni_repeat_kv_skip = True  # type: ignore[attr-defined]

    # 保留原函数以便回退（或诊断用）
    kv_utils._orig_repeat_kv = orig_repeat_kv
    kv_utils.repeat_kv = repeat_kv_skip

    # 同步 patch transformer 模块里已经 import 好的引用
    # （`from vllm_omni.diffusion.utils.kv_utils import repeat_kv` 会把函数拷到
    #  transformer 模块名字空间，改 kv_utils 里的不够，transformer 里也要改）。
    for target_name in (
        "vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer",
    ):
        mod = sys.modules.get(target_name)
        if mod is not None and hasattr(mod, "repeat_kv"):
            mod._orig_repeat_kv = mod.repeat_kv
            mod.repeat_kv = repeat_kv_skip
            _log(f"patched {target_name}.repeat_kv reference")

    _log(
        "patched repeat_kv -> no-op "
        "(assumes attention backend supports GQA natively, e.g. mindiesd/aclnnFlashAttentionScoreV4)"
    )


def _apply_patches(mod) -> None:
    global _PATCH_DONE
    if _PATCH_DONE:
        return
    _PATCH_DONE = True
    try:
        _patch_rope(mod)
        _patch_decoder_layer(mod)
        _patch_mlp(mod)
        _patch_cos_sin_share(mod)
        _patch_skip_repeat_kv()
        _log("all fusion patches applied.")
    except Exception as e:  # pragma: no cover
        _log(f"apply_patches failed: {e!r}")


def _try_apply_now() -> bool:
    """尝试立即打补丁；成功返回 True，否则 False（目标模块还没就绪）。"""
    m = sys.modules.get(_TARGET_MOD)
    if m is not None and _target_ready(m):
        _apply_patches(m)
        return True
    return False


def _install_import_hook() -> None:
    """兜底：装一次性 import hook，等目标模块类齐全后 apply。"""
    _orig_import = builtins.__import__
    if getattr(_orig_import, "_hunyuan_image3_fusion_hook", False):
        return  # 已装过

    def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        mod_ret = _orig_import(name, globals, locals, fromlist, level)
        if not _PATCH_DONE:
            m2 = sys.modules.get(_TARGET_MOD)
            if m2 is not None and _target_ready(m2):
                _apply_patches(m2)
        return mod_ret

    _patched_import._hunyuan_image3_fusion_hook = True  # type: ignore[attr-defined]
    builtins.__import__ = _patched_import
    _log(f"import-hook installed (fallback), waiting for {_TARGET_MOD} to be fully loaded ...")


# NOTE: DiT class patch 的实际触发（`_try_apply_now()` / `_install_import_hook()`）
# 已经挪到文件末尾——mega_moe 段里会 wrap `_apply_patches` 以追加 shared-expert 解耦，
# 必须等 wrap 完成后再触发，否则 import-hook 里的旧闭包会用到未被 wrap 的旧函数。


# ==================================================================
# 6) NPU MegaMoE：把 MoE 的 dispatch / GEMM / SwiGLU / GEMM / combine
#    融合成一个大 kernel（bf16 no-quant 模式）
# ==================================================================
#
# 背景与决策
# ----------
# 前几轮尝试通过 patch vllm-ascend 的 `use_cann_megamoe` /
# `is_megamoe_supported_by_config` / `load_cann_mega_moe_ops` 来让 mega_moe
# 上线，但最后确认：**vllm-omni 的 DiT 走的是自己 `set_forward_context`，
# 从来不进 vllm-ascend 的 `set_ascend_forward_context`**，所以 `_EXTRA_CTX`
# 里的 `use_mega_moe` / `moe_comm_method` 永远不会被写入，vllm-ascend 那
# 整条 MoE comm 分派链在 DiT 场景下形同虚设。
#
# 因此改为**直接接管 `HunYuanSparseMoeBlock.forward` 的 routed 分支**：
#
#   shared_out = shared_mlp(x)                          # 主流串行
#   router_logits, _ = self.gate(x)
#   topk_w, topk_ids = self.experts.router.select_experts(x, router_logits)
#   routed_out, _ = npu_ops_transformer.ops.mega_moe(
#       x=x, topk_ids=topk_ids.int(), topk_weights=topk_w.bf16,
#       l1_weights=[self.experts.w13_weight],           # [E_local, 2N, H]
#       l2_weights=[self.experts.w2_weight],            # [E_local, H, N]
#       sym_buffer=_global_mega_moe_symm_buffer,        # 跨 32 层共享一份
#       # bf16 no-quant: scales/l1_sf/l2_sf/dispatch_quant_mode 均取默认(0/None)
#   )
#   return routed_out + shared_out
#
# 这条路径完全绕开 vllm 上游 FusedMoE.MoERunner / vllm-ascend
# FusedMC2CommImpl，直接对接 `npu_ops_transformer.ops.mega_moe`，跟
# `mega_moe_demo_real.py` / 官方 `doc/mega_moe.md` 完全同构。
#
# 兼容性
# ------
# * bf16 无量化：`dispatch_quant_mode=0`，`l1_weights_sf` / `l2_weights_sf`
#   均为 None，权重直接用 vllm FusedMoE 加载好的 bf16 `w13_weight` /
#   `w2_weight`（shape 与 mega_moe 期望完全一致，见 doc/mega_moe.md）。
# * Hunyuan Image 3 的 shape：EP=4, num_experts=192, expertPerRank=48,
#   topK=8, H=4096, N=1536。命中 doc 中 A5 硬检查 & 回归 catalog 范围。
# * SymmBuffer：doc 明确"同一进程同一 HCCL group 只能创建一次"，因此我们把
#   32 层 MoE 共享**同一个 SymmBuffer**（EP group / shape 参数完全相同）。
#   首次 forward 懒创建，进程存续期间不 destroy。


# ------------------------------------------------------------------
# 6.1) npu_ops_transformer 探测
# ------------------------------------------------------------------
_NPU_OPS_TRANSFORMER_AVAILABLE = False


def _detect_npu_ops_transformer() -> bool:
    """探测 npu_ops_transformer 是否可 import。"""
    global _NPU_OPS_TRANSFORMER_AVAILABLE
    try:
        import importlib.util

        _NPU_OPS_TRANSFORMER_AVAILABLE = (
            importlib.util.find_spec("npu_ops_transformer") is not None
        )
    except Exception:
        _NPU_OPS_TRANSFORMER_AVAILABLE = False
    return _NPU_OPS_TRANSFORMER_AVAILABLE


# ------------------------------------------------------------------
# 6.1a) 修复 vllm-ascend 的 is_megamoe_supported_by_config（**必须始终执行**）
# ------------------------------------------------------------------
# 这个 patch 与 DIT_ENABLE_MEGA_MOE 开关无关，是**启动崩溃修复**：
#
# vllm-ascend 的 AscendConfig.__init__（ascend_config.py:203）里三条件 and 的第 3
# 条会调 is_megamoe_supported_by_config；只要 yaml 里
#   additional_config.enable_fused_mc2: 1
# 且 _CANN_OPS_TRANSFORMER_AVAILABLE=True（本容器 npu_ops_transformer 探测到），
# 就一定会走到这个判定。
#
# 但 vllm-ascend 硬假设 moe_intermediate_size 是 int：
#   return moe_intermediate_size >= 1024 and ... and % 512 == 0
# 而 Hunyuan Image 3 的 config 里 moe_intermediate_size 是 per-layer list（长度
# num_hidden_layers）。直接 TypeError 崩溃，且发生在**每个 DiffusionWorker
# 子进程初始化时**，跟 mega_moe 是否启用无关。
#
# 因此本 patch **无条件执行**，且必须在 AscendConfig() 构造之前完成（vllm_omni
# import 时执行，早于 worker.init_device → init_ascend_config → AscendConfig()）。
def _patch_is_megamoe_supported_by_config() -> None:
    try:
        import vllm_ascend.ascend_config as _ac
    except Exception as e:  # pragma: no cover
        _log(f"vllm_ascend.ascend_config not importable ({e!r}); skip fix")
        return

    if getattr(_ac.is_megamoe_supported_by_config, "_hunyuan_image3_patched", False):
        return

    def _shape_ok(n) -> bool:
        return isinstance(n, int) and 1024 <= n <= 3072 and n % 512 == 0

    def _patched(vllm_config) -> bool:
        try:
            hf_text_config = vllm_config.model_config.hf_text_config
        except AttributeError:
            return False
        n = getattr(hf_text_config, "moe_intermediate_size", None)
        if n is None:
            return False
        if isinstance(n, int):
            return _shape_ok(n)
        if isinstance(n, (list, tuple)):
            return len(n) > 0 and all(_shape_ok(x) for x in n)
        return False

    _patched._hunyuan_image3_patched = True  # type: ignore[attr-defined]
    _ac.is_megamoe_supported_by_config = _patched
    # 同步给已把该函数抓走的其它模块
    m = sys.modules.get("vllm_ascend.ascend_forward_context")
    if m is not None and hasattr(m, "is_megamoe_supported_by_config"):
        setattr(m, "is_megamoe_supported_by_config", _patched)
    _log("is_megamoe_supported_by_config patched to accept per-layer list "
         "moe_intermediate_size (Hunyuan Image 3 uses list of length num_hidden_layers).")


# 立即执行（与 DIT_ENABLE_MEGA_MOE 无关，是启动崩溃修复）
_patch_is_megamoe_supported_by_config()


# ------------------------------------------------------------------
# 6.2) 全局 SymmBuffer：跨 32 层共享
# ------------------------------------------------------------------
# 说明（来自 doc/mega_moe.md）：
#   同一进程同一 HCCL group 只能创建一次 MegaMoe MC2 context。应在全部
#   调用期间保持并复用同一个 SymmBuffer，只在最后一次调用完成后、
#   group/process teardown 前执行 SymmBuffer.destroy()。当前 destroy()
#   释放设备 buffer，但不销毁 HCCL Engine context；销毁后不支持在同一
#   进程中用同一 group 再建一个 SymmBuffer。
class _MegaMoeGlobalState:
    """跨层复用 SymmBuffer / EP-group 信息 / 动态上界。"""

    def __init__(self) -> None:
        self.sym_buffer = None
        self.ep_group_device = None  # torch.distributed.ProcessGroup
        self.ep_world_size = 0
        self.num_experts = 0        # global
        self.local_experts = 0
        self.hidden = 0
        self.intermediate = 0
        self.topk = 0
        self.num_max_tokens_per_rank = 0
        self.warned_bs_over = False

    def ensure(
        self,
        local_experts: int,
        hidden: int,
        intermediate: int,
        topk: int,
        bs_hint: int,
    ) -> None:
        """首次调用时懒创建 SymmBuffer；后续调用只做 sanity check。"""
        if self.sym_buffer is not None:
            # 只在 BS 超过初始上界时打一次 warning 并 assert 保护
            if bs_hint > self.num_max_tokens_per_rank and not self.warned_bs_over:
                self.warned_bs_over = True
                _log(
                    f"[MegaMoE] BS={bs_hint} exceeds initial num_max_tokens_per_rank="
                    f"{self.num_max_tokens_per_rank}; will fall back to non-mega path."
                )
            return

        from vllm.distributed.parallel_state import get_ep_group
        from npu_ops_transformer.ops import get_symm_buffer_for_mega_moe

        ep = get_ep_group()
        self.ep_group_device = ep.device_group
        self.ep_world_size = ep.world_size
        self.num_experts = local_experts * self.ep_world_size
        self.local_experts = local_experts
        self.hidden = hidden
        self.intermediate = intermediate
        self.topk = topk

        # BS 上界估计：DiT 场景 dummy run 用 512x512 仅 ~1K tokens；正式 case 图像
        # 大小可到 1024x1024 → ~4K tokens/rank。设一个宽松的上界，兼容首次 dummy
        # 与后续正常 case，且不超过 doc 上限 16384。
        upper = max(bs_hint * 4, 4096)
        upper = min(upper, 16384)
        upper = (upper + 15) // 16 * 16  # 16 对齐
        self.num_max_tokens_per_rank = upper

        # max_recv_token_num 是 dispatch 后单 rank 收到的 token 上限。**不能传 0**：
        # C++ kernel 内部把这个值当分子做整数除法，为 0 会 segfault。
        # 参照 vllm-ascend `_init_mega_moe_symm_buffer`（moe_comm_method.py:337-340）：
        #   absolute_safe = num_max_tokens_per_rank * ep_world_size * min(num_topk, expert_per_rank)
        max_recv = upper * self.ep_world_size * min(topk, local_experts)
        max_recv = max(max_recv, 1)

        # 无条件打印（关键调试信息，不依赖 DIT_FUSE_LOG）
        print(
            f"[MegaMoE] Creating SymmBuffer: ep_ws={self.ep_world_size} "
            f"num_experts={self.num_experts} local_E={local_experts} "
            f"topK={topk} H={hidden} N={intermediate} "
            f"num_max_tokens_per_rank={upper} max_recv_token_num={max_recv} "
            f"(hint BS={bs_hint})",
            flush=True,
        )
        self.sym_buffer = get_symm_buffer_for_mega_moe(
            group=self.ep_group_device,
            num_experts=self.num_experts,
            num_max_tokens_per_rank=upper,
            num_topk=topk,
            hidden=hidden,
            intermediate_hidden=intermediate,
            max_recv_token_num=max_recv,
            dispatch_quant_mode=0,  # bf16 no-quant
            dispatch_quant_out_dtype=28,  # 忽略（no-quant 模式）
            combine_quant_mode=0,
            comm_alg="",
        )
        _log("[MegaMoE] SymmBuffer ready.")


_MEGA_MOE_GLOBAL = _MegaMoeGlobalState()


# ------------------------------------------------------------------
# 6.3) 核心 patch：接管 HunYuanSparseMoeBlock.forward
# ------------------------------------------------------------------
def _patch_moe_block_with_mega_moe(mod_transformer) -> None:
    """接管 HunYuanSparseMoeBlock：
      * __init__：走原生构造，事后清除 experts._shared_experts 引用（防止
        MoERunner 走 double-stream shared+routed 路径，因为我们 forward 里
        根本不会再调用 self.experts 的 forward）。
      * forward：完全自实现。shared_mlp 串行 + mega_moe kernel + 相加。
    """
    if not FUSE_MEGA_MOE:
        _log("skip MoeBlock MegaMoE takeover (DIT_ENABLE_MEGA_MOE=0)")
        return
    if not HAS_NPU:
        _log("skip MoeBlock MegaMoE takeover (torch_npu unavailable)")
        return
    if not _detect_npu_ops_transformer():
        _log("skip MoeBlock MegaMoE takeover (npu_ops_transformer unimportable)")
        return

    cls = getattr(mod_transformer, "HunYuanSparseMoeBlock", None)
    if cls is None:
        _log("HunYuanSparseMoeBlock not found; skip MegaMoE takeover")
        return
    if getattr(cls, "_omni_mega_moe_patched", False):
        return

    orig_init = cls.__init__

    def _init_with_mega_moe(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        # 记住 shared_mlp（可能为 None，如 use_mixed_mlp_moe==False 时）
        self._omni_shared_mlp = getattr(self, "shared_mlp", None)

        # 把 AscendMoERunner / MoERunner 里的 shared_experts 引用抹掉，避免它自建
        # double-stream 分支。同时重新 bind _forward_entry（否则 tuple/tensor 不匹配）。
        experts = getattr(self, "experts", None)
        if experts is not None and hasattr(experts, "_shared_experts"):
            experts._shared_experts = None
            if hasattr(experts, "_select_forward"):
                try:
                    experts._forward_entry = experts._select_forward()
                except Exception:
                    pass

        # **不要**在这里访问 w13_weight / w2_weight：
        #   1) 权重此时可能还没通过 quant_method.create_weights 建好；
        #   2) AscendMoERunner 把权重挂在 `routed_experts` 上，不是 runner 本身；
        #   3) 加载完后 AscendUnquantizedFusedMoEMethod.process_weights_after_loading
        #      会对权重做 transpose(1,2)，shape 从 [E, 2N, H] 变成 [E, H, 2N]
        #      （w2 同理从 [E, H, N] 变成 [E, N, H]）。
        # 我们在首次 forward 时懒初始化：从 routed_experts 拿权重，做逆 transpose
        # 到 mega_moe 需要的 [E, 2N, H] 与 [E, H, N]，并缓存到 module 属性上。
        self._omni_weights_ready = False
        self._omni_w13 = None
        self._omni_w2 = None
        self._omni_local_experts = 0
        self._omni_intermediate = 0
        self._omni_hidden = 0
        self._omni_topk = 0
        # router：AscendMoERunner 继承基类的 self.router；也可能在 routed_experts
        # 上。查两遍。
        router = getattr(experts, "router", None) if experts is not None else None
        if router is None and experts is not None:
            re = getattr(experts, "routed_experts", None)
            if re is not None:
                router = getattr(re, "router", None)
        self._omni_router = router

    def _lazy_init_weights(self) -> bool:
        """首次 forward 时把权重从 routed_experts 拿出来并逆 transpose。
        成功返回 True，失败返回 False（导致回退）。"""
        if self._omni_weights_ready:
            return True
        experts = getattr(self, "experts", None)
        if experts is None:
            return False
        # 权重的实际宿主：AscendMoERunner → routed_experts；
        # 基类 MoERunner 可能直接挂在 self。两种都尝试。
        holder = getattr(experts, "routed_experts", None) or experts
        w13 = getattr(holder, "w13_weight", None)
        w2 = getattr(holder, "w2_weight", None)
        # 若 vllm-ascend 的 mega_moe 分支已把权重 unbind 到 *_weight_list，
        # 就直接用（已经是 list of [1, 2N, H] / [1, H, N]，但每张只有 1 个 E）。
        # 但注意 mega_moe 要求 l1_weights[e] 是 [expertPerRank, N, H]（含 E 维），
        # 所以 list 版本我们没法直接用；这里只支持整块 weight 场景。
        if w13 is None or w2 is None:
            return False
        if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
            return False

        # 逆 transpose：process_weights_after_loading 里做的是 [E, 2N, H] → [E, H, 2N]
        # 与 [E, H, N] → [E, N, H]。我们要求 mega_moe 的 l1/l2 是 [E, 2N, H] 与 [E, H, N]。
        # 逆 transpose 即再来一次 transpose(1, 2)。用 .contiguous() 保证 kernel 期望 layout。
        w13_for_mega = w13.transpose(1, 2).contiguous()  # [E, 2N, H]
        w2_for_mega = w2.transpose(1, 2).contiguous()    # [E, H, N]

        E_local, two_N, H = int(w13_for_mega.shape[0]), int(w13_for_mega.shape[1]), int(w13_for_mega.shape[2])
        if two_N % 2 != 0:
            _log(f"[MegaMoE] w13.shape[1]={two_N} not even, skip take-over")
            return False
        N = two_N // 2
        # w2 shape 应当是 [E, H, N]
        if w2_for_mega.shape != (E_local, H, N):
            _log(
                f"[MegaMoE] w2 shape {tuple(w2_for_mega.shape)} != "
                f"expected ({E_local}, {H}, {N})，skip take-over"
            )
            return False

        self._omni_w13 = w13_for_mega
        self._omni_w2 = w2_for_mega
        self._omni_local_experts = E_local
        self._omni_intermediate = N
        self._omni_hidden = H
        # topk：AscendMoERunner 会挂 self.top_k，兜底看 self.gate.top_k
        topk = int(getattr(experts, "top_k", 0)) or int(getattr(self.gate, "top_k", 0)) or 0
        self._omni_topk = topk
        self._omni_weights_ready = True
        _log(
            f"[MegaMoE] Weights prepared for layer: E_local={E_local} "
            f"H={H} N={N} topK={topk} (w13→{tuple(w13_for_mega.shape)}, "
            f"w2→{tuple(w2_for_mega.shape)})"
        )
        return True

    def _forward_mega_moe(self, hidden_states):
        # 走不到 mega_moe 的兜底：权重还没 ready 或 shape 不合预期
        if not _lazy_init_weights(self):
            return _forward_shared_decoupled_only(self, hidden_states)

        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        x = hidden_states.view(-1, hidden_dim)
        bs = int(x.shape[0])

        # 1) shared 分支（可能为 None）
        shared_out = None
        shared_mlp = getattr(self, "_omni_shared_mlp", None)
        if shared_mlp is not None:
            shared_out = shared_mlp(x)
            if isinstance(shared_out, tuple):
                shared_out = shared_out[0]

        # 2) router：产出 topk_ids/topk_weights
        #
        # 注意：**不能**调用 router.select_experts(...)。它默认走 vllm 上游
        # fused_topk_router.py → vllm_topk_softmax → torch.ops._moe_C.topk_softmax，
        # 而 `_moe_C` 是 vllm 编译时的 CUDA/ROCm kernel namespace，在 NPU 环境根
        # 本没这个符号，会抛
        #   AttributeError: '_OpNamespace' '_moe_C' object has no attribute 'topk_softmax'
        # 直接用 NPU 融合算子（跟 vllm-ascend `_310p/fused_moe/experts_selector.py`
        # 完全一致）。
        # DiT 侧 HunYuanSparseMoeBlock 里 FusedMoE 的构造：
        #   top_k=8, renormalize=(top_k>1)=True, 无 grouped_topk / bias /
        #   custom_routing_function / scoring_func！=softmax
        # 所以我们只需要普通 topk_softmax + renormalize。
        router_logits, _ = self.gate(x)
        # 大 BS 时（>1024）310p 那份实现里做了 chunk，A5 上没这个约束，一次算完即可。
        topk_weights, topk_ids, _row_indices = torch_npu.npu_moe_gating_top_k_softmax(
            router_logits,
            None,  # finished, 不用
            k=self._omni_topk,
        )
        # renormalize（HunYuan 里 renormalize=top_k>1，DiT top_k=8 恒 True）
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        topk_ids = topk_ids.to(torch.int32)
        # doc: topk_weights 只支持 bf16
        if topk_weights.dtype != torch.bfloat16:
            topk_weights = topk_weights.to(torch.bfloat16)

        # 3) SymmBuffer 懒初始化（跨 32 层共享）
        _MEGA_MOE_GLOBAL.ensure(
            local_experts=self._omni_local_experts,
            hidden=self._omni_hidden,
            intermediate=self._omni_intermediate,
            topk=self._omni_topk,
            bs_hint=bs,
        )
        # 若 BS 超上界，回退
        if (_MEGA_MOE_GLOBAL.sym_buffer is None
                or bs > _MEGA_MOE_GLOBAL.num_max_tokens_per_rank):
            return _forward_shared_decoupled_only(self, hidden_states)

        # 4) 直接调 mega_moe（bf16 no-quant）
        from npu_ops_transformer.ops import mega_moe as _mega_moe_fn

        # 第一次调用时无条件 dump 所有输入 tensor 的元信息，便于崩溃时定位。
        if not getattr(_MEGA_MOE_GLOBAL, "_dumped", False):
            _MEGA_MOE_GLOBAL._dumped = True

            def _t(name, t):
                if t is None:
                    return f"  {name}: None"
                return (
                    f"  {name}: shape={tuple(t.shape)} dtype={t.dtype} "
                    f"device={t.device} contig={t.is_contiguous()}"
                )

            print(
                "[MegaMoE] First mega_moe call tensor dump:\n"
                + _t("x", x) + "\n"
                + _t("topk_ids", topk_ids) + "\n"
                + _t("topk_weights", topk_weights) + "\n"
                + _t("l1_weights[0] (w13)", self._omni_w13) + "\n"
                + _t("l2_weights[0] (w2)", self._omni_w2) + "\n"
                + f"  sym_buffer: type={type(_MEGA_MOE_GLOBAL.sym_buffer).__name__} "
                + f"num_experts={_MEGA_MOE_GLOBAL.sym_buffer.num_experts} "
                + f"ep_ws={_MEGA_MOE_GLOBAL.sym_buffer.ep_world_size} "
                + f"num_max_tokens_per_rank={_MEGA_MOE_GLOBAL.sym_buffer.num_max_tokens_per_rank} "
                + f"num_topk={_MEGA_MOE_GLOBAL.sym_buffer.num_topk} "
                + f"H={_MEGA_MOE_GLOBAL.sym_buffer.hidden} "
                + f"N={_MEGA_MOE_GLOBAL.sym_buffer.intermediate_hidden} "
                + f"max_recv_token_num={_MEGA_MOE_GLOBAL.sym_buffer.max_recv_token_num} "
                + f"global_bs={_MEGA_MOE_GLOBAL.sym_buffer.global_bs}",
                flush=True,
            )

        routed_out, _expert_token_nums = _mega_moe_fn(
            x=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            l1_weights=[self._omni_w13],
            l2_weights=[self._omni_w2],
            sym_buffer=_MEGA_MOE_GLOBAL.sym_buffer,
            scales=None,
            l1_weights_sf=None,
            l2_weights_sf=None,
            x_active_mask=None,
        )

        # 5) 相加
        final = routed_out if shared_out is None else routed_out + shared_out
        return final.view(orig_shape)

    def _forward_shared_decoupled_only(self, hidden_states):
        """兜底：跟前一版本一致的实现（shared 串行 + vllm 上游 MoERunner）。"""
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        x = hidden_states.view(-1, hidden_dim)
        shared_mlp = getattr(self, "_omni_shared_mlp", None)
        shared_out = None
        if shared_mlp is not None:
            shared_out = shared_mlp(x)
            if isinstance(shared_out, tuple):
                shared_out = shared_out[0]
        router_logits, _ = self.gate(x)
        routed_out = self.experts(hidden_states=x, router_logits=router_logits)
        final = routed_out if shared_out is None else routed_out + shared_out
        return final.view(orig_shape)

    cls.__init__ = _init_with_mega_moe
    cls.forward = _forward_mega_moe
    cls._omni_mega_moe_patched = True  # type: ignore[attr-defined]

    _log(
        "HunYuanSparseMoeBlock patched with MegaMoE takeover "
        "(bf16 no-quant, shared_mlp serial, SymmBuffer shared across all layers)."
    )


# ------------------------------------------------------------------
# 6.4) 挂到 _apply_patches：DiT transformer 加载好之后一并触发
# ------------------------------------------------------------------
_orig_apply_patches = _apply_patches  # type: ignore[has-type]


def _apply_patches_with_mega_moe(mod_transformer) -> None:  # type: ignore[no-redef]
    _orig_apply_patches(mod_transformer)
    try:
        _patch_moe_block_with_mega_moe(mod_transformer)
    except Exception as e:  # pragma: no cover
        _log(f"MegaMoE takeover failed: {e!r}")


_apply_patches = _apply_patches_with_mega_moe  # type: ignore[assignment]


# ------------------------------------------------------------------
# 6.5) 触发 DiT class 补丁
# ------------------------------------------------------------------
# 此时 _apply_patches 已被 wrap 成 _apply_patches_with_mega_moe，
# import-hook 里 `_apply_patches(m2)` 的运行时全局查找会拿到新版本，
# DiT MoeBlock 的 mega_moe 接管一并生效。
if not _try_apply_now():
    _install_import_hook()
