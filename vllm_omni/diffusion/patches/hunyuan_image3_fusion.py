# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage3 的额外 Ascend 平台补丁。

模型内已经正式支持 RoPE Q/K、cos/sin 预展开、压缩 KV 和 Add+RMSNorm，
这里不再重复替换对应的模型方法，只保留两项独立能力：

* 为 vLLM 的 ``SiluAndMul`` 补充 Ascend ``npu_swiglu`` 实现；
* 可选的 HunyuanImage3 MegaMoE 接管。

环境变量：
  DIT_FUSE_SWIGLU       默认 1（开）
  DIT_ENABLE_MEGA_MOE   默认 0（关）
  DIT_FUSE_LOG          默认 1（打印生效日志）
"""

from __future__ import annotations

import os
import sys


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip() not in ("0", "", "false", "False")


FUSE_SWIGLU = _flag("DIT_FUSE_SWIGLU")
# CANN MegaMoE：把 MoE 的 dispatch / grouped-matmul / routing / combine 一次融合。
# 仅在显式设置 DIT_ENABLE_MEGA_MOE=1、torch_npu 可用且已安装
# npu_ops_transformer 时接管 HunYuanSparseMoeBlock；默认关闭。
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
# SwiGLU 融合 (SiluAndMul -> npu_swiglu)
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
# 都会得到融合，HunyuanImage3 的普通 MLP 和 shared expert 因而共用同一条
# 清晰的算子级实现，不再额外替换各自的 ``forward``。
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


_patch_silu_and_mul_forward_oot()


# ==================================================================
# 可选 NPU MegaMoE：把 MoE 的 dispatch / GEMM / SwiGLU / GEMM / combine
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
# 1) npu_ops_transformer 探测
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
# 2) 修复 vllm-ascend 的 is_megamoe_supported_by_config（**必须始终执行**）
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
# 3) 全局 SymmBuffer：跨 32 层共享
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
# 4) 核心 patch：接管 HunYuanSparseMoeBlock.forward
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


def apply_hunyuan_image3_patches(mod_transformer) -> None:
    """在 HunyuanImage3 模块加载完成后安装可选的模型专用补丁。"""
    try:
        _patch_moe_block_with_mega_moe(mod_transformer)
    except Exception as e:  # pragma: no cover
        _log(f"MegaMoE takeover failed: {e!r}")


__all__ = ["apply_hunyuan_image3_patches"]
