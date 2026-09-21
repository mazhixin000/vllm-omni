# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPUGraph (ACL graph) support for the HunyuanImage3 SigLIP2 ViT.

整块功能以 **monkey patch** 形式注入，不修改
``vllm_omni/model_executor/models/hunyuan_image3/siglip2.py`` 与
``vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py``
的源码，与 0918 分支既有的 patch 组织方式保持一致
（``platforms/npu/models/hunyuan_image3.py``、``diffusion/patches/*``）。

图只捕获 SigLIP2 的静态 packed 核心（patch embedding + position embedding +
encoder + post layernorm），动态的 pack / unpack、aligner、prompt scatter
仍走普通 eager 路径。

开关（全部走 stage 的 ``additional_config``，默认全关）：
  hunyuan_vit_aclgraph                  总开关
  hunyuan_vit_aclgraph_strict           禁止回退 eager，失败直接抛错
  hunyuan_vit_aclgraph_lazy_capture     运行期按需捕获（默认 true）
  hunyuan_vit_aclgraph_warmups          捕获前 warmup 次数
  hunyuan_vit_aclgraph_max_graphs       图数量上限
  hunyuan_vit_aclgraph_dedicated_stream 使用独立 stream 捕获/回放
  hunyuan_vit_aclgraph_capture_layouts  预捕获 layout 列表
  hunyuan_vit_bucket_pad                预处理阶段把 ViT 输入 pad 到固定桶
  hunyuan_vit_aclgraph_add_layer_norm   图模式下是否保留 npu_add_layer_norm
                                        融合算子（auto=启动期探测，1=强制开，0=强制关）
"""

from __future__ import annotations

import builtins
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from vllm.logger import init_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vllm_omni.model_executor.models.hunyuan_image3.siglip2 import Siglip2VisionTransformer

logger = init_logger(__name__)

_TARGET_MODULE = "vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3"


# =============================================================================
# Layout / config dataclasses
# =============================================================================


@dataclass(frozen=True)
class Siglip2GraphLayout:
    batch_size: int
    max_patches: int
    patch_dim: int
    token_count: int
    spatial_shapes: tuple[tuple[int, int], ...]
    max_seqlen: int


@dataclass
class Siglip2PreparedGraphInputs:
    packed_pixels: torch.Tensor
    packed_position_embeddings: torch.Tensor
    cu_seqlens: torch.Tensor
    unpack_indices: torch.Tensor
    output_shape: tuple[int, int, int]
    layout: Siglip2GraphLayout


@dataclass(frozen=True)
class HunyuanImage3VitGraphKey:
    batch_size: int
    max_patches: int
    patch_dim: int
    token_count: int
    spatial_shapes: tuple[tuple[int, int], ...]
    input_dtype: torch.dtype
    model_dtype: torch.dtype
    device_index: int
    attention_backend: str
    # 图内容取决于编码器里走的是融合的 Add+LayerNorm 还是原生 add + nn.LayerNorm，
    # 两者不能共用同一张图，因此必须进 key。
    add_layer_norm: bool


@dataclass
class HunyuanImage3VitGraphConfig:
    enabled: bool = False
    strict: bool = False
    lazy_capture: bool = False
    warmups: int = 3
    max_graphs: int = 8
    capture_layouts: tuple[Siglip2GraphLayout, ...] = ()
    use_dedicated_stream: bool = True
    # None => auto：启动期用一次迷你图捕获探测后决定。
    fused_add_layer_norm: bool | None = None

    @staticmethod
    def parse_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return default

    @staticmethod
    def parse_optional_bool(value: Any) -> bool | None:
        """Parse a tri-state switch: ``True``/``False``，``None`` 表示 auto。"""
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "auto", "default"}:
            return None
        return HunyuanImage3VitGraphConfig.parse_bool(value, True)

    @staticmethod
    def parse_int(value: Any, default: int) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def parse_spatial_shapes(value: Any) -> tuple[tuple[int, int], ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            shapes: list[tuple[int, int]] = []
            for chunk in value.split(";"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                sep = "x" if "x" in chunk else ","
                h_str, w_str = chunk.split(sep, 1)
                shapes.append((int(h_str), int(w_str)))
            return tuple(shapes)
        if isinstance(value, Sequence):
            shapes = []
            for item in value:
                if isinstance(item, str):
                    sep = "x" if "x" in item else ","
                    h_str, w_str = item.split(sep, 1)
                    shapes.append((int(h_str), int(w_str)))
                else:
                    shapes.append((int(item[0]), int(item[1])))
            return tuple(shapes)
        return ()

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None, vit_config: Any) -> HunyuanImage3VitGraphConfig:
        config = config or {}
        patch_size = int(getattr(vit_config, "patch_size", 16) or 16)
        num_channels = int(getattr(vit_config, "num_channels", 3) or 3)
        patch_dim = num_channels * patch_size * patch_size
        default_max_patches = int(getattr(vit_config, "num_patches", 1024) or 1024)

        layouts: list[Siglip2GraphLayout] = []
        for item in config.get("hunyuan_vit_aclgraph_capture_layouts", ()) or ():
            if not isinstance(item, Mapping):
                continue
            spatial_shapes = cls.parse_spatial_shapes(item.get("spatial_shapes"))
            if not spatial_shapes:
                continue
            batch_size = int(item.get("batch_size", len(spatial_shapes)))
            if batch_size != len(spatial_shapes):
                logger.warning(
                    "Ignoring HunyuanImage3 ViT graph layout with batch_size=%s but %s spatial shapes.",
                    batch_size,
                    len(spatial_shapes),
                )
                continue
            token_count = sum(h * w for h, w in spatial_shapes)
            max_seqlen = max(h * w for h, w in spatial_shapes)
            layouts.append(
                Siglip2GraphLayout(
                    batch_size=batch_size,
                    max_patches=int(item.get("max_patches", default_max_patches)),
                    patch_dim=int(item.get("patch_dim", patch_dim)),
                    token_count=int(item.get("token_count", token_count)),
                    spatial_shapes=spatial_shapes,
                    max_seqlen=int(item.get("max_seqlen", max_seqlen)),
                )
            )

        return cls(
            enabled=cls.parse_bool(config.get("hunyuan_vit_aclgraph")),
            strict=cls.parse_bool(config.get("hunyuan_vit_aclgraph_strict")),
            lazy_capture=cls.parse_bool(config.get("hunyuan_vit_aclgraph_lazy_capture")),
            warmups=max(0, cls.parse_int(config.get("hunyuan_vit_aclgraph_warmups"), 3)),
            max_graphs=max(0, cls.parse_int(config.get("hunyuan_vit_aclgraph_max_graphs"), 8)),
            capture_layouts=tuple(layouts),
            use_dedicated_stream=cls.parse_bool(config.get("hunyuan_vit_aclgraph_dedicated_stream"), True),
            fused_add_layer_norm=cls.parse_optional_bool(config.get("hunyuan_vit_aclgraph_add_layer_norm")),
        )


@dataclass
class HunyuanImage3VitGraphEntry:
    key: HunyuanImage3VitGraphKey
    graph: Any
    static_packed_pixels: torch.Tensor
    static_position_embeddings: torch.Tensor
    static_cu_seqlens: torch.Tensor
    static_output: torch.Tensor
    stream: Any
    replay_done_event: Any
    consume_done_event: Any | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    capture_time_ms: float = 0.0
    memory_bytes: int = 0
    replay_count: int = 0
    disabled_reason: str | None = None
    poisoned_reason: str | None = None


# =============================================================================
# Graph manager
# =============================================================================


class HunyuanImage3VitAclGraphManager:
    """Capture/replay exact-layout HunyuanImage3 ViT NPUGraphs."""

    def __init__(
        self,
        vision_model: Siglip2VisionTransformer,
        config: HunyuanImage3VitGraphConfig,
        device: torch.device,
    ) -> None:
        self.vision_model = vision_model
        self.config = config
        self.device = device
        self._graphs: dict[HunyuanImage3VitGraphKey, HunyuanImage3VitGraphEntry] = {}
        self._disabled: dict[HunyuanImage3VitGraphKey, str] = {}
        self._position_cache: dict[
            tuple[tuple[tuple[int, int], ...], torch.dtype, tuple[str, int | None]],
            torch.Tensor,
        ] = {}
        self._capture_lock = threading.Lock()
        self._pool = self._get_graph_pool()
        self._stats: dict[str, Any] = {
            "capture_count": 0,
            "capture_time_ms": 0.0,
            "replay_hits": 0,
            "replay_misses": 0,
            "eager_fallbacks": 0,
            "misses_by_reason": {},
        }

    @classmethod
    def from_od_config(
        cls,
        vision_model: Siglip2VisionTransformer,
        od_config: Any,
        device: torch.device,
    ) -> HunyuanImage3VitAclGraphManager | None:
        extra = getattr(od_config, "additional_config", {}) or {}
        cfg = HunyuanImage3VitGraphConfig.from_mapping(extra, getattr(vision_model, "config", None))
        if not cfg.enabled:
            return None
        return cls(vision_model=vision_model, config=cfg, device=device)

    # ------------------------------------------------------------------ #
    # NPU stream / graph helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _npu_graph_available() -> bool:
        npu = getattr(torch, "npu", None)
        return npu is not None and hasattr(npu, "NPUGraph") and hasattr(npu, "graph")

    @staticmethod
    def _get_graph_pool() -> Any | None:
        npu = getattr(torch, "npu", None)
        if npu is None or not hasattr(npu, "graph_pool_handle"):
            return None
        try:
            return npu.graph_pool_handle()
        except Exception as exc:  # noqa: BLE001
            logger.warning("HunyuanImage3 ViT NPUGraph pool creation failed: %s", exc)
            return None

    @staticmethod
    def _current_stream() -> Any:
        try:
            return torch.npu.current_stream()
        except TypeError:
            return torch.npu.current_stream(None)

    @staticmethod
    def _new_stream(device: torch.device) -> Any:
        try:
            return torch.npu.Stream(device=device)
        except TypeError:
            return torch.npu.Stream()

    @staticmethod
    def _record_event(stream: Any | None = None) -> Any:
        event = torch.npu.Event()
        if stream is None:
            event.record()
            return event
        try:
            event.record(stream)
        except TypeError:
            with torch.npu.stream(stream):
                event.record()
        return event

    @staticmethod
    def _wait_event(stream: Any, event: Any) -> None:
        if event is None:
            return
        if hasattr(stream, "wait_event"):
            stream.wait_event(event)
        else:
            event.wait(stream)

    def _wait_stream(self, stream: Any, wait_for: Any) -> None:
        if hasattr(stream, "wait_stream"):
            stream.wait_stream(wait_for)
            return
        with torch.npu.stream(wait_for):
            event = self._record_event()
        self._wait_event(stream, event)

    # ------------------------------------------------------------------ #
    # Stats / fallback
    # ------------------------------------------------------------------ #

    def _note_miss(self, reason: str) -> None:
        self._stats["replay_misses"] += 1
        misses = self._stats["misses_by_reason"]
        misses[reason] = misses.get(reason, 0) + 1

    def _fallback(
        self,
        prepared: Siglip2PreparedGraphInputs,
        reason: str,
        exc: Exception | None = None,
    ) -> torch.Tensor:
        self._note_miss(reason)
        self._stats["eager_fallbacks"] += 1
        if self.config.strict:
            if exc is not None:
                raise RuntimeError(f"HunyuanImage3 ViT NPUGraph fallback is forbidden: {reason}") from exc
            raise RuntimeError(f"HunyuanImage3 ViT NPUGraph fallback is forbidden: {reason}")
        if exc is not None:
            logger.warning_once("HunyuanImage3 ViT NPUGraph fallback (%s): %s", reason, exc)
        else:
            logger.debug("HunyuanImage3 ViT NPUGraph fallback: %s", reason)
        return self._eager(prepared)

    # ------------------------------------------------------------------ #
    # Input preparation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _spatial_shapes_tuple(spatial_shapes: torch.Tensor) -> tuple[tuple[int, int], ...]:
        shapes = spatial_shapes.detach().cpu().tolist()
        return tuple((int(shape[0]), int(shape[1])) for shape in shapes)

    def _get_packed_position_embeddings(
        self,
        spatial_shapes: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        embeddings = self.vision_model.embeddings
        shapes = self._spatial_shapes_tuple(spatial_shapes)
        cache_key = (shapes, dtype, (device.type, device.index))
        cached = self._position_cache.get(cache_key)
        if cached is not None:
            return cached

        positional_embeddings = embeddings.position_embedding.weight.reshape(
            embeddings.position_embedding_size,
            embeddings.position_embedding_size,
            -1,
        )
        pe_for_resize = positional_embeddings.permute(2, 0, 1).unsqueeze(0)
        if pe_for_resize.device.type == "cpu":
            pe_for_resize = pe_for_resize.to(torch.float32)

        position_embs: list[torch.Tensor] = []
        for height, width in shapes:
            resized = F.interpolate(
                pe_for_resize,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            resized = resized.reshape(embeddings.embed_dim, height * width).transpose(0, 1)
            position_embs.append(resized.to(device=device, dtype=dtype))

        packed_position_embs = torch.cat(position_embs, dim=0).contiguous()
        self._position_cache[cache_key] = packed_position_embs
        return packed_position_embs

    def prepare_graph_inputs(
        self,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> Siglip2PreparedGraphInputs:
        batch_size, max_patches, patch_dim = pixel_values.shape
        mask_bool = attention_mask.bool()
        packed_pixels = pixel_values[mask_bool].contiguous()
        seq_lens = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).to(torch.int32)

        token_count = int(seq_lens.sum().item())
        if token_count != int(packed_pixels.shape[0]):
            raise ValueError(
                "SigLIP2 attention_mask and spatial_shapes disagree: "
                f"mask tokens={int(packed_pixels.shape[0])}, spatial tokens={token_count}."
            )

        spatial_shapes_tuple = self._spatial_shapes_tuple(spatial_shapes)
        cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=pixel_values.device)
        cu_seqlens[1:] = seq_lens.cumsum(0)
        max_seqlen = max((height * width for height, width in spatial_shapes_tuple), default=0)

        layout = Siglip2GraphLayout(
            batch_size=batch_size,
            max_patches=max_patches,
            patch_dim=patch_dim,
            token_count=token_count,
            spatial_shapes=spatial_shapes_tuple,
            max_seqlen=max_seqlen,
        )

        packed_position_embeddings = self._get_packed_position_embeddings(
            spatial_shapes,
            dtype=self.vision_model.embeddings.patch_embedding.weight.dtype,
            device=pixel_values.device,
        )

        unpack_indices = mask_bool.reshape(-1).nonzero(as_tuple=False).reshape(-1).to(torch.long)
        return Siglip2PreparedGraphInputs(
            packed_pixels=packed_pixels,
            packed_position_embeddings=packed_position_embeddings,
            cu_seqlens=cu_seqlens,
            unpack_indices=unpack_indices,
            output_shape=(batch_size, max_patches, self.vision_model.embed_dim),
            layout=layout,
        )

    # ------------------------------------------------------------------ #
    # Captured region
    # ------------------------------------------------------------------ #

    def graph_forward(
        self,
        packed_pixels: torch.Tensor,
        packed_position_embeddings: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Static core that is captured into the NPUGraph."""
        target_dtype = self.vision_model.embeddings.patch_embedding.weight.dtype
        hidden_states = self.vision_model.embeddings.patch_embedding(packed_pixels.to(dtype=target_dtype))
        hidden_states = hidden_states + packed_position_embeddings.to(
            dtype=target_dtype,
            device=hidden_states.device,
        )
        hidden_states = self.vision_model.encoder(hidden_states, cu_seqlens)
        return self.vision_model.post_layernorm(hidden_states)

    def unpack_graph_output(
        self,
        packed_output: torch.Tensor,
        prepared: Siglip2PreparedGraphInputs,
    ) -> torch.Tensor:
        output = packed_output.new_zeros(prepared.output_shape)
        output.reshape(-1, self.vision_model.embed_dim).index_copy_(0, prepared.unpack_indices, packed_output)
        return output

    def _eager(self, prepared: Siglip2PreparedGraphInputs) -> torch.Tensor:
        packed_output = self.graph_forward(
            prepared.packed_pixels,
            prepared.packed_position_embeddings,
            prepared.cu_seqlens,
        )
        return self.unpack_graph_output(packed_output, prepared)

    # ------------------------------------------------------------------ #
    # Entry points
    # ------------------------------------------------------------------ #

    def execute(
        self,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(pixel_values.shape[0])
        if batch_size <= 1:
            return self.execute_padded(self.prepare_graph_inputs(pixel_values, attention_mask, spatial_shapes))

        outputs = []
        for row in range(batch_size):
            prepared = self.prepare_graph_inputs(
                pixel_values[row : row + 1],
                attention_mask[row : row + 1],
                spatial_shapes[row : row + 1],
            )
            outputs.append(self.execute_padded(prepared))
        return torch.cat(outputs, dim=0)

    def prepare_key(self, prepared: Siglip2PreparedGraphInputs) -> HunyuanImage3VitGraphKey:
        device_index = prepared.packed_pixels.device.index
        if device_index is None:
            try:
                device_index = int(torch.npu.current_device())
            except Exception:  # noqa: BLE001
                device_index = -1
        model_dtype = next(self.vision_model.parameters()).dtype
        attention_backend = "npu_fia_single" if prepared.layout.batch_size == 1 else "mm_encoder_attention"
        return HunyuanImage3VitGraphKey(
            batch_size=prepared.layout.batch_size,
            max_patches=prepared.layout.max_patches,
            patch_dim=prepared.layout.patch_dim,
            token_count=prepared.layout.token_count,
            spatial_shapes=prepared.layout.spatial_shapes,
            input_dtype=prepared.packed_pixels.dtype,
            model_dtype=model_dtype,
            device_index=int(device_index),
            attention_backend=attention_backend,
            add_layer_norm=is_vit_graph_fused_add_layer_norm_enabled(),
        )

    def _validate_supported(self, prepared: Siglip2PreparedGraphInputs) -> str | None:
        if not self.config.enabled:
            return "disabled"
        if prepared.packed_pixels.device.type != "npu":
            return f"device={prepared.packed_pixels.device.type}"
        if not self._npu_graph_available():
            return "torch.npu.NPUGraph is unavailable"
        if prepared.layout.batch_size != 1:
            return "only batch_size=1 is supported in the first ViT graph path"
        if len(prepared.layout.spatial_shapes) != 1:
            return "only single-image layouts are supported in the first ViT graph path"
        if prepared.layout.token_count <= 0:
            return "empty ViT token layout"
        return None

    @torch.inference_mode()
    def capture(self, prepared: Siglip2PreparedGraphInputs) -> HunyuanImage3VitGraphEntry:
        key = self.prepare_key(prepared)
        unsupported = self._validate_supported(prepared)
        if unsupported is not None:
            raise RuntimeError(unsupported)

        with self._capture_lock:
            existing = self._graphs.get(key)
            if existing is not None:
                return existing
            if key in self._disabled:
                raise RuntimeError(self._disabled[key])
            if len(self._graphs) >= self.config.max_graphs:
                reason = f"max_graphs={self.config.max_graphs} reached"
                self._disabled[key] = reason
                raise RuntimeError(reason)

            try:
                return self._capture_layout(prepared, key)
            except Exception as exc:  # noqa: BLE001
                reason = str(exc) or type(exc).__name__
                # 探测通过但真实捕获仍失败时，关掉融合的 Add+LayerNorm 并用原生路径重试
                # 一次。只有在一张图都还没捕获成功时才允许切换，避免新旧图混用两套 kernel。
                if not self._graphs and _turn_off_vit_graph_fused_add_layer_norm_on_failure(reason):
                    retry_key = self.prepare_key(prepared)
                    try:
                        return self._capture_layout(prepared, retry_key)
                    except Exception as retry_exc:  # noqa: BLE001
                        self._disabled[retry_key] = str(retry_exc) or type(retry_exc).__name__
                        raise
                self._disabled[key] = reason
                raise

    @torch.inference_mode()
    def _capture_layout(
        self,
        prepared: Siglip2PreparedGraphInputs,
        key: HunyuanImage3VitGraphKey,
    ) -> HunyuanImage3VitGraphEntry:
        static_packed_pixels = torch.empty_like(prepared.packed_pixels)
        static_position_embeddings = prepared.packed_position_embeddings.detach().clone()
        static_cu_seqlens = prepared.cu_seqlens.detach().clone()
        stream = self._new_stream(self.device) if self.config.use_dedicated_stream else self._current_stream()
        current_stream = self._current_stream()

        start = time.perf_counter()
        static_output = None
        try:
            self._wait_stream(stream, current_stream)
            with torch.npu.stream(stream):
                static_packed_pixels.copy_(prepared.packed_pixels, non_blocking=True)
                for _ in range(self.config.warmups):
                    static_output = self.graph_forward(
                        static_packed_pixels,
                        static_position_embeddings,
                        static_cu_seqlens,
                    )
                stream.synchronize()

                graph = torch.npu.NPUGraph()
                if self._pool is None:
                    with torch.npu.graph(graph):
                        static_output = self.graph_forward(
                            static_packed_pixels,
                            static_position_embeddings,
                            static_cu_seqlens,
                        )
                else:
                    with torch.npu.graph(graph, pool=self._pool):
                        static_output = self.graph_forward(
                            static_packed_pixels,
                            static_position_embeddings,
                            static_cu_seqlens,
                        )
                replay_done_event = self._record_event(stream)
            self._wait_stream(current_stream, stream)
        except Exception as exc:  # noqa: BLE001
            logger.debug("HunyuanImage3 ViT NPUGraph capture failed for key=%s: %s", key, exc)
            raise

        if static_output is None:
            raise RuntimeError("capture produced no output")

        capture_time_ms = (time.perf_counter() - start) * 1000.0
        memory_bytes = (
            static_packed_pixels.numel() * static_packed_pixels.element_size()
            + static_position_embeddings.numel() * static_position_embeddings.element_size()
            + static_cu_seqlens.numel() * static_cu_seqlens.element_size()
            + static_output.numel() * static_output.element_size()
        )
        entry = HunyuanImage3VitGraphEntry(
            key=key,
            graph=graph,
            static_packed_pixels=static_packed_pixels,
            static_position_embeddings=static_position_embeddings,
            static_cu_seqlens=static_cu_seqlens,
            static_output=static_output,
            stream=stream,
            replay_done_event=replay_done_event,
            capture_time_ms=capture_time_ms,
            memory_bytes=memory_bytes,
        )
        self._graphs[key] = entry
        self._stats["capture_count"] += 1
        self._stats["capture_time_ms"] += capture_time_ms
        logger.info(
            "Captured HunyuanImage3 ViT NPUGraph key=%s in %.2f ms (static buffers %.2f MiB).",
            key,
            capture_time_ms,
            memory_bytes / (1024**2),
        )
        return entry

    @torch.inference_mode()
    def execute_padded(self, prepared: Siglip2PreparedGraphInputs) -> torch.Tensor:
        unsupported = self._validate_supported(prepared)
        if unsupported is not None:
            return self._fallback(prepared, unsupported)

        key = self.prepare_key(prepared)
        entry = self._graphs.get(key)
        if entry is None and self.config.lazy_capture:
            try:
                entry = self.capture(prepared)
            except Exception as exc:  # noqa: BLE001
                return self._fallback(prepared, "capture failed", exc)
        if entry is None:
            reason = self._disabled.get(key, "graph key miss")
            return self._fallback(prepared, reason)
        if entry.disabled_reason or entry.poisoned_reason:
            return self._fallback(prepared, entry.disabled_reason or entry.poisoned_reason or "disabled graph")

        with entry.lock:
            current_stream = self._current_stream()
            try:
                if entry.consume_done_event is not None:
                    self._wait_event(entry.stream, entry.consume_done_event)
                self._wait_stream(entry.stream, current_stream)
                with torch.npu.stream(entry.stream):
                    entry.static_packed_pixels.copy_(prepared.packed_pixels, non_blocking=True)
                    entry.graph.replay()
                    entry.replay_done_event = self._record_event(entry.stream)
                self._wait_event(current_stream, entry.replay_done_event)
                output = self.unpack_graph_output(entry.static_output, prepared)
                entry.consume_done_event = self._record_event(current_stream)
                entry.replay_count += 1
                self._stats["replay_hits"] += 1
                return output
            except Exception as exc:  # noqa: BLE001
                entry.poisoned_reason = str(exc) or type(exc).__name__
                return self._fallback(prepared, "replay failed", exc)

    @torch.inference_mode()
    def capture_configured_graphs(self) -> None:
        if not self.config.enabled or not self.config.capture_layouts:
            return
        if not self._npu_graph_available() or self.device.type != "npu":
            if self.config.strict:
                raise RuntimeError("HunyuanImage3 ViT NPUGraph requested on unsupported platform.")
            logger.info("HunyuanImage3 ViT NPUGraph pre-capture skipped on device %s.", self.device)
            return

        model_dtype = next(self.vision_model.parameters()).dtype
        for layout in sorted(self.config.capture_layouts, key=lambda item: item.token_count, reverse=True):
            spatial_shapes = torch.tensor(layout.spatial_shapes, dtype=torch.long, device=self.device)
            pixel_values = torch.zeros(
                (layout.batch_size, layout.max_patches, layout.patch_dim),
                dtype=torch.float32,
                device=self.device,
            )
            attention_mask = torch.zeros(
                (layout.batch_size, layout.max_patches),
                dtype=torch.long,
                device=self.device,
            )
            for row, (height, width) in enumerate(layout.spatial_shapes):
                attention_mask[row, : height * width] = 1
            prepared = self.prepare_graph_inputs(pixel_values, attention_mask, spatial_shapes)
            try:
                self.capture(prepared)
            except Exception as exc:  # noqa: BLE001
                if self.config.strict:
                    raise
                logger.warning("Skipping HunyuanImage3 ViT NPUGraph layout=%s: %s", layout, exc)
        logger.info(
            "HunyuanImage3 ViT NPUGraph stats after pre-capture: %s (model_dtype=%s)",
            self.stats(),
            model_dtype,
        )

    def clear(self) -> None:
        with self._capture_lock:
            self._graphs.clear()
            self._disabled.clear()
            self._position_cache.clear()
            self._pool = self._get_graph_pool()
            logger.info("Cleared HunyuanImage3 ViT NPUGraph cache.")

    def stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "graph_count": len(self._graphs),
            "disabled_count": len(self._disabled),
            "keys": [str(key) for key in self._graphs],
            "disabled": {str(key): reason for key, reason in self._disabled.items()},
            "fused_add_layer_norm": vit_graph_fused_add_layer_norm_state(),
        }


# =============================================================================
# SigLIP2 融合算子在 ACL graph 下的取舍（patch，不改 siglip2.py 源码）
# =============================================================================
# siglip2.py 里 ``npu_fused_infer_attention_score`` / ``npu_add_layer_norm``
# 两条融合路径由 ``_get_hunyuan_image3_vit_npu_op`` 统一取算子（并由
# VLLM_OMNI_HUNYUAN_IMAGE3_FUSED_VIT 控制）。图模式下能否保留，取决于它能否被
# ACL graph 捕获：
#
# - ``npu_fused_infer_attention_score``：不能降级。降级后 SigLIP2 会改走
#   ``MMEncoderAttention`` 的 varlen 路径，而 NPU 上的 Ascend 实现（不在 vllm-ascend
#   的 encoder graph 上下文中时会走 ``_forward_eager_fia``）需要先把 cu_seqlens 拷回
#   host、再以 host 侧 actual_seq_lengths 调用 FIA，捕获期间这一来一回都会触发
#   ``aclrtMemcpy: stream is captured``，导致 ViT 图全部捕获失败并永久回退 eager
#   （实测 3 张参考图 100ms 而非图模式的 ~20ms）。融合分支在 batch=1 时走 BSND
#   全注意力，完全不接触 cu_seqlens，是静态可捕获的。
# - ``npu_add_layer_norm``：以前是一律降级（它曾对 packed 输入引入非静态 shape）。
#   现在改为**启动期探测**：用静态 buffer 走一次「warmup + 迷你图捕获 + replay」，
#   能捕获就保留融合实现（省掉一次 residual add kernel 与 LayerNorm 的额外读写），
#   不能捕获才降级成原生 ``add + nn.LayerNorm``。可用
#   ``hunyuan_vit_aclgraph_add_layer_norm`` 强制 1/0/auto。
#
# 探测结论不是最终保证：真实捕获若因该算子失败，且此时还没有任何已捕获的图，
# 会自动关掉它并用原生路径重试一次，避免所有 layout 被一次性打进黑名单。

_VIT_GRAPH_ACTIVE = False
_VIT_GRAPH_FUSED_ADD_LAYER_NORM = False
_VIT_GRAPH_ADD_LAYER_NORM_REASON = "not evaluated"

_GRAPH_SAFE_FUSED_OPS = frozenset({"npu_fused_infer_attention_score"})
_ADD_LAYER_NORM_OP = "npu_add_layer_norm"
_ORIGINAL_GET_VIT_NPU_OP: Callable[[str], Callable[..., object] | None] | None = None


def _patch_siglip2_graph_friendly_ops() -> None:
    """Route SigLIP2's fused operators according to ACL graph capture support.

    安装后 ``_VIT_GRAPH_ACTIVE`` 仍为 False 时等价于透传，因此可以先装 hook、
    再探测算子、最后才激活图模式。
    """
    from vllm_omni.model_executor.models.hunyuan_image3 import siglip2

    global _ORIGINAL_GET_VIT_NPU_OP

    if getattr(siglip2, "_omni_vit_graph_ops_patched", False):
        return

    original_get_op = siglip2._get_hunyuan_image3_vit_npu_op
    _ORIGINAL_GET_VIT_NPU_OP = original_get_op

    def _get_hunyuan_image3_vit_npu_op(name: str):
        if _VIT_GRAPH_ACTIVE:
            if name in _GRAPH_SAFE_FUSED_OPS:
                return original_get_op(name)
            if name == _ADD_LAYER_NORM_OP and _VIT_GRAPH_FUSED_ADD_LAYER_NORM:
                return original_get_op(name)
            return None
        return original_get_op(name)

    siglip2._get_hunyuan_image3_vit_npu_op = _get_hunyuan_image3_vit_npu_op
    siglip2._omni_vit_graph_ops_patched = True


def _set_vit_graph_fused_add_layer_norm(allowed: bool, reason: str) -> None:
    global _VIT_GRAPH_FUSED_ADD_LAYER_NORM, _VIT_GRAPH_ADD_LAYER_NORM_REASON
    _VIT_GRAPH_FUSED_ADD_LAYER_NORM = bool(allowed)
    _VIT_GRAPH_ADD_LAYER_NORM_REASON = reason


def is_vit_graph_fused_add_layer_norm_enabled() -> bool:
    """Return whether the fused Add+LayerNorm stays enabled inside the ViT graph."""
    return _VIT_GRAPH_FUSED_ADD_LAYER_NORM


def vit_graph_fused_add_layer_norm_state() -> dict[str, Any]:
    """Expose the Add+LayerNorm decision for logging and ``stats()``."""
    return {
        "operator": _ADD_LAYER_NORM_OP,
        "enabled": _VIT_GRAPH_FUSED_ADD_LAYER_NORM,
        "reason": _VIT_GRAPH_ADD_LAYER_NORM_REASON,
    }


def _turn_off_vit_graph_fused_add_layer_norm_on_failure(reason: str) -> bool:
    """Disable the fused operator after a capture failure.

    Returns ``True`` only when it was enabled and has just been switched off,
    i.e. when a capture retry with the native path makes sense.
    """
    if not _VIT_GRAPH_FUSED_ADD_LAYER_NORM:
        return False
    _set_vit_graph_fused_add_layer_norm(False, f"disabled after capture failure: {reason}")
    logger.warning(
        "HunyuanImage3 ViT NPUGraph turned off %s because a graph capture failed (%s); "
        "retrying with the native add + LayerNorm path.",
        _ADD_LAYER_NORM_OP,
        reason,
    )
    return True


@torch.inference_mode()
def _probe_fused_add_layer_norm(
    fused_add_layer_norm: Callable[..., object],
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden_size: int,
    token_count: int = 64,
    epsilon: float = 1e-6,
    warmups: int = 1,
) -> str | None:
    """Return ``None`` when the fused op can be captured, otherwise the reason.

    探测用一张一次性的迷你 NPUGraph 完成：静态 buffer 上先 warmup，再尝试捕获并
    replay。任何异常（典型是算子内部触发 host 回读导致的
    ``aclrtMemcpy: stream is captured``）都视为不可入图。
    """
    manager_cls = HunyuanImage3VitAclGraphManager
    if not manager_cls._npu_graph_available():
        return "torch.npu.NPUGraph is unavailable"

    outputs: Any = None
    try:
        x1 = torch.zeros((token_count, hidden_size), dtype=dtype, device=device)
        x2 = torch.zeros_like(x1)
        gamma = torch.ones(hidden_size, dtype=dtype, device=device)
        beta = torch.zeros(hidden_size, dtype=dtype, device=device)

        stream = manager_cls._new_stream(device)
        current_stream = manager_cls._current_stream()
        manager_cls._wait_stream(stream, current_stream)
        with torch.npu.stream(stream):
            for _ in range(max(0, warmups)):
                fused_add_layer_norm(x1, x2, gamma, beta, epsilon, True)
            stream.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                outputs = fused_add_layer_norm(x1, x2, gamma, beta, epsilon, True)
            graph.replay()
        manager_cls._wait_stream(current_stream, stream)
    except Exception as exc:  # noqa: BLE001
        # 探测图随局部变量一起释放，不占用图池显存。
        return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__

    if not isinstance(outputs, tuple) or len(outputs) < 4:
        return f"unexpected {_ADD_LAYER_NORM_OP} output signature: {type(outputs).__name__}"
    return None


def _resolve_vit_graph_fused_add_layer_norm(
    vision_model: Any,
    config: HunyuanImage3VitGraphConfig,
    device: torch.device,
) -> None:
    """Decide whether the fused Add+LayerNorm may stay enabled in graph mode."""
    if config.fused_add_layer_norm is False:
        _set_vit_graph_fused_add_layer_norm(False, "disabled by hunyuan_vit_aclgraph_add_layer_norm")
        return
    if _ORIGINAL_GET_VIT_NPU_OP is None:
        _set_vit_graph_fused_add_layer_norm(False, "SigLIP2 operator hook is not installed")
        return

    fused_add_layer_norm = _ORIGINAL_GET_VIT_NPU_OP(_ADD_LAYER_NORM_OP)
    if fused_add_layer_norm is None:
        _set_vit_graph_fused_add_layer_norm(
            False,
            f"{_ADD_LAYER_NORM_OP} unavailable (VLLM_OMNI_HUNYUAN_IMAGE3_FUSED_VIT off or torch-npu too old)",
        )
        return
    if config.fused_add_layer_norm is True:
        _set_vit_graph_fused_add_layer_norm(True, "forced on by hunyuan_vit_aclgraph_add_layer_norm")
        return

    hidden_size = int(getattr(vision_model, "embed_dim", 0) or 0)
    if hidden_size <= 0:
        _set_vit_graph_fused_add_layer_norm(False, "unknown ViT hidden size")
        return

    reason = _probe_fused_add_layer_norm(
        fused_add_layer_norm,
        device=device,
        dtype=next(vision_model.parameters()).dtype,
        hidden_size=hidden_size,
        epsilon=float(getattr(getattr(vision_model, "config", None), "layer_norm_eps", 1e-6) or 1e-6),
        warmups=config.warmups,
    )
    if reason is None:
        _set_vit_graph_fused_add_layer_norm(True, "capture probe succeeded")
    else:
        _set_vit_graph_fused_add_layer_norm(False, f"capture probe failed: {reason}")


# =============================================================================
# Pipeline monkey patches
# =============================================================================

_PATCHED = False
_HOOK_INSTALLED = False
_ORIGINAL_IMPORT = None
_ORIGINAL_PRE_PROCESS_FACTORY = None
_ORIGINAL_PIPELINE_INIT = None
_ORIGINAL_LOAD_WEIGHTS = None
_ORIGINAL_INSTANTIATE_VIT_IMAGE_TOKENS = None


def _resize_and_pad_center(image, target_width: int, target_height: int):
    """Resize保持长宽比后居中 pad 到目标尺寸，避免 ViT token 数随输入抖动。"""
    tw, th = target_width, target_height
    w, h = image.size
    scale = min(tw / w, th / h)
    resize_width = max(1, int(round(w * scale)))
    resize_height = max(1, int(round(h * scale)))
    resized = image.resize((resize_width, resize_height), PILImage.Resampling.LANCZOS)
    canvas = PILImage.new(image.mode, (tw, th), 0)
    left = (tw - resize_width) // 2
    top = (th - resize_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def _patched_pre_process_factory(od_config):
    """Wrap the upstream pre-process func with optional ViT bucket padding."""
    assert _ORIGINAL_PRE_PROCESS_FACTORY is not None
    base_pre_process = _ORIGINAL_PRE_PROCESS_FACTORY(od_config)
    extra_config = getattr(od_config, "additional_config", {}) or {}
    if not HunyuanImage3VitGraphConfig.parse_bool(extra_config.get("hunyuan_vit_bucket_pad"), False):
        return base_pre_process

    from vllm.transformers_utils.config import get_config

    from vllm_omni.diffusion.models.hunyuan_image3 import pipeline_hunyuan_image3 as pipeline_mod

    hf_config = get_config(od_config.model, trust_remote_code=True)
    image_processor = pipeline_mod.HunyuanImage3ImageProcessor(hf_config)
    vit_patch_size = getattr(image_processor.vision_encoder_processor, "patch_size", 1)
    if isinstance(vit_patch_size, tuple | list):
        vit_patch_size = int(vit_patch_size[0])

    def pre_process_func(request):
        request = base_pre_process(request)
        prompt = request.prompt
        if not isinstance(prompt, dict):
            return request

        multi_modal_data = prompt.get("multi_modal_data") or {}
        raw_images = multi_modal_data.get("image")
        if raw_images is None:
            raw_images = prompt.get("pil_image")
        if raw_images is None:
            return request

        image_list = raw_images if isinstance(raw_images, list) else [raw_images]
        additional_info = prompt.get("additional_information") or {}
        cond_infos = additional_info.get("batch_cond_image_info") or []
        if not cond_infos:
            return request

        for raw_image, cond_info in zip(image_list, cond_infos):
            if not isinstance(cond_info, dict):
                continue
            pil_image = pipeline_mod._to_pil_image(raw_image).convert("RGB")
            orig_width, orig_height = pil_image.size
            _, ratio_idx = image_processor.reso_group.get_base_size_and_ratio_index(orig_width, orig_height)
            reso = image_processor.reso_group[int(ratio_idx)]
            target_width = int(reso.width)
            target_height = int(reso.height)
            vit_input = _resize_and_pad_center(pil_image, target_width, target_height)
            vit_inputs = image_processor.vision_encoder_processor(vit_input, return_tensors="pt")
            vit_tensor = vit_inputs["pixel_values"]
            spatial_shapes = vit_inputs["spatial_shapes"].squeeze(0)
            pixel_attention_mask = vit_inputs["pixel_attention_mask"].squeeze(0)
            vit_token_h = int(spatial_shapes[0].item())
            vit_token_w = int(spatial_shapes[1].item())
            vit_info = pipeline_mod.ImageInfo(
                image_type="siglip2",
                image_tensor=vit_tensor,
                image_width=vit_token_w * vit_patch_size,
                image_height=vit_token_h * vit_patch_size,
                token_width=vit_token_w,
                token_height=vit_token_h,
                image_token_length=int(vit_tensor.shape[1]),
            )
            cond_info["vision_image_info"] = pipeline_mod._image_info_to_payload(vit_info)
            cond_info["vision_encoder_kwargs"] = {
                "spatial_shapes": spatial_shapes,
                "pixel_attention_mask": pixel_attention_mask,
            }
        return request

    return pre_process_func


def _patched_pipeline_init(self, *args, **kwargs) -> None:
    assert _ORIGINAL_PIPELINE_INIT is not None
    _ORIGINAL_PIPELINE_INIT(self, *args, **kwargs)
    self._vit_graph_manager = None
    self._vit_graph_init_attempted = False


def _patched_load_weights(self, weights):
    assert _ORIGINAL_LOAD_WEIGHTS is not None
    loaded_params = _ORIGINAL_LOAD_WEIGHTS(self, weights)
    self._ensure_vit_graph_manager()
    return loaded_params


def _vit_graph_enabled(self) -> bool:
    extra_config = getattr(self.od_config, "additional_config", {}) or {}
    return HunyuanImage3VitGraphConfig.parse_bool(extra_config.get("hunyuan_vit_aclgraph"), False)


def _ensure_vit_graph_manager(self):
    global _VIT_GRAPH_ACTIVE

    if getattr(self, "_vit_graph_manager", None) is not None:
        return self._vit_graph_manager
    if getattr(self, "_vit_graph_init_attempted", False) or not self._vit_graph_enabled():
        return None

    extra_config = getattr(self.od_config, "additional_config", {}) or {}
    strict = HunyuanImage3VitGraphConfig.parse_bool(extra_config.get("hunyuan_vit_aclgraph_strict"), False)
    try:
        device = next(self.vision_model.parameters()).device
        if device.type != "npu":
            if strict:
                raise RuntimeError(f"HunyuanImage3 ViT NPUGraph requires NPU, got {device}.")
            logger.info("HunyuanImage3 ViT NPUGraph disabled on non-NPU device %s.", device)
            return None
        if self.od_config.enable_cpu_offload or self.od_config.enable_layerwise_offload:
            if strict:
                raise RuntimeError("HunyuanImage3 ViT NPUGraph does not support offload mode.")
            logger.info("HunyuanImage3 ViT NPUGraph disabled because offload mode is enabled.")
            return None
        # ViT 张量并行时 RowParallelLinear 会插入集合通信，无法被 ACL graph 捕获。
        from vllm_omni.model_executor.models.hunyuan_image3.siglip2 import (
            is_hunyuan_image3_vit_data_parallel_enabled,
        )

        if not is_hunyuan_image3_vit_data_parallel_enabled():
            if strict:
                raise RuntimeError("HunyuanImage3 ViT NPUGraph requires VLLM_OMNI_HUNYUAN_IMAGE3_VIT_DATA_PARALLEL=1.")
            logger.info("HunyuanImage3 ViT NPUGraph disabled because ViT data parallel is off.")
            return None

        # 先装算子 hook（此时 _VIT_GRAPH_ACTIVE 仍为 False，等价于透传），再决定
        # Add+LayerNorm 能否入图，最后才激活图模式。
        _patch_siglip2_graph_friendly_ops()
        _resolve_vit_graph_fused_add_layer_norm(
            self.vision_model,
            HunyuanImage3VitGraphConfig.from_mapping(
                getattr(self.od_config, "additional_config", {}) or {},
                getattr(self.vision_model, "config", None),
            ),
            device,
        )
        logger.info(
            "HunyuanImage3 ViT NPUGraph fused Add+LayerNorm: %s",
            vit_graph_fused_add_layer_norm_state(),
        )
        _VIT_GRAPH_ACTIVE = True

        self._vit_graph_init_attempted = True
        manager = HunyuanImage3VitAclGraphManager.from_od_config(self.vision_model, self.od_config, device)
        if manager is None:
            return None
        self._vit_graph_manager = manager
        manager.capture_configured_graphs()
        return manager
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise
        logger.warning("HunyuanImage3 ViT NPUGraph initialization failed; using eager ViT: %s", exc)
        return None


def _clear_vit_graphs(self) -> None:
    manager = getattr(self, "_vit_graph_manager", None)
    if manager is not None:
        manager.clear()
        self._vit_graph_manager = None
    self._vit_graph_init_attempted = False


def _vit_encode(
    self,
    pixel_values: torch.Tensor,
    attention_mask: torch.Tensor,
    spatial_shapes: torch.Tensor,
) -> torch.Tensor:
    manager = self._ensure_vit_graph_manager()
    if manager is not None:
        return manager.execute(pixel_values, attention_mask=attention_mask, spatial_shapes=spatial_shapes)
    return self.vision_model(pixel_values, attention_mask=attention_mask, spatial_shapes=spatial_shapes)


def _patched_instantiate_vit_image_tokens(self, x, cond_vit_images, cond_vit_image_mask, vit_kwargs):
    cond_vit_image_embeds = []
    for batch_idx, image in enumerate(cond_vit_images):
        cur_kwargs = {k: v[batch_idx] for k, v in vit_kwargs.items()}
        image_embed = self._vit_encode(
            image,
            attention_mask=cur_kwargs["attention_mask"],
            spatial_shapes=cur_kwargs["spatial_shapes"],
        )
        image_embed = self.vision_aligner(image_embed)
        n, seq_len, dim = image_embed.shape
        image_embed = image_embed.reshape(n * seq_len, dim)
        cond_vit_image_embeds.append(image_embed)

    batch_size, seq_len, n_embd = x.shape
    index = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
    for i, (image_embed, mask) in enumerate(zip(cond_vit_image_embeds, cond_vit_image_mask)):
        image_scatter_index = index[i : i + 1].masked_select(mask.bool()).reshape(1, -1)
        x[i : i + 1].scatter_(
            dim=1,
            index=image_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
            src=image_embed.reshape(1, -1, n_embd),
        )
    return x


# =============================================================================
# Patch installation
# =============================================================================


def _apply_to_pipeline_module(pipeline_mod) -> None:
    global _PATCHED, _HOOK_INSTALLED, _ORIGINAL_IMPORT
    global _ORIGINAL_PRE_PROCESS_FACTORY, _ORIGINAL_PIPELINE_INIT
    global _ORIGINAL_LOAD_WEIGHTS, _ORIGINAL_INSTANTIATE_VIT_IMAGE_TOKENS

    if _PATCHED:
        return

    _ORIGINAL_PRE_PROCESS_FACTORY = pipeline_mod.get_hunyuan_image_3_pre_process_func
    pipeline_mod.get_hunyuan_image_3_pre_process_func = _patched_pre_process_factory

    pipeline_cls = pipeline_mod.HunyuanImage3Pipeline
    targets = list(getattr(pipeline_cls, "_PROFILER_TARGETS", []))
    if "_vit_encode" not in targets:
        targets.insert(1, "_vit_encode")
        pipeline_cls._PROFILER_TARGETS = targets

    _ORIGINAL_PIPELINE_INIT = pipeline_cls.__init__
    _ORIGINAL_LOAD_WEIGHTS = pipeline_cls.load_weights
    _ORIGINAL_INSTANTIATE_VIT_IMAGE_TOKENS = pipeline_cls.instantiate_vit_image_tokens
    pipeline_cls.__init__ = _patched_pipeline_init
    pipeline_cls.load_weights = _patched_load_weights
    pipeline_cls._vit_graph_enabled = _vit_graph_enabled
    pipeline_cls._ensure_vit_graph_manager = _ensure_vit_graph_manager
    pipeline_cls.clear_vit_graphs = _clear_vit_graphs
    pipeline_cls._vit_encode = _vit_encode
    pipeline_cls.instantiate_vit_image_tokens = _patched_instantiate_vit_image_tokens
    _PATCHED = True

    if _HOOK_INSTALLED and _ORIGINAL_IMPORT is not None:
        builtins.__import__ = _ORIGINAL_IMPORT
        _HOOK_INSTALLED = False
    logger.info("Installed HunyuanImage3 ViT NPUGraph patch for NPU.")


def _pipeline_module_ready(module) -> bool:
    return (
        module is not None
        and hasattr(module, "get_hunyuan_image_3_pre_process_func")
        and hasattr(module, "HunyuanImage3Pipeline")
    )


def _try_apply_now() -> bool:
    module = sys.modules.get(_TARGET_MODULE)
    if not _pipeline_module_ready(module):
        return False
    _apply_to_pipeline_module(module)
    return True


def _install_import_hook() -> None:
    global _HOOK_INSTALLED, _ORIGINAL_IMPORT
    if _HOOK_INSTALLED:
        return
    _ORIGINAL_IMPORT = builtins.__import__

    def hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
        assert _ORIGINAL_IMPORT is not None
        module = _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
        target = sys.modules.get(_TARGET_MODULE)
        if _pipeline_module_ready(target):
            _apply_to_pipeline_module(target)
        return module

    builtins.__import__ = hooked_import
    _HOOK_INSTALLED = True


def apply_hunyuan_image3_vit_graph_patch() -> None:
    """Install Ascend-only HunyuanImage3 ViT graph monkey patches lazily."""
    if _PATCHED:
        return
    if not _try_apply_now():
        _install_import_hook()


__all__ = [
    "HunyuanImage3VitAclGraphManager",
    "HunyuanImage3VitGraphConfig",
    "HunyuanImage3VitGraphEntry",
    "HunyuanImage3VitGraphKey",
    "Siglip2GraphLayout",
    "Siglip2PreparedGraphInputs",
    "apply_hunyuan_image3_vit_graph_patch",
    "is_vit_graph_fused_add_layer_norm_enabled",
    "vit_graph_fused_add_layer_norm_state",
]
