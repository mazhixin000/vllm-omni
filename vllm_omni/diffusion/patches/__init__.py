# SPDX-License-Identifier: Apache-2.0
"""模型级 monkey-patch 集合。

这里放的是"启动即生效"的运行时 patch：进程加载 vllm_omni 时会经由
`vllm_omni/__init__.py -> vllm_omni/patch.py` 触发本包 import，从而
自动挂载每个子 patch 模块。

新增 patch 的规则：
  - 在本目录新建一个模块（如 xxx_fusion.py）
  - 内部用 try/except 包住可能失败的挂载（找不到目标 / torch_npu 不可用 等）
  - 用环境变量做开关，方便对比 baseline
  - 在 `vllm_omni/patch.py` 末尾追加一次 `from vllm_omni.diffusion.patches import xxx_fusion`
"""
