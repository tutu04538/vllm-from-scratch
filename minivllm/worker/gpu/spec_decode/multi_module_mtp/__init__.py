"""多模块 MTP 的输入布局与 eager 执行（需求 080A / 080B）。

80A 提供可独立测试的输入准备函数；80B 的 `MultiModuleMTPSpeculator` 把它们接成
V2 的 `propose()` 调用链。CUDA Graph（80C）、Gemma4/Step3p5/Q-only/稀疏索引
（80D–80G）不在本子包当前范围内。
"""

from .speculator import MultiModuleMTPSpeculator

__all__ = ["MultiModuleMTPSpeculator"]
