"""一次 forward 的上下文（**兼容层**：实现已搬到 `minivllm/forward_context.py`）。

解决的问题：**模型层怎么知道自己这一层的 AttentionMetadata？**

模型不该拿 Runner 的参数（`forward()` 的签名里没有块表、没有 metadata）。做法是：Runner 在
forward 之前把"层名 → AttentionMetadata"放进一个临时的上下文，`Attention.forward()` 再按**自己
唯一的层名**取出来。

三条约定：**只在一次 forward 内有效**（`finally` 里恢复）、**按层名索引**、**上下文里没有
Request/Scheduler/KV 池对象**。

69 关把 `ForwardContext` / `set_forward_context()` 提到了包根，与上游 `vllm/forward_context.py`
对齐：那里同时住着 CUDA Graph 的批次键 `BatchDescriptor`，而图包装器与注意力层都要用它
（留在 `attention/` 下会造成 `attention → 图包装器 → attention` 的循环依赖）。
本文件只做转发，保留旧导入路径（`minivllm.attention.forward_context`）不破。
"""

from ..forward_context import (BatchDescriptor, ForwardContext, create_forward_context,
                               get_forward_context, is_forward_context_available,
                               override_forward_context, set_forward_context)

__all__ = ["BatchDescriptor", "ForwardContext", "create_forward_context",
           "get_forward_context", "is_forward_context_available",
           "override_forward_context", "set_forward_context"]
