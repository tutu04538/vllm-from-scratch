"""CUDA Graph 包装器（对应 vLLM `compilation/cuda_graph.py` 的子集，逐行为对齐）。

它把"一个可调用对象"变成"能捕获/能重放的图"，**只做三件事**：

    读 forward 上下文里的 (运行模式, 图键)
    运行模式与自己的模式不符 → 直通（图外那段照常 eager 跑）
    相符 → 键不存在就**捕获**一次并存起来；键存在就 `replay()`

三条边界（上游注释里写得很清楚，这里照抄并且真的不做）：

1. **包装器不持有输入缓冲**：它假设调用方每次都喂**同一批地址**的静态缓冲。这是刻意的：
   包装器不知道输入形状怎么来的，硬要它去拷输入就要引入"形状推断"，而那属于 Runner。
   代价是"喂了别的张量"会静默读旧数据——所以调试模式下会逐张量比对 `data_ptr()`（本仓库
   把这条检查做成**始终开启**：这类错不报错，只是结果不对，正是最该拦的一类）。
2. **输出缓冲是捕获时那一块**：图重放写回的是捕获时分配的同一块张量，所以每次 `replay()`
   之后返回的是同一个张量对象（内容是新算的）。本仓库**直接持有强引用**——上游用弱引用是为了
   让"分段图之间传递的中间结果"尽快释放，而本仓库只有一张全图、输出马上要拿去采样，
   强引用更简单也更不容易出"重放时缓冲已经没了"的问题。
3. **不做"图外还能捕获"的懒加载**：捕获必须在
   `set_cudagraph_capturing_enabled(True)` 的窗口里（`monitor.py`），窗口外一律抛错。

与上游的差异：没有 offloader 同步（本仓库没有权重/激活卸载）、没有 gc patch 与
`compilation_counter` 指标。
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar
import torch

from ..config import CUDAGraphMode
from ..forward_context import (BatchDescriptor, get_forward_context,
                               is_forward_context_available)
from .monitor import validate_cudagraph_capturing_enabled


@contextmanager
def graph_capture(device: str = "cuda"):
    """捕获期的公共环境（对应上游 `graph_capture`）：同步 + 释放缓存 + 用一张**专用显存池**。

    为什么要专用池：图会把捕获期间的显存分配**录下来**（重放时不再走分配器），所以这些块
    必须独占、生命周期与图一致。不隔离的话，别的张量复用同一段地址，重放就会写花别人的数据。
    """
    pool = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        yield pool
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()


@dataclass
class CUDAGraphEntry:
    """一张图 + 它的键 + 捕获时的输入地址（地址用来验"重放时喂的是不是同一批缓冲"）。"""

    batch_descriptor: BatchDescriptor
    cudagraph: torch.cuda.CUDAGraph | None = None
    output: Any | None = None
    input_addresses: list[int] | None = None


@dataclass
class CUDAGraphOptions:
    """上游同名字段里与本仓库相关的那个：捕获时要不要打日志。"""

    debug_log_enable: bool = True


class CUDAGraphWrapper:
    """给一个可调用对象加上"捕获一次、之后重放"的能力（上游同名类）。"""

    #: 所有实例（上游用它做 `clear_all_graphs()`，测试里换权重/换配置时要把图丢掉）
    _all_instances: ClassVar[set["CUDAGraphWrapper"]] = set()

    @classmethod
    def clear_all_graphs(cls) -> None:
        for instance in list(cls._all_instances):
            instance.clear_graphs()

    def __init__(self, runnable, vllm_config, runtime_mode: CUDAGraphMode,
                 cudagraph_options: CUDAGraphOptions | None = None) -> None:
        if runtime_mode == CUDAGraphMode.NONE or not runtime_mode.is_valid_runtime_mode():
            raise ValueError(
                f"只有具体运行模式（FULL / PIECEWISE）才需要包装器，收到 {runtime_mode}")
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config
        self.cudagraph_options = cudagraph_options or CUDAGraphOptions()
        # 键 → 图。**唯一**的图缓存（不在别处再存一份，AGENTS §9：不自造一套图缓存）
        self.concrete_cudagraph_entries: dict[BatchDescriptor, CUDAGraphEntry] = {}
        self.num_captures = 0
        self.num_replays = 0
        # 本次 replay 是否有**输入地址与捕获时不一致**（伪造/调试用；正常路径恒 False）。
        # 不做静默纠正：地址不一致意味着"重放会读旧数据"，唯一正确的处理是报错。
        self.last_replay_checked = False
        CUDAGraphWrapper._all_instances.add(self)

    # 属性转发：包起来之后，外面仍然能 `model.compute_logits(...)` / `model.named_modules()`
    def __getattr__(self, key: str) -> Any:
        return getattr(self.runnable, key)

    def unwrap(self):
        """拿到原来的可调用对象（不需要图的路径用）。"""
        return self.runnable

    @property
    def cudagraph_wrapper(self) -> "CUDAGraphWrapper":
        return self

    def clear_graphs(self) -> None:
        self.concrete_cudagraph_entries.clear()

    # -------- 一次调用 --------

    def __call__(self, *args: Any, **kwargs: Any):
        if not is_forward_context_available():
            # 不在推理路径里（例如权重加载后的自检前向）：直通，不捕获也不重放
            return self.runnable(*args, **kwargs)

        forward_context = get_forward_context()
        runtime_mode = forward_context.cudagraph_runtime_mode
        batch_descriptor = forward_context.batch_descriptor

        if runtime_mode == CUDAGraphMode.NONE or runtime_mode != self.runtime_mode:
            # 模式不匹配 = 这一轮不该走我这层的图（嵌套多层包装器时也靠这条各管各的）
            return self.runnable(*args, **kwargs)

        if batch_descriptor is None:
            raise RuntimeError(
                f"运行模式是 {runtime_mode} 却没有批次描述（batch_descriptor）："
                f"图键是分派的依据，缺了就只能猜——猜错会重放到别的形状上")

        entry = self.concrete_cudagraph_entries.get(batch_descriptor)
        if entry is None:
            entry = CUDAGraphEntry(batch_descriptor=batch_descriptor)
            self.concrete_cudagraph_entries[batch_descriptor] = entry

        if entry.cudagraph is None:
            validate_cudagraph_capturing_enabled()
            entry.input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
            cudagraph = torch.cuda.CUDAGraph()
            pool = torch.cuda.graph_pool_handle()
            # 注意引用管理：捕获期间 `output` 由图的显存池管理，出了 `with` 之后只留弱引用，
            # 否则这块显存看起来永远"有人用"。
            with torch.cuda.graph(cudagraph, pool=pool):
                output = self.runnable(*args, **kwargs)
            # 强引用留在 entry 里：这块张量就是图的输出缓冲，重放时写回同一块
            entry.output = output
            entry.cudagraph = cudagraph
            self.num_captures += 1
            if self.cudagraph_options.debug_log_enable:
                import logging

                logging.getLogger(__name__).debug(
                    "捕获了一张 %s 图：%s", self.runtime_mode.name, batch_descriptor)
            # 第一次是**捕获**，返回强引用给调用方（这一份输出刚算出来，调用方要立刻用）
            return output

        new_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
        if new_addresses != entry.input_addresses:
            raise RuntimeError(
                f"CUDA Graph 重放时输入张量的地址与捕获时不一致：\n"
                f"  捕获时 {entry.input_addresses}\n  现在   {new_addresses}\n"
                f"图里记的是**指针**，地址变了就会读到别的数据（旧缓冲/别人的张量），"
                f"而且不会报错、只是结果不对。请让 Runner 使用静态输入缓冲"
                f"（GPUModelRunner 的 padded 工作区）")
        self.last_replay_checked = True
        entry.cudagraph.replay()
        self.num_replays += 1
        return entry.output
