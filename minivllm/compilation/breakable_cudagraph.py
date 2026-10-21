"""Breakable CUDA Graph（对应上游 `vllm/compilation/breakable_cudagraph.py` 的核心子集）。

与 Phase 1/2 的 FULL 图不同：PIECEWISE 不把整段模型录成一张图，而是用 stream capture
录若干段，在 attention 调用处“断开”：

    graph 段 0（embedding / qkv / 前段 MLP）
    eager attention（读 runtime metadata，写固定 output 缓冲）
    graph 段 1（o_proj / 后段 MLP）
    ...

`eager_break_during_capture` 就是断点装饰器：只在 `BreakableCUDAGraphCapture` 活跃且
forward context 不是 FULL 时触发；FULL 图仍把 attention 录进去。

裁剪（本仓库没有）：offloader、breakable 的 sequence parallel、`envs` 开关（本仓库用
显式的 `BreakableCUDAGraphWrapper` 只在 PIECEWISE 时启用，不新增环境变量）。
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import threading
from collections.abc import Callable
from typing import Any

import torch

from ..config import CUDAGraphMode
from ..forward_context import (BatchDescriptor, get_forward_context,
                               is_forward_context_available)


def _graph_pool():
    from .cuda_graph import _get_global_graph_pool

    return _get_global_graph_pool()


_CAPTURE_STREAM: torch.cuda.Stream | None = None


def _capture_stream() -> torch.cuda.Stream:
    """全进程复用的捕获流；不要每次 capture 建/销毁（pinned host allocator 会踩到
    `CachingHostAllocator` 的 use_count 断言）。"""
    global _CAPTURE_STREAM
    if _CAPTURE_STREAM is None:
        _CAPTURE_STREAM = torch.cuda.Stream()
    return _CAPTURE_STREAM


class BreakableCUDAGraphCapture:
    """一段或多段 `torch.cuda.CUDAGraph` + 夹在中间的 eager callable。"""

    _tls = threading.local()

    @classmethod
    def current(cls) -> "BreakableCUDAGraphCapture | None":
        return getattr(cls._tls, "active", None)

    @classmethod
    def is_active(cls) -> bool:
        return cls.current() is not None

    def __init__(self, pool: Any | None = None):
        self.pool = pool
        self.segments: list[Callable[[], Any]] = []
        self._graph: torch.cuda.CUDAGraph | None = None
        self._capturing = False
        self._prev_stream: torch.cuda.Stream | None = None
        self._stream: torch.cuda.Stream | None = None

    def __enter__(self):
        if getattr(self._tls, "active", None) is not None:
            raise RuntimeError("不支持嵌套 BreakableCUDAGraphCapture")
        self._tls.active = self
        # `capture_begin()` 必须非默认 stream；上游 `graph_capture` 负责切流，
        # 本仓库的 `compilation.graph_capture` 只做 gc/同步，所以这里自己切。
        self._prev_stream = torch.cuda.current_stream()
        self._stream = _capture_stream()
        try:
            self._stream.wait_stream(self._prev_stream)
            torch.cuda.set_stream(self._stream)
            self._begin_segment()
        except BaseException:
            torch.cuda.set_stream(self._prev_stream)
            self._tls.active = None
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self._end_segment()
        finally:
            if self._prev_stream is not None:
                if self._stream is not None:
                    # 捕获/eager 段跑在 side stream；调用方接下来在默认 stream 读输出，
                    # 不 join 会读到未完成的结果（不报错、只是错）。
                    self._prev_stream.wait_stream(self._stream)
                torch.cuda.set_stream(self._prev_stream)
            self._tls.active = None

    def _begin_segment(self) -> None:
        assert not self._capturing
        graph = torch.cuda.CUDAGraph()
        if self.pool is not None:
            graph.capture_begin(pool=self.pool)
        else:
            graph.capture_begin()
        self._graph = graph
        self._capturing = True

    def _end_segment(self) -> None:
        if not self._capturing:
            return
        assert self._graph is not None
        self._graph.capture_end()
        self.segments.append(self._graph.replay)
        self._graph = None
        self._capturing = False

    def add_eager(self, fn: Callable[[], Any]) -> Any:
        """结束当前图段 → eager 执行 → 记录 → 开始下一段（上游同名方法）。"""
        self._end_segment()
        result = fn()
        self.segments.append(_EagerCallable(fn))
        self._begin_segment()
        return result

    def replay(self) -> None:
        for segment in self.segments:
            segment()

    @property
    def num_graphs(self) -> int:
        return sum(1 for s in self.segments if not isinstance(s, _EagerCallable))


class _EagerCallable:
    """给 replay 列表里的 eager 段打个标记（只用于计数/调试）。"""

    def __init__(self, fn: Callable[[], Any]):
        self.fn = fn

    def __call__(self):
        return self.fn()


def eager_break_during_capture(fn: Callable) -> Callable:
    """把 attention 调用变成 breakable capture 的 eager 断点（上游同名装饰器）。

    只在 `BreakableCUDAGraphCapture` 活跃时生效；FULL 图不打断（整段 attention 进图）。
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        capture = BreakableCUDAGraphCapture.current()
        if capture is None or not capture._capturing:
            return fn(*args, **kwargs)
        if is_forward_context_available():
            mode = get_forward_context().cudagraph_runtime_mode
            if mode == CUDAGraphMode.FULL:
                return fn(*args, **kwargs)
        # 本仓库的图段输出张量没有上游那种"图池持有"的强生命周期；用强引用，
        # 避免 replay 时 weakref 已失效（上游注释里的 weakref 前提在本仓库不成立）。
        eager_fn = lambda: fn(*args, **kwargs)

        return capture.add_eager(eager_fn)

    return wrapper


@dataclasses.dataclass
class _BreakableEntry:
    batch_descriptor: BatchDescriptor
    capture: BreakableCUDAGraphCapture | None = None
    output: Any = None


class BreakableCUDAGraphWrapper:
    """只在 PIECEWISE 运行模式下捕获/重放；其它模式直通。

    与上游 drop-in wrapper 的差异：本仓库只把它包在 target 模型外层，FULL 图仍由
    `CudaGraphManager` 的 `torch.cuda.graph` 负责；因此这里只需要 PIECEWISE 分支。
    """

    def __init__(self, runnable: Callable[..., Any], pool: Any | None = None):
        self.runnable = runnable
        self.pool = pool
        self.entries: dict[BatchDescriptor, _BreakableEntry] = {}
        self.num_replays = 0

    def __getattr__(self, key: str) -> Any:
        return getattr(self.runnable, key)

    def clear_graphs(self) -> None:
        self.entries.clear()

    def capture(self, *args: Any, **kwargs: Any) -> None:
        """在 forward context 已设成 PIECEWISE 时录一段（由 Runner 在 execute 前调用）。"""
        context = get_forward_context()
        desc = context.batch_descriptor
        if desc is None:
            raise RuntimeError("PIECEWISE 捕获缺少 batch_descriptor")
        entry = self.entries.setdefault(desc, _BreakableEntry(batch_descriptor=desc))
        if entry.capture is not None:
            return
        with BreakableCUDAGraphCapture(self.pool or _graph_pool()) as capture:
            entry.output = self.runnable(*args, **kwargs)
        entry.capture = capture

    def __call__(self, *args: Any, **kwargs: Any):
        if not is_forward_context_available():
            return self.runnable(*args, **kwargs)
        context = get_forward_context()
        if context.cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE:
            return self.runnable(*args, **kwargs)
        desc = context.batch_descriptor
        if desc is None:
            raise RuntimeError("PIECEWISE 运行缺少 batch_descriptor，无法定位图键")
        entry = self.entries.get(desc)
        if entry is None or entry.capture is None:
            raise RuntimeError("PIECEWISE 图还没捕获：Runner 应先 capture() 再 replay()")
        entry.capture.replay()
        self.num_replays += 1
        return entry.output
