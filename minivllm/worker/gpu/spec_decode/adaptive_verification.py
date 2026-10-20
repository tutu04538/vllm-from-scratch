# SPDX-License-Identifier: Apache-2.0
"""DSpark 的自适应验证（对应 vLLM `v1/worker/gpu/spec_decode/adaptive_verification.py`）。

## 它在做什么

草稿宽度固定为 K，但 target **不必把每条请求的 K 枚草稿都验证**。每个草稿位置有一个
置信度 `c_i`（DSpark 的 confidence head，77 关已算好），"前 i 枚都能被接受"的存活概率是

    survival_i = c_0 × c_1 × ... × c_i

存活概率沿位置单调不增，所以按它做**全局 top-k** 时，每个请求分到的必然是**连续前缀**
（不会出现"验了第 3 枚、却没验第 2 枚"）。CPU 侧先用上一轮拷回来的 stale 置信度枚举
总预算 `b`，用实测成本表比较

    期望接受数(b) / (draft_cost_ms[B] + verify_cost_ms[非草稿行数 + b])

选一个预算；GPU 侧再用**当前**置信度重新做全局 top-k，得到每个请求实际获准验证的前缀长度，
并写回真实的 `cu_num_logits`/`query_start_loc`。CPU 侧只保留一个"预算均分"的**上界**。

## 两条硬边界

1. **CPU 数组可能只是上界**：除了 budget=0 的精确 offsets 路径，CPU 的
   `cu_num_logits_np` 不能拿去切 logits 块；真实长度只有 GPU 知道。这也是拒绝采样器
   单块分支的保护条件（`logits.shape[0] <= max_chunk_logits`）。
2. **置信度只消费一次**：`_batch_budget` 由 `get_num_tokens()` 写入、由
   `reallocate_drafts()` 取走并清空；取消/异常/slot 复用都不能再吃旧请求的置信度。
   新请求的 slot 由 `add_request()` 重置成 1.0，并进入 `_pending_resets`。

## 与上游的差异（详见 `docs/step79_alignment.md` §5）

- 本仓库没有 TP 进程组，`set_cost_curves()` 的广播退化为单进程直通。
- V2 的 **target CUDA graph** 尚未实现（74 关边界），所以 runner 只在 target 真会捕获
  varlen 图时才要求 `AttentionCGSupport.ALWAYS`；默认参数仍与上游一致。
- 成本表 profiling 由 runner 的 eager dummy step 生成；`set_initial_cost_curves()` 在
  "没有任何 full-cudagraph 样本"时回退到全部样本，否则 draft 曲线会为空。
"""

from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

import numpy as np
import torch

from ....attention.backend import AttentionCGSupport
from ..async_utils import StepTimingSample, stream
from ..attn_utils import (AttentionCGSupportInfo,
                          get_query_lens_mismatch_unsupported_backend)
from ..buffer_utils import CpuGpuBuffer, async_copy_to_gpu

if TYPE_CHECKING:
    from ..input_batch import InputBatch
    from ..states import RequestState


_PROFILE_REPLAYS = 5


def _assign_draft_token_budget(
    confidence_probs: torch.Tensor,
    idx_mapping: torch.Tensor,
    capacities: torch.Tensor,
    draft_budget: int,
    num_steps: int,
) -> None:
    """原地选全局最好的 `draft_budget` 个草稿槽位（上游同名内核的 Python 版）。

    每个 `(request, step)` 槽位的分数是存活概率（逐位置置信度的连乘）；`capacities` 进来时
    是"该请求原调度了多少枚草稿"（用于把越界槽位打成 -inf），出去时变成"获准验证几枚"。
    调用方只在 `draft_budget < capacities.sum()` 时调用，所以每个赢家都是真实草稿槽位。

    存活概率沿 step 单调不增 ⇒ 全局 top-k 天然是每个请求的连续前缀。这个性质不能用手写的
    "按单点 confidence 排序"替代：那会选出"第 3 枚高、第 2 枚低"的非法前缀。
    """
    survival = confidence_probs[idx_mapping].cumprod(dim=1)
    steps = torch.arange(num_steps, device=survival.device)
    # 越界槽位打成 -inf，永远排不到真实草稿前面。
    out_of_range = steps[None, :] >= capacities[:, None]
    survival = survival.masked_fill(out_of_range, -float("inf"))
    flat = survival.flatten()
    winners = flat.topk(draft_budget).indices
    admitted = torch.zeros_like(flat, dtype=torch.bool).index_fill_(0, winners, True)
    torch.sum(admitted.view_as(survival), dim=1, dtype=capacities.dtype, out=capacities)


# 与上游一致：小内核用 torch.compile(dynamic=True)。编译失败不改变语义，退化为 eager。
try:
    _assign_draft_token_budget_compiled = torch.compile(
        _assign_draft_token_budget, dynamic=True
    )
except Exception:  # pragma: no cover - 只在没有 torch.compile 的运行环境走到
    _assign_draft_token_budget_compiled = _assign_draft_token_budget


def build_cost_tables_from_curves(
    draft_curve: list[tuple[int, float]],
    verify_curve: list[tuple[int, float]],
    max_num_reqs: int,
    max_batch_tokens: int,
    cudagraph_limit: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """把实测曲线变成按"行数"索引的成本表（上游同名函数，逐行同义）。

    Args:
        cudagraph_limit: 最大的 cudagraph 捕获尺寸。**小于等于它**时执行会补齐到下一个
            捕获档位，所以成本是阶梯函数；**大于它**没有补齐，成本才是连续的。
    """

    def build_table(limit: int, curve: list[tuple[int, float]]) -> np.ndarray:
        xs, ys = np.asarray(curve, dtype=np.float64).T
        # 成本必须随规模单调不降：profiling 噪声不能制造"多一行反而更便宜"的假曲线。
        ys = np.maximum.accumulate(ys)
        values = np.arange(limit + 1)
        # 执行会补齐到下一个捕获尺寸，所以成本是"补齐后尺寸"的阶梯函数；
        # 在 pad bucket 内做平滑插值会凭空发明不存在的边际成本。
        idx = np.searchsorted(xs, values, side="left")
        result = ys[np.minimum(idx, len(xs) - 1)]
        # 超过捕获上限后不再补齐，成本才真的连续；但"跨过上限"本身会丢掉 cudagraph，
        # 是一个真实的不连续点，所以只用**上限以上**的实测点做插值/外推。
        if cudagraph_limit:
            smooth = values > cudagraph_limit
            above = xs > cudagraph_limit
            if smooth.any() and above.any():
                result[smooth] = np.interp(values[smooth], xs[above], ys[above])
        if len(xs) > 1:
            after = values > xs[-1]
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            result[after] = ys[-1] + slope * (values[after] - xs[-1])
        return result

    draft_table = np.maximum(build_table(max_num_reqs, draft_curve), 0.0)
    verify_table = np.maximum(build_table(max_batch_tokens, verify_curve), 1e-6)
    return draft_table, verify_table


def _median_curve(points: Iterable[tuple[int, float]]) -> list[tuple[int, float]]:
    grouped: defaultdict[int, list[float]] = defaultdict(list)
    for key, value in points:
        grouped[key].append(value)
    return [(k, float(np.median(v))) for k, v in sorted(grouped.items())]


class AdaptiveVerificationManager:
    """成本曲线 + CPU stale 置信度 + GPU 当前置信度 + 一次性预算（上游同名类）。"""

    def __init__(
        self,
        req_states: "RequestState",
        query_start_loc: torch.Tensor,
        num_bonus_tokens: int,
        max_total_logits: int,
    ):
        self.req_states = req_states
        self.num_speculative_steps = req_states.num_speculative_steps
        device = req_states.device
        self._copy_stream = torch.cuda.Stream(device)

        self.num_bonus_tokens = num_bonus_tokens
        # 拒绝采样一次验证一整块连续 logits；分块路径按**未裁剪**的 CPU offsets 切块，
        # 索引不到压紧后的布局。所以预算必须能装进一个 chunk（上游同款理由）。
        self._max_total_logits = max_total_logits
        self.query_start_loc = query_start_loc
        self.cost_tables: tuple[np.ndarray, np.ndarray] | None = None
        self.cost_curves: tuple[list[tuple[int, float]],
                                list[tuple[int, float]]] | None = None
        # 最大 cudagraph 捕获 token 数；超过它没有补齐。
        self._cudagraph_limit = 0
        self._batch_budget: tuple[dict[str, int], dict[str, int], int] | None = None
        max_num_reqs = req_states.max_num_reqs
        # 当前每 slot 的置信度（GPU）。
        self._confidence_probs = torch.empty(
            (max_num_reqs, self.num_speculative_steps),
            dtype=torch.float32,
            device=device,
        )
        self._batch_draft_capacity = torch.empty(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self._num_non_draft_tokens = torch.empty_like(query_start_loc[:-1])
        self._cu_num_logits = torch.empty_like(query_start_loc)

        # 两个 D2H 槽位：预算读的是上一轮已经落地的 CPU 数据。
        self._stale_confidences = [
            CpuGpuBuffer(
                max_num_reqs,
                self.num_speculative_steps,
                dtype=torch.float32,
                device=device,
            )
            for _ in range(2)
        ]
        self._copy_events = [torch.cuda.Event(blocking=True) for _ in range(2)]
        self._pending_resets: list[int] = []
        self._stale_idx = 0
        for slot in self._stale_confidences:
            slot.np.fill(1.0)

        # 诊断/基准用：最近一轮原调度草稿数与实际预算（不是"接受数"）。
        self.last_scheduled_drafts = 0
        self.last_draft_budget = 0
        self.last_num_sampling_requests = 0

    # -------- 请求生命周期 --------

    def add_request(self, req_idx: int) -> None:
        """新请求占用 slot 时把两代置信度都重置成 1.0（上游同款）。

        不重置的话，slot 复用后的第一步会拿**上一个请求**的置信度算预算——不报错，
        只是预算莫名其妙地偏大/偏小。
        """
        self._stale_confidences[self._stale_idx].np[req_idx].fill(1.0)
        self._pending_resets.append(req_idx)
        self._confidence_probs[req_idx].fill_(1.0)

    # -------- 成本曲线 --------

    def batches_to_profile(self, capture_sizes: list[int]) -> Iterator[dict[str, int]]:
        """生成 dummy-run 的 kwargs；每个尺寸重复 `_PROFILE_REPLAYS` 次取中位数。"""
        import os

        max_num_tokens = self.req_states.max_num_batched_tokens
        size = self._cudagraph_limit = capture_sizes[-1] if capture_sizes else 0
        # 捕获上限之外也要 profile：真实 step 会在那里跑（eager），线性外推会严重低估。
        tail_sizes: set[int] = set()
        if size:
            tail_sizes.add(min(size + size // 2, max_num_tokens))
            while size < max_num_tokens:
                size = min(size * 2, max_num_tokens)
                tail_sizes.add(size)
            tail_sizes -= set(capture_sizes)
        context_len = int(os.environ.get(
            "VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN", "0"))
        for num_tokens in capture_sizes + sorted(tail_sizes):
            for _ in range(_PROFILE_REPLAYS):
                yield {"num_tokens": num_tokens, "context_len": context_len}

    def set_initial_cost_curves(self, samples: list[StepTimingSample]) -> None:
        """把 profiling 样本变成两条中位数曲线（上游同名方法）。

        上游只允许 full-cudagraph 样本给 draft 曲线定价（eager target 的 CPU 启动间隙会
        把 drafter 的 event 时间撑大）。本仓库 target 图尚未实现，profiling 全是 eager
        样本；若严格过滤会得到空曲线，所以这里**在没有 full 样本时回退到全部样本**。
        有 full 样本时仍严格按上游过滤。
        """
        full_samples = [s for s in samples if s.full_cudagraph]
        draft_samples = full_samples or samples
        draft_curve = _median_curve((s.num_reqs, s.drafter_ms) for s in draft_samples)
        verify_curve = _median_curve(
            (s.num_target_tokens, s.forward_ms) for s in samples
        )
        self.set_cost_curves(draft_curve, verify_curve)

    def set_cost_curves(
        self,
        draft_curve: list[tuple[int, float]],
        verify_curve: list[tuple[int, float]],
    ) -> None:
        """安装两条 ms 曲线（上游这里还会从 rank 0 广播；本仓库单进程直通）。"""
        if not draft_curve or not verify_curve:
            raise RuntimeError(
                "Adaptive verification could not profile step costs. Pass "
                "`enable_adaptive_verification=false` in the speculative config to "
                "verify a fixed number of drafts instead."
            )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # 上游 `get_tp_group().broadcast_object(...)` 的等价物：多 rank 时以 rank 0 为准，
            # 保证每张卡用同一张成本表（否则各 rank 会选出不同预算）。
            obj = [list(draft_curve), list(verify_curve)]
            torch.distributed.broadcast_object_list(obj, src=0)
            draft_curve, verify_curve = obj[0], obj[1]
        self.cost_curves = (list(draft_curve), list(verify_curve))
        self.cost_tables = build_cost_tables_from_curves(
            draft_curve,
            verify_curve,
            self.req_states.max_num_reqs,
            self.req_states.max_num_batched_tokens,
            self._cudagraph_limit,
        )

    # -------- 每轮：置信度 -> CPU 预算 -> GPU 布局 --------

    def record_confidences(
        self,
        confidence_probs: torch.Tensor,
        input_batch: "InputBatch",
    ) -> None:
        """发布本轮原始置信度给 GPU 排序内核，并启动到 CPU 的异步拷贝。"""
        num_reqs = input_batch.num_reqs
        ready_idx = self._stale_idx ^ 1
        # 上一轮往 ready 槽的拷贝必须先落地，才能覆盖它。
        self._copy_events[ready_idx].synchronize()
        if self._pending_resets:
            self._stale_confidences[ready_idx].np[self._pending_resets] = 1.0
            self._pending_resets.clear()
        # 上一轮拷贝已经落地：预算读它；这一轮开始覆盖另一个槽。
        self._stale_idx, write_idx = ready_idx, self._stale_idx

        self._confidence_probs[input_batch.idx_mapping] = confidence_probs[:num_reqs]
        write_slot = self._stale_confidences[write_idx]
        write_slot.gpu.copy_(self._confidence_probs)

        current_stream = torch.cuda.current_stream(self.req_states.device)
        self._copy_stream.wait_stream(current_stream)
        with stream(self._copy_stream, current_stream):
            write_slot.copy_to_cpu()
            self._copy_events[write_idx].record()

    def get_num_tokens(
        self,
        num_tokens_per_req: dict[str, int],
        draft_tokens: dict[str, list[int]],
    ) -> int:
        """选总预算并把结果暂存在 `_batch_budget`，返回裁剪后的 target 行数。

        CPU 用 **stale** 置信度估期望接受数；GPU 侧 `reallocate_drafts()` 会用当前置信度
        重新分配前缀。两者不是同一份数据，这是有意为之（CPU 拿不到本轮的 GPU 结果）。
        """
        assert self.cost_tables is not None, (
            "成本表还没 profiling（Runner 应在 capture_model() 里调 "
            "set_initial_cost_curves()）")
        req_ids = list(num_tokens_per_req)
        num_reqs = len(req_ids)
        scheduled_tokens = np.fromiter(
            num_tokens_per_req.values(), dtype=np.int32, count=num_reqs
        )
        scheduled_drafts = np.fromiter(
            (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        num_non_draft_tokens = scheduled_tokens - scheduled_drafts
        slots = np.fromiter(
            (self.req_states.req_id_to_index[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        stale_confidences = self._stale_confidences[self._stale_idx].np[slots]
        survival_probability = np.cumprod(
            stale_confidences.astype(np.float64), axis=1)
        steps = np.arange(self.num_speculative_steps)
        valid = steps[None, :] < scheduled_drafts[:, None]
        scores = np.sort(survival_probability[valid])[::-1]
        num_non_draft_tokens_total = int(num_non_draft_tokens.sum())
        max_draft_budget = min(
            int(scheduled_drafts.sum()),
            max(0, self._max_total_logits - num_reqs * self.num_bonus_tokens),
        )
        scores = scores[:max_draft_budget]
        draft_cost_ms, verify_cost_ms = self.cost_tables
        num_sampling_requests = np.count_nonzero(
            self.req_states.num_computed_tokens_np[slots] + num_non_draft_tokens
            >= self.req_states.prefill_len.np[slots]
        )
        self.last_num_sampling_requests = int(num_sampling_requests)
        # b=0 时只有"采样请求各一个 bonus token"；每多验证一枚草稿，期望多
        # survival_i 个接受 token。这是期望值，不是实际接受数。
        num_tokens_to_estimated_accepted_tokens = np.concatenate(
            ([num_sampling_requests], num_sampling_requests + np.cumsum(scores))
        )
        costs = (
            draft_cost_ms[len(req_ids)]
            + verify_cost_ms[
                num_non_draft_tokens_total : num_non_draft_tokens_total
                + max_draft_budget
                + 1
            ]
        )
        num_drafts_per_req = {
            req_id: int(num_drafts)
            for req_id, num_drafts in zip(req_ids, scheduled_drafts, strict=True)
        }
        num_non_draft_tokens_per_req = {
            req_id: int(num_tokens)
            for req_id, num_tokens in zip(req_ids, num_non_draft_tokens, strict=True)
        }
        # np.argmax 在并列时取**第一个**（更小的预算）；这是参考实现的 tie 规则，
        # 不要改成"随机/最大预算"之类自创规则。
        draft_budget = int(np.argmax(
            num_tokens_to_estimated_accepted_tokens / costs))
        self._batch_budget = (
            num_drafts_per_req,
            num_non_draft_tokens_per_req,
            draft_budget,
        )
        self.last_scheduled_drafts = int(scheduled_drafts.sum())
        self.last_draft_budget = draft_budget
        return sum(num_non_draft_tokens_per_req.values()) + draft_budget

    def compact_batch(
        self,
        num_draft_tokens_per_req: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        cu_num_logits_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """把 CPU 批次压到选定预算；返回 (每请求 token 数, CPU cu_num_logits)。

        返回的 CPU offsets **不一定精确**：
          - budget == 全部草稿：原样返回；
          - budget == 0：精确（每个请求只有 bonus 行），这条路径专治"超出 logits 分块"；
          - 其它：把预算**均分**给有草稿的请求，只是上界；GPU 会换成真实的非均匀前缀。
        """
        batch_budget = self._batch_budget
        assert batch_budget is not None
        _, _, draft_budget = batch_budget
        num_drafts = int(num_draft_tokens_per_req.sum())
        if draft_budget == num_drafts:
            return num_scheduled_tokens, cu_num_logits_np

        num_non_draft_tokens = num_scheduled_tokens - num_draft_tokens_per_req
        if draft_budget == 0:
            # budget=0 时 CPU 可以精确知道 offsets：每请求只剩 bonus 行。
            num_reqs = num_scheduled_tokens.shape[0]
            cu_num_logits_np = (
                np.arange(num_reqs + 1, dtype=cu_num_logits_np.dtype)
                * self.num_bonus_tokens
            )
            return num_non_draft_tokens, cu_num_logits_np

        is_verification_request = num_draft_tokens_per_req > 0
        num_verification_reqs = int(is_verification_request.sum())
        # sort_batch_req_ids 把有草稿的请求排在前面；这条断言就是那条排序约定。
        assert np.all(is_verification_request[:num_verification_reqs])
        # CPU 侧只均分；GPU 的 `reallocate_drafts()` 会按当前置信度非均匀分配。
        draft_lens_cpu = np.zeros_like(num_non_draft_tokens)
        draft_lens_cpu[:num_verification_reqs] = (
            draft_budget // num_verification_reqs)
        draft_lens_cpu[: draft_budget % num_verification_reqs] += 1
        return num_non_draft_tokens + draft_lens_cpu, cu_num_logits_np

    def reallocate_drafts(
        self, req_ids: list[str], idx_mapping: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """用 GPU 当前置信度分配前缀，并写回真实 offsets（上游同名方法）。"""
        batch_budget, self._batch_budget = self._batch_budget, None
        assert batch_budget is not None
        num_drafts_per_req, num_non_draft_tokens_per_req, draft_budget = batch_budget
        num_reqs = idx_mapping.shape[0]
        scheduled_drafts = np.fromiter(
            (num_drafts_per_req[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        num_non_draft_tokens = np.fromiter(
            (num_non_draft_tokens_per_req[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        num_tokens = int(num_non_draft_tokens.sum()) + draft_budget

        # capacities 进来是"原有效候选数"（内核用它排除越界槽位），出去是"获准验证数"。
        capacities = self._batch_draft_capacity[:num_reqs]
        if draft_budget == 0:
            capacities.zero_()
        else:
            async_copy_to_gpu(scheduled_drafts, out=capacities)
            if draft_budget < int(scheduled_drafts.sum()):
                _assign_draft_token_budget_compiled(
                    self._confidence_probs,
                    idx_mapping,
                    capacities,
                    draft_budget,
                    self.num_speculative_steps,
                )

        num_non_draft_tokens_gpu = self._num_non_draft_tokens[:num_reqs]
        async_copy_to_gpu(num_non_draft_tokens, out=num_non_draft_tokens_gpu)
        self._cu_num_logits[:1].zero_()
        torch.cumsum(
            capacities + self.num_bonus_tokens,
            dim=0,
            out=self._cu_num_logits[1 : num_reqs + 1],
        )
        self.query_start_loc[:1].zero_()
        torch.cumsum(
            capacities + num_non_draft_tokens_gpu,
            dim=0,
            out=self.query_start_loc[1 : num_reqs + 1],
        )
        self.query_start_loc[num_reqs + 1 :].fill_(num_tokens)
        return (
            self._cu_num_logits[: num_reqs + 1],
            self.query_start_loc,
            draft_budget,
        )


def maybe_create_adaptive_verification_manager(
    *,
    enable_adaptive_verification: bool,
    attn_groups: list[list],
    attn_cg_support: AttentionCGSupportInfo | None,
    req_states: "RequestState",
    query_start_loc: torch.Tensor,
    num_bonus_tokens: int,
    max_total_logits: int,
    require_varlen_cudagraph: bool = True,
) -> AdaptiveVerificationManager | None:
    """按上游的两条能力检查决定是否建 manager（上游同名函数）。

    `require_varlen_cudagraph` 是本仓库的唯一扩展：默认 True，语义与上游完全一致
    （要求 `AttentionCGSupport.ALWAYS`）。V2 的 target 图尚未实现时，runner 传 False，
    因为这条执行路径不会捕获 target varlen 图——但**语义**能力
    （`supports_device_cpu_query_lens_mismatch`）仍然必须满足。
    """
    if not enable_adaptive_verification:
        return None

    # 选择器会排除不支持的后端；但硬编码后端的模型不会走选择器，所以这里再查一遍。
    backend = get_query_lens_mismatch_unsupported_backend(attn_groups)
    if backend is not None:
        raise ValueError(
            "Adaptive verification trims verification requests on device, which"
            f" the {backend} attention backend does not support. Pass "
            "enable_adaptive_verification=false in the speculative config, or "
            "use a backend that does."
        )

    if require_varlen_cudagraph:
        assert attn_cg_support is not None
        if attn_cg_support.min_cg_support != AttentionCGSupport.ALWAYS:
            raise ValueError(
                "Adaptive verification captures varlen decode cudagraphs, so every"
                " attention builder must report AttentionCGSupport.ALWAYS, but"
                f" {attn_cg_support.min_cg_attn_backend} reports"
                f" {attn_cg_support.min_cg_support}. Pass "
                "enable_adaptive_verification=false in the speculative config, or "
                "use a backend that does."
            )

    return AdaptiveVerificationManager(
        req_states,
        query_start_loc,
        num_bonus_tokens,
        max_total_logits=max_total_logits,
    )
