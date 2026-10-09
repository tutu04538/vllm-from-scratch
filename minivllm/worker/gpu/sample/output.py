"""对应 vLLM `vllm/v1/worker/gpu/sample/output.py`（逐行移植）。

`_pack_sampling_mask_kernel`、`SamplingMaskTensors`、`SamplerOutput` 的**字段与默认值、
内核体、算法**逐行保持上游原文，只改 import；此外把上游放在 `v1/outputs.py` 的
`SamplingMaskLists` 一起搬进来（理由见第 3 条）。逐条列出删改点：

1. `from vllm.triton_utils import tl, triton` → `from minivllm.triton_utils import tl, triton`
   （路径映射 `minivllm/` ↔ `vllm/`，写法不变）。`from vllm.v1.outputs import LogprobsTensors,
   SamplingMaskLists` → `from minivllm.outputs import LogprobsTensors`：本项目那份
   `LogprobsTensors` 的字段与上游**逐字一致**（`logprob_token_ids, logprobs,
   selected_token_ranks, cu_num_generated_tokens`），本文件只把它当类型用，无需改字段名。
2. **`SamplerOutput` 采用方案 (b)：在本文件里定义 V2 版**，不从 `minivllm.outputs` 复用。
   理由（先读了两个文件核对）：本项目 68 关按上游 **V1** 写的
   `minivllm/outputs.py::SamplerOutput` 只有两个字段
   （`sampled_token_ids`、`logprobs_tensors=None`），而 V2 的采样通路需要额外四件东西：
   `num_nans`（`torch.isnan` 的逐行计数，用于"出现 NaN 就整行回退成全 mask"）、
   `num_sampled`（每行真正采出的 token 数，采样掩码按它判断该行是否 active）、
   `num_rejected`（投机路径的逐行被拒草稿数，默认 `None`）、
   `sampling_mask_tensors`（下面这个 `SamplingMaskTensors`，异步 D2H 的掩码句柄）。
   V1 版**没有**这四个字段，硬塞会让 V1 的调用方（`minivllm/sample/sampler.py` 等）
   看到一堆 `None`；而在 `minivllm/outputs.py` 里加字段又会改到本项目已有文件
   （本次移植不允许）。所以按"上游 V2 文件里本来就是它自己定义"的原样保留在这里。
   注意：**两个同名 dataclass 是两条通路的类型**（V1 无 num_*、V2 有），不是重复定义。
3. **`SamplingMaskLists` 从上游 `v1/outputs.py` 一并搬进本文件**：本项目
   `minivllm/outputs.py` 里**没有**这个 NamedTuple（68 关只搬了 `LogprobsTensors/LogprobsLists`），
   而 `SamplingMaskTensors.tolists()` 的返回值就是它。字段与上游逐字一致
   （`token_ids, offsets, cu_num_generated_tokens`），方法（`slice_request`/`to_nested_list`/
   `merge`）也一并搬来，保持 CSR 语义与上游一致——本文件是它在本项目唯一合理的落点。
   本项目目前**还没有生产调用方**（73 关的引擎侧/异步交付还没接上来，只有差分测试在用）；
   接上时用到的是"字段 + `tolists()` 的构造"，`slice_request`/`to_nested_list`/`merge`
   暂无调用方——一并搬来是为了保持 CSR 语义与上游逐行一致，不必再回头改这个文件。
4. `from __future__ import annotations` 保留：`SamplerOutput` 里 `SamplingMaskTensors`
   这个注解在上游也是"先引用、后定义"（类型注解不求值），改成一个文件后同样成立。

掩码这条通路要解决什么问题：采样会让某些 token 的 logits 变成 `-inf`（结构化输出的语法掩码、
`min_p`/`top-k` 等）；异步调度（70 关）里 CPU 需要在**不等采样器结果**的前提下知道
"这一轮每个位置还允许哪些 token"，于是 GPU 侧把 finite-logit 支持集**按位打包**
（`packed_mask`，每 8 个 token 一个字节、低位在前）并数出个数（`counts`），
交给 CPU 时再展成 CSR（`SamplingMaskLists`）——比传整份 `[num_reqs, vocab]` 布尔掩码省得多。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from minivllm.outputs import LogprobsTensors
from minivllm.triton_utils import tl, triton


# NOTE(本仓库)：这一段逐行来自上游 `vllm/v1/outputs.py`（`SamplingMaskLists`）。
# 上游 `vllm/v1/worker/gpu/sample/output.py` 是从 `vllm.v1.outputs` import 它的；
# 本项目 `minivllm/outputs.py` 没有这个类，所以整段搬到这里（字段与语义不变）。
class SamplingMaskLists(NamedTuple):
    # [num_kept_tokens]
    token_ids: np.ndarray
    # [num_generated_tokens + 1]
    offsets: np.ndarray
    # [num_reqs + 1]
    cu_num_generated_tokens: list[int] | None = None

    def slice_request(self, req_idx: int, num_positions: int) -> "SamplingMaskLists":
        if self.cu_num_generated_tokens is None:
            start_idx = req_idx
        else:
            start_idx = self.cu_num_generated_tokens[req_idx]
        end_idx = start_idx + num_positions
        flat_start = self.offsets[start_idx]
        flat_end = self.offsets[end_idx]
        return SamplingMaskLists(
            self.token_ids[flat_start:flat_end],
            self.offsets[start_idx : end_idx + 1] - flat_start,
            None,
        )

    def to_nested_list(self) -> list[list[int]]:
        """Convert CSR representation to ``list[list[int]]``."""
        return [
            self.token_ids[int(self.offsets[i]) : int(self.offsets[i + 1])].tolist()
            for i in range(len(self.offsets) - 1)
        ]

    @staticmethod
    def merge(chunks: Sequence["SamplingMaskLists"]) -> "SamplingMaskLists":
        token_ids = np.concatenate([chunk.token_ids for chunk in chunks])
        counts = np.concatenate([np.diff(chunk.offsets) for chunk in chunks])
        offsets = np.empty(len(counts) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, dtype=np.int64, out=offsets[1:])
        return SamplingMaskLists(token_ids, offsets)


@dataclass
class SamplerOutput:
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None
    num_nans: torch.Tensor | None
    num_sampled: torch.Tensor | None
    num_rejected: torch.Tensor | None = None
    sampling_mask_tensors: SamplingMaskTensors | None = None


@triton.jit
def _pack_sampling_mask_kernel(
    logits_ptr,
    logits_row_stride,
    logits_col_stride,
    num_sampled_tokens_ptr,
    packed_mask_ptr,
    packed_mask_row_stride,
    counts_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_active = tl.load(num_sampled_tokens_ptr + req_idx) > 0
    count = tl.zeros((), dtype=tl.int32)

    for start_idx in range(0, vocab_size, BLOCK_SIZE):
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        valid = offsets < vocab_size
        logits = tl.load(
            logits_ptr + req_idx * logits_row_stride + offsets * logits_col_stride,
            mask=valid,
            other=-float("inf"),
        )
        keep = (logits > -float("inf")) & (logits < float("inf")) & is_active
        count += tl.sum(keep).to(tl.int32)

        keep = tl.reshape(keep.to(tl.int32), (BLOCK_SIZE // 8, 8))
        bit_shifts = tl.arange(0, 8)[None, :]
        packed = tl.sum(keep << bit_shifts, axis=1).to(tl.uint8)
        byte_offsets = start_idx // 8 + tl.arange(0, BLOCK_SIZE // 8)
        tl.store(
            packed_mask_ptr + req_idx * packed_mask_row_stride + byte_offsets,
            packed,
            mask=byte_offsets < tl.cdiv(vocab_size, 8),
        )

    tl.store(counts_ptr + req_idx, count)


class SamplingMaskTensors(NamedTuple):
    """Bit-packed device-side sampling mask data pending async D2H."""

    # [num_requests, ceil(vocab_size / 8)]
    packed_mask: torch.Tensor
    # [num_requests]
    counts: torch.Tensor
    vocab_size: int

    @classmethod
    def from_logits(
        cls,
        logits: torch.Tensor,
        num_sampled_tokens: torch.Tensor,
    ) -> SamplingMaskTensors:
        """Pack the finite-logit support for requests that sampled tokens."""
        num_reqs, vocab_size = logits.shape
        packed_width = (vocab_size + 7) // 8

        packed_mask = torch.empty(
            (num_reqs, packed_width), dtype=torch.uint8, device=logits.device
        )
        counts = torch.empty(num_reqs, dtype=torch.int32, device=logits.device)
        _pack_sampling_mask_kernel[(num_reqs,)](
            logits,
            logits.stride(0),
            logits.stride(1),
            num_sampled_tokens,
            packed_mask,
            packed_mask.stride(0),
            counts,
            vocab_size,
            BLOCK_SIZE=8192,
        )

        return cls(packed_mask, counts, vocab_size)

    def to_cpu_nonblocking(self) -> SamplingMaskTensors:
        if self.packed_mask.device.type == "cpu":
            return self
        return SamplingMaskTensors(
            self.packed_mask.to("cpu", non_blocking=True),
            self.counts.to("cpu", non_blocking=True),
            self.vocab_size,
        )

    def tolists(self, num_sampled_tokens: np.ndarray) -> SamplingMaskLists:
        """Convert the packed masks to the scheduler's CSR representation."""
        sampled_rows = np.flatnonzero(num_sampled_tokens)
        counts = self.counts.cpu().numpy()[sampled_rows]
        offsets = np.empty(len(counts) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, dtype=np.int64, out=offsets[1:])
        unpacked = np.unpackbits(
            self.packed_mask.cpu().numpy()[sampled_rows],
            axis=1,
            count=self.vocab_size,
            bitorder="little",
        )
        token_ids = np.nonzero(unpacked)[1].astype(np.int32, copy=False)
        return SamplingMaskLists(
            token_ids=token_ids,
            offsets=offsets,
            cu_num_generated_tokens=np.cumsum(
                np.concatenate(([0], num_sampled_tokens))
            ).tolist(),
        )
