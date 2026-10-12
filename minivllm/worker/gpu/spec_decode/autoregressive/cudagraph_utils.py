"""草稿侧 CUDA Graph 管理器（对应 vLLM
`v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py`）。

它给基类 `CudaGraphManager` 补上的**唯一**一件事：把"捕获一张图"翻译成草稿模型能跑的调用
——即"准备一份假输入 → 调提议者的 `_prefill` / `_generate_draft` / `_generate_fused_drafts`"。
提议循环本身仍写在 `speculator.py`，这里不复制它（需求 074 §2 的划分）。

与上游的一处重要差别（**输入缓冲是"谁的那一份"**）：

    prefill 图  用 **target 的** `InputBuffers` + **target 的**注意力元数据构造器。
                为什么：草稿 prefill 复用 target 本轮的注意力元数据与 slot_mapping
                （批形状与 KV 布局完全相同），所以那份假输入也必须写在 target 的常驻缓冲上，
                否则图里记的指针与运行期不是同一批。
    decode 图   用**草稿自己的** `InputBuffers`（每次草稿步只有 1 行/请求，元数据也是草稿
                自己按 `seq_lens + step` 造的）。

上游在这里还有 `num_tokens_across_dp`（DP 同步）与 breakable CUDA graph（无 torch.compile
的 PW 图）两个分支，本仓库单卡且没有编译路径，两者都没有对应物——**不留空壳**。
"""

from collections.abc import Callable

import torch

from .....config import CUDAGraphMode
from ...block_table import BlockTables
from ...cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
    prepare_inputs_to_capture,
)
from ...input_batch import InputBatch, InputBuffers


class SpeculatorCudaGraphManager(CudaGraphManager):
    """草稿 prefill / decode 图的捕获与重放（上游同名类）。

    上游类的 docstring 里那句要点照抄：**每次热身与捕获都要重新造一份假输入与注意力元数据**。
    复用上一次捕获时建好的元数据会让内核按**过期的缓冲内容**执行（共享常驻缓冲的
    `query_start_loc` / `seq_lens` / 调度元数据都是每张图各写一份的），而 cuBLAS/Triton 的
    自动调优又必须在图外先发生一次。
    """

    def capture(
        self,
        forward_fn: Callable,
        build_attn_metadata: Callable[[InputBatch], dict],
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        def create_forward_fn(
            desc: BatchExecutionDescriptor,
            warmup: bool,
        ) -> Callable[[CUDAGraphMode], None]:
            num_tokens = desc.num_tokens
            # PIECEWISE 的 desc 里 num_reqs 是 None（不补请求行）；本仓库的草稿图只有 FULL，
            # 这里保留上游的兜底写法，免得以后的分段图静默取到 None。
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
            attn_state = prepare_inputs_to_capture(
                num_reqs,
                num_tokens,
                input_buffers,
                block_tables,
                build_attn_metadata,
                full_cudagraph=desc.cg_mode == CUDAGraphMode.FULL,
            )

            return lambda cg_mode: forward_fn(
                num_reqs,
                num_tokens,
                attn_state.attn_metadata,
                attn_state.slot_mappings,
                cg_mode,
            )

        super().capture(create_forward_fn, progress_bar_desc)
