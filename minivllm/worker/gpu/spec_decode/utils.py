"""V2 投机通路的两个小工具（对应 vLLM `v1/worker/gpu/spec_decode/utils.py`）。

`DraftTokensHandler`：把**草稿 token 交给调度器**的那一步。普通情况下草稿不需要回 CPU——
调度器下一轮直接从 worker 侧的状态里拿（V2 的 `req_states.draft_tokens` 常驻 GPU，
结构化输出要校验时也走 GPU 侧掩码）。**只有"结构化输出 + 投机"同时开**时，
grammar 的校验在调度器进程里做，才必须把草稿拷回 CPU：这条 D2H 走侧流、用事件同步，
不阻塞主流程。

`get_parallel_drafting_token_id`：并行提议（PARD/P-EAGLE）的 mask token 从 checkpoint
配置里取，取值顺序与上游一致（dflash → 顶层 mask_token_id → dspark → pard → ptd）；
**一个都没有就报错**——静默退化成某个默认 token 会让 K 枚草稿全部基于错的输入。
"""

import numpy as np
import torch

from ....outputs import DraftTokenIds
from ..input_batch import InputBatch


def async_copy_to_np(tensor: torch.Tensor) -> np.ndarray:
    """把张量拷回 pinned CPU 内存（对应上游 `v1/worker/gpu/async_utils.py` 的同名函数）。"""
    return tensor.to("cpu", non_blocking=True).numpy()


class DraftTokensHandler:
    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.copy_stream = torch.cuda.Stream(device) if torch.cuda.is_available() else None
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = (
            torch.cuda.Event(blocking=True) if torch.cuda.is_available() else None
        )

        self.req_ids: list[str] = []
        self.draft_tokens_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0

    def set_draft_tokens(
        self, input_batch: InputBatch, draft_tokens: torch.Tensor
    ) -> None:
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]
        if not input_batch.has_structured_output_reqs:
            # No draft token validation needs to be performed by
            # the scheduler for this batch.
            self.draft_tokens_np = None
            return

        # For spec decoding + structured outputs, we must transfer the
        # draft tokens back to the scheduler for grammar validation.
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            self.draft_tokens_np = async_copy_to_np(draft_tokens)
            # draft_tokens is a temporary allocation on the main stream and read here on
            # copy_stream; without record_stream, the caching allocator may reuse its
            # memory before the async copy executes.
            draft_tokens.record_stream(self.copy_stream)
            self.copy_event.record()

    def get_draft_tokens(self) -> DraftTokenIds | None:
        if self.draft_tokens_np is not None:
            self.copy_event.synchronize()
            draft_token_ids = self.draft_tokens_np.tolist()
        else:
            # This case only happens when async scheduling is disabled.
            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]
        return DraftTokenIds(self.req_ids, draft_token_ids)


def get_parallel_drafting_token_id(hf_config) -> int:
    """Resolve the mask token id used for parallel drafting slots.

    Checks (in order): `dflash_config.mask_token_id`, top-level `mask_token_id`,
    `dspark_noise_token_id`, `pard_token`, `ptd_token_id`. Raises ValueError if
    none are present.
    """
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    if "mask_token_id" in dflash_config:
        return int(dflash_config["mask_token_id"])
    if getattr(hf_config, "mask_token_id", None) is not None:
        return int(hf_config.mask_token_id)
    if hasattr(hf_config, "dspark_noise_token_id"):
        return int(hf_config.dspark_noise_token_id)
    if hasattr(hf_config, "pard_token"):
        return int(hf_config.pard_token)
    if hasattr(hf_config, "ptd_token_id"):
        return int(hf_config.ptd_token_id)
    raise ValueError(
        "Model config must specify `dflash_config.mask_token_id`,"
        " `mask_token_id`, `dspark_noise_token_id`, `pard_token`, or"
        " `ptd_token_id` for parallel drafting."
    )
