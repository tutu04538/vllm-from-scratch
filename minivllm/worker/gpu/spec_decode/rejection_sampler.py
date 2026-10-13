"""V2 的验证采样器（对应 vLLM `v1/worker/gpu/spec_decode/rejection_sampler.py`）。

职责与 V1 的 `minivllm/sample/rejection_sampler.py` 相同（同一套拒绝采样数学，059 §3），
但**寻址方式与切分位置不同**：

- 输入是"**扁平**的 logits + `cu_num_logits`（含前导 0）"，而不是 `[B, K+1]` 的规整张量：
  每条请求占 `1 + K_i` 行（K_i 可以不同，chunked prefill 的请求 0 行），
  `expanded_idx_mapping`/`expanded_local_pos` 说明每一行属于哪个 slot、是该请求的第几行。
- 采样参数（温度/种子/惩罚）按 **slot** 取（`temperature.gpu[expanded_idx_mapping]`），
  不需要 CPU 先按行展开一份。
- 结果 `sampled [num_logits]` + `num_sampled [num_reqs]` 都是 GPU 张量，
  `num_rejected = num_logits - num_sampled` 由内核算出（`get_num_sampled_and_rejected`）。

**三种验证方式**（75 关补齐后，与上游 V2 一致；分派点只有 `__init__` 一处）：

    standard    Leviathan et al. 2023：逐位置概率比检验，第一次拒绝就停。
    block       Sun et al. 2024（arXiv:2403.10444）：整块联合验证，阈值 h 由"前缀联合比
                `p_i`"与"下一位置的残差质量"给出，接受长度取**最后**通过测试的位置；
                拒绝位的恢复分布是 `max(p_tau * target - draft, 0)` 的归一化形式（不是 standard
                的 `max(target - draft, 0)`）。递推在
                `rejection_sampler_utils._compute_cumulative_log_p_kernel` 里，**不能**换成
                "每步 min(1, p/q) 再累乘"（075 §3.2）。
    synthetic   不按 p/q 判定，按配置给的**条件**接受率 `u < c_i` 接受草稿——用于把"接受长度"
                当实验输入（调度/显存/图的行为与模型质量解耦）。接受率表在配置期从
                `synthetic_acceptance_rates`（无条件）换算成条件率。
                ⚠️ synthetic **不证明模型分布无损**（它绕过了概率比检验），交付物必须写明这一点
                （075 §4）。

`block` 与 `synthetic` 互斥：块验证的阈值与残差质量都建立在真实 p/q 上，塞一张合成率表进去
就没有意义了——`rejection_sampler_utils.rejection_sample` 里有一条 assert 挡着（上游同款）。
"""

from collections.abc import Iterable, Iterator

import numpy as np
import torch

from ....config import SpeculativeConfig
from ....outputs import LogprobsTensors
from ....spec_decode.utils import unconditional_to_conditional_rates
from ....triton_utils import tl, triton
from ..input_batch import InputBatch, get_num_sampled_and_rejected
from ..sample.logprob import compute_topk_scores
from ..sample.output import SamplerOutput
from ..sample.sampler import Sampler
from ..sample.states import NO_LOGPROBS
from .rejection_sampler_utils import rejection_sample

# Cap on the FP32 target-logits buffer materialized by apply_sampling_params.
MAX_CHUNK_BYTES = 2**30  # 1GB
_FP32_BYTES = 4


def get_max_chunk_logits(vocab_size: int) -> int:
    """Largest number of logits rows one verification chunk may hold."""
    return max(1, MAX_CHUNK_BYTES // (vocab_size * _FP32_BYTES))


def _iter_request_chunks(
    cu_num_logits: np.ndarray, max_chunk_logits: int
) -> Iterator[tuple[int, int]]:
    """Yield maximally packed request ranges without splitting requests."""
    assert max_chunk_logits > 0
    num_reqs = cu_num_logits.size - 1
    start = 0
    while start < num_reqs:
        max_logit = int(cu_num_logits[start]) + max_chunk_logits
        end = int(np.searchsorted(cu_num_logits, max_logit, side="right") - 1)
        end = min(num_reqs, max(start + 1, end))
        yield start, end
        start = end


@triton.jit
def _flatten_sampled_kernel(
    # [num_logits]
    flat_sampled_ptr,
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    for i in range(num_sampled):
        token_id = tl.load(sampled_ptr + req_idx * sampled_stride + i)
        tl.store(flat_sampled_ptr + start_idx + i, token_id)


def flatten_sampled(
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_logits: int,
) -> torch.Tensor:
    """`[num_reqs, K+1]` 的采样结果按请求压平成 `[num_logits]`（上游同款内核包装）。"""
    num_reqs = cu_num_logits.shape[0] - 1
    flat_sampled = torch.zeros(
        num_logits, dtype=sampled.dtype, device=sampled.device
    )
    _flatten_sampled_kernel[(num_reqs,)](
        flat_sampled,
        sampled,
        sampled.stride(0),
        num_sampled,
        cu_num_logits,
        num_warps=1,
    )
    return flat_sampled


class RejectionSampler:
    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig,
        device: torch.device,
    ):
        self.sampler = sampler
        self.num_speculative_steps = spec_config.num_speculative_tokens
        # 75 关：验证方式的**唯一分派点**（上游 `rejection_sampler.py:84-96` 逐行同义）。
        # 两个开关互斥：`synthetic` 只放一张条件率表、`block` 只置一个布尔；`standard` 两者都不设。
        rejection_sample_method = spec_config.rejection_sample_method
        self.use_block_verification: bool = False
        self.synthetic_conditional_rates: torch.Tensor | None = None
        if rejection_sample_method == "synthetic":
            # 配置期已经把 rates/length 归一成 `synthetic_acceptance_rates`（长度 == K）；这里的
            # assert 只是"有人绕过配置直接建"时的兜底，与上游同款。
            assert spec_config.synthetic_acceptance_rates is not None
            self.synthetic_conditional_rates = torch.tensor(
                unconditional_to_conditional_rates(
                    spec_config.synthetic_acceptance_rates
                ),
                dtype=torch.float32,
                device=device,
            )
        elif rejection_sample_method == "block":
            self.use_block_verification = True

    def _get_logprobs_tensors(
        self,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        logits: torch.Tensor,
        cu_num_logits: torch.Tensor,
        cu_num_logits_np: np.ndarray,
        max_num_logprobs: int,
    ) -> LogprobsTensors | None:
        if max_num_logprobs == NO_LOGPROBS:
            return None

        num_reqs = cu_num_logits.shape[0] - 1
        num_logits = logits.shape[0]
        flat_sampled = flatten_sampled(sampled, num_sampled, cu_num_logits, num_logits)
        expanded_logits = num_logits != num_reqs
        return compute_topk_scores(
            logits,
            max_num_logprobs,
            flat_sampled,
            cu_num_logits_np.tolist() if expanded_logits else None,
            logits_mode=self.sampler.logprobs_mode
            in ("raw_logits", "processed_logits"),
        )

    def _verify(
        self,
        logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        processed_logits = self.sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            draft_sampled,
            expanded_local_pos,
        )
        sampled, num_sampled = rejection_sample(
            processed_logits,
            draft_logits,
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
            self.num_speculative_steps,
            self.synthetic_conditional_rates,
            use_fp64=self.sampler.use_fp64_gumbel,
            use_block_verification=self.use_block_verification,
        )
        return processed_logits, sampled, num_sampled

    def _verify_in_chunks(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        max_chunk_logits: int,
        max_num_logprobs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, LogprobsTensors | None]:
        cu_num_logits_np = input_batch.cu_num_logits_np
        use_processed_logits = self.sampler.logprobs_mode in (
            "processed_logprobs",
            "processed_logits",
        )
        num_reqs = input_batch.num_reqs

        if logits.shape[0] <= max_chunk_logits:
            # 单块：整批一次验完。**这条分支同时是 79 关的保护**——adaptive verification 会
            # 把 logits 压紧（只留通过置信度筛选的行）却**不更新** `cu_num_logits_np`（它保留
            # 压紧前的布局），于是那些"陈旧的 CPU 前缀和"不能再用来挑块边界；79 关给压紧后的
            # 批做了预算上限，保证它一定落在"行数 ≤ max_chunk_logits"这一档里，所以这里直接
            # 一整块（上游同款注释与理由）。
            request_chunks: Iterable[tuple[int, int]] = ((0, num_reqs),)
        else:
            request_chunks = _iter_request_chunks(cu_num_logits_np, max_chunk_logits)

        sampled_chunks: list[torch.Tensor] = []
        num_sampled_chunks: list[torch.Tensor] = []
        logprobs_chunks: list[LogprobsTensors] = []

        for start, end in request_chunks:
            lo = int(cu_num_logits_np[start])
            hi = int(cu_num_logits_np[end])
            chunk_cu_num_logits_np = cu_num_logits_np[start : end + 1] - lo
            chunk_cu_num_logits = input_batch.cu_num_logits[start : end + 1] - lo
            # draft_logits uses persistent request-state indices and stays global.
            processed_logits, sampled, num_sampled = self._verify(
                logits[lo:hi],
                draft_logits,
                draft_sampled[lo:hi],
                pos[lo:hi],
                chunk_cu_num_logits,
                input_batch.idx_mapping[start:end],
                input_batch.idx_mapping_np[start:end],
                input_batch.expanded_idx_mapping[lo:hi],
                input_batch.expanded_local_pos[lo:hi],
            )
            chunk_logprobs = self._get_logprobs_tensors(
                sampled,
                num_sampled,
                processed_logits if use_processed_logits else logits[lo:hi],
                chunk_cu_num_logits,
                chunk_cu_num_logits_np,
                max_num_logprobs,
            )
            if chunk_logprobs is not None:
                logprobs_chunks.append(chunk_logprobs)
            del processed_logits
            sampled_chunks.append(sampled)
            num_sampled_chunks.append(num_sampled)

        if len(sampled_chunks) == 1:
            logprobs_tensors = logprobs_chunks[0] if logprobs_chunks else None
            return sampled_chunks[0], num_sampled_chunks[0], logprobs_tensors

        logprobs_tensors = None
        if logprobs_chunks:
            expanded_logits = logits.shape[0] != input_batch.num_reqs
            logprobs_tensors = LogprobsTensors.cat(
                logprobs_chunks,
                cu_num_generated_tokens=(
                    cu_num_logits_np.tolist() if expanded_logits else None
                ),
            )

        sampled = torch.cat(sampled_chunks)
        num_sampled = torch.cat(num_sampled_chunks)
        return sampled, num_sampled, logprobs_tensors

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        # 上游这里先算 num_nans（`metrics/logits.py` 的内核 + `VLLM_COMPUTE_NANS_IN_LOGITS`
        # 开关，默认关）。本仓库没有这套指标通路 → 恒 None（差异账本"未接指标"条）。
        num_nans = None

        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        pos = input_batch.positions[input_batch.logits_indices]

        max_num_logprobs = self.sampler.sampling_states.max_num_logprobs(
            input_batch.idx_mapping_np
        )
        chunk_logit_limit = get_max_chunk_logits(logits.shape[1])
        sampled, num_sampled, logprobs_tensors = self._verify_in_chunks(
            logits,
            input_batch,
            draft_logits,
            draft_sampled,
            pos,
            chunk_logit_limit,
            max_num_logprobs,
        )

        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu,
        )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
