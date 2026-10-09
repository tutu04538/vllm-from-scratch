"""V2 的采样器（对应 vLLM `v1/worker/gpu/sample/sampler.py`）。

与 V1 采样器的差别全在**寻址**：V2 按 **slot** 取采样参数、按 **logits 行**施加约束：

    expanded_idx_mapping [num_logits]   logits 第 i 行属于哪个 slot
    expanded_local_pos   [num_logits]   第 i 行是该请求的第几行（温度/惩罚按行取）
    idx_mapping          [num_reqs]     batch 行 → slot

于是这一层**不需要 CPU 先按行展开一份 `SampleMetadata`**：所有参数都在 UVA 张量里按 slot 放着，
内核自己按 `expanded_idx_mapping` 取（`sampling_states.temperature.gpu[expanded_idx_mapping]`）。

本关支持的处理顺序（与上游一致，`apply_sampling_params`）：
**惩罚 → 温度 → min_p → top_k/top_p →（gumbel 采样）**。

裁剪（差异账本逐条）：
- `logit_bias` / `bad_words` / `thinking_budget`：本仓库在**请求期**明确拒绝这些字段
  （68 关白名单），所以这里不建对应状态类——建了也没有生产者。
- `flashinfer` 采样后端：本仓库没有 flashinfer 依赖 → 恒走 Torch 的 `apply_top_k_top_p` + gumbel。
- `num_nans` 指标：无通路 → 恒 None。
"""

import numpy as np
import torch

from ....config import PROCESSED_LOGPROBS_MODES
from ....sample.ops.topk_topp_sampler import apply_top_k_top_p
from ....sampling_params import SamplingParams
from ..input_batch import InputBatch, get_num_sampled_and_rejected
from ..states import RequestState
from .gumbel import gumbel_sample
from .logprob import LogprobTokenIdsState, compute_topk_scores
from .output import SamplerOutput
from .penalties import PenaltiesState
from .states import NO_LOGPROBS, SamplingStates


class Sampler:
    def __init__(
        self,
        max_num_reqs: int,
        vocab_size: int,
        device: torch.device,
        req_states: RequestState,
        logprobs_mode: str = "raw_logprobs",
        num_speculative_tokens: int = 1,
        use_fp64_gumbel: bool = False,
    ):
        self.logprobs_mode = logprobs_mode
        # 上游读 `VLLM_COMPUTE_NANS_IN_LOGITS`（默认 False）；本仓库没有 NaN 指标通路。
        self.compute_nans = False
        self.use_fp64_gumbel = use_fp64_gumbel

        self.req_states = req_states
        self.sampling_states = SamplingStates(max_num_reqs, vocab_size)
        self.penalties_state = PenaltiesState(req_states)
        # 上游同一条调用链：`logprob_token_ids` 允许请求指定"只交付这几个 token 的 logprobs"。
        # 本仓库该字段在**请求期**就明确拒绝（68 关白名单），所以这个状态类实际上恒为空——
        # 但路径照上游接好（`compute_topk_scores` 的慢路径因此有真实生产者，而不是死代码）。
        self.logprob_token_ids_state = LogprobTokenIdsState(max_num_reqs, device)
        self.needs_logits_processing = np.zeros(max_num_reqs, dtype=bool)
        self.num_speculative_tokens = num_speculative_tokens
        self.device = device

    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None:
        self.sampling_states.add_request(req_idx, sampling_params)
        self.penalties_state.add_request(req_idx, sampling_params)
        self.logprob_token_ids_state.add_request(req_idx, sampling_params)

        states = self.sampling_states
        temperature = states.temperature.np[req_idx]
        self.needs_logits_processing[req_idx] = (
            self.penalties_state.use_penalty[req_idx]
            or (temperature != 0.0 and temperature != 1.0)
            or states.min_p.np[req_idx] != 0.0
            or states.top_k.np[req_idx] != states.vocab_size
            or states.top_p.np[req_idx] != 1.0
        )

    def apply_staged_writes(self) -> None:
        self.sampling_states.apply_staged_writes()
        self.penalties_state.apply_staged_writes()
        self.logprob_token_ids_state.apply_staged_writes()

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
    ) -> SamplerOutput:
        expanded_idx_mapping = input_batch.expanded_idx_mapping
        idx_mapping = input_batch.idx_mapping
        idx_mapping_np = input_batch.idx_mapping_np
        cu_num_logits_np = input_batch.cu_num_logits_np
        expanded_local_pos = input_batch.expanded_local_pos
        pos = input_batch.positions[input_batch.logits_indices]
        input_ids = input_batch.input_ids[input_batch.logits_indices]

        max_num_logprobs = self.sampling_states.max_num_logprobs(idx_mapping_np)
        max_per_req_token_ids = self.logprob_token_ids_state.max_num_token_ids(
            idx_mapping_np
        )
        return_logprobs = max_num_logprobs != NO_LOGPROBS or max_per_req_token_ids > 0

        sampled, processed_logits = self.sample(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            return_logprobs=return_logprobs,
        )

        if return_logprobs:
            if self.logprobs_mode in PROCESSED_LOGPROBS_MODES:
                logits = processed_logits
            expanded_logits = logits.shape[0] != idx_mapping_np.shape[0]
            cu_num_logits = cu_num_logits_np.tolist() if expanded_logits else None
            num_logprobs = max_num_logprobs if max_num_logprobs != NO_LOGPROBS else 0
            logprobs_tensors = compute_topk_scores(
                logits,
                num_logprobs,
                sampled,
                cu_num_logits,
                logprob_token_ids_state=self.logprob_token_ids_state,
                expanded_idx_mapping=input_batch.expanded_idx_mapping,
                max_per_req_token_ids=max_per_req_token_ids,
                logits_mode=self.logprobs_mode in ("raw_logits", "processed_logits"),
            )
        else:
            logprobs_tensors = None

        # 1 sampled token per request, except chunked-prefill requests
        # (seq_len < prefill_len) which aren't done prefilling and produce no
        # output token. num_rejected is always 0 here (one logit per request).
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            input_batch.seq_lens.new_ones(input_batch.num_reqs),
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.req_states.prefill_len.gpu,
        )

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=logprobs_tensors,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
        return sampler_output

    def apply_sampling_params(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        skip_top_k_top_p: bool = False,
    ) -> torch.Tensor:
        if not np.any(self.needs_logits_processing[idx_mapping_np]):
            return logits

        # Copy logits to a new FP32 tensor.
        logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)

        # Apply penalties in place.
        self.penalties_state.apply_penalties(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            input_ids,
            expanded_local_pos,
        )

        # Apply temperature in place.
        self.sampling_states.apply_temperature(
            logits, expanded_idx_mapping, idx_mapping_np
        )

        # Apply min_p in place.
        self.sampling_states.apply_min_p(logits, expanded_idx_mapping, idx_mapping_np)

        if skip_top_k_top_p:
            return logits

        # Apply top_k and/or top_p. This might or might not return a new tensor.
        return self.sampling_states.apply_top_k_top_p(
            logits, expanded_idx_mapping, idx_mapping_np
        )

    def sample(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        return_logprobs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        processed_logits = self.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            skip_top_k_top_p=True,
        )
        top_k, top_p = self.sampling_states.get_top_k_top_p(
            expanded_idx_mapping, idx_mapping_np
        )
        # 上游这里优先走 flashinfer 采样内核；本仓库没有该依赖 → 恒走 Torch 路径
        # （`apply_top_k_top_p` + gumbel，与上游的 else 分支逐行相同）。
        processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)
        sampled = gumbel_sample(
            processed_logits,
            expanded_idx_mapping,
            self.sampling_states.temperature.gpu,
            self.sampling_states.seeds.gpu,
            pos,
            apply_temperature=False,
            use_fp64=self.use_fp64_gumbel,
        )
        return sampled, processed_logits
