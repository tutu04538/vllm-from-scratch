"""V2 的自回归提议者（对应 vLLM `v1/worker/gpu/spec_decode/autoregressive/speculator.py`）。

EAGLE/EAGLE3/MTP（以及 DFlash/DSpark 的串行部分）在 V2 里共用这套流程，只有"模型从哪来"
与"位置要不要步进"不同。**关键差别在输入组装**：V1 用 `copy_and_expand_eagle_inputs_kernel`
把 target 的行扩容/左移；V2 直接在 draft 的输入缓冲上做同一件事（`_prepare_prefill_inputs_kernel`）：

    draft_input_ids[i-1] = target_input_ids[i]        （整块左移一行）
    draft_input_ids[query_start + query_len - 1] = next_token
        其中 query_len = target_query_len - num_rejected，next_token = last_sampled
        （还在 chunked prefill 的请求则取 `next_prefill_tokens`）
    positions 原样拷贝（一行不差——位置是绝对量，左移的是**行**不是位置）

于是"锚点（采样行）= 最后一枚**有效**行"这条 63 关修出来的不变量，在 V2 里由
`query_len -= num_rejected` 一句话表达，且 `num_rejected` 是 GPU 张量——**CPU 不需要知道**
被拒了几枚（这就是 V2 能把异步调度做起来的原因）。

draft 自回归步（step = 1..K-1）每步只喂 **1 行/请求**：上一步采出的草稿当输入 token、
上一步的 hidden 当输入特征，position/seq_len 各 +1，`slot_mapping` 按 block table 现算。

**裁剪**：CUDA Graph（74 关）、融合多步 decode（需要注意力后端支持
`update_draft_decode_metadata`，本仓库后端没有 → 走上游的"逐步重建 metadata"回退路径）、
多模态输入、DP 同步、EPLB。
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .....config import VllmConfig
from .....forward_context import BatchDescriptor, set_forward_context
from .....triton_utils import tl, triton
from ...block_table import PAD_SLOT_ID
from ...input_batch import InputBatch, InputBuffers
from ..speculator import DraftModelSpeculator


class AutoRegressiveSpeculator(DraftModelSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )
        self.current_draft_step = torch.tensor(0, dtype=torch.int64, device=device)
        self.last_token_indices = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )

        # 注意力元数据构造器（草稿层自己的那份；块表与 target 共享）。
        from .....attention.backends.torch_sdpa import TorchAttentionBackend

        self.metadata_builder = TorchAttentionBackend.get_builder_cls()(
            vllm_config.cache_config.block_size
        )
        # 本关 eager：图管理器恒为空（74 关补）。
        self.prefill_cudagraph_manager = None
        self.decode_cudagraph_manager = None
        self.use_fused_multi_step_decode = False

    # -------- 生命周期钩子（上游同名；本仓库被子类覆写用于"注意力标志"这类切换） --------

    def on_prefill_begin(self, num_reqs: int) -> None: ...

    def on_prefill_end(self, num_reqs: int) -> None: ...

    def on_multi_step_decode_begin(self, num_reqs: int) -> None: ...

    def on_multi_step_decode_end(self, num_reqs: int) -> None: ...

    @property
    def advance_draft_positions(self) -> bool:
        """每一步是否推进 positions/seq_lens（EAGLE/标准 MTP：True；Q-only 的 MTP：False）。"""
        return True

    def init_cudagraph_manager(self, cudagraph_mode) -> None:
        """V2 的图属 74 关：本关只允许 eager（NONE），否则明确报错。"""
        if getattr(cudagraph_mode, "is_valid_runtime_mode", None) is not None and (
            cudagraph_mode.value if hasattr(cudagraph_mode, "value") else str(cudagraph_mode)
        ) != "NONE":
            raise NotImplementedError(
                "V2 提议者的 CUDA Graph（prefill/decode 两张图、融合多步 decode）属 74 关；"
                f"73 关只支持 eager，收到 cudagraph_mode={cudagraph_mode}"
            )

    def capture(self) -> None:
        """没有图管理器就是空操作（本关只做 eager）。"""
        return None

    # -------- 主流程 --------

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [num_prefill_lookahead, max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        dummy_run: bool = False,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_tokens = input_batch.num_tokens
        num_tokens_padded = input_batch.num_tokens_after_padding
        num_reqs = input_batch.num_reqs
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(
            max_seq_len + self.num_speculative_steps, self.max_model_len
        )

        # NOTE(woosuk): To avoid CPU-GPU synchronization without CPU knowing the
        # number of rejected tokens, we maintain the size of input_ids and
        # hidden_states the same as the target model's. This means, we pad each
        # request's query length to include any rejected positions. By doing so,
        # we can also reuse the attention metadata (e.g., query_start_loc,
        # seq_lens) of the target model.
        if aux_hidden_states:
            assert self.method == "eagle3"
            hidden_states = self.model.combine_hidden_states(
                torch.cat(aux_hidden_states, dim=-1)
            )
        else:
            hidden_states = last_hidden_states
        self.hidden_states[:num_tokens_padded].copy_(hidden_states)

        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
        )

        # Get the input ids and last token indices for the speculator.
        prepare_prefill_inputs(
            self.last_token_indices,
            self.current_draft_step,
            self.input_buffers,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self.max_num_reqs,
        )

        self.on_prefill_begin(num_reqs)
        # 本关 eager：target 的 attention metadata 与 slot_mapping 可以直接给 draft prefill
        # 用，因为批形状与 KV 布局完全相同（上游注释同款）。
        self._prefill(
            num_reqs,
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
        )
        self.on_prefill_end(num_reqs)

        if self.num_speculative_steps == 1:
            # Early exit.
            return self.draft_tokens[:num_reqs, :1]

        # Prepare the inputs for the decode steps.
        prepare_decode_inputs(
            self.draft_tokens[:num_reqs, 0],
            input_batch.seq_lens,
            num_rejected,
            self.input_buffers,
            self.max_model_len,
            self.max_num_reqs,
            advance_draft_positions=self.advance_draft_positions,
        )

        self.on_multi_step_decode_begin(num_reqs)
        # Generate the remaining num_speculative_steps - 1 draft tokens.
        self._multi_step_decode(num_reqs, input_batch.seq_lens_cpu_upper_bound)
        self.on_multi_step_decode_end(num_reqs)

        return self.draft_tokens[:num_reqs]

    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=self._runtime_mode(),
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            ret_hidden_states = self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
            )
        # Some MTP models declare a single-tensor contract but return
        # (logits_hidden, feedback_hidden) for final-norm correctness.
        if isinstance(ret_hidden_states, tuple):
            last_hidden_states, hidden_states = ret_hidden_states
        else:
            last_hidden_states = ret_hidden_states
            hidden_states = ret_hidden_states
        return last_hidden_states, hidden_states

    @staticmethod
    def _runtime_mode():
        from .....config import CUDAGraphMode

        return CUDAGraphMode.NONE

    def _prefill(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
    ) -> None:
        last_token_indices = self.last_token_indices[:num_reqs]
        positions = self.input_buffers.positions[last_token_indices]
        idx_mapping = self.idx_mapping[:num_reqs]

        last_hidden_states, hidden_states = self._run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
        )
        sample_hidden_states = last_hidden_states[last_token_indices]

        self.draft_tokens[:num_reqs, 0] = self.sample_draft(
            sample_hidden_states,
            positions,
            idx_mapping,
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
        )
        if last_hidden_states is hidden_states:
            self.hidden_states[:num_reqs] = sample_hidden_states
        else:
            self.hidden_states[:num_reqs] = hidden_states[last_token_indices]
        self.input_buffers.positions[:num_reqs] = positions

    def _multi_step_decode(
        self,
        num_reqs: int,
        seq_lens_cpu_upper_bound: torch.Tensor,
    ) -> None:
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]

        for step in range(1, self.num_speculative_steps):
            # 每一步重建（本仓库注意力后端没有 `update_draft_decode_metadata`，
            # 所以走上游的非融合回退路径，见 `_configure_fused_multi_step_decode`）。
            slot_mappings = self.block_tables.compute_slot_mappings(
                idx_mapping,
                query_start_loc,
                positions,
                num_reqs,
            )
            attn_metadata = self._build_draft_attn_metadata(
                num_reqs=num_reqs,
                num_tokens=num_reqs,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=step,
                slot_mappings=slot_mappings,
            )

            self.current_draft_step.fill_(step)
            self._generate_draft(
                num_reqs,
                num_reqs,
                attn_metadata,
                slot_mappings,
            )

    def _build_draft_attn_metadata(
        self,
        num_reqs: int,
        num_tokens: int,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        slot_mappings: torch.Tensor,
    ) -> dict[str, Any]:
        """草稿 decode 步的注意力元数据：每请求 1 行 query，seq_len = target 上界 + step。"""
        query_start_loc_cpu = torch.clamp(
            self.arange[: num_reqs + 1], max=num_reqs
        ) * 1
        block_tables = [
            x[:num_reqs] for x in self.block_tables.input_block_tables
        ]
        draft_seq_lens_cpu_upper_bound = torch.zeros(
            num_reqs, dtype=torch.int32, device="cpu"
        )
        torch.add(
            seq_lens_cpu_upper_bound[:num_reqs],
            step,
            out=draft_seq_lens_cpu_upper_bound[:num_reqs],
        )
        draft_seq_lens_cpu_upper_bound[:num_reqs].clamp_(max=self.max_model_len)
        metadata = self.metadata_builder.build(
            query_start_loc=self.input_buffers.query_start_loc[: num_reqs + 1],
            seq_lens=self.input_buffers.seq_lens[:num_reqs],
            block_table=block_tables[0],
            slot_mapping=slot_mappings[0][:num_tokens]
            if slot_mappings.dim() > 1
            else slot_mappings[:num_tokens],
            num_reqs=num_reqs,
        )
        # `query_start_loc_cpu` / `draft_seq_lens_cpu_upper_bound` 只用于形状校验与图键，
        # 本关 eager 不需要（图属 74 关），保留计算是为了与上游同形、便于 74 关接线。
        del query_start_loc_cpu, draft_seq_lens_cpu_upper_bound
        return {name: metadata for name in self._attention_layer_names()}

    def _attention_layer_names(self) -> list[str]:
        names = getattr(self, "draft_attn_layer_names", None)
        if names:
            return list(names)
        return [
            name
            for name, module in self.model.named_modules()
            if hasattr(module, "kv_cache")
        ]

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
    ) -> None:
        idx_mapping = self.idx_mapping[:num_reqs]
        positions = self.input_buffers.positions[:num_reqs]
        # Run the draft model forward pass.
        last_hidden_states, hidden_states = self._run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
        )
        last_hidden_states = last_hidden_states[:num_reqs]

        sample_positions = positions
        if not self.advance_draft_positions:
            # The forward pass holds positions fixed (Q-only, shared target KV),
            # but Gumbel sampling still needs the absolute draft position.
            sample_positions = positions + self.current_draft_step

        # Sample the draft tokens.
        draft_tokens = self.sample_draft(
            last_hidden_states,
            sample_positions,
            idx_mapping,
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
        )

        # Update the inputs for the next step.
        update_draft_inputs(
            draft_tokens,
            self.current_draft_step,
            hidden_states,
            self.draft_tokens,
            self.hidden_states,
            self.input_buffers,
            num_reqs,
            self.max_model_len,
            self.num_speculative_steps,
            advance_draft_positions=self.advance_draft_positions,
        )

    def _configure_fused_multi_step_decode(self) -> None:
        """本仓库注意力后端没有 `update_draft_decode_metadata` → 恒走非融合回退路径。"""
        self.use_fused_multi_step_decode = False


@triton.jit
def _prepare_prefill_inputs_kernel(
    last_token_indices_ptr,
    draft_current_step_ptr,
    draft_input_ids_ptr,
    draft_positions_ptr,
    draft_query_start_loc_ptr,
    draft_seq_lens_ptr,
    target_input_ids_ptr,
    target_positions_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + req_idx)

    # Get the true query length and next token after accounting for rejected tokens.
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    query_len -= num_rejected

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        next_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefilling.
        # Get the next prefill token.
        next_token = tl.load(next_prefill_tokens_ptr + req_state_idx)

    # Shift target_input_ids by one.
    for i in range(1, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        input_ids = tl.load(target_input_ids_ptr + query_start + block, mask=mask)
        tl.store(draft_input_ids_ptr + query_start + block - 1, input_ids, mask=mask)

    last_token_index = query_start + query_len - 1
    tl.store(last_token_indices_ptr + req_idx, last_token_index)
    tl.store(draft_input_ids_ptr + last_token_index, next_token)

    # Copy positions.
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        target_pos = tl.load(target_positions_ptr + query_start + block, mask=mask)
        tl.store(draft_positions_ptr + query_start + block, target_pos, mask=mask)

    # Copy query start locations.
    tl.store(draft_query_start_loc_ptr + req_idx, query_start)
    # Copy sequence lengths.
    tl.store(draft_seq_lens_ptr + req_idx, seq_len)
    if req_idx == (num_reqs - 1):
        # Reset the current draft step to 0.
        tl.store(draft_current_step_ptr, 0)
        # Pad query_start_loc for CUDA graphs.
        for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs + 1
            tl.store(draft_query_start_loc_ptr + block, query_end, mask=mask)
        # Pad seq_lens for CUDA graphs.
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(draft_seq_lens_ptr + block, 0, mask=mask)
        # Pad last_token_indices for CUDA graphs.
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(last_token_indices_ptr + block, 0, mask=mask)


def prepare_prefill_inputs(
    # [num_reqs]
    last_token_indices: torch.Tensor,
    current_draft_step: torch.Tensor,
    input_buffers: InputBuffers,
    input_batch: InputBatch,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [max_num_reqs]
    last_sampled: torch.Tensor,
    # [max_num_reqs]
    next_prefill_tokens: torch.Tensor,
    max_num_reqs,
) -> torch.Tensor:
    num_reqs = input_batch.num_reqs
    _prepare_prefill_inputs_kernel[(num_reqs,)](
        last_token_indices,
        current_draft_step,
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        input_batch.input_ids,
        input_batch.positions,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        input_batch.query_start_loc,
        input_batch.seq_lens,
        max_num_reqs,
        BLOCK_SIZE=1024,
    )
    return last_token_indices


@triton.jit
def _prepare_decode_inputs_kernel(
    draft_tokens_ptr,
    draft_tokens_stride,
    target_seq_lens_ptr,
    num_rejected_ptr,
    input_ids_ptr,
    positions_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    max_model_len,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
    ADVANCE_DRAFT_POSITIONS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_idx == num_reqs:
        # Compute query_start_loc. Pad it with the last query_start_loc
        # for CUDA graphs.
        for i in range(0, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            q = tl.where(block < num_reqs, block, num_reqs)
            mask = block < max_num_reqs + 1
            tl.store(query_start_loc_ptr + block, q, mask=mask)
        # Pad seq_lens for CUDA graphs.
        for i in range(req_idx, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    # draft token -> input id.
    draft_token = tl.load(draft_tokens_ptr + req_idx * draft_tokens_stride)
    tl.store(input_ids_ptr + req_idx, draft_token)

    target_seq_len = tl.load(target_seq_lens_ptr + req_idx)
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    seq_len = target_seq_len - num_rejected
    if ADVANCE_DRAFT_POSITIONS:
        # Compute position and seq_lens.
        # NOTE(woosuk): To prevent out-of-range access, we clamp these values
        # if they reach the max model length.
        position = tl.load(positions_ptr + req_idx)
        position = tl.minimum(position + 1, max_model_len - 1)
        tl.store(positions_ptr + req_idx, position)
        seq_len = tl.minimum(seq_len + 1, max_model_len)
    tl.store(seq_lens_ptr + req_idx, seq_len)


def prepare_decode_inputs(
    draft_tokens: torch.Tensor,
    target_seq_lens: torch.Tensor,
    num_rejected: torch.Tensor,
    input_buffers: InputBuffers,
    max_model_len: int,
    max_num_reqs: int,
    advance_draft_positions: bool = True,
):
    num_reqs = draft_tokens.shape[0]
    _prepare_decode_inputs_kernel[(num_reqs + 1,)](
        draft_tokens,
        draft_tokens.stride(0),
        target_seq_lens,
        num_rejected,
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        max_model_len,
        max_num_reqs,
        BLOCK_SIZE=1024,
        ADVANCE_DRAFT_POSITIONS=advance_draft_positions,
    )


@triton.jit
def _update_draft_inputs_kernel(
    output_draft_tokens_ptr,
    output_draft_tokens_stride,
    next_input_hidden_states_ptr,
    next_input_hidden_states_stride,
    input_ids_ptr,
    positions_ptr,
    seq_lens_ptr,
    draft_tokens_ptr,
    current_draft_step_ptr,
    hidden_states_ptr,
    hidden_states_stride,
    hidden_size,
    max_model_len,
    num_speculative_steps,
    BLOCK_SIZE: tl.constexpr,
    ADVANCE_DRAFT_POSITIONS: tl.constexpr,
):
    req_idx = tl.program_id(0)

    # Write the sampled draft token into self.draft_tokens[req_idx, step].
    draft_token = tl.load(draft_tokens_ptr + req_idx)
    step = tl.load(current_draft_step_ptr)
    tl.store(
        output_draft_tokens_ptr + req_idx * output_draft_tokens_stride + step,
        draft_token,
    )

    if step >= num_speculative_steps - 1:
        # This is the final step. Skip updating draft forward inputs.
        return

    # Write the sampled draft token into the input ids tensor for the next
    # forward pass.
    tl.store(input_ids_ptr + req_idx, draft_token)

    # Copy hidden states into the input hidden states tensor for the next
    # forward pass.
    for i in range(0, hidden_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < hidden_size
        hidden_states = tl.load(
            hidden_states_ptr + req_idx * hidden_states_stride + block,
            mask=mask,
        )
        tl.store(
            next_input_hidden_states_ptr
            + req_idx * next_input_hidden_states_stride
            + block,
            hidden_states,
            mask=mask,
        )

    if ADVANCE_DRAFT_POSITIONS:
        # Increment position and seq_lens.
        # NOTE(woosuk): To prevent out-of-range access, we clamp these values
        # if they reach the max model length.
        position = tl.load(positions_ptr + req_idx)
        position = tl.minimum(position + 1, max_model_len - 1)
        tl.store(positions_ptr + req_idx, position)

        seq_len = tl.load(seq_lens_ptr + req_idx)
        seq_len = tl.minimum(seq_len + 1, max_model_len)
        tl.store(seq_lens_ptr + req_idx, seq_len)


def update_draft_inputs(
    draft_tokens: torch.Tensor,
    current_draft_step: torch.Tensor,
    hidden_states: torch.Tensor,
    output_draft_tokens: torch.Tensor,
    next_input_hidden_states: torch.Tensor,
    input_buffers: InputBuffers,
    num_reqs: int,
    max_model_len: int,
    num_speculative_steps: int,
    advance_draft_positions: bool = True,
):
    _, hidden_size = hidden_states.shape
    _update_draft_inputs_kernel[(num_reqs,)](
        output_draft_tokens,
        output_draft_tokens.stride(0),
        next_input_hidden_states,
        next_input_hidden_states.stride(0),
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.seq_lens,
        draft_tokens,
        current_draft_step,
        hidden_states,
        hidden_states.stride(0),
        hidden_size,
        max_model_len,
        num_speculative_steps,
        BLOCK_SIZE=1024,
        ADVANCE_DRAFT_POSITIONS=advance_draft_positions,
    )
