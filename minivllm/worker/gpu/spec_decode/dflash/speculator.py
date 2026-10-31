"""DFlash 的 V2 提议者（对应 vLLM `v1/worker/gpu/spec_decode/dflash/`）。

V2 的 DFlash 与 V1 的**算法**完全一样，差别在"输入怎么进 GPU"：V1 是 CPU 拼行再上传，
V2 是全张量、常驻缓冲（`_prepare_dflash_inputs_kernel` 直接按 target 的 `InputBatch` 写
draft 的输入缓冲）。于是本文件只有三件事：

    _prepare_dflash_inputs_kernel   行/位置/槽位/采样映射的 GPU 组装（上游同名内核）
    DFlashSpeculator.propose()      组合：特征投影 → 组装输入 → **图外**上下文 KV 预计算 →
                                    一次 query forward（可重放 FULL 图）→ 采 K 枚草稿
    DFlashSpeculator.capture()      捕获那个 query 前向（每请求固定 1+K 行 ⇒ 单一固定形状）

**为什么上下文 KV 预计算在图外**（需求 076 §3.3，上游同款）：它的行数每步都变（本轮
target 处理了多少个 token），录不进"固定形状"的图。所以它每轮 eager 跑一次，只有 query 前向
进图。

**裁剪**（逐条记在 `docs/step76_alignment.md` §5）：多 KV group（混合滑窗/全注意力）、
DP/EPLB/多模态、`is_neox_style` 从 target 拷（本仓库的 RoPE 只有一种风格）。这些都是
**明确拒绝或明确不用**，不是静默降级。内核的 `SAMPLE_FROM_ANCHOR` 两种布局在 77 关补齐
（DFlash 恒 False；DSpark 由 `DSparkSpeculator` 传自己的值，见 `dspark/speculator.py`）。
"""

from typing import Any

import torch
import torch.nn as nn

from .....config import CUDAGraphMode, VllmConfig
from .....forward_context import BatchDescriptor, set_forward_context
from .....triton_utils import tl, triton
from ...block_table import PAD_SLOT_ID
from ...cudagraph_utils import BatchExecutionDescriptor
from ...input_batch import InputBatch, InputBuffers
from ..speculator import DraftModelSpeculator
from ..autoregressive.cudagraph_utils import SpeculatorCudaGraphManager


def _mask_token_id(hf_config: dict) -> int:
    """并行槽位填的占位符（与 `SpeculativeConfig._dflash_mask_token_id` 同一条取值链）。"""
    dflash_config = hf_config.get("dflash_config") or {}
    for value in (dflash_config.get("mask_token_id"), hf_config.get("mask_token_id"),
                  hf_config.get("dspark_noise_token_id"), hf_config.get("pard_token"),
                  hf_config.get("ptd_token_id")):
        if value is not None:
            return int(value)
    raise ValueError("DFlash 的配置里没有 mask token（dflash_config.mask_token_id 等一处都没有）")


class DFlashSpeculator(DraftModelSpeculator):
    """DFlash 的 V2 提议者（上游 `DFlashSpeculator(DraftModelSpeculator)`）。"""

    _speculator_name = "DFlash"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        draft_hf = self.speculative_config.draft_model_config.hf_config or {}
        dflash_config = draft_hf.get("dflash_config") or {}
        # DFlash 的布局固定是 `1 + K`（anchor = bonus、只有 mask 行采样）。`sample_from_anchor`
        # 是 **DSpark** 的开关（77 关），而且上游 DSpark 读的是**顶层**字段
        # `hf_config.sample_from_anchor`——与这里读的 `dflash_config.sample_from_anchor` 不是同一个
        # 键。所以 `method="dspark"` 放行（它在自己的 `__init__` 里按顶层字段重设布局）。
        if self.method != "dspark" and dflash_config.get("sample_from_anchor", False):
            raise NotImplementedError(
                "DFlash 不支持 `dflash_config.sample_from_anchor=True`（那是 DSpark 的形状，77 关）："
                "DFlash 的 anchor 就是 target 刚采出的 bonus token、不参与采样")
        # 本次草稿的 query 布局：DFlash 恒 `1 + K`（anchor 不采样）；DSpark 在自己 `__init__` 里改写。
        self.sample_from_anchor = False

        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device)
        # 多模态输入不支持（上游同款；本仓库的 V2 本来也没有多模态通路）
        self.supports_mm_inputs = False

        self.num_query_per_req = 1 + self.num_speculative_steps
        self.parallel_drafting_token_id = _mask_token_id(draft_hf)
        self.use_aux_hidden_state = bool(dflash_config.get("use_aux_hidden_state", True))

        # 上下文 KV 的位置（由内核填）——**不**进 query 图（形状每步都变）
        self.context_positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device)
        self._context_slot_mapping = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device)

        # 采样行的三张映射表（每请求 K 行；上游同名缓冲）
        max_num_sampled = self.max_num_reqs * self.num_speculative_steps
        self.sample_indices = torch.zeros(max_num_sampled, dtype=torch.int64, device=device)
        self.sample_pos = torch.zeros(max_num_sampled, dtype=torch.int64, device=device)
        # **-1 = 惰性行**：图捕获/重放时，补齐出来的采样行绝不能散射到 0 号请求的槽位上
        # （上游注释：zero would make every padding row scatter into request slot 0）。
        self.sample_idx_mapping = torch.full(
            (max_num_sampled,), -1, dtype=torch.int32, device=device)
        # [0, 1, …, K−1, 0, 1, …]：每枚草稿在 `draft_logits[req, step, :]` 里的列号
        self.sample_col = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        ).repeat(self.max_num_reqs)

        self.query_cudagraph_manager: SpeculatorCudaGraphManager | None = None
        # 元数据构造器在 `set_attn()` 里建（与 target 同一个后端、同一套 builder）
        self.metadata_builder = None
        # 每层因果性在 `load_model()` 里读模型（配置期已保证全层一致）
        self.dflash_causal = True

    # -------- 加载与注意力上下文 --------

    def load_draft_model(self, target_model: nn.Module) -> nn.Module:
        from .....model_loader import get_model

        return get_model(self.speculative_config.draft_model_config, self.device)

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        causal = self.model.get_draft_attn_causal()
        if len(set(causal)) != 1:
            raise NotImplementedError(
                f"DFlash 的逐层因果性不一致（{causal}）：本仓库只支持全部因果或全部非因果")
        self.dflash_causal = bool(causal[0])

    def set_attn(self, block_tables, target_input_buffers: InputBuffers,
                 target_build_attn_metadata) -> None:
        """接上草稿的注意力设施（对应上游 `DFlashSpeculator.set_attn`，只留真正消费的部分）。

        三个参数里 DFlash 只用 `block_tables`：

            block_tables               内核查 query/context 槽位、把 query 槽位写回
                                       `slot_mappings[0]`（图重放读的就是这块常驻缓冲）、
                                       捕获图时造假块表/假槽位
            target_input_buffers       AR 提议者（EAGLE/MTP）捕获草稿 prefill 图时复用 target 的
            target_build_attn_metadata 输入缓冲与元数据；DFlash 的 query 前向用**自己的**
                                       `input_buffers` 和自造的元数据，不消费这两个

        上游基类把三样都存下来（DFlash 分支同样没读后两个）——这里不调 `super()` 是为了不留
        "存了没人用"的引用（差异账本 D-76-9）。`metadata_builder` 需要 block_size，所以在
        接线点建一次，`propose()` 每轮直接用它造元数据。
        """
        self.block_tables = block_tables
        # 087 收口：跟随 target 的 attention backend；DFlash 的“非因果”是配置差异，
        # 不是换回 Torch 的理由。builder 负责给出对应的 workspace_pool/图能力。
        backend = self.draft_attention_backend
        self.metadata_builder = backend.get_builder_cls()(
            self.vllm_config.cache_config.block_size)

    # -------- 图 --------

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        """只建**一张**图：query 前向（上游 `DFlashCudaGraphManager` 的等价物）。

        三条规则与上游一致：

            PIECEWISE 不支持（上游注释原文）→ 只有解码全图或 eager 两种
            满足条件时收敛成 `FULL_DECODE_ONLY`（DFlash 每轮形状都是 num_reqs × (1+K)）
            不满足就 eager（本仓库的 Torch 后端恒满足 `UNIFORM_BATCH`，所以走前者）
        """
        draft_mode = (
            CUDAGraphMode.FULL_DECODE_ONLY
            if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            else CUDAGraphMode.NONE
        )
        self.query_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            draft_mode,
            decode_query_len=self.num_query_per_req,
        )
        # 供 runner 的 `capture_model()` 统一遍历（本仓库的账本接口）
        self.cudagraph_managers = {"query": self.query_cudagraph_manager}

    def capture(self) -> None:
        assert self.query_cudagraph_manager is not None, "先调 init_cudagraph_manager()"
        # 捕获期的补齐采样行必须是惰性的（否则会写进 0 号请求的草稿槽位）
        self.sample_indices.zero_()
        self.sample_pos.zero_()
        self.sample_idx_mapping.fill_(-1)
        self.query_cudagraph_manager.capture(
            self._generate_draft,
            self._capture_attn_metadata,
            self.input_buffers,
            self.block_tables,
            progress_bar_desc="Capturing DFlash query CUDA graphs",
        )

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
        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        num_query_tokens = num_reqs * self.num_query_per_req
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(
            max_seq_len + self.num_query_per_req, self.max_model_len)

        # 特征与 target 同源：多辅助层拼接后过一次 `fc`（上游 `combine_hidden_states`），
        # 行数 = target 本轮的行数（含被拒的草稿行——它们的 hidden 也在批里，但
        # `num_rejected` 会让内核只把有效前缀写进上下文）。
        if aux_hidden_states:
            hidden_states = self.model.combine_hidden_states(
                torch.cat(aux_hidden_states, dim=-1))
        else:
            hidden_states = last_hidden_states
        self.hidden_states[:num_target_tokens].copy_(hidden_states[:num_target_tokens])

        self._copy_request_inputs(num_reqs, input_batch.idx_mapping, temperature, seeds)

        prepare_dflash_inputs(
            self.input_buffers,
            self.block_tables.slot_mappings[0],
            self.context_positions,
            self._context_slot_mapping,
            self.sample_indices,
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            self.block_tables.input_block_tables[0],
            int(self.block_tables.kernel_block_sizes[0]),
            self.parallel_drafting_token_id,
            self.num_query_per_req,
            self.num_speculative_steps,
            self.max_num_reqs,
            self.max_num_tokens,
            self.max_model_len,
            self.sample_from_anchor,
        )

        # 上下文 K/V 预计算：**图外** eager（形状每步都变）；写的是草稿自己的缓存。
        if dummy_run:
            context_slots = None
        else:
            context_slots = self._context_slot_mapping[:num_target_tokens]
        self.model.precompute_and_store_context_kv(
            self.hidden_states[:num_target_tokens],
            self.context_positions[:num_target_tokens],
            context_slots,
        )

        desc = self.query_cudagraph_manager.dispatch(
            num_reqs, num_query_tokens,
            uniform_token_count=self.num_query_per_req,
            need_eager=is_profile,
        ) if self.query_cudagraph_manager is not None else BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE, num_tokens=num_query_tokens, num_reqs=num_reqs)
        if self.query_cudagraph_manager is not None:
            self.query_cudagraph_manager.record_selection(desc, num_reqs, num_query_tokens)

        num_reqs_padded = desc.num_reqs or num_reqs
        num_tokens_padded = desc.num_tokens
        if desc.cg_mode == CUDAGraphMode.FULL:
            # 图里记的是指针：上面两个内核已经把这一轮的数据写进同一批常驻缓冲
            self.query_cudagraph_manager.run_fullgraph(desc)
        else:
            self._generate_draft(
                num_reqs,
                num_tokens_padded,
                self._build_attn_metadata(num_reqs_padded, num_tokens_padded),
                {"slot_mapping": self.block_tables.slot_mappings[:, :num_tokens_padded]},
                desc.cg_mode,
            )
        return self.draft_tokens[:num_reqs]

    # -------- 前向与采样 --------

    def _build_attn_metadata(self, num_reqs_padded: int,
                             num_tokens_padded: int) -> dict[str, Any]:
        """DFlash query 前向的注意力元数据（**非因果**，由每层配置决定）。

        与 target 的元数据是两份：query 行是"槽位"，可以互相看（`causal=False`），
        `seq_lens` 也已经包含这些 query 位置。**不能**复用 target 那份——它服务的是
        target 自己的因果前向。
        """
        metadata = self.metadata_builder.build(
            query_start_loc=self.input_buffers.query_start_loc[:num_reqs_padded + 1],
            seq_lens=self.input_buffers.seq_lens[:num_reqs_padded],
            block_table=self.block_tables.input_block_tables[0][:num_reqs_padded],
            slot_mapping=self.block_tables.slot_mappings[0][:num_tokens_padded],
            num_reqs=num_reqs_padded,
            uniform_query_len=self.num_query_per_req,
            causal=self.dflash_causal,
        )
        return {name: metadata for name in self.draft_attn_layer_names}

    def _capture_attn_metadata(self, input_batch: InputBatch) -> dict[str, Any]:
        """捕获期回调（`prepare_inputs_to_capture` 之后的假批）。"""
        return self._build_attn_metadata(input_batch.num_reqs, input_batch.num_tokens)

    def _run_model(self, num_tokens: int, attn_metadata: dict[str, Any] | None,
                   slot_mappings: dict[str, torch.Tensor] | None,
                   cudagraph_runtime_mode: CUDAGraphMode) -> torch.Tensor:
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            return self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
            )

    def _generate_draft(self, num_reqs: int, num_tokens: int,
                        attn_metadata: dict[str, Any] | None,
                        slot_mappings: dict[str, torch.Tensor] | None,
                        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE) -> None:
        """一次 query 前向 + 只采 mask 行（上游同名方法，逐行同义）。

        `sample_pos - 2` 不是笔误：`sample_pos` 是"被预测 token 的位置 Q"，而验证侧的
        Gumbel 噪声按**前一个位置**（Q−1）取；`sample_draft()` 内部又 +1，所以传 Q−2。
        """
        last_hidden_states = self._run_model(num_tokens, attn_metadata, slot_mappings,
                                             cudagraph_runtime_mode)
        num_sample = num_reqs * self.num_speculative_steps
        sample_hidden_states = last_hidden_states[self.sample_indices[:num_sample]]
        draft_tokens = self.sample_draft(
            sample_hidden_states,
            self.sample_pos[:num_sample] - 2,
            self.sample_idx_mapping[:num_sample],
            self.temperature,
            self.seeds,
            self.sample_col[:num_sample],
            self.draft_logits,
        )
        self.draft_tokens[:num_reqs] = draft_tokens.view(
            num_reqs, self.num_speculative_steps)



@triton.jit
def _prepare_dflash_inputs_kernel(
    out_input_ids_ptr,
    out_query_positions_ptr,
    out_query_start_loc_ptr,
    out_seq_lens_ptr,
    out_query_slot_mapping_ptr,
    out_context_positions_ptr,
    out_context_slot_mapping_ptr,
    out_sample_indices_ptr,
    out_sample_pos_ptr,
    out_sample_idx_mapping_ptr,
    out_temperature_ptr,
    out_seeds_ptr,
    # 本轮 target 的事实
    target_positions_ptr,
    target_query_start_loc_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    # 采样参数（按 slot）
    temperature_ptr,
    seeds_ptr,
    block_table_ptr,
    block_table_stride,
    # 标量
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
    max_model_len,
    SAMPLE_FROM_ANCHOR: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """上游 `_prepare_dflash_inputs_kernel` 的逐行等价（含 `SAMPLE_FROM_ANCHOR` 两种布局）。

    每请求产出（与上游同序）：

        context 行    位置 = target 的位置；槽位按块表查；被拒尾部写 `PAD_SLOT_ID`
                      （它们不是有效上下文，**不能**写草稿的 KV）
        query 行      位置 = `last_valid_pos + 1 + off`；第 0 行是 bonus token，其余是 mask token
                      （`num_query_per_req` 行：DFlash 是 `1 + K`，DSpark 的 anchor-as-first 是 `K`）
        采样行        `off >= sample_off` 的那些：DFlash 只采 mask 行（`off >= 1`，位置 = 自己），
                      DSpark 的 anchor-as-first 连 anchor 一起采（`off >= 0`，位置 = 自己 + 1，
                      因为 anchor 这个位置是真实 token、它预测的是**下一个**位置）
        seq_lens      `last_valid_pos + 1 + num_query_per_req`（绝对长度）
        补齐          `query_start_loc`/`seq_lens`/采样映射按 `max_num_reqs` 铺满，
                      **补齐采样行的映射写 -1**（图重放时不许写真实槽位）

    与上游的一处裁剪：没有块 0 判断（上游那是对滑窗淘汰的防御；本仓库不支持混合滑窗，
    块表里不会出现 0 号块）。
    """
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
    ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start

    num_rejected = tl.load(num_rejected_ptr + req_idx)
    valid_ctx_end = ctx_end - num_rejected
    num_valid_ctx = valid_ctx_end - ctx_start

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        bonus_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # 分块 prefill：本轮不产出 token，用下一段 prefill 的首 token 当 bonus
        bonus_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)

    last_valid_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_base = req_idx * num_query_per_req

    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    is_ctx = j < num_ctx
    is_valid_ctx = j < num_valid_ctx
    is_query = (j >= num_valid_ctx) & (j < num_valid_ctx + num_query_per_req)
    query_off = j - num_valid_ctx

    # ---- context 行：位置原样；有效行查块表，被拒尾部写哨兵 ----
    ctx_pos_idx = ctx_start + tl.where(is_ctx, j, 0)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_valid_ctx, other=0)
    ctx_block_num = ctx_pos // block_size
    ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=is_valid_ctx, other=0).to(tl.int64)
    ctx_slot = tl.where(is_valid_ctx, ctx_block_id * block_size + (ctx_pos % block_size),
                        PAD_SLOT_ID)
    # 整个 [0, num_ctx) 跨度都要写：被拒行写 0/哨兵，保证图重放读不到上一批的残值
    tl.store(out_context_positions_ptr + ctx_start + j, ctx_pos, mask=is_ctx)
    tl.store(out_context_slot_mapping_ptr + ctx_start + j, ctx_slot, mask=is_ctx)

    # ---- query 行：[bonus, mask, ...]，位置接在 last_valid_pos 之后 ----
    query_pos = last_valid_pos + 1 + query_off
    query_idx = query_base + query_off
    is_bonus = is_query & (query_off == 0)
    input_id = tl.where(is_bonus, bonus_token, parallel_drafting_token_id)

    q_block_num = query_pos // block_size
    q_block_num = tl.minimum(q_block_num, block_table_stride - 1)
    q_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + q_block_num,
        mask=is_query, other=0).to(tl.int64)
    q_slot = q_block_id * block_size + (query_pos % block_size)
    q_slot = tl.where(is_query, q_slot, PAD_SLOT_ID)

    tl.store(out_input_ids_ptr + query_idx, input_id, mask=is_query)
    clamped_query_pos = tl.minimum(query_pos, max_model_len - 1)
    tl.store(out_query_positions_ptr + query_idx, clamped_query_pos, mask=is_query)
    tl.store(out_query_slot_mapping_ptr + query_idx, q_slot, mask=is_query)

    # ---- 采样行 ----
    # DFlash（`SAMPLE_FROM_ANCHOR=False`）：只采 mask 行（off ≥ 1），位置 = 它自己的 query 位置
    # ——mask 就坐在"被预测的那个位置"上。DSpark（`True`）：每个 query 位置都预测**下一个**
    # token（anchor 行是真实的 bonus token），所以 off ≥ 0 且采样位置 = query 位置 + 1。
    sample_off = 0 if SAMPLE_FROM_ANCHOR else 1
    is_sample = is_query & (query_off >= sample_off)
    sample_idx = req_idx * num_speculative_steps + (query_off - sample_off)
    sample_pos = query_pos + 1 if SAMPLE_FROM_ANCHOR else query_pos
    tl.store(out_sample_indices_ptr + sample_idx, query_idx, mask=is_sample)
    tl.store(out_sample_pos_ptr + sample_idx, sample_pos, mask=is_sample)
    tl.store(out_sample_idx_mapping_ptr + sample_idx, req_state_idx, mask=is_sample)

    if block_idx == 0:
        tl.store(out_query_start_loc_ptr + req_idx, query_base)
        tl.store(
            out_seq_lens_ptr + req_idx,
            tl.minimum(last_valid_pos + 1 + num_query_per_req, max_model_len))
        # 采样参数按 slot 抄一份（草稿采样只看温度与 seed，与 target 同款）
        tl.store(out_temperature_ptr + req_state_idx,
                 tl.load(temperature_ptr + req_state_idx))
        tl.store(out_seeds_ptr + req_state_idx, tl.load(seeds_ptr + req_state_idx))
        if req_idx == num_reqs - 1:
            # 补齐 per-request 缓冲到 max_num_reqs（图重放可能比真实批大）
            last_query_end = num_reqs * num_query_per_req
            for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs + 1
                tl.store(out_query_start_loc_ptr + block, last_query_end, mask=mask)
            for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs
                tl.store(out_seq_lens_ptr + block, 0, mask=mask)
            # 补齐的采样行：索引指 0（合法行，避免越界读），映射 -1（不写任何请求）
            pad_start = num_reqs * num_speculative_steps
            pad_end = max_num_reqs * num_speculative_steps
            for i in range(pad_start, pad_end, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < pad_end
                tl.store(out_sample_indices_ptr + block, 0, mask=mask)
                tl.store(out_sample_pos_ptr + block, 0, mask=mask)
                tl.store(out_sample_idx_mapping_ptr + block, -1, mask=mask)
            # 补齐的 query 槽位写哨兵（不写 KV）
            q_pad_start = num_reqs * num_query_per_req
            for i in range(q_pad_start, max_num_tokens, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_tokens
                tl.store(out_query_slot_mapping_ptr + block, PAD_SLOT_ID, mask=mask)


def prepare_dflash_inputs(
    input_buffers: InputBuffers,
    query_slot_mapping: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    input_batch: InputBatch,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    input_temperature: torch.Tensor,
    input_seeds: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    parallel_drafting_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    sample_from_anchor: bool = False,
) -> None:
    """启动 `_prepare_dflash_inputs_kernel`（上游同名包装，参数逐个对应）。

    `sample_from_anchor` 只有 DSpark 会传 True（anchor-as-first：K 个 query 全采样）；
    DFlash 用默认的 False（`1 + K` 布局，anchor 是 bonus）。
    """
    num_reqs = input_batch.num_reqs
    assert num_reqs > 0
    max_target_query_len = int(input_batch.num_scheduled_tokens.max())
    max_tokens_per_req = max_target_query_len + num_query_per_req
    BLOCK_SIZE = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
    num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
    _prepare_dflash_inputs_kernel[(num_reqs, num_blocks)](
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        temperature,
        seeds,
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        input_temperature,
        input_seeds,
        block_table,
        block_table.stride(0),
        parallel_drafting_token_id,
        int(block_size),
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
        SAMPLE_FROM_ANCHOR=sample_from_anchor,
        PAD_SLOT_ID=PAD_SLOT_ID,
        BLOCK_SIZE=BLOCK_SIZE,
    )
