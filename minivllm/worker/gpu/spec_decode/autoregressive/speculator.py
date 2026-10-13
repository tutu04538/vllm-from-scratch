"""V2 的自回归提议者（对应 vLLM `v1/worker/gpu/spec_decode/autoregressive/speculator.py`）。

EAGLE/EAGLE3/MTP/Gemma4-MTP 在 V2 里共用这套流程（上游类层次：
`DraftModelSpeculator` → `AutoRegressiveSpeculator` → `EagleSpeculator` / `MTPSpeculator` /
`Gemma4Speculator`），只有"模型从哪来"与"位置要不要步进"不同。

**"整块左移"是这一支（吃 target hidden 的 AR/EAGLE 系）的布局，不是 V2 的普遍规则**，
所以这个内核里没有 `if method` 分支——分叉在"选哪个 speculator 类"（上游
`spec_decode/__init__.py::init_speculator`）里：`DFlashSpeculator`（→ DFlash2/DSpark）有**自己的**
输入组装内核（`_prepare_dflash_inputs_kernel`：上下文原样 + 追加 `num_query_per_req` 行，
query 0 = bonus token、其余是 mask token，**不左移**），`MultiModuleMTPSpeculator` 也有自己的
内核（同一套 `num_sampled ? last_sampled : next_prefill_tokens` 规则，但读 K 枚 lookahead 给后续
模块做 override）。V1 里"要不要左移"反而是**运行期参数**：上游同一个
`copy_and_expand_eagle_inputs_kernel` 靠 `shift_input_ids` 分叉（True = EAGLE 系跳过本请求第 0 枚
token；False = PARD/普通 draft 整块原样 + 追加槽位），本仓库 V1 的等价实现见
`spec_decode/utils.py::expand_parallel_draft_inputs(..., shift_input_ids=...)`。

**关键差别在输入组装**：V1 用 `copy_and_expand_eagle_inputs_kernel`
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

### 74 关：两张图 + 融合多步 decode

这一步要解决的是"K 个草稿步各提交一次 Python/驱动栈"的开销。有两条路，**选哪条由后端的
capability 决定**（不猜、也不伪造）：

    fallback（`_multi_step_decode`）   每个草稿步：CPU 侧重建 slot_mapping + 注意力元数据，
                                       然后跑一次前向。后端不支持"元数据原地更新"时必须走这条。
    fused（`_fused_multi_step_decode`）元数据**只建一次**（step=1），K−1 个草稿步的循环整体
                                       录进**一张图**；步与步之间只做两件图内的事：
                                       `compute_slot_mappings()` 原算 + `update_draft_decode_metadata()`
                                       刷新步相关元数据。

于是"重放一次 = 走完 K−1 步"，而不是"每步一张图"。两张图（prefill 与 decode）由
`SpeculatorCudaGraphManager` 分别管理，与 target 的图**完全独立**：

    prefill 图  草稿位置 0（每请求 1+K 行），复用 **target 本轮**的注意力元数据与 slot_mapping
    decode 图   草稿位置 >0（每请求 1 行），用草稿自己的元数据（`seq_lens + step`）

**裁剪**（逐条记在 `docs/step74_alignment.md`）：多模态输入、DP 同步、EPLB、local-argmax
规约、多模块 MTP；PIECEWISE 分段图在 V2 未接（草稿的两张图都只做 FULL）。
"""

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .....config import CUDAGraphMode, VllmConfig
from .....forward_context import BatchDescriptor, set_forward_context
from .....triton_utils import tl, triton
from ....utils import AttentionGroup, get_uniform_decode_token_count
from ...block_table import PAD_SLOT_ID
from ...cudagraph_utils import BatchExecutionDescriptor
from ...input_batch import InputBatch, InputBuffers
from ..speculator import DraftModelSpeculator
from .cudagraph_utils import SpeculatorCudaGraphManager

logger = logging.getLogger(__name__)


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

        # 草稿层自己的注意力组（`set_attn()` 里建；组数恒为 1，见 worker/utils.py）。
        self.attn_groups: list[list[AttentionGroup]] = []
        self.block_tables = None
        self.target_input_buffers: InputBuffers | None = None
        self.target_build_attn_metadata = None

        self.prefill_cudagraph_manager: SpeculatorCudaGraphManager | None = None
        self.decode_cudagraph_manager: SpeculatorCudaGraphManager | None = None
        self.use_fused_multi_step_decode = False

    # -------- 生命周期钩子（上游同名；子类覆写用于"注意力标志"这类切换） --------

    def on_prefill_begin(self, num_reqs: int) -> None: ...

    def on_prefill_end(self, num_reqs: int) -> None: ...

    def on_multi_step_decode_begin(self, num_reqs: int) -> None: ...

    def on_multi_step_decode_end(self, num_reqs: int) -> None: ...

    @property
    def advance_draft_positions(self) -> bool:
        """每一步是否推进 positions/seq_lens（EAGLE/标准 MTP：True；Q-only 的 MTP：False）。"""
        return True

    # -------- 注意力上下文 --------

    def set_attn(
        self,
        block_tables,
        target_input_buffers: InputBuffers,
        target_build_attn_metadata,
    ) -> None:
        """把 draft 的注意力上下文接到 target 的块表与输入缓冲上（对应上游同名方法）。

        上游签名是 `set_attn(model_state, kv_cache_config, block_tables,
        target_input_buffers, target_attn_groups)`，在其中用 `init_attn_backend(...,
        active_layer_names=self.draft_attn_layer_names)` 建**草稿自己的**注意力组。本仓库只有
        一个后端、没有 ModelState/KVCacheSpec 对象，所以：

            `model_state`            → `target_build_attn_metadata`（runner 侧"造 target 那份
                                       元数据"的回调；草稿 prefill 图复用它）
            `kv_cache_config`        → 不需要（块大小从 `vllm_config.cache_config` 取）
            `target_attn_groups`     → 由上面那个回调覆盖（上游用它的理由见
                                       `docs/step74_alignment.md` 的差异账本 D-74-3）

        建完组之后立刻定"融合还是逐步"（上游同位置：`_configure_fused_multi_step_decode()`）。
        """
        super().set_attn(block_tables, target_input_buffers, target_build_attn_metadata)
        self.attn_groups = self._init_attn_groups()
        self._configure_fused_multi_step_decode()

    def _init_attn_groups(self) -> list[list[AttentionGroup]]:
        """建草稿自己的注意力组（对应上游 `init_attn_backend(..., active_layer_names=

        self.draft_attn_layer_names)`）。

        本仓库只有一个后端、一个 KV group，所以恒为"一组、一个构造器"。**测试的接缝放在这里**：
        74 关的验收要"支持/不支持 `update_draft_decode_metadata` 的两个后端都跑一遍"，而本仓库
        生产路径只有 Torch 后端（它声明支持），所以测试用 `minivllm/testing/` 里的替身构造器
        替换这个方法——被验证的仍然是**生产的那条门**（`_configure_fused_multi_step_decode()`），
        只是换了一份 capability 声明。
        """
        from .....attention.backends.torch_sdpa import TorchAttentionBackend

        builder = TorchAttentionBackend.get_builder_cls()(
            self.vllm_config.cache_config.block_size
        )
        return [[
            AttentionGroup(
                backend=TorchAttentionBackend,
                layer_names=list(self.draft_attn_layer_names),
                metadata_builders=[builder],
            )
        ]]

    def _configure_fused_multi_step_decode(self) -> None:
        """决定"K−1 个草稿步是逐步重建元数据，还是融进一张图"（上游同名方法，逐条对齐）。

        三条规则，顺序不能改：

            1. `K == 1` → 没有第二步，不融合（融合的收益恰好是"省掉 K−1 次重建"）。
            2. `advance_draft_positions == False`（Q-only 的 MTP：位置不动、共享 target 的 KV）
               → **强制融合**。这条不是优化而是正确性：逐步路径会在每个草稿步重建
               `slot_mapping`，而"位置不动"的模型每步的槽位完全相同，重建只会把元数据里的
               `seq_lens` 推着走，反而算错。
            3. 否则看**后端 capability**：只要有任何一个注意力组不支持原地更新，就回退逐步重建。
               **不伪造 capability**：本仓库的 Torch 后端声明支持（步相关字段都是常驻缓冲的
               切片，见 `attention/backends/torch_sdpa.py`），而"不支持"的后端（例如把
               `scheduler_metadata` 拷了一份出来的 FA3）走回退路径——那条路每步重建，慢但正确。
        """
        if self.num_speculative_steps == 1:
            self.use_fused_multi_step_decode = False
            return

        if not self.advance_draft_positions:
            self.use_fused_multi_step_decode = True
            return

        unsupported_backends = sorted({
            attn_group.backend.get_name()
            for attn_groups in self.attn_groups
            for attn_group in attn_groups
            if not attn_group.supports_draft_decode_metadata_update
        })
        self.use_fused_multi_step_decode = not unsupported_backends
        if unsupported_backends:
            logger.info(
                "融合多步草稿 decode 不被注意力后端 %s 支持；回退为「每步重建注意力元数据」"
                "（正确性不变，只少省下 CPU 侧重建与 kernel 启动开销）",
                ", ".join(unsupported_backends),
            )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        """建两张图管理器（上游同名方法；两处差别都写在下面）。

        草稿图只做 FULL，所以先把 target 的最终模式收敛成"decode 用不用全图"这一个问题：

            prefill 图（草稿位置 0） 上游把 target 的**原始**模式传进来，于是混合批也会有一张
                                    PIECEWISE 键（混合批的草稿 prefill 走 target 同款的分段图）。
                                    本仓库 V2 没有分段图（`CompilationMode` 只有 NONE、V2 不给
                                    模型装分段件），所以这里只保留 FULL 的那一半：混合批一律走
                                    eager，**不会**出现"选中 PIECEWISE 却什么都没捕获"的假命中。
            decode 图（草稿位置 >0）上游规则照抄：PIECEWISE 图不支持草稿 decode，所以
                                    `FULL → FULL_DECODE_ONLY`，其余一律 `NONE`。
        """
        draft_mode = (
            CUDAGraphMode.FULL_DECODE_ONLY
            if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            else CUDAGraphMode.NONE
        )
        self.prefill_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            draft_mode,
            self.num_speculative_steps + 1,
            # 草稿 prefill 图复用 **target 本轮**的注意力元数据，而本仓库 V2 的 target 批不补齐
            # （target 图不属 74 关），所以这张图必须精确命中形状——理由见
            # `cudagraph_utils._is_compatible` 的长注释。
            allow_shape_padding=False,
        )
        self.decode_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            draft_mode,
            decode_query_len=1,
        )

    # -------- 捕获 --------

    def capture(self) -> None:
        """捕获 prefill / decode 两张图（上游同名方法）。

        两处"先做的小事"都有理由：

            把 `last_token_indices` / `idx_mapping` **清零**：捕获期的假输入会读它们，留着
                上一轮的旧值就是越界索引（上游注释：prevent stale values from prior dummy
                runs to cause out-of-bounds indexing during capture）。
            `on_*_begin/end` 钩子**在捕获与运行两侧成对触发**（上游注释：so that any state
                they toggle is identical at capture time and replay time）。MTP 的
                `skip_topk` 就是靠这条：图里录的是"复用 top-k 索引"那段代码，重放前必须
                同样把它打开。§3.6 的另一半（异常后不残留）由 `on_prefill_begin()` 无条件
                清除保证。
        """
        assert self.prefill_cudagraph_manager is not None, "先调 init_cudagraph_manager()"
        self.last_token_indices.zero_()
        self.idx_mapping.zero_()

        self.on_prefill_begin(self.max_num_reqs)
        self.prefill_cudagraph_manager.capture(
            self._prefill,
            self.target_build_attn_metadata,
            self.target_input_buffers,
            self.block_tables,
            progress_bar_desc="Capturing draft prefill CUDA graphs",
        )
        self.on_prefill_end(self.max_num_reqs)

        if self.num_speculative_steps == 1:
            return

        assert self.decode_cudagraph_manager is not None
        self.on_multi_step_decode_begin(self.max_num_reqs)
        decode_fn = (
            self._generate_fused_drafts
            if self.use_fused_multi_step_decode
            else self._generate_draft
        )
        self.decode_cudagraph_manager.capture(
            decode_fn,
            self._draft_attn_metadata_for_capture,
            self.input_buffers,
            self.block_tables,
            progress_bar_desc="Capturing draft decode CUDA graphs",
        )
        self.on_multi_step_decode_end(self.max_num_reqs)

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

        # 统一 decode 的判定用的是**未补齐**的行数（上游注释：without padding added by the
        # target model during FULL cudagraph）；而分派用的是**补齐后**的行数。
        uniform_token_count = get_uniform_decode_token_count(
            num_reqs,
            num_tokens,
            int(input_batch.num_scheduled_tokens.max()),
            input_batch.has_prefill,
        )
        prefill_batch_desc = self._dispatch_prefill(
            num_reqs, num_tokens_padded, uniform_token_count, need_eager=is_profile
        )

        self.on_prefill_begin(num_reqs)
        if prefill_batch_desc.cg_mode == CUDAGraphMode.FULL:
            # 重放草稿 prefill 全图（输入已经在**常驻缓冲**里：上面的
            # `prepare_prefill_inputs` 与 `hidden_states` 拷贝写的就是那些缓冲）。
            assert self.prefill_cudagraph_manager is not None
            self.prefill_cudagraph_manager.run_fullgraph(prefill_batch_desc)
        else:
            # 本关 eager：target 的 attention metadata 与 slot_mapping 可以直接给 draft prefill
            # 用，因为批形状与 KV 布局完全相同（上游注释同款）。
            self._prefill(
                num_reqs,
                prefill_batch_desc.num_tokens,
                attn_metadata,
                slot_mappings,
                prefill_batch_desc.cg_mode,
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

        # 每个草稿步每请求恰好 1 行 → 统一 decode 图能直接重放。
        decode_batch_desc = self._dispatch_decode(
            num_reqs, num_reqs, need_eager=is_profile
        )

        self.on_multi_step_decode_begin(num_reqs)
        # Generate the remaining num_speculative_steps - 1 draft tokens.
        skip_attn = bool(dummy_run)
        decode_fn = (
            self._fused_multi_step_decode
            if self.use_fused_multi_step_decode
            else self._multi_step_decode
        )
        decode_fn(num_reqs, skip_attn, decode_batch_desc)
        self.on_multi_step_decode_end(num_reqs)

        return self.draft_tokens[:num_reqs]

    # -------- 图分派（薄封装，唯一的写入点是 `record_selection`） --------
    def _dispatch_prefill(self, num_reqs: int, num_tokens: int,
                          uniform_token_count: int | None,
                          need_eager: bool = False) -> BatchExecutionDescriptor:
        manager = self.prefill_cudagraph_manager
        if manager is None:
            # 没建图管理器（图模式 NONE，但 74 关仍然会建"空管理器"——这条只为防御）。
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE, num_tokens=num_tokens, num_reqs=num_reqs
            )
        desc = manager.dispatch(
            num_reqs, num_tokens, uniform_token_count, need_eager=need_eager
        )
        manager.record_selection(desc, num_reqs, num_tokens)
        return desc

    def _dispatch_decode(self, num_reqs: int, num_tokens: int,
                         need_eager: bool = False) -> BatchExecutionDescriptor:
        manager = self.decode_cudagraph_manager
        if manager is None:
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE, num_tokens=num_tokens, num_reqs=num_reqs
            )
        desc = manager.dispatch(
            num_reqs, num_tokens, uniform_token_count=1, need_eager=need_eager
        )
        manager.record_selection(desc, num_reqs, num_tokens)
        return desc

    @property
    def cudagraph_selections(self) -> list[dict]:
        """两张图各自的"真实选中"账本（prefill + decode，按发生顺序）。"""
        selections = []
        for name, manager in (("prefill", self.prefill_cudagraph_manager),
                              ("decode", self.decode_cudagraph_manager)):
            if manager is None:
                continue
            selections += [dict(entry, graph=name) for entry in manager.selections]
        return selections

    # -------- 前向 --------

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
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

    def _prefill(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_token_indices = self.last_token_indices[:num_reqs]
        positions = self.input_buffers.positions[last_token_indices]
        idx_mapping = self.idx_mapping[:num_reqs]

        last_hidden_states, hidden_states = self._run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            cudagraph_runtime_mode,
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
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
    ) -> None:
        """逐步重建元数据的回退路径（对应上游同名方法）。

        每一步都要重建 `slot_mapping` 与注意力元数据——因为位置在步进，而"这一步的槽位"
        只有按新位置现算才对。这条路径**正确但慢**：K−1 次 CPU 侧重建 + K−1 次独立提交。
        """
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]

        attn_metadata = None
        slot_mappings = None
        for step in range(1, self.num_speculative_steps):
            # Rebuild every step when positions advance, or just once
            # on the first step when positions are constant (Gemma4 MTP).
            if not skip_attn and (self.advance_draft_positions or step == 1):
                slot_mappings = self._slot_mappings_by_layer(
                    self.block_tables.compute_slot_mappings(
                        idx_mapping,
                        query_start_loc,
                        positions,
                        batch_desc.num_tokens,
                    )
                )
                attn_metadata = self._build_draft_attn_metadata(
                    num_reqs=num_reqs,
                    num_reqs_padded=batch_desc.num_reqs or num_reqs,
                    num_tokens_padded=batch_desc.num_tokens,
                    step=step,
                )

            self.current_draft_step.fill_(step)

            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                assert self.decode_cudagraph_manager is not None
                self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            else:
                self._generate_draft(
                    num_reqs,
                    batch_desc.num_tokens,
                    attn_metadata,
                    slot_mappings,
                    batch_desc.cg_mode,
                )

    def _fused_multi_step_decode(
        self,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
    ) -> None:
        """融合多步：元数据只建一次，K−1 个草稿步在**一张图**里走完（上游同名方法）。

        为什么可以只建一次：步相关的量（`seq_lens`、`slot_mapping`）都是常驻缓冲的切片，
        步进时由内核原位改写；后端只要声明 `supports_draft_decode_metadata_update`，就表示
        "它没有第二份需要重建的步相关状态"（见 `_configure_fused_multi_step_decode`）。
        """
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]

        attn_metadata = None
        slot_mappings = None
        if not skip_attn:
            slot_mappings = self._slot_mappings_by_layer(
                self.block_tables.compute_slot_mappings(
                    idx_mapping,
                    query_start_loc,
                    positions,
                    batch_desc.num_tokens,
                )
            )
            attn_metadata = self._build_draft_attn_metadata(
                num_reqs=num_reqs,
                num_reqs_padded=batch_desc.num_reqs or num_reqs,
                num_tokens_padded=batch_desc.num_tokens,
                step=1,
            )

        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.decode_cudagraph_manager is not None
            self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            return

        self._generate_fused_drafts(
            num_reqs,
            batch_desc.num_tokens,
            attn_metadata,
            slot_mappings,
            batch_desc.cg_mode,
        )

    def _generate_fused_drafts(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        """K−1 个草稿步的循环本体（**这就是被录进图的那个函数**，上游同名方法）。

        注意这个函数里**只有捕获安全的操作**：

            `current_draft_step.fill_(step)`          设备张量原地写（图内允许）
            `block_tables.compute_slot_mappings(...)` Triton 内核，按新位置原算槽位
            `update_draft_decode_metadata(...)`       刷新步相关元数据（本仓库空操作，见上）

        任何"CPU 侧重建 metadata 对象"的写法都会让图**捕获时长得对、重放时根本没执行**
        （重放不跑 Python），所以这一层不允许出现。
        """
        idx_mapping = self.idx_mapping[:num_reqs]
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        attn_groups = (
            [group for groups in self.attn_groups for group in groups]
            if attn_metadata is not None
            else []
        )

        for step in range(1, self.num_speculative_steps):
            self.current_draft_step.fill_(step)
            self._generate_draft(
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                cudagraph_runtime_mode,
            )
            if (
                step < self.num_speculative_steps - 1
                and attn_metadata is not None
                and self.advance_draft_positions
            ):
                self.block_tables.compute_slot_mappings(
                    idx_mapping,
                    query_start_loc,
                    positions,
                    num_tokens_padded,
                )
                for attn_group in attn_groups:
                    attn_group.update_draft_decode_metadata(attn_metadata)

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        """一个草稿步：前向 → 采样 → 把结果写回下一步的输入（上游同名方法）。"""
        idx_mapping = self.idx_mapping[:num_reqs]
        positions = self.input_buffers.positions[:num_reqs]
        # Run the draft model forward pass.
        last_hidden_states, hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            cudagraph_runtime_mode,
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

    # -------- 注意力元数据 --------

    def _build_draft_attn_metadata(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        step: int,
    ) -> dict[str, Any] | None:
        """草稿 decode 步的注意力元数据（对应上游同名方法）。

        口径（每一步 1 行/请求）：

            `num_reqs_padded`     补齐后的请求数（decode 图可能比真实批大，多出来的请求
                                  `seq_lens=0`、槽位是哨兵 → 整行被掩掉）
            `num_tokens_padded`   补齐后的行数（与 `num_reqs_padded` 相等：每请求 1 行）
            `uniform_query_len=1` 固定形状路径（`_forward_padded`）要的每请求行数

        上游这里还有 `seq_lens_cpu_upper_bound`（FA3 用它避免一次 D2H 同步）与
        `causal` / `query_start_loc_np`（多模块 MTP 的非均匀布局）。本仓库的 Torch 后端
        **只用设备上的 `input_buffers.seq_lens`**（它就是"target 上下文 + 已走的步数"，由
        `prepare_decode_inputs` / `update_draft_inputs` 两个内核维护），所以不再需要 CPU 上界；
        多模块 MTP 属 80 关。**不留没人读的参数**：这里只保留真正被消费的三个形状量。
        """
        if not self.attn_groups:
            return None
        builder = self.attn_groups[0][0].get_metadata_builder()
        block_table = self.block_tables.input_block_tables[0][:num_reqs_padded]
        metadata = builder.build(
            query_start_loc=self.input_buffers.query_start_loc[: num_reqs_padded + 1],
            seq_lens=self.input_buffers.seq_lens[:num_reqs_padded],
            block_table=block_table,
            slot_mapping=self.block_tables.slot_mappings[0][:num_tokens_padded],
            num_reqs=num_reqs_padded,
            uniform_query_len=1,
        )
        return {name: metadata for name in self._draft_layer_names()}

    def _draft_attn_metadata_for_capture(self, input_batch: InputBatch) -> dict[str, Any]:
        """捕获期用的元数据回调（喂给 `prepare_inputs_to_capture`）。

        与运行期唯一的不同：这里的 `seq_lens` 是**假值**（`InputBatch.make_dummy` 写的
        "每请求 = 本请求的 query 行数"），`hidden_states`/`input_ids` 也还是捕获前的初值。

        **这里刻意不调 `build_for_cudagraph_capture()`**（它会顺手把 `seq_lens` 全填 1）。
        两条理由，第二条是实测踩出来的：

            1. "填 1 更快"对本后端的图内路径**不成立**：`_forward_padded` 无论如何都要按块表
               宽度把整段上下文 gather 出来（`num_positions = 块表宽度 × block_size`），
               `seq_lens` 只参与 mask，不决定 work 量。
            2. 填 1 会让**热身的 eager 那一遍**出现"整行被掩掉"：`make_dummy` 把 query 行数
               摊成 `num_tokens / num_reqs`（统一 decode 图就是 `1+K`），而 `seq_lens=1` 时
               `past_len = seq_len − query_len < 0` → 逐请求路径（`_attend`）里整行都被
               `-inf` 掩掉 → softmax 出 **NaN** → NaN 写进 KV 缓存与 `hidden_states`，
               后面每一次重放都跟着烂（实测：捕获完 `hidden_states[0]` 与 0 号块都是 NaN，
               重放出来的草稿 token 全是 0）。热身那一遍走的就是图的**同一个函数体**，
               所以它的数值必须也是干净的。
        """
        metadata = self._build_draft_attn_metadata(
            num_reqs=input_batch.num_reqs,
            num_reqs_padded=input_batch.num_reqs,
            num_tokens_padded=input_batch.num_tokens,
            step=1,
        )
        assert metadata is not None
        # 形状必须已经在**图外**定死（捕获之后改 `uniform_query_len` 等于换一张图）。
        for layer_metadata in metadata.values():
            if layer_metadata.uniform_query_len is None:
                raise ValueError(
                    "捕获草稿 decode 图必须是统一 decode 形状（每请求 1 行）："
                    f"收到 uniform_query_len=None，num_reqs={input_batch.num_reqs}、"
                    f"num_tokens={input_batch.num_tokens}")
        return metadata

    def _draft_layer_names(self) -> list[str]:
        if self.attn_groups:
            return list(self.attn_groups[0][0].layer_names)
        return list(self.draft_attn_layer_names)

    @staticmethod
    def _slot_mappings_by_layer(slot_mappings: torch.Tensor) -> dict[str, torch.Tensor]:
        """槽位张量 → forward 上下文要的字典形态（与 V2 Runner 同一条口径）。

        本仓库的注意力从 **metadata** 里取槽位（`AttentionMetadata.slot_mapping`），forward
        上下文里这份字典只有 64 关的 cache-only 层在读，而 V2 没有那条通路。留着它是为了
        "同一个字段只有一种形态"——同一个字段有时是张量、有时是字典，是最容易写错的那种接口。
        """
        return {"slot_mapping": slot_mappings}


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
