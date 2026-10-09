"""V2 的模型执行器（对应 vLLM `v1/worker/gpu/model_runner.py::GPUModelRunner`）。

与 V1（`worker/gpu_model_runner.py`）的分工相同、**数据流不同**。V2 的一轮分四个阶段，
顺序与上游逐条对齐（`execute_model` → `sample_tokens` 两段式，引擎侧本来就按这个协议调用）：

    1. 更新请求状态   `finish_requests / free_states / add_requests / update_requests`
                      → 改的是**常驻 slot** 上的状态（`RequestState`），与 batch 行无关
    2. 组装 batch      `gather_batch_req_state` → `prepare_inputs` → `prepare_attn`
                      → `idx_mapping`（行→slot）、`cu_num_logits`（含前导 0）、
                        `expanded_idx_mapping`（logits 行→slot）、块表 gather、slot_mapping 现算
    3. 前向            eager 调模型（本关不做图，74 关补），留下 hidden states 与辅助层特征
    4. 采样与回写      `sample`（普通采样器 or 拒绝采样）→ `postprocess_sampled`
                      → `post_update` 内核按 slot 写回 all_token_ids/total_len/last_sampled/已算数

第 4 步的"按 slot 写回"是 V2 的关键：CPU 侧**不需要知道**这一轮采了几个、拒了几个，
所以结果可以异步交付（`AsyncOutput` 在侧流上拷，见 `async_utils.py`）。

裁剪（差异账本逐条）：LoRA / 多模态 / PP / DCP / PCP / MoE+EPLB / KV connector / 池化模型 /
CUDA Graph（74 关）/ adaptive verification（79 关）/ prompt logprobs / `_dummy_run`（本仓库容量由
配置给定，不做显存 profiling）。
"""

from typing import Any, NamedTuple

import numpy as np
import torch

from ...config import VllmConfig
from ...outputs import ModelRunnerOutput
from ..gpu.async_utils import AsyncOutput
from ..gpu.block_table import BlockTables
from ..gpu.input_batch import (
    InputBatch,
    InputBuffers,
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    post_update,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)
from ..gpu.sample.sampler import Sampler
from ..gpu.spec_decode import init_speculator
from ..gpu.spec_decode.rejection_sampler import RejectionSampler
from ..gpu.spec_decode.speculator import DraftModelSpeculator
from ..gpu.spec_decode.utils import DraftTokensHandler
from ..gpu.states import RequestState
from ..gpu.structured_outputs import StructuredOutputsWorker
from ..gpu.buffer_utils import async_copy_to_gpu, set_default_max_concurrency


class ExecuteModelState(NamedTuple):
    input_batch: InputBatch
    attn_metadata: dict[str, Any] | None
    slot_mappings_by_layer: dict[str, torch.Tensor] | None
    hidden_states: torch.Tensor | None
    aux_hidden_states: list[torch.Tensor] | None
    finished_req_ids: set[str]


class BatchReqState(NamedTuple):
    """CPU request state for a scheduled batch, in batch (sorted) order."""

    req_ids: list[str]
    num_scheduled_tokens: np.ndarray  # [num_reqs]
    num_tokens: int
    idx_mapping_np: np.ndarray  # [num_reqs]
    prefill_len_np: np.ndarray  # [num_reqs]
    num_computed_prefill_tokens_np: np.ndarray  # [num_reqs]
    is_prefilling_np: np.ndarray  # [num_reqs]
    has_prefill: bool


def sort_batch_req_ids(
    num_tokens_per_req: dict[str, int],
    draft_tokens: dict[str, list[int]],
    decode_query_len: int,
) -> list[str]:
    # Order verification/decode -> short_extend -> prefill;
    # split_decodes_and_prefills relies on decode-like requests leading.
    key = lambda r: (
        not draft_tokens.get(r),
        (num := num_tokens_per_req[r]) != decode_query_len,
        num,
    )
    return sorted(num_tokens_per_req, key=key)


class GPUModelRunner:
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config

        self.device = device
        self.dtype = self.model_config.dtype
        self.block_size = vllm_config.cache_config.block_size

        self.vocab_size = self.model_config.get_vocab_size()
        self.max_model_len = self.model_config.max_model_len
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.scheduler_config.max_num_seqs

        self.output_copy_stream = (
            torch.cuda.Stream(self.device) if torch.cuda.is_available() else None
        )

        # Size the UVA buffer pools to the max number of concurrent in-flight
        # steps. Must run before any pooled buffer is constructed.
        set_default_max_concurrency(
            getattr(vllm_config, "max_concurrent_batches", 2)
        )

        # Speculative decoding.
        self.speculator = None
        self.use_aux_hidden_state_outputs = False
        self.num_speculative_steps = vllm_config.num_speculative_tokens
        # `num_new_sampled_tokens_per_step`（上游 `ModelState` 的字段）：本仓库的模型每步
        # 采 1 个 bonus token（扩散模型是 0，本仓库没有）。
        self.num_new_sampled_tokens_per_step = 1
        self.decode_query_len = (
            self.num_speculative_steps + self.num_new_sampled_tokens_per_step
        )

        # Draft tokens propagation - for spec-dec + struct outputs.
        self.draft_tokens_handler = DraftTokensHandler(self.device)

        self.req_states = RequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=self.num_speculative_steps,
            vocab_size=self.vocab_size,
            device=self.device,
        )
        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=self.device,
        )

        self.sampler: Sampler | None = None
        self.rejection_sampler: RejectionSampler | None = None
        self.structured_outputs_worker: StructuredOutputsWorker | None = None

        # For transferring state from execute_model to subsequent sample_tokens call.
        self.execute_model_state: ExecuteModelState | None = None

        self.kv_cache_config = None
        self.block_tables: BlockTables | None = None
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.model = None
        self.failure: str | None = None
        if not torch.cuda.is_available():
            # V2 的输入组装（prepare_inputs / 采样 / 验证）全是 Triton 内核，且常驻状态走 UVA。
            raise NotImplementedError(
                "V2 Model Runner 需要 CUDA（输入组装与采样是 Triton 内核、常驻状态走 UVA）；"
                "没有 CUDA 时请用 V1 路径（VLLM_USE_V2_MODEL_RUNNER=0）"
            )
        # 70 关：执行侧要知道"草稿从哪来"；异步调度在本仓库一律在引擎层拒绝，
        # 所以这里恒 False（放开属 74/75 关，见 AGENTS §1）。
        self.async_scheduling = False

    def update_max_model_len(self, max_model_len: int) -> None:
        self.max_model_len = max_model_len
        self.req_states.max_model_len = max_model_len

    # -------- 模型与缓存 --------

    def load_model(self) -> None:
        from ...model_loader import get_model

        self.model = get_model(self.model_config, self.device)

        if self.speculative_config is not None:
            self.speculator = init_speculator(self.vllm_config, self.device)
            self.speculator.load_model(self.model)
            if self.speculative_config.method == "eagle3":
                # Drafting may require auxiliary hidden states from target model outputs.
                self.use_aux_hidden_state_outputs = True
                # 与 V1 同一条取值链：draft 配置里的辅助层 id，缺省用模型的默认值。
                layers = self.speculative_config.eagle_aux_hidden_state_layers() \
                    or self.model.get_eagle3_default_aux_hidden_state_layers()
                self.model.set_aux_hidden_state_layers(layers)

        # Initialize samplers. Model states may override via custom_sampler().
        self.sampler = Sampler(
            max_num_reqs=self.max_num_reqs,
            vocab_size=self.vocab_size,
            device=self.device,
            req_states=self.req_states,
            logprobs_mode=self.model_config.logprobs_mode,
            num_speculative_tokens=self.num_speculative_steps,
        )
        if self.speculative_config is not None:
            self.rejection_sampler = RejectionSampler(
                self.sampler, self.speculative_config, self.device
            )

        self.structured_outputs_worker = StructuredOutputsWorker(
            max_num_logits=self.max_num_reqs * self.decode_query_len,
            vocab_size=self.vocab_size,
            device=self.device,
            mask_stride=self.decode_query_len,
            num_bonus_tokens=self.num_new_sampled_tokens_per_step,
        )

    def get_kv_cache_spec(self):
        raise NotImplementedError(
            "本仓库的 KV 容量由配置给定（`CacheConfig`），不经 Runner 做显存 profiling"
        )

    def initialize_kv_cache(self, kv_cache_config) -> dict[str, torch.Tensor]:
        """按 KV 规格分配物理缓存并绑定到 Attention 层，同时建 V2 的块表集合。

        顺序与上游一致：**先建块表**（`BlockTables` 的行 = 常驻 slot）、再把块表与输入缓冲
        交给提议者（`set_attn`）、最后分配物理缓存并绑定到 Attention 层。
        """
        if self.model is None:
            raise RuntimeError("先 load_model() 再初始化 KV 缓存（需要模型的 dtype/qk 头数）")
        if kv_cache_config.block_size != self.block_size:
            raise ValueError(
                f"KV 规格里的 block_size={kv_cache_config.block_size} 与 Runner 的 "
                f"{self.block_size} 不一致：块大小决定 slot 怎么算，两边不同会出现"
                f"'写进去的和读出来的不是同一块'这种最难查的错"
            )
        self.kv_cache_config = kv_cache_config

        max_num_blocks_per_req = -(-self.max_model_len // self.block_size)
        self.block_tables = BlockTables(
            block_sizes=[self.block_size],
            max_num_reqs=self.max_num_reqs,
            max_num_batched_tokens=self.max_num_tokens,
            max_num_blocks_per_group=[max_num_blocks_per_req],
            device=self.device,
            kernel_block_sizes=[self.block_size],
        )
        if isinstance(self.speculator, DraftModelSpeculator):
            self.speculator.set_attn(self.block_tables, self.input_buffers)

        dtype = next(self.model.parameters()).dtype
        self.kv_caches = {}
        for layer_name, layer in self._attention_layers().items():
            cache = torch.zeros(
                2,
                kv_cache_config.num_gpu_blocks,
                self.block_size,
                layer.num_kv_heads,
                layer.head_size,
                dtype=dtype,
                device=self.device,
            )
            layer.kv_cache = cache
            self.kv_caches[layer_name] = cache
        if self.speculator is not None:
            # 草稿层的物理缓存：与 target **同形同块数**（块号在两边代表同一段位置），
            # 但 K/V 各写各的（与 V1 提议者的 `_allocate_kv_caches()` 同一套做法）。
            self._allocate_draft_kv_caches(kv_cache_config)
        return self.kv_caches

    def _allocate_draft_kv_caches(self, kv_cache_config) -> None:
        from ...attention.layer import Attention

        draft_model = self.speculator.model
        draft_dtype = next(draft_model.parameters()).dtype
        names: set[str] = set()
        for name, module in draft_model.named_modules():
            if not isinstance(module, Attention):
                continue
            cache = torch.zeros(
                2,
                kv_cache_config.num_gpu_blocks,
                self.block_size,
                module.num_kv_heads,
                module.head_size,
                dtype=draft_dtype,
                device=self.device,
            )
            module.kv_cache = cache
            names.add(name)
        if not names:
            raise ValueError(
                f"{type(draft_model).__name__} 里没有 Attention 层，无法绑定草稿的 KV 缓存"
            )
        self.speculator.draft_attn_layer_names = names

    def _attention_layers(self):
        from ...attention.layer import Attention

        layers = {}
        for name, module in self.model.named_modules():
            if not isinstance(module, Attention):
                continue
            if module.layer_name != name:
                raise ValueError(
                    f"Attention 层名与模块路径不一致：layer_name={module.layer_name!r}，"
                    f"实际路径={name!r}。forward 上下文按层名取 metadata，两者必须相同"
                )
            layers[name] = module
        if not layers:
            raise ValueError(f"{type(self.model).__name__} 里没有 Attention 层，无法绑定 KV 缓存")
        return layers

    def capture_model(self) -> None:
        """V2 的 CUDA Graph 属 74 关：本关明确不做（不是"悄悄不做"）。"""
        return None

    # -------- 请求状态（按常驻 slot） --------

    def _remove_request(self, req_id: str) -> bool:
        req_idx = self.req_states.remove_request(req_id)
        if req_idx is None:
            return False
        return True

    def finish_requests(self, scheduler_output) -> None:
        finished_req_ids = scheduler_output.finished_req_ids
        preempted_req_ids = getattr(scheduler_output, "preempted_req_ids", None) or set()
        for req_id in set(finished_req_ids) | set(preempted_req_ids):
            self._remove_request(req_id)

    def free_states(self, scheduler_output) -> None:
        """上游这里释放多模态编码缓存；本仓库没有该通路（差异账本）。"""
        return None

    def add_requests(self, scheduler_output) -> None:
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            # Streaming input update: request already exists from a prior
            # chunk. Remove old state so it can be cleanly re-added below
            # with the updated prompt_token_ids.
            self._remove_request(req_id)

            prompt_len = len(new_req_data.prompt_token_ids)
            all_token_ids = getattr(new_req_data, "prefill_token_ids", None)
            if all_token_ids is None:
                # 非 V2 的调度器不送 prefill_token_ids：那时"喂进去的长度"就是 prompt。
                all_token_ids = list(new_req_data.prompt_token_ids)
            sampling_params = new_req_data.sampling_params
            self.req_states.add_request(
                req_id=req_id,
                prompt_len=prompt_len,
                all_token_ids=all_token_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                max_tokens=sampling_params.max_tokens if sampling_params else 1,
            )
            req_index = self.req_states.req_id_to_index[req_id]

            self.block_tables.append_block_ids(
                req_index, new_req_data.block_ids, overwrite=True
            )

            if sampling_params is not None:
                assert self.sampler is not None
                self.sampler.add_request(req_index, prompt_len, sampling_params)

        if scheduler_output.scheduled_new_reqs:
            self.req_states.apply_staged_writes()
        if self.sampler is not None:
            self.sampler.apply_staged_writes()

    def update_requests(self, scheduler_output) -> None:
        # Add new blocks and update num_computed_tokens for the existing requests.
        reqs = scheduler_output.scheduled_cached_reqs
        num_computed_tokens_np = self.req_states.num_computed_tokens_np
        for req_id, num_computed_tokens, req_new_block_ids in zip(
            reqs.req_ids, reqs.num_computed_tokens, reqs.new_block_ids
        ):
            req_index = self.req_states.req_id_to_index[req_id]
            num_computed_tokens_np[req_index] = num_computed_tokens
            if req_new_block_ids is not None:
                self.block_tables.append_block_ids(
                    req_index, req_new_block_ids, overwrite=False
                )

        # Update CPU num_computed_prefill_tokens.
        np.minimum(
            self.req_states.num_computed_tokens_np,
            self.req_states.prefill_len.np,
            out=self.req_states.num_computed_prefill_tokens,
        )

    def gather_batch_req_state(
        self, scheduler_output
    ) -> BatchReqState:
        """Gather CPU request state for the scheduled batch, in batch order."""
        num_tokens_per_req = scheduler_output.num_scheduled_tokens
        num_reqs = len(num_tokens_per_req)
        num_toks = scheduler_output.total_num_scheduled_tokens

        draft_tokens = scheduler_output.scheduled_spec_decode_tokens
        # batch_idx -> req_id
        req_ids = sort_batch_req_ids(
            num_tokens_per_req, draft_tokens, self.decode_query_len
        )

        numtoks_iter = map(num_tokens_per_req.__getitem__, req_ids)
        num_scheduled_tokens = np.fromiter(numtoks_iter, dtype=np.int32, count=num_reqs)

        idx_mapping_iter = map(self.req_states.req_id_to_index.__getitem__, req_ids)
        idx_mapping_np = np.fromiter(idx_mapping_iter, dtype=np.intp, count=num_reqs)
        prefill_len_np = self.req_states.prefill_len.np[idx_mapping_np]
        num_computed_prefill_tokens_np = self.req_states.num_computed_prefill_tokens[
            idx_mapping_np
        ]
        is_prefilling_np = num_computed_prefill_tokens_np < prefill_len_np

        return BatchReqState(
            req_ids=req_ids,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_toks,
            idx_mapping_np=idx_mapping_np,
            prefill_len_np=prefill_len_np,
            num_computed_prefill_tokens_np=num_computed_prefill_tokens_np,
            is_prefilling_np=is_prefilling_np,
            has_prefill=bool(is_prefilling_np.any()),
        )

    # -------- 输入组装 --------

    def prepare_inputs(
        self, scheduler_output, batch_req_state: BatchReqState
    ) -> InputBatch:
        num_tokens = batch_req_state.num_tokens
        num_tokens_after_padding = num_tokens
        assert num_tokens > 0

        req_ids = batch_req_state.req_ids
        num_scheduled_tokens_np = batch_req_state.num_scheduled_tokens
        idx_mapping_np = batch_req_state.idx_mapping_np
        idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)
        num_reqs = len(req_ids)

        # Get the number of draft tokens for each request.
        draft_tokens = scheduler_output.scheduled_spec_decode_tokens
        num_draft_tokens_per_req = None
        if not draft_tokens:
            # No draft token scheduled (common case).
            total_num_draft_tokens = 0
            total_num_logits = num_reqs
            cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.arange(
                num_reqs + 1, device=self.device, dtype=torch.int32
            )
            expanded_idx_mapping = idx_mapping
            expanded_local_pos = torch.zeros(
                num_reqs, dtype=torch.int32, device=self.device
            )
        else:
            num_draft_tokens_per_req = np.fromiter(
                (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
                dtype=np.int32,
                count=num_reqs,
            )
            num_bonus_tokens = self.num_new_sampled_tokens_per_step
            total_num_draft_tokens = int(num_draft_tokens_per_req.sum())
            total_num_logits = num_reqs * num_bonus_tokens + total_num_draft_tokens
            num_logits = num_draft_tokens_per_req + num_bonus_tokens
            cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
            cu_num_logits_np[0] = 0
            np.cumsum(num_logits, out=cu_num_logits_np[1:])
            cu_num_logits = async_copy_to_gpu(cu_num_logits_np, device=self.device)

        # Get query_start_loc.
        num_reqs_padded = num_reqs
        query_start_loc_np = np.empty(self.max_num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens_np, out=query_start_loc_np[1 : num_reqs + 1])
        # Pad for full CUDA graph mode.
        # Some attention backends like FA3 require query_start_loc to be non-decreasing.
        query_start_loc_np[num_reqs + 1 :] = num_tokens
        query_start_loc = self.input_buffers.query_start_loc
        async_copy_to_gpu(query_start_loc_np, out=query_start_loc)
        if draft_tokens:
            expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
                idx_mapping, total_num_logits, cu_num_logits, self.decode_query_len
            )
        query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]
        query_start_loc = query_start_loc[: num_reqs_padded + 1]

        # Get prefill tokens if any.
        if batch_req_state.has_prefill:
            prepare_prefill_inputs(
                self.input_buffers.input_ids,
                self.req_states.next_prefill_tokens,
                idx_mapping,
                query_start_loc,
                self.req_states.all_token_ids.gpu,
                self.req_states.prefill_len.gpu,
                self.req_states.num_computed_tokens.gpu,
            )

        # Prepare positions and seq_lens.
        prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            self.req_states.num_computed_tokens.gpu,
            self.input_buffers.positions,
            self.input_buffers.seq_lens,
        )
        seq_lens = self.input_buffers.seq_lens[:num_reqs_padded]

        # Some input token ids are directly read from the last sampled tokens
        # and draft tokens. Also, get the logits indices to sample tokens from.
        logits_indices = combine_sampled_and_draft_tokens(
            self.input_buffers.input_ids,
            idx_mapping,
            self.req_states.last_sampled_tokens,
            query_start_loc,
            seq_lens,
            self.req_states.prefill_len.gpu,
            self.req_states.draft_tokens,
            cu_num_logits,
            total_num_logits,
            self.num_new_sampled_tokens_per_step,
        )

        # CPU upper bound on seq_lens; padded entries left at zero.
        num_computed_tokens_np = self.req_states.num_computed_tokens_np[idx_mapping_np]
        seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
        np.add(
            num_computed_tokens_np,
            num_scheduled_tokens_np,
            out=seq_lens_cpu_upper_bound_np[:num_reqs],
        )
        seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens_cpu_upper_bound_np)

        return InputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs_padded,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens_np,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            num_draft_tokens=total_num_draft_tokens,
            num_draft_tokens_per_req=num_draft_tokens_per_req,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            num_computed_tokens_np=num_computed_tokens_np,
            prefill_len_np=batch_req_state.prefill_len_np,
            num_computed_prefill_tokens_np=batch_req_state.num_computed_prefill_tokens_np,
            is_prefilling_np=batch_req_state.is_prefilling_np,
            has_prefill=batch_req_state.has_prefill,
            input_ids=self.input_buffers.input_ids[:num_tokens_after_padding],
            positions=self.input_buffers.positions[:num_tokens_after_padding],
            is_padding=self.input_buffers.is_padding[:num_tokens_after_padding],
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=scheduler_output.has_structured_output_requests,
        )

    def prepare_attn(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        # Block tables: num_kv_cache_groups x [num_reqs_padded, max_num_blocks].
        block_tables = self.block_tables.gather_block_tables(
            input_batch.idx_mapping,
            num_reqs_padded=input_batch.num_reqs_after_padding,
        )
        # Slot mappings: [num_kv_cache_groups, num_tokens_padded].
        # Kernel pads beyond num_tokens with PAD_SLOT_ID.
        slot_mappings = self.block_tables.compute_slot_mappings(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            input_batch.positions,
            num_tokens_padded=input_batch.num_tokens_after_padding,
        )
        return block_tables, slot_mappings

    def _build_attn_metadata(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
    ) -> dict[str, Any]:
        """把 V2 的批次视图翻成注意力层要的元数据（本项目只有一个后端，一份 metadata）。

        上游这里是 `model_state.prepare_attn(...)`，按 `attn_groups` 分组构造**每组的后端
        metadata**；本仓库只有一个 Torch 后端、一个 KV group，所以"一组"就是全部层。
        """
        from ...attention.backends.torch_sdpa import TorchAttentionBackend

        builder = TorchAttentionBackend.get_builder_cls()(self.block_size)
        metadata = builder.build(
            query_start_loc=input_batch.query_start_loc,
            seq_lens=input_batch.seq_lens,
            block_table=block_tables[0],
            slot_mapping=slot_mappings[0][: input_batch.num_tokens_after_padding],
            num_reqs=input_batch.num_reqs_after_padding,
        )
        return {name: metadata for name in self.kv_caches}

    # -------- 一轮 --------

    def _check_usable(self) -> None:
        if self.failure is not None:
            raise RuntimeError(f"这个 Runner 已经失败，不再接受新的一轮：{self.failure}")

    @torch.inference_mode()
    def execute_model(self, scheduler_output, non_block: bool = False):
        self._check_usable()
        if self.execute_model_state is not None:
            raise RuntimeError(
                "上一轮 execute_model() 的结果还没被 sample_tokens() 消费，不能开始新的一轮"
            )

        # Update the request states.
        self.finish_requests(scheduler_output)
        self.free_states(scheduler_output)
        self.add_requests(scheduler_output)
        self.update_requests(scheduler_output)
        self.block_tables.apply_staged_writes()
        if scheduler_output.total_num_scheduled_tokens == 0:
            # No need to run the model.
            return ModelRunnerOutput.make_empty()

        batch_req_state = self.gather_batch_req_state(scheduler_output)
        input_batch = self.prepare_inputs(scheduler_output, batch_req_state)
        block_tables, slot_mappings = self.prepare_attn(input_batch)
        slot_mappings_by_layer = {"slot_mapping": slot_mappings}
        attn_metadata = self._build_attn_metadata(
            input_batch, block_tables, slot_mappings
        )

        from ...config import CUDAGraphMode
        from ...forward_context import BatchDescriptor, set_forward_context

        try:
            with set_forward_context(
                attn_metadata,
                num_tokens=input_batch.num_tokens_after_padding,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                batch_descriptor=BatchDescriptor(
                    num_tokens=input_batch.num_tokens_after_padding
                ),
                slot_mapping=slot_mappings_by_layer,
            ):
                model_output = self.model(
                    input_ids=input_batch.input_ids,
                    positions=input_batch.positions,
                )
        except Exception as exc:  # noqa: BLE001 —— 任何异常都让 Runner 停摆（205 §5）
            self.execute_model_state = None
            self.failure = f"{type(exc).__name__}: {exc}"
            raise

        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = model_output
        else:
            hidden_states = model_output
            aux_hidden_states = None

        self.execute_model_state = ExecuteModelState(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            hidden_states=hidden_states,
            aux_hidden_states=aux_hidden_states,
            finished_req_ids=scheduler_output.finished_req_ids,
        )
        return None

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output,
    ):
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        if grammar_output is not None:
            # Apply grammar bitmask to the logits in-place.
            assert self.structured_outputs_worker is not None
            self.structured_outputs_worker.apply_grammar_bitmask(
                logits,
                input_batch,
                grammar_output.structured_output_request_ids,
                grammar_output.grammar_bitmask,
            )

        if input_batch.num_draft_tokens == 0 or self.rejection_sampler is None:
            assert self.sampler is not None
            sampler_output = self.sampler(logits, input_batch)
        else:
            # Rejection sampling for spec decoding.
            assert self.rejection_sampler is not None
            assert self.speculator is not None
            sampler_output = self.rejection_sampler(
                logits,
                input_batch,
                # Draft logits are needed for probabilistic rejection sampling.
                self.speculator.draft_logits,
            )

        return sampler_output, sampler_output.num_sampled, sampler_output.num_rejected

    def postprocess_sampled(
        self,
        idx_mapping: torch.Tensor,  # May include -1 for masked entries
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        query_start_loc: torch.Tensor | None = None,
    ) -> None:
        # Update the number of computed tokens.
        output_bin_counts = self.sampler.penalties_state.output_bin_counts
        post_update(
            idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.last_sampled_tokens,
            output_bin_counts,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
            self.req_states.all_token_ids.gpu,
            self.req_states.total_len.gpu,
        )

    @torch.inference_mode()
    def sample_tokens(self, grammar_output, non_block: bool = False):
        if self.execute_model_state is None:
            # The prior execute_model call must have failed.
            return None

        input_batch = self.execute_model_state.input_batch
        attn_metadata = self.execute_model_state.attn_metadata
        slot_mappings_by_layer = self.execute_model_state.slot_mappings_by_layer
        hidden_states = self.execute_model_state.hidden_states
        aux_hidden_states = self.execute_model_state.aux_hidden_states
        finished_req_ids = self.execute_model_state.finished_req_ids
        self.execute_model_state = None

        # Last rank: sample tokens
        sampler_output, num_sampled, num_rejected = self.sample(
            hidden_states, input_batch, grammar_output
        )

        # Prepare the model runner output.
        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            # NOTE(woosuk): req_id_to_index is unused in this model runner.
            # Only for compatibility with the existing model runner and scheduler.
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            sampled_token_ids=None,  # type: ignore
        )
        if torch.cuda.is_available():
            # Start async output copy here so that it can overlap with speculator proposal.
            async_output = AsyncOutput(
                model_runner_output=model_runner_output,
                sampler_output=sampler_output,
                num_sampled_tokens=num_sampled,
                main_stream=torch.cuda.current_stream(self.device),
                copy_stream=self.output_copy_stream,
            )
        else:
            async_output = None

        # Postprocess results and update request states.
        # NOTE: This is intentionally done after creating the AsyncOutput,
        # ensuring that `copy_event` is recorded before calling postprocess.
        self.postprocess_sampled(
            input_batch.idx_mapping,
            sampler_output.sampled_token_ids,
            num_sampled,
            num_rejected,
            input_batch.query_start_loc,
        )

        if self.speculator is not None:
            assert self.sampler is not None
            with torch.inference_mode():
                draft_tokens = self.speculator.propose(
                    input_batch,
                    attn_metadata,
                    slot_mappings_by_layer,
                    hidden_states,
                    aux_hidden_states,
                    num_sampled,
                    num_rejected,
                    self.req_states.last_sampled_tokens,
                    self.req_states.next_prefill_tokens,
                    self.sampler.sampling_states.temperature.gpu,
                    self.sampler.sampling_states.seeds.gpu,
                )
            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens

        if self.num_speculative_steps > 0:
            # Spec-decode and diffusion LLMs both use draft tokens but the latter does
            # not have a speculator (i.e., self.speculator is None)
            self.draft_tokens_handler.set_draft_tokens(
                input_batch,
                self.req_states.draft_tokens[input_batch.idx_mapping],
            )

        if async_output is not None and non_block:
            return async_output
        if async_output is not None:
            # 同步路径：交付边界就是这里（与 V1 同款；`step()` 拿到的是 CPU 结果）。
            return async_output.get_output()
        return None

    def take_draft_token_ids(self):
        return self.draft_tokens_handler.get_draft_tokens()
