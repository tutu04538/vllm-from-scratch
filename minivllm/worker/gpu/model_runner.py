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

from ...config import CUDAGraphMode, VllmConfig
from ...core.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroup, SlidingWindowSpec)
from ...outputs import ModelRunnerOutput
from ..gpu.async_utils import AsyncOutput, StepTimingCollector
from ..gpu.block_table import BlockTables, PAD_SLOT_ID
from ..gpu.cudagraph_utils import BatchExecutionDescriptor, ModelCudaGraphManager
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
from ..gpu.spec_decode.rejection_sampler import RejectionSampler, get_max_chunk_logits
from ..gpu.spec_decode.speculator import DraftModelSpeculator
from ...attention.backend import AttentionCGSupport
from ...attention.backends import get_attention_backend
from ..utils import AttentionGroup, get_uniform_decode_token_count
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
        self.attention_backend_name = str(
            getattr(vllm_config, "attention_backend", "torch")).lower()
        self.attention_backend = get_attention_backend(self.attention_backend_name)
        self.attn_builder_cls = self.attention_backend.get_builder_cls()
        self.attn_metadata_builder = self.attn_builder_cls(self.block_size)

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

        # 多模块 MTP 在 chunked prefill 时每个模块吃一枚 future prompt token，
        # 所以 lookahead 要按 K 行准备；其它提议者只读紧邻的一枚（上游同款）。
        num_prefill_lookahead = (
            self.num_speculative_steps
            if self.speculative_config is not None
            and self.speculative_config.use_multi_module_mtp()
            else 1
        )
        self.req_states = RequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=self.num_speculative_steps,
            vocab_size=self.vocab_size,
            device=self.device,
            num_prefill_lookahead=num_prefill_lookahead,
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
        # 79 关：自适应验证 manager 在 `initialize_kv_cache()` 里按实际注意力后端能力建；
        # 成本曲线由 `capture_model()` 里的 eager dummy step 实测生成。
        self.adaptive_verification = None
        self.step_timing = StepTimingCollector()
        # Phase 1：target 的 FULL_DECODE_ONLY 图；None = target 走 eager。
        self.target_cudagraph_manager: ModelCudaGraphManager | None = None
        if not torch.cuda.is_available():
            # V2 的输入组装（prepare_inputs / 采样 / 验证）全是 Triton 内核，且常驻状态走 UVA。
            raise NotImplementedError(
                "V2 Model Runner 需要 CUDA（输入组装与采样是 Triton 内核、常驻状态走 UVA）；"
                "没有 CUDA 时请用 V1 路径（VLLM_USE_V2_MODEL_RUNNER=0）"
            )
        # 70 关：执行侧要知道"草稿从哪来"；异步调度在本仓库一律在引擎层拒绝，
        # 所以这里恒 False（放开属 74/75 关，见 AGENTS §1）。
        self.async_scheduling = False
        # 74 关：图模式与"元数据的键集合"都在 `initialize_kv_cache()` 里定；先给安全默认值，
        # 免得"没初始化就查"读到属性不存在的假象（默认 NONE = 不捕获任何图）。
        self.cudagraph_mode = CUDAGraphMode.NONE
        self.cudagraph_capture_stats: dict | None = None
        self._attn_metadata_layer_names: set[str] = set()
        # 80E.E2/81：layer_name → KV group。target 的 Q-only 层与 assistant 的 Q-only 层
        # 都取**源层**的 group；metadata 必须按这个映射逐层给，不能全批共用第 0 组。
        self._attn_metadata_layer_groups: dict[str, int] = {}

    def update_max_model_len(self, max_model_len: int) -> None:
        self.max_model_len = max_model_len
        self.req_states.max_model_len = max_model_len

    # -------- 模型与缓存 --------

    def load_model(self) -> None:
        from ...attention.backends import TorchAttentionImpl
        from ...attention.layer import set_attention_impl_cls
        from ...model_loader import get_model

        # 只有 target 用选定的后端；speculator 的草稿层仍是 Torch 参考后端。
        set_attention_impl_cls(self.attention_backend.get_impl_cls())
        try:
            self.model = get_model(self.model_config, self.device)
        finally:
            set_attention_impl_cls(TorchAttentionImpl)

        if self.speculative_config is not None:
            self.speculator = init_speculator(self.vllm_config, self.device)
            self.speculator.load_model(self.model)
            # 草稿的注意力层名：`set_attn()` 要用它建"草稿自己的"注意力组，而 target 的
            # 注意力元数据也要**覆盖这些层名**（上游 `init_attn_backend` 在 runner 侧不带
            # `active_layer_names`，于是 target 的 metadata 字典天然包含草稿层；草稿 prefill
            # 复用 target 那份 metadata 时，草稿层就是靠这个键找到自己的元数据）。
            self.speculator.draft_attn_layer_names = self._draft_attention_layer_names()
            if self.speculative_config.method in ("eagle3", "dflash", "dspark"):
                # Drafting may require auxiliary hidden states from target model outputs.
                # 76/77 关：DFlash 与 DSpark 的上下文 KV 就是这些特征翻译出来的，**必须**采集。
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
        """返回按 Attention 层推导的真实 `KVCacheConfig`（对应上游 `get_kv_cache_spec()`）。

        容量仍由配置给定（本仓库不做显存 profiling）；这里只回答**分组与每组形状**。
        分组键是 `(类型, head_size, num_kv_heads, sliding_window)`：同组层必须共用块表。
        Q-only 层不拥有 cache，不进入 group；它们的层名在 `build_kv_cache_config()` 后
        由 `_attn_metadata_layer_groups` 指向源层 group。
        """
        if self.model is None:
            raise RuntimeError("先 load_model() 再推导 KV 规格")
        from ...attention.layer import Attention

        dtype = next(p.dtype for p in self.model.parameters() if p.is_floating_point())
        groups: list[KVCacheGroup] = []
        group_keys: dict[tuple, int] = {}
        layer_to_group: dict[str, int] = {}
        q_only_layers: list[tuple[str, Attention]] = []

        for name, layer in self._attention_layers().items():
            if layer.q_only:
                q_only_layers.append((name, layer))
                continue
            sliding = layer.per_layer_sliding_window
            key = (layer.head_size, layer.num_kv_heads, sliding)
            if key not in group_keys:
                spec = (SlidingWindowSpec(
                            block_size=self.block_size,
                            num_kv_heads=layer.num_kv_heads,
                            head_size=layer.head_size,
                            dtype=dtype,
                            sliding_window=int(sliding),
                        ) if sliding is not None else FullAttentionSpec(
                            block_size=self.block_size,
                            num_kv_heads=layer.num_kv_heads,
                            head_size=layer.head_size,
                            dtype=dtype,
                        ))
                group_keys[key] = len(groups)
                groups.append(KVCacheGroup(spec, [name]))
            else:
                groups[group_keys[key]].layer_names.append(name)
            group_id = group_keys[key]
            layer_to_group[name] = group_id
            layer.kv_cache_group_id = group_id

        for name, layer in q_only_layers:
            source = layer.kv_sharing_target_layer_name
            if source is None:
                raise ValueError(
                    f"Q-only 层 {name!r} 没有 kv_sharing_target_layer_name："
                    f"不能为它单独建一份 KV（会静默读错上下文）")
            if source not in layer_to_group:
                raise KeyError(
                    f"Q-only 层 {name!r} 的源层 {source!r} 不存在或本身也是 Q-only")
            group_id = layer_to_group[source]
            layer_to_group[name] = group_id
            layer.kv_cache_group_id = group_id
            source_layer = self._attention_layers()[source]
            if (layer.num_kv_heads, layer.head_size) != (
                    source_layer.num_kv_heads, source_layer.head_size):
                raise ValueError(
                    f"Q-only 层 {name!r} 与源层 {source!r} 的 KV 形状不一致："
                    f"{(layer.num_kv_heads, layer.head_size)} vs "
                    f"{(source_layer.num_kv_heads, source_layer.head_size)}")

        self._attn_metadata_layer_groups = layer_to_group
        return KVCacheConfig(
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            kv_cache_groups=groups,
            enable_prefix_caching=bool(self.cache_config.enable_prefix_caching),
        )

    def build_kv_cache_config(self) -> KVCacheConfig:
        """Executor/Engine 拿容量前调用的入口（上游 `get_kv_cache_spec()` 的别名包装）。"""
        return self.get_kv_cache_spec()

    def initialize_kv_cache(self, kv_cache_config) -> dict[str, torch.Tensor]:
        """按 KV 规格分配物理缓存并绑定到 Attention 层，同时建 V2 的块表集合。

        顺序与上游一致：**先建块表**（`BlockTables` 的行 = 常驻 slot）、再把块表与输入缓冲
        交给提议者（`set_attn`）、最后分配物理缓存并绑定到 Attention 层。
        """
        if self.model is None:
            raise RuntimeError("先 load_model() 再初始化 KV 缓存（需要模型的 dtype/qk 头数）")
        if not isinstance(kv_cache_config, KVCacheConfig):
            raise TypeError(
                "V2 的 initialize_kv_cache 需要 KVCacheConfig（含 KV group）；"
                f"收到 {type(kv_cache_config).__name__}")
        for spec in kv_cache_config.block_sizes:
            if spec != self.block_size:
                raise NotImplementedError(
                    f"本阶段的多 group 执行仍要求所有组 block_size 相同："
                    f"Runner={self.block_size}，group={spec}。不同块大小的 manager 已在控制面"
                    f"支持，但 BlockTables/attention metadata 的逐组 kernel 编译属下一阶段，"
                    f"不能在这里静默按同一块大小跑")
        self.kv_cache_config = kv_cache_config

        group_block_sizes = kv_cache_config.block_sizes
        max_num_blocks_per_group = [
            -(-self.max_model_len // size) for size in group_block_sizes]
        self.block_tables = BlockTables(
            block_sizes=list(group_block_sizes),
            max_num_reqs=self.max_num_reqs,
            max_num_batched_tokens=self.max_num_tokens,
            max_num_blocks_per_group=max_num_blocks_per_group,
            device=self.device,
            kernel_block_sizes=list(group_block_sizes),
        )
        dtype = next(self.model.parameters()).dtype
        target_layers = self._attention_layers()
        self.kv_caches = {}
        self._attn_metadata_layer_groups = {}
        from ..kv_cache_alloc import (allocate_packed_target_caches,
                                      allocate_per_layer_target_caches)

        if kv_cache_config.num_kv_cache_groups > 1:
            self.kv_caches, self.packed_backing, self.packed_block_stride = (
                allocate_packed_target_caches(
                    kv_cache_config, target_layers, self.device))
        else:
            self.packed_backing = None
            self.packed_block_stride = 0
            self.kv_caches = allocate_per_layer_target_caches(
                kv_cache_config, target_layers, self.device)
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in group.layer_names:
                layer = target_layers[layer_name]
                layer.set_output_buffer(
                    self.max_num_tokens, dtype=dtype, device=self.device)
                self._attn_metadata_layer_groups[layer_name] = group_id

        # Q-only 的 target 层（YOCO 后半段）：只绑源层 cache，不分配第二份。
        for layer_name, layer in target_layers.items():
            if not layer.q_only:
                continue
            source = layer.kv_sharing_target_layer_name
            if source is None or source not in self.kv_caches:
                raise KeyError(
                    f"target Q-only 层 {layer_name!r} 的源层 {source!r} 没有物理 cache"
                    f"（可用源层：{sorted(self.kv_caches)}）")
            source_layer = target_layers[source]
            layer.bind_shared_kv_cache(
                source_layer, self.kv_caches[source], draft_dtype=dtype)
            layer.kv_cache_group_id = source_layer.kv_cache_group_id
            layer.set_output_buffer(
                self.max_num_tokens, dtype=dtype, device=self.device)
            self._attn_metadata_layer_groups[layer_name] = source_layer.kv_cache_group_id

        if self.speculator is not None:
            self._allocate_draft_kv_caches(kv_cache_config)
            # 多 group 的草稿注意力组需要知道 target 组的 spec（滑窗/形状）；
            # EAGLE/MTP 单组路径不用它，但统一挂上是无副作用的。
            self.speculator.kv_cache_config = kv_cache_config
            # set_attn 必须在草稿层完成 group 绑定之后：它按 `kv_cache_group_id`
            # 建草稿的注意力组。物理 cache 已就绪，元数据构造器只是引用块表。
            if isinstance(self.speculator, DraftModelSpeculator):
                self.speculator.set_attn(
                    self.block_tables, self.input_buffers,
                    self._build_capture_attn_metadata)
        # 74 关：图模式必须在 `init_cudagraph_manager()` **之前**定下来（上游同序）。
        self.initialize_cudagraph_capture()
        if self.speculator is not None:
            self.speculator.init_cudagraph_manager(self.cudagraph_mode)
        self._attn_metadata_layer_names = set(self._attn_metadata_layer_groups)
        if self.speculator is not None:
            for name in self.speculator.draft_attn_layer_names:
                if name not in self._attn_metadata_layer_groups:
                    raise KeyError(
                        f"草稿层 {name!r} 没有 KV group 映射；Q-only 草稿的源层映射"
                        f"必须先于 metadata 键集合建立")
        # 79 关：能力检查 + manager（必须在 KV 缓存与注意力层名都就绪之后；但要在
        # `capture_model()` 之前，后者会用它来做成本 profiling）。
        self.adaptive_verification = self._create_adaptive_verification_manager()
        return self.kv_caches

    def _target_varlen_cudagraphs_enabled(self) -> bool:
        """是否要求上游那条“adaptive 必须 ALWAYS”的硬门（79 关）。

        **当前恒 False，但不是“target 不捕获 varlen”**：Phase 2 起 target 在 adaptive
        开启时会捕获 varlen decode FULL 图（`varlen_decode=True`），裁剪后的 1..1+K 行批
        走 `TorchAttentionImpl._forward_varlen_padded()`。

        这里仍返回 False 是为了**关闭上游的 ALWAYS 硬门**：Torch 后端能力档位仍是
        `UNIFORM_BATCH`，target varlen 图是只服务 decode 的窄路径，不满足上游“任意批都能
        进图”的 ALWAYS 语义。`maybe_create_adaptive_verification_manager` 因此只检查
        “后端不依赖 CPU query 长度做规划”，不要求 ALWAYS。

        要严格对齐上游：需要后端声明 ALWAYS（FlashAttention/Triton 那类）并把这里改成
        True；否则不能声称完整支持。
        """
        return (self.attn_builder_cls.get_cudagraph_support(self.vllm_config)
                == AttentionCGSupport.ALWAYS)

    def _create_adaptive_verification_manager(self):
        """按实际注意力后端能力建 `AdaptiveVerificationManager`（79 关）。

        上游在 `init_attn_backend()` 之后、图捕获之前建；本仓库只有一个 Torch 后端、
        一个 KV group，所以这里显式造一个 `AttentionGroup`（后端 + 层名 + 元数据构造器）
        喂给同一个 `maybe_create_adaptive_verification_manager()`。
        """
        from ..gpu.attn_utils import AttentionCGSupportInfo
        from ..gpu.spec_decode.adaptive_verification import (
            maybe_create_adaptive_verification_manager,
        )

        enable = bool(getattr(self.speculator, "enable_adaptive_verification", False))
        builder_cls = self.attn_builder_cls
        group = AttentionGroup(
            backend=self.attention_backend,
            layer_names=sorted(self._attn_metadata_layer_names),
            metadata_builders=[builder_cls(self.block_size)],
        )
        attn_cg_support = AttentionCGSupportInfo(
            builder_cls.get_cudagraph_support(self.vllm_config),
            self.attention_backend.get_name(),
        )
        return maybe_create_adaptive_verification_manager(
            enable_adaptive_verification=enable,
            attn_groups=[[group]],
            attn_cg_support=attn_cg_support,
            req_states=self.req_states,
            query_start_loc=self.input_buffers.query_start_loc,
            num_bonus_tokens=self.num_new_sampled_tokens_per_step,
            max_total_logits=get_max_chunk_logits(self.vocab_size),
            # 本仓库 target 已捕获 varlen decode 图，但后端仍是 UNIFORM_BATCH；
            # 这里关闭上游的 ALWAYS 硬门（见上一个方法的说明）。
            require_varlen_cudagraph=self._target_varlen_cudagraphs_enabled(),
        )

    def initialize_cudagraph_capture(self) -> None:
        """定下 V2 的图模式（对应 V1 `GPUModelRunner.initialize_cudagraph_capture()` 的前两步）。

        顺序与上游 V2 一致：**注意力后端先开口**（能力档位决定 FULL 能不能成立），再决定最终
        模式，最后把这个模式交给草稿的两张图管理器。写回 `compilation_config.cudagraph_mode`，
        所以"实际用了什么模式"在配置对象上可查，不是只在日志里。

        本仓库已接 Phase 3 的 breakable PIECEWISE：混合 prefill/decode 批走 PIECEWISE，
        统一 decode 批走 FULL；但 `execute_model()` 对**纯 prefill 批**显式回退 NONE
        （`is_prefilling_np.all()`）。这不是 PIECEWISE 的语义限制，而是本仓库的捕获时机
        取舍：PIECEWISE 惰性捕获，若在启动后的第一轮纯 prefill 就录 breakable 段，会和已录
        的 FULL 图/本机 pinned 分配器互相干扰；因此把首次 capture 推迟到真正的混合批。
        纯 prefill 的 eager 路径正确性不变，只是没有图；要放开需补纯 prefill 捕获与测试。
        """
        from ...config import CUDAGraphMode

        builder_cls = self.attn_builder_cls
        mode = self.vllm_config.compilation_config.cudagraph_mode
        if mode != CUDAGraphMode.NONE and not torch.cuda.is_available():
            # 环境事实：配置写着 cuda 不等于这台机器有 CUDA。真去捕获只会炸，不如在这里退 eager。
            mode = CUDAGraphMode.NONE
            self.vllm_config.compilation_config.cudagraph_mode = mode
            self.vllm_config.compilation_config.cudagraph_capture_sizes = []
            self.vllm_config.compilation_config.max_cudagraph_capture_size = 0
        if mode != CUDAGraphMode.NONE:
            min_support = builder_cls.get_cudagraph_support(self.vllm_config)
            mode = self.vllm_config.resolve_cudagraph_mode_and_sizes(
                min_support, builder_cls.__name__, self.decode_query_len)
        self.cudagraph_mode = mode
        # target 图管理器：Phase 1 FULL_DECODE_ONLY；Phase 2 varlen；Phase 3
        # FULL_AND_PIECEWISE（混合批走 breakable PIECEWISE）。
        self.target_cudagraph_manager = None
        if mode.has_full_cudagraphs():
            # Phase 2：adaptive 开启时 target 用 varlen decode 图（每请求 1..1+K 行）；
            # 否则仍是统一 decode 图（每请求 1+K 行）。
            if mode == CUDAGraphMode.FULL:
                # ALWAYS 后端（Triton paged）：混合批也录整张 FULL 图。
                target_mode = CUDAGraphMode.FULL
            elif mode.has_piecewise_cudagraphs():
                target_mode = CUDAGraphMode.FULL_AND_PIECEWISE
            else:
                target_mode = CUDAGraphMode.FULL_DECODE_ONLY
            self.target_cudagraph_manager = ModelCudaGraphManager(
                self.vllm_config, self.device,
                target_mode,
                self.decode_query_len,
                allow_shape_padding=True,
                varlen_decode=bool(getattr(
                    self.speculator, "enable_adaptive_verification", False)),
            )

    def _draft_attention_layer_names(self) -> set[str]:
        """草稿模型里所有 `Attention` 层的模块路径（= metadata 的键）。"""
        from ...attention.layer import Attention

        # Gemma4 assistant 与 target 都从 `model.layers.0...` 起名，且 Gemma4 有多个
        # KV group / target 自身 Q-only 层。如果直接拿模块路径当 metadata 键，assistant
        # 的层会读到 target 同号层的组（这一层 source 可能不是它）。只对 Gemma4 加别名，
        # 不改变其它草稿（EAGLE/MTP）的既有键名。
        needs_alias = getattr(
            getattr(self.speculator.model, "model", None),
            "backbone_hidden_size", None) is not None
        names: set[str] = set()
        for name, module in self.speculator.model.named_modules():
            if not isinstance(module, Attention):
                continue
            alias = module.layer_name
            if needs_alias and not alias.startswith("draft::"):
                alias = "draft::" + name
                module.layer_name = alias
            names.add(alias)
        return names

    def _allocate_draft_kv_caches(self, kv_cache_config) -> None:
        from ...attention.layer import Attention

        if not hasattr(self, "_attn_metadata_layer_groups"):
            self._attn_metadata_layer_groups = {}
        draft_model = self.speculator.model
        # **必须取浮点参数**：缩小词表的草稿（EAGLE3 / DFlash / DSpark）把 `d2t` 注册成
        # int64 的 `nn.Parameter`，而它（顶层参数）在 `named_parameters()` 里排在子模块**之前**
        # ——`next(parameters()).dtype` 会拿到 int64，KV 缓存就变成整型，写 K/V 时当场
        # `index_copy_` 的 dtype 不匹配（77 关做 DSpark 缩减词表时踩到；DFlash 同款）。
        draft_dtype = next(p.dtype for p in draft_model.parameters() if p.is_floating_point())
        # 80D：Q-only 层直接绑 target 源层的物理 cache。先把 target 的层对象按层名找到；
        # 只按名字从 `kv_caches` 里拿 cache 不够——兼容性检查还要读源层的 KV heads/head_size。
        target_layers = self._attention_layers()
        names: set[str] = set()
        for name, module in draft_model.named_modules():
            if not isinstance(module, Attention):
                continue
            shared_name = module.kv_sharing_target_layer_name
            if shared_name is not None:
                if shared_name not in self.kv_caches:
                    raise KeyError(
                        f"Q-only 草稿层 {name!r} 要共享 target 层 {shared_name!r} 的 KV，"
                        f"但 target 没有这个 cache（可用层：{sorted(self.kv_caches)}）"
                    )
                target_layer = target_layers.get(shared_name)
                if target_layer is None:
                    raise KeyError(
                        f"Q-only 草稿层 {name!r} 的源层 {shared_name!r} 不是 target 的 "
                        f"Attention 层"
                    )
                cache = self.kv_caches[shared_name]
                if (cache.shape[0] != 2
                        or cache.shape[1] != kv_cache_config.num_gpu_blocks
                        or cache.shape[2] != self.block_size):
                    raise ValueError(
                        f"Q-only 源层 {shared_name!r} 的 KV cache 布局与本轮不一致："
                        f"cache={tuple(cache.shape)}，期望 "
                        f"[2, {kv_cache_config.num_gpu_blocks}, {self.block_size}, "
                        f"kv_heads, head_size]"
                    )
                module.bind_shared_kv_cache(
                    target_layer, cache, draft_dtype=draft_dtype)
                module.kv_cache_group_id = target_layer.kv_cache_group_id
                self._attn_metadata_layer_groups[module.layer_name] = (
                    target_layer.kv_cache_group_id)
                names.add(module.layer_name)
                continue
            if module.q_only:
                raise NotImplementedError(
                    f"草稿层 {name!r} 声明为 Q-only，但没有 kv_sharing_target_layer_name。"
                    f"Q-only 共享映射未接线时不能给它分配独立 KV：那会静默读错上下文。"
                    f"先由 speculator 的映射建立源层，再初始化 KV cache。"
                )
            # 非 Q-only 草稿层：按 (head_size, kv_heads, sliding_window) 匹配 target
            # 的 KV group。Step3p5 的 MTP 层可能落在 sliding group 而不是 group0；
            # 不能像单组 Qwen 那样直接写死 0。
            key = (module.head_size, module.num_kv_heads,
                   module.per_layer_sliding_window)
            matches = [
                gid for gid, group in enumerate(kv_cache_config.kv_cache_groups)
                if (group.kv_cache_spec.head_size == key[0]
                    and group.kv_cache_spec.num_kv_heads == key[1]
                    and getattr(group.kv_cache_spec, "sliding_window", None) == key[2])
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"非 Q-only 草稿层 {name!r} 的 KV 形状 {key} 在 target 的 "
                    f"{len(kv_cache_config.kv_cache_groups)} 个 group 中匹配到 "
                    f"{matches}：必须恰好匹配一个 group（否则 metadata 会用错块表）")
            gid = matches[0]
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
            module.kv_cache_group_id = gid
            self._attn_metadata_layer_groups[module.layer_name] = gid
            names.add(module.layer_name)
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

    def capture_model(self) -> dict:
        """捕获草稿的两张图（对应上游 V2 `GPUModelRunner.capture_model()` 的提议者那一半）。

        上游的顺序是"先捕获 target 的图，再在草稿 workspace lane 里捕获提议者的图"。本仓库
        V2 的 **target 图**不属 74 关（`CudaGraphManager` 只为提议者建），所以这里只有第二步。

        捕获窗口由 `monitor.set_cudagraph_capturing_enabled()` 划定：窗口外任何"顺手捕获"
        都会被 `validate_cudagraph_capturing_enabled()` 拦下（那会让服务中的某一步突然卡顿）。
        没有请求时也要能跑——捕获用的输入全是 `InputBatch.make_dummy()` 造的假输入，
        写的是**常驻缓冲**（图里记的是指针），所以捕获完这些缓冲会被真实的下一轮覆盖。
        """
        import time

        from ...compilation import graph_capture, set_cudagraph_capturing_enabled
        from ...config import CUDAGraphMode

        mode = getattr(self, "cudagraph_mode", CUDAGraphMode.NONE)
        stats = {"mode": str(mode), "captured": 0, "capture_seconds": 0.0,
                 "graphs": {}}
        if self.speculator is None and self.target_cudagraph_manager is None:
            return stats
        if mode == CUDAGraphMode.NONE:
            # 79 关：即使不捕获任何图，自适应验证也需要成本表；profiling 仍然要跑。
            if self.adaptive_verification is not None:
                stats["adaptive_verification"] = self.profile_adaptive_verification()
            return stats
        start = time.perf_counter()
        torch.cuda.synchronize()
        start_free = torch.cuda.mem_get_info()[0]
        set_cudagraph_capturing_enabled(True)
        try:
            # 上游顺序：先 target，再 speculator。target 图要先把 hidden/aux 写成静态缓冲，
            # 后面草稿的 propose 才从同一批缓冲读（与 eager 路径同源）。
            if self.target_cudagraph_manager is not None:
                self.target_cudagraph_manager.capture(
                    self.model,
                    self.input_buffers,
                    self.block_tables,
                    self._build_target_capture_attn_metadata,
                    use_aux_hidden_state_outputs=self.use_aux_hidden_state_outputs,
                )
            if self.speculator is not None:
                with graph_capture(self.device):
                    self.speculator.capture()
        finally:
            # 无论捕获成功与否都要关掉窗口（69 关的理由：开着的话后面每一次"没见过的形状"
            # 都会顺手录一张图：延迟尖峰 + 显存悄悄涨）。
            set_cudagraph_capturing_enabled(False)
        torch.cuda.synchronize()
        end_free = torch.cuda.mem_get_info()[0]
        if self.target_cudagraph_manager is not None:
            stats["target"] = {
                "captured": len(self.target_cudagraph_manager.graphs),
                "token_counts": self.target_cudagraph_manager.captured_token_counts(),
            }
            stats["captured"] += len(self.target_cudagraph_manager.graphs)
        if self.speculator is not None:
            managers = getattr(self.speculator, "cudagraph_managers", None)
            if managers is None:
                managers = {"prefill": getattr(self.speculator, "prefill_cudagraph_manager", None),
                            "decode": getattr(self.speculator, "decode_cudagraph_manager", None)}
            for name, manager in managers.items():
                if manager is not None:
                    stats["graphs"][name] = {
                        "captured": len(manager.graphs),
                        "token_counts": manager.captured_token_counts(),
                    }
                    stats["captured"] += len(manager.graphs)
        stats["capture_seconds"] = time.perf_counter() - start
        stats["graph_pool_bytes"] = start_free - end_free
        stats["fused_multi_step_decode"] = bool(
            getattr(self.speculator, "use_fused_multi_step_decode", False))
        if self.adaptive_verification is not None:
            stats["adaptive_verification"] = self.profile_adaptive_verification()
        self.cudagraph_capture_stats = stats
        return stats

    def _build_target_capture_attn_metadata(self, input_batch: InputBatch):
        """给 target 捕获用的 metadata：块表/槽位都是 `prepare_inputs_to_capture()` 刚写的
        常驻 dummy 视图（行数 = 捕获请求数），不能再新建一份。"""
        block_tables = tuple(
            bt[: input_batch.num_reqs]
            for bt in self.block_tables.input_block_tables
        )
        return self._build_attn_metadata(
            input_batch, block_tables, self.block_tables.slot_mappings)

    # -------- 79 关：自适应验证的成本 profiling --------

    def _adaptive_profile_sizes(self) -> list[int]:
        """成本 profiling 的 target token 数网格。

        取 `decode_query_len × {1,2,4,…}`，保证 dummy 批是"每请求 1+K 行"的整齐形状；
        末尾会由 `batches_to_profile()` 自动补捕获上限外的 1.5×/2×… 尾巴。
        """
        sizes: set[int] = set()
        for reqs in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            if reqs > self.max_num_reqs:
                break
            sizes.add(min(self.max_num_tokens, reqs * self.decode_query_len))
        sizes.add(min(self.max_num_tokens, self.max_num_reqs * self.decode_query_len))
        return sorted(s for s in sizes if s > 0)

    @torch.inference_mode()
    def _dummy_run_adaptive(self, num_tokens: int, context_len: int = 0) -> None:
        """跑一个 dummy 前向/图重放 + 草稿提议，供 `StepTimingCollector` 记录两段 GPU 时间。

        Phase 2：target 已有 varlen decode 图时，这里走 `run_fullgraph()`；没有对应图的
        尾部尺寸仍走 eager（上游 `batches_to_profile` 也会为捕获上限之外补尺寸）。
        """
        from ...forward_context import BatchDescriptor, set_forward_context

        num_tokens = max(1, min(int(num_tokens), self.max_num_tokens))
        num_reqs = max(1, num_tokens // self.decode_query_len)
        num_reqs = min(self.max_num_reqs, num_tokens, num_reqs)

        batch_desc = None
        if self.target_cudagraph_manager is not None:
            batch_desc = self.target_cudagraph_manager.dispatch(
                num_reqs, num_tokens, None,
                max_query_len=self.decode_query_len)
            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                num_tokens = batch_desc.num_tokens
                num_reqs = batch_desc.num_reqs or num_reqs

        context_len = max(0, min(int(context_len), self.max_model_len - 1))
        max_per_req = max(1, self.max_model_len - context_len)
        num_tokens = min(num_tokens, num_reqs * max_per_req)
        num_reqs = min(num_reqs, num_tokens)
        input_batch = InputBatch.make_dummy(
            num_reqs, num_tokens, self.input_buffers,
            max_query_len=(batch_desc.max_query_len if batch_desc is not None else None),
        )
        if batch_desc is not None and batch_desc.uniform_token_count is None:
            input_batch.uniform_query_len = None
        if context_len:
            input_batch.seq_lens.add_(context_len)
            input_batch.seq_lens_cpu_upper_bound.add_(context_len)
            qsl = torch.as_tensor(input_batch.query_start_loc_np, dtype=torch.int64)
            lengths = qsl[1:] - qsl[:-1]
            starts = torch.repeat_interleave(qsl[:-1], lengths)
            self.input_buffers.positions[:num_tokens].copy_(
                context_len + torch.arange(num_tokens, dtype=torch.int64) - starts
            )

        is_full = (batch_desc is not None
                   and batch_desc.cg_mode == CUDAGraphMode.FULL)
        if is_full:
            # 图重放不读 Python 侧 metadata；但草稿 propose 要一份与 input_batch 同源的
            # slot_mapping/metadata，所以仍按 dummy 缓冲构造一份。
            self.block_tables.get_dummy_block_tables(num_reqs)
            slot_mappings = self.block_tables.get_dummy_slot_mappings(num_tokens)
            block_tables = tuple(
                bt[:num_reqs] for bt in self.block_tables.input_block_tables)
            attn_metadata = self._build_attn_metadata(
                input_batch, block_tables, slot_mappings)
        else:
            block_tables, slot_mappings = self.prepare_attn(input_batch)
            attn_metadata = self._build_attn_metadata(
                input_batch, block_tables, slot_mappings)
        slot_mappings_by_layer = {"slot_mapping": slot_mappings}

        self.step_timing.record_batch(input_batch, is_full)
        self.step_timing.forward_start()
        if is_full:
            assert self.target_cudagraph_manager is not None
            model_output = self.target_cudagraph_manager.run_fullgraph(batch_desc)
        else:
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
        self.step_timing.forward_end()
        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = model_output
        else:
            hidden_states, aux_hidden_states = model_output, None

        assert self.sampler is not None and self.speculator is not None
        num_sampled = torch.ones(
            input_batch.num_reqs, dtype=torch.int32, device=self.device)
        num_rejected = torch.zeros_like(num_sampled)
        self.step_timing.drafter_start()
        self.speculator.propose(
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
            dummy_run=True,
        )
        self.step_timing.drafter_end()

    def profile_adaptive_verification(self) -> dict:
        """实测 draft/target 两段成本并安装成本表（79 关）。

        这是**真实 GPU event 测量**，不是常数表：`batches_to_profile()` 给出尺寸与重复次数，
        `_dummy_run_adaptive()` 跑真实模型前向与真实草稿提议，`set_initial_cost_curves()`
        按中位数建表。dummy 输入与图捕获同源（常驻缓冲），所以量到的不是另一套代码路径。
        """
        manager = self.adaptive_verification
        assert manager is not None
        if manager.cost_tables is not None:
            return {
                "cost_curves": manager.cost_curves,
                "num_samples": 0,
                "cached": True,
            }
        target_manager = self.target_cudagraph_manager
        captured = (target_manager.captured_token_counts()
                    if target_manager is not None else [])
        sizes = captured or self._adaptive_profile_sizes()
        with self.step_timing.collect() as timings:
            for batch in manager.batches_to_profile(sizes):
                self._dummy_run_adaptive(**batch)
        # Phase 2：target 有 FULL 图时，捕获上限内的成本是图阶梯；上限外仍 eager。
        # 没有 target 图（例如只有草稿图）时才把 limit 置 0。
        manager._cudagraph_limit = max(captured) if captured else 0
        manager.set_initial_cost_curves(timings)
        return {
            "cost_curves": manager.cost_curves,
            "num_samples": len(timings),
            "cudagraph_limit": manager._cudagraph_limit,
            "profile_sizes": sizes,
            "cached": False,
        }

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

            if self.adaptive_verification is not None:
                # 79 关：slot 复用 / 新请求必须把两代置信度重置成 1.0，否则第一步会拿
                # 上一个请求的 stale 置信度选预算（不报错，只是预算偏大/偏小）。
                self.adaptive_verification.add_request(req_index)

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
        num_toks = scheduler_output.total_num_scheduled_tokens
        if self.adaptive_verification is not None and draft_tokens:
            # 79 关：CPU 先按 stale 置信度选总预算；`num_tokens` 是**裁剪后**的 target
            # 行数（可能小于 scheduler 排的 1+K）。真实每请求长度在 GPU 上再定。
            num_toks = self.adaptive_verification.get_num_tokens(
                num_tokens_per_req, draft_tokens
            )

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
        self, scheduler_output, batch_req_state: BatchReqState,
        batch_desc: BatchExecutionDescriptor | None = None,
    ) -> InputBatch:
        num_tokens = batch_req_state.num_tokens
        num_tokens_after_padding = (
            batch_desc.num_tokens if batch_desc is not None else num_tokens)
        assert num_tokens > 0

        req_ids = batch_req_state.req_ids
        num_scheduled_tokens_np = batch_req_state.num_scheduled_tokens
        idx_mapping_np = batch_req_state.idx_mapping_np
        idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)
        num_reqs = len(req_ids)
        # Phase 1：target FULL 图要求把批次补齐到捕获档位；eager 路径不补。
        num_reqs_padded = (
            batch_desc.num_reqs
            if batch_desc is not None and batch_desc.num_reqs is not None
            else num_reqs
        )

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

        # 79 关：CPU 侧先把草稿预算压到选定值；GPU 侧稍后用**当前**置信度重新分配。
        # `num_scheduled_tokens` 在这里分成两份：`num_scheduled_tokens_np` 是压紧后的布局
        # （用于 query_start_loc），`num_scheduled_tokens_upper_bound` 是调度器原计划
        # （用于 seq_lens 上界与 InputBatch 的对外口径）。
        adaptive_verification = (
            self.adaptive_verification if num_draft_tokens_per_req is not None else None
        )
        cu_num_logits_np_exact = True
        num_scheduled_tokens_upper_bound = num_scheduled_tokens_np
        if adaptive_verification is not None:
            num_scheduled_tokens_np, cu_num_logits_np = (
                adaptive_verification.compact_batch(
                    num_draft_tokens_per_req,
                    num_scheduled_tokens_np,
                    cu_num_logits_np,
                )
            )

        # Get query_start_loc.
        query_start_loc_np = np.empty(self.max_num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens_np, out=query_start_loc_np[1 : num_reqs + 1])
        # Pad for full CUDA graph mode.
        # Some attention backends like FA3 require query_start_loc to be non-decreasing.
        # Padded requests have zero query length: their start/end both point at the
        # real total, so graph attention sees q_len=0 and never writes/reads their KV.
        query_start_loc_np[num_reqs + 1 :] = num_tokens
        query_start_loc = self.input_buffers.query_start_loc
        async_copy_to_gpu(query_start_loc_np, out=query_start_loc)
        if adaptive_verification is not None:
            # GPU 用当前置信度做全局 top-k，写回真实每请求前缀长度与 offsets。
            # 返回的 `query_start_loc` 是常驻缓冲（长度 max_num_reqs+1）。
            cu_num_logits, query_start_loc, total_num_draft_tokens = (
                adaptive_verification.reallocate_drafts(req_ids, idx_mapping)
            )
            total_num_logits = num_reqs * num_bonus_tokens + total_num_draft_tokens
        if adaptive_verification is not None:
            # 中间预算的 CPU offsets 是"均分上界"；0 / 满预算才是精确值。
            cu_num_logits_np_exact = (
                total_num_draft_tokens == 0
                or total_num_draft_tokens == int(num_draft_tokens_per_req.sum())
            )
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
        if num_tokens_after_padding > num_tokens:
            # Padding 行必须每轮显式写死：图重放会读满捕获形状，上一轮的残值可能是
            # 越界的 token/position（embedding/RoPE 直接越界）。
            self.input_buffers.input_ids[num_tokens:num_tokens_after_padding].zero_()
            self.input_buffers.positions[num_tokens:num_tokens_after_padding].zero_()
            self.input_buffers.is_padding[num_tokens:num_tokens_after_padding].fill_(True)
        self.input_buffers.is_padding[:num_tokens].fill_(False)

        # CPU upper bound on seq_lens; padded entries left at zero.
        num_computed_tokens_np = self.req_states.num_computed_tokens_np[idx_mapping_np]
        seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
        np.add(
            num_computed_tokens_np,
            num_scheduled_tokens_upper_bound,
            out=seq_lens_cpu_upper_bound_np[:num_reqs],
        )
        seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens_cpu_upper_bound_np)

        # 74 关：这一批是不是"统一 decode"（每条请求都恰好 `1 + K` 行、且没有请求在 prefill）。
        # 是的话它的形状是固定的、可以被图重放——草稿 prefill 图正是吃这个形状；不是就回退
        # eager（`uniform_query_len=None` → 注意力走逐请求路径）。
        # 79 关：自适应验证裁剪后每请求行数可以不同。Phase 2 起 target 有 varlen FULL 图：
        # dispatch 会给出 `max_query_len=1+K`、`uniform_token_count=None` 的描述符；
        # 这里把描述符的上界写进 metadata，注意力走 `_forward_varlen_padded()`。
        graph_max_query_len = None
        if batch_desc is not None and batch_desc.cg_mode == CUDAGraphMode.FULL:
            # Target FULL 图的形状由描述符定死；注意力必须走固定形状路径，不能用
            # “按真实批重新推”的值。uniform 图用 uniform_query_len；varlen 图用 max_query_len。
            uniform_query_len = batch_desc.uniform_token_count
            graph_max_query_len = (
                batch_desc.max_query_len or batch_desc.uniform_token_count)
        else:
            uniform_query_len = (
                None
                if adaptive_verification is not None
                else get_uniform_decode_token_count(
                    num_reqs,
                    num_tokens,
                    int(num_scheduled_tokens_np.max()),
                    bool(batch_req_state.has_prefill),
                )
            )

        return InputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs_padded,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens_upper_bound,
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
            cu_num_logits_np_exact=cu_num_logits_np_exact,
            max_query_len=graph_max_query_len,
            uniform_query_len=uniform_query_len,
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
        if input_batch.num_tokens_after_padding > input_batch.num_tokens:
            # `compute_slot_mappings` 只按**真实请求数**启动；捕获形状多出来的行必须
            # 显式写哨兵，否则图会按上一轮的残值写 KV。
            slot_mappings[:, input_batch.num_tokens:].fill_(PAD_SLOT_ID)
        return block_tables, slot_mappings

    def _build_attn_metadata(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
    ) -> dict[str, Any]:
        """把 V2 的批次视图翻成注意力层要的元数据（本项目只有一个后端，一份 metadata）。

        上游这里是 `model_state.prepare_attn(...)`，按 `attn_groups` 分组构造**每组的后端
        metadata**；本仓库只有一个 Torch 后端、一个 KV group，所以"一组"就是全部层
        （`_attn_metadata_layer_names` = target 层 ∪ 草稿层）。

        `uniform_query_len` 来自 `InputBatch`（74 关）：统一 decode 批（每请求 `1 + K` 行）里
        它是 `1 + K`，其它批是 None。**它决定图内注意力的形状**——草稿 prefill 图重放时正是
        靠 target 这份元数据里的这个 Python int 走 `_forward_padded()`。
        """
        builder = self.attn_metadata_builder
        group_metadata: dict[int, Any] = {}
        for group_id, group in enumerate(
                getattr(self.kv_cache_config, "kv_cache_groups", [])):
            spec = group.kv_cache_spec
            group_metadata[group_id] = builder.build(
                query_start_loc=input_batch.query_start_loc,
                seq_lens=input_batch.seq_lens,
                block_table=block_tables[group_id],
                slot_mapping=slot_mappings[group_id][
                    : input_batch.num_tokens_after_padding],
                num_reqs=input_batch.num_reqs_after_padding,
                uniform_query_len=input_batch.uniform_query_len,
                max_query_len=input_batch.max_query_len,
                sliding_window=getattr(spec, "sliding_window", None),
            )
        if not group_metadata:
            raise RuntimeError("KV cache 还没有初始化：先 initialize_kv_cache()")
        result = {}
        for name in self._attn_metadata_layer_names:
            group_id = self._attn_metadata_layer_groups.get(name)
            if group_id is None:
                # 兼容旧路径：没有显式映射时，单组模型的全部层都属于 group 0。
                if len(group_metadata) != 1:
                    raise KeyError(
                        f"注意力层 {name!r} 没有 KV group 映射，但模型有 "
                        f"{len(group_metadata)} 个 group：不能默认读到第 0 组")
                group_id = 0
            result[name] = group_metadata[group_id]
        return result

    def _build_capture_attn_metadata(self, input_batch: InputBatch) -> dict[str, Any]:
        """74 关：捕获草稿 prefill 图时造 target 那份"假"元数据（下游是
        `SpeculatorCudaGraphManager.create_forward_fn`）。

        `prepare_inputs_to_capture()` 已经把假输入写进了**常驻缓冲**（块表清零、slot_mapping
        全哨兵、`query_start_loc`/`seq_lens` 按 `num_tokens` 摊开），所以这里只按假批的请求数
        切块表（与运行期 `prepare_attn()` 同一条切片口径），再用**同一个**
        `_build_attn_metadata()` 造元数据——不同口径重建一份会让图里记的指针与运行期不一致，
        那是"重放读到旧数据、且不报错"的经典来源。

        与 V1 的差别（`docs/step74_alignment.md` 差异账本 D-74-5）：这里**不调**
        `build_for_cudagraph_capture()`（它把 `seq_lens` 全填 1）。理由是热身的 eager 那一遍
        会被填 1 弄出"整行掩掉 → softmax NaN"，而 NaN 会写进 KV 与特征缓冲、污染后续所有重放；
        且"填 1 更快"对本后端的图内路径不成立（它按块表宽度 gather 整段上下文）。
        形状守门改成显式断言 `uniform_query_len` 已定死。
        """
        block_tables = tuple(
            bt[: input_batch.num_reqs_after_padding]
            for bt in self.block_tables.input_block_tables
        )
        metadata = self._build_attn_metadata(
            input_batch, block_tables, self.block_tables.slot_mappings
        )
        for layer_metadata in metadata.values():
            if layer_metadata.uniform_query_len is None:
                raise ValueError(
                    "捕获草稿 prefill 图必须是统一 decode 形状（每请求 1+K 行）："
                    f"收到 uniform_query_len=None，num_reqs={input_batch.num_reqs}、"
                    f"num_tokens={input_batch.num_tokens}")
        return metadata

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
        # Phase 1：统一 decode（每请求 1+K 行）才可能命中 target FULL 图；
        # 混合 prefill/decode 批由 dispatch 回退成 NONE（eager）。
        num_reqs = len(batch_req_state.req_ids)
        max_query_len = int(batch_req_state.num_scheduled_tokens.max())
        uniform_tok_count = get_uniform_decode_token_count(
            num_reqs,
            batch_req_state.num_tokens,
            max_query_len,
            batch_req_state.has_prefill,
        )
        batch_desc = None
        if self.target_cudagraph_manager is not None:
            batch_desc = self.target_cudagraph_manager.dispatch(
                num_reqs,
                batch_req_state.num_tokens,
                uniform_tok_count,
                max_query_len=max_query_len,
            )
            # varlen/uniform FULL 只服务 decode；短 prefill 的 query_len 也可能 <=1+K，
            # 不能误命中。混合批要等 dispatch 给出 PIECEWISE 描述符。
            if (batch_req_state.has_prefill
                    and batch_desc.cg_mode == CUDAGraphMode.FULL
                    and (batch_desc.uniform_token_count is not None
                         or batch_desc.max_query_len is not None)):
                # decode FULL 图（uniform / varlen）不能吃 prefill；混合 FULL 图的
                # 两个字段都为 None，交给 ALWAYS 后端。
                batch_desc = None
            # 纯 prefill 也暂不碰 breakable：Phase 3 的惰性捕获只在**混合批**
            # （至少一条 prefill + 至少一条 decode）上触发，避免纯 prefill 首次
            # 捕获与已录好的 FULL 图共享图池时互相干扰。
            if (batch_desc is not None
                    and batch_desc.cg_mode == CUDAGraphMode.PIECEWISE
                    and batch_req_state.is_prefilling_np.all()):
                batch_desc = None
            if batch_desc is not None:
                self.target_cudagraph_manager.record_selection(
                    batch_desc, num_reqs, batch_req_state.num_tokens)
        input_batch = self.prepare_inputs(
            scheduler_output, batch_req_state, batch_desc)
        block_tables, slot_mappings = self.prepare_attn(input_batch)
        slot_mappings_by_layer = {"slot_mapping": slot_mappings}
        attn_metadata = self._build_attn_metadata(
            input_batch, block_tables, slot_mappings
        )

        from ...forward_context import BatchDescriptor, set_forward_context

        try:
            if batch_desc is not None and batch_desc.cg_mode == CUDAGraphMode.FULL:
                # 图重放：输入已经写进常驻缓冲；不能再调 model()，否则会跑 eager。
                assert self.target_cudagraph_manager is not None
                model_output = self.target_cudagraph_manager.run_fullgraph(batch_desc)
            elif (batch_desc is not None
                  and batch_desc.cg_mode == CUDAGraphMode.PIECEWISE):
                assert self.target_cudagraph_manager is not None
                # breakable PIECEWISE：graph 段由 wrapper 重放；attention 在 eager 段里
                # 读**当前** forward context 的 metadata。
                pw_context = dict(
                    attn_metadata=attn_metadata,
                    num_tokens=input_batch.num_tokens_after_padding,
                    cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                    batch_descriptor=BatchDescriptor(
                        num_tokens=batch_desc.num_tokens,
                        num_reqs=None,
                    ),
                    slot_mapping=slot_mappings_by_layer,
                )
                wrapper = self.target_cudagraph_manager.pw_model
                assert wrapper is not None
                # 惰性捕获：先按**这一轮真实输入**录段，再重放。捕获点必须在 model()
                # 之外，不能藏在 wrapper.__call__ 里（否则 capture 的 eager attention
                # 与后续 replay 共用同一份输入，容易读到半成品）。
                if wrapper.entries.get(pw_context["batch_descriptor"]) is None:
                    with set_forward_context(**pw_context):
                        self.target_cudagraph_manager.capture_pw_graph(
                            input_batch.input_ids, input_batch.positions)
                with set_forward_context(**pw_context):
                    model_output = self.target_cudagraph_manager.run_pw_graph(
                        input_batch.input_ids, input_batch.positions)
            else:
                with set_forward_context(
                    attn_metadata,
                    num_tokens=input_batch.num_tokens_after_padding,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    batch_descriptor=BatchDescriptor(
                        num_tokens=input_batch.num_tokens_after_padding,
                        num_reqs=input_batch.num_reqs,
                        uniform=input_batch.uniform_query_len is not None,
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
            if self.adaptive_verification is not None:
                # 79 关：本轮**新提议**的置信度发布给下一步（CPU 用 stale、GPU 用当前）。
                self.adaptive_verification.record_confidences(
                    self.speculator.draft_token_confidence_probs, input_batch
                )

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
