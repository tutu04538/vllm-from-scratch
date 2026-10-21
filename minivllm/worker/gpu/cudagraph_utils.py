"""V2 的 CUDA Graph 管理器（对应 vLLM `v1/worker/gpu/cudagraph_utils.py` 的核心子集）。

与 69 关的 V1 图机制**不是同一套东西**，别混：

    V1（`compilation/cuda_graph.py::CUDAGraphWrapper`）
        包装一个可调用对象，**从 forward 上下文里读**运行模式与图键，第一次遇到就捕获。
        它不持有输入缓冲——"喂同一批地址"是调用方的责任。
    V2（本文件 `CudaGraphManager`）
        **手工**管理器：捕获期由 Runner 主动调 `capture()`（每个键一次），运行期由 Runner
        主动调 `dispatch()` 挑键、`run_fullgraph()` 重放。图在 `self.graphs` 里，键是
        `BatchExecutionDescriptor`。上游 V2 就是这么做的——因为 V2 的输入组装是"常驻缓冲 +
        内核现写"，不需要（也不该）让包装器去猜形状。

四件事，逐个对应上游同名成员：

    _init_candidates()   从档位表列出"每个 token 数该用哪张图"（含**向上取整**规则）
    capture()            逐张捕获（PIECEWISE 先、FULL 后；每张先热身一次）
    dispatch()           运行时把真实形状翻成图键；没有匹配的图 → 回退 `NONE`（不猜）
    run_fullgraph()      重放；**输入必须已经写进常驻缓冲**（这是本类的核心不变量）

裁剪项（本仓库没有对应能力，逐条记在 `docs/step74_alignment.md`）：LoRA（捕获档位不按
LoRA 数展开）、breakable CUDA graph（没有 torch.compile）、varlen decode（属 79 关的自适应
验证）、DP 同步（单卡）、offloader（没有权重/激活卸载）、MTP 多模块。**没有"简化算法"**：
候选表、取整、捕获顺序、重放契约都与上游逐行同义。
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from typing import Any, NamedTuple, Protocol

import torch

from ...compilation import graph_capture
from ...config import CUDAGraphMode
from ...forward_context import BatchDescriptor, set_forward_context
from .block_table import BlockTables
from .input_batch import InputBatch, InputBuffers


class AttentionState(NamedTuple):
    """捕获一份输入时要准备好的两样东西（上游同名 NamedTuple）。"""

    attn_metadata: dict[str, Any] | None
    slot_mappings: dict[str, torch.Tensor]


@dataclass(frozen=True)
class BatchExecutionDescriptor:
    """批次形状 + 图模式（上游同名 dataclass，逐字对齐）。

    `frozen=True` 不是为了好看：它让这个对象**可哈希**，于是可以直接当 `self.graphs` 的键。
    `cg_mode` 参与相等比较——同样的数字配 FULL 与配 PIECEWISE 是两张不同的图。

    `num_reqs=None` 是 PIECEWISE 的契约（"不需要补齐请求行"）；`uniform_token_count=None`
    表示这张图不约束每请求行数；`max_query_len` 是 varlen decode 用的上界（本仓库不用）。
    """

    cg_mode: CUDAGraphMode
    num_tokens: int
    num_reqs: int | None  # None means no request padding is needed (PIECEWISE graphs)
    uniform_token_count: int | None = None
    # Upper bound on per-request query length. Varlen decode graphs leave
    # uniform_token_count unset, so this is what keeps a prefill batch out of one.
    max_query_len: int | None = None
    num_active_loras: int = 0


class CreateForwardFn(Protocol):
    """"准备输入（在图外）→ 返回可捕获的 forward"的工厂（上游同名 Protocol）。

    调用时机有讲究：热身（warmup=True）与**真正捕获**（warmup=False）各调一次。上游这么
    做的理由是"注意力后端可能在热身时改元数据或懒初始化"，第二次调用能拿到**干净**的
    元数据。本仓库照抄这个协议（74 关的草稿图确实两样都会发生：热身那次会把 Triton 内核的
    自动调优做掉）。
    """

    def __call__(
        self,
        desc: BatchExecutionDescriptor,
        warmup: bool,
    ) -> Callable[[CUDAGraphMode], None]: ...


def _is_compatible(
    desc: BatchExecutionDescriptor,
    num_reqs: int,
    num_tokens: int,
    uniform_token_count: int | None,
    num_active_loras: int,
    max_query_len: int | None,
    allow_shape_padding: bool,
) -> bool:
    """捕获的这张图能不能服务这一轮的批（上游 `_is_compatible` + 本仓库一条收紧）。

    上游规则（逐条同义）：
      * `desc.uniform_token_count is None`（PIECEWISE）能吃任意 uniform_token_count；
        否则必须**精确相等**——统一 decode 图的形状就是每请求 `1+K` 行，差一行都不行。
      * `desc.max_query_len is None` 表示不约束；否则要求调用方给了上界且不超过它。
      * 请求数：`desc.num_reqs >= num_reqs`（多出来的请求用补齐行）。
      * 行数：`desc.num_tokens >= num_tokens`（多出来的行是补齐行）。
      * LoRA 数必须相等（本仓库恒 0）。

    **本仓库收紧的一条**（`allow_shape_padding=False`）：上面那两条 `>=` 的前提是"调用方会
    把整批**补齐**到 `desc` 的形状"（上游 target 侧由 `ModelCudaGraphManager` 负责补齐）。
    草稿的 prefill 图直接复用 **target 的**注意力元数据与 slot_mapping，而本仓库 V2 的 target
    批不补齐（target 图不在 74 关范围内）——那种图一旦服务了更小的批，图里记的 `num_reqs`
    就会去读 target 缓冲里**没被写过**的行（旧的块号/seq_len）→ 越界或读到别人的 KV，
    而且不报错。所以那张图要求形状**精确相等**；多出来的请求/行只允许出现在"图自己拥有
    那份缓冲"的草稿 decode 图上（`allow_shape_padding=True`，与上游一致）。
    """
    if not allow_shape_padding:
        if desc.num_tokens != num_tokens or desc.num_reqs != num_reqs:
            return False
    else:
        if not (desc.num_reqs is None or desc.num_reqs >= num_reqs):
            return False
        if desc.num_tokens < num_tokens:
            return False
    return (
        (desc.uniform_token_count is None or desc.uniform_token_count == uniform_token_count)
        and (
            desc.max_query_len is None
            or (max_query_len is not None and desc.max_query_len >= max_query_len)
        )
        and desc.num_active_loras == num_active_loras
    )


class CudaGraphManager:
    """一批形状相同的前向的"捕获 / 挑键 / 重放"（上游同名类的子集）。"""

    def __init__(
        self,
        vllm_config,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        allow_shape_padding: bool = True,
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.compilation_config = vllm_config.compilation_config
        self.cudagraph_mode = cudagraph_mode
        #: "统一 decode"批的每请求行数（草稿 prefill 图 = K+1，草稿 decode 图 = 1）
        self.decode_query_len = decode_query_len
        #: 见 `_is_compatible` 的长注释：图能不能服务更小的批
        self.allow_shape_padding = allow_shape_padding

        #: 键 → 图。**唯一的图缓存**（AGENTS §9：不自造第二套图缓存）
        self.graphs: dict[BatchExecutionDescriptor, torch.cuda.CUDAGraph] = {}
        self._graphs_captured = False
        #: 运行时 token 数 → 候选图（按序遍历，第一个兼容的胜出）
        self._candidates: dict[int, list[BatchExecutionDescriptor]] = {}
        #: 每个模式要捕获哪些键（捕获时按 num_tokens **从大到小**）
        self._capture_descs: dict[CUDAGraphMode, list[BatchExecutionDescriptor]] = {}
        #: 观测用账本：这一轮真实选中了什么（验收要求"断言实际选中"）
        self.selections: list[dict] = []
        #: 真正 `replay()` 了几次（与 `selections` 不是一回事：融合多步 decode 一轮只
        #: **挑一次键**，但图里走完 K−1 步；逐步路径则是同一个键重放 K−1 次）。
        self.num_replays = 0

        self._init_candidates()

    # -------- 候选表 --------

    def _init_candidates(self) -> None:
        """按档位表列出候选图键（上游 `_init_candidates` 的等价实现）。

        两类键，规则都来自上游：
          * **统一 decode 图**（`separate_routine() and decode_mode`）：档位向上取整到
            `decode_query_len` 的倍数，请求数 = 行数 / `decode_query_len`；超过
            `max_num_reqs × decode_query_len`、超过 `max_cudagraph_capture_size`、
            或者请求数超过 `max_num_reqs` 的一律丢掉（那种形状不可能出现）。
          * **混合批图**（`mixed_mode` 非 NONE）：`num_reqs=None`（PIECEWISE 契约）。

        然后把"运行时 token 数 → 候选图"的映射铺满：**每个真实 token 数映射到"能装下它的
        最小捕获档位"**（上游注释：round the runtime token count up to the nearest captured
        shape）。这样 `dispatch()` 是一次字典查表，不用每次二分。
        """
        capture_sizes = self.compilation_config.cudagraph_capture_sizes
        if not (self.cudagraph_mode and capture_sizes):
            return

        capture_sizes = sorted(capture_sizes)
        max_decode_tokens = self.max_num_reqs * self.decode_query_len
        decode_mode = self.cudagraph_mode.decode_mode()
        mixed_mode = self.cudagraph_mode.mixed_mode()
        separate_decode_routine = self.cudagraph_mode.separate_routine()
        max_cg_capture_size = self.compilation_config.max_cudagraph_capture_size

        descs_by_token: dict[int, list[BatchExecutionDescriptor]] = defaultdict(list)
        descs_by_mode: dict[CUDAGraphMode, list[BatchExecutionDescriptor]] = defaultdict(list)

        for num_tokens in capture_sizes:
            if separate_decode_routine and decode_mode:
                rounded_num_tokens = -(-num_tokens // self.decode_query_len) * self.decode_query_len
                rounded_num_reqs = rounded_num_tokens // self.decode_query_len
                if not (
                    rounded_num_tokens > max_decode_tokens
                    or rounded_num_tokens > max_cg_capture_size
                    or rounded_num_reqs > self.max_num_reqs
                ):
                    desc = BatchExecutionDescriptor(
                        cg_mode=decode_mode,
                        num_tokens=rounded_num_tokens,
                        num_reqs=rounded_num_reqs,
                        uniform_token_count=self.decode_query_len,
                    )
                    if desc not in descs_by_mode[decode_mode]:
                        descs_by_mode[decode_mode].append(desc)
                        descs_by_token[rounded_num_tokens].append(desc)

            if mixed_mode:
                # PIECEWISE 图重放时不限制请求数（不需要请求补齐），所以是 None。
                num_reqs = None
                if mixed_mode == CUDAGraphMode.FULL:
                    num_reqs = min(num_tokens, self.max_num_reqs)
                desc = BatchExecutionDescriptor(
                    cg_mode=mixed_mode, num_tokens=num_tokens, num_reqs=num_reqs
                )
                descs_by_mode[mixed_mode].append(desc)
                descs_by_token[num_tokens].append(desc)

        if not descs_by_token:
            return

        all_token_counts = sorted(descs_by_token)
        current_range_start = 0
        for token_cg_size in all_token_counts:
            for i in range(current_range_start, token_cg_size + 1):
                self._candidates[i] = descs_by_token[token_cg_size]
            current_range_start = token_cg_size + 1

        for mode, descs in descs_by_mode.items():
            descs.sort(key=lambda d: d.num_tokens, reverse=True)
            self._capture_descs[mode] = descs

    def needs_capture(self) -> bool:
        return len(self._capture_descs) > 0

    def captured_token_counts(self) -> list[int]:
        """已经捕获了哪些 token 数（上游同名方法；观测 + 测试用）。"""
        return sorted({desc.num_tokens for desc in self.graphs})

    # -------- 捕获 --------

    @torch.inference_mode()
    def capture(
        self,
        create_forward_fn: CreateForwardFn,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """逐张捕获（上游 `CudaGraphManager.capture` 的等价实现）。

        三处顺序不能改，每处都有实测理由（69 关踩过）：

            1. **大档位先捕获**（`_capture_descs` 已按 num_tokens 降序）：小图能复用大图占下
               的显存池；反过来每张小图都要单独开池。
            2. **每张图先热身一次**：第一次跑某个形状时 cuBLAS handle / workspace、Triton 自动
               调优、缓存块分配都会发生，而这些**不允许在捕获里创建**（实测
               `CUBLAS_STATUS_NOT_INITIALIZED then cublasCreate`）。
            3. FULL 图要**重新准备一次输入**（`warmup=False`）再录：热身那次可能改过元数据
               （上游注释：attention backends may mutate or lazily initialize metadata），
               录进图的必须是"干净的那一份"。

        ⚠️ **热身的运行模式与图里要跑的那条路径一致**（上游热身传 `NONE`，因为它的注意力按
        metadata 选路径）：本仓库的 Torch 后端按 forward 上下文里的运行模式在"逐请求 eager"与
        "固定形状批量"之间分叉，两条路的 cuBLAS 形状完全不同。只用 `NONE` 热身的话，图里那条
        路径就是**在捕获窗口里第一次执行**——第一次要做的懒初始化（cuBLAS 计划/workspace、
        Triton 加载）落在捕获里是"不允许的操作"。所以热身也走 `desc.cg_mode`。

        本仓库没有 torch.compile / breakable graph，所以 PIECEWISE 那一支只是**照常前向**
        （分段图由模型层的包装器自己负责），不在这里建 `CUDAGraph`——与上游"PIECEWISE 不进
        `self.graphs`"一致。
        """
        with graph_capture(self.device):
            for mode in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL):
                if mode not in self._capture_descs:
                    continue
                for desc in self._capture_descs[mode]:
                    forward_fn = create_forward_fn(desc, True)
                    warmup_mode = (desc.cg_mode if desc.cg_mode == CUDAGraphMode.FULL
                                   else CUDAGraphMode.NONE)
                    forward_fn(warmup_mode)          # 热身（走图里那条路径）
                    forward_fn = create_forward_fn(desc, False)
                    if desc.cg_mode == CUDAGraphMode.PIECEWISE:
                        forward_fn(CUDAGraphMode.PIECEWISE)
                        continue
                    assert desc not in self.graphs, f"这个形状已经捕获过图了：{desc}"
                    graph = torch.cuda.CUDAGraph()
                    # 捕获用的图模式必须是**重放时要执行的那条路径**。上游这里传 NONE，
                    # 因为它的注意力后端靠 metadata（`for_capture=True`）决定走固定形状路径；
                    # 本仓库的 Torch 后端靠 forward 上下文里的运行模式决定（`_graph_dispatched()`），
                    # 传 NONE 会把**逐请求 eager 路径录进图**（那里有 `int(张量)` 同步，捕获当场
                    # 就会炸）。所以这里传 FULL ——差异记在 `docs/step74_alignment.md`。
                    with torch.cuda.graph(graph, _get_graph_pool()):
                        forward_fn(CUDAGraphMode.FULL)
                    self.graphs[desc] = graph
        self._graphs_captured = True

    # -------- 运行期 --------

    def dispatch(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int = 0,
        max_query_len: int | None = None,
        need_eager: bool = False,
    ) -> BatchExecutionDescriptor:
        """真实形状 → 图键（上游 `dispatch` 的等价实现）。

        回退成 `NONE` 的情形有四种，**都不猜**：图还没捕获完、token 数为 0、候选表里没有这个
        token 数、候选都不兼容。回退返回的描述符带的是**调用方给的未补齐数字**（上游同款），
        所以调用方可以放心用它去跑 eager。
        """
        if need_eager or not self._graphs_captured or num_tokens <= 0:
            return self._fallback(num_reqs, num_tokens, num_active_loras)
        for desc in self._candidates.get(num_tokens, ()):
            if _is_compatible(
                desc,
                num_reqs,
                num_tokens,
                uniform_token_count,
                num_active_loras,
                max_query_len,
                self.allow_shape_padding,
            ):
                return desc
        return self._fallback(num_reqs, num_tokens, num_active_loras)

    @staticmethod
    def _fallback(num_reqs: int, num_tokens: int,
                  num_active_loras: int) -> BatchExecutionDescriptor:
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=num_active_loras,
        )

    def record_selection(self, desc: BatchExecutionDescriptor, num_reqs: int,
                         num_tokens: int) -> None:
        """记下这一轮**真实选中**的描述符（"实际选中了哪条路"的唯一写入点）。

        验收要求"断言实际选中 fused/fallback"，所以这不能只是日志：测试直接读这个列表。
        """
        self.selections.append({
            "num_reqs": num_reqs,
            "num_tokens": num_tokens,
            "desc_num_reqs": desc.num_reqs,
            "desc_num_tokens": desc.num_tokens,
            "mode": str(desc.cg_mode),
            "replayed": desc in self.graphs,
        })

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> None:
        """重放一张 FULL 图（上游同名方法）。

        **不准备任何输入**：图里记的是指针，调用方必须已经把这一轮的数据写进那批常驻缓冲。
        这正是"V2 图路径"的全部前提，写成断言而不是注释。
        """
        assert desc.cg_mode == CUDAGraphMode.FULL, f"期望 FULL 图，收到 {desc.cg_mode}"
        assert desc in self.graphs, f"没有这张图：{desc}"
        self.graphs[desc].replay()
        self.num_replays += 1


def _get_graph_pool():
    """全进程共享的图显存池（与 69 关的 `graph_capture` 用同一个）。

    为什么共享而不是一张图一个池：同一条流上任何时刻只有一张图在重放，共享池是安全的；
    而每个池都要按"图里录下的分配"独占显存，一张一个池会让显存按图数增长。
    """
    from ...compilation.cuda_graph import _get_global_graph_pool

    return _get_global_graph_pool()


class ModelCudaGraphManager(CudaGraphManager):
    """target 模型的图管理器（对应上游 `ModelCudaGraphManager`，Phase 1 子集）。

    与 speculator 的图管理器共用 `CudaGraphManager` 的候选表/捕获/分派/重放；
    额外只做两件 target 特有的事：

        **准备输入**：`create_forward_fn()` 在图外调 `prepare_inputs_to_capture()`，
            把 dummy 批写进常驻输入缓冲，并用 `_build_attn_metadata` 造一份图内 metadata；
        **持有输出**：图重放写进静态 hidden/aux 缓冲，`run_fullgraph()` 返回它们的切片。
            `sample_tokens` 不能拿到一个重放时新分配的张量——图里记的是捕获时的地址。
    """

    def __init__(self, vllm_config, device: torch.device,
                 cudagraph_mode: CUDAGraphMode, decode_query_len: int,
                 allow_shape_padding: bool = True):
        super().__init__(
            vllm_config, device, cudagraph_mode, decode_query_len,
            allow_shape_padding=allow_shape_padding,
        )
        self.hidden_states: torch.Tensor | None = None
        self.aux_hidden_states: list[torch.Tensor] = []
        self.use_aux_hidden_state_outputs = False

    @torch.inference_mode()
    def capture(
        self,
        model,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        build_attn_metadata: Callable[[InputBatch], dict[str, Any]],
        use_aux_hidden_state_outputs: bool = False,
        progress_bar_desc: str = "Capturing target CUDA graphs",
    ) -> None:
        self.use_aux_hidden_state_outputs = use_aux_hidden_state_outputs

        def create_forward_fn(desc: BatchExecutionDescriptor, warmup: bool):
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
            attn_state = prepare_inputs_to_capture(
                num_reqs, num_tokens, input_buffers, block_tables,
                build_attn_metadata,
                full_cudagraph=desc.cg_mode == CUDAGraphMode.FULL,
            )
            batch_descriptor = BatchDescriptor(
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                uniform=desc.uniform_token_count is not None,
            )

            def forward_fn(cg_mode: CUDAGraphMode) -> None:
                with set_forward_context(
                    attn_state.attn_metadata,
                    num_tokens=num_tokens,
                    cudagraph_runtime_mode=cg_mode,
                    batch_descriptor=batch_descriptor,
                    slot_mapping=attn_state.slot_mappings,
                ):
                    model_output = model(
                        input_ids=input_buffers.input_ids[:num_tokens],
                        positions=input_buffers.positions[:num_tokens],
                    )
                if self.use_aux_hidden_state_outputs:
                    hidden_states, aux_hidden_states = model_output
                else:
                    hidden_states, aux_hidden_states = model_output, None
                if self.hidden_states is None:
                    self.hidden_states = torch.empty_like(hidden_states)
                self.hidden_states[:num_tokens] = hidden_states
                if self.use_aux_hidden_state_outputs:
                    if not self.aux_hidden_states:
                        self.aux_hidden_states = [
                            torch.empty_like(x) for x in aux_hidden_states]
                    for i, aux in enumerate(aux_hidden_states):
                        self.aux_hidden_states[i][:num_tokens] = aux

            return forward_fn

        super().capture(create_forward_fn, progress_bar_desc)

    def run_fullgraph(self, desc: BatchExecutionDescriptor):
        super().run_fullgraph(desc)
        assert self.hidden_states is not None
        hidden_states = self.hidden_states[: desc.num_tokens]
        if not self.use_aux_hidden_state_outputs:
            return hidden_states
        return hidden_states, [x[: desc.num_tokens] for x in self.aux_hidden_states]


def prepare_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    build_attn_metadata: Callable[[InputBatch], dict[str, Any]],
    full_cudagraph: bool,
) -> AttentionState:
    """为一次捕获准备"假"输入与注意力元数据（上游同名函数的等价实现）。

    两件事必须同时成立，否则录出来的图不能重放：

        **张量地址**：`make_dummy` 只往**常驻缓冲**里写（`input_ids` / `positions` /
            `query_start_loc` / `seq_lens` / 块表 / slot_mapping 都是常驻的），所以捕获时与
            运行期是同一批指针。
        **内容安全**：补齐行必须是"写了也没人读"的值——`input_ids`/`positions` 清零（会被
            embedding/RoPE 查表读到，超范围就是 device 端越界），slot_mapping 全填哨兵
            `-1`（写 KV 时被跳过或夹到 0 号垃圾桶），块表补齐行清零（0 号块是垃圾桶，
            留着旧块号会读到别人的 KV）。

    与上游的差异：没有 DCP 的 `dcp_local_seq_lens`、没有 `model_state`（本仓库的注意力元数据
    由 Runner 侧一个回调构造，而不是 ModelState 对象）。
    """
    input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)
    block_tables.get_dummy_block_tables(num_reqs)
    block_tables.get_dummy_slot_mappings(num_tokens)
    attn_metadata = build_attn_metadata(input_batch)
    return AttentionState(attn_metadata, {"slot_mapping": block_tables.slot_mappings[:, :num_tokens]})
