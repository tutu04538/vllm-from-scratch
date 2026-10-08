"""`GPUModelRunner`：把 `SchedulerOutput` 变成模型输入、跑模型、产出采样结果
（对应 vLLM `v1/worker/gpu_model_runner.py` 的子集）。

198 §1 把旧 `TinyCausalLM` 里揉在一起的职责拆开了，这里就是那些职责的新位置：

    从请求提取本轮 token、算 positions        → `_prepare_inputs`
    常驻输入缓冲、batch 行管理、slot_mapping   → `InputBatch` / `BlockTable`
    生成 AttentionMetadata                    → `AttentionMetadataBuilder`
    QKV / RoPE / norm / MLP                  → 模型与 layers（`models/qwen3.py`）
    KV 写入与 paged attention                → `Attention` + 后端实现
    按采样行选 hidden、算 logits              → `_run_model` + `model.compute_logits()`
    采样后构造结果、维护执行端镜像             → `sample_tokens` / `_bookkeeping_sync`

**两条边界不能破**：

1. 模型 `forward(input_ids, positions)` 拿不到 Request / KV 池 / sampling_params；它要知道的
   一切（这一层该用哪份 metadata、KV 写到哪）都经由 `set_forward_context()` 与绑定的 `kv_cache`
   传进去。
2. Scheduler **不碰 GPU 张量**：它只发 `SchedulerOutput`（纯数据），执行侧自己的缓冲、
   块表镜像、generator 都在这里。`requests` 是 Worker 的缓存，**不是** `Scheduler.requests` 的别名。

**三种角色的状态**（198 §2）：控制端权威（Scheduler）→ 执行端镜像（`CachedRequestState` +
`InputBatch`）→ GPU 输入副本。镜像的进度（`num_computed_tokens`）**每轮由协议校正**，
执行侧不自增：自增会让"谁是真相"出现第二个来源。

**执行与采样分两步**（198 §5）：`execute_model()` 跑完模型把 logits 存进
`execute_model_state` 并返回 `None`；`sample_tokens()` 才消费它。"上一轮没消费就再来一轮"
必须报错，模型执行异常也不能把旧 logits 留给下一次采样——简单做法是让这个 Runner 进入
失败状态，不再接受新的一轮（不做没有依据的回滚）。
"""

from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np
import torch

from ..attention import Attention, AttentionMetadataBuilder
from ..attention.backends.torch_sdpa import TorchAttentionBackend
from ..compilation import (CUDAGraphLogging, CUDAGraphStat, CUDAGraphWrapper, graph_capture,
                           set_cudagraph_capturing_enabled)
from ..config import CUDAGraphMode
from ..cudagraph_dispatcher import CudagraphDispatcher
from ..forward_context import BatchDescriptor, set_forward_context
from ..outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from ..outputs import DraftTokenIds
from ..sample import RejectionSampler, Sampler, SamplingMetadata
from ..spec_decode.metadata import SpecDecodeMetadata
from ..spec_decode.utils import PADDING_SLOT_ID, TargetRows
from .gpu_input_batch import InputBatch


class CachedRequestState:
    """执行端的**请求镜像**。与 `Scheduler` 的 `Request` 是两份状态，靠协议数据对齐。

    保存 `generator` 而不是"第几行"：batch 行会重排（`condense`），随机流必须跟着请求走。
    """

    def __init__(self, req_id: str, prompt_token_ids: list[int], sampling_params,
                 generator, block_ids: list[list[int]], num_computed_tokens: int = 0,
                 output_token_ids: list[int] | None = None) -> None:
        self.req_id = req_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.sampling_params = sampling_params
        self.generator = generator
        # 每个 KV group 一张块表；本关只有一个 group
        self.block_ids = [list(group) for group in block_ids]
        self.num_computed_tokens = num_computed_tokens
        self.output_token_ids: list[int] = list(output_token_ids or [])

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def all_token_ids(self) -> list[int]:
        """执行侧认为的完整历史。它与控制端的长度会**短暂**不一致（控制端可能已经提交了
        执行侧还不知道的 token），下一轮协议里的 `num_output_tokens` 会把镜像校正回来。"""
        return list(self.prompt_token_ids) + list(self.output_token_ids)


class AsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):
    """异步结果句柄（对应 vLLM `v1/worker/gpu_model_runner.py::AsyncGPUModelRunnerOutput`）。

    它持有三样东西，**在 `get_output()` 之前都不能被覆盖**：

        sampler_output   产生结果的 device 张量（拷贝的源）
        _cpu_buffers     这块结果专用的 pinned 主机缓冲（拷贝的目标）
        event            侧流上的完成事件（判断"值可不可用"的唯一依据）

    生命周期由 Runner 保证：`get_output()` 只允许调用一次，调用后归还缓冲；下一轮的
    `execute_model()` 会先把欠账结清，所以"同一块 pinned 缓冲被两轮同时用"不会发生
    （`test_async_buffer_lifetime.py` 用**事件门闩**把这条钉住，而不是靠 sleep 猜时序）。
    """

    def __init__(self, runner, state, sampler_output, kind: str = "plain") -> None:
        import torch

        self._runner = runner
        self._state = state
        # "plain" = 普通采样（每行 1 个 token）；"spec" = 验证批（`[B, K+1]`，被拒位置 -1）。
        # 两者的 device 张量同形（都是二维），差别在**怎么解析**：验证批要走
        # `RejectionSampler.parse_output`（同一张 valid_mask 裁 token 与 logprobs）。
        self._kind = kind
        self._sampler_output = sampler_output
        self._delivered = False
        self._result: ModelRunnerOutput | None = None
        # 事件用 blocking=True：等的时候让出 CPU，不做忙轮询（上游同款注释）
        self._event = torch.cuda.Event(blocking=True) if torch.cuda.is_available() else None
        self._started = False
        # 构造即发起拷贝（上游同款：`__init__` 里就把 D2H 发到侧流上）。
        # **别等到 get_output() 才拷**：那样"异步"就只剩一个名字了。
        self.start_non_blocking_copy()

    def start_non_blocking_copy(self) -> None:
        """在侧流上发起 D2H（非阻塞）并记事件。

        为什么要单独一条流：拷贝要和"下一次前向"重叠。为什么要 pinned 内存：只有 pinned
        的 `non_blocking=True` 拷贝才是真异步（普通页内存会退化成同步拷贝，等于没异步）。
        """
        import torch

        if self._started or not torch.cuda.is_available():
            return
        self._started = True
        sampler_output = self._sampler_output
        # 结果就是这份 device 张量；拷贝到 pinned 主机缓冲（生命周期与句柄绑定）
        self._tokens_cpu = torch.empty_like(sampler_output.sampled_token_ids,
                                            device="cpu", pin_memory=True)
        source = sampler_output.sampled_token_ids
        if source.is_cuda:
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                stream.wait_stream(torch.cuda.current_stream())
                self._tokens_cpu.copy_(source, non_blocking=True)
                self._event.record(stream)
        else:
            self._tokens_cpu.copy_(source)
            self._event.record()

    def get_output(self) -> ModelRunnerOutput:
        """等拷贝完成 → 解析 → 记账 + 提议 → 交出 CPU 结果。

        **记账部分只做一次**（`_delivered` 守住）：那是一段有副作用的逻辑（写镜像、提草稿），
        重跑会把同一批 token 提交两遍。但**结果可以重复读**：交付之后拿到的是我们自己的
        Python list，不再指向被复用的缓冲，所以"再来一次 `get_output()`"是安全的——
        本仓库有两个等待点（下一步 `execute_model()` 的输入组装前、引擎的交付边界），
        两边都可能先到。
        """
        if self._delivered:
            return self._result
        self._delivered = True
        if self._event is not None:
            self._event.synchronize()
        runner = self._runner
        try:
            # 拷贝回来的那份替换掉 device 张量：后半段与同步路径共用同一段实现
            sampler_output = replace(self._sampler_output,
                                     sampled_token_ids=self._tokens_cpu
                                     if self._started else self._sampler_output
                                     .sampled_token_ids)
            if self._kind == "spec":
                sampled, logprobs_by_req = runner._parse_spec_sampler_output(
                    self._state, sampler_output)
            else:
                sampled = sampler_output.sampled_token_ids.tolist()
                logprobs_by_req = runner._logprobs_by_request(
                    sampler_output.logprobs_tensors, self._state.sample_rows)
            try:
                self._result = runner._finish_async_output(
                    self._state, sampled, logprobs_by_req)
            except Exception as exc:                 # noqa: BLE001 —— 与同步路径同一条约定
                # 70 关：异步下"记账 + 提议"发生在这里（同步路径发生在 sample_tokens），
                # 所以失败态也要在这里记：清掉未交付的草稿、让 Runner 停摆（下一轮在调度前
                # 就拒绝）。漏了这一步 = 半轮状态被当成正常状态继续跑。
                runner.pending_draft_token_ids = None
                runner.pending_draft_probs = None
                runner.failure = f"{type(exc).__name__}: {exc}"
                raise
            return self._result
        finally:
            runner._pending_async_output = None
            self._sampler_output = None          # 释放 device 张量引用（缓冲可复用）


class _PaddedRun(NamedTuple):
    """图分派路径的一次前向所需要的全部信息（批次键 + 运行模式 + 补齐后的行数）。

    单独一个类型而不是散着传：`_dummy_run`（热身/捕获）与 `execute_model`（真实重放）必须
    走**同一条**代码路径，否则"录下来的图"和"真实路径"就不是一回事了。
    """

    batch_descriptor: BatchDescriptor
    mode: CUDAGraphMode
    num_tokens: int


class ExecuteModelState(NamedTuple):
    """`execute_model()` 与 `sample_tokens()` 之间的临时状态。

    只放"这一轮采样需要的东西"：调度快照、logits、以及 **logits 行 → batch 行** 的映射。
    不放整个 Scheduler，也不放模型输入张量（那些用完即弃，留着只会把显存占住）。
    """

    scheduler_output: object
    logits: torch.Tensor
    sample_rows: list[int]
    spec_metadata: object = None
    # 68 关：这一轮的语法掩码（`EngineCore.step()` 里算好、随 `sample_tokens()` 传进来）。
    # 它只在这一步有效：掩码行与 logits 行一一对应，打完就丢。
    grammar_output: object = None
    # 69 关：本轮 target 的 `query_start_loc`（device 版）。给 `prepare_inputs_padded` 用——
    # 它按"每请求 query 块的最后一行 − 被拒行数"算该从哪一行采样（上游同一个 kernel 的输入）。
    query_start_loc: torch.Tensor | None = None
    # 63 关（EAGLE）：提议者要照上游那样对本轮的输入做"整体左移 + 打补丁"，所以要把
    # **本轮真正的输入**（含被拒草稿那几行）留到提议时刻。只留 **CPU** 张量（不占显存），
    # 而且只在 `capture_aux_hidden_states` 时才留——上游是把它们当场当参数传进 proposer 的。
    target_token_ids_cpu: torch.Tensor | None = None
    target_positions_cpu: torch.Tensor | None = None
    # 64 关（extract_hidden_states）：cache-only 层要把特征写进**与 target KV 相同的槽位**，
    # 所以提议者要拿到本轮那份 attention 元数据（槽位就在它里面）。只存**引用**、不复制，
    # 生命周期只在 execute→sample 之间（上游的 execute state 同样持有 common_attn_metadata）。
    common_attn_metadata: object = None


@dataclass
class PreparedInputs:
    """`_prepare_inputs()` 的产物：**CPU 上的张量**（还没上传）。

    分开的理由是 198 §4 那句"张量准备与模型数学的职责必须先分开"：数字能单独打印、单独对照，
    上传与模型计算是后面的事。字段名与 vLLM 的循环里那些变量同名，便于源码对照。
    """

    input_ids: torch.Tensor                 # [num_tokens]       本轮要算的 token
    positions: torch.Tensor                 # [num_tokens]       绝对位置
    query_start_loc: torch.Tensor           # [num_reqs + 1]     每请求的 query 起止（前缀和）
    seq_lens: torch.Tensor                  # [num_reqs]         本轮之后每请求的上下文长度
    slot_mapping: torch.Tensor              # [num_tokens]       KV 写到哪个物理槽位
    logits_indices: torch.Tensor            # [num_reqs]         每请求最后一个 query 行
    sample_rows: list[int] = field(default_factory=list)   # 其中"可以采样"的 batch 行
    num_scheduled_tokens: list[int] = field(default_factory=list)
    # 投机批的形状（没有草稿时是 None）：行号怎么切、哪几行要 logits，都在它里面
    spec_metadata: "SpecDecodeMetadata | None" = None

    @property
    def num_tokens(self) -> int:
        return int(self.input_ids.shape[0])


class GPUModelRunner:
    def __init__(self, vllm_config, device: str, model=None) -> None:
        self.vllm_config = vllm_config
        self.device = device
        self.model = model
        self.max_model_len = vllm_config.model_config.max_model_len
        self.block_size = vllm_config.cache_config.block_size

        # 执行端自己的状态
        self.requests: dict[str, CachedRequestState] = {}
        self.input_batch = InputBatch(
            max_num_reqs=vllm_config.scheduler_config.max_num_seqs,
            max_model_len=self.max_model_len,
            device=device,
            block_size=self.block_size,
            # 词表大小来自 **config**（不是等模型加载完再问）：InputBatch 在 __init__ 就要建，
            # 而 top_k 的归一化规则要用它。假执行路径的 config 没有 hf_config → None
            vocab_size=(vllm_config.model_config.hf_config or {}).get("vocab_size"))
        # 用**后端自己的** builder（而不是基类）：71 关的能力协商要读它声明的图能力档位
        # （`get_cudagraph_support()`），基类的档位是 NEVER（= 不能进图）。
        self.attn_metadata_builder = TorchAttentionBackend.get_builder_cls()(self.block_size)
        # 68 关：logprobs 的四种模式是**引擎级**配置（上游 `ModelConfig.logprobs_mode`），
        # 采样器与拒绝采样器都要知道它——拒绝采样器还要据此决定"交付 raw 还是 processed 那份"。
        self.logprobs_mode = vllm_config.model_config.logprobs_mode
        self.sampler = None                       # load_model() 之后才有（采样要用模型精度）
        # 投机的两个部件（57E）：拒绝采样器复用普通采样器（对应上游 sample/rejection_sampler.py，
        # 59 关搬到 `sample/` 下并换成 Triton 批量内核）；提议器由 load_model() 按配置建
        self.rejection_sampler = RejectionSampler(
            Sampler(self.logprobs_mode), vllm_config.speculative_config)
        # 投机元数据的索引算术要在 CPU 侧反复做，`arange` 预分配到最大批（上游 `arange_np` 同款）
        arange_size = max(vllm_config.scheduler_config.max_num_batched_tokens,
                          vllm_config.scheduler_config.max_num_seqs + 1)
        self._arange_np = np.arange(arange_size, dtype=np.int64)
        self._arange_scratch = np.empty(arange_size, dtype=np.int64)
        self.proposer = None
        self.speculative_config = vllm_config.speculative_config
        # 本轮验证时顺手提的下一轮草稿（`post_step` 取走）；概率留到下一轮按采用的前缀重排
        self.pending_draft_token_ids = None
        self.pending_draft_probs = None
        # 本轮协议里"整表替换过块表"的请求（draft 侧据此重置自己的进度）
        self._resumed_req_ids: set[str] = set()
        # 60 关：ngram_gpu 的显存历史缓冲在 `_build_proposer()` 里分配（只有这个方法用得到）
        self.capture_aux_hidden_states = False
        self.aux_hidden_states: torch.Tensor | None = None
        # 65 关（MTP）：提议者吃的是 target 的**最后一层** hidden（不是辅助层），所以这里留一份
        # `_run_model()` 每轮算出来的 hidden 引用（同一轮 execute→sample 之间有效）
        self.target_hidden_states: torch.Tensor | None = None
        self.eagle_aux_hidden_state_layers: tuple[int, ...] = ()
        # 本轮那份 attention 元数据（`_build_attn_metadata()` 每轮重设）。64 关的 cache-only
        # 提议者要用它的 `slot_mapping`（与 target 的 KV 同一份槽位），所以留一个引用。
        self.attn_metadata = None
        self.num_tokens_no_spec_gpu: torch.Tensor | None = None
        self.token_ids_gpu_tensor: torch.Tensor | None = None
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.execute_model_state: ExecuteModelState | None = None
        # 70 关：异步调度下"已经算完、还没拷回"的那一份结果（None = 没有欠账）。
        self._pending_async_output: "AsyncGPUModelRunnerOutput | None" = None
        # 是否走异步调度（由 Worker 在 load_model 时按"配置 + executor 能力"解析后写入；
        # 默认 False = 同步，直接构造 Runner 的测试不受影响）
        self.async_scheduling = False
        # 本轮"真正要验证的草稿"（同步 = 协议里的值；异步 = 执行侧补的真实值）。
        # 由 `_update_states()` 每轮重设；放在这里只是给"还没跑过一轮"的读取一个默认值。
        self._resolved_spec_decode_tokens: dict[str, list[int]] = {}
        self.failure: str | None = None

        # ---------------- 69 关：CUDA Graph ----------------
        self.compilation_config = vllm_config.compilation_config
        # 统一 decode 批的每请求行数（普通 decode = 1，投机 = 1 + K）。它是图的形状参数：
        # 只有"每请求恰好这么多行"的批才有固定形状（见 forward_context.BatchDescriptor）。
        self.uniform_decode_query_len = 1 + vllm_config.num_speculative_tokens
        self.cudagraph_dispatcher = CudagraphDispatcher(vllm_config)
        # 静态输入工作区（按 max_num_batched_tokens / max_num_reqs 开一次，地址恒定）。
        # 图里记的是**指针**，所以这些缓冲必须常驻、每轮只覆盖内容。
        self._padded_buffers: dict[str, torch.Tensor] | None = None
        # 图键 → 捕获时建好的那份 metadata（它的张量就是上面的静态缓冲，每轮原地更新）。
        # **不能每轮新建 metadata**：新对象里的张量是别人的地址，图重放时读不到。
        self._padded_metadata: dict[BatchDescriptor, object] = {}
        # 真实分派记录（每轮一条）：测试与 profiler 记录"这一轮选了哪个 mode/键"靠它，
        # 不是靠日志猜。字段名与 README/对齐文档里的表一致。
        self.cudagraph_selections: list[dict] = []
        self.cudagraph_capture_stats: dict | None = None
        # 图命中统计（`CUDAGraphLogging`）：`initialize_cudagraph_capture()` 里按最终模式建
        self.cudagraph_logging: CUDAGraphLogging | None = None

    # -------- 初始化 --------

    def load_model(self):
        """装模型。等价于 vLLM 的 `GPUModelRunner.load_model()`（那里的 `self.model = ...`）。"""
        from ..model_loader import get_model

        if self.model is None:
            self.model = get_model(self.vllm_config.model_config, self.device)
        from ..sample import Sampler

        self.sampler = Sampler(self.logprobs_mode)
        self.proposer = self._build_proposer()
        # 63/64 关：EAGLE3 与 extract_hidden_states 都要 target **顺带输出若干辅助层**的 hidden
        # states（EAGLE-1 只要最后一层，不吃 aux）。层号来自 draft 配置，缺省用 target 的默认值
        # `(2, n//2, n-3)`（与上游同一套）；extract 例外——它的层号是必需的（配置期已校验），
        # 因为"存哪几层"没有默认值可言。
        self.capture_aux_hidden_states = False
        if self.speculative_config is not None and (
                self.speculative_config.eagle3_use_aux_hidden_state()
                or self.speculative_config.uses_extract_hidden_states()):
            # 注意：**MTP 不走这里**——它吃的是 target 的最后一层 hidden（`_run_model` 本来就会
            # 算出来），不需要 target 顺带输出辅助层（上游 `use_aux_hidden_state_outputs` 同样是
            # False）。把 MTP 塞进这条分支会让 target 白算几层特征。
            self.capture_aux_hidden_states = True
            layers = self.speculative_config.eagle_aux_hidden_state_layers() \
                or self.model.get_eagle3_default_aux_hidden_state_layers()
            self.model.set_aux_hidden_state_layers(layers)
            self.eagle_aux_hidden_state_layers = tuple(layers)
        # 69 关：图键必须在**注意力后端与 KV 缓存确定之后**才好定（键里含着请求数上限），
        # 但"要不要建图"此刻就能定（设备、enforce_eager、模式都是配置事实）。
        self.initialize_cudagraph_capture()
        return self.model

    def _target_hidden_states_by_req(self, scheduler_output, num_reqs: int):
        """把本轮 target 的 hidden states 按**请求**切片（EAGLE/MTP 的第一遍要逐行配对）。

        两份来源，取决于方法（上游 `gpu_model_runner.py:5325-5365` 的同一个判断）：

            use_aux_hidden_state_outputs（EAGLE3）：多个辅助层拼在最后一维 → [T, L*H]
            否则（**MTP** / EAGLE-1）      ：target 本轮每个 query 行的**最后一层 hidden** → [T, H]

        行序与 `_prepare_inputs` 的展平顺序一致（批行序 × 每请求 `num_scheduled_tokens`），
        所以偏移量就是 `num_scheduled_tokens` 的前缀和。特征只是**本轮快照**：抢占/重排之后
        必须由 target 重新算，不能长期留着（需求 063 §2）。
        """
        features = (self.aux_hidden_states if self.capture_aux_hidden_states
                    else self.target_hidden_states)
        if features is None:
            return None
        hidden_by_req = {}
        offset = 0
        for req_id in self.input_batch.req_ids[:num_reqs]:
            length = int(scheduler_output.num_scheduled_tokens[req_id])
            hidden_by_req[req_id] = features[offset:offset + length]
            offset += length
        return hidden_by_req

    def _build_proposer(self):
        """按配置建提议器：`ngram` 用历史匹配，`draft_model` 再加载一个小模型（57E）。

        没有投机配置 → 没有提议器 → `take_draft_token_ids()` 恒为 None，
        Scheduler 那边也不会收到草稿（整条路径是关的）。

        **分派只在这里做一次**，而且只认 `speculative_config.method`——"用哪种投机"的推断
        已经在配置期归一化完了（62 关需求 §3.1：不让 CLI 判一次、Runner 再猜一次）。
        `custom_class` 放在最前，与上游 `gpu_model_runner.py:645` 的分支顺序一致。
        """
        config = self.speculative_config
        if config is None:
            return None
        if config.method == "custom_class":
            # 62 关：用户类直接就是 Runner 持有的提议者（不包 Adapter）。
            # 上游同款：构造参数只有 `VllmConfig`，拿不到 Request / KVCacheManager。
            from ..spec_decode.custom_class_proposer import create_custom_proposer

            return create_custom_proposer(self.vllm_config)
        if config.use_eagle():
            # 63/65 关：EAGLE3 与 MTP 都走 `EagleProposer`（上游 `use_eagle()` 就把 mtp 算进来，
            # Runner 同样建 EagleProposer）——它们都是"吃 target hidden 的迭代提议"，
            # 提议循环一行都不用改。
            from ..spec_decode.eagle import EagleProposer

            if config.uses_mtp() and config.draft_model_config is None:
                # MTP 的权重在 target 的 checkpoint 里，draft 配置由 target 配置派生
                # （上游在 `SpeculativeConfig.__post_init__` 里做，本仓库拿到 target 的时机在这里；
                # `replace` 会重跑配置校验，K 与 n_predict 的整除关系也在这一步判）
                config = replace(
                    config,
                    draft_model_config=config.derive_mtp_draft_config(
                        self.vllm_config.model_config))
            proposer = EagleProposer(config, self.vllm_config, self.device)
            proposer.load_model()
            if self.model is not None:
                proposer.share_embeddings(self.model)
            return proposer
        if config.method == "ngram":
            from ..spec_decode.ngram_proposer import NgramProposer

            return NgramProposer(self.vllm_config)
        if config.method == "ngram_gpu":
            from ..spec_decode.ngram_proposer_gpu import NgramProposerGPU

            # 60 关：GPU 提议者要一份**常驻显存**的历史 + 长度表，每步只增量写新采样的 token
            # （上游在 Runner 里分配同样两个张量）
            self.num_tokens_no_spec_gpu = torch.zeros(
                self.input_batch.max_num_reqs, dtype=torch.int32, device=self.device)
            self.token_ids_gpu_tensor = torch.zeros(
                self.input_batch.max_num_reqs, self.max_model_len, dtype=torch.int32,
                device=self.device)
            return NgramProposerGPU(self.vllm_config, self.device, self)
        if config.method == "draft_model":
            from ..spec_decode.draft_model import DraftModelProposer

            proposer = DraftModelProposer(config, self.vllm_config, self.device)
            proposer.load_model()
            return proposer
        if config.method == "suffix":
            # 61 关：suffix decoding 没有模型要装（`load_model()` 是空操作，上游同款），
            # 树与匹配全在外部包 `arctic_inference.suffix_decoding` 里
            from ..spec_decode.suffix_decoding import SuffixDecodingProposer

            return SuffixDecodingProposer(self.vllm_config)
        if config.uses_medusa():
            # 66 关：Medusa 的 head 是**用户给的**独立小目录。调用形态与上游逐字一致
            # （`MedusaProposer(vllm_config=..., device=...)`）；"与 target 对齐词表"那步需要
            # target 配置，由提议者在构造时自己做（与 64 关同样处理）。
            # 分支位置与上游一致（suffix 之后、extract 之前）。
            from ..spec_decode.medusa import MedusaProposer

            proposer = MedusaProposer(self.vllm_config, self.device)
            proposer.load_model()
            return proposer
        if config.uses_extract_hidden_states():
            # 64 关：它不猜 token（"草稿"就是 target 自己采出的那一列），只负责把 target 的
            # 辅助层特征写进自己的 cache-only 缓存。分支位置与上游一致（排在 eagle 之后）。
            from ..spec_decode.extract_hidden_states import ExtractHiddenStatesProposer

            proposer = ExtractHiddenStatesProposer(self.vllm_config, self.device)
            proposer.load_model()
            return proposer
        raise ValueError(f"未知的投机方法 {config.method!r}"
                         f"（本关支持 'ngram' / 'ngram_gpu' / 'draft_model' / 'suffix' / "
                         f"'custom_class' / 'eagle' / 'eagle3' / 'extract_hidden_states' / "
                         f"'mtp' / 'medusa'）")

    def initialize_kv_cache(self, kv_cache_config) -> dict[str, torch.Tensor]:
        """按 KV 规格分配物理缓存并**绑定到每个 Attention 层**。

        `num_gpu_blocks` 由上层给定（vLLM 靠显存 profiling 自动定容，本关按配置）。
        形状 `[2, num_blocks, block_size, num_kv_heads, head_size]`：第 0 片是 K、第 1 片是 V，
        其余维与"块表 + slot_mapping"的寻址方式一一对应。
        """
        if self.model is None:
            raise RuntimeError("先 load_model() 再初始化 KV 缓存（需要模型的 dtype/qk 头数）")
        if kv_cache_config.block_size != self.block_size:
            raise ValueError(
                f"KV 规格里的 block_size={kv_cache_config.block_size} 与 Runner 的 "
                f"{self.block_size} 不一致：块大小决定 slot 怎么算，两边不同会出现"
                f"'写进去的和读出来的不是同一块'这种最难查的错")
        dtype = next(self.model.parameters()).dtype
        self.kv_caches = {}
        for layer_name, layer in self._attention_layers().items():
            cache = torch.zeros(2, kv_cache_config.num_gpu_blocks, self.block_size,
                                layer.num_kv_heads, layer.head_size,
                                dtype=dtype, device=self.device)
            layer.kv_cache = cache
            self.kv_caches[layer_name] = cache
        # 64 关：cache-only 层的缓存是提议者自己分配的（与 draft 的 KV 同一套做法），但它必须
        # 与本轮的块规格一致——上游在 `initialize_kv_cache_tensors()` 之后调
        # `validate_same_kv_cache_group()`，本仓库在这里做对应的校验。
        if self.speculative_config is not None \
                and self.speculative_config.uses_extract_hidden_states():
            self.proposer.validate_same_kv_cache_group(kv_cache_config)
        return self.kv_caches

    def _attention_layers(self) -> dict[str, Attention]:
        """模型里所有 `Attention` 层，按**层名**索引——层名就是 forward 上下文的键。

        顺手校验"层名 == 模块路径"：名字对不上时 metadata 会查不到（`Attention.forward`
        会 KeyError），而那种错很难从现象上看出来，所以在建 map 的时候就报。
        """
        layers: dict[str, Attention] = {}
        for name, module in self.model.named_modules():
            if not isinstance(module, Attention):
                continue
            if module.layer_name != name:
                raise ValueError(
                    f"Attention 层名与模块路径不一致：layer_name={module.layer_name!r}，"
                    f"实际路径={name!r}。forward 上下文按层名取 metadata，两者必须相同")
            layers[name] = module
        if not layers:
            raise ValueError(f"{type(self.model).__name__} 里没有 Attention 层，无法绑定 KV 缓存")
        return layers

    # -------- 一轮：更新状态 --------

    def _check_usable(self) -> None:
        if self.failure is not None:
            raise RuntimeError(f"这个 Runner 已经失败，不再接受新的一轮：{self.failure}")

    def _update_states(self, scheduler_output) -> None:
        """按 198 §3 的七步把协议数据合进镜像。

        顺序有理由，不能随便调换：**先删（1/2）再加（3/7）**，否则同一个 ID 的旧状态会把新
        状态盖掉；**镜像重建（6）在入批（7）之前**，因为 `InputBatch.add_request` 要读
        `all_token_ids` 写缓冲。
        """
        # 60 关：ngram_gpu 的历史缓冲按**行**索引，而行号会因为结束/压实/新请求而变，
        # 所以要拿"上一轮的行号映射"来判断哪些行搬了家。上游把它存在 InputBatch 上，
        # 本机在进入 `_update_states` 时快照一份（此时还是上一轮结束后的状态）。
        prev_req_id_to_index = dict(self.input_batch.req_id_to_index)

        # 1) 结束的请求：删镜像 + 删批行（本轮结束的请求，执行侧不再为它保留任何状态）
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.input_batch.remove_request(req_id)
        # 57E：提议者的状态也要按"结束"显式删除（进度 + 随机流）。
        # **这里必须用控制端的结束通知，不能拿"不在本轮 batch 里"当结束**：预算不够没排上、
        # 被抢占等待恢复的请求都不在 batch 里，但它们的状态得留着（205 §4.1/§4.3）。
        # 0-token 的结束清理轮也会走到这里——最后一条请求结束后复用 ID 才不会继承旧进度。
        # （草稿概率 q 存在 `pending_draft_probs` 上、每轮整体重算，所以它不需要按请求清。）
        #
        # 62 关：**自定义提议者不保证有 `remove_requests`**（上游对 `propose` 之外的钩子没有任何
        # 契约，`create_custom_proposer` 只检查 `propose`），所以这里按"有才调"处理。这不是静默
        # 降级：插件本来就没有状态要清，缺这个方法只是"没有可选钩子"，不影响草稿正确性。
        if scheduler_output.finished_req_ids and self.proposer is not None:
            remove_requests = getattr(self.proposer, "remove_requests", None)
            if remove_requests is not None:
                remove_requests(scheduler_output.finished_req_ids)

        # 2) 本轮没被调度的活跃请求：**移出批，但保留 CachedRequestState**
        #    （未调度 ≠ 结束：它可能是被抢占、或者这一轮预算不够）
        scheduled_req_ids = set(scheduler_output.num_scheduled_tokens)
        for req_id in list(self.input_batch.req_id_to_index):
            if req_id not in scheduled_req_ids:
                self.input_batch.remove_request(req_id)

        # 3) 新请求：建镜像（含按请求建的 generator）
        reqs_to_add: list[CachedRequestState] = []
        for data in scheduler_output.scheduled_new_reqs:
            if data.req_id in self.requests:
                raise RuntimeError(
                    f"{data.req_id!r} 已经在执行端镜像里，却又出现在 scheduled_new_reqs："
                    f"同一个 ID 的新请求必须等旧状态清理完之后才能再出现")
            state = CachedRequestState(
                req_id=data.req_id,
                prompt_token_ids=data.prompt_token_ids,
                sampling_params=data.sampling_params,
                generator=self._make_generator(data.sampling_params),
                block_ids=data.block_ids,
                num_computed_tokens=data.num_computed_tokens,
            )
            self.requests[data.req_id] = state
            reqs_to_add.append(state)

        # 4/5/6) 续跑请求：校正进度、裁剪未提交输出、更新块表、（必要时）重建镜像
        cached = scheduler_output.scheduled_cached_reqs
        for index, req_id in enumerate(cached.req_ids):
            state = self.requests[req_id]
            # 4) 进度以**协议**为准（执行侧不自增）
            state.num_computed_tokens = cached.num_computed_tokens[index]
            num_output_tokens = cached.num_output_tokens[index]
            row_index = self.input_batch.req_id_to_index.get(req_id)
                        # 70 关（异步调度）：协议的进度里含**乐观预留**的行，而执行侧的 positions、
            # ready 判据、提议起点都必须建立在**已确认历史**上——所以这里校正一次：
            # 只要这条请求还有占位没兑现（协议报的输出数 > 执行侧已提交的输出数），行起点就是
            # "最后那个已确认 token 的位置" = `num_tokens_no_spec - 1`（与同步路径的不变量同形）。
            # 上游在 GPU 侧做同一件事（`update_num_computed_tokens_for_batch_change`），
            # 本仓库的输入组装在 CPU 上，所以用 CPU 镜像校正——注意这一步能成立，是因为
            # 上一轮的异步结果已经在 `execute_model()` 入口结清（镜像里的历史是最新的）。
            if self.async_scheduling:
                has_pending_placeholders = (
                    num_output_tokens > len(state.output_token_ids)
                    if row_index is not None else False)
                if has_pending_placeholders:
                    state.num_computed_tokens = max(
                        self.input_batch.num_tokens_no_spec[row_index] - 1, 0)
            if num_output_tokens < len(state.output_token_ids):
                # 控制端认为只提交了这么多：未提交的尾部（如 EOS 之后被截掉的候选）要丢掉
                del state.output_token_ids[num_output_tokens:]
                if row_index is not None:
                    # 缓冲里的对应部分也要退回去，否则下一轮会把丢弃的 token 当输入算进去
                    self.input_batch.num_tokens_no_spec[row_index] = (
                        len(state.prompt_token_ids) + num_output_tokens)

            new_block_ids = cached.new_block_ids[index]
            resumed = req_id in cached.resumed_req_ids
            if resumed:
                # 5) resumed：**整表替换**，旧物理编号一个都不留
                if new_block_ids is None:
                    raise RuntimeError(f"{req_id!r} 是 resumed 请求，协议必须给整张块表")
                state.block_ids = [list(group) for group in new_block_ids]
            elif new_block_ids is not None:
                # 普通续跑：把**新增**块接到后面
                for group, added in zip(state.block_ids, new_block_ids):
                    group.extend(added)

            if row_index is not None:
                # 请求已经在批里：**镜像也要跟着协议改**——进度与块表两样都要。
                # （异步下上面可能已经把 `state.num_computed_tokens` 校正过，这里写的是校正值）
                #
                # 进度用协议值覆盖（不是自增）：`positions` 就是拿这个值算的，它必须是
                # "本轮开始前算到哪"。漏了这一步的话镜像会一直停在创建时的值，
                # 于是 `seq_lens` 算小、永远判不出 ready，整条请求原地空转。
                self.input_batch.num_computed_tokens_cpu[row_index] = state.num_computed_tokens
                # 块表也要改（只改 CachedRequestState 是不够的，模型读的是镜像）：不做的话，
                # 请求长到新块之后 slot_mapping 会算出一个镜像里不存在的块号，
                # `BlockTable.compute_slot_mapping` 当场报越界。本关只有一个 KV group，取第 0 组。
                if resumed:
                    self.input_batch.block_table.set_row(row_index, state.block_ids[0])
                elif new_block_ids is not None:
                    self.input_batch.block_table.append_to_row(row_index, new_block_ids[0])

            if row_index is None:
                # 6) 它不在批里（上一轮没被调度 / 刚从抢占恢复）：镜像要重建。
                #    协议只在"上一轮没被调度"时带 all_token_ids——缺了就是协议违约。
                all_token_ids = cached.all_token_ids.get(req_id)
                if all_token_ids is None:
                    raise RuntimeError(
                        f"{req_id!r} 不在执行端的批里，重建镜像需要完整 token 历史，"
                        f"但协议没带 all_token_ids（Scheduler 只在'上一轮没调度'时才带）")
                state.output_token_ids = list(all_token_ids)[len(state.prompt_token_ids):]
                reqs_to_add.append(state)

        # 7) 入批（紧凑行）并重排
        for state in reqs_to_add:
            self.input_batch.add_request(state)
        self.input_batch.condense()

        # 不变量：本轮被调度的请求**都在**批里，且批里没有 0 token 的行。
        # 这条断言依赖"同一个 ID 不会在结束清理的同时又被调度"——本关的 Scheduler 禁止
        # 在清理消息送出去之前复用 ID（vLLM 靠代际编号允许复用，那种情况下这里要放宽）。
        self._resumed_req_ids = set(scheduler_output.scheduled_cached_reqs.resumed_req_ids)

        # 投机：把本轮采用的草稿写进输入缓冲（在**插块表/进度之后**做，因为要按
        # "已提交历史"的位置写；协议里没带的请求会被清空）。
        #
        # 草稿的**值**来自协议（`scheduled_spec_decode_tokens`）。异步调度下协议里会是
        # 占位符（-1）、值由执行侧补——那条路要求提议器与输入组装都在 GPU 侧，本仓库还没接线，
        # 所以在配置期就拒绝了 `async_scheduling=True` + 投机（见 `config.resolve_async_scheduling`）。
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id in scheduled_req_ids:
            self.input_batch.update_req_spec_token_ids(req_id, spec_decode_tokens)
        self._resolved_spec_decode_tokens = spec_decode_tokens

        # 60 关：批已经稳定（增删/压实都做完）→ 增量维护 ngram_gpu 的显存历史。
        # 顺序与上游一致（上游也在 `_update_states` 末尾做）。搬运规则见
        # `ngram_proposer_gpu.update_ngram_gpu_tensors_incremental`。
        if self.speculative_config is not None and self.speculative_config.use_ngram_gpu():
            from ..spec_decode.ngram_proposer_gpu import update_ngram_gpu_tensors_incremental

            update_ngram_gpu_tensors_incremental(
                self.input_batch, self.token_ids_gpu_tensor, self.num_tokens_no_spec_gpu,
                new_req_ids={state.req_id for state in reqs_to_add},
                prev_req_id_to_index=prev_req_id_to_index, device=self.device)

        assert set(self.input_batch.req_id_to_index) == scheduled_req_ids, (
            "批里的请求与本轮被调度的请求必须一致："
            f"批里多出 {set(self.input_batch.req_id_to_index) - scheduled_req_ids}，"
            f"少了 {scheduled_req_ids - set(self.input_batch.req_id_to_index)}")

    def _make_generator(self, sampling_params):
        """有 seed 才建 generator：`seed=None` 用全局 RNG（与 vLLM 的 RANDOM 一致）。

        generator 建在**这个设备**上：`torch.multinomial` 要求 generator 与概率张量同设备，
        建错了会在正确性之外的地方炸（"Expected a 'cuda' device type for generator"）。
        """
        if sampling_params.seed is None:
            return None
        generator = torch.Generator(device=self.device)
        generator.manual_seed(sampling_params.seed)
        return generator

    # -------- 一轮：准备输入 --------

    def _prepare_inputs(self, scheduler_output) -> PreparedInputs:
        """把批状态变成模型输入。**全部在 CPU 上算**（198 §4：先用 Torch/CPU 算术，别提前上 kernel）。

        以 198 §4 的两条请求为例（A 起始 computed=3、本轮 2 个；B 起始 0、本轮 3 个，
        block_size=4、块表 A=[7,2] B=[9]）就是：

            input_ids = [A[3], A[4], B[0], B[1], B[2]]      positions = [3, 4, 0, 1, 2]
            query_start_loc = [0, 2, 5]                      seq_lens = [5, 3]
            slot_mapping = [31, 8, 36, 37, 38]               logits_indices = [1, 4]
        """
        num_reqs = self.input_batch.num_reqs
        if num_reqs == 0:
            raise ValueError("批是空的：0 token 的轮次不该走到 _prepare_inputs")
        num_scheduled = [int(scheduler_output.num_scheduled_tokens[req_id])
                         for req_id in self.input_batch.req_ids]
        if any(count <= 0 for count in num_scheduled):
            raise ValueError(f"批里有请求本轮排了 0 个 token：{num_scheduled}；"
                             f"Scheduler 不该把 0 token 的请求放进批")

        # 每行各占几个 token。`req_indices` 把行号按 token 数展开，`query_pos` 是行内序号：
        #   [2, 3] → req_indices=[0,0,1,1,1]，query_pos=[0,1,0,1,2]
        req_indices = torch.repeat_interleave(
            torch.arange(num_reqs), torch.tensor(num_scheduled, dtype=torch.int64))
        query_pos = torch.cat([torch.arange(count, dtype=torch.int64) for count in num_scheduled])

        # 绝对位置 = 该行已算到的位置 + 行内序号（不是"本轮的第几个"！）
        positions = self.input_batch.num_computed_tokens_cpu[:num_reqs][req_indices] + query_pos

        # 取输入 token：把 [max_num_reqs, max_model_len] 的缓冲摊平后按 (行, 位置) 索引
        token_indices = positions + req_indices * self.input_batch.token_ids_cpu.shape[1]
        input_ids = self.input_batch.token_ids_cpu.flatten().index_select(0, token_indices)

        query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int64)
        query_start_loc[1:] = torch.tensor(num_scheduled, dtype=torch.int64).cumsum(0)
        # 本轮之后每请求的上下文长度：进度快照 + 本轮算的
        seq_lens = (self.input_batch.num_computed_tokens_cpu[:num_reqs]
                    + torch.tensor(num_scheduled, dtype=torch.int64))
        slot_mapping = self.input_batch.block_table.compute_slot_mapping(
            num_reqs, positions, req_indices)
        # 每请求最后一个 query 行：它的 hidden 才需要过 LM head
        logits_indices = query_start_loc[1:] - 1
        # 投机批：带草稿的请求要的是**它那 K+1 行**的 logits（b 一行 + K 枚草稿），
        # 不是"最后一行"；整批都没有草稿时（`scheduled_spec_decode_tokens` 为空）走普通路径，
        # 与上游 `use_spec_decode = len(scheduled_spec_decode_tokens) > 0` 同一条判据
        spec_metadata = None
        if self.speculative_config is not None and scheduler_output.scheduled_spec_decode_tokens:
            # 逐请求的草稿数（上游从 `scheduled_spec_decode_tokens` 填同一个数组）
            num_draft_tokens_np = np.zeros(num_reqs, dtype=np.int32)
            for req_id, draft_token_ids in \
                    scheduler_output.scheduled_spec_decode_tokens.items():
                num_draft_tokens_np[self.input_batch.req_id_to_index[req_id]] = \
                    len(draft_token_ids)
            cu_num_scheduled_tokens_np = np.cumsum(
                np.array(num_scheduled, dtype=np.int32))
            spec_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens_np, cu_num_scheduled_tokens_np, input_ids,
                self._resolved_spec_decode_tokens)
            logits_indices = spec_metadata.logits_indices

        # 只对 **ready 行** 采样：算完之后已经追平已知历史（prompt + 已产出）才谈得上"下一个 token"。
        # 中间 prefill 块（chunked prefill 的前几块）虽然也有 hidden，但它们的"下一个 token"
        # 还不该产生（历史本身还没算完）。
        # ready = 算完之后追平了"已提交历史 + 本轮草稿"（没有草稿时就是已提交历史）
        known_tokens = torch.tensor(
            [self.input_batch.num_tokens_with_spec(row) for row in range(num_reqs)],
            dtype=torch.int64)
        ready = seq_lens == known_tokens
        sample_rows = [row for row in range(num_reqs) if bool(ready[row])]

        return PreparedInputs(input_ids=input_ids, positions=positions,
                              query_start_loc=query_start_loc, seq_lens=seq_lens,
                              slot_mapping=slot_mapping, logits_indices=logits_indices,
                              sample_rows=sample_rows, num_scheduled_tokens=num_scheduled,
                              spec_metadata=spec_metadata)

    # -------- 一轮：投机元数据（59 关）--------

    def _get_cumsum_and_arange(self, num_tokens: np.ndarray, arange_out: np.ndarray,
                               cumsum_dtype=None) -> np.ndarray:
        """累积和 + "每段内部从 0 开始"的 arange（上游同名方法）。

        例：`[2, 5, 3]` → 返回 `[2, 7, 10]`，并把 `[0,1,0,1,2,3,4,0,1,2]` 写进
        `arange_out[:10]`（本关的 `arange_out` 是预分配的 `_arange_scratch`）。
        """
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        np.subtract(self._arange_np[:total_num_tokens], cumsums_offsets,
                    out=arange_out[:total_num_tokens])
        return cu_num_tokens

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
        input_ids: torch.Tensor,
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ) -> SpecDecodeMetadata:
        """构造一轮验证的索引（上游同名方法的 3+3 步）。

        ```text
        cu_num_scheduled_tokens: [  4, 104, 107, 207, 209]
        num_draft_tokens:        [  3,   0,   2,   0,   1]
        cu_num_draft_tokens:     [  3,   3,   5,   5,   6]
        logits_indices:          [0,1,2,3, 103, 104,105,106, 206, 207,208]
        target_logits_indices:   [0,1,2, 5,6, 9]
        bonus_logits_indices:    [3, 4, 7,8, 10]
        ```

        **本机与上游的两点差异**（都不改值）：

        1. 上游的草稿 token 从常驻 GPU 输入缓冲取
           （`input_ids.gpu[logits_indices][target_logits_indices + 1]`）；本机还没有那个缓冲
           （69 关 CUDA Graph 才有），所以对**同一份 CPU 输入行**做同样的索引，再随元数据一起上传。
        2. 多一个"输入行 vs 协议"的一致性检查：本机的输入缓冲与 `SchedulerOutput` 是两份状态，
           草稿取错来源（比如缓冲没同步上）会**静默地验证别的 token**，所以这里当场比一遍。
        """
        # 带草稿的请求，query 必须恰好是 K+1 行：`logits_indices` 取的是该请求**最后 K+1 行**，
        # 多出来的行会让 `target+1` 取到别的 token（上游靠调度侧保证，这里显式挡住）
        num_scheduled_per_req = np.diff(
            np.concatenate([[0], cu_num_scheduled_tokens]))
        for index, num_draft in enumerate(num_draft_tokens):
            num_scheduled = int(num_scheduled_per_req[index])
            if num_draft > 0 and num_scheduled != num_draft + 1:
                raise ValueError(
                    f"{self.input_batch.req_ids[index]!r} 本轮排了 {num_scheduled} 个 token，"
                    f"却带了 {num_draft} 枚草稿：带草稿的请求 query 必须恰好是 K+1 行"
                    f"（b + K 枚草稿）。不一致说明调度侧算错了草稿的采用数")

        # Step 1：每请求 K+1 行的累积末端与段内 arange
        num_sampled_tokens = num_draft_tokens + 1
        cu_num_sampled_tokens = self._get_cumsum_and_arange(
            num_sampled_tokens, self._arange_scratch, cumsum_dtype=np.int32)
        # Step 2：每请求取它 query 块的**最后 K+1 行**
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        # Step 3：段内偏移
        logits_indices = logits_indices + self._arange_scratch[:cu_num_sampled_tokens[-1]]
        # bonus 行 = 每请求取完之后的第 K 行（紧凑坐标系）
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # 验证行 = 每请求前 K 行（紧凑坐标系）
        cu_num_draft_tokens = self._get_cumsum_and_arange(
            num_draft_tokens, self._arange_scratch, cumsum_dtype=np.int32)
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
        target_logits_indices = (target_logits_indices
                                 + self._arange_scratch[:cu_num_draft_tokens[-1]])

        # 草稿 token：草稿本来就是**输入行**，取"验证行的下一行"就是它们自己
        logits_indices_t = torch.from_numpy(logits_indices)
        draft_token_ids = input_ids[logits_indices_t][
            torch.from_numpy(target_logits_indices) + 1]

        # 与协议逐请求比一遍（顺序 = batch 行序）
        expected = [token for req_id in self.input_batch.req_ids
                    for token in scheduled_spec_decode_tokens.get(req_id, ())]
        if draft_token_ids.tolist() != expected:
            raise RuntimeError(
                f"输入行里取出的草稿 {draft_token_ids.tolist()} 与本轮协议 "
                f"`scheduled_spec_decode_tokens` 的 {expected} 不一致：输入缓冲与调度快照"
                f"不同步（草稿必须是协议里那几枚，否则验证的不是 q 对应的候选）")

        def to_device(array, dtype=torch.int32):
            return torch.from_numpy(np.ascontiguousarray(array)).to(
                device=self.device, dtype=dtype)

        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids.to(device=self.device, dtype=torch.int32),
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=to_device(cu_num_draft_tokens),
            cu_num_sampled_tokens=to_device(cu_num_sampled_tokens),
            target_logits_indices=to_device(target_logits_indices),
            bonus_logits_indices=to_device(bonus_logits_indices),
            logits_indices=to_device(logits_indices),
        )

    # -------- 69 关：CUDA Graph（键 → 热身 → 捕获 → 分派 → 重放）--------

    def initialize_cudagraph_capture(self) -> None:
        """确定图模式、建键表、包装模型、开静态工作区（对应上游
        `initialize_cudagraph_capture()` + `CudagraphDispatcher.initialize_cudagraph_keys()`）。

        顺序与上游一致，四步：

            1. 环境事实：没有 CUDA 就没有图（退 eager 并记录原因）
            2. **能力协商**：注意力后端的能力档位决定最终模式（`full` 可能被降级）
            3. 建键表（PIECEWISE + FULL 两张），再按模式包装模型：
               含 FULL → 外层包一张全图；含 PIECEWISE → 让模型给每层装两段分段图
            4. 开静态工作区（图里记的是指针，缓冲必须常驻）

        与上游的差异只有一条、而且是环境事实：**本机没有 CUDA 时强制 NONE**。
        `VllmConfig` 是按 `DeviceConfig.device` 字符串解析的，而"字符串写着 cuda"不等于
        "这台机器真的有 CUDA"（跨机器跑同一份配置时就会不同）；真去 `torch.cuda.CUDAGraph()`
        只会在捕获时炸，不如在这里就退成 eager 并**记录下来**。
        """
        mode = self.compilation_config.cudagraph_mode
        if mode != CUDAGraphMode.NONE and not torch.cuda.is_available():
            self.cudagraph_unsupported_reason = (
                "配置要求 CUDA Graph，但这台机器没有可用的 CUDA：退成 eager")
            mode = CUDAGraphMode.NONE
            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
            self.compilation_config.cudagraph_capture_sizes = []
            self.compilation_config.max_cudagraph_capture_size = 0
        else:
            self.cudagraph_unsupported_reason = None
        # 能力协商（对应上游 `_check_and_update_cudagraph_mode`）：**注意力后端先开口**，
        # 再决定最终模式。为什么必须在图键初始化之前：`FULL` 的含义是"混合批也录全图"，
        # 而本仓库的图内注意力只认"每请求行数相同"的批（能力档位 UNIFORM_BATCH），
        # 所以请求 full 会被降级成 FULL_AND_PIECEWISE（混合批改走分段图）并打 warning。
        if mode != CUDAGraphMode.NONE:
            builder_cls = type(self.attn_metadata_builder)
            min_support = builder_cls.get_cudagraph_support(self.vllm_config)
            mode = self.vllm_config.resolve_cudagraph_mode_and_sizes(
                min_support, builder_cls.__name__, self.uniform_decode_query_len)
        # 命中统计（上游 `CUDAGraphStat`/`CUDAGraphLogging`）：**每一轮**都记一条，
        # 包括"这一轮没走图"（模式 NONE）——验收要能回答"为什么没走图"，而只记命中时
        # 这个问题的答案恰好缺失。
        self.cudagraph_logging = CUDAGraphLogging(
            mode, self.compilation_config.cudagraph_capture_sizes)
        self.cudagraph_dispatcher.initialize_cudagraph_keys(
            mode, self.uniform_decode_query_len)
        # draft 侧单独一套键（上游 `self.drafter.initialize_cudagraph_keys(cudagraph_mode)`）：
        # 草稿模型的图属 74 关，本仓库的 drafter 恒为 NONE。
        if self.proposer is not None:
            initialize_keys = getattr(self.proposer, "initialize_cudagraph_keys", None)
            if initialize_keys is not None:
                initialize_keys(mode)
        if mode != CUDAGraphMode.NONE:
            self._init_padded_buffers()
            # 图包装器只包模型前向（`compute_logits` 留在图外：它只在"要采样的行"上做，
            # 行数每轮不同，属于动态索引）。上游同款：`self.model = CUDAGraphWrapper(...)`。
            # 只有模式里**含 FULL** 时才包外层全图；纯 PIECEWISE 时外层不包（每层两段各自进图）。
            if mode.has_full_cudagraphs():
                self.model = CUDAGraphWrapper(self.model, self.vllm_config,
                                              CUDAGraphMode.FULL)
            if mode.has_piecewise_cudagraphs():
                # 分段图：让模型把每层切成"注意力前段 / 后段"并各包一个 PIECEWISE 包装器。
                # 段间缓冲按**输入工作区容量**（max_num_batched_tokens）开一次，此后不重建：
                # 它服务的是"所有模式"（超过最大档位的批会回退 eager，但那时这些缓冲照样要用），
                # 所以不能用"图档位上限"当容量——实测那样会在第 6 行就写不下（档位上限才 4）。
                self.model.enable_piecewise_pieces(
                    self.vllm_config,
                    self.vllm_config.scheduler_config.max_num_batched_tokens)

    def _init_padded_buffers(self) -> None:
        """静态输入工作区（对应上游 `_init_input_buffers()` 里与图相关的那几块）。

        容量口径全部来自既有配置（AGENTS §8：**不加"调大一点"的旋钮**）：

            input_ids / positions / slot_mapping   max_num_batched_tokens 行
            query_start_loc / seq_lens             max_num_seqs (+1) 行
            block_table                            复用 InputBatch 的块表镜像（本来就去定长）

        `slot_mapping` 初值填 `PADDING_SLOT_ID(-1)`：没被覆盖到的行**必须是哨兵**，
        不能是 0（0 是真实槽位，会往 0 号块/第一块写垃圾）。
        """
        max_num_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.input_batch.max_num_reqs
        device = self.device
        self._padded_buffers = {
            "input_ids": torch.zeros(max_num_tokens, dtype=torch.int64, device=device),
            "positions": torch.zeros(max_num_tokens, dtype=torch.int64, device=device),
            "slot_mapping": torch.full((max_num_tokens,), PADDING_SLOT_ID,
                                       dtype=torch.int64, device=device),
            "query_start_loc": torch.zeros(max_num_reqs + 1, dtype=torch.int64, device=device),
            "seq_lens": torch.zeros(max_num_reqs, dtype=torch.int64, device=device),
        }

    @staticmethod
    def _is_uniform_decode(max_num_scheduled_tokens: int, uniform_decode_query_len: int,
                           num_tokens: int, num_reqs: int,
                           force_uniform_decode: bool | None = None) -> bool:
        """"所有请求的 query 长度都一样"的批（上游同名静态方法，逐字对齐）。

        两个条件缺一不可：最长请求恰好 `1+K` 行，且总行数 = `(1+K) × 请求数`。
        只看第一个的话，"一条请求 K+1 行、另一条 1 行"也会被误判成统一批——那样的批
        行数不是请求数的整数倍，图的形状根本对不上。
        """
        return (((max_num_scheduled_tokens == uniform_decode_query_len)
                 and (num_tokens == max_num_scheduled_tokens * num_reqs))
                if force_uniform_decode is None else force_uniform_decode)

    def _determine_batch_execution_and_padding(
            self, num_tokens: int, num_reqs: int, num_scheduled_tokens: list[int],
            max_num_scheduled_tokens: int, force_eager: bool = False,
            force_uniform_decode: bool | None = None,
    ) -> tuple[CUDAGraphMode, BatchDescriptor]:
        """真实形状 → `(运行模式, 补齐后的图键)`（上游同名方法的本仓库子集）。

        上游这里还有 cascade attention、DP 协调、LoRA 计数、microbatch —— 那几样本仓库都没有
        （单卡、无 LoRA、无 cascade），所以只留下"判统一 decode → 问分派器"这两步。
        返回的键里的 `num_tokens` 是**补齐后**的，调用方必须按它准备输入。
        """
        uniform_decode = self._is_uniform_decode(
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            uniform_decode_query_len=self.uniform_decode_query_len,
            num_tokens=num_tokens, num_reqs=num_reqs,
            force_uniform_decode=force_uniform_decode)
        mode, batch_descriptor = self.cudagraph_dispatcher.dispatch(
            num_tokens=num_tokens, uniform_decode=uniform_decode,
            valid_modes={CUDAGraphMode.NONE} if force_eager else None)
        return mode, batch_descriptor

    def _record_selection(self, mode: CUDAGraphMode, batch_descriptor: BatchDescriptor,
                          num_tokens: int, num_reqs: int) -> None:
        """记下这一轮**真实**选中的模式与键（验收要求："记录真实选中的 mode/key"）。

        不是日志：测试与 profiler 直接读这个列表，所以它必须由分派的那一处写、别处不写。
        """
        self.cudagraph_selections.append({
            "num_tokens": num_tokens, "num_reqs": num_reqs,
            "padded_tokens": batch_descriptor.num_tokens,
            "padded_reqs": batch_descriptor.num_reqs,
            "uniform": batch_descriptor.uniform,
            "mode": str(mode), "key": str(batch_descriptor),
        })
        # 上游在 `_determine_batch_execution_and_padding()` 里建这条记录、由调用方 observe；
        # 本仓库把两者放在这一处（同一个写入点），数值口径完全一样。
        if self.cudagraph_logging is not None:
            self.cudagraph_logging.observe(CUDAGraphStat(
                num_unpadded_tokens=num_tokens,
                num_padded_tokens=batch_descriptor.num_tokens,
                num_paddings=batch_descriptor.num_tokens - num_tokens,
                runtime_mode=str(mode)))

    def _padded_attn_metadata(self, batch_descriptor: BatchDescriptor):
        """取（或第一次建）这个图键的 attention 元数据对象。

        对象只建一次，因为它的张量必须是**静态缓冲本身**：每轮只改缓冲内容，图重放时按同一批
        地址去读。每轮新建 metadata 会让图读到旧地址（不报错、只是算错）。
        """
        metadata = self._padded_metadata.get(batch_descriptor)
        if metadata is None:
            raise RuntimeError(
                f"图键 {batch_descriptor} 没有对应的元数据：说明它在 capture_model() 之前就被"
                f"分派到了。图必须**先在捕获窗口里建好**（含元数据），运行时只重放")
        return metadata

    def _build_padded_attn_metadata(self, batch_descriptor: BatchDescriptor,
                                    for_cudagraph_capture: bool = False,
                                    num_tokens_padded: int | None = None,
                                    num_reqs_padded: int | None = None,
                                    uniform_query_len: int | None = None):
        """按图键建一份元数据（张量全部指向静态工作区）。

        默认按**图键**（补齐后的形状）建；热身（mode=NONE）那一次按**未补齐**形状建，
        因为那一轮根本不走图（上游 `dummy_run` 里 `pad_attn = mode == FULL` 是同一个意思）。
        """
        buffers = self._padded_buffers
        if num_tokens_padded is None:
            num_tokens_padded = batch_descriptor.num_tokens
        if num_reqs_padded is None:
            num_reqs_padded = batch_descriptor.num_reqs
            if num_reqs_padded is None:
                raise ValueError("本仓库只有 FULL 图，键里必须有精确的 num_reqs")
        if uniform_query_len is None:
            uniform_query_len = (self.uniform_decode_query_len
                                 if batch_descriptor.uniform else None)
        if uniform_query_len is not None and \
                uniform_query_len * num_reqs_padded != num_tokens_padded:
            raise ValueError(
                f"统一 decode 的行数对不上：{uniform_query_len} × {num_reqs_padded} != "
                f"{num_tokens_padded}（键自己矛盾，不可能有这张图）")
        metadata = self.attn_metadata_builder.build(
            query_start_loc=buffers["query_start_loc"][:num_reqs_padded + 1],
            seq_lens=buffers["seq_lens"][:num_reqs_padded],
            block_table=self.input_batch.block_table.gpu[:num_reqs_padded],
            slot_mapping=buffers["slot_mapping"][:num_tokens_padded],
            num_reqs=num_reqs_padded, uniform_query_len=uniform_query_len)
        if for_cudagraph_capture:
            self.attn_metadata_builder.build_for_cudagraph_capture(metadata)
        return metadata

    def _fill_padded_buffers(self, num_tokens: int, num_reqs: int,
                             num_scheduled_tokens: list[int] | None,
                             num_tokens_padded: int, num_reqs_padded: int,
                             uniform_query_len: int | None,
                             source: PreparedInputs | None = None) -> None:
        """把这一轮的真实输入写进静态缓冲（图内路径的"更新数据"这一步）。

        三条规则，都是"图上不能有分支"逼出来的：

            padding 行的槽位一律 `PADDING_SLOT_ID(-1)`（图里 clamp 到 0 号垃圾桶）
            padding 行的 token/position 写成确定的 0（它们**会被 embedding/RoPE 查表读到**，
                超出词表/位置表长度就是 device 端越界；不能靠"上一轮的残值恰好合法"）
            padding 请求的 `seq_lens = 0`（它的 attention 整行被掩掉，输出是垃圾但有限）
            padding 请求的块表行**清零**（指向 0 号块，绝不能留着上一轮的块号）

        `source=None` 是 dummy_run（热身/捕获）：token/position 填 0 即可，槽位仍是哨兵
        （上游注释同款："dummy runs have no real slot assignments — fill with -1 so the
        cache kernels skip the KV write"）。
        """
        buffers = self._padded_buffers
        device = self.device
        # 先把**整段**槽位打成哨兵，再覆盖真实行：顺序反过来的话，补齐的行会留着上一轮
        # 的槽位（可能是别人的真实槽位）——那是不报错的覆盖。
        buffers["slot_mapping"][:num_tokens_padded].fill_(PADDING_SLOT_ID)
        if source is not None:
            buffers["input_ids"][:num_tokens].copy_(source.input_ids.to(device))
            buffers["positions"][:num_tokens].copy_(source.positions.to(device))
            buffers["slot_mapping"][:num_tokens].copy_(source.slot_mapping.to(device))
            buffers["seq_lens"][:num_reqs].copy_(source.seq_lens.to(device))
            buffers["query_start_loc"][:num_reqs + 1].copy_(
                source.query_start_loc.to(device))
            # 补齐的**行**也要把 token/位置写成确定的 0。它们的结果会被丢掉、槽位也是哨兵，
            # 但内容**会被模型查表读到**：embedding 按 input_id 索引、RoPE 按 position 索引，
            # 超出词表/位置表长度就是 device 端越界（实测 `device-side assert triggered`）。
            # 留上一轮的残值在正常路径里恰好都合法（都是自己写进去的真 token/真位置），
            # 但"恰好合法"不是不变量——这里显式写死，让补齐行不依赖上一轮发生过什么。
            buffers["input_ids"][num_tokens:num_tokens_padded].zero_()
            buffers["positions"][num_tokens:num_tokens_padded].zero_()
        else:
            # dummy：位置连续、token 全 0（只求形状对；输出会被丢掉）
            buffers["positions"][:num_tokens].copy_(
                torch.arange(num_tokens, dtype=torch.int64, device=device))
            buffers["input_ids"][:num_tokens].zero_()
            if uniform_query_len is not None:
                cumsum = torch.arange(num_reqs + 1, dtype=torch.int64,
                                      device=device) * uniform_query_len
            else:
                cumsum = torch.tensor(
                    [0, *list(__import__("itertools").accumulate(num_scheduled_tokens))],
                    dtype=torch.int64, device=device)
            buffers["query_start_loc"][:num_reqs + 1].copy_(cumsum)
            buffers["seq_lens"][:num_reqs].fill_(1)
        # 补齐的行/请求
        if uniform_query_len is not None:
            buffers["query_start_loc"][num_reqs + 1:num_reqs_padded + 1].fill_(
                num_tokens)
        else:
            buffers["query_start_loc"][num_reqs + 1:num_reqs_padded + 1].fill_(num_tokens)
        buffers["seq_lens"][num_reqs:num_reqs_padded].zero_()
        # 块表：padding 请求的行必须清零（0 号块 = 垃圾桶；留旧块号会读到别人的 KV）
        self.input_batch.block_table.cpu[num_reqs:num_reqs_padded].zero_()
        self.input_batch.block_table.commit_block_table(num_reqs_padded)

    def _piecewise_attn_metadata(self, num_reqs: int, num_tokens_padded: int):
        """分段图这一轮的元数据（**每轮新建**，不缓存）。

        为什么与 FULL 相反、可以不缓存：分段图里**注意力在图外**，它的元数据没有任何张量会被
        图读到——所以不必满足"张量必须是静态缓冲"，也不必把 `num_reqs` 定死在键里。
        口径：

            `num_reqs`               = **真实**请求数（不是补齐后的）
            `uniform_query_len=None` → 注意力走逐请求的 eager 路径（图外允许 CPU 同步）
            `slot_mapping`           = **补齐后**的静态缓冲（必须与图输出的行数同形；
                                        padding 行是 -1，写 KV 时被掩掉）
        """
        buffers = self._padded_buffers
        return self.attn_metadata_builder.build(
            query_start_loc=buffers["query_start_loc"][:num_reqs + 1],
            seq_lens=buffers["seq_lens"][:num_reqs],
            block_table=self.input_batch.block_table.gpu[:num_reqs],
            slot_mapping=buffers["slot_mapping"][:num_tokens_padded],
            num_reqs=num_reqs, uniform_query_len=None)

    def _prepare_inputs_padded(self, inputs: PreparedInputs, mode: CUDAGraphMode,
                              batch_descriptor: BatchDescriptor) -> None:
        """eager 的紧凑输入 → 图要的补齐输入（对应上游 `prepare_inputs_padded` 的效果）。

        两种模式的**补齐口径不同**，这是 PIECEWISE 与 FULL 的关键区别：

            FULL       token 行与**请求行**都补齐（图里的注意力按 `num_reqs` 定形状）
            PIECEWISE  只补 token 行；请求行保持真实数（注意力在图外，按真实请求数算）

        与上游的差别要说清楚：上游的 padding 发生在**草稿第一遍**（drafter 侧，
        `SpecDecodeBaseProposer.prepare_inputs_padded()` 用 Triton kernel 算
        `token_indices_to_sample` / `num_rejected_tokens_gpu`），target 侧是按"整批 K+1 行"
        直接准备的。本仓库的紧凑布局在**统一 decode 批**下已经与"整批 K+1 行"逐行相同
        （每请求恰好 `1+K` 行连续排布），所以这里只需要在**尾部**补上档位差：
        补齐的行槽位是哨兵、补齐的请求 seq_len=0 → 既不改动真实行的索引，
        也不需要动采样/掩码/logprobs 的行映射（它们是这个前缀的子集，见
        `tests/step69/test_spec_cudagraph.py::test_padded_rows_extend_the_compact_layout`）。
        """
        if mode == CUDAGraphMode.PIECEWISE:
            num_reqs_padded = self.input_batch.num_reqs
            uniform_query_len = None
        else:
            num_reqs_padded = batch_descriptor.num_reqs
            uniform_query_len = (self.uniform_decode_query_len
                                 if batch_descriptor.uniform else None)
        self._fill_padded_buffers(inputs.num_tokens, self.input_batch.num_reqs,
                                  inputs.num_scheduled_tokens,
                                  batch_descriptor.num_tokens, num_reqs_padded,
                                  uniform_query_len, source=inputs)
        if mode == CUDAGraphMode.FULL and batch_descriptor not in self._padded_metadata:
            # 正常路径不会走到这里（键要么在 capture_model 里建过，要么分派器不会返回它）。
            # 走到这里说明"图没建好就想重放"：明确报错，不要顺手补一个（那会带着旧地址建图）。
            raise RuntimeError(
                f"图键 {batch_descriptor} 还没在捕获窗口里建过元数据："
                f"请先跑 capture_model()（Worker.compile_or_warm_up_model()）")

    def _dummy_run(self, num_tokens: int, uniform_decode: bool = False,
                   cudagraph_runtime_mode: CUDAGraphMode | None = None,
                   is_graph_capturing: bool = False) -> torch.Tensor:
        """跑一次"假"前向：用来热身、也用来**在捕获窗口里录图**（上游同名方法的子集）。

        与上游一致的三点：

        1. 形状由分派器决定（`force_uniform_decode=uniform_decode`），不是自己拼；
        2. 捕获时把 `seq_lens` 全填 1（见 `AttentionMetadataBuilder.build_for_cudagraph_capture`）；
        3. 走的是**和真实路径同一个** `_run_model_padded`，不是"在别处重建一套前向"
           ——否则录下来的图与真实路径的算子/顺序不一致，"捕获成功"就没有意义。
        """
        if uniform_decode:
            uniform_query_len = self.uniform_decode_query_len
            num_reqs = min(self.input_batch.max_num_reqs,
                           -(-num_tokens // uniform_query_len))
            num_scheduled_tokens = [uniform_query_len] * num_reqs
            if num_tokens % uniform_query_len != 0:
                num_scheduled_tokens[-1] = num_tokens % uniform_query_len
        else:
            num_reqs = min(num_tokens, self.input_batch.max_num_reqs)
            per_req = num_tokens // num_reqs
            num_scheduled_tokens = [per_req] * num_reqs
            num_scheduled_tokens[-1] += num_tokens % num_reqs
        if sum(num_scheduled_tokens) != num_tokens:
            raise RuntimeError("dummy_run 的 token 分配与 num_tokens 不一致")
        max_num_scheduled_tokens = max(num_scheduled_tokens)

        mode, batch_descriptor = self._determine_batch_execution_and_padding(
            num_tokens=num_tokens, num_reqs=num_reqs,
            num_scheduled_tokens=num_scheduled_tokens,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            force_eager=(cudagraph_runtime_mode == CUDAGraphMode.NONE),
            force_uniform_decode=uniform_decode)
        if cudagraph_runtime_mode is None:
            cudagraph_runtime_mode = mode
        elif cudagraph_runtime_mode != mode:
            raise RuntimeError(
                f"dummy_run 期望的模式 {cudagraph_runtime_mode} 与分派器给的 {mode} 不一致："
                f"热身/捕获必须与真实路径选到同一张图，否则热身白做")

        uniform_query_len = (self.uniform_decode_query_len
                             if batch_descriptor.uniform else None)
        if cudagraph_runtime_mode == CUDAGraphMode.NONE:
            # 热身：**不补齐**（上游 `pad_attn = mode == FULL` 同义），跑的是未补齐的形状，
            # 元数据临时建、不占用图键——它是"把算子和 handle 热起来"，不是"建图"。
            self._fill_padded_buffers(num_tokens, num_reqs, num_scheduled_tokens,
                                      num_tokens, num_reqs, uniform_query_len)
            metadata = self._build_padded_attn_metadata(
                batch_descriptor, num_tokens_padded=num_tokens,
                num_reqs_padded=num_reqs, uniform_query_len=uniform_query_len)
        elif cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            # 分段图：只补 token 行，请求行保持真实数；元数据每轮新建（注意力在图外）
            self._fill_padded_buffers(num_tokens, num_reqs, num_scheduled_tokens,
                                      batch_descriptor.num_tokens, num_reqs,
                                      uniform_query_len)
            metadata = self._piecewise_attn_metadata(num_reqs,
                                                     batch_descriptor.num_tokens)
        else:
            num_reqs_padded = batch_descriptor.num_reqs or num_reqs
            self._fill_padded_buffers(num_tokens, num_reqs, num_scheduled_tokens,
                                      batch_descriptor.num_tokens, num_reqs_padded,
                                      uniform_query_len)
            if batch_descriptor not in self._padded_metadata:
                self._padded_metadata[batch_descriptor] = self._build_padded_attn_metadata(
                    batch_descriptor, for_cudagraph_capture=is_graph_capturing)
            metadata = self._padded_metadata[batch_descriptor]
        padded = _PaddedRun(batch_descriptor=batch_descriptor, mode=cudagraph_runtime_mode,
                            num_tokens=metadata.slot_mapping.shape[0])
        hidden = self._run_model_padded(padded, metadata=metadata)
        # dummy 前向的产物**不能留在 Runner 上**：`hidden_states` / `aux_hidden_states` /
        # `target_hidden_states` 都是"本轮 target 的事实"，提议者会直接吃它们。留着热身/捕获
        # 那一轮的零张量，会让"还没跑过真实一轮"的状态看起来像有数据（测试当场抓到了这一点），
        # 更糟的是失败路径下可能真的喂给提议者。
        self.aux_hidden_states = None
        self.target_hidden_states = None
        self.attn_metadata = None
        return hidden

    def _warmup_and_capture(self, batch_descriptor: BatchDescriptor,
                            cudagraph_runtime_mode: CUDAGraphMode) -> None:
        """热身若干次 → 捕获一次（上游同名方法）。

        为什么要热身：第一次跑某个形状时，cuBLAS 的 handle / workspace、kernel 的 lazy init、
        `torch.empty` 的缓存块都会"在录制过程中"发生。cuBLAS 的 handle **不允许在捕获里创建**
        （实测报 `CUBLAS_STATUS_NOT_INITIALIZED when calling cublasCreate(handle)`，随后整个
        捕获流被作废），所以捕获前至少要在这个（形状/权重/流都相同的）上下文里真跑过一次。

        上游的 `cudagraph_num_of_warmups` 默认 0，是因为它在此之前已经做过 profile run
        （定容量那一步就是若干次 `_dummy_run(mode=NONE)`）；本仓库没有显存 profiling，
        于是把"至少一次 eager 前向"作为捕获的**必需前置**：`max(1, cudagraph_num_of_warmups)`。
        这不改变图的语义，只保证捕获能成功。
        """
        num_warmups = max(1, self.compilation_config.cudagraph_num_of_warmups)
        for _ in range(num_warmups):
            self._dummy_run(batch_descriptor.num_tokens,
                            uniform_decode=batch_descriptor.uniform,
                            cudagraph_runtime_mode=CUDAGraphMode.NONE)
        torch.cuda.synchronize()
        self._dummy_run(batch_descriptor.num_tokens,
                        uniform_decode=batch_descriptor.uniform,
                        cudagraph_runtime_mode=cudagraph_runtime_mode,
                        is_graph_capturing=True)

    def capture_model(self) -> dict:
        """按分派器的键表逐张捕获（上游 `capture_model()` 的子集）。

        **大档位先捕获**（`get_capture_descs()` 已排好序）：小图能复用大图占下的显存池。
        捕获窗口由 `monitor.set_cudagraph_capturing_enabled()` 划定；窗口之外任何"顺手捕获"
        都会被 `validate_cudagraph_capturing_enabled()` 拦下（那会让服务中的某一步突然卡顿）。
        """
        import time

        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
            return {"mode": "NONE", "captured": 0, "capture_seconds": 0.0}
        start = time.perf_counter()
        torch.cuda.synchronize()
        start_free = torch.cuda.mem_get_info()[0]
        set_cudagraph_capturing_enabled(True)
        captured = 0
        try:
            with graph_capture(self.device):
                for runtime_mode, batch_descriptors in \
                        self.cudagraph_dispatcher.get_capture_descs():
                    for batch_descriptor in batch_descriptors:
                        self._warmup_and_capture(batch_descriptor, runtime_mode)
                        captured += 1
                        torch.cuda.synchronize()
        finally:
            # 无论捕获成功与否都要**关掉窗口**：开着的话，后面每一次"没见过的形状"都会
            # 顺手录一张图（延迟尖峰 + 显存悄悄涨）
            set_cudagraph_capturing_enabled(False)
        torch.cuda.synchronize()
        end_free = torch.cuda.mem_get_info()[0]
        self.cudagraph_capture_stats = {
            "mode": str(self.compilation_config.cudagraph_mode),
            "captured": captured,
            "capture_seconds": time.perf_counter() - start,
            "graph_pool_bytes": start_free - end_free,
            "capture_sizes": list(self.compilation_config.cudagraph_capture_sizes or []),
        }
        return self.cudagraph_capture_stats

    def _run_model_padded(self, padded: "_PaddedRun", metadata=None,
                          inputs: PreparedInputs | None = None) -> torch.Tensor:
        """图分派路径的前向：静态缓冲 + 固定的 metadata + 上下文里的图模式/键。

        与 `_run_model()` 的关系：**同一个模型、同一份 forward 上下文协议**，区别只在
        "输入从哪来"（静态缓冲）与"上下文里写什么"（运行模式 + 键）。图包装器就是靠上下文里
        这两个字段决定捕获/重放的；注意力层也靠运行模式决定走固定形状那条路。

        `metadata` 由调用方给：捕获/重放给的是**存在图键下的那一份**（张量必须是静态缓冲本身，
        每轮只改内容），热身给的是临时建的那一份（NONE 模式不走图，形状也不必补齐）。
        """
        buffers = self._padded_buffers
        num_tokens_padded = metadata.slot_mapping.shape[0]
        layer_names = self._attention_layers()
        attn_metadata = {name: metadata for name in layer_names}
        # 64 关的 cache-only 提议者要拿"本轮 target 的 slot_mapping"，而它按**紧凑行**工作
        # （它的特征缓冲也是紧凑的）。所以在图路径里额外留一份紧凑元数据给它，而不是把
        # 补齐的那份（多出来的行是 padding，喂过去会写错位置/长度对不上）。
        needs_compact_metadata = (inputs is not None
                                  and self.speculative_config is not None
                                  and self.speculative_config.uses_extract_hidden_states())
        if needs_compact_metadata:
            self.attn_metadata = next(iter(self._build_attn_metadata(inputs).values()))
        else:
            self.attn_metadata = None
        input_ids = buffers["input_ids"][:num_tokens_padded]
        positions = buffers["positions"][:num_tokens_padded]
        with set_forward_context(
                attn_metadata, num_tokens=num_tokens_padded,
                cudagraph_runtime_mode=padded.mode,
                batch_descriptor=padded.batch_descriptor,
                slot_mapping={"slot_mapping": metadata.slot_mapping}):
            if self.capture_aux_hidden_states:
                hidden_states, aux = self.model(input_ids, positions)
                self.aux_hidden_states = torch.cat(list(aux), dim=-1)
            else:
                hidden_states = self.model(input_ids, positions)
        self.target_hidden_states = hidden_states
        return hidden_states

    # -------- 一轮：跑模型 --------

    def _run_model(self, inputs: PreparedInputs) -> torch.Tensor:
        """上传输入 → 建 metadata → 设 forward 上下文 → 跑模型，返回 hidden states。

        数值测试会直接用它（拿全部位置的 hidden），生产路径经由 `execute_model()`。
        """
        device = self.device
        input_ids = inputs.input_ids.to(device)
        positions = inputs.positions.to(device)
        attn_metadata = self._build_attn_metadata(inputs)
        with set_forward_context(attn_metadata, num_tokens=inputs.num_tokens):
            if self.capture_aux_hidden_states:
                hidden_states, aux = self.model(input_ids, positions)
                # 多个辅助层拼在最后一维（draft 的 `combine_hidden_states` 按同样的顺序切块）
                self.aux_hidden_states = torch.cat(list(aux), dim=-1)
            else:
                hidden_states = self.model(input_ids, positions)
        # 65 关：MTP 要的就是这份"本轮每个 query 行的最后一层 hidden"（上游
        # `target_hidden_states = hidden_states[:total_num_tokens]`，因为它的
        # `use_aux_hidden_state_outputs` 是 False）。留引用不复制。
        self.target_hidden_states = hidden_states
        return hidden_states

    def _build_attn_metadata(self, inputs: PreparedInputs) -> dict:
        """一个 KV group 一份 metadata，同 group 的层**共用同一个对象**（vLLM 也这么分）。

        层 → metadata 的对应关系全靠层名：模型层不知道块表怎么排，Runner 也不知道某一层
        "特殊在哪"。
        """
        num_reqs = int(inputs.query_start_loc.shape[0]) - 1
        self.input_batch.block_table.commit_block_table(num_reqs)
        metadata = self.attn_metadata_builder.build(
            query_start_loc=inputs.query_start_loc.to(self.device),
            seq_lens=inputs.seq_lens.to(self.device),
            block_table=self.input_batch.block_table.gpu[:num_reqs],
            slot_mapping=inputs.slot_mapping.to(self.device),
            num_reqs=num_reqs)
        # 64 关：本轮这份元数据留一个引用给 cache-only 提议者（它要的就是里面的 `slot_mapping`：
        # 特征必须写进**与 target KV 相同的槽位**，"同源"是这一关的正确性前提）。
        self.attn_metadata = metadata
        return {layer_name: metadata for layer_name in self._attention_layers()}

    # -------- 执行侧的两步协议 --------

    @torch.inference_mode()
    def execute_model(self, scheduler_output, non_block: bool = False):
        """第一步：合并状态、跑模型、存下 logits。返回 `None` 表示"等 sample_tokens"。

        `non_block`（70 关，异步调度）：本仓库不在这里分叉——CUDA 的 kernel 提交本来就是
        异步的，"不阻塞"由上层体现（executor 把返回值包成 Future、`sample_tokens` 交异步
        句柄）。参数收下是为了接口与上游三层同参，不是装饰。

        **`torch.inference_mode()` 不是装饰性的**（vLLM 在同样的位置也有这个装饰器）：模型的
        参数默认 `requires_grad=True`，而 KV 写入是 `index_copy_`——没有这个边界的话，每次
        写入都会记一个 `CopySlices` 反向图挂在 KV 缓存上，并且**一步一步累积**（每一步多 30 个
        图节点）。推理引擎里这意味着显存随步数单调上涨（验收方的独立探针就是查这个）。

        为什么不在 `load_model` 上也加：那会让权重本身变成"推理张量"，而这些权重还要被
        测试里的直接前向用到；本关只在**每步的入口**（执行与采样）划这条线。
        """
        self._check_usable()
        # 70 关：先把上一轮欠下的异步结果结清（见 `wait_for_pending_async_output` 的说明）。
        # 放在最前面：`_update_states()` 要按镜像里"上一轮产出的 token"校正缓冲。
        self.wait_for_pending_async_output()
        if self.execute_model_state is not None:
            raise RuntimeError(
                "上一轮 execute_model() 的结果还没被 sample_tokens() 消费，不能开始新的一轮："
                "覆盖它会让上一轮的 logits 与这一轮的请求对不上")
        self._update_states(scheduler_output)
        if scheduler_output.total_num_scheduled_tokens == 0:
            # 空轮（只有结束清理）：不碰模型
            return ModelRunnerOutput.make_empty()
        if self.model is None or self.sampler is None:
            raise RuntimeError("Runner 还没 load_model()")

        inputs = self._prepare_inputs(scheduler_output)
        # 69 关：先问分派器"这一轮能不能走图、走哪张图"。它返回的 num_tokens 是**补齐后**的，
        # 下面准备输入与设上下文都要按它来。模式 NONE = 这一轮 eager（形状不匹配 / 没建图 /
        # 关掉了图），这也是上游 `dispatch()` 的规则，不是"我们悄悄降级"。
        mode, batch_descriptor = self._determine_batch_execution_and_padding(
            num_tokens=inputs.num_tokens, num_reqs=self.input_batch.num_reqs,
            num_scheduled_tokens=inputs.num_scheduled_tokens,
            max_num_scheduled_tokens=max(inputs.num_scheduled_tokens))
        self._record_selection(mode, batch_descriptor, inputs.num_tokens,
                               self.input_batch.num_reqs)
        try:
            if mode == CUDAGraphMode.NONE:
                hidden_states = self._run_model(inputs)
            else:
                self._prepare_inputs_padded(inputs, mode, batch_descriptor)
                # FULL 用**缓存**在键下的那份元数据（张量必须是静态缓冲）；PIECEWISE 每轮
                # 新建（注意力在图外，没有"图读元数据"这回事）。
                metadata = (self._piecewise_attn_metadata(
                                self.input_batch.num_reqs, batch_descriptor.num_tokens)
                            if mode == CUDAGraphMode.PIECEWISE
                            else self._padded_attn_metadata(batch_descriptor))
                hidden_states = self._run_model_padded(
                    _PaddedRun(batch_descriptor=batch_descriptor, mode=mode,
                               num_tokens=batch_descriptor.num_tokens),
                    metadata=metadata, inputs=inputs)
                # 图路径的 hidden 缓冲是**补齐后**的行数（含尾部 padding 行）。后续所有消费者
                # （采样行选择、logprobs、掩码、提议者取特征）用的都是**紧凑行号**，而紧凑行
                # 正好是补齐布局的前缀，所以在这里切一刀即可——不是"重新映射"，是"去掉尾巴"。
                hidden_states = hidden_states[:inputs.num_tokens]
                if self.aux_hidden_states is not None:
                    self.aux_hidden_states = self.aux_hidden_states[:inputs.num_tokens]
                self.target_hidden_states = hidden_states
            # LM head 只做在**要采样的行**上（每请求末行、且已 ready）。这是本关"按采样行选
            # hidden"的落点：隐藏态是给所有 token 算的，词表 GEMM 不是。
            #
            # 投机批例外：一个请求要 K+1 行（验证 + bonus），行是按请求块排的、不在
            # `sample_rows` 的坐标系里，所以整块都要算（不 ready 的请求由结果侧丢弃）
            spec_metadata = inputs.spec_metadata
            if spec_metadata is not None and spec_metadata.draft_token_ids.shape[0] > 0:
                sample_indices = inputs.logits_indices
            else:
                sample_indices = inputs.logits_indices[inputs.sample_rows]
            logits = self.model.compute_logits(
                hidden_states.index_select(0, sample_indices.to(self.device)))
        except Exception as exc:                     # noqa: BLE001 —— 任何异常都让 Runner 停摆
            self.execute_model_state = None
            self.failure = f"{type(exc).__name__}: {exc}"
            raise
        # 63/65 关：EAGLE 系（EAGLE3 与 **MTP**）的第一遍都要"本轮 target 真正喂进去的那两行"
        # 做整体左移 + 打补丁，所以要把它们留到提议时刻。判据是**方法**而不是
        # `capture_aux_hidden_states`：MTP 不吃辅助层（那个开关是 False），但同样需要这两份输入。
        needs_target_rows = (self.speculative_config is not None
                             and self.speculative_config.use_eagle())
        self.execute_model_state = ExecuteModelState(
            scheduler_output=scheduler_output, logits=logits, sample_rows=inputs.sample_rows,
            spec_metadata=inputs.spec_metadata,
            query_start_loc=inputs.query_start_loc.to(self.device),
            target_token_ids_cpu=(inputs.input_ids if needs_target_rows else None),
            target_positions_cpu=(inputs.positions if needs_target_rows else None),
            # 64 关：只有 cache-only 提议者需要本轮的元数据（槽位）；别的方法不用，就不留引用
            common_attn_metadata=(self.attn_metadata
                                  if self.speculative_config is not None
                                  and self.speculative_config.uses_extract_hidden_states()
                                  else None))
        return None

    @torch.inference_mode()
    def sample_tokens(self, grammar_output=None, non_block: bool = False):
        """第二步：消费 logits 采样，并产出 `ModelRunnerOutput`（同样在推理边界内）。

        `non_block=True`（70 关，异步调度）：**不在这里把结果拷回 CPU**，而是把采样结果
        （device 张量）连同一条侧流上的异步拷贝包成一个句柄返回；真正的拷回、解析与记账
        发生在 `get_output()`（= 交付边界）。这样 CPU 可以去调度下一轮，而不必干等拷贝。

        `grammar_output`（68 关）：Scheduler 算好的语法掩码。**先打掩码再采样**是上游的顺序
        （`apply_grammar_bitmask` 在 `_sample` 之前），执行与采样分开的意义之一就是给这类
        "采样前还要改一遍 logits"的路径留位置。

        **异常 → 明确失败态**（205 §5 / 204 §9.1）：采样与提议（尤其是提议）失败时，这一轮
        已经处在"镜像更新了、权威输出还没提交"的半截状态。`inference_mode()` 只管梯度记录，
        不会处理这种事务中断，所以这里统一记 `failure`、清掉未交付的状态再抛出去；
        `EngineCore.step()` 在下一轮**调度之前**就会拒绝执行，不会带着半轮状态继续推进。
        """
        self._check_usable()
        state = self.execute_model_state
        if state is None:
            raise RuntimeError("没有待采样的 execute 结果：sample_tokens() 必须跟在 "
                               "返回 None 的 execute_model() 之后")
        self.execute_model_state = None
        # 掩码在这一步才拿到（它在 Scheduler 手里），所以要在这里补进 state
        state = state._replace(grammar_output=grammar_output)

        try:
            if non_block:
                return self._sample_and_propose_async(state)
            return self._sample_and_propose(state)
        except Exception as exc:                     # noqa: BLE001 —— 任何异常都让 Runner 停摆
            # 未交付的草稿/概率不能留着：下一轮若被消费，会把失败轮的假设当成有效提议
            self.pending_draft_token_ids = None
            self.pending_draft_probs = None
            self.failure = f"{type(exc).__name__}: {exc}"
            raise

    def _sample_and_propose_async(self, state) -> "AsyncGPUModelRunnerOutput":
        """异步路径：采样照旧**同步执行**（内核发射本来就是异步的），只是**不把结果拷回来**。

        与上游的差异（写在 `docs/step70_alignment.md` §6）：上游的采样结果本身就以 device
        张量形式存在，账也在 GPU 侧记；本仓库的 `_bookkeeping_sync` 与提议器都是 CPU 驱动的
        （要 token 的 Python 值），所以"记账 + 提议"只能推迟到 `get_output()`——也就是
        "等到值真的被需要时"。收益没有消失：**调度下一轮（预算/KV/占位）与这次拷贝重叠**，
        而调度正是异步调度要抢出来的那段时间。
        """
        self._apply_grammar_bitmask(state)
        # 草稿统一从 `_resolved_spec_decode_tokens` 读（本轮 `_update_states()` 里写的同一份：
        # 同步路径就是协议里的值）——"验证的候选"与"写进输入缓冲的候选"因此永远是同一份。
        scheduled_spec = self._resolved_spec_decode_tokens
        spec_metadata = state.spec_metadata
        if spec_metadata is not None and spec_metadata.draft_token_ids.shape[0] > 0:
            sampler_output = self._run_spec_sampler(state, spec_metadata, scheduled_spec)
            kind = "spec"
        else:
            sampling_metadata = SamplingMetadata.from_input_batch(
                self.input_batch, state.sample_rows, device=self.device,
                scheduled_spec_decode_tokens=scheduled_spec,
                logprobs_mode=self.logprobs_mode)
            sampler_output = self.sampler.forward(state.logits, sampling_metadata)
            kind = "plain"
        handle = AsyncGPUModelRunnerOutput(
            runner=self, state=state, sampler_output=sampler_output, kind=kind)
        self._pending_async_output = handle
        return handle

    def _finish_async_output(self, state, sampled, logprobs_by_req) -> ModelRunnerOutput:
        """`get_output()` 里的后半段：记账 + 提议（同步路径后半段的同一段实现）。

        传进来的是**已经解析成 CPU 结果**的 token/logprobs，所以"异步只改时机、不改语义"：
        两条路径共用同一段记账与提议代码，不可能算出不同结果。
        """
        output = self._bookkeeping_sync(state, sampled, logprobs_by_req)
        self.pending_draft_token_ids = self._propose_draft_tokens(state, sampled)
        return output

    def wait_for_pending_async_output(self) -> None:
        """把上一轮还没交付的异步结果**结清**（幂等）。

        调用点是下一步 `execute_model()` 的最前面：那一步的输入组装要读执行端镜像
        （`token_ids_cpu` 等），而镜像是上一轮记账写进去的——**没结清就读 = 读到旧历史**
        （不报错、只是喂给模型错的 token）。上游靠"把上一轮采样的 token 直接 scatter 进
        GPU 输入缓冲"避免这次等待；本仓库的输入组装在 CPU 上，所以必须在组装前等一次。
        """
        handle = self._pending_async_output
        if handle is None:
            return
        handle.get_output()

    def _sample_and_propose(self, state):
        """`sample_tokens()` 的正常路径（单独一层，好让失败态只包一层 try）。"""
        # 68 关：结构化输出的掩码必须在**采样之前**打到 logits 上（上游同序：
        # `apply_grammar_bitmask` 在 `_sample` 之前）。它原地改 `state.logits`，
        # 所以掩码之后的那份就是采样器看到的那一份——`processed_*` 模式的 logprobs
        # 因此天然包含掩码，不存在"交付的 logprobs 认为非法 token 还有概率"。
        self._apply_grammar_bitmask(state)
        # 草稿的"值来源"：同步路径就是协议里的（执行侧照抄了它），异步路径是执行侧自己补的
        # 那份（`_update_states` 里解析并记下）。两条路都用 `_resolved_spec_decode_tokens`，
        # 于是"验证的候选"与"写进输入缓冲的候选"永远是同一份。
        scheduled_spec = self._resolved_spec_decode_tokens
        spec_metadata = state.spec_metadata
        if spec_metadata is not None and spec_metadata.draft_token_ids.shape[0] > 0:
            # ---- 投机路径：验证草稿 ----
            # 元数据要按**被调度的请求**建（草稿的"假设历史"来自协议里的 spec tokens），
            # 行序与 spec_metadata 的请求顺序一致，长度是 K+1 而不是 1
            sampled, logprobs_by_req = self._sample_with_spec(
                state, spec_metadata, scheduled_spec)
        else:
            # ---- 普通路径（含全批 K=0）----
            sampling_metadata = SamplingMetadata.from_input_batch(
                self.input_batch, state.sample_rows, device=self.device,
                scheduled_spec_decode_tokens=scheduled_spec,
                logprobs_mode=self.logprobs_mode)
            sampler_output = self.sampler.forward(state.logits, sampling_metadata)
            sampled = sampler_output.sampled_token_ids.tolist()
            logprobs_by_req = self._logprobs_by_request(sampler_output.logprobs_tensors,
                                                        state.sample_rows)

        # 先记账、再提草稿（顺序不能反，199 §4 的时序）：
        # `_bookkeeping_sync` 把本轮采样结果写进镜像，**提议必须看到它**——
        # 否则草稿是基于"少一个 token 的历史"算出来的，draft 侧的进度也会比 target 落后一格，
        # 于是发布出去的完整块可能还没被 draft 算过（验收方的独立探针分别抓到了这两点）
        output = self._bookkeeping_sync(state, sampled, logprobs_by_req)
        self.pending_draft_token_ids = self._propose_draft_tokens(state, sampled)
        return output

    # -------- 结构化输出（68 关）--------

    def _apply_grammar_bitmask(self, state) -> None:
        """把语法掩码打到这一轮的 logits 上（上游 `sample_tokens` 里的同一步）。

        行映射由本 Runner 给出（`_logit_row_of_req`）：上游用它的 InputBatch 现算，
        但本仓库的非投机路径**只为要采样的行算 logits**（57 关以来的教学差异），所以
        "请求 → 第一行"只有这里知道。
        """
        grammar_output = state.grammar_output
        if grammar_output is None:
            return
        from ..structured_output.utils import apply_grammar_bitmask

        apply_grammar_bitmask(state.logits, grammar_output,
                              state.scheduler_output.scheduled_spec_decode_tokens,
                              self._logit_row_of_req(state))

    def _logit_row_of_req(self, state) -> dict[str, int]:
        """请求 ID → 它在本轮 `logits` 里的第一行行号。

        两种批的行布局（与 `execute_model()` 里选 `sample_indices` 的两条分支一一对应）：

            投机批（有草稿）：所有批行、每请求 `1 + K_i` 行，顺序 = `input_batch.req_ids`
            普通批：只有 `sample_rows` 那些行，每行一条请求

        行数对不上时直接报错：掩码打错行**不会**报错，只会让模型在该填数字的地方写出字母。
        """
        spec_metadata = state.spec_metadata
        rows: dict[str, int] = {}
        if spec_metadata is not None and spec_metadata.draft_token_ids.shape[0] > 0:
            offset = 0
            for row, req_id in enumerate(self.input_batch.req_ids):
                rows[req_id] = offset
                offset += 1 + spec_metadata.num_draft_tokens[row]
        else:
            for index, row in enumerate(state.sample_rows):
                rows[self.input_batch.req_id_at(row)] = index
        if len(rows) and max(rows.values()) >= state.logits.shape[0]:
            raise RuntimeError(
                f"请求 → logits 行的映射越界了（最大行号 {max(rows.values())}，"
                f"logits 只有 {state.logits.shape[0]} 行）：掩码会打到别的行上，"
                f"那是不报错的静默错，所以直接停下")
        return rows

    def _logprobs_by_request(self, logprobs_tensors, sample_rows: list[int]):
        """把"按采样行排列"的 logprobs 摊成 `{req_id: LogprobsLists}`（68 关）。

        为什么要摊开：`_bookkeeping_sync` 按 **Scheduler 的请求顺序**交付结果，而 logprobs
        是**按批行/采样行**算出来的，两者顺序可能不同。逐请求摊开之后按目标顺序重排，
        就不会出现"行号猜对了但内容属于别人"这种静默错。
        """
        if logprobs_tensors is None:
            return None
        lists = logprobs_tensors.tolists()
        return {self.input_batch.req_id_at(row): self._slice_logprobs(lists, index, 1)
                for index, row in enumerate(sample_rows)}

    @staticmethod
    def _slice_logprobs(logprobs_lists, start: int, num_positions: int):
        """从 `LogprobsLists` 里切出连续 `num_positions` 行（numpy 切片，不拷贝数据）。"""
        end = start + num_positions
        return type(logprobs_lists)(
            logprobs_lists.logprob_token_ids[start:end],
            logprobs_lists.logprobs[start:end],
            logprobs_lists.sampled_token_ranks[start:end],
            None)

    # -------- 投机（57E）--------

    def _sample_with_spec(self, state, spec_metadata, scheduled_spec):
        """验证草稿，摊成**每条请求一串 token**（无效位置裁掉、非 ready 的给空）。

        拒绝采样器给的是 `[B, max_spec_len+1]` 的 padded 张量（无效位置 `-1`）；裁剪交给
        `RejectionSampler.parse_output`（上游同一处：那里是整条验证路径唯一的 D2H）。
        **元数据按整个批建**（行序 = `input_batch.req_ids`），因为草稿的"假设历史"来自协议里的
        spec tokens，而不是我们筛出来的采样行。

        68 关：logprobs 与 token **用同一张 valid_mask 裁**（同一处 `parse_output`），
        所以被拒绝的候选位在两边同时消失——这就是"截断后的尾部不能漏出"的机制。
        """
        sampler_output = self._run_spec_sampler(state, spec_metadata, scheduled_spec)
        return self._parse_spec_sampler_output(state, sampler_output)

    def _run_spec_sampler(self, state, spec_metadata, scheduled_spec):
        """**只跑验证内核**（结果仍是 device 张量，不拷回）——异步路径在这里就返回句柄。"""
        sampling_metadata = SamplingMetadata.from_input_batch(
            self.input_batch, list(range(self.input_batch.num_reqs)), device=self.device,
            scheduled_spec_decode_tokens=scheduled_spec,
            logprobs_mode=self.logprobs_mode)
        draft_probs = self._get_spec_decode_draft_probs(spec_metadata)
        return self.rejection_sampler(
            spec_metadata, draft_probs, state.logits, sampling_metadata)

    def _parse_spec_sampler_output(self, state, sampler_output):
        """验证内核的 device 结果 → 每请求一串 token（无效位置裁掉、非 ready 的给空）。

        **这是投机路径上唯一的 D2H**（`parse_output` 里的 `cpu().numpy()`）：同步路径当场做，
        异步路径推迟到 `get_output()`（交付边界）。
        """
        # 未 ready 的行整行丢弃（中间 prefill 块：logits 有效，但这一轮不该产出 token）。
        # 上游用同一个 `discard_req_indices` 参数（它的 `discard_request_mask`），
        # 并在那里顺手把对应 generator 的 offset 退回去。
        ready = set(state.sample_rows)
        discard = [row for row in range(self.input_batch.num_reqs) if row not in ready]
        rows, logprobs_lists = RejectionSampler.parse_output(
            sampler_output.sampled_token_ids, self.input_batch.vocab_size, discard,
            sampler_output.logprobs_tensors)
        sampled_by_row = {row: (list(rows[row]) if row in ready else [])
                          for row in range(self.input_batch.num_reqs)}
        sampled = [sampled_by_row[row] for row in state.sample_rows]

        # logprobs 按请求切开：`cu_num_generated_tokens` 给出每请求的起始行（投机时每请求
        # 交付 0 ~ K+1 个位置，行数各不相同，所以不能拿请求序号当行号）
        logprobs_by_req = None
        if logprobs_lists is not None:
            cu = logprobs_lists.cu_num_generated_tokens or []
            logprobs_by_req = {}
            for row in range(self.input_batch.num_reqs):
                if row not in ready:
                    continue
                start = cu[row] if cu else row
                num_positions = (cu[row + 1] - cu[row]) if cu else 1
                logprobs_by_req[self.input_batch.req_id_at(row)] = self._slice_logprobs(
                    logprobs_lists, start, num_positions)
        return sampled, logprobs_by_req

    def _propose_draft_tokens(self, state, sampled):
        """按配置提**下一轮**的草稿（199 §4：轮 t 验证时顺手提，轮 t+1 才采用）。

        这里只做一件事：把本轮 target 侧的**事实**摊成 `TargetRows`（58 §5 要求起点/终点
        写清楚），提议者据此决定 draft 第一遍写哪些位置：

            start        本轮起点 = 执行侧收到的调度快照里的 `num_computed_tokens`
                         （**含 prefix 命中起点**，所以命中过的前缀不会被重算）
            target_rows  本轮 target 执行的行数（调度快照里的 `num_scheduled_tokens`）
            num_rejected 本轮采用的草稿里被拒的条数（= 采用数 - 接受数）
            history_end  记账后的**有效历史**末尾（ready 行）；未 ready = start + target_rows
            next_token_id 扩容行的 token：ready = 最后一个新采样 token；
                         未 ready = backup（本段历史的最后一个 token，上游同款）

        中间 prefill 块（未 ready）也要过一遍提议者：它们这一轮也要把 draft 的 KV 同步到
        target 算完的位置，否则 target 这一轮发布的完整块会带着一层没写过的 draft KV
        （199 §9）。草稿本身只给 ready 行提（提了 Scheduler 也会丢）。
        """
        if self.proposer is None:
            return None
        # 71 关（动态投机长度）：本轮要提几枚草稿由**调度快照**说了算（上游
        # `propose_draft_token_ids()` 里同一行：`num_spec_tokens_to_schedule =
        # scheduler_output.num_spec_tokens_to_schedule`）。静态 K 时它等于配置的 K，
        # 所以"提议宽度只有一个来源"这件事在两条路径上都成立。
        # 上界必须现场判：工作区/KV 预留/块表都按配置的最大 K 开，超了就是调度侧口径错了。
        num_spec_tokens = state.scheduler_output.num_spec_tokens_to_schedule
        max_num_spec_tokens = self.speculative_config.num_speculative_tokens
        if not 0 <= num_spec_tokens <= max_num_spec_tokens:
            raise RuntimeError(
                f"本轮 K={num_spec_tokens} 超出 [0, {max_num_spec_tokens}]：动态投机长度的"
                f"查找表只被最大 K 裁剪（工作区/图/掩码缓冲按它开），调度侧不该发出更大的值")
        req_ids = list(self.input_batch.req_ids)
        if not req_ids:
            return None
        sampled_by_row = {row: list(tokens) for row, tokens in zip(state.sample_rows, sampled)}
        adopted = state.scheduler_output.scheduled_spec_decode_tokens
        all_token_ids = {req_id: self.requests[req_id].all_token_ids for req_id in req_ids}

        rows: list[TargetRows] = []
        for req_id in req_ids:
            row = self.input_batch.req_id_to_index[req_id]
            start = int(self.input_batch.num_computed_tokens_cpu[row])
            target_rows = int(state.scheduler_output.num_scheduled_tokens[req_id])
            tokens = sampled_by_row.get(row, [])
            num_drafted = len(adopted.get(req_id, ()))
            if tokens:
                # ready：接受 a 枚 + 1 个纠正/奖励 token；被拒的 K-a 枚不进新提议上下文
                num_accepted = max(len(tokens) - 1, 0)
                num_rejected = num_drafted - num_accepted
                history_end = self.input_batch.num_tokens(row)
                # [start, history_end) = 本轮 target query 里的**有效行** + 1 个扩容行：
                # 有效行 = n - 被拒行；扩容行 = 最后那个新采样 token（位置 history_end-1）
                if history_end != start + (target_rows - num_rejected) + 1:
                    raise RuntimeError(
                        f"{req_id!r} 的本轮口径对不上：记账后历史 {history_end} != "
                        f"起点 {start} + (target 行数 {target_rows} - 被拒 {num_rejected}) + 1。"
                        f"这是调度快照/记账/草稿采用数三者不一致，属于执行侧 bug")
                if num_rejected < 0:
                    raise RuntimeError(
                        f"{req_id!r} 接受了 {num_accepted} 枚草稿，但本轮只采用了 "
                        f"{num_drafted} 枚：采用数不该小于接受数")
            else:
                # 中间 prefill 块：只同步到 target 本轮**算完**的位置（已提交历史里那些还没算的
                # 位置，KV 槽位也还没分配，同步过去就是越界写，204 §6.1）
                num_rejected = 0
                history_end = start + target_rows
            # 扩容行的 token：ready 用刚采出的最后一个；未 ready 用 backup（=本段最后一个 token）
            backup_index = start + target_rows - 1
            next_token_id = int(tokens[-1]) if tokens else int(all_token_ids[req_id][backup_index])
            rows.append(TargetRows(req_id=req_id, row=row, start=start,
                                   target_rows=target_rows, num_rejected=num_rejected,
                                   history_end=history_end, next_token_id=next_token_id,
                                   ready=bool(tokens)))
        # 按方法分派（草稿来源不同：用户类 / CPU、GPU 的 ngram / draft 模型）
        if self.speculative_config is not None and \
                self.speculative_config.method == "custom_class":
            # 62 关：**照上游 `custom_class` 分支的参数**调用用户类，一个都不多、一个都不少：
            #     drafter.propose(sampled_token_ids, num_tokens_no_spec, token_ids_cpu,
            #                     slot_mappings=slot_mappings)
            # 三个对象就是 InputBatch 自己的缓冲（不做拷贝、不转格式）；返回 `list[list[int]]`
            # 再包成本仓库 Runner 的 `DraftTokenIds`（协议出口只有这一个，插件不需要知道它）。
            #
            # 行对齐：`sampled_token_ids` 必须**按批行**给满（未采样的行给空列表），
            # 因为用户类会拿它和 `num_tokens_no_spec[row]` 配对——只有长度 == 批行数时
            # "第 i 项 ↔ 第 i 行"才成立（61 关讨论过 i 与 req_id_to_index 的关系）。
            num_reqs = len(self.input_batch.req_ids)
            sampled_token_ids: list[list[int]] = [[] for _ in range(num_reqs)]
            for row, tokens in sampled_by_row.items():
                sampled_token_ids[row] = list(tokens)
            draft_token_ids = self.proposer.propose(
                sampled_token_ids,
                self.input_batch.num_tokens_no_spec,
                self.input_batch.token_ids_cpu,
                slot_mappings=None,  # 本仓库还没有 slot_mappings 对象（69 关 CUDA Graph 时才有）
            )
            # 插件契约：逐行返回，行数 == 批行数（每行枚数随意，可为 0）。
            # 上游不检查这一条；本仓库按"协议违约立刻报错"处理——因为外部插件的行错位
            # 在这里是**静默**的：少给几行只会让那几条请求没草稿（看起来"能用"），
            # 报错比让人以为插件写对了更省事（差异记在 docs/step62_alignment.md §3）。
            if len(draft_token_ids) != num_reqs:
                raise RuntimeError(
                    f"自定义提议者 {type(self.proposer).__name__} 返回了 "
                    f"{len(draft_token_ids)} 行草稿，但本轮批里有 {num_reqs} 行："
                    f"必须按批行逐行返回（未采样的行给空列表）")
            drafts = DraftTokenIds(req_ids=list(self.input_batch.req_ids),
                                   draft_token_ids=[list(draft) for draft in draft_token_ids])
        elif self.speculative_config is not None and self.speculative_config.use_ngram_gpu():
            # 60 关：GPU 提议者收显存里的历史与长度，交回**固定宽度**的草稿 + 每行有效个数
            drafts = self.proposer.propose_drafts(
                rows, all_token_ids, self.input_batch, sampled_by_row=sampled_by_row,
                sample_rows=state.sample_rows, token_ids_gpu=self.token_ids_gpu_tensor,
                num_tokens_no_spec_gpu=self.num_tokens_no_spec_gpu)
        elif self.speculative_config is not None and self.speculative_config.method == "ngram":
            # 71 关：ngram（CPU）收本轮 K——上游 `drafter.propose(num_spec_tokens_to_schedule, ...)`。
            # 它不跑模型、不写 KV，所以 K=0 就是"空草稿"，没有第一遍要同步。
            drafts = self.proposer.propose_drafts(
                rows, all_token_ids, self.input_batch,
                num_speculative_tokens=num_spec_tokens)
        elif self.speculative_config is not None and self.speculative_config.method == "suffix":
            # 61 关：suffix decoding 的输入就是 InputBatch 的三个 CPU 缓冲
            # （token_ids_cpu / num_tokens_no_spec / num_prompt_tokens），
            # `sampled_by_row` 告诉它本轮哪些行真的采到了 token（空 = 中间 prefill 块）。
            drafts = self.proposer.propose_drafts(
                rows, all_token_ids, self.input_batch, sampled_by_row=sampled_by_row)
        elif self.speculative_config is not None and \
                self.speculative_config.uses_extract_hidden_states():
            # 64 关：cache-only 提议者走**自己那套特殊协议**（上游 `propose_draft_token_ids()`
            # 里的 extract 分支），不吃 `TargetRows`/`all_token_ids`——它不跑 draft 模型。
            drafts = self._propose_extract_hidden_states(state, sampled_by_row)
        elif self.speculative_config is not None and self.speculative_config.uses_medusa():
            # 66 关：Medusa 也走自己的协议（上游 `propose_draft_token_ids()` 的 medusa 分支）：
            # 它只要"每条请求最后一个已算过的 token"的 hidden，跑 N 个 head、取 argmax。
            drafts = self._propose_medusa(state, sampled_by_row)
        else:
            # 恢复过的请求：它的块表整表换过 → draft 只从本轮协议给的有效前缀重新开始
            kwargs = {"reset_req_ids": set(self._resumed_req_ids)}
            # 71 关：LLM 系提议者（draft_model / EAGLE / MTP）的 `propose()` 收本轮 K——
            # 上游同一处传 `num_speculative_tokens=num_spec_tokens_to_schedule`
            # （`gpu_model_runner.py:5380`）。K=0 时它**仍然跑第一遍**同步 draft KV，
            # 只是返回 `[B, 0]`（需求 071 §3.4）。
            kwargs["num_speculative_tokens"] = num_spec_tokens
            if getattr(self.proposer, "supports_padded_first_pass", False):
                # 69 关：按上游 `drafter.prepare_inputs_padded(...)` 的**同一套输入**算出
                # "该从哪一行采样"与"每请求被拒了几行"，再按上游的参数名传进 propose()。
                # 上游是在 device 上算的（`eagle_prepare_inputs_padded_kernel`），本仓库的
                # draft 第一遍仍是 CPU 拼行（58 关的设计），所以这两个张量在本仓库里的作用
                # 是**校验与对齐口径**：提议者会把它们与自己算的那份逐值比一遍（不一致就报错），
                # 而不是"两条路各算一次、谁错都看不出来"。
                kwargs["token_indices_to_sample"], kwargs["num_rejected_tokens_gpu"] = \
                    self._padded_draft_token_indices(state, sampled_by_row)
            if getattr(self.proposer, "pass_hidden_states_to_model", False):
                # EAGLE 系（含 MTP）的提议者才收特征；普通 draft/假提议者的签名不变。
                # `target_token_ids` / `target_positions` 就是**本轮 target 真正喂进去的行**
                # （含被拒草稿）：上游 `set_inputs_first_pass` 拿的正是这两份 + 特征。
                kwargs["target_hidden_states"] = self._target_hidden_states_by_req(
                    state.scheduler_output, len(self.input_batch.req_ids))
                kwargs["target_token_ids"] = state.target_token_ids_cpu
                kwargs["target_positions"] = state.target_positions_cpu
            drafts = self.proposer.propose(rows, all_token_ids, self.input_batch, **kwargs)
        # 概率按请求存：下一轮可能只采用每条请求的**前缀**，所以要留下每条的块边界
        self.pending_draft_probs = drafts if drafts.draft_probs is not None else None
        return drafts

    def _padded_draft_token_indices(self, state, sampled_by_row):
        """算 draft 侧 padded 批的两个逐请求索引（上游 `prepare_inputs_padded` 的输入）。

            valid_sampled_tokens_count[i]  这条请求本轮采到了几个有效 token（= 接受数 + 1）
            cu_num_draft_tokens            本轮采用的草稿数的**包含式**前缀和（spec_metadata 给）

        两者都按**批行序**给，长度 = 本轮批的请求数（上游同款）。没有带草稿的请求时
        `cu_num_draft_tokens` 会是全 0，`prepare_inputs_padded` 得到 num_rejected=0、
        index=每请求最后一行——与"没有草稿"的语义一致（那一行就是纠正/bonus 行）。
        """
        from ..spec_decode.utils import prepare_inputs_padded

        num_reqs = self.input_batch.num_reqs
        valid_counts = torch.tensor(
            [len(sampled_by_row.get(row, ())) for row in range(num_reqs)],
            dtype=torch.int32, device=self.device)
        spec_metadata = state.spec_metadata
        if spec_metadata is not None:
            cu_num_draft_tokens = spec_metadata.cu_num_draft_tokens
        else:
            cu_num_draft_tokens = torch.zeros(num_reqs, dtype=torch.int32,
                                              device=self.device)
        if state.query_start_loc is None:
            raise RuntimeError("这一轮没有留下 query_start_loc：prepare_inputs_padded 需要它")
        return prepare_inputs_padded(cu_num_draft_tokens, valid_counts,
                                     state.query_start_loc, num_reqs)

    def _propose_extract_hidden_states(self, state, sampled_by_row) -> DraftTokenIds:
        """64 关：`extract_hidden_states` 的**特殊协议**分支（上游同名分支）。

        它不猜 token：`propose()` 的返回值是 `sampled_token_ids[:, :1]`——target 本轮采出的
        第一列，直接当成下一轮的草稿。所以这里不做任何"提议质量"的判断，也没有 `draft_probs`
        （点质量 q，验证时走 59 关的 `NO_DRAFT_PROBS` 分支，采样分布仍然精确是 target 的分布）。

        真正的产物是**副作用**：target 本轮每个 query 行的辅助层特征被写进了 cache-only 层的
        分页缓存（槽位与本轮 KV 完全相同）。

        行 → 请求的映射与其它提议者一致：给满批行数，没采到 token 的行（中间 prefill 块）用
        `-1` 占位（上游的 padded 张量同样用 `-1` 表示"这一行没有有效采样"），它们不产生草稿。
        """
        if not self.capture_aux_hidden_states or self.aux_hidden_states is None:
            raise RuntimeError(
                "extract_hidden_states 需要本轮 target 的辅助层特征（aux_hidden_states）："
                "没有特征就没有东西可缓存，这条路径不该出现")
        if state.common_attn_metadata is None:
            raise RuntimeError(
                "extract_hidden_states 需要本轮的 attention 元数据（槽位要与 target KV 同源）："
                "execute_model() 没把它留下来")
        num_rows = len(self.input_batch.req_ids)
        sampled = self._padded_sampled_token_ids(sampled_by_row, num_rows)
        # 上游调用形态：propose(K, sampled_token_ids, target_hidden_states, common_attn_metadata)
        draft_tokens = self.proposer.propose(
            num_speculative_tokens=self.speculative_config.num_speculative_tokens,
            sampled_token_ids=sampled,
            target_hidden_states=self.aux_hidden_states,
            common_attn_metadata=state.common_attn_metadata)
        # 只取第 0 列当草稿（上游：宽度可能 >1，仍然只返回规定的那一列）
        column = draft_tokens[:, 0].tolist()
        return DraftTokenIds(
            req_ids=list(self.input_batch.req_ids),
            draft_token_ids=[[int(token)] if int(token) >= 0 else [] for token in column])

    def _propose_medusa(self, state, sampled_by_row) -> DraftTokenIds:
        """66 关：Medusa 的**特殊协议**分支（上游 `gpu_model_runner.py:5206-5225`）。

        上游那段只有两件事：挑出每条请求"最后一个已算过的 token"的 hidden，然后
        `drafter.propose(K, hidden_states, sampling_metadata, slot_mappings)` 拿回
        `[B, num_heads]`。行号算式在 `MedusaProposer.select_target_hidden_states()` 里
        （本仓库把它收进提议者，见那里的说明）。

        **本仓库的两处差异**（docs/step66_alignment.md §3）：

        1. 中间 prefill 块（本轮排了多行、没有采到 token）**跳过不提草稿**。上游会给它算一个
           `-1` 行号（取到最后一行）并把结果一起交回去，靠 Scheduler 事后把 prefill 块的草稿
           丢掉；本仓库在提议这一步就不做无意义的计算（那个 hidden 行也不代表"最后一个 token"）。
        2. 行数口径用调度快照的 `num_scheduled_tokens`（见 `select_target_hidden_states`）。
        """
        if self.target_hidden_states is None:
            raise RuntimeError(
                "Medusa 需要本轮 target 的 hidden states（`_run_model()` 留下的那份）："
                "head 就是在这份 hidden 上做预测的，没有它这条路径不该出现")
        req_ids = list(self.input_batch.req_ids)
        spec = self.speculative_config
        ready_req_ids: list[str] = []
        rows_per_request: list[int] = []
        num_sampled: list[int] = []
        for row, req_id in enumerate(req_ids):
            tokens = sampled_by_row.get(row, [])
            if not tokens:
                continue          # 中间 prefill 块：没有"最后一个已算过的 token"
            ready_req_ids.append(req_id)
            rows_per_request.append(int(state.scheduler_output.num_scheduled_tokens[req_id]))
            num_sampled.append(len(tokens))
        if not ready_req_ids:
            return DraftTokenIds(req_ids=req_ids, draft_token_ids=[[] for _ in req_ids])
        hidden_states = self.proposer.select_target_hidden_states(
            self.target_hidden_states, rows_per_request, num_sampled)
        # 上游调用形态：propose(K, hidden_states, sampling_metadata, slot_mappings=None)。
        # 后两个参数上游收下但不用（argmax 提议不看采样参数、也不写 KV），本仓库不为了
        # "凑参数"去建一份 SamplingMetadata。
        draft_tokens = self.proposer.propose(
            num_speculative_tokens=spec.num_speculative_tokens,
            target_hidden_states=hidden_states)
        if draft_tokens.shape[1] != spec.num_speculative_tokens:
            raise RuntimeError(
                f"Medusa 交回 {draft_tokens.shape[1]} 列草稿，但 K="
                f"{spec.num_speculative_tokens}：head 数与调度侧的 K 必须一致")
        proposed = {req_id: [int(token) for token in draft_tokens[index].tolist()]
                    for index, req_id in enumerate(ready_req_ids)}
        return DraftTokenIds(req_ids=req_ids,
                             draft_token_ids=[proposed.get(req_id, []) for req_id in req_ids])

    def _padded_sampled_token_ids(self, sampled_by_row, num_rows: int) -> torch.Tensor:
        """把"每行采到了哪些 token"摊成 `[num_rows, K+1]` 的定宽张量（无效位置 `-1`）。

        上游在 padded drafter batch 下拿到的就是这个形状的 GPU 张量；本仓库的采样结果是
        **每行的变长列表**（`RejectionSampler.parse_output` 之后），所以在这里补成同形状——
        `propose()` 的 `[:, :1]` 切片语义才和上游一致（K=1 时一行最多 2 列：
        验证通过的草稿 + bonus）。
        """
        width = self.speculative_config.num_speculative_tokens + 1
        padded = torch.full((num_rows, width), -1, dtype=torch.int32)
        for row, tokens in sampled_by_row.items():
            row_tokens = list(tokens)[:width]
            if row_tokens:
                padded[row, :len(row_tokens)] = torch.tensor(row_tokens, dtype=torch.int32)
        return padded

    def _get_spec_decode_draft_probs(self, spec_metadata) -> torch.Tensor | None:
        """按**本轮采用的行序**拼出 `[P, V]` 的 q（上游同名方法；199 §5 的 q 对齐）。

        上一轮提草稿的顺序与本轮采用的顺序一般**不同**（行序会变、每条还可能被预算截短），
        所以要按"每请求取前 K_i 行"重新拼，**不能**拿原矩阵的前 P 行——那会把 A 的概率
        配到 B 的草稿上。确定性提议（ngram）没有概率，返回 None（点质量 q，内核走
        `NO_DRAFT_PROBS` 分支）。

        与上游的差异：上游在"某条请求的概率没缓存"时打 warning 然后**退回无 q**，那会让接受
        判定变成 `p[d] >= u`（接受率虚高）；本机按约定**明确报错**，不静默降级。
        """
        previous = self.pending_draft_probs
        if previous is None or previous.draft_probs is None:
            return None
        offsets: dict[str, int] = {}
        cursor = 0
        for req_id, draft_ids in zip(previous.req_ids, previous.draft_token_ids):
            offsets[req_id] = cursor
            cursor += len(draft_ids)
        rows: list[int] = []
        for req_id, num_draft in zip(self.input_batch.req_ids,
                                     spec_metadata.num_draft_tokens):
            if num_draft == 0:
                continue
            start = offsets.get(req_id)
            if start is None:
                raise RuntimeError(
                    f"{req_id!r} 本轮采用了草稿，但上一轮没有为它提过："
                    f"q 对不上（草稿必须与本轮采用的前缀同源）")
            if start + num_draft > cursor:
                raise RuntimeError(
                    f"{req_id!r} 本轮采用了 {num_draft} 枚草稿，但上一轮只为它提了 "
                    f"{cursor - start} 枚：采用数不该超过提议数")
            rows.extend(range(start, start + num_draft))
        if not rows:
            return None
        return previous.draft_probs[rows].contiguous()

    def take_draft_token_ids(self):
        """取走本轮提的草稿（`EngineCore.post_step` 调，发生在 `sample_tokens` 之后）。

        对应 199 §4 的时序：轮 t 的采样里验证并提议 → post_step 取回 → Scheduler 记下 →
        轮 t+1 的 `schedule()` 才决定采用几枚。**提议不改本轮计划**。
        """
        drafts, self.pending_draft_probs = self.pending_draft_token_ids, self.pending_draft_probs
        self.pending_draft_token_ids = None
        # 本轮协议里"整表替换过块表"的请求（draft 侧据此重置自己的进度）
        self._resumed_req_ids: set[str] = set()
        return drafts

    # -------- 采样之后的记账 --------

    def _bookkeeping_sync(self, state: ExecuteModelState, sampled: list[list[int]],
                          logprobs_by_req=None):
        """把采样结果**散射回所有被调度的请求**，并更新镜像。

        未 ready 的请求返回 `[]`（198 §4 允许的显式教学差异：本机是"先算采样行、再在
        `_bookkeeping_sync` 里丢掉不该提交的结果"，本关直接从采样阶段就不算它们）。

        这里最容易错的是"筛行之后还拿原列表 zip"：`sampled` 的行号是 **batch 行**，
        结果要按 `req_id` 摊回 `num_scheduled_tokens` 的顺序。所以映射
        `sample_row → req_id` 必须显式重建。

        68 关的 logprobs 走同一条重建：按 `req_ids` 的顺序拼起来，并用
        `cu_num_generated_tokens` 给出每请求的起始行——投机时一条请求可能交付 0~K+1 个位置，
        Scheduler 侧 `slice_request(req_index, n)` 就靠这个偏移切（上游同款容器与用法）。
        """
        sampled_by_req: dict[str, list[int]] = {}
        for row, token_ids in zip(state.sample_rows, sampled):
            req_id = self.input_batch.req_id_at(row)     # 行号 → ID（唯一对应关系）
            sampled_by_req[req_id] = list(token_ids)
            self._commit_tokens_to_mirror(req_id, row, list(token_ids))

        req_ids = list(state.scheduler_output.num_scheduled_tokens)
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
            sampled_token_ids=[sampled_by_req.get(req_id, []) for req_id in req_ids],
            logprobs=self._concat_logprobs_in_req_order(logprobs_by_req, req_ids),
        )

    @staticmethod
    def _concat_logprobs_in_req_order(logprobs_by_req, req_ids: list[str]):
        """按 Scheduler 的请求顺序拼接 logprobs，并给出每请求的行偏移（68 关）。

        没有 logprobs 的请求（没要、或这一轮没采到 token）插**0 行**的空段：偏移表因此与
        `req_ids` 严格对齐，`slice_request(req_index, n)` 不会切到别人的行上。
        上游在这一层用 `LogprobsTensors.cat`；本项目在 CPU numpy 上做同一件事
        （跨执行边界传的本来就是 numpy）。
        """
        if logprobs_by_req is None:
            return None
        import numpy as np

        from ..outputs import LogprobsLists

        width = None
        for block in logprobs_by_req.values():
            if block.logprob_token_ids.shape[0] > 0:
                width = block.logprob_token_ids.shape[1]
                break
        if width is None:
            return None

        token_ids, logprobs, ranks, cu = [], [], [], []
        offset = 0
        for req_id in req_ids:
            block = logprobs_by_req.get(req_id)
            cu.append(offset)
            if block is None or block.logprob_token_ids.shape[0] == 0:
                continue
            token_ids.append(block.logprob_token_ids)
            logprobs.append(block.logprobs)
            ranks.append(block.sampled_token_ranks)
            offset += block.logprob_token_ids.shape[0]
        cu.append(offset)

        def _cat(parts, dtype, columns):
            if not parts:
                return np.empty((0, columns), dtype=dtype)
            return np.concatenate(parts, axis=0)

        return LogprobsLists(
            _cat(token_ids, np.int32, width),
            _cat(logprobs, np.float32, width),
            (np.concatenate(ranks, axis=0) if ranks else np.empty((0,), dtype=np.int32)),
            cu)

    def _commit_tokens_to_mirror(self, req_id: str, row: int, token_ids: list[int]) -> None:
        """把本轮产出的 token 写进执行端镜像：`CachedRequestState` + CPU 缓冲两份都要更新。

        缓冲区里的历史是**下一轮的输入来源**，落一份在这里，下一轮就不必让 Scheduler 重发
        整段历史（那正是 `all_token_ids` 只在必要时才带的原因）。
        """
        if not token_ids:
            return
        state = self.requests[req_id]
        start = self.input_batch.num_tokens(row)
        end = start + len(token_ids)
        if end > self.max_model_len:
            raise RuntimeError(
                f"{req_id!r} 采样后的长度 {end} 超过 max_model_len={self.max_model_len}："
                f"上下文上限的裁剪是 Scheduler 的责任（要为本轮的采样结果留位置）")
        self.input_batch.token_ids_cpu[row, start:end].copy_(
            torch.tensor(token_ids, dtype=torch.int64))
        self.input_batch.num_tokens_no_spec[row] = end
        state.output_token_ids.extend(token_ids)
        # 草稿区到此为止：被接受的已经写进上面这段，被拒的不该再影响 ready 判据
        # （`num_tokens_with_spec` 会把它们算进去）
        self.input_batch.spec_token_ids[row].clear()
