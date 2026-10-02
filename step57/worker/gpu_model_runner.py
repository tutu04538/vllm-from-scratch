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

from dataclasses import dataclass, field
from typing import NamedTuple

import torch

from ..attention import Attention, AttentionMetadataBuilder, set_forward_context
from ..outputs import ModelRunnerOutput
from ..outputs import DraftTokenIds
from ..sample import Sampler, SamplingMetadata
from ..spec_decode.metadata import SpecDecodeMetadata
from ..spec_decode.rejection_sampler import RejectionSampler
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


class ExecuteModelState(NamedTuple):
    """`execute_model()` 与 `sample_tokens()` 之间的临时状态。

    只放"这一轮采样需要的东西"：调度快照、logits、以及 **logits 行 → batch 行** 的映射。
    不放整个 Scheduler，也不放模型输入张量（那些用完即弃，留着只会把显存占住）。
    """

    scheduler_output: object
    logits: torch.Tensor
    sample_rows: list[int]
    spec_metadata: object = None


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
        self.attn_metadata_builder = AttentionMetadataBuilder(self.block_size)
        self.sampler = None                       # load_model() 之后才有（采样要用模型精度）
        # 投机的两个部件（57E）：拒绝采样器复用普通采样器；提议器由 load_model() 按配置建
        self.rejection_sampler = RejectionSampler(Sampler())
        self.proposer = None
        self.speculative_config = vllm_config.speculative_config
        # 本轮验证时顺手提的下一轮草稿（`post_step` 取走）；概率留到下一轮按采用的前缀重排
        self.pending_draft_token_ids = None
        self.pending_draft_probs = None
        # 本轮协议里"整表替换过块表"的请求（draft 侧据此重置自己的进度）
        self._resumed_req_ids: set[str] = set()
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.execute_model_state: ExecuteModelState | None = None
        self.failure: str | None = None

    # -------- 初始化 --------

    def load_model(self):
        """装模型。等价于 vLLM 的 `GPUModelRunner.load_model()`（那里的 `self.model = ...`）。"""
        from ..model_loader import get_model

        if self.model is None:
            self.model = get_model(self.vllm_config.model_config, self.device)
        from ..sample import Sampler

        self.sampler = Sampler()
        self.proposer = self._build_proposer()
        return self.model

    def _build_proposer(self):
        """按配置建提议器：`ngram` 用历史匹配，`draft_model` 再加载一个小模型（57E）。

        没有投机配置 → 没有提议器 → `take_draft_token_ids()` 恒为 None，
        Scheduler 那边也不会收到草稿（整条路径是关的）。
        """
        config = self.speculative_config
        if config is None:
            return None
        if config.method == "ngram":
            from ..spec_decode.ngram_proposer import NgramProposer

            return NgramProposer(config.num_speculative_tokens)
        if config.method == "draft_model":
            from ..spec_decode.draft_model import DraftModelProposer

            proposer = DraftModelProposer(config, self.vllm_config, self.device)
            proposer.load_model()
            return proposer
        raise ValueError(f"未知的投机方法 {config.method!r}（本关支持 'ngram' / 'draft_model'）")

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
        # 1) 结束的请求：删镜像 + 删批行（本轮结束的请求，执行侧不再为它保留任何状态）
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.input_batch.remove_request(req_id)
            # 57E：这里还要丢掉它的草稿/提议状态与随机流

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
        # "已提交历史"的位置写；协议里没带的请求会被清空）
        for req_id in scheduled_req_ids:
            self.input_batch.update_req_spec_token_ids(
                req_id, scheduler_output.scheduled_spec_decode_tokens)

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
        # 不是"最后一行"；K=0 的请求退化成最后一行，与上面完全一致
        spec_metadata = None
        if self.speculative_config is not None:
            spec_metadata = SpecDecodeMetadata.from_scheduled(
                scheduler_output.scheduled_spec_decode_tokens,
                scheduler_output.num_scheduled_tokens, self.input_batch.req_ids)
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
            hidden_states = self.model(input_ids, positions)
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
        return {layer_name: metadata for layer_name in self._attention_layers()}

    # -------- 执行侧的两步协议 --------

    @torch.inference_mode()
    def execute_model(self, scheduler_output):
        """第一步：合并状态、跑模型、存下 logits。返回 `None` 表示"等 sample_tokens"。

        **`torch.inference_mode()` 不是装饰性的**（vLLM 在同样的位置也有这个装饰器）：模型的
        参数默认 `requires_grad=True`，而 KV 写入是 `index_copy_`——没有这个边界的话，每次
        写入都会记一个 `CopySlices` 反向图挂在 KV 缓存上，并且**一步一步累积**（每一步多 30 个
        图节点）。推理引擎里这意味着显存随步数单调上涨（验收方的独立探针就是查这个）。

        为什么不在 `load_model` 上也加：那会让权重本身变成"推理张量"，而这些权重还要被
        测试里的直接前向用到；本关只在**每步的入口**（执行与采样）划这条线。
        """
        self._check_usable()
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
        try:
            hidden_states = self._run_model(inputs)
            # LM head 只做在**要采样的行**上（每请求末行、且已 ready）。这是本关"按采样行选
            # hidden"的落点：隐藏态是给所有 token 算的，词表 GEMM 不是。
            #
            # 投机批例外：一个请求要 K+1 行（验证 + bonus），行是按请求块排的、不在
            # `sample_rows` 的坐标系里，所以整块都要算（不 ready 的请求由结果侧丢弃）
            spec_metadata = inputs.spec_metadata
            if spec_metadata is not None and spec_metadata.num_draft_tokens_total > 0:
                sample_indices = inputs.logits_indices
            else:
                sample_indices = inputs.logits_indices[inputs.sample_rows]
            logits = self.model.compute_logits(
                hidden_states.index_select(0, sample_indices.to(self.device)))
        except Exception as exc:                     # noqa: BLE001 —— 任何异常都让 Runner 停摆
            self.execute_model_state = None
            self.failure = f"{type(exc).__name__}: {exc}"
            raise
        self.execute_model_state = ExecuteModelState(
            scheduler_output=scheduler_output, logits=logits, sample_rows=inputs.sample_rows,
            spec_metadata=inputs.spec_metadata)
        return None

    @torch.inference_mode()
    def sample_tokens(self, grammar_output=None):
        """第二步：消费 logits 采样，并产出 `ModelRunnerOutput`（同样在推理边界内）。

        `grammar_output`（结构化输出）本关不用，但接口留着：执行与采样分开的意义之一就是给
        这类"采样前还要改一遍 logits"的路径留位置。
        """
        self._check_usable()
        state = self.execute_model_state
        if state is None:
            raise RuntimeError("没有待采样的 execute 结果：sample_tokens() 必须跟在 "
                               "返回 None 的 execute_model() 之后")
        self.execute_model_state = None

        scheduled_spec = state.scheduler_output.scheduled_spec_decode_tokens
        spec_metadata = state.spec_metadata
        if spec_metadata is not None and spec_metadata.num_draft_tokens_total > 0:
            # ---- 投机路径：验证草稿 ----
            # 元数据要按**被调度的请求**建（草稿的"假设历史"来自协议里的 spec tokens），
            # 行序与 spec_metadata 的请求顺序一致，长度是 K+1 而不是 1
            sampled = self._sample_with_spec(state, spec_metadata, scheduled_spec)
        else:
            # ---- 普通路径（含全批 K=0）----
            sampling_metadata = SamplingMetadata.from_input_batch(
                self.input_batch, state.sample_rows, device=self.device,
                scheduled_spec_decode_tokens=scheduled_spec)
            sampler_output = self.sampler.forward(state.logits, sampling_metadata)
            sampled = sampler_output.sampled_token_ids.tolist()

        # 先记账、再提草稿（顺序不能反，199 §4 的时序）：
        # `_bookkeeping_sync` 把本轮采样结果写进镜像，**提议必须看到它**——
        # 否则草稿是基于"少一个 token 的历史"算出来的，draft 侧的进度也会比 target 落后一格，
        # 于是发布出去的完整块可能还没被 draft 算过（验收方的独立探针分别抓到了这两点）
        output = self._bookkeeping_sync(state, sampled)
        # 只对**本轮已 ready**（历史算完）的请求提草稿：中间 prefill 块既没有"next token"，
        # 它的 KV 槽位也还没分配完（草稿要写在 target query 之外）。
        # vLLM 在 Scheduler.update_draft_token_ids 里同样跳过 prefill 块
        # （"Ignore draft tokens for prefill chunks"）——这里更早一步就不提。
        self.pending_draft_token_ids = self._propose_draft_tokens(ready_rows=state.sample_rows)
        return output

    # -------- 投机（57E）--------

    def _sample_with_spec(self, state, spec_metadata, scheduled_spec):
        """验证草稿，摊成**每条请求一串 token**（无效位置裁掉、非 ready 的给空）。

        拒绝采样器给的是 `[B, max_spec_len+1]` 的 padded 张量（无效位置 -1）；这里做裁剪与
        "请求 → 行"的映射。**元数据按整个批建**（行序 = `input_batch.req_ids`），因为草稿的
        "假设历史"来自协议里的 spec tokens，而不是我们筛出来的采样行。
        """
        sampling_metadata = SamplingMetadata.from_input_batch(
            self.input_batch, list(range(self.input_batch.num_reqs)), device=self.device,
            scheduled_spec_decode_tokens=scheduled_spec)
        draft_probs = self._align_draft_probs(spec_metadata)
        output = self.rejection_sampler.forward(
            spec_metadata, state.logits, draft_probs, sampling_metadata)
        rows = output.sampled_token_ids.tolist()

        ready = set(state.sample_rows)
        sampled_by_row: dict[int, list[int]] = {}
        for index, req_id in enumerate(self.input_batch.req_ids):
            row = self.input_batch.req_id_to_index[req_id]
            sampled_by_row[row] = (
                [token for token in rows[index] if token != -1] if row in ready else [])
        return [sampled_by_row[row] for row in state.sample_rows]

    def _propose_draft_tokens(self, ready_rows=None):
        """按配置提**下一轮**的草稿（199 §4：轮 t 验证时顺手提，轮 t+1 才采用）。

        提议的输入只看**已提交历史**（不看本轮自己的草稿区）；返回的草稿与概率只留到
        下一轮的 `sample_tokens`，届时按实际采用的条数重排 q。
        `ready_rows` 非空时只给这些行提（中间 prefill 块不提）。
        """
        if self.proposer is None:
            return None
        if ready_rows is None:
            req_ids = list(self.input_batch.req_ids)
        else:
            req_ids = [self.input_batch.req_id_at(row) for row in ready_rows]
            if not req_ids:
                return None
        all_token_ids = {req_id: self.requests[req_id].all_token_ids for req_id in req_ids}
        num_tokens_no_spec = {}
        for req_id in req_ids:
            row = self.input_batch.req_id_to_index[req_id]
            num_tokens_no_spec[req_id] = self.input_batch.num_tokens(row)
        # 恢复过的请求：它的块表整表换过，draft 侧的历史进度不再成立 → 重置
        drafts = self.proposer.propose(req_ids, all_token_ids, num_tokens_no_spec,
                                       self.input_batch,
                                       reset_req_ids=set(self._resumed_req_ids))
        # 概率按请求存：下一轮可能只采用每条请求的**前缀**，所以要留下每条的块边界
        self.pending_draft_probs = drafts if drafts.draft_probs is not None else None
        return drafts

    def _align_draft_probs(self, spec_metadata):
        """把上一轮存的 `[P_prev, V]` 草稿概率重排成本轮 `[P, V]`（199 §5 的 q 对齐）。

        上一轮提草稿的顺序与本轮采用的顺序一般**不同**（行序会变、每条还可能被预算截短），
        所以要按"每请求取前 K_i 行"重新拼，**不能**拿原矩阵的前 P 行——那会把 A 的概率
        配到 B 的草稿上。确定性提议（ngram）没有概率，返回 None。
        """
        previous = self.pending_draft_probs
        if previous is None:
            return None
        offsets: dict[str, int] = {}
        cursor = 0
        for req_id, draft_ids in zip(previous.req_ids, previous.draft_token_ids):
            offsets[req_id] = cursor
            cursor += len(draft_ids)
        rows: list[int] = []
        for req_id, num_draft in zip(spec_metadata.req_ids,
                                     spec_metadata.num_draft_tokens):
            if num_draft == 0:
                continue
            start = offsets.get(req_id)
            if start is None:
                raise RuntimeError(
                    f"{req_id!r} 本轮采用了草稿，但上一轮没有为它提过："
                    f"q 对不上（草稿必须与本轮采用的前缀同源）")
            rows.extend(range(start, start + num_draft))
        if not rows:
            return None
        return previous.draft_probs[rows]

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

    def _bookkeeping_sync(self, state: ExecuteModelState, sampled: list[list[int]]):
        """把采样结果**散射回所有被调度的请求**，并更新镜像。

        未 ready 的请求返回 `[]`（198 §4 允许的显式教学差异：本机是"先算采样行、再在
        `_bookkeeping_sync` 里丢掉不该提交的结果"，本关直接从采样阶段就不算它们）。

        这里最容易错的是"筛行之后还拿原列表 zip"：`sampled` 的行号是 **batch 行**，
        结果要按 `req_id` 摊回 `num_scheduled_tokens` 的顺序。所以映射
        `sample_row → req_id` 必须显式重建。
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
            draft_computed_tokens=(
                {req_id: self.proposer.draft_computed(req_id)
                 for req_id in self.input_batch.req_ids}
                if self.proposer is not None else None),
        )

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
