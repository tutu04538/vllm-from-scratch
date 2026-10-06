"""`InputBatch`：执行侧的**批状态缓冲**（对应 vLLM `v1/worker/gpu_input_batch.py` 的子集）。

它是"请求 → batch 行"的账本。Runner 每轮要的不是"新建一堆模型对象"，而是**在一组定长缓冲
上原地写**：

    [max_num_reqs]                  req_ids / req_id_to_index
    [max_num_reqs, max_model_len]   token_ids_cpu      ← 已知的 token 历史（prompt + 已产出）
    [max_num_reqs]                  num_computed_tokens_cpu / num_tokens_no_spec / num_prompt_tokens
    block_table                     [max_num_reqs, max_num_blocks_per_req]（每个 KV group 一份）
    sampling_params / generators    每行的采样配置与随机流（本关放普通 list，没编成 GPU 张量）
    采样用的定长 CPU 张量            temperature / top_k / top_p / min_p / 三种惩罚（57D、68）

`req_output_token_ids[row]` 存的是**请求镜像那个 list 的引用**（vLLM 同款）：`_bookkeeping_sync`
往里 append 之后，下一轮建 `SamplingMetadata` 时立刻能看到——惩罚历史因此不需要另存一份。

**行号不是身份**：`req_id_to_index` 是唯一的对应关系，行号会因为 `condense()` 而变。所以
generator、采样参数都必须跟着**行**搬运（见 `_move_row`），不能让 seed 跟着显存行号走——
"同一行换个请求，随机流就串了"是这类设计最典型的错。

**三种角色**（198 §2）：Scheduler 是权威；`CachedRequestState`（在 Runner 里）是执行端的
请求镜像；本类的缓冲是**GPU 输入的 CPU 副本**。缓冲里的历史来自执行侧已知的输出，
Scheduler 通过进度/输出长度快照校正它——三者不是三处都能随便改的真相。

**没有实现**（vLLM 有、本关不需要）：多 KV group 的块表列表、sampling metadata 的 GPU 张量化、
`num_accepted_tokens`（投机）、logprobs、lora、多模态占位。
"""

import torch

from ..sampling_params import SamplingParams
from .block_table import BlockTable


class InputBatch:
    def __init__(self, max_num_reqs: int, max_model_len: int, device: str,
                 block_size: int, vocab_size: int | None = None) -> None:
        # 词表大小只用于一件事：判断 top_k 是不是"等于不筛"（见 _write_sampling_params）。
        # None = 不知道（假执行路径不采样）；此时任何 top_k > 0 都按"要筛"处理。
        self.vocab_size = vocab_size
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.device = device

        self._req_ids: list[str | None] = []          # 允许空洞（remove 之后、condense 之前）
        self.req_id_to_index: dict[str, int] = {}
        self.token_ids_cpu = torch.zeros((max_num_reqs, max_model_len), dtype=torch.int64)
        self.num_computed_tokens_cpu = torch.zeros(max_num_reqs, dtype=torch.int64)
        self.num_tokens_no_spec = torch.zeros(max_num_reqs, dtype=torch.int64)
        # 每行的采样配置与随机流。用 list 而不是张量：本关的采样器是逐请求跑的（57D 才把
        # 温度/top_k 编成 GPU 张量、变成一次批采样）。
        self.sampling_params: list[SamplingParams | None] = [None] * max_num_reqs
        self.generators: list[torch.Generator | None] = [None] * max_num_reqs
        # 已提交的输出（与 CachedRequestState 共享同一个 list 对象，见模块说明）
        self.req_output_token_ids: list[list[int]] = [[] for _ in range(max_num_reqs)]
        # 本轮采用的草稿（57E）：写在 `token_ids_cpu` 里"已提交历史"之后，**不算**已提交。
        # 目标模型的 query 就是 [b, d1..dK]，所以它们必须在输入缓冲里（vLLM 同款）
        self.spec_token_ids: list[list[int]] = [[] for _ in range(max_num_reqs)]
        self.num_prompt_tokens = torch.zeros(max_num_reqs, dtype=torch.int64)
        # 采样用的按行定长 CPU 张量：建一次、每行原地写（对应 vLLM 的 *_cpu_tensor）
        self.temperature_cpu = torch.zeros(max_num_reqs, dtype=torch.float32)
        self.top_k_cpu = torch.zeros(max_num_reqs, dtype=torch.int64)
        self.top_p_cpu = torch.zeros(max_num_reqs, dtype=torch.float32)
        # 68 关：min_p（上游是 MinPLogitsProcessor 的状态张量；这里与其它采样参数并排存）
        self.min_p_cpu = torch.zeros(max_num_reqs, dtype=torch.float32)
        self.presence_penalties_cpu = torch.zeros(max_num_reqs, dtype=torch.float32)
        self.frequency_penalties_cpu = torch.zeros(max_num_reqs, dtype=torch.float32)
        self.repetition_penalties_cpu = torch.zeros(max_num_reqs, dtype=torch.float32)

        # 本关只有一个 KV group，所以只有一张块表（vLLM 是每个 group 一张的列表）
        max_num_blocks_per_req = (max_model_len + block_size - 1) // block_size
        self.block_table = BlockTable(block_size, max_num_reqs, max_num_blocks_per_req, device)

    # -------- 视图 --------

    @property
    def req_ids(self) -> list[str]:
        """紧凑批的有效 ID。`condense()` 之后 `_req_ids` 里没有 None。"""
        return [req_id for req_id in self._req_ids if req_id is not None]

    @property
    def num_reqs(self) -> int:
        """当前批的**行数**（含中间空洞）。它决定模型要跑几行，所以 `condense()` 之后才有意义。"""
        return len(self._req_ids)

    def num_tokens(self, row_index: int) -> int:
        """该行已知的 token 数（prompt + 已产出）。"""
        return int(self.num_tokens_no_spec[row_index])

    def num_tokens_with_spec(self, row_index: int) -> int:
        """已提交 token + 本轮草稿。**执行侧的 ready 判据用它**（与 Request 上的同名属性同义）。"""
        return self.num_tokens(row_index) + len(self.spec_token_ids[row_index])

    def update_req_spec_token_ids(self, req_id: str, scheduled_spec_tokens: dict) -> None:
        """按协议把本轮采用的草稿写进输入缓冲（对应 vLLM 同名方法）。

        写在"已提交历史"之后、并单独记一份：这样既能当模型输入（草稿是 query 的一部分），
        又能被下一轮**整体覆盖**——草稿被拒之后就不该留在缓冲里。
        """
        row_index = self.req_id_to_index.get(req_id)
        if row_index is None:
            return
        spec_token_ids = list(scheduled_spec_tokens.get(req_id, ()))
        self.spec_token_ids[row_index].clear()
        if not spec_token_ids:
            return
        start = self.num_tokens(row_index)
        end = start + len(spec_token_ids)
        if end > self.max_model_len:
            raise ValueError(
                f"{req_id!r} 的草稿写到第 {end} 个 token，超过 max_model_len="
                f"{self.max_model_len}：调度侧该为本轮的草稿留出位置")
        self.token_ids_cpu[row_index, start:end].copy_(
            torch.tensor(spec_token_ids, dtype=torch.int64))
        self.spec_token_ids[row_index].extend(spec_token_ids)

    def prompt_token_ids(self, row_index: int) -> list[int]:
        """该行的 prompt token（从输入缓冲里切出来，不另存一份）。"""
        return self.token_ids_cpu[row_index, :int(self.num_prompt_tokens[row_index])].tolist()

    def req_id_at(self, row_index: int) -> str:
        """行号 → 请求 ID。**行号不是身份**，它是可变的（`condense` 之后会变），
        所以"采样结果属于谁"必须经过这张表，不能在筛行之后拿原列表直接 zip。"""
        req_id = self._req_ids[row_index]
        if req_id is None:
            raise ValueError(f"第 {row_index} 行是空的（已被 remove，还没 condense）")
        return req_id

    # -------- 增删 --------

    def add_request(self, request) -> int:
        """把一个 `CachedRequestState` 放进批里，返回行号。

        行从"第一个空洞"或"末尾"取（缓冲复用），**并把用不到的部分清零**：不清零的话，
        新请求可能读到上一条请求留下的 token（`token_ids_cpu` 是定长二维缓冲，不是新张量）。
        """
        if request.req_id in self.req_id_to_index:
            raise ValueError(f"{request.req_id!r} 已经在批里（行 "
                             f"{self.req_id_to_index[request.req_id]}）")
        if len(self._req_ids) >= self.max_num_reqs:
            empty = [index for index, req_id in enumerate(self._req_ids) if req_id is None]
            if not empty:
                raise ValueError(
                    f"batch 行数已满（max_num_reqs={self.max_num_reqs}）："
                    f"Scheduler 的 max_num_seqs 必须 ≤ 执行侧的 max_num_reqs")
            row_index = empty[0]
        else:
            row_index = len(self._req_ids)
            self._req_ids.append(None)

        all_token_ids = request.all_token_ids
        if len(all_token_ids) > self.max_model_len:
            raise ValueError(f"{request.req_id!r} 有 {len(all_token_ids)} 个 token，"
                             f"超过 max_model_len={self.max_model_len}")
        self._req_ids[row_index] = request.req_id
        self.req_id_to_index[request.req_id] = row_index
        # 原地写（`copy_`），不重新分配：缓冲是复用的，写进去的是"这一行"的历史
        self.token_ids_cpu[row_index].zero_()
        self.token_ids_cpu[row_index, :len(all_token_ids)].copy_(
            torch.tensor(all_token_ids, dtype=torch.int64))
        self.num_computed_tokens_cpu[row_index] = request.num_computed_tokens
        self.num_tokens_no_spec[row_index] = len(all_token_ids)
        self.sampling_params[row_index] = request.sampling_params
        self.generators[row_index] = request.generator
        self.num_prompt_tokens[row_index] = len(request.prompt_token_ids)
        # **引用**请求镜像的输出列表（不复制）：append 之后采样侧立刻可见
        self.req_output_token_ids[row_index] = request.output_token_ids
        self._write_sampling_params(row_index, request.sampling_params)
        self.block_table.add_row(row_index, request.block_ids[0])
        return row_index

    def remove_request(self, req_id: str) -> None:
        """把一行标记为空。**不清缓冲**：下一行搬进来时（`_move_row` 或下次 `add_request`）
        才会被覆盖，这正是"复用缓冲"的代价与好处。"""
        row_index = self.req_id_to_index.pop(req_id, None)
        if row_index is None:
            return
        self._req_ids[row_index] = None
        self.sampling_params[row_index] = None
        self.generators[row_index] = None
        self.req_output_token_ids[row_index] = []
        self.num_prompt_tokens[row_index] = 0
        self.spec_token_ids[row_index] = []

    # -------- 重排 --------

    def condense(self) -> None:
        """把有效行压到前面，丢掉尾部的空洞（**保序**压实）。

        为什么必须压实：`_prepare_inputs` 用 `[0, num_reqs)` 这一段行去跑模型，中间有空洞就会
        把已移除的请求也算进去。vLLM 用"尾部行交换进空洞"来少搬几行；本关用保序压实——
        行序 == 加入顺序，`_prepare_inputs` 的数字才好逐值对照（差异账本里记着）。
        """
        write = 0
        for read in range(len(self._req_ids)):
            if self._req_ids[read] is None:
                continue
            if write != read:
                self._move_row(read, write)
            write += 1
        del self._req_ids[write:]

    def _write_sampling_params(self, row_index: int, sampling_params: SamplingParams) -> None:
        """把请求的采样参数写进按行的定长张量（采样时整块取用，不必逐行现读对象）。

        `top_k` 要按 vLLM 的规则**归一化**：只有 `0 < top_k < vocab_size` 才算"要筛"，
        其余（`-1` / `0` / `>= vocab_size`）一律写成 `vocab_size`——因为 `V - k = 0` 时
        阈值取到最小值，top-k 掩码等于什么都不屏蔽。这样"不筛"的行也能安全地留在同一张
        张量里混批，不需要在算子内部做分支或夹取。
        """
        self.temperature_cpu[row_index] = sampling_params.temperature
        top_k = sampling_params.top_k
        if self.vocab_size is not None and not 0 < top_k < self.vocab_size:
            top_k = self.vocab_size
        self.top_k_cpu[row_index] = top_k
        self.top_p_cpu[row_index] = sampling_params.top_p
        self.min_p_cpu[row_index] = sampling_params.min_p
        self.presence_penalties_cpu[row_index] = sampling_params.presence_penalty
        self.frequency_penalties_cpu[row_index] = sampling_params.frequency_penalty
        self.repetition_penalties_cpu[row_index] = sampling_params.repetition_penalty

    def _move_row(self, src: int, dst: int) -> None:
        """搬一行：**所有**按行存的缓冲都要跟着搬，漏一个就会出现"token 是 A 的、
        采样参数是 B 的"这类错。generator 尤其重要——它决定随机流跟谁走。"""
        req_id = self._req_ids[src]
        assert req_id is not None
        self._req_ids[dst] = req_id
        self._req_ids[src] = None
        self.req_id_to_index[req_id] = dst
        self.token_ids_cpu[dst] = self.token_ids_cpu[src]
        self.token_ids_cpu[src].zero_()
        self.num_computed_tokens_cpu[dst] = self.num_computed_tokens_cpu[src]
        self.num_tokens_no_spec[dst] = self.num_tokens_no_spec[src]
        self.sampling_params[dst] = self.sampling_params[src]
        self.generators[dst] = self.generators[src]
        self.req_output_token_ids[dst] = self.req_output_token_ids[src]
        self.num_prompt_tokens[dst] = self.num_prompt_tokens[src]
        self.spec_token_ids[dst] = self.spec_token_ids[src]
        self.spec_token_ids[src] = []
        for name in ("temperature", "top_k", "top_p", "min_p", "presence_penalties",
                     "frequency_penalties", "repetition_penalties"):
            tensor = getattr(self, f"{name}_cpu")
            tensor[dst] = tensor[src]
        self.block_table.move_row(src, dst)
