"""用一个**小模型**当提议者（对应 vLLM `v1/spec_decode/llm_base_proposer.py` +
`draft_model.py`）。

拆成两个类，位置与 vLLM 一致：

    SpecDecodeBaseProposer   提议的**步骤**：自己的输入缓冲、自己的 KV、自己的 attention 元数据
    DraftModelProposer       与"用哪个模型"有关的：造 draft 的 VllmConfig、加载、规格校验

### 它为什么不是"第二套引擎"

它没有调度器、没有请求状态、没有输出提交：只做一件事——**给定本轮 target 侧的事实，提 K 枚草稿**。

    第一遍 forward：把 [本轮起点, 采样后有效历史末尾) 喂给 draft 模型（含 prefix 命中之后的续算），
                    写进自己的 KV；扩容行的 hidden 顺便得到**第一枚**草稿
    自回归 K-1 步：每步一行（上一枚草稿当输入），**复用同一个工作区**，得到其余草稿

第一遍为什么需要：draft 要提出**像样**的草稿，它的 KV 必须覆盖与 target 相同的那段历史；
57 的设计是"每轮与 target 跑同一段位置"（中间 prefill 块也同步），所以发布出去的完整块
在 draft 那几层也都写过（199 §9 的不变量）。

### 58：第一遍的输入怎么组织（padded + mask，与上游同形）

普通自回归 draft 的第一遍，每条请求的**物理行**是：

    [有效行 (n - num_rejected)] + [1 行扩容行（新采出的 token）] + [num_rejected 行被拒行]
    总物理行 = Σ(target 本轮物理行数 + 1)          ← 与 target 的 input_budget 对得上

- **有效行** = target 本轮 query 里真正成为历史的那部分（起点 `start` 到采样后有效历史末尾）；
- **扩容行** = target 刚采出的那个 token（普通 draft 比 target 多要的就是这 1 行）；
  中间 prefill 块没有新 token，用 backup（本段历史最后一个 token）占位，草稿不提；
- **被拒行** = 留在工作区里但被屏蔽：token=padding、position=0、`slot=PADDING_SLOT_ID(-1)`，
  于是它既不写 KV、也不进新提议的上下文。展开规则与上游
  `copy_and_expand_eagle_inputs_kernel`（`shift_input_ids=False`）逐值一致，见 `utils.py`。

**prefix 命中**：起点直接用调度快照里的 `num_computed_tokens`（含命中起点），命中段不再重算——
它的前提正是上面的 199 §9 不变量（缓存交回来的块在 draft 每一层都有效）。58 之前用
`_draft_computed` 当第二套权威，新请求=0 于是命中 120 个 token 也会从 0 重算。

**固定工作区**：`input_ids/positions/slot_mapping/masks/query_start_loc/seq_lens/block_table`
在初始化时按 `max_num_batched_tokens` / `max_num_seqs` 开好，每轮只覆盖前 `[:num_tokens]` /
`[:num_reqs]`，`data_ptr()` 稳定；自回归步骤复用同一块工作区，不再重复计入输入预算。

### KV：共用一个 group，但每层是自己的 tensor

draft 与 target 共用**逻辑块表**与分配生命周期（同一张块表、同一套 slot 编号），
但每个 Attention 层绑自己的物理 tensor。所以同一块号在两边代表同一段位置，**不是同一份 K/V**。
加载时校验规格（词表、dtype、KV head 数、head_size、block_size），不兼容就明确报错——
不假装"所有小模型都能配对"。

### 本关的差异（写清楚，不假装已实现）

- `num_lookahead_tokens` = K：草稿里"下一轮才验证"的那 K 枚写在 target 本轮 query **之外**，
  Scheduler 分配块时多留 K 个槽位；上下文快满时预留会被 `max_model_len` 截掉，所以自回归
  每写一枚前还要过逻辑上界 + `BlockTable.covers` 两道检查（205 §3），过不了就少提几枚。
- 仍然**没有** vLLM 的 EAGLE / MTP / 并行提议（PARD/DFlash）分支：`max_num_new_slots_for_drafting`
  只有普通 draft 的 1 与 ngram 的 0 两条路径；`is_masked_token_mask` 缓冲留着但对齐的是
  "并行提议的多 query 槽位"，本关恒为 False。
- 中间 prefill 块**不提草稿**（只同步 KV）：vLLM 跑完 drafter 再让 Scheduler 丢掉草稿，
  本关在采样阶段就不采（少跑 K 次试探性前向，57E 起就记在差异账本里）。
- draft 不共享 target 的 random stream：自己按 `seed` 建 generator（可复现），
  与 target 的采样流相互独立（vLLM 也把 draft 的随机数分开算）。
- 不做 CUDA Graph / 编译（那是 69 关）：本关只保证输入工作区稳定复用，不要求中间算子零分配。
"""

import torch

from ..outputs import DraftTokenIds
from ..sample import Sampler
from ..sample.metadata import SAMPLING_EPS
from ..sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample
from ..attention import Attention, AttentionMetadataBuilder, set_forward_context
from .utils import (DraftInputRows, FirstPassPlan, TargetRows, compute_new_slot_mapping,
                    expand_draft_inputs, extend_all_queries_by_N)


def _dtype(name) -> torch.dtype:
    """配置里的 dtype 字符串 → torch dtype（本关只用到这两种）。"""
    return {"float32": torch.float32, "bfloat16": torch.bfloat16,
            "float16": torch.float16}.get(str(name), torch.float32)


class SpecDecodeBaseProposer:
    """提议步骤的骨架：**没有调度、没有请求状态**，只有"本轮哪些行进、草稿怎么出"。

    63 关：EAGLE 系的提议者还吃 target 的 hidden states，所以这里留了两个开关——
    `pass_hidden_states_to_model`（第一遍要传特征）与 `model_returns_tuple`（模型返回
    `(hidden_for_lm_head, hidden_for_next_step)` 两个张量，上游 `model_returns_tuple()` 同义）。
    """

    # 普通 draft 只吃 token；EAGLE 子类改成 True（上游 `EagleProposer.__init__` 传的就是它）
    pass_hidden_states_to_model = False
    # 模型是否返回 tuple（EAGLE3 的 forward 返回 `(hidden_states, hidden_prenorm)`）
    model_returns_tuple = False

    def __init__(self, spec_config, vllm_config, device: str) -> None:
        self.spec_config = spec_config
        self.method = spec_config.method      # 上游 `SpecDecodeBaseProposer` 同样存这份
        self.vllm_config = vllm_config
        self.device = device
        self.num_speculative_tokens = spec_config.num_speculative_tokens
        self.block_size = vllm_config.cache_config.block_size
        # **逻辑**上界。块表容量是按块向上取整的（10 个位置可能给 12 个槽位），
        # 所以"物理槽位够"不等于"模型允许写这个位置"——两个边界要分别检查（205 §3）。
        self.max_model_len = vllm_config.model_config.max_model_len
        # 普通自回归 draft：第一遍比 target query 多要 1 行输入（= 新采出的那个 token）
        self.num_new_slots_per_request = spec_config.max_num_new_slots_for_drafting
        self.model = None                     # 子类加载
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.metadata_builder = AttentionMetadataBuilder(self.block_size)
        self.sampler = Sampler()
        # **观测字段，不参与任何决策**（58 §5）：上一次第一遍把 draft 的 KV 覆盖到哪。
        # 58 之前的版本用它当"从哪开始补算"的第二套权威（新请求=0 于是命中前缀也整段重算）；
        # 现在起点由本轮调度快照的 `TargetRows.start` 决定，这里只留给测试/排查对账。
        self._draft_computed: dict[str, int] = {}
        self._draft_generators: dict[str, torch.Generator] = {}
        # 63 关：AR 步要用的 draft 自己的 hidden（每请求一行，`_forward` 的第二返回值）
        self._ar_hidden: dict[str, torch.Tensor] = {}
        self.num_drafts_proposed = 0
        # ---------------- 固定输入工作区（58 §7） ----------------
        # 一次性开好、每轮只覆盖前 N 行/N 个请求：**不每轮新建张量**（`data_ptr()` 稳定，
        # 为后面的编译/CUDA Graph 留地基）。CPU 上 staging 与 device 侧是同一份；
        # CUDA 上先写 staging、再只上传有效前缀。
        scheduler_config = vllm_config.scheduler_config
        self.max_num_reqs = scheduler_config.max_num_seqs
        # 容量口径与 Scheduler 的 input_budget 起点**同一个配置**（58 §5），不另加一个
        # "调大一点逃避预算"的旋钮。
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_blocks_per_req = max(1, -(-self.max_model_len // self.block_size))

        def _buffer(size, dtype):
            cpu = torch.zeros(size, dtype=dtype)
            if device == "cpu":
                return cpu, cpu
            return cpu, torch.zeros(size, dtype=dtype, device=device)

        self.input_ids_cpu, self.input_ids = _buffer((self.max_num_tokens,), torch.int64)
        self.positions_cpu, self.positions = _buffer((self.max_num_tokens,), torch.int64)
        self.slot_mapping_cpu, self.slot_mapping = _buffer((self.max_num_tokens,), torch.int64)
        self.is_rejected_token_mask_cpu, self.is_rejected_token_mask = _buffer(
            (self.max_num_tokens,), torch.bool)
        # 并行提议（DFlash/PARD）才用到的 mask；普通 draft 恒 False，缓冲先留着对齐布局
        self.is_masked_token_mask_cpu, self.is_masked_token_mask = _buffer(
            (self.max_num_tokens,), torch.bool)
        self.query_start_loc_cpu, self.query_start_loc = _buffer((self.max_num_reqs + 1,),
                                                                 torch.int64)
        self.seq_lens_cpu, self.seq_lens = _buffer((self.max_num_reqs,), torch.int64)
        self.block_table_cpu, self.block_table = _buffer(
            (self.max_num_reqs, self.max_blocks_per_req), torch.int64)
        # 63 关：EAGLE 的第一遍要带 target 的 hidden states（工作区定长、原位覆盖）
        hidden_size = int((vllm_config.model_config.hf_config or {}).get("hidden_size", 0))
        self.hidden_size = hidden_size
        # EAGLE3 收的是**多个辅助层拼接后**的特征（宽度 = hidden × 辅助层数），EAGLE-1 是 1 层
        draft_hf = (spec_config.draft_model_config.hf_config or {}) \
            if spec_config.draft_model_config is not None else {}
        aux_ids = draft_hf.get("eagle_aux_hidden_state_layer_ids") or \
            (draft_hf.get("eagle_config") or {}).get("eagle_aux_hidden_state_layer_ids")
        num_aux = len(aux_ids) if aux_ids else int(draft_hf.get("num_aux_layers", 1) or 1)
        self.num_aux_layers = max(int(num_aux), 1)
        dtype = _dtype(vllm_config.model_config.dtype)
        # 模型收到的是**投影后**的特征（宽度 = draft 的 hidden）：多辅助层的拼接与 fc 投影
        # 由提议者在写缓冲之前用 `combine_hidden_states()` 完成（上游同在 proposer 里做）
        self.hidden_states_cpu, self.hidden_states = _buffer(
            (self.max_num_tokens, hidden_size), dtype)

    # -------- 交给子类 --------

    def load_model(self) -> None:
        raise NotImplementedError

    # -------- 提议 --------

    def propose(self, rows: list[TargetRows], all_token_ids: dict[str, list[int]], input_batch,
                reset_req_ids: set[str] | None = None,
                target_hidden_states: dict[str, torch.Tensor] | None = None) -> DraftTokenIds:
        """对每个被调度的请求都跑一遍：**同步 KV**，并给其中 ready 的那些提草稿。

        `rows` 是本轮 target 侧的事实（`TargetRows`：起点 `start`、本轮行数 `target_rows`、
        被拒数 `num_rejected`、采样后有效历史末尾 `history_end`）。**起点直接来自调度快照**
        ——prefix 命中过的请求 `start` 就是命中末尾，draft 不再从位置 0 重算（58 §4/§6）。

        中间 prefill 块也要同步（`ready=False`）：同一个逻辑块在 target/draft 的每一层都有
        各自的 tensor，draft 没写过的那一层会让这个块不能算"完整可复用"，而 target 的完整块
        这一轮就会发布出去（199 §9）。**草稿只给 ready 的请求提**：中间 prefill 块没有可验证的
        next token，提了 Scheduler 也会丢（vLLM 的 `update_draft_token_ids` 同款规则）。

        **生命周期**（205 §4.4，与 Controller 的分工）：没被调度的请求根本不进 `rows`，
        状态保留；抢占恢复（`reset_req_ids`）只重置进度、随机流继续；结束/abort 由
        `remove_requests()` 显式删除。
        """
        self._reset_requests(reset_req_ids or set())
        req_ids = [target.req_id for target in rows]
        drafts: dict[str, list[int]] = {req_id: [] for req_id in req_ids}
        probs: dict[str, list[torch.Tensor]] = {req_id: [] for req_id in req_ids}
        if not rows:
            return DraftTokenIds(req_ids=[], draft_token_ids=[], draft_probs=None)

        # ---- 第一遍：把 [start, history_end) 这段写进 draft 的 KV（含 prefix 命中的跳过）----
        self._fill_block_table_rows([target.row for target in rows], input_batch.block_table)
        plan = self.set_inputs_first_pass(rows, all_token_ids, target_hidden_states)
        hidden = self._forward(plan.num_tokens, plan.num_reqs)
        hidden = self._split_hidden(hidden, plan, rows)
        for target in rows:
            self._draft_computed[target.req_id] = target.history_end      # 观测用
        if plan.sample_rows:
            self._sample_draft_tokens(hidden,
                                      list(zip(plan.sample_req_ids, plan.sample_rows)),
                                      input_batch, drafts, probs)

        # ---- 自回归补足 K 枚：上一枚当输入，位置接在它后面（**复用同一个工作区**）----
        #
        # 第 k 枚草稿写在位置 `history_end + k - 2`，那是 target 本轮 query 之外的位置：
        # 调度侧为此预留了 `num_lookahead_tokens` 个 KV 槽位。但预留可能被 `max_model_len`
        # 截掉、上下文也可能刚好走到尽头，所以每写一枚都要过**两个**边界：模型自己的位置范围
        # （逻辑）与块表覆盖（物理）。过不了就少提几枚——草稿只是候选，不写就不会越界。
        while True:
            pending: list[tuple[TargetRows, int]] = []
            for target in rows:
                req_id = target.req_id
                if not 0 < len(drafts[req_id]) < self.num_speculative_tokens:
                    continue
                position = target.history_end + len(drafts[req_id]) - 1
                if not 0 <= position < self.max_model_len:
                    continue
                if not input_batch.block_table.covers(target.row, position):
                    continue
                pending.append((target, position))
            if not pending:
                break
            self._set_autoregressive_inputs(pending, drafts, input_batch)
            hidden = self._forward(len(pending), len(pending))
            if self.model_returns_tuple:
                logits_hidden, next_hidden = hidden
                for index, (target, _) in enumerate(pending):
                    self._ar_hidden[target.req_id] = next_hidden[index]
                hidden = logits_hidden
            self._sample_draft_tokens(
                hidden, [(target.req_id, index) for index, (target, _) in enumerate(pending)],
                input_batch, drafts, probs)

        probs_rows = [row for req_id in req_ids for row in probs[req_id]]
        self.num_drafts_proposed += sum(len(drafts[req_id]) for req_id in req_ids)
        return DraftTokenIds(
            req_ids=list(req_ids),
            draft_token_ids=[drafts[req_id] for req_id in req_ids],
            draft_probs=torch.stack(probs_rows) if probs_rows else None)

    def _split_hidden(self, hidden, plan: FirstPassPlan, rows: list[TargetRows]):
        """`model_returns_tuple` 时拆开 `(for_lm_head, for_next_step)` 并记住 AR 要用的那份。

        第一遍每请求的采样行在 `plan.sample_rows`（= 扩容行的全局行号），AR 步用的特征就是
        这一行对应请求的 `for_next_step`。
        """
        if not self.model_returns_tuple:
            return hidden
        logits_hidden, next_hidden = hidden
        for req_id, row in zip(plan.sample_req_ids, plan.sample_rows):
            self._ar_hidden[req_id] = next_hidden[row]
        return logits_hidden

    # -------- 第一遍输入（对应上游 set_inputs_first_pass） --------

    def set_inputs_first_pass(self, rows: list[TargetRows],
                              all_token_ids: dict[str, list[int]],
                              target_hidden_states=None) -> FirstPassPlan:
        """把第一遍的输入写进工作区，返回物理行数、采样行与各请求的 AR 起点。

        每条请求的物理行 = [有效行 (n - num_rejected)] + [1 行扩容行] + [被拒行]，展开规则与
        上游 `copy_and_expand_eagle_inputs_kernel`（`shift_input_ids=False`）一致，见
        `spec_decode/utils.py`；槽位用上游同款 `compute_new_slot_mapping()` 算，query/seq
        长度用 `extend_all_queries_by_N()` 扩。

        **拒绝尾部留在工作区里但被屏蔽**：token 取 padding、position 取 0、slot 取哨兵，
        于是它既不写 KV、也不进新提议的上下文（58 §6）。
        """
        input_rows = []
        for target in rows:
            tokens = all_token_ids[target.req_id]
            input_rows.append(DraftInputRows(
                valid_token_ids=[int(tokens[position])
                                 for position in range(target.start,
                                                       target.start + target.num_valid)],
                start=target.start,
                next_token_id=target.next_token_id,
                num_rejected=target.num_rejected))
        input_ids, positions, is_rejected, sample_indices = expand_draft_inputs(input_rows)
        num_tokens = len(input_ids)
        if num_tokens > self.max_num_tokens:
            raise RuntimeError(
                f"draft 第一遍要 {num_tokens} 行，超过输入工作区 {self.max_num_tokens} 行："
                f"Scheduler 的 input_budget 没兜住，属于控制面/执行面口径不一致（不是模型问题）")
        query_lens = [target.target_rows for target in rows]
        # target 的 query_start_loc 与 seq_lens，交给 extend_all_queries_by_N 各 +1 行/+1 长度
        query_start_loc = [0]
        for length in query_lens:
            query_start_loc.append(query_start_loc[-1] + length)
        seq_lens = [target.start + target.target_rows for target in rows]
        query_start_loc, seq_lens = extend_all_queries_by_N(
            query_start_loc, seq_lens, self.num_new_slots_per_request)

        self.input_ids_cpu[:num_tokens] = torch.tensor(input_ids, dtype=torch.int64)
        self.positions_cpu[:num_tokens] = torch.tensor(positions, dtype=torch.int64)
        self.is_rejected_token_mask_cpu[:num_tokens] = torch.tensor(is_rejected, dtype=torch.bool)
        self.query_start_loc_cpu[:len(query_start_loc)] = torch.tensor(query_start_loc,
                                                                      dtype=torch.int64)
        self.seq_lens_cpu[:len(seq_lens)] = torch.tensor(seq_lens, dtype=torch.int64)
        # slot mapping 在 **CPU** 上算：块表镜像是 CPU 结构，索引也必须是 CPU 张量
        # （否则会撞上 "Expected all tensors to be on the same device"，204 §5）。
        self.slot_mapping_cpu[:num_tokens] = compute_new_slot_mapping(
            self.block_table_cpu[:len(rows)], query_lens, self.positions_cpu[:num_tokens],
            self.is_rejected_token_mask_cpu[:num_tokens], self.block_size,
            self.num_new_slots_per_request, self.max_model_len)
        self._check_valid_positions(rows)
        ready = [index for index, target in enumerate(rows) if target.ready]
        return FirstPassPlan(
            num_tokens=num_tokens, num_reqs=len(rows),
            sample_rows=[sample_indices[index] for index in ready],
            sample_req_ids=[rows[index].req_id for index in ready],
            history_end={target.req_id: target.history_end for target in rows})

    def _check_valid_positions(self, rows: list[TargetRows]) -> None:
        """内部错误检查（对应 205 §3 保留的那条断言）：有效行/扩容行必须在模型位置上界内。

        被拒行不在检查范围（它们是 padding，position=0，本来就不参与上下文）。
        """
        for target in rows:
            last = target.start + target.num_valid          # 扩容行的位置
            if not 0 <= last < self.max_model_len:
                raise RuntimeError(
                    f"{target.req_id!r} 的 draft 第一遍要写位置 {last}，超出 "
                    f"max_model_len={self.max_model_len}：调度快照与 draft 输入口径不一致")

    def _set_autoregressive_inputs(self, pending: list[tuple[TargetRows, int]],
                                   drafts: dict, input_batch) -> None:
        """后续自回归步骤：**复用同一个工作区的前 B 行**（每活跃请求一行，58 §7）。"""
        num_reqs = len(pending)
        tokens = [drafts[target.req_id][-1] for target, _ in pending]
        positions = [position for _, position in pending]
        batch_rows = [target.row for target, _ in pending]
        self.input_ids_cpu[:num_reqs] = torch.tensor(tokens, dtype=torch.int64)
        self.positions_cpu[:num_reqs] = torch.tensor(positions, dtype=torch.int64)
        if self.pass_hidden_states_to_model:
            # 第 k 枚草稿的输入特征 = 第 k-1 枚那一步 draft 自己吐出的 hidden（上游同款）
            rows_hidden = [self._ar_hidden[target.req_id] for target, _ in pending]
            self.hidden_states_cpu[:num_reqs] = torch.stack(rows_hidden).to(
                self.hidden_states_cpu.dtype)
        # 槽位与 target 同一份公式、同一张块表（物理容量已由 covers() 确认过）
        self.slot_mapping_cpu[:num_reqs] = input_batch.block_table.compute_slot_mapping(
            input_batch.num_reqs, torch.tensor(positions, dtype=torch.int64),
            torch.tensor(batch_rows, dtype=torch.int64))
        self.is_rejected_token_mask_cpu[:num_reqs] = False
        self.query_start_loc_cpu[:num_reqs + 1] = torch.arange(num_reqs + 1, dtype=torch.int64)
        # 第 k 枚草稿的上下文 = 它自己的位置 + 1（之前几枚的 KV 已经写进缓存）
        self.seq_lens_cpu[:num_reqs] = torch.tensor(
            [position + 1 for _, position in pending], dtype=torch.int64)
        self._fill_block_table_rows(batch_rows, input_batch.block_table)

    def _fill_block_table_rows(self, batch_rows: list[int], block_table) -> None:
        """把这几条请求的块表行拷进**工作区块表**（只覆盖前 len(batch_rows) 行）。

        尾部残留的块号不会被读到：metadata 只带 `[:num_reqs]` 的有效切片（58 §7）。
        """
        for index, row in enumerate(batch_rows):
            count = block_table.num_blocks(row)
            if count > self.max_blocks_per_req:
                raise RuntimeError(
                    f"第 {row} 行有 {count} 个块，超过工作区 {self.max_blocks_per_req} 列："
                    f"工作区是按 max_model_len/block_size 开的，说明块表与本配置不匹配")
            self.block_table_cpu[index, :count] = torch.tensor(
                block_table.cpu[row, :count].tolist(), dtype=torch.int64)
            self.block_table_cpu[index, count:].zero_()

    def _upload(self, num_tokens: int, num_reqs: int) -> None:
        """只上传有效前缀（CPU 上 staging 与 device 侧是同一份，直接返回）。"""
        if self.device == "cpu":
            return
        self.input_ids[:num_tokens].copy_(self.input_ids_cpu[:num_tokens])
        self.positions[:num_tokens].copy_(self.positions_cpu[:num_tokens])
        self.slot_mapping[:num_tokens].copy_(self.slot_mapping_cpu[:num_tokens])
        self.is_rejected_token_mask[:num_tokens].copy_(
            self.is_rejected_token_mask_cpu[:num_tokens])
        self.query_start_loc[:num_reqs + 1].copy_(self.query_start_loc_cpu[:num_reqs + 1])
        self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs])
        self.block_table[:num_reqs].copy_(self.block_table_cpu[:num_reqs])

    def _forward(self, num_tokens: int, num_reqs: int, hidden_states=None):
        """把工作区的前 `num_tokens` / `num_reqs` 行交给 draft 模型，返回 hidden states。

        **只传有效切片**：缓冲尾部的残留 token / 块号不能被 attention 读到——用长度表达有效，
        不是"内容恰好是 0"（58 §7）。
        """
        self._upload(num_tokens, num_reqs)
        metadata = self.metadata_builder.build(
            query_start_loc=self.query_start_loc[:num_reqs + 1],
            seq_lens=self.seq_lens[:num_reqs],
            block_table=self.block_table[:num_reqs],
            slot_mapping=self.slot_mapping[:num_tokens],
            num_reqs=num_reqs)
        attn_metadata = {name: metadata for name in self.kv_caches}
        with set_forward_context(attn_metadata, num_tokens=num_tokens):
            if self.pass_hidden_states_to_model:
                features = self.hidden_states if hidden_states is None else hidden_states
                out = self.model(self.input_ids[:num_tokens], self.positions[:num_tokens],
                                 features[:num_tokens])
                if self.model_returns_tuple:
                    # 上游：`last_hidden_states, hidden_states = ret_hidden_states`
                    # —— lm_head 用前者，下一步 draft 用后者（EAGLE3 的 prenorm）
                    for_logits, for_next = out
                    return for_logits, for_next
                return out
            return self.model(self.input_ids[:num_tokens], self.positions[:num_tokens])

    def _reset_requests(self, reset_req_ids: set[str]) -> None:
        """丢掉不再成立的 draft 侧**进度**：请求刚被抢占恢复（块表整表换过）。

        恢复之后旧物理编号上的 KV 已经不属于它了，**必须**从头补，不能接着用。
        随机流**不重置**：请求还活着，只是换了块，重新 seed 会改变它的采样序列
        （205 §4.4：抢占恢复 ≠ 新请求）。
        """
        for req_id in reset_req_ids:
            self._draft_computed.pop(req_id, None)

    def remove_requests(self, req_ids) -> None:
        """请求**结束/abort**：进度与随机流一并删除（草稿概率 q 存在 Runner 上、
        每轮整体重算，不需要按请求清）。

        只有控制端明确说"这条结束了"才调它。**不能**用"不在本轮的 req_ids 里"代替：
        预算不够没排上、被抢占等待恢复的请求都不在 batch 里，但它们的状态必须留着
        （205 §4.1/§4.3 就是这两种情况混在一起造成的）。
        0-token 的结束清理轮也要调——否则最后一条请求结束后，复用的 ID 会继承旧进度。
        """
        for req_id in req_ids:
            self._draft_computed.pop(req_id, None)
            self._draft_generators.pop(req_id, None)

    # -------- 采样草稿 --------

    def _sample_draft_tokens(self, hidden: torch.Tensor, row_refs: list[tuple[str, int]],
                             input_batch, drafts: dict, probs: dict) -> None:
        """对给定行各采一枚草稿，同时记下它来自的分布（q）。

        `q` 必须是**实际提议时用的分布**（199 §7），所以这里与 `Sampler.sample` 走同一条
        约束链（温度 → top-k/top-p → 指数竞赛），把 softmax 之后的整行留下来。
        贪心行的"分布"是草稿位置上的点质量（one-hot），与 `draft_probs=None` 的点质量提议同义。
        """
        # **先过 LM head**：hidden 是隐藏态，采样要的是词表上的 logits（与 Runner 同一条路：
        # `compute_logits` 只对需要的行做词表 GEMM）
        logits = self.model.compute_logits(
            hidden[torch.tensor([row for _, row in row_refs],
                                dtype=torch.int64, device=hidden.device)]).to(torch.float32)
        for index, (req_id, _) in enumerate(row_refs):
            parameter = input_batch.sampling_params[input_batch.req_id_to_index[req_id]]
            row_logits = logits[index]
            probs_row = self._row_probs(row_logits, parameter)
            token = int(row_logits.argmax()) if parameter.is_greedy else int(
                random_sample(probs_row.unsqueeze(0),
                              {0: self._generator(req_id, parameter)})[0])
            drafts[req_id].append(token)
            probs[req_id].append(probs_row)

    def _row_probs(self, row_logits: torch.Tensor, parameter) -> torch.Tensor:
        """一行的提议分布：贪心 → one-hot（点质量）；否则与普通采样同一条约束链。"""
        if parameter.is_greedy:
            one_hot = torch.zeros_like(row_logits)
            one_hot[int(row_logits.argmax())] = 1.0
            return one_hot
        logits = row_logits / max(parameter.temperature, SAMPLING_EPS)
        top_k = None if parameter.top_k in (-1, 0) else torch.tensor(
            [parameter.top_k], device=logits.device)
        top_p = None if parameter.top_p >= 1.0 else torch.tensor(
            [parameter.top_p], device=logits.device)
        logits = apply_top_k_top_p(logits.unsqueeze(0), top_k, top_p)
        return logits.softmax(dim=-1, dtype=torch.float32)[0]

    def _generator(self, req_id: str, parameter):
        """draft 自己的随机流：按请求的 seed 建一次，之后**每轮接着用**（不是每轮重置）。"""
        generator = self._draft_generators.get(req_id)
        if generator is None and parameter.seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(parameter.seed + 1)   # 与 target 的流分开，避免逐位重合
            self._draft_generators[req_id] = generator
        return generator

    def _attention_layer_names(self) -> list[str]:
        return sorted(self.kv_caches)



class DraftModelProposer(SpecDecodeBaseProposer):
    """用**另一个模型目录**当提议者（对应 vLLM `v1/spec_decode/draft_model.py`）。"""

    def __init__(self, spec_config, vllm_config, device: str) -> None:
        super().__init__(spec_config, vllm_config, device)
        if spec_config.draft_model_config is None:
            raise ValueError("draft_model 投机必须在 SpeculativeConfig 里给 draft_model_config")
        self.draft_model_config = spec_config.draft_model_config

    # -------- 加载与校验 --------

    def load_model(self) -> None:
        from ..model_loader import get_model

        self._validate_configs()
        self.model = get_model(self.draft_model_config, self.device)
        self._allocate_kv_caches()
        return self.model

    def _validate_configs(self) -> None:
        """词表与 KV 规格必须兼容（199 §9）：不满足就**明确报错**，不静默降级。"""
        target_config = self.vllm_config.model_config.hf_config or {}
        draft_config = self.draft_model_config.hf_config or {}
        target_vocab = target_config.get("vocab_size")
        draft_vocab = draft_config.get("vocab_size")
        if target_vocab != draft_vocab:
            # 63 关：EAGLE3 允许 draft 词表更小 + 带 `d2t` 映射（异构词表 TLI）；
            # 但**采样空间**的完整语义属 67 关，这里只要求"映射存在"，否则明确报错。
            if not (self.method == "eagle3" and draft_config.get("draft_vocab_size")):
                raise ValueError(
                    f"draft 与 target 的词表不一致（{draft_vocab} vs {target_vocab}）："
                    f"草稿的 token 在 target 的词表里是另一个意思，验证没有意义")
        if str(self.draft_model_config.dtype) != str(self.vllm_config.model_config.dtype):
            raise ValueError("draft 与 target 的 dtype 必须一致（KV 缓存要放进同一个 group）")
        for key in ("num_key_value_heads", "head_dim", "max_position_embeddings"):
            if draft_config.get(key) != target_config.get(key):
                raise ValueError(
                    f"draft 与 target 的 {key} 不一致"
                    f"（{draft_config.get(key)} vs {target_config.get(key)}）："
                    f"本关只支持 KV 规格相同的 draft/target 共用一个 KV group")

    def _allocate_kv_caches(self) -> None:
        """给 draft 的每个 Attention 层分配**自己的**物理缓存。

        与 target 同形、同块数：块号在两边代表同一段位置。但**不是同一份 K/V**——
        target 写它自己的 tensor，draft 写自己的。
        """
        cache_config = self.vllm_config.cache_config
        dtype = next(self.model.parameters()).dtype
        self.kv_caches = {}
        for name, module in self.model.named_modules():
            if not isinstance(module, Attention):
                continue
            cache = torch.zeros(2, cache_config.num_gpu_blocks, self.block_size,
                                module.num_kv_heads, module.head_size, dtype=dtype,
                                device=self.device)
            module.kv_cache = cache
            self.kv_caches[name] = cache

