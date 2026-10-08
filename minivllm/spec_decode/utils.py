"""投机解码第一遍输入的纯函数与数据（对应 vLLM `v1/spec_decode/utils.py` 的相关两块）。

上游把 draft 第一遍的输入**在 GPU 上用 Triton kernel 拷贝 + 扩容**出来
（`copy_and_expand_eagle_inputs_kernel`），再做两件事：

    compute_new_slot_mapping(...)   算每一行的 KV 槽位（拒绝行/越界行 → 哨兵 PADDING_SLOT_ID）
    extend_all_queries_by_N(...)    把每请求的 query 行数与上下文长度各 + N

本地生产路径要能在 CPU 上跑，所以这里是**逐行 Torch/Python 的等价实现**，语义对齐 kernel 的
`shift_input_ids=False` 分支（普通自回归 draft，`num_padding_slots_per_request=1`）：

    每条请求的物理行 = [有效行 (n - num_rejected)] + [1 行扩容行] + [num_rejected 行被拒行]
    总物理行         = target 本轮物理行数 + 请求数          （58 §6 的不变量）
    positions        = start + 行内序号；被拒行取 0（反正它的 slot 是哨兵）
    token ids        = 有效行取历史 token、扩容行取新采样 token、被拒行取 padding
    is_rejected      = 只有被拒尾部是 1（它们物理存在，但**不能成为有效上下文**）
    token_indices_to_sample[i] = 第 i 条请求扩容行的全局行号

逐值对照见 `tests/step58/test_draft_inputs.py`（CUDA 上会与真 kernel 做差分测试）。
"""

from dataclasses import dataclass, field

import torch

# 与上游 `v1/spec_decode/utils.py` 同名：拒绝行 / 超出 max_model_len 的行用它，
# attention 的写入侧必须跳过（否则 index_copy_ 会把它当成"最后一个槽位"）。
PADDING_SLOT_ID = -1
# 上游 kernel 的 padding_token_id 由调用方传（runner 里传 0），本地保持一致。
PADDING_TOKEN_ID = 0


@dataclass(frozen=True)
class TargetRows:
    """一条请求本轮 **target 侧的事实**（由 Runner 从调度快照 + 记账结果填，提议者只读）。

    拆成这些字段是为了让"起点/终点"没有歧义（58 §5）：以前一个含糊的
    `num_computed_tokens` 同时想表达"target 算到哪"和"draft 该从哪补"，改名之后：

        start        本轮 target 起点 C（协议里的 `num_computed_tokens`，**含 prefix 命中起点**）
        target_rows  本轮 target 执行的行数 n
        history_end  采样后**有效历史**的末尾；未 ready 时 = start + target_rows
        num_rejected 本轮采用的草稿里被拒的条数（被拒尾部不进入新提议上下文）
    """

    req_id: str
    row: int                 # batch 行号（草稿读共享块表要用）
    start: int               # C
    target_rows: int         # n
    num_rejected: int
    history_end: int
    next_token_id: int       # 扩容行的 token：ready = 最后一个新采样 token；未 ready = backup
    ready: bool

    @property
    def num_valid(self) -> int:
        """第一遍的**有效行数** = n - 被拒行数（≤ n，配上扩容行后 ≤ n+1）。"""
        return self.target_rows - self.num_rejected


@dataclass(frozen=True)
class DraftInputRows:
    """一条请求第一遍要写进工作区的内容（`TargetRows` + 历史 token 展开后的结果）。"""

    valid_token_ids: list[int]
    start: int
    next_token_id: int
    num_rejected: int


@dataclass
class FirstPassPlan:
    """`set_inputs_first_pass()` 的结果：物理行数 + 采样行 + 每条请求的 AR 起点。"""

    num_tokens: int
    num_reqs: int
    # 要取 hidden 去采样第一枚草稿的行（= ready 请求扩容行的全局行号）。
    # 上游会给**所有**请求都填一行再让 Scheduler 丢掉未 ready 的草稿；本关连采样都不做，
    # 这是 57E 就记在差异账本里的"更早一步"（少跑 K 次试探性前向）。
    sample_rows: list[int] = field(default_factory=list)
    sample_req_ids: list[str] = field(default_factory=list)
    # 每条请求**采集样行自身的 position**（与 `sample_req_ids` 同序）。
    # 自回归步的起点就是它：上游 `positions = self.positions[token_indices_to_sample]`
    # （`llm_base_proposer.py:634`），之后每步 +1。EAGLE 与 draft 的采样行**不是同一行**
    # （EAGLE 是 target 行块的最后一行、draft 是尾部扩容行），所以这个位置必须由布局自己给出，
    # 不能在外面用 `history_end` 之类的量反推——反推只在 `rejected == 1` 时巧合相等
    # （实测 rejected=3 时反推出来的位置会落回第一遍刚写过的那几行，把 KV 覆盖掉）。
    sample_positions: list[int] = field(default_factory=list)
    # 第一遍用的逐请求 `seq_lens`（**乐观值**：含本轮 query 的全部行）。
    # 自回归步的上下文起点 = 它减去被拒行数（上游 `seq_lens -= num_rejected_tokens_gpu`），
    # 之后每步 +1。
    seq_lens: list[int] = field(default_factory=list)
    # 每条请求的 AR 起点（= 有效历史末尾）：第 k 枚草稿的输入位置 = history_end + k - 2
    history_end: dict[str, int] = field(default_factory=dict)
    # 69 关：每条请求本轮**被拒**的行数（= 采用数 - 接受数）。它决定"draft 的上下文长度要从
    # 乐观值里减掉多少"，也决定 padded 批里哪些行是 padding（见 `prepare_inputs_padded`）。
    num_rejected: list[int] = field(default_factory=list)


def expand_draft_inputs(rows: list[DraftInputRows]) -> tuple[list[int], list[int], list[int],
                                                             list[int]]:
    """上游 `copy_and_expand_eagle_inputs_kernel`（`shift_input_ids=False`）的等价展开。

    返回 `(input_ids, positions, is_rejected, token_indices_to_sample)` 四个扁平 list。
    这里不写 `is_masked_token_mask`：它对应的是**并行提议**的多 query 槽位（DFlash/PARD），
    普通 draft 分支里上游也全为 False（本关不做并行提议）。
    """
    input_ids: list[int] = []
    positions: list[int] = []
    is_rejected: list[int] = []
    token_indices_to_sample: list[int] = []
    for row in rows:
        base = len(input_ids)
        num_valid = len(row.valid_token_ids)
        input_ids.extend(row.valid_token_ids)
        positions.extend(range(row.start, row.start + num_valid))
        is_rejected.extend([0] * num_valid)
        # 扩容行：普通 draft 比 target query 多要的那 1 行
        input_ids.append(row.next_token_id)
        positions.append(row.start + num_valid)
        is_rejected.append(0)
        token_indices_to_sample.append(base + num_valid)
        # 被拒尾部：留在物理工作区里，但 token/position 是 padding，slot 会被打成哨兵
        input_ids.extend([PADDING_TOKEN_ID] * row.num_rejected)
        positions.extend([0] * row.num_rejected)
        is_rejected.extend([1] * row.num_rejected)
    return input_ids, positions, is_rejected, token_indices_to_sample


@dataclass
class ParallelFirstPass:
    """并行提议（PARD / P-EAGLE）第一遍的展开结果（上游 kernel 的全部输出）。

    与普通 draft 的 `expand_draft_inputs()` 是**同一个算法的两个分支**：上游把它们写在同一个
    `copy_and_expand_eagle_inputs_kernel` 里，靠 `shift_input_ids` 与 `num_padding_slots_per_request`
    两个参数分叉。字段逐个对应：

        input_ids / positions   每个物理行的 token 与位置
        is_rejected             被拒尾部（物理存在、不写 KV、不算上下文）
        is_masked               并行槽位（token = mask token；EAGLE 的 hidden 也会被换成 mask hidden）
        token_indices_to_sample 要采样的行（= 锚点 + 所有 mask 行，按 请求序 × 行内序 排列）
        hidden_state_mapping    {源行 → 目标行}：只有 shift 模式有（hidden **不跟着移**）
        num_tokens             总物理行数（= Σ(target 行数 + net)）
    """

    input_ids: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    is_rejected: list[int] = field(default_factory=list)
    is_masked: list[int] = field(default_factory=list)
    token_indices_to_sample: list[int] = field(default_factory=list)
    hidden_state_mapping: dict[int, int] = field(default_factory=dict)
    num_tokens: int = 0


def expand_parallel_draft_inputs(target_token_ids, target_positions, rows, *,
                                 extra_slots_per_request: int,
                                 parallel_drafting_token_id: int,
                                 shift_input_ids: bool) -> ParallelFirstPass:
    """上游 `copy_and_expand_eagle_inputs_kernel`（L308-454）的**逐行等价实现**。

    两种布局（`shift_input_ids`）就在"第 0 行放什么"上分叉：

        shift=True  （P-EAGLE / EAGLE 系）：跳过本请求 target 块的第 0 个 token，
                     行 j 放 `target[q_start + 1 + j]` → 有效行数 = target 行数 − 被拒 − 1
        shift=False （PARD / 普通 draft）：整块原样复制，
                     行 j 放 `target[q_start + j]`     → 有效行数 = target 行数 − 被拒

    之后两者的**块内结构完全一样**（j 从 0 数起）：

        [0, num_valid)                              有效行（token/位置来自 target）
        j == num_valid                              **锚点**：target 刚采出的 token（要采样）
        (num_valid, num_valid + extra)              并行槽位：token = mask token（要采样）
        [num_valid + extra, + num_rejected)         被拒尾部：token=padding、position=0、slot=哨兵

    位置一律 `start_pos + j`（**不随 shift 移动**，上游注释："Positions are NOT shifted"），
    被拒行取 0。要采样的行数 = `extra`（1 个锚点 + extra−1 个 mask），与 K 相等。

    **每请求多占的行数** = 总行数 − target 行数 = `num_valid + extra + rejected − target_rows`
    = `extra − (1 if shift else 0)`，也就是 `net_num_new_slots_per_request`
    （P-EAGLE：K−1，PARD：K）—— 调度侧的 `draft_slots` 就是按它给的。

    `rows` 是 `TargetRows`（顺序 = 批行序），`target_token_ids` / `target_positions` 必须是
    **本轮真正喂进 target 的那份缓冲**（63 关的不变量：同源，不许按"看起来等价"重建）。
    """
    out = ParallelFirstPass()
    cursor = 0                      # target 缓冲里本请求块的起点
    for row in rows:
        rejected = int(row.num_rejected)
        target_rows = int(row.target_rows)
        # 有效输入区的**闭区间**终点（上游 `query_end_loc = query_start_loc[1:] - 1 - rejected`）
        num_valid = target_rows - rejected - (1 if shift_input_ids else 0)
        if num_valid < 0:
            raise ValueError(
                f"{row.req_id!r} 的被拒行数 {rejected} 吃掉了整个 target 块"
                f"（{target_rows} 行）：并行第一遍至少要留 1 行有效输入"
                f"（shift 模式要留 2 行）")
        start_pos = int(target_positions[cursor])
        out_start = len(out.input_ids)
        for j in range(num_valid):
            index = cursor + (1 if shift_input_ids else 0) + j
            out.input_ids.append(int(target_token_ids[index]))
            out.positions.append(start_pos + j)
            out.is_rejected.append(0)
            out.is_masked.append(0)
        # 锚点行：target 本轮采出的 token（上游 `is_bonus_region`）
        out.input_ids.append(int(row.next_token_id))
        out.positions.append(start_pos + num_valid)
        out.is_rejected.append(0)
        out.is_masked.append(0)
        out.token_indices_to_sample.append(out_start + num_valid)
        # 并行槽位：K−1 个 mask token（上游 `is_parallel_draft_region`）
        for j in range(num_valid + 1, num_valid + extra_slots_per_request):
            out.input_ids.append(int(parallel_drafting_token_id))
            out.positions.append(start_pos + j)
            out.is_rejected.append(0)
            out.is_masked.append(1)
            out.token_indices_to_sample.append(out_start + j)
        # 被拒尾部：padding 行（不写 KV、不算上下文）
        for _ in range(rejected):
            out.input_ids.append(PADDING_TOKEN_ID)
            out.positions.append(0)
            out.is_rejected.append(1)
            out.is_masked.append(0)
        if shift_input_ids:
            # hidden **不跟着移**：源行 i（target 的第 i 行）的特征放到目标行 `out_start + i`
            for i in range(target_rows):
                out.hidden_state_mapping[cursor + i] = out_start + i
        cursor += target_rows
    out.num_tokens = len(out.input_ids)
    # 总行数 = Σ(target 行数) + 请求数 × net；被拒行**已经在 target 行数里**（不能重复计）
    expected = cursor + len(rows) * (extra_slots_per_request - (1 if shift_input_ids else 0))
    if out.num_tokens != expected:
        raise RuntimeError(
            f"并行第一遍展开出 {out.num_tokens} 行，按公式应为 {expected} 行："
            f"展开逻辑与 net_num_new_slots_per_request 的口径不一致")
    return out


def compute_new_slot_mapping(block_table: torch.Tensor,
                             query_lens: list[int],
                             new_positions: torch.Tensor,
                             is_rejected_token_mask: torch.Tensor,                             block_size: int,
                             num_new_tokens: int,
                             max_model_len: int) -> torch.Tensor:
    """上游 `compute_new_slot_mapping()` 的本地版（形状/公式逐行一致）。

    `block_table` 是**工作区**里的块表 `[num_reqs, max_blocks]`；`query_lens` 是 target 本轮
    每请求的物理行数（上游取 `cad.naive_query_lens()`），`num_new_tokens` 是每请求扩容的行数
    （普通 draft = 1）。于是 `new_positions` 的行数必须是 `sum(query_lens) + B * num_new_tokens`。

    规则（与上游同）：
      slot = block_table[req, pos // block_size] * block_size + pos % block_size
      pos >= max_model_len  → PADDING_SLOT_ID（越界行不写缓存）
      is_rejected_token_mask → PADDING_SLOT_ID（被拒尾部不写缓存）
    """
    batch_size, n_blocks_per_req = block_table.shape
    device = block_table.device
    req_indices = torch.repeat_interleave(
        torch.arange(batch_size, device=device),
        torch.tensor(query_lens, device=device, dtype=torch.int64) + num_new_tokens,
        output_size=new_positions.shape[0])
    # 先夹住再查表：越界位置本来就查不到块，夹住只是为了让索引合法（随后会被哨兵覆盖）
    clamped = new_positions.clamp(max=max_model_len - 1)
    block_nums = block_table.reshape(-1)[req_indices * n_blocks_per_req + clamped // block_size]
    slot_mapping = block_nums * block_size + clamped % block_size
    slot_mapping = slot_mapping.masked_fill(new_positions >= max_model_len, PADDING_SLOT_ID)
    slot_mapping = slot_mapping.masked_fill(is_rejected_token_mask, PADDING_SLOT_ID)
    return slot_mapping


def extend_all_queries_by_N(query_start_loc: list[int], seq_lens: list[int],
                            num_new_tokens: int) -> tuple[list[int], list[int]]:
    """上游 `extend_all_queries_by_N()` 的本地版：每请求多 N 行、上下文长 N。

    上游返回一个新的 `CommonAttentionMetadata`（`query_start_loc + N*arange`、`seq_lens + N`、
    `num_actual_tokens += B*N`、`max_query_len/max_seq_len += N`）。本地 attention metadata 只有
    `query_start_loc` / `seq_lens` 两个字段需要改（`num_tokens` 由调用方按 `Σ(n_i+N)` 传）。
    """
    return ([loc + num_new_tokens * index for index, loc in enumerate(query_start_loc)],
            [length + num_new_tokens for length in seq_lens])


def update_scheduler_for_invalid_drafts(spec_token_ids: list[int],
                                        num_valid_draft_tokens: int | None) -> list[int]:
    """把"占位草稿"裁到有效个数（上游 `update_scheduler_for_invalid_drafts` 的核心效应）。

    上游在 Runner 侧对 `scheduler_output` 做这件事（`num_scheduled_tokens[req] -= 占位-有效`、
    `scheduled_spec_decode_tokens[req] = spec[:valid]`、有效数 0 就 pop），因为异步调度让
    Scheduler 手里的计划是**乐观**的（按固定宽度 K 占位），必须等 GPU 的有效个数异步回来再裁。
    本机没有异步调度，所以在**草稿交接给 Scheduler 时**做同样的事：裁完之后的计划、预算、统计
    都只包含真实候选（见 docs/step60_alignment.md §3）。

    除了按有效数截断，还**再滤一遍 -1**：需求 059 §3.5 的不变量是"哨兵永远不能变成真实 token"，
    就算有效个数算错了也不许漏出去。
    """
    if num_valid_draft_tokens is None:
        return list(spec_token_ids)
    valid = max(0, min(int(num_valid_draft_tokens), len(spec_token_ids)))
    return [token for token in spec_token_ids[:valid] if token >= 0]


# ---------------- 69 关：投机输入的 padding（对应上游同名两个 kernel 的 eager 版）----------------


def prepare_inputs_padded(cu_num_draft_tokens: torch.Tensor,
                          valid_sampled_tokens_count: torch.Tensor,
                          query_start_loc: torch.Tensor, num_reqs: int,
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """上游 `eagle_prepare_inputs_padded_kernel`（`v1/spec_decode/utils.py:136-175`）的等价实现。

    它回答两个问题，**都在 device 上算、不做 CPU 同步**（上游的整个 padding 设计的要点）：

        token_indices_to_sample[i]   第 i 条请求要从**它的哪一行**取 hidden 去采样第一枚草稿
        num_rejected_tokens_gpu[i]   这条请求本轮采用 K_i 枚草稿、实际活了 v_i 枚，
                                     被拒的是 `K_i + 1 - v_i` 行（v 含纠正/奖励那一枚）

    为什么要用"补空位"的坐标来算：padding 之后每条请求的第一遍输入**包含会被拒的那些行**
    （它们物理存在、槽位是哨兵、值是 padding），所以"该从哪一行采样"不能再拿"最后一行"当答案，
    必须由"有效个数"反推——这正是本函数存在的理由。

    约定与上游一致：`cu_num_draft_tokens` 是**包含式**前缀和（第 0 项就是第 0 条请求的草稿数，
    不是 0），`query_start_loc` 长度 `num_reqs + 1`。返回两个 int32 张量，长度都为 `num_reqs`。
    """
    if cu_num_draft_tokens.shape[0] != num_reqs or valid_sampled_tokens_count.shape[0] != num_reqs:
        raise ValueError(
            f"逐请求张量长度必须是 num_reqs={num_reqs}：收到 "
            f"cu_num_draft_tokens={tuple(cu_num_draft_tokens.shape)} / "
            f"valid={tuple(valid_sampled_tokens_count.shape)}")
    if query_start_loc.shape[0] != num_reqs + 1:
        raise ValueError(
            f"query_start_loc 长度必须是 num_reqs+1={num_reqs + 1}，"
            f"收到 {tuple(query_start_loc.shape)}")
    device = cu_num_draft_tokens.device
    # num_draft[i] = cu[i] - cu[i-1]（第 0 项前面补 0，与内核里的分支等价）
    previous_cu = torch.cat([torch.zeros(1, dtype=cu_num_draft_tokens.dtype, device=device),
                             cu_num_draft_tokens[:-1]])
    num_draft_tokens = cu_num_draft_tokens - previous_cu
    # 没有草稿的请求（K_i=0）不算"被拒"：它这一轮的 1 行是纠正/bonus 行
    num_rejected = torch.where(
        num_draft_tokens > 0,
        num_draft_tokens + 1 - valid_sampled_tokens_count.to(num_draft_tokens.dtype),
        torch.zeros_like(num_draft_tokens))
    # query_start_loc[i+1] - 1 = 这条请求 query 块的最后一行（全局行号）
    index_to_sample = query_start_loc[1:num_reqs + 1] - 1 - num_rejected
    return index_to_sample.to(torch.int32), num_rejected.to(torch.int32)


def eagle_step_update_slot_mapping_and_metadata(
        positions_1d: torch.Tensor, block_table_tensor: torch.Tensor,
        seq_lens: torch.Tensor, block_size: int, max_model_len: int,
        out_clamped_positions: torch.Tensor, out_slot_mapping: torch.Tensor,
        input_batch_size: int | None = None) -> None:
    """上游 `eagle_step_update_slot_mapping_and_metadata()`（同文件 L88-133）的等价实现。

    EAGLE 自回归提议的每一步都要做同样的三件事，上游把它们**融进一个 kernel**（省两次
    launch）：位置 +1、按位置查块表得到 KV 槽位、上下文长度 +1。本仓库没有 Triton 版本的
    必要（draft 的提议循环本来就是 CPU 驱动的），但**算法逐条对齐**：

        new_position = position + 1
        超过 max_model_len          → 位置钳到 0、槽位 = PADDING_SLOT_ID(-1)、seq_len 回到 1
        block_number = min(new_pos // block_size, n_blocks_per_req - 1)     ← 越界时钳住再查表
        slot = block_table[req, block_number] * block_size + new_pos % block_size

    两个张量参数是**原地**语义（与上游一致）：`seq_lens` 原地 +1；`out_*` 是输出缓冲。
    `input_batch_size > batch_size` 时，多出来的行（CUDA Graph 的 padding 行）只写哨兵槽位
    ——它们不是任何请求的 KV，写进去会把别人的缓存覆盖掉。
    """
    batch_size = positions_1d.shape[0]
    if input_batch_size is None:
        input_batch_size = batch_size
    if out_slot_mapping.shape[0] < input_batch_size:
        raise ValueError(
            f"槽位缓冲只有 {out_slot_mapping.shape[0]} 行，放不下 input_batch_size="
            f"{input_batch_size}：padding 行会写到缓冲外面")
    n_blocks_per_req = block_table_tensor.shape[1]

    new_position = positions_1d[:batch_size].to(torch.int64) + 1
    exceeds_max = new_position >= max_model_len
    clamped_position = torch.where(exceeds_max,
                                   torch.zeros_like(new_position), new_position)
    block_number = torch.minimum(clamped_position // block_size,
                                 torch.tensor(n_blocks_per_req - 1,
                                              device=clamped_position.device))
    req_index = torch.arange(batch_size, device=clamped_position.device)
    block_id = block_table_tensor[req_index, block_number].to(torch.int64)
    slot_id = block_id * block_size + clamped_position % block_size
    slot_id = torch.where(exceeds_max,
                          torch.full_like(slot_id, PADDING_SLOT_ID), slot_id)

    out_clamped_positions[:batch_size].copy_(clamped_position.to(
        out_clamped_positions.dtype))
    out_slot_mapping[:batch_size].copy_(slot_id.to(out_slot_mapping.dtype))
    if input_batch_size > batch_size:
        # padding 行：只有槽位需要打哨兵（位置/长度没人读）
        out_slot_mapping[batch_size:input_batch_size].fill_(PADDING_SLOT_ID)
    # 上下文长度 +1；越界的那种行回到 1（上游语义：这条请求从"新一轮"开始，不再累积）
    new_seq_len = torch.where(exceeds_max, torch.ones_like(new_position),
                              seq_lens[:batch_size].to(torch.int64) + 1)
    seq_lens[:batch_size].copy_(
        torch.minimum(new_seq_len,
                      torch.tensor(max_model_len, device=new_seq_len.device)
                      ).to(seq_lens.dtype))
