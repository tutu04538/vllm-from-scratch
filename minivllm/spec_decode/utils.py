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
    # 每条请求的 AR 起点（= 有效历史末尾）：第 k 枚草稿的输入位置 = history_end + k - 2
    history_end: dict[str, int] = field(default_factory=dict)


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


def compute_new_slot_mapping(block_table: torch.Tensor,
                             query_lens: list[int],
                             new_positions: torch.Tensor,
                             is_rejected_token_mask: torch.Tensor,
                             block_size: int,
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
