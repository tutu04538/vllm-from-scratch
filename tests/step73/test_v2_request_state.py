"""73 关：V2 的**常驻 slot 请求状态**与输入组装（对应需求 073 §3/§4 的前两组验收）。

这一组测试要钉住的是"V1 的行号即身份"与"V2 的身份与行号分离"之间的差别：

    I1  请求进引擎时分到一个**常驻 slot**，被调度期间不变（`req_id_to_index`）
    I2  本轮 batch 行 → slot 的映射是 `idx_mapping`；同一条请求换行不影响它的任何状态
    I3  `cu_num_logits` 含**前导 0**（`cu[0] == 0`），第 i 条请求占 `[cu[i], cu[i+1])` 行
    I4  `expanded_idx_mapping` 是 logits 行 → slot；`expanded_local_pos` 是"第几行"
    I5  `last_sampled_tokens` / `draft_tokens` / `all_token_ids` / `total_len` 都按 **slot** 存
    I6  `prompt_len`（用户给的）与 `prefill_len`（喂进 runner 的）分开，恢复时后者更大
    I7  物理块表按 slot 登录、按 `idx_mapping` gather 成 batch 顺序
"""

import numpy as np
import pytest
import torch

from spec73_helpers import DEVICE, requires_cuda  # noqa: E402  (tests/step73 在 sys.path 上)

from minivllm.worker.gpu.block_table import PAD_SLOT_ID, BlockTables  # noqa: E402
from minivllm.worker.gpu.buffer_utils import async_copy_to_gpu  # noqa: E402
from minivllm.worker.gpu.input_batch import (  # noqa: E402
    InputBuffers,
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    post_update,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)
from minivllm.worker.gpu.states import RequestState  # noqa: E402

MAX_REQS = 8
MAX_TOKENS = 64
BLOCK_SIZE = 4
K = 2


def make_states(device="cuda"):
    return RequestState(
        max_num_reqs=MAX_REQS,
        max_model_len=MAX_TOKENS,
        max_num_batched_tokens=MAX_TOKENS,
        num_speculative_steps=K,
        vocab_size=32,
        device=torch.device(device),
    )


def reserve_slot(states: RequestState, slot: int) -> None:
    """用占位请求吃掉 `free_indices` 栈顶，使**下一个** `add_request` 拿到 `slot`。

    `free_indices` 是栈（LIFO）：`add_request` 从尾部 pop。所以"指定 slot"＝先把栈顶那些
    不想要的 slot 占掉，加完真请求之后再释放占位请求（它们不会再被这次分配用到）。
    """
    holders = []
    while states.free_indices[-1] != slot:
        holder_id = f"__hold_{states.free_indices[-1]}"
        states.add_request(holder_id, prompt_len=1, all_token_ids=[1],
                           num_computed_tokens=0, max_tokens=4)
        holders.append(holder_id)
    return holders


def release(states: RequestState, holders: list[str]) -> None:
    for holder_id in holders:
        states.remove_request(holder_id)


# ---------------------------------------------------------------------------
# 纯 CPU：slot 生命周期（不需要 CUDA）
# ---------------------------------------------------------------------------


def test_slots_are_persistent_and_reused_lifo():
    """I1：slot 来自 `free_indices`，请求走了才还回去，新请求**复用**它（LIFO）。"""
    states = make_states(device="cpu" if not torch.cuda.is_available() else "cuda")
    states.add_request("A", prompt_len=3, all_token_ids=[1, 2, 3], num_computed_tokens=0,
                       max_tokens=8)
    states.add_request("B", prompt_len=2, all_token_ids=[4, 5], num_computed_tokens=0,
                       max_tokens=8)
    slot_a, slot_b = states.req_id_to_index["A"], states.req_id_to_index["B"]
    assert (slot_a, slot_b) == (MAX_REQS - 1, MAX_REQS - 2)
    assert states.index_to_req_id[slot_a] == "A"

    freed = states.remove_request("A")
    assert freed == slot_a and slot_a in states.free_indices
    assert states.num_reqs == 1
    # 复用：下一个请求拿到刚还回来的那个 slot（只要它还在栈顶）。
    states.add_request("C", prompt_len=1, all_token_ids=[6], num_computed_tokens=0,
                       max_tokens=8)
    assert states.req_id_to_index["C"] == freed
    assert states.remove_request("nope") is None


def test_prompt_len_and_prefill_len_are_separate():
    """I6：抢占恢复时 `prefill_len > prompt_len`，两者互不覆盖。"""
    states = make_states(device="cpu" if not torch.cuda.is_available() else "cuda")
    # prompt 4 个 token，但恢复时要重算 [prompt + 已生成的 2 个] = 6 个。
    states.add_request("R", prompt_len=4, all_token_ids=[1, 2, 3, 4, 5, 6],
                       num_computed_tokens=4, max_tokens=8)
    slot = states.req_id_to_index["R"]
    assert int(states.prompt_len.np[slot]) == 4
    assert int(states.prefill_len.np[slot]) == 6
    assert int(states.max_seq_len[slot]) == 4 + 8
    assert int(states.num_computed_tokens_np[slot]) == 4
    assert int(states.num_computed_prefill_tokens[slot]) == 4
    # draft 缓冲按 slot 清零（新请求不能继承上一个用这个 slot 的请求的草稿）。
    states.draft_tokens[slot] = torch.tensor([7, 9], dtype=torch.int64,
                                             device=states.draft_tokens.device)
    states.add_request("S", prompt_len=1, all_token_ids=[2], num_computed_tokens=0,
                       max_tokens=4)
    states.remove_request("R")
    states.add_request("T", prompt_len=1, all_token_ids=[3], num_computed_tokens=0,
                       max_tokens=4)
    assert states.req_id_to_index["T"] == slot
    assert states.draft_tokens[slot].tolist() == [0, 0]


# ---------------------------------------------------------------------------
# 需求 §4 的验收例子：A=slot 5、B=slot 2、batch [B, A]
# ---------------------------------------------------------------------------


@requires_cuda
def test_slot_mapping_tokens_logits_rng_and_block_table():
    """需求 §4 第一条：A=5、B=2、batch [B,A] 时 tokens/logits/q/RNG/块表全部映射正确。"""
    states = make_states()
    hold_a = reserve_slot(states, 5)
    states.add_request("A", prompt_len=4, all_token_ids=[10, 11, 12, 13],
                       num_computed_tokens=0, max_tokens=16)
    hold_b = reserve_slot(states, 2)
    states.add_request("B", prompt_len=2, all_token_ids=[20, 21],
                       num_computed_tokens=0, max_tokens=16)
    release(states, hold_a + hold_b)
    assert states.req_id_to_index == {"A": 5, "B": 2}
    # staged write 要显式落盘：`total_len` / `all_token_ids` / `num_computed_tokens` 三张表
    # 在 GPU 上，`add_request` 只是"记账"（runner 在 `add_requests()` 末尾统一 apply）。
    states.apply_staged_writes()
    assert states.all_token_ids.gpu[5, :4].tolist() == [10, 11, 12, 13]
    assert states.all_token_ids.gpu[2, :2].tolist() == [20, 21]
    assert states.total_len.gpu.tolist()[5] == 4 and states.total_len.gpu.tolist()[2] == 2

    # batch 行 [B, A]：两行都是 decode（各 1 行 query，K 个草稿）
    req_ids = ["B", "A"]
    idx_mapping_np = np.array([states.req_id_to_index[r] for r in req_ids], dtype=np.intp)
    assert idx_mapping_np.tolist() == [2, 5]
    idx_mapping = async_copy_to_gpu(idx_mapping_np, device=DEVICE)

    # 块表：B 的 slot 2 用物理块 [3, 7]，A 的 slot 5 用 [1]
    block_tables = BlockTables(
        block_sizes=[BLOCK_SIZE], max_num_reqs=MAX_REQS, max_num_batched_tokens=MAX_TOKENS,
        max_num_blocks_per_group=[MAX_TOKENS // BLOCK_SIZE], device=torch.device(DEVICE),
        kernel_block_sizes=[BLOCK_SIZE])
    block_tables.append_block_ids(2, ([3, 7],), overwrite=True)
    # A 的上下文是 4 个 token + K 个草稿槽位 → 位置 4..6 落在**第 2 个逻辑块**上，
    # 所以块表必须有两项。只给一项时内核会读到期零填充的第 2 项（= 物理块 0），
    # 把草稿的 KV 悄悄写进 0 号块——"块表要覆盖本轮要写的位置"这条由调度器保证（58 §5）。
    block_tables.append_block_ids(5, ([1, 5],), overwrite=True)
    block_tables.apply_staged_writes()
    gathered = block_tables.gather_block_tables(idx_mapping, num_reqs_padded=2)
    assert gathered[0][0, :2].tolist() == [3, 7]      # 第 0 行 = B
    assert gathered[0][1, :2].tolist() == [1, 5]      # 第 1 行 = A

    # 输入组装：每请求 1 + K 行 query（decode），两次采样之后 A 在第 5 个位置、B 在第 3 个
    num_computed = torch.zeros(MAX_REQS, dtype=torch.int32, device=DEVICE)
    num_computed[5] = 4
    num_computed[2] = 2
    query_start_loc_np = np.array([0, 1 + K, 2 * (1 + K)] + [2 * (1 + K)] * (MAX_REQS - 1),
                                  dtype=np.int32)
    query_start_loc = async_copy_to_gpu(query_start_loc_np, device=DEVICE)
    input_buffers = InputBuffers(MAX_REQS, MAX_TOKENS, torch.device(DEVICE))
    positions = input_buffers.positions
    seq_lens = input_buffers.seq_lens
    prepare_pos_seq_lens(idx_mapping, query_start_loc, num_computed, positions, seq_lens)
    # B: 位置 2..4；A: 位置 4..6（含 K 个未验证槽位）
    assert seq_lens[:2].tolist() == [2 + 1 + K, 4 + 1 + K]
    assert positions[: 2 * (1 + K)].tolist() == [2, 3, 4, 4, 5, 6]

    # 上一轮采样结果与草稿按 **slot** 取：给 B、A 各放一份不同的草稿
    last_sampled = torch.zeros(MAX_REQS, 1, dtype=torch.int64, device=DEVICE)
    last_sampled[2] = 20
    last_sampled[5] = 10
    draft_tokens = torch.zeros(MAX_REQS, K, dtype=torch.int64, device=DEVICE)
    draft_tokens[2] = torch.tensor([21, 22], dtype=torch.int64, device=DEVICE)
    draft_tokens[5] = torch.tensor([11, 12], dtype=torch.int64, device=DEVICE)
    prefill_len = torch.zeros(MAX_REQS, dtype=torch.int32, device=DEVICE)
    prefill_len[5] = 4
    prefill_len[2] = 2
    cu_num_logits_np = np.array([0, 1 + K, 2 * (1 + K)], dtype=np.int32)
    cu_num_logits = async_copy_to_gpu(cu_num_logits_np, device=DEVICE)
    logits_indices = combine_sampled_and_draft_tokens(
        input_buffers.input_ids, idx_mapping, last_sampled, query_start_loc, seq_lens,
        prefill_len, draft_tokens, cu_num_logits, 2 * (1 + K))

    input_ids = input_buffers.input_ids[: 2 * (1 + K)]
    # 第 0 行（B）：[上次采样 20, 草稿 21, 22]；第 1 行（A）：[10, 11, 12]
    assert input_ids.tolist() == [20, 21, 22, 10, 11, 12]
    # 每请求要算 logits 的行 = 该请求 query 的最后 num_logits 行：B 是 0..2、A 是 3..5。
    # 投机下 K+1 行**每行都要 logits**（K 个候选位用于验证 + 1 个 bonus 位），
    # 所以这个张量是"扁平的全部 logits 行"，不是"每请求一行"。
    assert logits_indices.tolist() == [0, 1, 2, 3, 4, 5]

    # I3/I4：cu_num_logits 含前导 0，expanded 映射把 logits 行摊回 slot
    expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
        idx_mapping, 2 * (1 + K), cu_num_logits, 1 + K)
    assert cu_num_logits_np.tolist() == [0, 3, 6]
    assert expanded_idx_mapping.tolist() == [2, 2, 2, 5, 5, 5]   # 前 3 行是 B、后 3 行是 A
    assert expanded_local_pos.tolist() == [0, 1, 2, 0, 1, 2]

    # slot_mapping 按块表现算：B 的位置 2..4 → 块 3 的 slot 14,15 与块 7 的 slot 28；
    # A 的位置 4..6 → 块 5 的 slot 20,21,22
    slot_mappings = block_tables.compute_slot_mappings(
        idx_mapping, query_start_loc, positions, 2 * (1 + K))
    assert slot_mappings[0].tolist() == [14, 15, 28, 20, 21, 22]
    # 批尾（第 2*(1+K) 行之后）必须是哨兵，不能被当成合法槽位。切片只到 num_tokens，
    # 所以要看底层缓冲：内核最后一个 program 专门从**实际** token 数开始填 PAD。
    assert int(block_tables.slot_mappings[0, 2 * (1 + K)]) == PAD_SLOT_ID


@requires_cuda
def test_post_update_writes_by_slot_not_by_row():
    """I5：采样结果按 **slot** 落盘——batch 行换了，请求的状态不受影响。"""
    states = make_states()
    hold_a = reserve_slot(states, 5)
    states.add_request("A", prompt_len=2, all_token_ids=[10, 11],
                       num_computed_tokens=0, max_tokens=16)
    hold_b = reserve_slot(states, 2)
    states.add_request("B", prompt_len=2, all_token_ids=[20, 21],
                       num_computed_tokens=0, max_tokens=16)
    release(states, hold_a + hold_b)
    states.apply_staged_writes()

    last_sampled = torch.zeros(MAX_REQS, 1, dtype=torch.int64, device=DEVICE)
    num_computed = torch.zeros(MAX_REQS, dtype=torch.int32, device=DEVICE)
    total_len = states.total_len
    # 第一次：batch [B, A]，B 采了 2 个（1 接受 + bonus）、A 采了 3 个（2 接受 + bonus）
    idx_mapping = async_copy_to_gpu(np.array([2, 5], dtype=np.intp), device=DEVICE)
    query_start_loc = async_copy_to_gpu(np.array([0, 3, 6], dtype=np.int32), device=DEVICE)
    sampled = torch.tensor([[21, 22, -1], [12, 13, 14]], dtype=torch.int64, device=DEVICE)
    num_sampled = torch.tensor([2, 3], dtype=torch.int32, device=DEVICE)
    num_rejected = torch.tensor([1, 0], dtype=torch.int32, device=DEVICE)
    post_update(idx_mapping, num_computed, last_sampled, None, sampled, num_sampled,
                num_rejected, query_start_loc, states.all_token_ids.gpu, total_len.gpu)
    torch.cuda.synchronize()
    assert last_sampled[2].item() == 22 and last_sampled[5].item() == 14
    assert states.all_token_ids.gpu[2, :4].tolist() == [20, 21, 21, 22]
    assert states.all_token_ids.gpu[5, :5].tolist() == [10, 11, 12, 13, 14]
    assert total_len.gpu.tolist()[2] == 4 and total_len.gpu.tolist()[5] == 5
    # 已算 token 数 = 本轮 query 行数 − 被拒行数
    assert num_computed.tolist()[2] == 3 - 1 and num_computed.tolist()[5] == 3 - 0

    # 第二次：batch 变成 [A, B]（行换了）——状态按 slot 走，与行序无关
    idx_mapping = async_copy_to_gpu(np.array([5, 2], dtype=np.intp), device=DEVICE)
    sampled = torch.tensor([[15, -1, -1], [23, -1, -1]], dtype=torch.int64, device=DEVICE)
    num_sampled = torch.tensor([1, 1], dtype=torch.int32, device=DEVICE)
    num_rejected = torch.tensor([0, 0], dtype=torch.int32, device=DEVICE)
    post_update(idx_mapping, num_computed, last_sampled, None, sampled, num_sampled,
                num_rejected, None, states.all_token_ids.gpu, total_len.gpu)
    torch.cuda.synchronize()
    assert last_sampled[5].item() == 15 and last_sampled[2].item() == 23
    assert states.all_token_ids.gpu[5, :6].tolist() == [10, 11, 12, 13, 14, 15]
    assert states.all_token_ids.gpu[2, :5].tolist() == [20, 21, 21, 22, 23]


@requires_cuda
def test_prefill_reads_history_from_slot_and_lookahead():
    """I6 的另一半：prefill 行从 slot 的历史里取（`num_computed` 起），并留 lookahead。"""
    states = make_states()
    states.add_request("R", prompt_len=4, all_token_ids=[1, 2, 3, 4, 5, 6],
                       num_computed_tokens=4, max_tokens=16)
    slot = states.req_id_to_index["R"]
    states.apply_staged_writes()
    idx_mapping = async_copy_to_gpu(np.array([slot], dtype=np.intp), device=DEVICE)
    input_buffers = InputBuffers(MAX_REQS, MAX_TOKENS, torch.device(DEVICE))
    # 这一轮算 [4, 6)：喂 token 5、6（恢复重算），lookahead 取 1 个（已到 prefill 尾 → 0）
    query_start_loc = async_copy_to_gpu(np.array([0, 2], dtype=np.int32), device=DEVICE)
    prepare_prefill_inputs(input_buffers.input_ids, states.next_prefill_tokens, idx_mapping,
                           query_start_loc, states.all_token_ids.gpu,
                           states.prefill_len.gpu, states.num_computed_tokens.gpu)
    torch.cuda.synchronize()
    assert input_buffers.input_ids[:2].tolist() == [5, 6]
    assert states.next_prefill_tokens[:, slot].item() == 0
