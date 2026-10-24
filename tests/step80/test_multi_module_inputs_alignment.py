"""需求 080A：多模块 MTP 输入布局与跨轮缓存。

本文件**只测输入层**，不实例化 `MultiModuleMTPSpeculator`、不跑模型、不测 Graph。
80B 才接生产调用链；因此这里用 `SimpleNamespace` 提供生产函数实际读取的
`InputBatch` 字段，直接对 `InputBuffers` 做逐值差分。

冻结参考：本机 `vllm 0.28.0` 的
`v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py`。每个测试同时调用
本地函数与上游同名函数，用 `torch.equal` 比较**整个缓冲**，不是只比 shape 或
最终 token。

固定例子（需求 §4）：`max_num_seqs=4, B=1, K=3, idx_mapping=[2]`，
`query_start_loc=[0,1]`，输入行 0 token=42、hidden=7，cache 初值 -9。
正确结果是 `cached_ids[2,1]=42`，不是 `[0,0]`。
"""

import os
from types import SimpleNamespace

os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

import pytest
import torch

from minivllm.worker.gpu.block_table import PAD_SLOT_ID
from minivllm.worker.gpu.input_batch import InputBuffers
from minivllm.worker.gpu.spec_decode.multi_module_mtp import speculator as mine

up = pytest.importorskip(
    "vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator"
)

DEVICE = "cuda"
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="80A 的输入 kernels 是 Triton；CPU 只保留协议/形状检查",
)


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------


def _new_buffers(max_reqs: int, max_tokens: int, fill: int = -7) -> InputBuffers:
    """常驻缓冲；所有位置先填一个可辨认的哨兵。"""
    buffers = InputBuffers(max_reqs, max_tokens, torch.device(DEVICE))
    buffers.input_ids.fill_(fill)
    buffers.positions.fill_(fill)
    buffers.seq_lens.fill_(fill)
    buffers.query_start_loc.fill_(fill)
    return buffers


def _qsl(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device=DEVICE)


def _set_qsl(buffers: InputBuffers, values: list[int]) -> None:
    buffers.query_start_loc[: len(values)].copy_(_qsl(values))


def _batch(
    qsl: list[int],
    target_ids: torch.Tensor,
    target_positions: torch.Tensor,
    seq_lens: list[int],
    idx_mapping: list[int],
) -> SimpleNamespace:
    """生产函数实际读取的 InputBatch 字段；不是完整 InputBatch。"""
    total = int(qsl[-1])
    return SimpleNamespace(
        input_ids=target_ids[:total],
        positions=target_positions[:total],
        idx_mapping=torch.tensor(idx_mapping, dtype=torch.int32, device=DEVICE),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE),
        query_start_loc=_qsl(qsl),
    )


def _target_rows(total: int, id_start: int = 100, pos_start: int = 10):
    ids = torch.arange(
        id_start, id_start + total, dtype=torch.int32, device=DEVICE
    )
    positions = torch.arange(
        pos_start, pos_start + total, dtype=torch.int64, device=DEVICE
    )
    return ids, positions


def _last(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64, device=DEVICE)


def _assert_equal(name: str, got: torch.Tensor, want: torch.Tensor) -> None:
    assert torch.equal(got, want), (
        f"{name} 不一致\nlocal={got.tolist()}\nfrozen={want.tolist()}"
    )


def _run_prepare_pair(
    *,
    num_reqs: int,
    max_reqs: int,
    max_tokens: int,
    input_batch,
    cached_ids: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill: torch.Tensor,
    overrides: torch.Tensor,
    qsl: list[int],
    num_speculative_steps: int,
):
    """同一份输入分别跑本地/上游；两个 draft 缓冲初值相同。"""
    local_buf = _new_buffers(max_reqs, max_tokens)
    up_buf = _new_buffers(max_reqs, max_tokens)
    for buf in (local_buf, up_buf):
        _set_qsl(buf, qsl)
    local_last = torch.full((max_reqs,), -5, dtype=torch.int64, device=DEVICE)
    up_last = local_last.clone()
    local_ovr = overrides.clone()
    up_ovr = overrides.clone()

    mine.prepare_input_buffers(
        num_reqs,
        input_batch,
        cached_ids,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill,
        local_ovr,
        local_buf,
        local_last,
        max_reqs,
        num_speculative_steps,
    )
    up.prepare_input_buffers(
        num_reqs,
        input_batch,
        cached_ids,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill,
        up_ovr,
        up_buf,
        up_last,
        max_reqs,
        num_speculative_steps,
    )

    _assert_equal("input_ids", local_buf.input_ids, up_buf.input_ids)
    _assert_equal("positions", local_buf.positions, up_buf.positions)
    _assert_equal("seq_lens", local_buf.seq_lens, up_buf.seq_lens)
    _assert_equal("query_start_loc", local_buf.query_start_loc, up_buf.query_start_loc)
    _assert_equal("last_token_indices", local_last, up_last)
    _assert_equal("overrides", local_ovr, up_ovr)
    return local_buf, local_last, local_ovr


# ---------------------------------------------------------------------------
# 1. 缓存布局：固定例子 + 持久 slot + slot 复用
# ---------------------------------------------------------------------------


@requires_cuda
def test_cache_inputs_fixed_example_entire_buffer():
    """需求 §4 的固定例子：写 [2,1]，不是 [0,0]；整块 cache 逐值检查。"""
    max_reqs, max_tokens, hidden_size, K = 4, 8, 3, 3
    buffers = _new_buffers(max_reqs, max_tokens)
    buffers.input_ids[0] = 42
    _set_qsl(buffers, [0, 1])

    hidden = torch.full((max_tokens, hidden_size), -1.0, device=DEVICE)
    hidden[0] = 7.0
    embeds = torch.full((max_tokens, hidden_size), -2.0, device=DEVICE)
    embeds[0] = 5.0

    cached_ids = torch.full(
        (max_reqs, K - 1), -9, dtype=torch.int64, device=DEVICE
    )
    cached_hidden = torch.full(
        (max_reqs, K - 1, hidden_size), -9.0, device=DEVICE
    )
    cached_embeds = torch.full(
        (max_reqs, K - 1, hidden_size), -9.0, device=DEVICE
    )
    last = _last([0])
    idx = torch.tensor([2], dtype=torch.int32, device=DEVICE)

    mine.cache_inputs(
        buffers,
        embeds,
        hidden,
        cached_ids,
        cached_embeds,
        cached_hidden,
        last,
        idx,
        1,
        K,
        True,
    )

    want_ids = torch.full_like(cached_ids, -9)
    want_ids[2, 1] = 42
    want_hidden = torch.full_like(cached_hidden, -9.0)
    want_hidden[2, 1] = 7.0
    want_embeds = torch.full_like(cached_embeds, -9.0)
    want_embeds[2, 1] = 5.0
    assert torch.equal(cached_ids, want_ids), cached_ids.tolist()
    assert torch.equal(cached_hidden, want_hidden)
    assert torch.equal(cached_embeds, want_embeds)

    # 同一个输入喂上游：整块逐值相等。
    up_ids = torch.full_like(cached_ids, -9)
    up_hidden = torch.full_like(cached_hidden, -9.0)
    up_embeds = torch.full_like(cached_embeds, -9.0)
    up.cache_inputs(
        buffers,
        embeds,
        hidden,
        up_ids,
        up_embeds,
        up_hidden,
        last,
        idx,
        1,
        K,
        True,
    )
    _assert_equal("cache ids fixed", cached_ids, up_ids)
    _assert_equal("cache hidden fixed", cached_hidden, up_hidden)
    _assert_equal("cache embeds fixed", cached_embeds, up_embeds)


@requires_cuda
def test_cache_inputs_uses_persistent_slot_across_batch_reorder():
    """`idx_mapping=[3,1] → [1,3]`：数据跟 slot 走，不跟 batch 行号走。"""
    max_reqs, hidden_size, K = 4, 2, 3
    buffers = _new_buffers(max_reqs, 8)
    buffers.input_ids[:4] = torch.tensor(
        [10, 11, 12, 13], dtype=torch.int32, device=DEVICE
    )
    _set_qsl(buffers, [0, 2, 4])
    hidden = torch.arange(8 * hidden_size, dtype=torch.float32, device=DEVICE)
    hidden = hidden.reshape(8, hidden_size)
    embeds = hidden + 100
    last = _last([1, 3])
    idx = torch.tensor([3, 1], dtype=torch.int32, device=DEVICE)

    cached_ids = torch.full((max_reqs, K - 1), -9, dtype=torch.int64, device=DEVICE)
    cached_hidden = torch.full((max_reqs, K - 1, hidden_size), -9.0, device=DEVICE)
    cached_embeds = torch.full((max_reqs, K - 1, hidden_size), -9.0, device=DEVICE)
    mine.cache_inputs(
        buffers, embeds, hidden, cached_ids, cached_embeds, cached_hidden,
        last, idx, 2, K, True,
    )
    want_ids = torch.full_like(cached_ids, -9)
    want_ids[3] = torch.tensor([10, 11], dtype=torch.int64, device=DEVICE)
    want_ids[1] = torch.tensor([12, 13], dtype=torch.int64, device=DEVICE)
    assert torch.equal(cached_ids, want_ids), cached_ids.tolist()
    assert torch.equal(cached_hidden[3], hidden[0:2])
    assert torch.equal(cached_hidden[1], hidden[2:4])
    assert torch.equal(cached_embeds[3], embeds[0:2])
    assert torch.equal(cached_embeds[1], embeds[2:4])
    # 0/2 行没有被 batch 行号误写。
    assert torch.equal(cached_hidden[0], torch.full((K - 1, hidden_size), -9.0, device=DEVICE))
    assert torch.equal(cached_hidden[2], torch.full((K - 1, hidden_size), -9.0, device=DEVICE))

    # 批顺序反转，再写一遍：同一个 slot 被新数据覆盖，旧 batch 行不再拥有它。
    buffers.input_ids[:4] = torch.tensor(
        [20, 21, 22, 23], dtype=torch.int32, device=DEVICE
    )
    idx_reordered = torch.tensor([1, 3], dtype=torch.int32, device=DEVICE)
    mine.cache_inputs(
        buffers, embeds, hidden, cached_ids, cached_embeds, cached_hidden,
        last, idx_reordered, 2, K, True,
    )
    want_ids[1] = torch.tensor([20, 21], dtype=torch.int64, device=DEVICE)
    want_ids[3] = torch.tensor([22, 23], dtype=torch.int64, device=DEVICE)
    assert torch.equal(cached_ids, want_ids), cached_ids.tolist()


@requires_cuda
def test_reused_slot_reads_only_the_latest_cache_column():
    """slot 复用时，短窗口只让 re-prefill 读有效列，不能把左侧旧列当成数据。"""
    K, max_reqs, max_tokens, hidden_size = 3, 4, 8, 3
    # 上一轮这个 slot 只有一个有效 token（42），它落在列 1；列 0 是更早的残值。
    cached_ids = torch.full((max_reqs, K - 1), -9, dtype=torch.int64, device=DEVICE)
    cached_ids[2, 0] = -8
    cached_ids[2, 1] = 42
    cached_hidden = torch.full(
        (max_reqs, K - 1, hidden_size), -9.0, device=DEVICE
    )
    cached_hidden[2, 0] = -8.0
    cached_hidden[2, 1] = 7.0

    target_ids, target_positions = _target_rows(4, 600, 20)
    qsl = [0, 4]
    batch = _batch(qsl, target_ids, target_positions, [24], [2])
    num_sampled = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    # r=2 -> num_reprefill=1 -> 只读 cache 列 K-1-1=1（42/7），不读列 0。
    num_rejected = torch.tensor([2], dtype=torch.int32, device=DEVICE)
    last_sampled = torch.tensor([0, 0, 99, 0], dtype=torch.int64, device=DEVICE)
    next_prefill = torch.zeros(K, max_reqs, dtype=torch.int32, device=DEVICE)
    overrides = torch.full((max_reqs, K - 1), -1, dtype=torch.int64, device=DEVICE)
    buffers = _new_buffers(max_reqs, max_tokens)
    _set_qsl(buffers, qsl)
    last = torch.full((max_reqs,), -5, dtype=torch.int64, device=DEVICE)
    mine.prepare_input_buffers(
        1, batch, cached_ids, num_sampled, num_rejected, last_sampled,
        next_prefill, overrides, buffers, last, max_reqs, K,
    )
    assert int(last[0]) == 2
    assert int(buffers.input_ids[0]) == 42  # 不是 -8
    assert int(buffers.input_ids[1]) == int(target_ids[1])
    assert int(buffers.input_ids[2]) == 99
    assert int(buffers.positions[0]) == 19  # first_position(20)-1

    hidden = torch.full((max_tokens, hidden_size), -3.0, device=DEVICE)
    target_hidden = torch.arange(
        4 * hidden_size, dtype=torch.float32, device=DEVICE
    ).reshape(4, hidden_size)
    mine.prepare_input_hidden_states_and_embeddings(
        1, 4, hidden, target_hidden, cached_hidden, None, None,
        batch, buffers, num_rejected, K,
    )
    assert torch.equal(hidden[0], cached_hidden[2, 1])  # 7，不是 -8
    assert torch.equal(hidden[1], target_hidden[0])
    assert torch.equal(hidden[2], target_hidden[1])


@requires_cuda
def test_cache_inputs_skips_cudagraph_padding_slot():
    """idx_mapping=-1 的补齐请求不能写 0 号 slot。"""
    max_reqs, hidden_size, K = 3, 2, 3
    buffers = _new_buffers(max_reqs, 8)
    buffers.input_ids[:2] = torch.tensor([7, 8], dtype=torch.int32, device=DEVICE)
    _set_qsl(buffers, [0, 1, 2])
    hidden = torch.ones(8, hidden_size, device=DEVICE)
    cached_ids = torch.full((max_reqs, K - 1), -9, dtype=torch.int64, device=DEVICE)
    cached_hidden = torch.full((max_reqs, K - 1, hidden_size), -9.0, device=DEVICE)
    mine.cache_inputs(
        buffers, None, hidden, cached_ids, None, cached_hidden,
        _last([0, 1]), torch.tensor([2, -1], dtype=torch.int32, device=DEVICE),
        2, K, False,
    )
    assert cached_ids[2].tolist() == [-9, 7]
    assert torch.equal(cached_ids[0], torch.full((K - 1,), -9, dtype=torch.int64, device=DEVICE))
    assert torch.equal(cached_hidden[0], torch.full((K - 1, hidden_size), -9.0, device=DEVICE))


# ---------------------------------------------------------------------------
# 2. prepare_input_buffers：num_rejected=0..K、K=1..4、query 边界
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize(
    "K,r", [(K, r) for K in (1, 2, 3, 4) for r in range(K + 1)]
)
def test_prepare_input_buffers_differential(K, r):
    """整个 packed 输入逐值与冻结 kernel 相等；并手算关键行。"""
    max_reqs, B, hidden_size = 4, 2, 4
    query_len = K + 1
    total = B * query_len
    max_tokens = 32
    qsl = [0, query_len, total]
    target_ids, target_positions = _target_rows(total, 100, 10)
    target_seq_lens = [30 + query_len, 40 + query_len]
    idx_mapping = [3, 1]
    batch = _batch(qsl, target_ids, target_positions, target_seq_lens, idx_mapping)

    cached_ids = (
        torch.arange(max_reqs * (K - 1), dtype=torch.int64, device=DEVICE)
        .reshape(max_reqs, K - 1)
        + 1000
    )
    num_sampled = torch.tensor([1, 1], dtype=torch.int32, device=DEVICE)
    num_rejected = torch.tensor([r, r], dtype=torch.int32, device=DEVICE)
    # 持久 slot 1/3 的 last_sampled 不同；若代码误用 batch 行会读成另一条请求。
    last_sampled = torch.tensor([0, 71, 0, 73], dtype=torch.int64, device=DEVICE)
    next_prefill = (
        torch.arange(K * max_reqs, dtype=torch.int32, device=DEVICE)
        .reshape(K, max_reqs)
        + 500
    )
    overrides = torch.full(
        (max_reqs, K - 1), -1, dtype=torch.int64, device=DEVICE
    )

    local_buf, local_last, local_ovr = _run_prepare_pair(
        num_reqs=B,
        max_reqs=max_reqs,
        max_tokens=max_tokens,
        input_batch=batch,
        cached_ids=cached_ids,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
        last_sampled=last_sampled,
        next_prefill=next_prefill,
        overrides=overrides,
        qsl=qsl,
        num_speculative_steps=K,
    )

    num_reprefill = max(0, r - 1)
    num_input = query_len - r
    for req in range(B):
        qs = qsl[req]
        slot = idx_mapping[req]
        expected_last = qs + num_reprefill + num_input - 1
        assert int(local_last[req]) == expected_last
        assert int(local_buf.seq_lens[req]) == target_seq_lens[req] - num_reprefill
        assert int(local_buf.input_ids[expected_last]) == int(last_sampled[slot])

        # target 第 1..num_input-1 行 → draft 的 num_reprefill 起。
        for i in range(1, num_input):
            dst = qs + num_reprefill - 1 + i
            assert int(local_buf.input_ids[dst]) == int(target_ids[qs + i])
        # gap 从缓存列 [K-1-num_reprefill .. K-2] 读。
        for i in range(num_reprefill):
            read_col = K - 1 - num_reprefill + i
            assert int(local_buf.input_ids[qs + i]) == int(cached_ids[slot, read_col])
        # positions：gap 在 first_position 之前，随后是 target 的 0..num_input-1。
        first_pos = int(target_positions[qs])
        for i in range(num_reprefill):
            assert int(local_buf.positions[qs + i]) == first_pos - num_reprefill + i
        for i in range(num_input):
            assert int(local_buf.positions[qs + num_reprefill + i]) == int(
                target_positions[qs + i]
            )
        # 本请求尾部的 padding 行保持哨兵，不能被上一轮残值污染。
        for row in range(expected_last + 1, qsl[req + 1]):
            assert int(local_buf.input_ids[row]) == -7

    # 超出真实请求数的 qsl/seq_lens/last 必须被补齐成空值。
    assert int(local_buf.query_start_loc[B]) == total
    assert torch.equal(
        local_buf.query_start_loc[B + 1 :],
        torch.full_like(local_buf.query_start_loc[B + 1 :], total),
    )
    assert torch.equal(
        local_buf.seq_lens[B:],
        torch.zeros_like(local_buf.seq_lens[B:]),
    )
    assert torch.equal(
        local_last[B:], torch.zeros_like(local_last[B:])
    )
    # 本用例 num_sampled>0：所有 override 都必须是 -1。
    assert torch.equal(
        local_ovr, torch.full_like(local_ovr, -1)
    )


@requires_cuda
@pytest.mark.parametrize("query_len", [1, 2, 4])
def test_prepare_input_buffers_query_length_boundaries(query_len):
    """query 长度 < / = / > K-1（K=3）时，左移与补 next 的位置。"""
    K, max_reqs, max_tokens = 3, 2, 12
    target_ids, target_positions = _target_rows(query_len, 200, 50)
    qsl = [0, query_len]
    batch = _batch(qsl, target_ids, target_positions, [70 + query_len], [1])
    cached_ids = torch.full((max_reqs, K - 1), -9, dtype=torch.int64, device=DEVICE)
    num_sampled = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    num_rejected = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    last_sampled = torch.tensor([0, 900], dtype=torch.int64, device=DEVICE)
    next_prefill = torch.zeros(K, max_reqs, dtype=torch.int32, device=DEVICE)
    overrides = torch.full((max_reqs, K - 1), -1, dtype=torch.int64, device=DEVICE)

    local_buf, local_last, _ = _run_prepare_pair(
        num_reqs=1,
        max_reqs=max_reqs,
        max_tokens=max_tokens,
        input_batch=batch,
        cached_ids=cached_ids,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
        last_sampled=last_sampled,
        next_prefill=next_prefill,
        overrides=overrides,
        qsl=qsl,
        num_speculative_steps=K,
    )
    expected_last = query_len - 1
    assert int(local_last[0]) == expected_last
    assert int(local_buf.seq_lens[0]) == 70 + query_len
    # target 第 1 行起整块移进 draft 第 0 行；右端补 last_sampled。
    for i in range(1, query_len):
        assert int(local_buf.input_ids[i - 1]) == int(target_ids[i])
    assert int(local_buf.input_ids[expected_last]) == 900
    # 尾部 row query_len 保持哨兵；query_len=4 时没有尾部（缓冲 row 4 仍是哨兵）。
    assert int(local_buf.input_ids[query_len]) == -7
    for i in range(query_len):
        assert int(local_buf.positions[i]) == int(target_positions[i])


@requires_cuda
def test_prepare_input_buffers_partial_prefill_overrides_use_slot():
    """chunked prefill 的 future token 按持久 slot 取，不能 `next_prefill[:, :B]`。"""
    K, max_reqs, B = 3, 4, 2
    query_len = K + 1
    total = B * query_len
    target_ids, target_positions = _target_rows(total, 300, 20)
    qsl = [0, query_len, total]
    idx_mapping = [3, 1]
    batch = _batch(
        qsl, target_ids, target_positions, [50, 60], idx_mapping
    )
    cached_ids = torch.full((max_reqs, K - 1), -9, dtype=torch.int64, device=DEVICE)
    num_sampled = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    num_rejected = torch.tensor([0, 0], dtype=torch.int32, device=DEVICE)
    last_sampled = torch.tensor([0, 77, 0, 0], dtype=torch.int64, device=DEVICE)
    # 每一格都不同：若按 batch 行取，第 0 行会拿到 *0，而正确是 *3。
    next_prefill = (
        torch.arange(K * max_reqs, dtype=torch.int32, device=DEVICE)
        .reshape(K, max_reqs)
        + 500
    )
    overrides = torch.full((max_reqs, K - 1), -1, dtype=torch.int64, device=DEVICE)

    local_buf, local_last, local_ovr = _run_prepare_pair(
        num_reqs=B,
        max_reqs=max_reqs,
        max_tokens=32,
        input_batch=batch,
        cached_ids=cached_ids,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
        last_sampled=last_sampled,
        next_prefill=next_prefill,
        overrides=overrides,
        qsl=qsl,
        num_speculative_steps=K,
    )
    # req0 还在 chunked prefill：override 读 next_prefill[i, slot=3]。
    assert int(local_ovr[0, 0]) == int(next_prefill[1, 3])
    assert int(local_ovr[0, 1]) == int(next_prefill[2, 3])
    assert int(local_ovr[0, 0]) != int(next_prefill[1, 0])
    # req1 已有采样：override 全部 -1，next token 取 last_sampled[slot=1]。
    assert int(local_ovr[1, 0]) == -1
    assert int(local_ovr[1, 1]) == -1
    assert int(local_buf.input_ids[int(local_last[0])]) == int(next_prefill[0, 3])
    assert int(local_buf.input_ids[int(local_last[1])]) == 77


# ---------------------------------------------------------------------------
# 3. hidden / embedding：同一套 gap、同源数据
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize(
    "K,r",
    [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3),
     (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)],
)
def test_prepare_input_hidden_and_embeds_differential(K, r):
    """hidden 与 embedding 从同一个 cache 列补；整个缓冲与冻结 kernel 相等。"""
    max_reqs, B, hidden_size, max_tokens = 4, 2, 8, 32
    qlens = [1, K + 1] if r <= 1 else [r + 1, K + 1]
    qsl = [0, qlens[0], qlens[0] + qlens[1]]
    total = qsl[-1]
    target_ids, target_positions = _target_rows(total, 100, 30)
    batch = _batch(qsl, target_ids, target_positions, [20, 30], [3, 1])
    target_hidden = (
        torch.arange(total * hidden_size, dtype=torch.float32, device=DEVICE)
        .reshape(total, hidden_size)
        + 1000
    )
    input_embeds = (
        torch.arange(max_tokens * hidden_size, dtype=torch.float32, device=DEVICE)
        .reshape(max_tokens, hidden_size)
        + 2000
    )
    cached_hidden = (
        torch.arange(max_reqs * (K - 1) * hidden_size, dtype=torch.float32, device=DEVICE)
        .reshape(max_reqs, K - 1, hidden_size)
        + 3000
    )
    cached_embeds = cached_hidden + 4000
    num_rejected = torch.tensor([r, r], dtype=torch.int32, device=DEVICE)
    max_query_len = max(qlens)

    local_hidden = torch.full((max_tokens, hidden_size), -3.0, device=DEVICE)
    up_hidden = local_hidden.clone()
    local_embeds = input_embeds.clone()
    up_embeds = input_embeds.clone()
    local_buffers = _new_buffers(max_reqs, max_tokens)
    up_buffers = _new_buffers(max_reqs, max_tokens)
    for buf in (local_buffers, up_buffers):
        _set_qsl(buf, qsl)

    mine.prepare_input_hidden_states_and_embeddings(
        B, max_query_len, local_hidden, target_hidden, cached_hidden,
        local_embeds, cached_embeds, batch, local_buffers, num_rejected, K,
    )
    up.prepare_input_hidden_states_and_embeddings(
        B, max_query_len, up_hidden, target_hidden, cached_hidden,
        up_embeds, cached_embeds, batch, up_buffers, num_rejected, K,
    )
    _assert_equal("hidden differential", local_hidden, up_hidden)
    _assert_equal("embeds differential", local_embeds, up_embeds)

    num_reprefill = max(0, r - 1)
    for req in range(B):
        qs = qsl[req]
        num_input = qlens[req] - r
        slot = [3, 1][req]
        for i in range(num_input):
            assert torch.equal(
                local_hidden[qs + num_reprefill + i],
                target_hidden[qs + i],
            ), (req, i)
        for i in range(num_reprefill):
            read_col = K - 1 - num_reprefill + i
            assert torch.equal(
                local_hidden[qs + i], cached_hidden[slot, read_col]
            ), (req, i)
            # embedding 的 gap 同列同源。
            assert torch.equal(
                local_embeds[qs + i], cached_embeds[slot, read_col]
            ), (req, i)
        # 非 gap 的 embedding 不由本函数写：保持调用方原值。
        for row in range(qs + num_reprefill, qs + qlens[req]):
            assert torch.equal(local_embeds[row], input_embeds[row])


@requires_cuda
def test_prepare_input_hidden_without_embeds_branch():
    """无多模态模型：input_embeds 为空时只搬 hidden，不要求 embed cache。"""
    K, max_reqs, B, hidden_size, max_tokens = 3, 4, 1, 4, 8
    qsl = [0, 4]
    target_ids, target_positions = _target_rows(4, 500, 80)
    batch = _batch(qsl, target_ids, target_positions, [90], [2])
    target_hidden = torch.arange(4 * hidden_size, dtype=torch.float32, device=DEVICE).reshape(4, hidden_size)
    cached_hidden = torch.full((max_reqs, K - 1, hidden_size), -9.0, device=DEVICE)
    num_rejected = torch.tensor([2], dtype=torch.int32, device=DEVICE)
    local_hidden = torch.full((max_tokens, hidden_size), -3.0, device=DEVICE)
    up_hidden = local_hidden.clone()
    local_buffers = _new_buffers(max_reqs, max_tokens)
    up_buffers = _new_buffers(max_reqs, max_tokens)
    for buf in (local_buffers, up_buffers):
        _set_qsl(buf, qsl)

    mine.prepare_input_hidden_states_and_embeddings(
        B, 4, local_hidden, target_hidden, cached_hidden,
        None, None, batch, local_buffers, num_rejected, K,
    )
    up.prepare_input_hidden_states_and_embeddings(
        B, 4, up_hidden, target_hidden, cached_hidden,
        None, None, batch, up_buffers, num_rejected, K,
    )
    _assert_equal("hidden no-embeds", local_hidden, up_hidden)
    # r=2 -> 1 行 gap；gap 来自 cache 列 1，其余 target 行右移。
    assert torch.equal(local_hidden[0], cached_hidden[2, 1])
    assert torch.equal(local_hidden[1], target_hidden[0])
    assert torch.equal(local_hidden[2], target_hidden[1])


# ---------------------------------------------------------------------------
# 4. padding：尾部槽位必须真的变成 PAD
# ---------------------------------------------------------------------------


@requires_cuda
def test_pad_trailing_draft_slots_actually_masks_tail():
    """req0 有真实尾部要屏蔽；req1 传 query_end-1（空操作）不能代表通过。"""
    max_reqs, num_groups, num_tokens = 4, 2, 8
    qsl = _qsl([0, 4, 8])
    last = _last([2, 7])  # req0 尾部行 3；req1 尾部为空
    base = torch.arange(
        num_groups * num_tokens, dtype=torch.int64, device=DEVICE
    ).reshape(num_groups, num_tokens) + 5
    local = base.clone()
    up_slots = base.clone()

    mine.pad_trailing_draft_slots(local, qsl, last, 2)
    up.pad_trailing_draft_slots(up_slots, qsl, last, 2)
    _assert_equal("padded slots", local, up_slots)

    # req0：0..2 不动，3 变 PAD。
    assert torch.equal(local[:, 3], torch.full((num_groups,), PAD_SLOT_ID, dtype=torch.int64, device=DEVICE))
    assert torch.equal(local[:, :3], base[:, :3])
    # req1：last=7 是 query_end-1，整个请求原地不动。
    assert torch.equal(local[:, 4:8], base[:, 4:8])
    # 如果误把 last 传成 query_end-1，req0 的尾部不会被屏蔽——这里显式对照。
    noop = base.clone()
    mine.pad_trailing_draft_slots(noop, qsl, _last([3, 7]), 2)
    assert torch.equal(noop, base)


# ---------------------------------------------------------------------------
# 5. 窗口移动：丢最左、右边追加候选
# ---------------------------------------------------------------------------


@requires_cuda
def test_update_draft_inputs_shifts_and_appends_per_request():
    """两条请求边界不能串；token/embedding 同步移动。"""
    max_reqs, hidden_size = 4, 4
    qsl = [0, 3, 7]
    idx = torch.tensor([3, 1], dtype=torch.int32, device=DEVICE)
    last = _last([2, 6])
    base_ids = torch.arange(8, dtype=torch.int32, device=DEVICE) + 100
    local = _new_buffers(max_reqs, 8)
    up_buf = _new_buffers(max_reqs, 8)
    for buf in (local, up_buf):
        buf.input_ids[:8] = base_ids
        _set_qsl(buf, qsl)
    base_embeds = torch.arange(8 * hidden_size, dtype=torch.float32, device=DEVICE).reshape(8, hidden_size)
    local_embeds = base_embeds.clone()
    up_embeds = base_embeds.clone()
    draft_tokens = torch.tensor([50, 60], dtype=torch.int64, device=DEVICE)
    draft_embeds = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], device=DEVICE
    )

    mine.update_draft_inputs(
        draft_tokens, draft_embeds, local, local_embeds, last, idx, 2
    )
    up.update_draft_inputs(
        draft_tokens, draft_embeds, up_buf, up_embeds, last, idx, 2
    )
    _assert_equal("shifted input_ids", local.input_ids, up_buf.input_ids)
    _assert_equal("shifted embeds", local_embeds, up_embeds)

    assert local.input_ids[:3].tolist() == [101, 102, 50]
    assert local.input_ids[3:7].tolist() == [104, 105, 106, 60]
    assert int(local.input_ids[3]) == 104  # 请求 2 的起点没有被请求 1 的尾部覆盖
    assert int(local.input_ids[7]) == 107  # last 之后的行不动
    assert torch.equal(local_embeds[0], base_embeds[1])
    assert torch.equal(local_embeds[1], base_embeds[2])
    assert torch.equal(local_embeds[2], draft_embeds[0])
    assert torch.equal(local_embeds[6], draft_embeds[1])


# ---------------------------------------------------------------------------
# 6. K=1：空窗口不能被 max(1, K-1) 改成 1 列
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("K", [1, 2, 4])
def test_cache_width_is_exactly_k_minus_one(K):
    """协议宽度；K=1 就是 0 列，不是 1 列。"""
    cache = torch.zeros(3, K - 1, dtype=torch.int64)
    assert cache.shape[1] == K - 1


@requires_cuda
def test_k1_empty_window_end_to_end_layout():
    """K=1：cache 宽 0；prepare/cache/hidden 都不应写 cache，也不应崩。"""
    K, max_reqs, max_tokens, hidden_size = 1, 3, 8, 4
    target_ids, target_positions = _target_rows(1, 700, 90)
    qsl = [0, 1]
    batch = _batch(qsl, target_ids, target_positions, [95], [2])
    cached_ids = torch.zeros(max_reqs, K - 1, dtype=torch.int64, device=DEVICE)
    cached_hidden = torch.zeros(max_reqs, K - 1, hidden_size, device=DEVICE)
    num_sampled = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    num_rejected = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    last_sampled = torch.tensor([0, 0, 42], dtype=torch.int64, device=DEVICE)
    next_prefill = torch.zeros(K, max_reqs, dtype=torch.int32, device=DEVICE)
    overrides = torch.full((max_reqs, K - 1), -1, dtype=torch.int64, device=DEVICE)
    buffers = _new_buffers(max_reqs, max_tokens)
    _set_qsl(buffers, qsl)
    last = torch.full((max_reqs,), -5, dtype=torch.int64, device=DEVICE)

    mine.prepare_input_buffers(
        1, batch, cached_ids, num_sampled, num_rejected, last_sampled,
        next_prefill, overrides, buffers, last, max_reqs, K,
    )
    assert int(last[0]) == 0
    assert int(buffers.input_ids[0]) == 42
    assert int(buffers.seq_lens[0]) == 95

    hidden = torch.full((max_tokens, hidden_size), -3.0, device=DEVICE)
    target_hidden = torch.arange(
        hidden_size, dtype=torch.float32, device=DEVICE
    ).reshape(1, hidden_size)
    buffers.query_start_loc[:2] = _qsl(qsl)
    mine.prepare_input_hidden_states_and_embeddings(
        1, 1, hidden, target_hidden, cached_hidden, None, None,
        batch, buffers, num_rejected, K,
    )
    assert torch.equal(hidden[0], target_hidden[0])

    before_ids = cached_ids.clone()
    before_hidden = cached_hidden.clone()
    mine.cache_inputs(
        buffers, None, hidden, cached_ids, None, cached_hidden,
        last, batch.idx_mapping, 1, K, False,
    )
    assert torch.equal(cached_ids, before_ids)
    assert torch.equal(cached_hidden, before_hidden)
