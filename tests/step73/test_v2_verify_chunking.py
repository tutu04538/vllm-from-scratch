"""73 关：V2 投机验证的**分块上限**（`max_chunk_logits` / `_iter_request_chunks`）。

这个机制回答一个问题：**验证一批投机草稿时，那份 fp32 的 target logits 副本最多能有多大？**

`RejectionSampler._verify()` 先调 `sampler.apply_sampling_params()`，而它会把整段 logits
**升到 fp32 再原地施加**惩罚 / 温度 / min_p（`logits = torch.empty_like(logits, dtype=torch.float32)`）。
投机批的行数是 `Σ(1 + K_i)`，于是这份副本 = `行数 × vocab × 4` 字节：15 万词表时**一行就 0.58 MiB**，
512 条请求 × (1+4) 行 ≈ 1.5 GiB——不设上限，显存会随并发线性涨到不可预期。

上限就是 `get_max_chunk_logits(vocab_size) = 1 GiB // (vocab × 4)`（上游 `MAX_CHUNK_BYTES = 2**30`），
超过就按**整条请求**切块（`_iter_request_chunks`），逐块验证、逐块释放（`del processed_logits`），
最后把 `sampled` / `num_sampled` / logprobs 拼回去。

本用例只钉**切块规则本身**（纯 numpy，CPU 可跑）；数值正确性由
`tests/step73/test_v2_rejection_diff.py` 与上游内核逐值差分覆盖。

四条不变量：

    C1  每块的行数 ≤ `max_chunk_logits`（**除非单条请求自己就超**，那时也必须单独成块）
    C2  块边界落在请求边界上：块是 `[start, end)` 的**连续请求区间**，覆盖 0..num_reqs 无缝无重叠
    C3  每条请求只出现在一个块里（绝不切开一条请求——它的 `cu_num_logits` 前缀和是一个单位）
    C4  一定前进（`end ≥ start + 1`），不会死循环
"""

from __future__ import annotations

import numpy as np
import pytest

from minivllm.worker.gpu.spec_decode.rejection_sampler import (
    MAX_CHUNK_BYTES,
    _FP32_BYTES,
    _iter_request_chunks,
    get_max_chunk_logits,
)


def cu_of(rows_per_req: list[int]) -> np.ndarray:
    """按"每请求几行"造 `cu_num_logits`（含前导 0 的前缀和，与本模块的口径一致）。"""
    return np.concatenate(([0], np.cumsum(np.array(rows_per_req, dtype=np.int64))))


def test_chunk_budget_scales_with_vocab_size():
    """上限是"1 GiB 的 fp32 行"，不是固定的行数——词表越大，一次能放的行越少。"""
    assert get_max_chunk_logits(151936) == MAX_CHUNK_BYTES // (151936 * _FP32_BYTES) == 1766
    assert get_max_chunk_logits(32000) == 8388
    assert get_max_chunk_logits(128256) == 2092
    # 词表大到一行就超过预算时也不能算出 0 块（至少一块，哪怕那块只有一行）
    assert get_max_chunk_logits(MAX_CHUNK_BYTES) == 1


# 上限都取 ≥ 单条请求的最大行数（5）：上限比单条请求还小时是 C1 的例外，见下一个用例
@pytest.mark.parametrize("chunk", [5, 6, 7, 11, 64])
def test_chunks_respect_the_cap_and_stay_on_request_boundaries(chunk):
    """C1/C2/C3：不超上限、落在请求边界上、每条请求只出现在一个块里、且覆盖完整。"""
    rng = np.random.default_rng(0)
    rows_per_req = [int(x) for x in rng.integers(1, 6, size=40)]     # 40 条请求，每条约 1..5 行
    cu = cu_of(rows_per_req)

    assert max(rows_per_req) <= chunk, "本用例的前提：上限 ≥ 单条请求的最大行数"
    chunks = list(_iter_request_chunks(cu, chunk))
    assert chunks, "至少要产出一块"

    covered: list[int] = []
    for start, end in chunks:
        assert 0 <= start < end <= len(rows_per_req), (start, end)
        rows = int(cu[end] - cu[start])
        assert rows <= chunk, f"块 [{start},{end}) 有 {rows} 行，超过上限 {chunk}"
        covered += list(range(start, end))
    # C2/C3：无缝、无重叠、无遗漏（这正是"绝不切开请求"的等价说法）
    assert covered == list(range(len(rows_per_req)))


def test_oversized_request_still_gets_its_own_chunk():
    """C1 的例外+MUST：单条请求自己就超上限时，它单独成块（总不能把请求切开）。"""
    cu = cu_of([2, 2, 100, 2, 2])       # 中间那条请求 100 行
    chunks = list(_iter_request_chunks(cu, 10))
    assert chunks[0] == (0, 2), chunks          # 前面 2 条请求（4 行）先凑一块
    assert (2, 3) in chunks, chunks             # 超大的那条单独一块
    assert sum(end - start for start, end in chunks) == 5
    # 覆盖完整且不重叠
    assert [i for start, end in chunks for i in range(start, end)] == list(range(5))


def test_no_chunking_when_the_batch_fits():
    """批本身没超上限时，调用方**根本不切**（`logits.shape[0] <= max_chunk_logits` → 单块）。"""
    # 256 条请求 × (1+K=5) 行 = 1280 行 < 1766 → 一整块
    single = cu_of([5] * 256)
    assert int(single[-1]) <= get_max_chunk_logits(151936)
    # 512 条请求 × 5 行 = 2560 行 > 1766 → 至少两块，且每块 ≤ 1766 行
    big = cu_of([5] * 512)
    chunks = list(_iter_request_chunks(big, get_max_chunk_logits(151936)))
    assert len(chunks) == 2, chunks
    assert all(int(big[end] - big[start]) <= 1766 for start, end in chunks)
