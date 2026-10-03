"""step59：`SpecDecodeMetadata` 的索引口径（对应需求 059 §2 的算例与 vLLM `_calc_spec_decode_metadata`）。

两个坐标系必须分清：

    logits_indices          指向**扁平化的 forward 行**
    target/bonus indices    指向**取完之后**的 `[P+B, V]` 紧凑张量

下面每个数字都能在上游 `gpu_model_runner._calc_spec_decode_metadata` 的注释里逐行对上。
"""

import numpy as np
import pytest
import torch
from helpers import meta_drafts, metadata_for

from minivllm.testing.spec_metadata import make_metadata

# 上游注释里的算例：cu_num_scheduled_tokens / num_draft_tokens
DOCSTRING_CU_SCHEDULED = np.array([4, 104, 107, 207, 209], dtype=np.int32)
DOCSTRING_NUM_DRAFT = np.array([3, 0, 2, 0, 1], dtype=np.int32)
DOCSTRING_REQ_IDS = ["r0", "r1", "r2", "r3", "r4"]
# 输入行里逐请求的 (b, [draft...])：草稿行必须与协议一致（方法里会比对）。
# 注意第 2、4 条请求的 K=0 但排了 100 行——它们是**中间 prefill 块**（这正是上游那个
# 算例里 cu_num_scheduled 出现 4→104→107 跳跃的原因），所以输入行要按块起点摆。
DOCSTRING_ROWS = [(100, [11, 12, 13]), (200, []), (101, [21, 22]), (201, []), (102, [31])]


def stub_runner(req_ids, size=4096):
    """只带 `_calc_spec_decode_metadata` 需要的那几个属性的 Runner 替身。

    这个方法只用到 `device` / `_arange_np` / `_arange_scratch` / `input_batch.req_ids`
    与 `req_id_to_index`——不碰模型、不碰 KV，所以不用真的建一个 Runner。
    """
    from types import SimpleNamespace

    from minivllm.worker.gpu_model_runner import GPUModelRunner

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.device = "cuda"
    runner._arange_np = np.arange(size, dtype=np.int64)
    runner._arange_scratch = np.empty(size, dtype=np.int64)
    runner.input_batch = SimpleNamespace(
        req_ids=list(req_ids),
        req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)})
    return runner


def input_rows(rows, cu_scheduled):
    """把逐请求的 (b, [draft...]) 按**块起点**摆进一条扁平输入行数组。

    第 i 条请求占 `[cu_scheduled[i] - num_scheduled[i], cu_scheduled[i])`：最后 K 行是草稿、
    它前面一行是 b；K=0 的中间 prefill 块只有最后一行（b）会被取到 logits。
    """
    total = int(cu_scheduled[-1])
    flat = [0] * total
    for (backup, drafts), end in zip(rows, cu_scheduled):
        end = int(end)
        block = [backup] + list(drafts)
        start = end - len(block)
        flat[start:end] = block
        # b 行在草稿之前：块内位置是 end-len(block)..end-1，b 就在 end-len(block)
    return torch.tensor(flat, dtype=torch.int64)


def protocol_from_rows(rows):
    return {f"r{index}": drafts for index, (_, drafts) in enumerate(rows) if drafts}


def calc(rows=DOCSTRING_ROWS, req_ids=DOCSTRING_REQ_IDS, cu_scheduled=DOCSTRING_CU_SCHEDULED,
         num_draft=DOCSTRING_NUM_DRAFT, protocol=None):
    runner = stub_runner(req_ids)
    cu_scheduled = np.asarray(cu_scheduled, dtype=np.int32)
    return runner._calc_spec_decode_metadata(
        np.asarray(num_draft, dtype=np.int32), cu_scheduled,
        input_rows(rows, cu_scheduled),
        protocol_from_rows(rows) if protocol is None else protocol)


def test_upstream_docstring_example_indices(cuda_device):
    """上游注释里的那 5 条请求 → 逐值核对 6 个索引数组。"""
    meta = calc()
    assert meta.num_draft_tokens == [3, 0, 2, 0, 1]
    assert meta.cu_num_draft_tokens.tolist() == [3, 3, 5, 5, 6]
    assert meta.cu_num_sampled_tokens.tolist() == [4, 5, 8, 9, 11]
    assert meta.logits_indices.tolist() == [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
    assert meta.target_logits_indices.tolist() == [0, 1, 2, 5, 6, 9]
    assert meta.bonus_logits_indices.tolist() == [3, 4, 7, 8, 10]
    assert meta.max_spec_len == 3
    # 草稿 token 从"输入行的下一行"取（上游 `input_ids[logits_indices][target+1]`）
    assert meta.draft_token_ids.tolist() == [11, 12, 13, 21, 22, 31]


def test_dtypes_and_device_no_leading_zero(cuda_device):
    """索引张量是 GPU 上的 int32，且累积和**不带开头的 0**（059 §2）。"""
    meta = calc()
    for name in ("draft_token_ids", "cu_num_draft_tokens", "cu_num_sampled_tokens",
                 "target_logits_indices", "bonus_logits_indices", "logits_indices"):
        tensor = getattr(meta, name)
        assert tensor.dtype == torch.int32, name
        assert str(tensor.device).startswith("cuda"), name
    # 开头是第 0 条请求自己的 K，而不是 0
    assert meta.cu_num_draft_tokens[0] == 3
    assert meta.cu_num_draft_tokens.shape[0] == len(meta.num_draft_tokens)
    assert meta.cu_num_sampled_tokens.shape[0] == len(meta.num_draft_tokens)


def test_abc_ragged_example(cuda_device):
    """059 §2 的小例子：B=3、候选数 [3,0,1]、D=4。"""
    rows = [(7, [1, 2, 3]), (8, []), (9, [4])]
    meta = calc(rows=rows, req_ids=["r0", "r1", "r2"],
                cu_scheduled=np.array([4, 5, 7], dtype=np.int32),
                num_draft=np.array([3, 0, 1], dtype=np.int32))
    assert meta.num_draft_tokens == [3, 0, 1]
    assert meta.cu_num_draft_tokens.tolist() == [3, 3, 4]
    assert meta.cu_num_sampled_tokens.tolist() == [4, 5, 7]
    assert meta.logits_indices.tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert meta.target_logits_indices.tolist() == [0, 1, 2, 5]
    assert meta.bonus_logits_indices.tolist() == [3, 4, 6]
    assert meta.draft_token_ids.tolist() == [1, 2, 3, 4]


def test_batch_one(cuda_device):
    meta = calc(rows=[(7, [5, 6])], req_ids=["r0"],
                cu_scheduled=np.array([3], dtype=np.int32),
                num_draft=np.array([2], dtype=np.int32))
    assert meta.max_spec_len == 2
    assert meta.cu_num_draft_tokens.tolist() == [2]
    assert meta.cu_num_sampled_tokens.tolist() == [3]
    assert meta.logits_indices.tolist() == [0, 1, 2]
    assert meta.target_logits_indices.tolist() == [0, 1]
    assert meta.bonus_logits_indices.tolist() == [2]
    assert meta.draft_token_ids.tolist() == [5, 6]


def test_all_requests_without_drafts(cuda_device):
    """D=0：没有草稿的批也要能建元数据（每条请求的 bonus 行就是它唯一那行）。"""
    rows = [(7, []), (8, [])]
    meta = calc(rows=rows, req_ids=["r0", "r1"],
                cu_scheduled=np.array([1, 2], dtype=np.int32),
                num_draft=np.array([0, 0], dtype=np.int32))
    assert meta.max_spec_len == 0
    assert meta.draft_token_ids.tolist() == []
    assert meta.target_logits_indices.tolist() == []
    assert meta.cu_num_sampled_tokens.tolist() == [1, 2]
    assert meta.bonus_logits_indices.tolist() == [0, 1]
    assert meta.logits_indices.tolist() == [0, 1]


def test_k_zero_bonus_row_is_its_only_query_row(cuda_device):
    """K=0 的请求：bonus 行就是 b 行（普通 decode 是投机的特例）。"""
    rows = [(7, [9]), (8, [])]
    meta = calc(rows=rows, req_ids=["r0", "r1"],
                cu_scheduled=np.array([2, 3], dtype=np.int32),
                num_draft=np.array([1, 0], dtype=np.int32))
    # r0 的 query 是行 0,1；r1 的是行 2
    assert meta.logits_indices.tolist() == [0, 1, 2]
    assert meta.target_logits_indices.tolist() == [0]
    assert meta.bonus_logits_indices.tolist() == [1, 2]


def test_wrong_num_scheduled_raises(cuda_device):
    """带草稿的请求 query 必须是 K+1 行，否则 `target+1` 会取到别的 token。"""
    rows = [(7, [1, 2])]
    with pytest.raises(ValueError, match="K\\+1"):
        calc(rows=rows, req_ids=["r0"], cu_scheduled=np.array([4], dtype=np.int32),
             num_draft=np.array([2], dtype=np.int32))


def test_input_rows_must_match_protocol(cuda_device):
    """输入缓冲与调度快照不同步 → 明确报错（不静默验证别的 token）。"""
    rows = [(7, [1, 2])]
    with pytest.raises(RuntimeError, match="scheduled_spec_decode_tokens"):
        calc(rows=rows, req_ids=["r0"], cu_scheduled=np.array([3], dtype=np.int32),
             num_draft=np.array([2], dtype=np.int32), protocol={"r0": [1, 9]})


def test_testing_helper_matches_runner(cuda_device):
    """测试辅助 `make_metadata` 与 Runner 的算法必须给出同一份索引。"""
    drafts = [[1, 2, 3], [], [4]]
    runner_meta = calc(rows=[(7, [1, 2, 3]), (8, []), (9, [4])],
                       req_ids=["r0", "r1", "r2"],
                       cu_scheduled=np.array([4, 5, 7], dtype=np.int32),
                       num_draft=np.array([3, 0, 1], dtype=np.int32))
    helper_meta = make_metadata(drafts, device="cuda")
    for name in ("draft_token_ids", "cu_num_draft_tokens", "cu_num_sampled_tokens",
                 "target_logits_indices", "bonus_logits_indices", "logits_indices"):
        assert torch.equal(getattr(runner_meta, name), getattr(helper_meta, name)), name
    assert runner_meta.num_draft_tokens == helper_meta.num_draft_tokens
    assert meta_drafts(runner_meta) == drafts


def test_draft_slice_helper(cuda_device):
    meta = metadata_for([[1, 2, 3], [], [4]])
    assert meta_drafts(meta) == [[1, 2, 3], [], [4]]
