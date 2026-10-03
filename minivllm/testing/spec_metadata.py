"""测试辅助：不建 Runner 也能造出一份 `SpecDecodeMetadata`（只给测试）。

索引算法与 **Runner 的 `_calc_spec_decode_metadata`（上游同名方法）逐句相同**：
累积和 → `repeat` 展开 + `arange` 偏移 → `target/bonus/logits_indices`。
`tests/step59/test_metadata.py` 里有一条用例拿真实的 Runner 对着它逐值比，防止两边漂移。

草稿 token 也是按上游那条表达式取的：
`draft_token_ids = input_ids[logits_indices][target_logits_indices + 1]`
——"草稿本来就是输入行"，取它们的下一行就是草稿本身。没给 `input_ids` 时按
`[b][d1..dK]` 的布局现场拼一份（b 行只是一个占位 token，验证不用它）。
"""

import numpy as np
import torch

from ..spec_decode.metadata import SpecDecodeMetadata


def _cumsum_and_arange(num_tokens: np.ndarray, total: int) -> tuple[np.ndarray, np.ndarray]:
    """上游 `_get_cumsum_and_arange`：[2,5,3] → cu=[2,7,10]、arange=[0,1,0,1,2,3,4,0,1,2]。"""
    cu_num_tokens = np.cumsum(num_tokens, dtype=np.int32)
    offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
    arange = np.arange(total, dtype=np.int32) - offsets
    return cu_num_tokens, arange


def make_metadata(draft_token_ids: list[list[int]],
                  num_scheduled: list[int] | None = None,
                  input_ids: list[int] | None = None,
                  device: str | torch.device = "cpu") -> SpecDecodeMetadata:
    """`draft_token_ids`：逐请求的草稿（K=0 用空列表）。

    `num_scheduled` 缺省按"带草稿的请求排 K+1 行、其余排 1 行"给（本项目的调度约定）。
    """
    ks = [len(draft) for draft in draft_token_ids]
    if num_scheduled is None:
        num_scheduled = [k + 1 if k else 1 for k in ks]
    num_scheduled_arr = np.array(num_scheduled, dtype=np.int32)
    num_draft_arr = np.array(ks, dtype=np.int32)
    num_sampled_arr = num_draft_arr + 1

    if input_ids is None:
        # 上游的输入行布局：[b][d1..dK]；b 行（占位 0）只影响 metadata 之外的东西
        input_ids = []
        for draft in draft_token_ids:
            input_ids.extend([0] + list(draft))
    assert len(input_ids) == int(num_scheduled_arr.sum()), (
        f"input_ids 有 {len(input_ids)} 行，但按调度应该是 {int(num_scheduled_arr.sum())} 行")

    cu_num_sampled, arange_sampled = _cumsum_and_arange(
        num_sampled_arr, int(num_sampled_arr.sum()))
    # 每请求 query 块的起点 = 累积末端 - 本请求行数（**必须用累积值**：请求不是从 0 开始的）
    cu_num_scheduled = np.cumsum(num_scheduled_arr, dtype=np.int32)
    logits_indices = (np.repeat(cu_num_scheduled - num_sampled_arr, num_sampled_arr)
                      + arange_sampled)
    bonus_logits_indices = cu_num_sampled - 1

    cu_num_draft, arange_draft = _cumsum_and_arange(
        num_draft_arr, int(num_draft_arr.sum()))
    target_logits_indices = (np.repeat(cu_num_sampled - num_sampled_arr, num_draft_arr)
                             + arange_draft)

    gathered = np.array(input_ids, dtype=np.int64)[logits_indices]
    extracted = gathered[target_logits_indices + 1] if len(target_logits_indices) else []
    expected = [token for draft in draft_token_ids for token in draft]
    assert list(extracted) == expected, (
        f"输入行里取出来的草稿 {list(extracted)} 与传入的草稿 {expected} 不一致："
        f"说明 input_ids 的布局与本函数的 num_scheduled 假设不符")

    def to_device(array, dtype=torch.int32):
        return torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=dtype)

    return SpecDecodeMetadata(
        draft_token_ids=to_device(extracted),
        num_draft_tokens=ks,
        cu_num_draft_tokens=to_device(cu_num_draft),
        cu_num_sampled_tokens=to_device(cu_num_sampled),
        target_logits_indices=to_device(target_logits_indices),
        bonus_logits_indices=to_device(bonus_logits_indices),
        logits_indices=to_device(logits_indices),
    )
