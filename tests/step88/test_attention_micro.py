"""88 后续：2D grid 数学与 3D q>1 原型的正确性。"""

from __future__ import annotations

import torch

from benchmarks.check_step88_attention_micro import (
    _correctness,
    _grid_info,
    _prepare,
    case_matrix,
)

requires_cuda = torch.cuda.is_available()


def _case(label: str) -> dict:
    return next(case for case in case_matrix() if case["label"] == label)


def test_microbenchmark_cases_are_unique_and_cover_axes():
    cases = case_matrix()
    assert len(cases) == 22
    assert len({case["label"] for case in cases}) == 22
    assert {case["q_len"] for case in cases} >= {1, 2, 3, 4, 8}
    assert {case["context"] for case in cases} >= {128, 1024, 2048}
    assert {case["batch"] for case in cases} >= {1, 4, 8}
    assert {case["head_dim"] for case in cases} == {256, 512}
    assert {case["window"] for case in cases} == {-1, 512}


def test_2d_grid_does_not_grow_with_context_but_3d_does():
    case = _case("qlen3_ctx1024")
    two = _grid_info(case, is_3d=False)
    three = _grid_info(case, is_3d=True)
    # Gemma4 target: 8 query heads / 1 KV head → BLOCK_Q=2；3 行 query → 2 blocks。
    assert two["block_q"] == 2
    assert two["q_blocks"] == 2
    assert two["programs"] == 2
    assert two["tiles_per_program"] == 33
    # 3D 加 16 个 segment；每个 program 只扫约 1/16 的 KV。
    assert three["q_blocks"] == 2
    assert three["programs"] == 32
    assert three["tiles_per_program"] == 5

    longer = _case("qlen3_ctx2048")
    assert _grid_info(longer, is_3d=False)["programs"] == 2
    assert _grid_info(longer, is_3d=False)["tiles_per_program"] > 60

    batch4 = _case("qlen3_b4_ctx1024")
    assert _grid_info(batch4, is_3d=False)["programs"] == 10
    assert _grid_info(batch4, is_3d=True)["programs"] == 160


@torch.no_grad()
def test_3d_q_len_greater_than_one_matches_2d_for_long_context():
    if not requires_cuda:
        return
    for label in ("qlen3_ctx1024", "qlen3_ctx2048", "qlen3_hd512_win-1"):
        case = _case(label)
        prepared = _prepare(case)
        result = _correctness(case, prepared)
        assert result["ok"], (label, result)
        assert result["max_abs"] < 2e-2, (label, result["max_abs"])
        del prepared
