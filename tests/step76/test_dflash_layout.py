"""76 关：DFlash 的**输入布局**（context / query / anchor / mask / sample 行）。

这一关钉三件事（需求 076 §4 前两条）：

    L1  同一条规则在 V1 与 V2 里**逐值**一致，且与上游内核的原始输出一致
        （V1 的 `copy_and_expand_dflash_inputs_kernel`、V2 的 `_prepare_dflash_inputs_kernel`）
    L2  被拒尾部：V1 内核按 `is_ctx` 把它们的**位置**也写出来（上游口径），V2 内核把它们
        标成 position 0 / slot −1（不写草稿缓存）——两条口径各自与上游逐值相等
    L3  采样行 = mask 行（不含 anchor），补齐行映射到 −1；位置/槽位/seq_lens 按绝对位置算

"被拒数少减一行"是本关最容易犯的错：query 位置整体后移 1，草稿写在从未存在的位置上，
不报错、只是草稿质量崩掉。`test_query_positions_subtract_rejected_rows` 用反证钉它。
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from spec76_helpers import (  # noqa: E402
    MASK_TOKEN_ID,
    fake_rows,
    layout_case,
    model_configs,
    requires_cuda,
    run_our_v1_expansion,
    run_our_v2_kernel,
    run_upstream_v1_kernel,
    run_upstream_v2_kernel,
)

from minivllm.config import ModelConfig, SpeculativeConfig  # noqa: E402
from minivllm.spec_decode.utils import (  # noqa: E402
    compute_slot_mapping_for_rows,
    expand_dflash_inputs,
)


# ---------------- L1/L2：与上游内核逐值差分 ----------------

@requires_cuda
def test_v1_layout_matches_the_upstream_kernel():
    """V1：位置/槽位/采样行与上游内核**逐值相等**（含被拒尾部整段写位置）。"""
    case = layout_case("cuda")
    upstream = run_upstream_v1_kernel(case)
    ours = run_our_v1_expansion(case)
    for name in ("input_ids", "context_positions", "query_positions",
                 "context_slot_mapping", "query_slot_mapping", "sample_indices"):
        assert torch.equal(upstream[name], ours[name]), name
    # 语义复核（手算）：请求 0 有效 2 行（位置 7,8），query 从 9 开始；请求 1 从 22 开始
    assert upstream["query_positions"].tolist() == [9, 10, 11, 12, 22, 23, 24, 25]
    assert upstream["context_positions"].tolist() == [7, 8, 9, 10, 20, 21]
    assert upstream["sample_indices"].tolist() == [1, 2, 3, 5, 6, 7]
    # mask 行的 token 全是占位符、第 0 行是 bonus（target 刚采出的 token）
    assert upstream["input_ids"].tolist() == [11, MASK_TOKEN_ID, MASK_TOKEN_ID, MASK_TOKEN_ID,
                                              22, MASK_TOKEN_ID, MASK_TOKEN_ID, MASK_TOKEN_ID]


@requires_cuda
def test_v2_layout_matches_the_upstream_kernel():
    """V2：12 个输出与上游内核逐值相等（含补齐行、被拒行的 position 0 / slot −1）。"""
    case = layout_case("cuda")
    upstream = run_upstream_v2_kernel(case)
    ours = run_our_v2_kernel(case)
    for name, want in upstream.items():
        assert torch.equal(want, ours[name]), name
    # 被拒尾部（请求 0 的位置 9、10）不写草稿缓存
    assert upstream["context_positions"][:6].tolist() == [7, 8, 0, 0, 20, 21]
    assert upstream["context_slot_mapping"][:6].tolist() == [7, 8, -1, -1, 84, 85]
    # query：位置接着**最后一个有效**位置（8/21），槽位按块表查
    assert upstream["query_positions"][:8].tolist() == [9, 10, 11, 12, 22, 23, 24, 25]
    assert upstream["seq_lens"][:2].tolist() == [13, 26]


@requires_cuda
def test_v2_padding_is_inert():
    """补齐行：映射 −1（不写任何请求）、槽位 −1、seq_lens 0、query_start_loc 收到批尾。

    图重放可能比真实批大，这些补齐行**每条都不能有副作用**——`sample_idx_mapping` 写 0
    就会把草稿散射进 0 号请求的槽位（上游注释点名的那个坑）。
    """
    case = layout_case("cuda")
    out = run_upstream_v2_kernel(case, max_tokens=64, max_num_reqs=8)
    assert out["query_start_loc"][:9].tolist() == [0, 4, 8, 8, 8, 8, 8, 8, 8]
    assert out["seq_lens"][:4].tolist() == [13, 26, 0, 0]
    assert out["sample_idx_mapping"][6:].tolist() == [-1] * (8 * 3 - 6)
    assert out["query_slot_mapping"][8:].tolist() == [-1] * (64 - 8)
    # 真实行的映射按 slot 给（idx_mapping = [0, 5]）
    assert out["sample_idx_mapping"][:6].tolist() == [0, 0, 0, 5, 5, 5]


# ---------------- L3：CPU 侧的规则（不依赖 CUDA） ----------------

def test_cpu_rules_for_positions_tokens_and_samples():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    case = layout_case(device)
    rows = fake_rows(case)
    expanded = expand_dflash_inputs(
        [int(value) for value in case.target_positions.tolist()],
        rows, num_query_per_req=case.num_query_per_req, mask_token_id=MASK_TOKEN_ID)
    assert expanded.input_ids == [11, MASK_TOKEN_ID, MASK_TOKEN_ID, MASK_TOKEN_ID,
                                  22, MASK_TOKEN_ID, MASK_TOKEN_ID, MASK_TOKEN_ID]
    assert expanded.positions == [9, 10, 11, 12, 22, 23, 24, 25]
    assert expanded.query_start_loc == [0, 4, 8]
    assert expanded.seq_lens == [13, 26]
    assert expanded.token_indices_to_sample == [1, 2, 3, 5, 6, 7]
    assert expanded.context_positions == [7, 8, 9, 10, 20, 21]
    assert expanded.num_context == 6


def test_query_positions_subtract_rejected_rows():
    """反证：漏减被拒行 → query 从 11 而不是 9 开始（草稿整体错位 2 行）。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    case = layout_case(device)
    rows = fake_rows(case)
    # 把请求 0 的被拒数强制改成 0（模拟"忘了减"）
    wrong_rows = [replace(rows[0], num_rejected=0), rows[1]]
    expanded = expand_dflash_inputs(
        [int(value) for value in case.target_positions.tolist()],
        wrong_rows, num_query_per_req=case.num_query_per_req, mask_token_id=MASK_TOKEN_ID)
    assert expanded.positions[:4] == [11, 12, 13, 14]      # 应然是 [9, 10, 11, 12]
    assert expanded.seq_lens[0] == 15                      # 应然是 13


def test_slot_mapping_marks_out_of_range_positions():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    case = layout_case(device)
    positions = torch.tensor([7, 9, 63, 64], dtype=torch.int64, device=device)
    req = torch.tensor([0, 0, 0, 0], dtype=torch.int64, device=device)
    slots = compute_slot_mapping_for_rows(
        positions, req, case.block_table, case.block_size, case.max_model_len)
    # 64 是越界位置（max_model_len=64）→ 哨兵；其余按 block*4 + pos%4 查表
    assert slots.tolist() == [7, 9, 63, -1]


def test_mask_token_must_be_rejected_for_bad_cases():
    """空 mask token / 0 个 query 行都当场报错（不静默跑）。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    case = layout_case(device)
    rows = fake_rows(case)
    with pytest.raises(ValueError, match="num_query_per_req"):
        expand_dflash_inputs([7, 8], rows, num_query_per_req=0,
                             mask_token_id=MASK_TOKEN_ID)


# ---------------- 配置边界 ----------------

def _dflash_spec(**overrides):
    _, draft = model_configs()
    kwargs = dict(method="dflash", num_speculative_tokens=3, draft_model_config=draft)
    kwargs.update(overrides)
    return SpeculativeConfig(**kwargs)


def test_config_forces_parallel_drafting_and_slot_budget():
    spec = _dflash_spec()
    assert spec.parallel_drafting is True                      # 上游 config 同款强制
    assert spec.max_num_new_slots_for_drafting == 3            # 1 个 bonus + K 个 mask，均在 target 之外
    assert spec.use_eagle() and spec.uses_dflash()
    assert spec.eagle_aux_hidden_state_layers() == (0, 1)
    assert spec.eagle3_use_aux_hidden_state() is True


def test_config_rejects_sample_from_anchor():
    _, draft = model_configs()
    hf = dict(draft.hf_config)
    hf["dflash_config"] = {**hf["dflash_config"], "sample_from_anchor": True}
    with pytest.raises(NotImplementedError, match="sample_from_anchor"):
        _dflash_spec(draft_model_config=ModelConfig(
            model=draft.model, dtype="float32", max_model_len=64, hf_config=hf))


def test_config_rejects_mixed_sliding_window():
    _, draft = model_configs()
    hf = dict(draft.hf_config)
    hf["dflash_config"] = {**hf["dflash_config"], "use_swa": True}
    with pytest.raises(NotImplementedError, match="单 KV group"):
        _dflash_spec(draft_model_config=ModelConfig(
            model=draft.model, dtype="float32", max_model_len=64, hf_config=hf))


def test_config_rejects_missing_mask_token():
    _, draft = model_configs()
    hf = dict(draft.hf_config)
    hf["dflash_config"] = {k: v for k, v in hf["dflash_config"].items()
                           if k != "mask_token_id"}
    with pytest.raises(ValueError, match="mask token"):
        _dflash_spec(draft_model_config=ModelConfig(
            model=draft.model, dtype="float32", max_model_len=64, hf_config=hf))


def test_config_requires_a_draft_directory():
    with pytest.raises(ValueError, match="draft_model_config"):
        SpeculativeConfig(method="dflash", num_speculative_tokens=3)


def test_aux_layers_fall_back_to_dflash_target_layer_ids():
    """没有 `eagle_aux_hidden_state_layer_ids` 时用 `dflash_config.target_layer_ids + 1`。"""
    _, draft = model_configs()
    hf = dict(draft.hf_config)
    hf.pop("eagle_aux_hidden_state_layer_ids")
    hf["dflash_config"] = {**hf["dflash_config"], "target_layer_ids": [2, 5]}
    spec = _dflash_spec(draft_model_config=ModelConfig(
        model=draft.model, dtype="float32", max_model_len=64, hf_config=hf))
    assert spec.eagle_aux_hidden_state_layers() == (3, 6)
