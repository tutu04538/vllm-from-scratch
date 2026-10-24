"""需求 080D：Q-only 共享 KV 的映射与只读契约。

本文件不加载真实 Gemma4，也不造一个"能服务的 tiny Gemma4"；它只钉住四件事：

1. 映射：target `layer_types` / `num_kv_shared_layers` + draft `layer_types` →
   "同类型最后一个**非共享源层**"，并与冻结上游方法逐例差分。
2. 只读：Q-only 层的 `forward()` 不写 target 的 KV cache；执行前后逐字节一致。
3. 所有权：Runner 只绑 target 的 cache 对象，不分配第二份、不释放源 cache；
   未接线的 Q-only 层明确报错，不静默退回独立 cache。
4. 位置固定：Q-only 的同轮多个草稿步不推进 positions/seq_lens（Triton 内核实跑）。

混合 SWA/full 的真实执行仍不在本关：这里只验证"选源"和"只读"，不把映射测试
当成 81 关的多 group/backend 已完成。
"""

from __future__ import annotations

import gc
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from minivllm.attention.layer import Attention
from minivllm.attention.metadata import AttentionMetadata
from minivllm.config import CUDAGraphMode
from minivllm.forward_context import BatchDescriptor, set_forward_context
from minivllm.worker.gpu.model_runner import GPUModelRunner
from minivllm.worker.gpu.spec_decode.gemma4.kv_sharing import (
    DEFAULT_TARGET_LAYER_PREFIX,
    select_gemma4_kv_sharing_targets,
    setup_gemma4_kv_sharing,
    target_layer_prefix,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Q-only 位置固定的 Triton 内核需要 CUDA"
)

S = "sliding_attention"
F = "full_attention"
TARGET_NAME = "model.layers.2.self_attn.attn"
DRAFT_NAME = "draft.layers.0.self_attn.attn"


# ---------------------------------------------------------------------------
# 假对象：只提供冻结映射函数真正读取的结构接口
# ---------------------------------------------------------------------------


def _fake_draft_model(num_layers: int):
    attns = []
    layers = []
    for _ in range(num_layers):
        attn = SimpleNamespace(q_only=False, kv_sharing_target_layer_name=None)
        attns.append(attn)
        layers.append(SimpleNamespace(self_attn=SimpleNamespace(attn=attn)))
    return SimpleNamespace(model=SimpleNamespace(layers=layers)), attns


def _text_config(layer_types, num_kv_shared_layers=None):
    config = SimpleNamespace(layer_types=list(layer_types))
    if num_kv_shared_layers is not None:
        config.num_kv_shared_layers = num_kv_shared_layers
    return config


def _upstream_mapping(target_types, num_shared, draft_types, prefix):
    """调用已安装的冻结 vLLM 参考方法，返回 {draft_idx: target_layer_name}。"""
    from vllm.v1.worker.gpu.spec_decode.gemma4.speculator import Gemma4Speculator

    target_text = _text_config(target_types, num_shared)
    draft_text = _text_config(draft_types)
    target_hf = SimpleNamespace(get_text_config=lambda: target_text)
    draft_hf = SimpleNamespace(get_text_config=lambda: draft_text)
    fake_self = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=draft_hf)),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(hf_config=target_hf)),
    )
    model, attns = _fake_draft_model(len(draft_types))
    names = {
        f"{prefix}.{idx}.self_attn.attn"
        for idx in range(len(target_types))
    }
    Gemma4Speculator._setup_gemma4_kv_sharing(fake_self, model, names)
    return {
        idx: attn.kv_sharing_target_layer_name
        for idx, attn in enumerate(attns)
    }


# ---------------------------------------------------------------------------
# 1. 映射：固定例子 + 冻结方法差分
# ---------------------------------------------------------------------------


def test_mapping_fixed_example_skips_shared_tail_layer():
    """`[S,F,S,F,S]` + 最后 1 层共享：draft `[S,F]` 应映射到 2、3，而不是 4。"""
    plan = select_gemma4_kv_sharing_targets(
        target_layer_types=[S, F, S, F, S],
        target_num_kv_shared_layers=1,
        draft_layer_types=[S, F],
        num_draft_layers=2,
    )
    assert [(item.target_idx, item.target_layer_name) for item in plan] == [
        (2, "model.layers.2.self_attn.attn"),
        (3, "model.layers.3.self_attn.attn"),
    ]
    assert all(item.reason is None for item in plan)


def test_mapping_writes_q_only_and_target_name_on_model():
    model, attns = _fake_draft_model(2)
    mapped = setup_gemma4_kv_sharing(
        model,
        {"model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn"},
        _text_config([S, F]),
        _text_config([S, F, S, F, S], num_kv_shared_layers=1),
    )
    assert [item.target_idx for item in mapped] == [2, 3]
    assert all(attn.q_only is True for attn in attns)
    assert [attn.kv_sharing_target_layer_name for attn in attns] == [
        "model.layers.2.self_attn.attn",
        "model.layers.3.self_attn.attn",
    ]


def test_mapping_missing_type_and_short_draft_types():
    """缺同类型源层时保留 None；draft 类型列表短时按 full_attention 补。"""
    plan = select_gemma4_kv_sharing_targets(
        target_layer_types=[F],
        target_num_kv_shared_layers=0,
        draft_layer_types=[S],
        num_draft_layers=2,
    )
    assert plan[0].target_idx is None
    assert "没有类型" in plan[0].reason
    assert plan[1].draft_layer_type == F
    assert plan[1].target_idx == 0

    # 模型侧也要把"未接线"标出来：q_only=True + 源名 None，Runner 才能拒绝它，
    # 而不是把它当成普通草稿层去分配一份独立 KV。
    model, attns = _fake_draft_model(1)
    setup_gemma4_kv_sharing(
        model, {"model.layers.0.self_attn.attn"},
        _text_config([S]), _text_config([F]))
    assert attns[0].q_only is True
    assert attns[0].kv_sharing_target_layer_name is None


def test_mapping_preserves_real_target_layer_prefix():
    plan = select_gemma4_kv_sharing_targets(
        target_layer_types=[S, F],
        target_num_kv_shared_layers=0,
        draft_layer_types=[F],
        num_draft_layers=1,
        target_prefix=target_layer_prefix(
            {"language_model.model.layers.0.self_attn.attn"}),
    )
    assert plan[0].target_layer_name == (
        "language_model.model.layers.1.self_attn.attn")


def test_target_layer_prefix_defaults_to_model_layers():
    assert target_layer_prefix(set()) == DEFAULT_TARGET_LAYER_PREFIX


def test_mapping_rejects_invalid_shared_layer_count():
    with pytest.raises(ValueError, match="不能为负"):
        select_gemma4_kv_sharing_targets([S], -1, [S], 1)
    with pytest.raises(ValueError, match="超过 target 层数"):
        select_gemma4_kv_sharing_targets([S, F], 3, [S], 1)


@pytest.mark.parametrize(
    ("target_types", "num_shared", "draft_types", "prefix"),
    [
        ([S, F, S, F, S], 1, [S, F], "model.layers"),
        ([S, F], 0, [S], "model.layers"),
        ([F], 0, [S, F], "model.layers"),
        ([S, F, S], 1, [S, F], "language_model.model.layers"),
        ([S, F], 2, [S, F], "model.layers"),
    ],
)
def test_mapping_matches_frozen_upstream_method(
        target_types, num_shared, draft_types, prefix):
    """逐例对照冻结 `Gemma4Speculator._setup_gemma4_kv_sharing`。"""
    want = _upstream_mapping(target_types, num_shared, draft_types, prefix)

    model, _ = _fake_draft_model(len(draft_types))
    mapped = setup_gemma4_kv_sharing(
        model,
        {f"{prefix}.{idx}.self_attn.attn"
         for idx in range(len(target_types))},
        _text_config(draft_types),
        _text_config(target_types, num_shared),
    )
    got = {item.draft_idx: item.target_layer_name for item in mapped}
    assert got == want


# ---------------------------------------------------------------------------
# 2. 只读：Q-only 前向与独立数学参考
# ---------------------------------------------------------------------------


def _attention_case(*, mode: CUDAGraphMode):
    torch.manual_seed(0)
    heads, kv_heads, head_size = 4, 2, 4
    block_size, num_blocks, seq_len, q_len = 2, 8, 5, 2
    cache = torch.randn(
        2, num_blocks, block_size, kv_heads, head_size, dtype=torch.float32)
    target = Attention(heads, head_size, num_kv_heads=kv_heads,
                       layer_name=TARGET_NAME)
    target.kv_cache = cache
    draft = Attention(heads, head_size, num_kv_heads=kv_heads,
                      layer_name=DRAFT_NAME)
    draft.bind_shared_kv_cache(target, cache, draft_dtype=torch.float32)

    block_table = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    meta = AttentionMetadata(
        query_start_loc=torch.tensor([0, q_len], dtype=torch.int32),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        block_table=block_table,
        slot_mapping=torch.arange(q_len, dtype=torch.int64),
        block_size=block_size,
        num_reqs=1,
        uniform_query_len=q_len if mode == CUDAGraphMode.FULL else None,
    )
    query = torch.randn(q_len, heads * head_size, dtype=torch.float32)
    return cache, target, draft, meta, query


def _manual_attention(query, cache, block_table, seq_len, *, heads, kv_heads,
                      head_size, block_size, q_len):
    flat = cache.view(2, -1, kv_heads, head_size)
    positions = torch.arange(seq_len)
    slots = block_table[0, positions // block_size] * block_size + positions % block_size
    keys = flat[0][slots].transpose(0, 1).float()
    values = flat[1][slots].transpose(0, 1).float()
    if heads != kv_heads:
        keys = keys.repeat_interleave(heads // kv_heads, dim=0)
        values = values.repeat_interleave(heads // kv_heads, dim=0)
    q = query.view(q_len, heads, head_size).transpose(0, 1).float()
    scores = torch.matmul(q, keys.transpose(-1, -2)) / (head_size ** 0.5)
    q_pos = torch.arange(seq_len - q_len, seq_len)
    key_pos = torch.arange(seq_len)
    scores = scores.masked_fill(
        key_pos[None, :] > q_pos[:, None], torch.finfo(torch.float32).min)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, values)
    return out.transpose(0, 1).reshape(q_len, heads * head_size)


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
def test_q_only_forward_is_read_only_and_matches_math(mode):
    cache, _target, draft, meta, query = _attention_case(mode=mode)
    before = cache.clone()
    want = _manual_attention(
        query, cache, meta.block_table, int(meta.seq_lens[0]),
        heads=4, kv_heads=2, head_size=4, block_size=2, q_len=2)

    with set_forward_context(
        {DRAFT_NAME: meta},
        num_tokens=query.shape[0],
        cudagraph_runtime_mode=mode,
        batch_descriptor=BatchDescriptor(num_tokens=2, num_reqs=1),
    ):
        got = draft.forward(query, torch.empty(0), torch.empty(0))

    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
    assert torch.equal(cache, before), "Q-only 前向写入了源 KV cache"
    assert draft.kv_cache is cache


def test_q_only_allows_different_q_head_count():
    """GQA 下 draft 的 Q 头数可以不同；只要求 KV heads/head_size 一致。"""
    cache = torch.zeros(2, 8, 2, 2, 4, dtype=torch.float32)
    target = Attention(4, 4, num_kv_heads=2, layer_name=TARGET_NAME)
    target.kv_cache = cache
    draft = Attention(8, 4, num_kv_heads=2, layer_name=DRAFT_NAME)
    draft.bind_shared_kv_cache(target, cache, draft_dtype=torch.float32)
    assert draft.q_only is True
    assert draft.kv_cache is cache


def test_q_only_rejects_kv_head_mismatch():
    cache = torch.zeros(2, 8, 2, 2, 4, dtype=torch.float32)
    target = Attention(4, 4, num_kv_heads=2, layer_name=TARGET_NAME)
    target.kv_cache = cache
    draft = Attention(4, 4, num_kv_heads=4, layer_name=DRAFT_NAME)
    with pytest.raises(ValueError, match="KV heads"):
        draft.bind_shared_kv_cache(target, cache, draft_dtype=torch.float32)


def test_q_only_rejects_dtype_mismatch():
    cache = torch.zeros(2, 8, 2, 2, 4, dtype=torch.float32)
    target = Attention(4, 4, num_kv_heads=2, layer_name=TARGET_NAME)
    target.kv_cache = cache
    draft = Attention(4, 4, num_kv_heads=2, layer_name=DRAFT_NAME)
    with pytest.raises(ValueError, match="dtype"):
        draft.bind_shared_kv_cache(
            target, cache, draft_dtype=torch.float16)


def test_q_only_rejects_head_size_mismatch():
    cache = torch.zeros(2, 8, 2, 2, 4, dtype=torch.float32)
    target = Attention(4, 4, num_kv_heads=2, layer_name=TARGET_NAME)
    target.kv_cache = cache
    draft = Attention(4, 8, num_kv_heads=2, layer_name=DRAFT_NAME)
    with pytest.raises(ValueError, match="head_size"):
        draft.bind_shared_kv_cache(target, cache, draft_dtype=torch.float32)


def test_q_only_rejects_non_contiguous_or_wrong_layout_cache():
    base = torch.zeros(2, 8, 2, 2, 4, dtype=torch.float32)
    target = Attention(4, 4, num_kv_heads=2, layer_name=TARGET_NAME)
    draft = Attention(4, 4, num_kv_heads=2, layer_name=DRAFT_NAME)

    non_contiguous = torch.zeros(2, 8, 2, 2, 8, dtype=torch.float32)[..., ::2]
    assert non_contiguous.shape == (2, 8, 2, 2, 4)
    assert not non_contiguous.is_contiguous()
    target.kv_cache = non_contiguous
    with pytest.raises(ValueError, match="连续"):
        draft.bind_shared_kv_cache(
            target, non_contiguous, draft_dtype=torch.float32)

    wrong_layout = base.transpose(0, 1)
    target.kv_cache = wrong_layout
    with pytest.raises(ValueError, match="布局"):
        draft.bind_shared_kv_cache(
            target, wrong_layout, draft_dtype=torch.float32)


def test_q_only_rejects_chained_source():
    cache = torch.zeros(2, 8, 2, 2, 4, dtype=torch.float32)
    source = Attention(4, 4, num_kv_heads=2, layer_name="model.layers.2.self_attn.attn")
    source.kv_cache = cache
    middle = Attention(4, 4, num_kv_heads=2, layer_name="draft.layers.0.self_attn.attn")
    middle.bind_shared_kv_cache(source, cache, draft_dtype=torch.float32)
    leaf = Attention(4, 4, num_kv_heads=2, layer_name="draft.layers.1.self_attn.attn")
    with pytest.raises(ValueError, match="自己也是 Q-only"):
        leaf.bind_shared_kv_cache(middle, cache, draft_dtype=torch.float32)


def test_gemma4_config_is_routed_away_from_generic_mtp():
    from minivllm import ModelConfig, SpeculativeConfig
    from minivllm.worker.gpu.spec_decode import init_speculator

    spec = SpeculativeConfig(
        method="mtp",
        num_speculative_tokens=1,
        draft_model_config=ModelConfig(
            model="/tmp/gemma4-assistant",
            hf_config={"model_type": "gemma4_mtp", "num_nextn_predict_layers": 1},
        ),
    )
    assert spec.use_gemma4_mtp() is True
    with pytest.raises(NotImplementedError, match="80E"):
        init_speculator(
            SimpleNamespace(speculative_config=spec), torch.device("cpu"))



def test_q_only_backend_without_forward_q_only_is_rejected():
    cache = torch.zeros(2, 8, 2, 2, 4, dtype=torch.float32)
    target = Attention(4, 4, num_kv_heads=2, layer_name=TARGET_NAME)
    target.kv_cache = cache
    draft = Attention(4, 4, num_kv_heads=2, layer_name=DRAFT_NAME)
    draft.bind_shared_kv_cache(target, cache, draft_dtype=torch.float32)
    draft.impl = object()
    meta = AttentionMetadata(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        slot_mapping=torch.zeros(1, dtype=torch.int64),
        block_size=2, num_reqs=1)
    with set_forward_context({DRAFT_NAME: meta}, num_tokens=1):
        with pytest.raises(NotImplementedError, match="Q-only"):
            draft.forward(torch.zeros(1, 16), torch.empty(0), torch.empty(0))


# ---------------------------------------------------------------------------
# 3. 所有权：Runner 只绑 cache，不分配/不释放
# ---------------------------------------------------------------------------


def _runner_case(*, draft_kv_heads=2, mapping=TARGET_NAME):
    block_size, num_blocks, kv_heads, head_size = 4, 8, 2, 4
    cache = torch.randn(2, num_blocks, block_size, kv_heads, head_size)
    target = Attention(4, head_size, num_kv_heads=kv_heads,
                       layer_name=TARGET_NAME)
    target.kv_cache = cache

    class _Draft(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))
            self.attn = Attention(4, head_size, num_kv_heads=draft_kv_heads,
                                  layer_name=DRAFT_NAME)
            self.attn.q_only = True
            self.attn.kv_sharing_target_layer_name = mapping

    class _Runner:
        device = torch.device("cpu")

        def __init__(self):
            self.block_size = block_size
            self.kv_caches = {TARGET_NAME: cache}
            self.speculator = SimpleNamespace(model=_Draft())
            self._target_layers = {TARGET_NAME: target}

        def _attention_layers(self):
            return self._target_layers

    return _Runner(), target, cache


def test_runner_binds_target_cache_without_allocation():
    runner, target, cache = _runner_case()
    before = cache.clone()
    with mock.patch("minivllm.worker.gpu.model_runner.torch.zeros",
                    side_effect=AssertionError("不应分配第二份草稿 KV")):
        GPUModelRunner._allocate_draft_kv_caches(
            runner, SimpleNamespace(num_gpu_blocks=8))

    draft_attn = runner.speculator.model.attn
    assert draft_attn.kv_cache is cache
    assert draft_attn.q_only is True
    assert draft_attn.kv_sharing_target_layer_name == TARGET_NAME
    assert runner.speculator.draft_attn_layer_names == {"attn"}
    assert runner.kv_caches == {TARGET_NAME: cache}
    assert target.kv_cache is cache
    assert torch.equal(cache, before)


def test_runner_rejects_unwired_q_only_layer():
    runner, _target, _cache = _runner_case(mapping=None)
    with pytest.raises(NotImplementedError, match="Q-only"):
        GPUModelRunner._allocate_draft_kv_caches(
            runner, SimpleNamespace(num_gpu_blocks=8))


def test_runner_rejects_kv_head_mismatch():
    runner, _target, _cache = _runner_case(draft_kv_heads=4)
    with pytest.raises(ValueError, match="KV heads"):
        GPUModelRunner._allocate_draft_kv_caches(
            runner, SimpleNamespace(num_gpu_blocks=8))


def test_rebinding_after_slot_reuse_keeps_source_cache():
    """草稿层重建/重新绑定时，源 cache 对象与内容都不变（没有 free/二次分配）。"""
    runner, target, cache = _runner_case()
    replacement = _runner_case()[0].speculator.model
    before = cache.clone()
    with mock.patch("minivllm.worker.gpu.model_runner.torch.zeros",
                    side_effect=AssertionError("不应分配第二份草稿 KV")):
        GPUModelRunner._allocate_draft_kv_caches(
            runner, SimpleNamespace(num_gpu_blocks=8))
        old_model = runner.speculator.model
        runner.speculator.model = replacement
        GPUModelRunner._allocate_draft_kv_caches(
            runner, SimpleNamespace(num_gpu_blocks=8))

    assert runner.speculator.model.attn.kv_cache is cache
    assert target.kv_cache is cache
    assert torch.equal(cache, before)
    del old_model
    gc.collect()
    assert target.kv_cache is cache
    assert torch.equal(cache, before)


# ---------------------------------------------------------------------------
# 4. 位置固定：Q-only 的草稿步不推进 positions/seq_lens
# ---------------------------------------------------------------------------


@requires_cuda
def test_q_only_update_draft_inputs_keeps_positions_and_seq_lens():
    from minivllm.worker.gpu.input_batch import InputBuffers
    from minivllm.worker.gpu.spec_decode.autoregressive.speculator import (
        update_draft_inputs,
    )

    device = torch.device("cuda")
    buffers = InputBuffers(max_num_reqs=2, max_num_tokens=8, device=device)
    buffers.positions[:2] = torch.tensor([7, 11], dtype=torch.int64, device=device)
    buffers.seq_lens[:2] = torch.tensor([9, 13], dtype=torch.int32, device=device)
    draft_tokens = torch.tensor([101, 202], dtype=torch.int32, device=device)
    current_step = torch.tensor(1, dtype=torch.int64, device=device)
    hidden = torch.randn(2, 4, device=device)
    output_tokens = torch.zeros(2, 3, dtype=torch.int32, device=device)
    next_hidden = torch.zeros(2, 4, device=device)

    update_draft_inputs(
        draft_tokens, current_step, hidden, output_tokens, next_hidden,
        buffers, num_reqs=2, max_model_len=64, num_speculative_steps=3,
        advance_draft_positions=False,
    )

    assert torch.equal(
        buffers.positions[:2], torch.tensor([7, 11], dtype=torch.int64, device=device))
    assert torch.equal(
        buffers.seq_lens[:2], torch.tensor([9, 13], dtype=torch.int32, device=device))
    assert torch.equal(
        buffers.input_ids[:2], torch.tensor([101, 202], dtype=torch.int32, device=device))
    assert torch.equal(output_tokens[:, 1], draft_tokens)
    assert torch.equal(next_hidden, hidden)
