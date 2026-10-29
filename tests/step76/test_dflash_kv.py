"""76 关：DFlash 的**上下文 KV 与端到端执行**。

这一关钉四件事（需求 076 §4 后三条）：

    K1  上下文 KV 的融合实现与"逐层各算各的"参考实现**逐值相等**（含 K-norm / RoPE / 槽位）；
        query 前向与冻结参考实现逐值相等，且**因果性真的起作用**（两种模式的输出不同）。
    K2  被拒尾部不进下一步的草稿缓存：每轮写进去的 context 槽位数 = target 行数 − 被拒数，
        哨兵槽位的**位置**恰好是被拒的那几行。
    K3  V1 与 V2（同一份 tiny 权重、贪心）逐 token 相同；V2 的图与 eager 逐 token 相同。
    K4  边界：缩小词表（d2t）、`midlayer` 历史命名、`mask_hidden` 权重拒绝、无 aux 特征路径。

参考实现是 `minivllm/testing/torch_dflash.py`（复用模型自己的参数，只换"逐层显式"的写法）。
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from spec76_helpers import (  # noqa: E402
    DEVICE,
    bind_caches,
    make_v1_engine,
    make_v2_engine,
    model_configs,
    requires_cuda,
    run_prompts,
    tiny_dflash_model,
)

from minivllm.attention.backends.torch_sdpa import TorchAttentionBackend  # noqa: E402
from minivllm.forward_context import set_forward_context  # noqa: E402
from minivllm.testing.torch_dflash import reference_context_kv, reference_forward  # noqa: E402

VOCAB = 11
NUM_CTX = 5
NUM_QUERY = 4
BLOCK_SIZE = 4


def _context_case(device=DEVICE):
    torch.manual_seed(76)
    model, hf = tiny_dflash_model()
    context = torch.randn(NUM_CTX, hf["hidden_size"], device=device)
    # 位置与槽位必须与块表一致：块表 arange(8)、block_size 4 ⇒ 位置 p 落在槽位 p
    context_positions = torch.arange(0, NUM_CTX, device=device)
    query_ids = torch.tensor([3, hf["dflash_config"]["mask_token_id"],
                              hf["dflash_config"]["mask_token_id"], 5], device=device)
    query_positions = torch.arange(NUM_CTX, NUM_CTX + NUM_QUERY, device=device)
    return model, hf, context, context_positions, query_ids, query_positions


# ---------------- K1：融合实现 vs 冻结参考 ----------------

@pytest.mark.parametrize("causal", [False, True])
def test_query_forward_matches_the_frozen_reference(causal):
    model, hf, context, context_positions, query_ids, query_positions = _context_case()
    names = model.get_draft_kv_cache_layer_names()
    slots = torch.arange(NUM_CTX, device=context_positions.device)
    model.precompute_and_store_context_kv(context, context_positions, slots)
    builder = TorchAttentionBackend.get_builder_cls()(BLOCK_SIZE)
    metadata = builder.build(
        query_start_loc=torch.tensor([0, NUM_QUERY], device=DEVICE),
        seq_lens=torch.tensor([NUM_CTX + NUM_QUERY], device=DEVICE),
        block_table=torch.arange(8, device=DEVICE).view(1, -1),
        slot_mapping=torch.arange(NUM_CTX, NUM_CTX + NUM_QUERY, device=DEVICE),
        num_reqs=1, causal=causal)
    with set_forward_context({name: metadata for name in names}, num_tokens=NUM_QUERY):
        got = model(query_ids, query_positions)
    want = reference_forward(model, context, context_positions, query_ids, query_positions,
                             causal=causal)
    max_diff = float((got - want).abs().max())
    assert max_diff < 1e-6, f"causal={causal} 时与参考实现差 {max_diff}"


def test_causality_actually_changes_the_output():
    """两种模式必须产生不同输出——否则本次测试只是"两边都没跑对"。"""
    model, hf, context, context_positions, query_ids, query_positions = _context_case()
    both = []
    for causal in (False, True):
        names = model.get_draft_kv_cache_layer_names()
        slots = torch.arange(NUM_CTX, device=context_positions.device)
        model.precompute_and_store_context_kv(context, context_positions, slots)
        builder = TorchAttentionBackend.get_builder_cls()(BLOCK_SIZE)
        metadata = builder.build(
            query_start_loc=torch.tensor([0, NUM_QUERY], device=DEVICE),
            seq_lens=torch.tensor([NUM_CTX + NUM_QUERY], device=DEVICE),
            block_table=torch.arange(8, device=DEVICE).view(1, -1),
            slot_mapping=torch.arange(NUM_CTX, NUM_CTX + NUM_QUERY, device=DEVICE),
            num_reqs=1, causal=causal)
        with set_forward_context({name: metadata for name in names}, num_tokens=NUM_QUERY):
            both.append(model(query_ids, query_positions))
    assert not torch.equal(both[0], both[1])


def test_fused_context_kv_matches_per_layer_reference():
    """融合路径写进缓存的 K/V == 逐层参考实现（同一槽位逐值相等）。"""
    model, hf, context, context_positions, query_ids, query_positions = _context_case()
    names = model.get_draft_kv_cache_layer_names()
    slots = torch.arange(NUM_CTX, device=context_positions.device)
    model.precompute_and_store_context_kv(context, context_positions, slots)
    reference = reference_context_kv(model, context, context_positions)
    modules = dict(model.named_modules())
    for layer_index, name in enumerate(names):
        cache = modules[name].kv_cache                       # [2, blocks, bs, nkv, hd]
        module = modules[name]
        got_k = cache[0].reshape(-1, module.num_kv_heads, module.head_size)[slots]
        got_v = cache[1].reshape(-1, module.num_kv_heads, module.head_size)[slots]
        want_k, want_v = reference[layer_index]
        assert torch.allclose(got_k, want_k, atol=1e-6), f"layer {layer_index} 的 K 不等"
        assert torch.allclose(got_v, want_v, atol=1e-6), f"layer {layer_index} 的 V 不等"


# ---------------- K2：被拒尾部不进下一步的草稿缓存 ----------------

@requires_cuda
def test_rejected_tail_is_not_written_to_the_draft_cache():
    engine, core, runner = make_v2_engine(budget=64, max_num_seqs=2)
    speculator = runner.speculator
    recorded: list[tuple[np.ndarray, np.ndarray, torch.Tensor]] = []

    # 记录每一轮的 (num_scheduled_tokens, num_rejected, context_slot_mapping)
    from minivllm.worker.gpu.spec_decode.dflash import speculator as module

    real_prepare = module.prepare_dflash_inputs

    def spy_prepare(*args, **kwargs):
        num_rejected = args[11]
        input_batch = args[9]
        result = real_prepare(*args, **kwargs)
        slots = speculator._context_slot_mapping.clone()
        recorded.append((input_batch.num_scheduled_tokens.copy(),
                         num_rejected.detach().cpu().numpy().copy(), slots))
        return result

    module.prepare_dflash_inputs = spy_prepare
    try:
        run_prompts(engine, [("a", [1, 2, 3, 4]), ("b", [5, 6])], max_tokens=8)
    finally:
        module.prepare_dflash_inputs = real_prepare

    assert recorded, "没有记录到任何一轮 propose"
    saw_rejection = False
    for scheduled, rejected, slots in recorded:
        num_rows = int(scheduled.sum())
        assert int((slots[:num_rows] == -1).sum()) == int(rejected.sum())
        # 哨兵槽位出现在**每请求的最后 rejected 行**上（不是别的位置）
        cursor = 0
        for length, count in zip(scheduled, rejected):
            row_slots = slots[cursor:cursor + length]
            if count:
                saw_rejection = True
                assert (row_slots[length - count:] == -1).all()
                assert (row_slots[:length - count] != -1).all()
            cursor += length
    assert saw_rejection, "随机 tiny 权重下 8 个 token 应该有被拒的草稿；没有就说明测试没覆盖"


# ---------------- K3：V1/V2 与图/eager 一致 ----------------

@requires_cuda
def test_v1_and_v2_produce_the_same_greedy_tokens():
    prompts = [("a", [1, 2, 3, 4]), ("b", [5, 6])]
    v1_engine, _, _ = make_v1_engine()
    v1 = run_prompts(v1_engine, prompts, max_tokens=8)
    v2_engine, _, _ = make_v2_engine()
    v2 = run_prompts(v2_engine, prompts, max_tokens=8)
    assert v1 == v2, f"V1 {v1} vs V2 {v2}"


@requires_cuda
def test_v2_graph_and_eager_agree():
    prompts = [("a", [1, 2, 3, 4])]
    eager_engine, _, eager_runner = make_v2_engine(mode="none")
    eager = run_prompts(eager_engine, prompts, max_tokens=8)
    graph_engine, _, graph_runner = make_v2_engine(
        capture_sizes=[8, 16], mode="full_decode_only")
    graph = run_prompts(graph_engine, prompts, max_tokens=8, shutdown=False)
    try:
        assert eager == graph
        stats = graph_runner.cudagraph_capture_stats or {}
        assert stats.get("captured", 0) >= 1, stats
        selections = graph_runner.speculator.query_cudagraph_manager.selections
        assert selections, "图管理器没有记录任何一次分派"
        assert any(entry["desc_num_tokens"] > entry["num_tokens"] for entry in selections), \
            "补齐到图档位这条路径没有被触发（测试没有覆盖到 padding 重放）"
    finally:
        graph_engine.shutdown()


# ---------------- K4：边界 ----------------

def test_reduced_vocab_scatter_and_no_aux_path():
    model, hf = tiny_dflash_model(draft_vocab_size=6)
    device = next(model.parameters()).device
    hidden = torch.randn(3, hf["hidden_size"], device=device)
    logits = model.compute_logits(hidden)
    assert logits.shape == (3, hf["vocab_size"])
    assert torch.isinf(logits[:, 6:]).all() and (logits[:, 6:] < 0).all()
    assert logits[:, :6].isfinite().all()

    plain_model, _ = tiny_dflash_model(use_aux_hidden_state=False)
    hidden_states = torch.randn(2, 5, device=next(plain_model.parameters()).device)
    # 无 aux 时 `combine_hidden_states` 原样返回（不经过 fc）
    assert torch.equal(plain_model.combine_hidden_states(hidden_states), hidden_states)
    assert plain_model.model.fc is None


def test_midlayer_naming_loads():
    model, hf = tiny_dflash_model(naming="midlayer", num_layers=1)
    assert model.model.layers[0].self_attn.qkv_proj.weight.isfinite().all()
    assert hf["dflash_config"]["causal"] is False


def test_mask_hidden_weight_is_rejected():
    model, hf = tiny_dflash_model()
    with pytest.raises(ValueError, match="mask_hidden"):
        model.load_weights(iter([("mask_hidden", torch.zeros(1, hf["hidden_size"]))]))


def test_combine_hidden_states_rejects_wrong_width():
    model, hf = tiny_dflash_model()
    with pytest.raises(ValueError, match="aux"):
        model.combine_hidden_states(torch.randn(2, hf["hidden_size"] + 1))
