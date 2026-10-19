"""step78：DFlash2 模型侧的四层差分（forward / selector / logits / verifier 的前三层）。

- **forward**：解码器层的 norm/卷积/注意力占位/MLP/残差顺序与显式公式逐值相等；注意力本身
  沿用 DFlash（76 关已与上游差分过），这里用同一个无参 stub 把它排除在变量之外。
- **selector**：`CandidateSelector` 与上游真实模块、与逐元素参考实现都对照（见
  `test_selector_walk.py` 的 A 段；这里补"码本/投影的形状与 top-k 来源"）。
- **logits**：`compute_candidates()` 与参考 top-k 公式逐值相等（含 `output_multiplier` /
  `final_logit_softcapping` 的顺序）。
- **verifier**：不动它——V2 的验证器与 75 关完全同一份代码。这里只证明"交给它的 q 是
  fp32、只在候选列有值、softmax 合法"，端到端无损由 `test_dflash2_e2e.py` 的 greedy 对照盯。
"""

import pytest
import torch
import torch.nn.functional as F

from dflash2_helpers import DEVICE, make_speculator, model_configs
from minivllm.model_loader import get_model
from minivllm.testing.torch_dflash2 import (reference_candidates,
                                            reference_decoder_layer)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DFlash2 的提议者只在 CUDA 上跑（Triton 内核），模型层测试跟随")


def _tiny_model(**kwargs):
    _, draft_config = model_configs(**kwargs)
    model = get_model(draft_config, "cpu")
    model.eval()
    return model, draft_config.hf_config


def test_forward_matches_the_explicit_reference(monkeypatch):
    """两层解码器（含残差流）与显式公式逐值相等；query 行跨两个请求块。"""
    model, hf = _tiny_model(k=3, final_logit_softcapping=5.0, output_multiplier=1.7)
    layer = model.model.layers[0]
    num_tokens = 2 * (1 + 3)  # 两个请求 × (1+K) 行：验证分组卷积不跨块

    def stub_attention(positions, hidden_states):
        return hidden_states * 0.5 + 0.1

    monkeypatch.setattr(layer.self_attn, "forward", stub_attention)
    generator = torch.Generator().manual_seed(11)
    hidden = torch.randn(num_tokens, hf["hidden_size"], generator=generator)
    positions = torch.arange(num_tokens)
    out, residual = layer(positions, hidden, None)
    ref_out, ref_residual = reference_decoder_layer(
        layer, lambda x, pos: stub_attention(pos, x), positions, hidden, None)
    assert torch.allclose(out, ref_out, atol=1e-6), (out - ref_out).abs().max()
    assert torch.allclose(residual, ref_residual, atol=1e-6)

    # 第二层的输入接第一层的残差（Python 里 residual 是"加过、还没归一化"的那份）
    layer1 = model.model.layers[1]
    monkeypatch.setattr(layer1.self_attn, "forward", stub_attention)
    out1, residual1 = layer1(positions, out, residual)
    ref_out1, ref_residual1 = reference_decoder_layer(
        layer1, lambda x, pos: stub_attention(pos, x), positions, out, residual)
    assert torch.allclose(out1, ref_out1, atol=1e-6)
    assert torch.allclose(residual1, ref_residual1, atol=1e-6)


def test_embedding_scale_is_applied_once():
    """`input_embedding_scale` 只乘在输入嵌入上（不是乘 hidden、也不是乘 logits）。"""
    model, hf = _tiny_model(k=3, input_embedding_scale=1.7)
    assert model.model.input_embedding_scale == pytest.approx(1.7)
    ids = torch.tensor([0, 3, 7, 10])
    ours = model.embed_input_ids(ids)
    expected = F.embedding(ids, model.model.embed_tokens.weight) * 1.7
    assert torch.allclose(ours, expected, atol=1e-6)


def test_compute_candidates_matches_the_reference_formula():
    """`compute_candidates()`：top-k 的 id 与 value 都和参考公式一致（fp32 数值路径）。"""
    model, hf = _tiny_model(k=3, final_logit_softcapping=5.0, output_multiplier=1.7)
    generator = torch.Generator().manual_seed(12)
    hidden = torch.randn(7, hf["hidden_size"], generator=generator)
    ids, values = model.compute_candidates(hidden)
    ref_ids, ref_values = reference_candidates(
        hidden, model.lm_head.weight, model.model.candidate_selector.top_k,
        scale=1.7, soft_cap=5.0)
    assert torch.equal(ids, ref_ids)
    assert torch.allclose(values, ref_values, atol=1e-6)
    assert values.dtype == torch.float32
    # soft cap 把值压进 (-5, 5)；且收尾顺序是"先乘 output_multiplier、再 tanh"
    assert values.abs().max().item() < 5.0
    _, raw_values = reference_candidates(
        hidden, model.lm_head.weight, model.model.candidate_selector.top_k)
    assert torch.allclose(values, torch.tanh(raw_values * 1.7 / 5.0) * 5.0, atol=1e-6)


def test_candidate_top_k_comes_from_the_checkpoint():
    """候选图宽度来自 `dflash_config.selector_top_k`，不是采样器的 top_k。"""
    for top_k in (1, 4):
        model, hf = _tiny_model(k=3, selector_top_k=top_k)
        assert model.model.candidate_selector.top_k == top_k
        assert model.get_candidate_top_k() == top_k
        ids, values = model.compute_candidates(torch.randn(5, hf["hidden_size"]))
        assert ids.shape == (5, top_k)
        assert values.shape == (5, top_k)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_block_size_follows_k_and_the_model_has_no_per_k_embedding(k):
    """不同 K：分组卷积的 `block_size = 1+K`，模型**没有**任何按 K 扩展的 embedding。

    需求 078 §5 要求"不同 K 相对 checkpoint 训练槽数量的行为按实际模型检查，不自行扩展
    缺失 embedding"：这里确认 ① 模型参数集合与 K 无关；② 卷积窗口跟着 K 走；③ 用 `1+K`
    行能跑通一层前向。
    """
    model, hf = _tiny_model(k=k)
    layer = model.model.layers[0]
    assert layer.attention_conv.block_size == 1 + k
    assert layer.mlp_conv.block_size == 1 + k
    # 参数名里没有任何随 K 变化的东西（模型构造只读了 K 来设 block_size）
    parameter_names = {name for name, _ in model.named_parameters()}
    for other_k in (1, 2, 5):
        other, _ = _tiny_model(k=other_k)
        assert {name for name, _ in other.named_parameters()} == parameter_names

    # `1+K` 行的一层前向（attention 用 stub，避免搭 KV 元数据）
    with torch.no_grad():
        def stub_attention(positions, hidden_states):
            return hidden_states

        layer.self_attn.forward = stub_attention
        try:
            out, _ = layer(torch.arange(1 + k), torch.randn(1 + k, hf["hidden_size"]),
                           None)
        finally:
            # 删掉实例属性，恢复类上的 `forward`（把 bound method 写回实例会把绑定关系丢掉）
            del layer.self_attn.forward
    assert out.shape == (1 + k, hf["hidden_size"])


@requires_cuda
def test_draft_q_is_fp32_and_a_valid_distribution():
    """交给验证器的 q：fp32、只在候选列有值、每行 softmax 和为 1。"""
    speculator, _ = make_speculator(k=3, draft_sample_method="probabilistic")
    assert speculator.draft_logits is not None
    assert speculator.draft_logits.dtype == torch.float32
    assert torch.all(speculator.draft_logits == float("-inf"))
    candidate_ids = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]], device=DEVICE)
    from dflash2_helpers import set_walk_inputs

    set_walk_inputs(speculator, anchors=[4], temperature=1.0, seeds=0, slot_ids=[0])
    scores = torch.randn(1, 3, 3, 3, device=DEVICE)
    speculator._sample_path(candidate_ids, scores, 1)
    speculator._cache_draft_logits(candidate_ids, 3)
    q = speculator.draft_logits[0]
    for step in range(3):
        finite = torch.isfinite(q[step])
        assert int(finite.sum()) == 3
        assert set(torch.nonzero(finite).flatten().tolist()) == \
            set(candidate_ids[0, step].tolist())
        assert torch.softmax(q[step], dim=-1).sum().item() == pytest.approx(1.0, abs=1e-6)
