"""需求 85：Gemma4 target 的 packed QKV、共享层与 k_eq_v 路由。"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers import Gemma4AssistantConfig, Gemma4ForCausalLM, Gemma4TextConfig

from minivllm.models.gemma4 import Gemma4Attention
from minivllm.models.gemma4_mtp import Gemma4MTP

VOCAB = 64
HIDDEN = 16
INTERMEDIATE = 32
NUM_HEADS = 2
NUM_KV_HEADS = 1


def _text_config(*, k_eq_v: bool = False, num_shared: int = 0,
                 num_layers: int = 2) -> Gemma4TextConfig:
    return Gemma4TextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=num_layers,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        num_global_key_value_heads=NUM_KV_HEADS,
        head_dim=8,
        global_head_dim=16,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        layer_types=(
            ["sliding_attention", "full_attention", "full_attention"][:num_layers]
            if num_layers == 3 else
            ["sliding_attention", "full_attention"][:num_layers]),
        num_kv_shared_layers=num_shared,
        sliding_window=4,
        use_double_wide_mlp=False,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=0,
        attention_k_eq_v=k_eq_v,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {
                "rope_type": "proportional", "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.5,
            },
        },
    )


def _ours(config: Gemma4TextConfig):
    from minivllm.models.gemma4 import Gemma4ForCausalLM as Ours
    return Ours(config).eval()


def _hf(config: Gemma4TextConfig):
    return Gemma4ForCausalLM(config).eval()


def _state(model) -> dict[str, torch.Tensor]:
    return {name: tensor for name, tensor in model.state_dict().items()
            if name != "lm_head.weight"}


def test_packed_loader_routes_qkv_and_loads_all_shards():
    config = _text_config()
    hf = _hf(config)
    ours = _ours(config)
    state = _state(hf)
    ours.load_weights(iter(state.items()))

    for idx, layer in enumerate(ours.model.layers):
        attn = layer.self_attn
        src = hf.model.layers[idx].self_attn
        qkv = attn.qkv_proj.weight
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=0)
        torch.testing.assert_close(q, src.q_proj.weight, rtol=0, atol=0)
        torch.testing.assert_close(k, src.k_proj.weight, rtol=0, atol=0)
        torch.testing.assert_close(v, src.v_proj.weight, rtol=0, atol=0)
        torch.testing.assert_close(attn.q_norm.weight, src.q_norm.weight,
                                   rtol=0, atol=0)
        torch.testing.assert_close(attn.k_norm.weight, src.k_norm.weight,
                                   rtol=0, atol=0)


def test_shared_layers_use_qkv_proj_but_skip_kv_processing():
    # HF 文本栈的共享层只提供 q_proj/q_norm；qkv 的 K/V 行仍存在但 forward 不用。
    config = _text_config(num_layers=3, num_shared=1)
    hf = _hf(config)
    ours = _ours(config)
    ours.load_weights(iter(_state(hf).items()))

    shared = ours.model.layers[2].self_attn
    assert shared.is_kv_shared_layer
    assert hasattr(shared, "qkv_proj")
    assert shared.k_norm is None and shared.v_norm is None
    qkv = shared.qkv_proj.weight
    q, k, v = qkv.split([shared.q_size, shared.kv_size, shared.kv_size], dim=0)
    torch.testing.assert_close(
        q, hf.model.layers[2].self_attn.q_proj.weight, rtol=0, atol=0)
    # K/V 段存在但没有被初始化/使用；不能把“结构存在”误写成“已加载”。
    assert k.shape == (shared.kv_size, HIDDEN)
    assert v.shape == (shared.kv_size, HIDDEN)


def test_attention_k_eq_v_routes_k_to_both_slots():
    config = _text_config(k_eq_v=True)
    hf = _hf(config)
    ours = _ours(config)
    ours.load_weights(iter(_state(hf).items()))

    full = ours.model.layers[1].self_attn
    assert full.layer_type == "full_attention"
    q, k, v = full.qkv_proj.weight.split(
        [full.q_size, full.kv_size, full.kv_size], dim=0)
    torch.testing.assert_close(k, v, rtol=0, atol=0)
    torch.testing.assert_close(k, hf.model.layers[1].self_attn.k_proj.weight,
                               rtol=0, atol=0)


def test_missing_shard_and_duplicate_override_are_rejected():
    config = _text_config(num_layers=1)
    hf = _hf(config)
    ours = _ours(config)
    state = _state(hf)
    state.pop("model.layers.0.self_attn.k_proj.weight")
    with pytest.raises(ValueError, match=r"缺少 \['k'\]"):
        ours.load_weights(iter(state.items()))

    config = _text_config(k_eq_v=True, num_layers=2)
    hf = _hf(config)
    ours = _ours(config)
    state = _state(hf)
    # 人为给 full 层补一个 v_proj：k_eq_v 路由已经复制了 K，再加载 V 就是重复覆盖。
    # 必须紧跟 k_proj 插入，保持 AutoWeightsLoader 的前缀分组连续。
    patched: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        patched[name] = tensor
        if name == "model.layers.1.self_attn.k_proj.weight":
            patched["model.layers.1.self_attn.v_proj.weight"] = torch.zeros(
                16, HIDDEN)
    with pytest.raises(ValueError, match="重复覆盖"):
        ours.load_weights(iter(patched.items()))


class _CaptureAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = None
        self.k = None
        self.v = None

    def forward(self, q, k, v):
        self.q, self.k, self.v = q.detach().clone(), k.detach().clone(), v.detach().clone()
        return torch.zeros_like(q)


def test_k_eq_v_forward_uses_raw_k_for_v_norm_not_rotated_k():
    """V 必须来自 K 的**原始投影**，走无权重 v_norm；不能直接用 k_norm/RoPE 后的 K。"""
    config = _text_config(k_eq_v=True)
    hf = _hf(config)
    attn = Gemma4Attention(config, 1, prefix="model.layers.1.self_attn")
    src = hf.model.layers[1].self_attn
    with torch.no_grad():
        attn.qkv_proj.weight[:attn.q_size].copy_(src.q_proj.weight)
        attn.qkv_proj.weight[attn.q_size:attn.q_size + attn.kv_size].copy_(
            src.k_proj.weight)
        attn.qkv_proj.weight[attn.q_size + attn.kv_size:].copy_(src.k_proj.weight)
        attn.q_norm.weight.copy_(src.q_norm.weight)
        attn.k_norm.weight.copy_(src.k_norm.weight)
        attn.o_proj.weight.zero_()

    capture = _CaptureAttention()
    attn.attn = capture
    hidden = torch.randn(3, HIDDEN, generator=torch.Generator().manual_seed(301))
    with torch.no_grad():
        attn(torch.arange(3), hidden)

    raw = nn.functional.linear(hidden, src.k_proj.weight)
    raw = raw.view(3, NUM_KV_HEADS, 16)
    expected_v = raw.float() * torch.rsqrt(
        raw.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    expected_v = expected_v.to(hidden.dtype).view(3, 16)
    torch.testing.assert_close(capture.v, expected_v, rtol=0, atol=0)

    # 反例：如果实现的是 `v = k`（k_norm + RoPE 后），数值应明显不同。
    assert not torch.allclose(capture.v, capture.k)


def test_assistant_remains_q_only_without_qkv_proj():
    text = Gemma4TextConfig(
        vocab_size=VOCAB, hidden_size=8, intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, global_head_dim=16, max_position_embeddings=32,
        rms_norm_eps=1e-6, tie_word_embeddings=True,
        layer_types=["sliding_attention", "full_attention"],
        num_kv_shared_layers=2, sliding_window=4,
        hidden_size_per_layer_input=0, vocab_size_per_layer_input=0,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {
                "rope_type": "proportional", "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.5},
        },
    )
    config = Gemma4AssistantConfig(
        text_config=text, backbone_hidden_size=16,
        use_ordered_embeddings=False, tie_word_embeddings=True,
        architectures=["Gemma4MTPModel"], model_type="gemma4_assistant")
    model = Gemma4MTP(config)
    for layer in model.model.layers:
        assert not hasattr(layer.self_attn, "qkv_proj")
        assert layer.self_attn.attn.q_only is True
