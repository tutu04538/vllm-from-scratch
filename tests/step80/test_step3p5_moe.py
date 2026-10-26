"""80F 前置：Step3p5 MoE 的路由/共享专家/加权语义（tiny 教学路径）。

真实 196B target 的 fused expert 权重不在本机；这里不声称 kernel 级对齐，只把
router→top-k→专家加权→shared expert 的数学关系钉住。
"""

from __future__ import annotations

import torch

from minivllm.models.step3p5 import Step3p5DecoderLayer, Step3p5MoE

CFG = {
    "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
    "num_attention_heads": 2, "num_attention_groups": 1, "head_dim": 8,
    "max_position_embeddings": 32, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
    "layer_types": ["sliding_attention"], "sliding_window": 4,
    "partial_rotary_factors": [1.0], "use_rope_layers": [],
    "yarn_only_types": [], "attention_other_setting": None,
    "use_moe": True, "moe_layers_enum": "0", "moe_num_experts": 4,
    "moe_top_k": 1, "moe_intermediate_size": 8, "share_expert_dim": 8,
    "moe_router_activation": "sigmoid", "norm_expert_weight": True,
    "moe_router_scaling_factor": 1.0, "swiglu_limits": [0], "swiglu_limits_shared": [0],
    "vocab_size": 32, "num_nextn_predict_layers": 0,
}


def test_moe_router_top1_and_shared_expert():
    moe = Step3p5MoE(CFG, prefix="model.layers.0.moe", layer_idx=0)
    # 所有专家/共享专家清零；只让 expert 0 产生已知输出。
    for p in moe.parameters():
        p.data.zero_()
    # router：expert 0 的 logits 最大（其余为负）。
    with torch.no_grad():
        moe.gate.weight[0] = 1.0
        moe.gate.weight[1:] = -1.0
        # expert 0：gate=1, up=1 -> silu(1)=0.731...；down 取前 8 维恒等。
        expert = moe.experts[0]
        expert.gate_up_proj.weight[:8] = 1.0 / 16.0
        expert.gate_up_proj.weight[8:] = 1.0 / 16.0
        expert.down_proj.weight.zero_()
        expert.down_proj.weight[:8, :8] = torch.eye(8)
        # 共享专家清零，确认 routed 输出不被它污染。
    hidden = torch.ones(3, 16)
    out = moe(hidden)
    expected = torch.full((3, 8), torch.nn.functional.silu(torch.tensor(1.0)).item())
    torch.testing.assert_close(out[:, :8], expected, rtol=1e-6, atol=1e-6)
    assert torch.count_nonzero(out[:, 8:]) == 0


def test_decoder_layer_selects_moe_by_layer_enum():
    layer = Step3p5DecoderLayer(CFG, prefix="model.layers.0")
    assert layer.use_moe is True
    assert isinstance(layer.mlp, Step3p5MoE)
    cfg_dense = {**CFG, "use_moe": False}
    dense = Step3p5DecoderLayer(cfg_dense, prefix="model.layers.0")
    assert dense.use_moe is False
    assert not isinstance(dense.mlp, Step3p5MoE)


def test_moe_top2_normalization():
    cfg = {**CFG, "moe_top_k": 2}
    moe = Step3p5MoE(cfg, prefix="model.layers.0.moe", layer_idx=0)
    with torch.no_grad():
        for p in moe.parameters():
            p.data.zero_()
        # 两个专家的输出分别恒为 1 和 2，router 选 0/1。
        for idx, value in ((0, 1.0), (1, 2.0)):
            expert = moe.experts[idx]
            expert.gate_up_proj.weight[:8] = 1.0 / 16.0
            expert.gate_up_proj.weight[8:] = 1.0 / 16.0
            expert.down_proj.weight.zero_()
            expert.down_proj.weight[0, 0] = value / torch.nn.functional.silu(
                torch.tensor(1.0)).item()
        moe.gate.weight[0] = 1.0
        moe.gate.weight[1] = 1.0
        moe.gate.weight[2:] = -10.0
    hidden = torch.ones(1, 16)
    out = moe(hidden)
    # sigmoid(1) 两个相等 -> 归一化后各 0.5，输出 (1+2)/2 = 1.5 的第 0 维。
    assert out[0, 0].item() == pytest.approx(1.5, rel=1e-5)


import pytest
