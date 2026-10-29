"""80F：真实 Step3.5 MTP 子集（约 5.9 GB）的加载/前向。

权重不入库；用 `benchmarks/download_step3p5_mtp.py` 只取 MTP 子集后，本文件才跑。
这里**不加载 196B target**，只验证 Step3p5MTP 的真实 3 层结构、shared_head、qkv 打包
和一次前向；完整 target 端到端仍需更大显存，不在本机声明。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from minivllm import ModelConfig
from minivllm.model_loader import get_model

ROOT = Path(__file__).resolve().parents[2]
MTP_DIR = ROOT / "models" / "Step-3.5-Flash-MTP"

pytestmark = pytest.mark.skipif(
    not (MTP_DIR / "model.safetensors").is_file()
    or not (MTP_DIR / "config.json").is_file(),
    reason="真实 Step3.5 MTP 子集未下载：运行 "
           "`python benchmarks/download_step3p5_mtp.py --output models/Step-3.5-Flash-MTP`；"
           "待验，不是通过")


def _config():
    config = json.loads((MTP_DIR / "config.json").read_text())
    config = {**config, "architectures": ["Step3p5MTP"],
              "model_type": "step3p5_mtp", "max_position_embeddings": 512}
    return config


def test_real_mtp_weight_shapes_and_forward():
    config = _config()
    model = get_model(
        ModelConfig(model=str(MTP_DIR), dtype="bfloat16", max_model_len=64,
                    hf_config=config), "cpu")
    layers = model.model.layers
    assert list(layers.keys()) == ["45", "46", "47"]
    layer = layers["45"]
    # 真实配置的 sliding 层用 96 个 Q heads、8 个 KV heads、head_dim 128
    assert layer.mtp_block.self_attn.num_heads == 96
    assert layer.mtp_block.self_attn.num_kv_heads == 8
    assert layer.mtp_block.self_attn.head_dim == 128
    assert tuple(layer.mtp_block.self_attn.qkv_proj.weight.shape) == (14336, 4096)
    assert tuple(layer.shared_head.head.weight.shape) == (128896, 4096)
    assert tuple(model.model.embed_tokens.weight.shape) == (128896, 4096)

    # 只验证模型层/权重加载；真实 KV 绑定由 V1/V2 端到端覆盖。
    class Identity(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, *args, **kwargs):
            return args[0]

    for module in model.modules():
        if hasattr(module, "attn") and type(module).__name__.endswith("Attention"):
            module.attn = Identity()
    model.eval()
    with torch.no_grad():
        hidden = model(
            torch.tensor([1, 2]),
            torch.tensor([0, 1]),
            torch.zeros(2, 4096, dtype=torch.bfloat16),
            spec_step_idx=0,
        )
        logits = model.compute_logits(hidden, spec_step_idx=0)
    assert hidden.shape == (2, 4096)
    assert logits.shape == (2, 128896)
    assert torch.isfinite(hidden).all()
    assert torch.isfinite(logits).all()
