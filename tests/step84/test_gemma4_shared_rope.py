"""84-A：真实 Gemma4 target/assistant 配置下的层间共享审计。

不加载权重，只按配置建 35 + 4 个 RoPE 层；表上限改成 256 以保持测试轻量，
但层类型、head_dim、base、partial factor 与真实 config 一致。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from minivllm.models.gemma4_mtp import (_layer_geometry, _layer_types,
                                       _make_rotary_embedding, _text_config)

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "models" / "gemma-4-E2B-it" / "config.json"
ASSISTANT = ROOT / "models" / "gemma-4-E2B-it-assistant" / "config.json"

pytestmark = pytest.mark.skipif(
    not (TARGET.exists() and ASSISTANT.exists()),
    reason="本机没有真实 Gemma4 config（权重仍不需要加载）")


def _load_rope_layers(path: Path, max_position: int = 256):
    config = copy.deepcopy(json.loads(path.read_text()))
    text_config = _text_config(config)
    text_config["max_position_embeddings"] = max_position
    types = _layer_types(text_config, int(text_config["num_hidden_layers"]))
    layers = []
    for layer_idx, layer_type in enumerate(types):
        resolved_type, head_dim, _ = _layer_geometry(text_config, layer_idx)
        assert resolved_type == layer_type
        layers.append(_make_rotary_embedding(
            text_config, layer_idx, head_dim, layer_type,
            is_sliding=layer_type == "sliding_attention"))
    return text_config, layers


def _unique(modules):
    by_ptr = {}
    for module in modules:
        ptr = module.cos_sin_cache.untyped_storage().data_ptr()
        by_ptr.setdefault(ptr, []).append(module)
    return by_ptr


def test_real_config_35_plus_4_layers_share_two_tables():
    target_cfg, target_layers = _load_rope_layers(TARGET)
    assistant_cfg, assistant_layers = _load_rope_layers(ASSISTANT)
    assert len(target_layers) == 35
    assert len(assistant_layers) == 4

    all_layers = target_layers + assistant_layers
    unique = _unique(all_layers)
    assert len(unique) == 2

    sliding = [m for m in all_layers if m.rope_type == "default"]
    full = [m for m in all_layers if m.rope_type == "proportional"]
    assert len(sliding) == 28 + 3
    assert len(full) == 7 + 1
    assert len(_unique(sliding)) == 1
    assert len(_unique(full)) == 1
    assert {m.head_size for m in sliding} == {256}
    assert {m.head_size for m in full} == {512}
    assert {m.rotary_dim for m in sliding} == {256}
    assert {m.rotary_dim for m in full} == {512}
    assert {m.declared_rotary_dim for m in full} == {128}

    max_position = int(target_cfg["max_position_embeddings"])
    # 紧凑 + 共享后的唯一字节由布局算出来，不写死 5.25 GiB。
    expected_bytes = (
        max_position * 256 * torch.empty((), dtype=torch.float32).element_size()
        + max_position * 512 * torch.empty((), dtype=torch.float32).element_size()
    )
    actual_bytes = sum(
        modules[0].cos_sin_cache.numel() * modules[0].cos_sin_cache.element_size()
        for modules in unique.values())
    assert actual_bytes == expected_bytes

    # 旧“每层一份”的名义字节与唯一字节的差就是本关省下的表。
    nominal_bytes = sum(
        m.cos_sin_cache.numel() * m.cos_sin_cache.element_size()
        for m in all_layers)
    assert nominal_bytes == (
        (28 + 3) * max_position * 256 * 4
        + (7 + 1) * max_position * 512 * 4)
    assert nominal_bytes > actual_bytes

    # 同一 key 的 target/assistant 层必须是同一 storage。
    assert target_layers[0].cos_sin_cache.data_ptr() == (
        assistant_layers[0].cos_sin_cache.data_ptr())
    assert target_layers[4].cos_sin_cache.data_ptr() == (
        assistant_layers[3].cos_sin_cache.data_ptr())


def test_bf16_conversion_still_has_two_storages_for_39_layers():
    _, target_layers = _load_rope_layers(TARGET)
    _, assistant_layers = _load_rope_layers(ASSISTANT)
    all_layers = target_layers + assistant_layers
    for module in all_layers:
        module.to(torch.bfloat16)
    unique = _unique(all_layers)
    assert len(unique) == 2
    assert all(m.cos_sin_cache.dtype == torch.bfloat16 for m in all_layers)
    max_position = 256
    expected_bytes = max_position * (256 + 512) * 2
    actual_bytes = sum(
        modules[0].cos_sin_cache.numel() * modules[0].cos_sin_cache.element_size()
        for modules in unique.values())
    assert actual_bytes == expected_bytes
