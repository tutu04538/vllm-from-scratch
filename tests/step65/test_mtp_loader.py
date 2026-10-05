"""step65：MTP 的**权重加载**（需求 065 §3.1/§4）。

MTP 的权重不是另一个模型目录，而是**排在 target 最后一层后面**的同一份 checkpoint。所以这一关的
加载要回答四个问题，每个都有用例：

1. **两派命名都认**：`mtp.*`（上游 qwen3_next_mtp.py）与 `model.layers.{N+i}.*`（上游 deepseek_mtp.py）；
2. **层号/名字没有搞混**：MTP 的 `layers.0.*` 必须来自 **spec 层**，不能是 target 的第 0 层；
3. **target 侧要跳过 spec 权重**（同一个文件里两套权重并存）；
4. **不支持的家族要明确报错**（DeepSeek 的 MLA/MoE 参数名在这里没有对应模块）。
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from minivllm.models.qwen3 import Qwen3ForCausalLM
from minivllm.models.qwen3_mtp import Qwen3MTP
from minivllm.models.utils import get_spec_layer_idx_from_weight_name, skip_spec_layer_weights
from minivllm.testing.tiny_models import tiny_mtp_dir

TARGET_LAYERS = 2          # tiny_gqa 的 num_hidden_layers
NEXTN = 1


def load_state(naming: str) -> tuple[dict, dict]:
    directory = tiny_mtp_dir("tiny_gqa", naming=naming)
    config = json.loads((Path(directory) / "config.json").read_text())
    state = load_file(str(Path(directory) / "model.safetensors"))
    return config, state


def spec_name(naming: str, suffix: str) -> str:
    """按命名约定给出 spec 层某个权重的检查点名。"""
    if naming == "mtp":
        return f"mtp.{suffix}" if suffix.startswith(("fc", "norm", "pre_fc_norm")) \
            else f"mtp.layers.0.{suffix}"
    absolute = TARGET_LAYERS
    if suffix.startswith(("fc", "norm", "pre_fc_norm")):
        return f"model.layers.{absolute}.{suffix}"
    return f"model.layers.{absolute}.{suffix}"


@pytest.mark.parametrize("naming", ["mtp", "absolute"])
def test_loads_both_naming_conventions(naming):
    """两派命名都能加载，且**每个参数都被填上**（AutoWeightsLoader 的覆盖检查）。"""
    config, state = load_state(naming)
    config = {**config, "architectures": ["Qwen3MTPModel"]}
    model = Qwen3MTP(config)
    loaded = model.load_weights(iter(state.items()))

    expected = {name for name, _ in model.named_parameters()}
    assert loaded == expected, f"未加载：{sorted(expected - loaded)}"
    # 加载数量：MTP 自己的参数 + 共享的词表两边
    assert len(loaded) == len(expected)


@pytest.mark.parametrize("naming", ["mtp", "absolute"])
def test_spec_layer_weights_come_from_the_spec_layer(naming):
    """MTP 的 `layers.0.*` 与胶水来自 **spec 层**，不是 target 的第 0 层（层号没搞混）。"""
    config, state = load_state(naming)
    model = Qwen3MTP({**config, "architectures": ["Qwen3MTPModel"]})
    model.load_weights(iter(state.items()))

    pairs = [
        ("model.fc.weight", spec_name(naming, "fc.weight")),
        ("model.pre_fc_norm_hidden.weight", spec_name(naming, "pre_fc_norm_hidden.weight")),
        ("model.norm.weight", spec_name(naming, "norm.weight")),
        ("model.layers.0.mlp.down_proj.weight",
         spec_name(naming, "mlp.down_proj.weight")),
    ]
    params = dict(model.named_parameters())
    for ours, checkpoint_name in pairs:
        assert torch.equal(params[ours], state[checkpoint_name]), \
            f"{ours} 不等于检查点里的 {checkpoint_name}"

    # 反证：MTP 的层权重**不能**等于 target 第 0 层的同名权重（搞混层号就会相等）
    target_layer0 = state[f"model.layers.0.mlp.down_proj.weight"]
    assert not torch.equal(params["model.layers.0.mlp.down_proj.weight"], target_layer0), \
        "spec 层与 target 第 0 层的权重被搞混了"


@pytest.mark.parametrize("naming", ["mtp", "absolute"])
def test_qkv_and_gate_up_packing_applied(naming):
    """打包映射同样生效：q/k/v → qkv_proj、gate/up → gate_up_proj（顺序不能错）。"""
    config, state = load_state(naming)
    model = Qwen3MTP({**config, "architectures": ["Qwen3MTPModel"]})
    model.load_weights(iter(state.items()))
    params = dict(model.named_parameters())

    q = state[spec_name(naming, "self_attn.q_proj.weight")]
    k = state[spec_name(naming, "self_attn.k_proj.weight")]
    v = state[spec_name(naming, "self_attn.v_proj.weight")]
    qkv = params["model.layers.0.self_attn.qkv_proj.weight"]
    assert torch.equal(qkv[:q.shape[0]], q)
    assert torch.equal(qkv[q.shape[0]:q.shape[0] + k.shape[0]], k)
    assert torch.equal(qkv[q.shape[0] + k.shape[0]:], v)

    gate = state[spec_name(naming, "mlp.gate_proj.weight")]
    up = state[spec_name(naming, "mlp.up_proj.weight")]
    gate_up = params["model.layers.0.mlp.gate_up_proj.weight"]
    assert torch.equal(gate_up[:gate.shape[0]], gate)
    assert torch.equal(gate_up[gate.shape[0]:], up)


@pytest.mark.parametrize("naming", ["mtp", "absolute"])
def test_shared_embedding_and_lm_head_loaded(naming):
    """共享的词表两边（`embed_tokens`/`lm_head`）按检查点加载（上游 shared_weight_names）。"""
    config, state = load_state(naming)
    model = Qwen3MTP({**config, "architectures": ["Qwen3MTPModel"]})
    model.load_weights(iter(state.items()))
    params = dict(model.named_parameters())
    assert torch.equal(params["model.embed_tokens.weight"], state["model.embed_tokens.weight"])
    assert torch.equal(params["lm_head.weight"], state["lm_head.weight"])
    # 两者是**各自**的参数（不是同一个对象）：上游 Qwen3NextMTP 也是这样各自建层、各自加载
    assert params["model.embed_tokens.weight"] is not params["lm_head.weight"]


@pytest.mark.parametrize("naming", ["mtp", "absolute"])
def test_target_model_skips_spec_weights(naming):
    """**target 侧**加载同一份 checkpoint：spec 权重被跳过，target 的层权重照常。

    这是"MTP 与 target 同文件"的另一半：上游 `qwen3_next.py:846` 用
    `skip_prefixes=["mtp."]`、`deepseek_v2.py:1575` 用 `get_spec_layer_idx_from_weight_name()`。
    """
    config, state = load_state(naming)
    target = Qwen3ForCausalLM(config)
    loaded = target.load_weights(iter(state.items()))
    params = dict(target.named_parameters())

    assert {name for name, _ in target.named_parameters()} == loaded
    assert not any("mtp." in name or f"layers.{TARGET_LAYERS}." in name for name in loaded), \
        "spec 层的名字不该出现在 target 的加载结果里"
    # target 的第 0 层来自 target 自己的权重（没被 spec 层覆盖）
    assert torch.equal(params["model.layers.0.mlp.down_proj.weight"],
                       state["model.layers.0.mlp.down_proj.weight"])


def test_get_spec_layer_idx_matches_upstream():
    """名字识别函数与上游 `model_executor/models/utils.py:496` 逐名一致（含层号前缀的坑）。"""
    from vllm.model_executor.models.utils import (
        get_spec_layer_idx_from_weight_name as upstream_fn)

    our_config = {"num_hidden_layers": 61, "num_nextn_predict_layers": 2}
    upstream_config = SimpleNamespace(num_hidden_layers=61, num_nextn_predict_layers=2)
    names = [
        "model.layers.61.eh_proj.weight", "model.layers.62.shared_head.head.weight",
        "layers.61.self_attn.q_proj.weight", "model.layers.6.self_attn.q_proj.weight",
        "model.layers.60.mlp.down_proj.weight", "model.embed_tokens.weight",
        "model.layers.610.weight", "mtp.fc.weight",
    ]
    for name in names:
        assert get_spec_layer_idx_from_weight_name(our_config, name) == \
            upstream_fn(upstream_config, name), f"{name} 的归属判定与上游不一致"


def test_skip_spec_layer_weights_drops_only_spec():
    """`skip_spec_layer_weights()` 只丢 spec 层（上游 deepseek_v2.py:1575 的两行）。"""
    config = {"num_hidden_layers": 2, "num_nextn_predict_layers": 1}
    weights = [("model.layers.0.a", 1), ("model.layers.2.b", 2), ("lm_head.weight", 3)]
    kept = [name for name, _ in skip_spec_layer_weights(config, iter(weights))]
    assert kept == ["model.layers.0.a", "lm_head.weight"]


def test_unsupported_family_params_fail_loudly():
    """DeepSeek 那一族的参数名（MLA/MoE）在这里没有对应模块 → **明确报错**，不静默跳过。"""
    config, state = load_state("absolute")
    model = Qwen3MTP(config)
    fake = dict(state)
    fake[f"model.layers.{TARGET_LAYERS}.self_attn.q_a_proj.weight"] = torch.zeros(4, 4)
    with pytest.raises(ValueError, match="q_a_proj"):
        model.load_weights(iter(fake.items()))


def test_missing_spec_weights_fail_loudly():
    """检查点里缺 spec 层（只有 target 的层）→ 覆盖检查报"这些参数没被加载"。"""
    config, state = load_state("mtp")
    model = Qwen3MTP(config)
    only_target = {name: value for name, value in state.items()
                   if not name.startswith("mtp.")}
    with pytest.raises(ValueError, match="没有被加载"):
        model.load_weights(iter(only_target.items()))
