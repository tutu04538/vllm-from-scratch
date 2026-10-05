"""step63：EAGLE3 模型适配（需求 063 §2/§3.2）。

覆盖：
  A. 注册名（上游 registry 的两个名字）
  B. `combine_hidden_states` 的数学（= 拼接 + fc 投影，不许用平均替代）
  C. 第一层的输入拼装（`cat([input_layernorm(embeds), hidden_norm(hidden)])`，qkv 输入宽度 2*hidden）
  D. 真实 checkpoint 加载：参数全覆盖、d2t/t2d 收到、`embed_tokens` 缺 → 标记与 target 共享
  E. draft 词表 id → target 词表 id 的映射（`compute_logits()` 里 scatter 回 target 宽度）

真实权重在 `models/Qwen3-1.7B-eagle3`（本机、不入库）；没有它时相关用例**跳过并写明原因**
（不是"通过"）。
"""

import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minivllm.models import get_model_class  # noqa: E402
from minivllm.models.qwen3_eagle3 import Eagle3ForCausalLM  # noqa: E402

REAL_DRAFT_DIR = ROOT / "models" / "Qwen3-1.7B-eagle3"

TINY_CONFIG = {
    "architectures": ["Eagle3LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 8,
    "intermediate_size": 24,
    "vocab_size": 32,
    "draft_vocab_size": 32,
    "max_position_embeddings": 64,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "target_hidden_size": 16,
    "num_aux_layers": 3,
}


def tiny_model(**overrides) -> Eagle3ForCausalLM:
    config = dict(TINY_CONFIG)
    config.update(overrides)
    torch.manual_seed(0)
    return Eagle3ForCausalLM(config)


# ---------------------------------------------------------------------------
# A. 注册名
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["Eagle3Qwen3ForCausalLM", "Eagle3LlamaForCausalLM",
                                 "LlamaForCausalLMEagle3"])
def test_registry_names(name):
    assert get_model_class(name) is Eagle3ForCausalLM


# ---------------------------------------------------------------------------
# B. combine_hidden_states
# ---------------------------------------------------------------------------


def test_combine_hidden_states_is_concat_then_fc():
    """`fc` 权重设成已知矩阵时，输出必须等于 `cat(aux) @ fc.T`（不是平均、不是取一块）。"""
    model = tiny_model()
    with torch.no_grad():
        model.model.fc.weight.copy_(torch.eye(model.model.hidden_size).repeat(
            1, model.model.num_aux_layers))
    aux = torch.randn(4, model.model.hidden_size * model.model.num_aux_layers)
    out = model.model.combine_hidden_states(aux)
    expected = torch.nn.functional.linear(
        aux, model.model.fc.weight)
    assert torch.allclose(out, expected, atol=1e-6)
    # 反例：如果实现成"按块平均"，结果会与 cat 不同
    chunks = aux.chunk(model.model.num_aux_layers, dim=-1)
    assert not torch.allclose(out, torch.stack(chunks).mean(0), atol=1e-6)


def test_combine_hidden_states_is_noop_without_aux():
    model = tiny_model(use_aux_hidden_state=False)
    aux = torch.randn(2, model.model.hidden_size)
    assert torch.equal(model.model.combine_hidden_states(aux), aux)


def test_fc_input_size_is_target_hidden_times_aux_layers():
    model = tiny_model(target_hidden_size=16, num_aux_layers=3)
    assert model.model.fc.in_features == 48
    assert model.model.fc.out_features == model.model.hidden_size


# ---------------------------------------------------------------------------
# C. 第一层的输入拼装
# ---------------------------------------------------------------------------


class _RecordingAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []
        self.inputs = []

    def forward(self, positions, hidden_states):
        self.seen.append(tuple(hidden_states.shape))
        self.inputs.append(hidden_states.detach().clone())
        return torch.zeros_like(hidden_states[..., :TINY_CONFIG["hidden_size"]])


def test_layer0_concatenates_normed_embeds_with_normed_hidden():
    """layer0 的 attention 输入宽度 = 2*hidden（embeds 与 hidden 各一路），且两路都过了 RMSNorm。"""
    model = tiny_model()
    recorder = _RecordingAttention()
    layer = model.model.layers[0]
    layer.self_attn = recorder           # 只留"输入怎么拼"这一段
    embeds = torch.randn(3, TINY_CONFIG["hidden_size"])
    hidden = torch.randn(3, TINY_CONFIG["hidden_size"])
    layer(torch.arange(3), embeds, hidden, None)
    assert recorder.seen == [(3, 2 * TINY_CONFIG["hidden_size"])]
    # 拼进去的两路：input_layernorm(embeds) 与 hidden_norm(hidden)，顺序也是这个
    expected = torch.cat([layer.input_layernorm(embeds), layer.hidden_norm(hidden)], dim=-1)
    assert torch.allclose(recorder.inputs[0], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# D. 真实 checkpoint 加载
# ---------------------------------------------------------------------------


pytestmark_real = pytest.mark.skipif(
    not (REAL_DRAFT_DIR / "model.safetensors").is_file(),
    reason="本机没有 models/Qwen3-1.7B-eagle3（真实 EAGLE3 draft）：机制测试通过，模型集成待验")


@pytestmark_real
def test_real_checkpoint_loads_every_parameter():
    """真实 checkpoint：**每个参数都落位**（加载器对账不允许静默漏参），词表映射也收到。"""
    from safetensors.torch import load_file

    config = json.loads((REAL_DRAFT_DIR / "config.json").read_text())
    config["eagle_aux_hidden_state_layer_ids"] = [2, 14, 25]   # target 侧的默认辅助层
    config["target_hidden_size"] = 2048
    model = get_model_class(config["architectures"][0])(config)
    state = load_file(str(REAL_DRAFT_DIR / "model.safetensors"))
    model.load_weights(iter(state.items()))

    params = dict(model.named_parameters())
    assert set(params) == {
        "lm_head.weight", "model.embed_tokens.weight", "model.fc.weight",
        "model.layers.0.hidden_norm.weight", "model.layers.0.input_layernorm.weight",
        "model.layers.0.mlp.down_proj.weight", "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.self_attn.o_proj.weight", "model.layers.0.self_attn.qkv_proj.weight",
        "model.norm.weight"}
    # 检查点里的张量名 → 模型参数名的对应（逐条写出来，便于人工核对权重名映射）
    assert tuple(params["model.fc.weight"].shape) == (2048, 6144), "3 个辅助层 × 2048"
    assert tuple(params["model.layers.0.self_attn.qkv_proj.weight"].shape) == (4096, 4096)
    assert tuple(params["lm_head.weight"].shape) == (32000, 2048), "draft 词表 32000"
    assert model.draft_vocab_size == 32000 and model.target_vocab_size == 151936
    assert model.d2t is not None and model.t2d is not None
    assert model.model.tie_embeddings is True, "检查点没有 embed_tokens → 与 target 共享"
    # draft 那层用的是 Llama 风格（没有 q/k norm）
    assert model.model.layers[0].self_attn.q_norm is None


@pytestmark_real
def test_real_checkpoint_combine_and_logits_shapes():
    """真实权重的 combine + lm_head 形状走通（不涉及 attention 上下文）。"""
    from safetensors.torch import load_file

    config = json.loads((REAL_DRAFT_DIR / "config.json").read_text())
    config["eagle_aux_hidden_state_layer_ids"] = [2, 14, 25]
    config["target_hidden_size"] = 2048
    model = get_model_class(config["architectures"][0])(config)
    model.load_weights(iter(load_file(str(REAL_DRAFT_DIR / "model.safetensors")).items()))
    model.eval()
    with torch.no_grad():
        hidden = model.model.combine_hidden_states(torch.randn(3, 6144) * 0.05)
        logits = model.compute_logits(hidden)
    assert tuple(hidden.shape) == (3, 2048)
    # 66 关收尾（原 63 关待办）：`compute_logits()` 现在与上游同名方法一致——**已映射回 target
    # 词表宽度**（draft 宽度那份由 `draft_vocab_logits()` 提供，只给断言用）
    assert tuple(model.draft_vocab_logits(hidden).shape) == (3, 32000)
    assert tuple(logits.shape) == (3, 151936)
    assert int(torch.isfinite(logits[0]).sum()) == 32000        # 只有 d2t 覆盖的 32000 个位置有值
    assert torch.isfinite(hidden).all() and torch.isfinite(logits[:, :1]).all()


@pytestmark_real
def test_compute_logits_maps_draft_ids_through_d2t():
    """草稿 id 一定是 **target 空间**的 id，且等于 `draft_id + d2t[draft_id]`（d2t 是偏移量）。

    这是 66 关收尾的那条：提议者直接对 `compute_logits()` 的结果 `argmax`，所以映射必须发生在
    这个方法里面（上游 `llama_eagle3.py:339-356` 就是这么做的）。
    """
    from safetensors.torch import load_file

    config = json.loads((REAL_DRAFT_DIR / "config.json").read_text())
    config["eagle_aux_hidden_state_layer_ids"] = [2, 14, 25]
    config["target_hidden_size"] = 2048
    model = get_model_class(config["architectures"][0])(config)
    state = load_file(str(REAL_DRAFT_DIR / "model.safetensors"))
    model.load_weights(iter(state.items()))
    torch.manual_seed(0)
    with torch.no_grad():
        hidden = model.combine_hidden_states(torch.randn(4, 6144) * 0.05)
        draft_ids = model.draft_vocab_logits(hidden).argmax(-1)      # draft 空间
        target_ids = model.compute_logits(hidden).argmax(-1)        # target 空间
    assert target_ids.tolist() == (draft_ids + state["d2t"][draft_ids]).tolist()
    assert int(target_ids.max()) < model.target_vocab_size
    assert bool(model.t2d[target_ids].all())            # 都落在 t2d 标记的可用集合里
    # 反证：不做映射时会交出的 id 与正确值不同（真实 checkpoint 上 99.6% 的偏移非 0）
    assert not torch.equal(draft_ids, target_ids)


def test_compute_logits_is_identity_without_d2t():
    """draft 与 target 同词表（没有 d2t）时，`compute_logits()` 就是裸的 lm_head 输出。"""
    model = tiny_model()
    hidden = torch.randn(3, TINY_CONFIG["hidden_size"])
    with torch.no_grad():
        assert torch.equal(model.compute_logits(hidden), model.draft_vocab_logits(hidden))
        assert tuple(model.compute_logits(hidden).shape) == (3, TINY_CONFIG["vocab_size"])
