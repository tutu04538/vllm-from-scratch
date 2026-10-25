"""需求 80E.E1：Gemma4 assistant 的模型层差分（不接真实 target/service）。

本文件只做 E1：

1. 用冻结上游 `Gemma4MTP` 的**同一份 tiny 权重**，比较我们实现的
   `Gemma4MTP` 的逐层 hidden、两个返回值（draft / backbone）与 logits；
2. 覆盖 backbone=16、assistant=8 的跨宽度数据流；
3. 覆盖 Q-only attention 的 q 投影 + q_norm + RoPE（sliding/full 两种层）；
4. 覆盖 masked embedder 的选簇、稀疏 logits、argmax；
5. 覆盖权重路由（pre/post projection、gate/up 打包、buffer）与加载覆盖检查。

注意力核心在两边的模型差分里被替换成"原样返回 q"的 identity 模块：E1 验证的是
assistant 的模型结构与胶水，Q-only 的真实 cache 读取已由 80D 契约覆盖；真实 target 的
attention/KV group 接入属 80E.E2。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from transformers import Gemma4AssistantConfig, Gemma4TextConfig

from minivllm import ModelConfig
from minivllm.layers import VocabParallelEmbedding
from minivllm.model_loader import get_model
from minivllm.models.gemma4_mtp import (Gemma4MTP, Gemma4MTPMaskedEmbedder,
                                        Gemma4MultiTokenPredictor)
from minivllm.models.registry import get_model_class

VOCAB = 32
HIDDEN = 8
BACKBONE = 16
INTERMEDIATE = 16
NUM_LAYERS = 2
NUM_CENTROIDS = 4
TOP_K = 2


@dataclass
class ModelPair:
    ours: Gemma4MTP
    upstream: Gemma4MTP
    assistant_config: Gemma4AssistantConfig
    state: dict[str, torch.Tensor]
    shared_embedding: torch.Tensor
    monkeypatch: pytest.MonkeyPatch


class _IdentityAttention(nn.Module):
    """把 attention core 换成 q 直通；q 投影/q_norm/RoPE 仍在模型里被比较。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, query, key=None, value=None):
        return query


def _tiny_config(*, use_ordered_embeddings: bool = False,
                 backbone: int = BACKBONE, assistant: int = HIDDEN,
                 final_logit_softcapping: float | None = None):
    text = Gemma4TextConfig(
        vocab_size=VOCAB,
        hidden_size=assistant,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        max_position_embeddings=64,
        layer_types=["sliding_attention", "full_attention"],
        # assistant 的每一层都共享 target KV；这是 Gemma4AssistantConfig 的约束。
        num_kv_shared_layers=NUM_LAYERS,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=0,
        rms_norm_eps=1e-6,
        final_logit_softcapping=final_logit_softcapping,
    )
    return Gemma4AssistantConfig(
        text_config=text,
        backbone_hidden_size=backbone,
        use_ordered_embeddings=use_ordered_embeddings,
        num_centroids=NUM_CENTROIDS,
        centroid_intermediate_top_k=TOP_K,
        tie_word_embeddings=True,
        architectures=["Gemma4MTPModel"],
        model_type="gemma4_assistant",
    )


def _make_state(*, assistant: int = HIDDEN, backbone: int = BACKBONE,
                masked: bool = False, seed: int = 7) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.empty(*shape).normal_(0.0, 0.05, generator=generator)

    embedding = rand(VOCAB, assistant)
    # 顺序必须按模块分组：下游 AutoWeightsLoader 用 itertools.groupby。
    state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": embedding,
    }
    for layer in range(NUM_LAYERS):
        prefix = f"model.layers.{layer}."
        state.update({
            prefix + "self_attn.q_proj.weight": rand(2 * 8, assistant),
            prefix + "self_attn.o_proj.weight": rand(assistant, 2 * 8),
            prefix + "self_attn.q_norm.weight": rand(8) + 1.0,
            prefix + "mlp.gate_proj.weight": rand(INTERMEDIATE, assistant),
            prefix + "mlp.up_proj.weight": rand(INTERMEDIATE, assistant),
            prefix + "mlp.down_proj.weight": rand(assistant, INTERMEDIATE),
            prefix + "input_layernorm.weight": rand(assistant) + 1.0,
            prefix + "post_attention_layernorm.weight": rand(assistant) + 1.0,
            prefix + "pre_feedforward_layernorm.weight": rand(assistant) + 1.0,
            prefix + "post_feedforward_layernorm.weight": rand(assistant) + 1.0,
            prefix + "layer_scalar": torch.full((1,), 1.3),
        })
    state["model.norm.weight"] = rand(assistant) + 1.0
    # 检查点把两个跨宽度投影放在顶层；模型里在 model. 下。
    state["pre_projection.weight"] = rand(assistant, 2 * backbone)
    state["post_projection.weight"] = rand(backbone, assistant)
    state["lm_head.weight"] = embedding.clone()
    if masked:
        state["masked_embedding.centroids.weight"] = rand(NUM_CENTROIDS, assistant)
        state["masked_embedding.token_ordering"] = torch.randperm(
            VOCAB, generator=generator)
    return state


def _replace_shared_embedding(pair: ModelPair) -> None:
    """模拟 E2 的 `_share_embeddings`：assistant embedding 换成 target 的 backbone 宽。"""
    state = pair.state["model.embed_tokens.weight"]
    backbone = pair.assistant_config.backbone_hidden_size

    ours_embed = VocabParallelEmbedding(VOCAB, backbone)
    ours_embed.weight.data.copy_(torch.randn(
        VOCAB, backbone, generator=torch.Generator().manual_seed(19)))
    pair.ours.model.embed_tokens = ours_embed

    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding as UpEmbedding,
    )

    upstream_embed = UpEmbedding(VOCAB, backbone)
    # vLLM 的 embedding 会把词表 pad 到 64；只填真实行，padding 行清零。
    upstream_embed.weight.data[:VOCAB].copy_(ours_embed.weight.data)
    upstream_embed.weight.data[VOCAB:].zero_()
    pair.upstream.model.embed_tokens = upstream_embed

    # normalizer 是 __init__ 里按 backbone 宽建的；这里显式确认它没被替换掉。
    assert pair.ours.model.normalizer.item() == pytest.approx(backbone ** 0.5)
    assert pair.upstream.model.normalizer.item() == pytest.approx(backbone ** 0.5)
    del state


def _build_upstream(assistant_config: Gemma4AssistantConfig, model_dir: Path):
    from vllm.config import (CacheConfig, DeviceConfig, ModelConfig as UpModelConfig,
                             ParallelConfig, SpeculativeConfig, VllmConfig,
                             set_current_vllm_config)
    from vllm.config.compilation import CompilationConfig, CompilationMode
    from vllm.config.scheduler import SchedulerConfig as UpSchedulerConfig
    from vllm.distributed import (init_distributed_environment,
                                  initialize_model_parallel)
    from vllm.distributed.parallel_state import model_parallel_is_initialized
    import vllm.model_executor.models.gemma4_mtp as upstream_module

    target = UpModelConfig(model=str(model_dir), dtype="float32", max_model_len=64)
    draft = UpModelConfig(model=str(model_dir), dtype="float32", max_model_len=64)
    parallel = ParallelConfig()
    spec = SpeculativeConfig(
        model=str(model_dir), method="mtp", num_speculative_tokens=1,
        target_model_config=target, draft_model_config=draft,
        target_parallel_config=parallel)
    cfg = VllmConfig(
        model_config=target, speculative_config=spec,
        cache_config=CacheConfig(block_size=16, gpu_memory_utilization=0.1),
        parallel_config=parallel, device_config=DeviceConfig("cpu"),
        scheduler_config=UpSchedulerConfig(
            max_num_seqs=1, max_num_batched_tokens=64, max_model_len=64,
            is_encoder_decoder=False),
        compilation_config=CompilationConfig(mode=CompilationMode.NONE))
    with set_current_vllm_config(cfg):
        if not model_parallel_is_initialized():
            init_distributed_environment(
                world_size=1, rank=0, local_rank=0,
                distributed_init_method="tcp://127.0.0.1:29911",
                backend="gloo")
            initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        model = upstream_module.Gemma4MTP(vllm_config=cfg, prefix="")
    return model


@pytest.fixture(scope="module")
def model_pair(tmp_path_factory):
    assistant_config = _tiny_config()
    model_dir = tmp_path_factory.mktemp("tiny_gemma4_e1")
    assistant_config.save_pretrained(model_dir)
    save_file({"dummy": torch.zeros(1)}, str(model_dir / "model.safetensors"))

    # 上游模型需要：identity attention、native RoPE、native GeGLU（CPU 上没有 CUDA 算子）。
    from vllm.model_executor.layers.activation import GeluAndMul as UpGelu
    from vllm.model_executor.layers.rotary_embedding.base import (
        RotaryEmbedding as UpRotary,
    )

    mp = pytest.MonkeyPatch()
    import minivllm.models.gemma4_mtp as ours_module
    import vllm.model_executor.models.gemma4_mtp as upstream_module

    mp.setattr(upstream_module, "Attention", _IdentityAttention)
    mp.setattr(ours_module, "Attention", _IdentityAttention)
    mp.setattr(UpRotary, "forward", UpRotary.forward_native)
    mp.setattr(UpGelu, "forward", UpGelu.forward_native)
    try:
        upstream_model = _build_upstream(assistant_config, model_dir)
        ours_model = Gemma4MTP(assistant_config)
        state = _make_state()
        ours_model.load_weights(iter(state.items()))
        upstream_model.load_weights(iter(state.items()))
        pair = ModelPair(
            ours=ours_model, upstream=upstream_model,
            assistant_config=assistant_config, state=state,
            shared_embedding=torch.empty(0), monkeypatch=mp,
        )
        _replace_shared_embedding(pair)
        yield pair
    finally:
        mp.undo()


def _forward_inputs(seed: int = 3):
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.tensor([1, 2, 3])
    positions = torch.tensor([0, 1, 2])
    hidden_states = torch.randn(3, BACKBONE, generator=generator)
    return input_ids, positions, hidden_states


def _capture_layers(model: Gemma4MTP):
    captured: list[torch.Tensor] = []
    handles = [
        layer.register_forward_hook(
            lambda module, inputs, output: captured.append(output[0].detach().clone()))
        for layer in model.model.layers
    ]
    return captured, handles


def test_parameter_shapes_and_no_kv_projections(model_pair):
    ours = model_pair.ours
    assert ours.model.pre_projection.weight.shape == (HIDDEN, 2 * BACKBONE)
    assert ours.model.post_projection.weight.shape == (BACKBONE, HIDDEN)
    assert ours.model.pre_projection.weight.shape == (HIDDEN, 2 * BACKBONE)
    # 共享 target embedding 后，lm_head 仍是 assistant 宽度的旧参数（上游
    # `_share_embeddings` 同样只替换 model.embed_tokens，不动 lm_head）。
    assert ours.lm_head.weight.shape == (VOCAB, HIDDEN)

    for layer in ours.model.layers:
        attn = layer.self_attn
        assert attn.is_kv_shared_layer is True
        assert attn.attn.q_only is True
        assert attn.scaling == 1.0
        assert attn.num_kv_heads == 1
        assert attn.head_dim == 8
        assert not hasattr(attn, "k_proj")
        assert not hasattr(attn, "v_proj")
        assert attn.rotary_emb.rotary_dim in (2, 8)


def test_weight_loader_routes_checkpoint_names(model_pair):
    state = model_pair.state
    ours = model_pair.ours
    # pre/post projection 的顶层名 → model.*
    torch.testing.assert_close(
        ours.model.pre_projection.weight, state["pre_projection.weight"])
    torch.testing.assert_close(
        ours.model.post_projection.weight, state["post_projection.weight"])
    # gate/up 两个独立权重 → 一个打包参数的 [0:I] 与 [I:2I]
    layer0 = ours.model.layers[0]
    torch.testing.assert_close(
        layer0.mlp.gate_up_proj.weight[:INTERMEDIATE],
        state["model.layers.0.mlp.gate_proj.weight"])
    torch.testing.assert_close(
        layer0.mlp.gate_up_proj.weight[INTERMEDIATE:],
        state["model.layers.0.mlp.up_proj.weight"])
    # 普通参数与 buffer
    torch.testing.assert_close(
        layer0.self_attn.q_norm.weight,
        state["model.layers.0.self_attn.q_norm.weight"])
    torch.testing.assert_close(
        layer0.layer_scalar, state["model.layers.0.layer_scalar"])
    # tied lm_head：加载器跳过检查点里的 lm_head.weight，覆盖检查仍通过。
    # 这里用未做 embedding sharing 的新模型验证 tie 本身仍成立。
    tied = Gemma4MTP(model_pair.assistant_config)
    assert tied.lm_head.weight is tied.model.embed_tokens.weight
    tied.load_weights(iter(state.items()))
    assert tied.lm_head.weight is tied.model.embed_tokens.weight


def test_weight_loader_rejects_missing_and_unknown(model_pair):
    fresh = Gemma4MTP(model_pair.assistant_config)
    state = _make_state(seed=23)
    missing = dict(state)
    missing.pop("model.norm.weight")
    with pytest.raises(ValueError, match="没有被加载"):
        fresh.load_weights(iter(missing.items()))

    fresh = Gemma4MTP(model_pair.assistant_config)
    unknown = dict(state)
    unknown["bogus.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="没有名为"):
        fresh.load_weights(iter(unknown.items()))


def test_get_model_loads_registered_architecture(tmp_path):
    assistant_config = _tiny_config()
    model_dir = tmp_path / "tiny_gemma4_loader"
    model_dir.mkdir()
    assistant_config.save_pretrained(model_dir)
    save_file(_make_state(seed=31), str(model_dir / "model.safetensors"))
    hf = json.loads((model_dir / "config.json").read_text())
    model_config = ModelConfig(
        model=str(model_dir), dtype="float32", max_model_len=64, hf_config=hf)
    model = get_model(model_config, torch.device("cpu"))
    assert isinstance(model, Gemma4MTP)
    assert model.model.embed_tokens.weight.shape == (VOCAB, HIDDEN)
    assert model.model.layers[0].layer_scalar.item() == pytest.approx(1.3)
    assert get_model_class("Gemma4MTPModel") is Gemma4MTP


def test_q_pipeline_matches_frozen_attention(model_pair):
    _, positions, _ = _forward_inputs()
    positions = torch.arange(4)
    generator = torch.Generator().manual_seed(41)
    hidden = torch.randn(4, HIDDEN, generator=generator)
    for layer_idx in range(NUM_LAYERS):
        ours_attn = model_pair.ours.model.layers[layer_idx].self_attn
        up_attn = model_pair.upstream.model.layers[layer_idx].self_attn
        got = ours_attn.attention_pre(positions, hidden)

        q, _ = up_attn.q_proj(hidden)
        q = q.unflatten(-1, (up_attn.num_heads, up_attn.head_dim))
        q = up_attn.q_norm(q)
        q = q.flatten(-2, -1)
        q, _ = up_attn.rotary_emb(positions, q, None)
        torch.testing.assert_close(got, q, rtol=1e-6, atol=1e-6)


def test_forward_layer_by_layer_matches_frozen(model_pair):
    input_ids, positions, hidden = _forward_inputs()
    ours_layers, ours_handles = _capture_layers(model_pair.ours)
    up_layers, up_handles = _capture_layers(model_pair.upstream)
    try:
        with torch.no_grad():
            ours_draft, ours_backbone = model_pair.ours(
                input_ids, positions, hidden)
            up_draft, up_backbone = model_pair.upstream(
                input_ids, positions, hidden)
    finally:
        for handle in ours_handles + up_handles:
            handle.remove()

    assert ours_draft.shape == (3, HIDDEN)
    assert ours_backbone.shape == (3, BACKBONE)
    assert len(ours_layers) == len(up_layers) == NUM_LAYERS
    for idx, (got, want) in enumerate(zip(ours_layers, up_layers)):
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6,
                                   msg=lambda msg: f"layer {idx}: {msg}")
    torch.testing.assert_close(ours_draft, up_draft, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(ours_backbone, up_backbone, rtol=1e-6, atol=1e-6)
    with torch.no_grad():
        ours_logits = model_pair.ours.compute_logits(ours_draft)
        up_logits = model_pair.upstream.compute_logits(up_draft)
    torch.testing.assert_close(ours_logits, up_logits, rtol=1e-6, atol=1e-6)


def test_backbone_and_assistant_widths_are_not_interchangeable(model_pair):
    input_ids, positions, hidden = _forward_inputs()
    with torch.no_grad():
        draft, backbone = model_pair.ours(input_ids, positions, hidden)
    assert draft.shape[-1] == HIDDEN
    assert backbone.shape[-1] == BACKBONE
    # 把 assistant 宽的 draft 当反馈喂回去，必须在入口明确报错，而不是在
    # pre_projection 里出现一个"24 宽"的静默形状错。
    with pytest.raises(ValueError, match="backbone"):
        model_pair.ours(input_ids, positions, draft)
    # 真正的下一步反馈用 backbone 宽，能正常跑。
    with torch.no_grad():
        next_draft, next_backbone = model_pair.ours(
            input_ids, positions, backbone)
    assert next_draft.shape == draft.shape
    assert next_backbone.shape == backbone.shape


def test_masked_embedder_matches_frozen():
    from vllm.model_executor.models.gemma4_mtp import (
        Gemma4MTPMaskedEmbedder as UpMasked,
    )

    generator = torch.Generator().manual_seed(51)
    up = UpMasked(HIDDEN, VOCAB, NUM_CENTROIDS, TOP_K)
    ours = Gemma4MTPMaskedEmbedder(HIDDEN, VOCAB, NUM_CENTROIDS, TOP_K)
    state = {
        "centroids.weight": torch.randn(
            NUM_CENTROIDS, HIDDEN, generator=generator),
        "token_ordering": torch.randperm(VOCAB, generator=generator),
    }
    up.load_state_dict(state)
    ours.load_state_dict(state)
    hidden = torch.randn(5, HIDDEN, generator=generator)
    lm_head = torch.randn(VOCAB, HIDDEN, generator=generator)
    with torch.no_grad():
        got = ours(hidden, lm_head)
        want = up(hidden, lm_head)
        got_top = ours.get_top_tokens(hidden, lm_head)
        want_top = up.get_top_tokens(hidden, lm_head)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert torch.equal(torch.isfinite(got), torch.isfinite(want))
    assert torch.equal(got_top, want_top)


def test_masked_loader_and_compute_logits(tmp_path):
    assistant_config = _tiny_config(use_ordered_embeddings=True)
    model = Gemma4MTP(assistant_config)
    state = _make_state(masked=True, seed=61)
    loaded = model.load_weights(iter(state.items()))
    assert model.masked_embedding is not None
    assert "masked_embedding.token_ordering" in loaded
    assert torch.equal(
        model.masked_embedding.token_ordering,
        state["masked_embedding.token_ordering"])
    hidden = torch.randn(4, HIDDEN, generator=torch.Generator().manual_seed(62))
    logits = model.compute_logits(hidden)
    assert logits.shape == (4, VOCAB)
    # 只有被选中的 16 列是真实分数；其余用 finfo.min 占位（冻结实现同样用有限最小值，
    # 不是 -inf）。
    masked_value = torch.finfo(logits.dtype).min
    assert int((logits == masked_value).sum()) == 4 * (VOCAB - TOP_K * (VOCAB // NUM_CENTROIDS))
    assert torch.equal(logits, model.masked_embedding(
        hidden, model.lm_head.weight))

    missing = dict(state)
    missing.pop("masked_embedding.token_ordering")
    with pytest.raises(ValueError, match="token_ordering"):
        Gemma4MTP(assistant_config).load_weights(iter(missing.items()))


def test_final_logit_softcap_is_applied():
    assistant_config = _tiny_config(final_logit_softcapping=30.0)
    model = Gemma4MTP(assistant_config)
    weight = torch.randn(VOCAB, HIDDEN, generator=torch.Generator().manual_seed(71))
    with torch.no_grad():
        model.lm_head.weight.copy_(weight)
    hidden = torch.randn(3, HIDDEN, generator=torch.Generator().manual_seed(72))
    got = model.compute_logits(hidden)
    logits = hidden @ weight.T
    want = torch.tanh(logits / 30.0) * 30.0
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)


def test_unsupported_rope_type_and_bad_centroids_are_rejected():
    text = Gemma4TextConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, global_head_dim=8, max_position_embeddings=64,
        layer_types=["full_attention"], num_kv_shared_layers=1,
        hidden_size_per_layer_input=0, vocab_size_per_layer_input=0,
    )
    # transformers 会在构造时校验已知 rope_type 的必需字段；本测试要验证的是
    # **我们的**适配边界，所以在构造后换成未适配的 llama3。
    text.rope_parameters = {
        "full_attention": {"rope_theta": 10000.0, "rope_type": "llama3"}}
    assistant = Gemma4AssistantConfig(
        text_config=text, backbone_hidden_size=BACKBONE,
        use_ordered_embeddings=False, tie_word_embeddings=True,
        architectures=["Gemma4MTPModel"], model_type="gemma4_assistant")
    with pytest.raises(ValueError, match="RoPE 类型"):
        Gemma4MTP(assistant)

    with pytest.raises(ValueError, match="num_centroids"):
        Gemma4MTPMaskedEmbedder(HIDDEN, VOCAB, 5, TOP_K)

    # 未适配的结构不能静默按稠密 GELU MLP 跑。
    text = Gemma4TextConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, global_head_dim=8, max_position_embeddings=64,
        layer_types=["full_attention"], num_kv_shared_layers=1,
        hidden_size_per_layer_input=0, vocab_size_per_layer_input=0,
    )
    text.enable_moe_block = True
    with pytest.raises(NotImplementedError, match="MoE"):
        Gemma4MTP(text)
    text = Gemma4TextConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, global_head_dim=8, max_position_embeddings=64,
        layer_types=["full_attention"], num_kv_shared_layers=1,
        hidden_size_per_layer_input=0, vocab_size_per_layer_input=0,
    )
    text.attn_logit_softcapping = 30.0
    with pytest.raises(NotImplementedError, match="softcap"):
        Gemma4MTP(text)


def test_heterogeneous_head_dims_follow_layer_type():
    """Gemma4 的 full 层可以用更大的 head_dim / 更多 KV heads。"""
    text = Gemma4TextConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, global_head_dim=16, max_position_embeddings=64,
        layer_types=["sliding_attention", "full_attention"],
        num_kv_shared_layers=2, attention_k_eq_v=True,
        num_global_key_value_heads=2, hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=0,
    )
    assistant = Gemma4AssistantConfig(
        text_config=text, backbone_hidden_size=BACKBONE,
        use_ordered_embeddings=False, tie_word_embeddings=True,
        architectures=["Gemma4MTPModel"], model_type="gemma4_assistant")
    model = Gemma4MTP(assistant)
    sliding, full = model.model.layers[0].self_attn, model.model.layers[1].self_attn
    assert (sliding.layer_type, sliding.head_dim, sliding.num_kv_heads) == (
        "sliding_attention", 8, 1)
    assert (full.layer_type, full.head_dim, full.num_kv_heads) == (
        "full_attention", 16, 2)
    assert sliding.q_proj.weight.shape == (2 * 8, HIDDEN)
    assert full.q_proj.weight.shape == (2 * 16, HIDDEN)
    assert sliding.q_norm.weight.shape == (8,)
    assert full.q_norm.weight.shape == (16,)
    with pytest.raises(ValueError, match="backbone"):
        model(torch.tensor([1]), torch.tensor([0]), torch.zeros(1, HIDDEN))
