"""step65：MTP 的**前向语义**（需求 065 §3.2/§3.3/§4）。

三块：

1. **胶水的顺序**：`norm(block(fc([norm_emb(embed(x)) ‖ norm_hidden(h)])))`——拼接顺序、
   两侧归一化、最终归一化各在哪一步，都要能逐项验；改写任何一处都必须让输出变（反证）；
2. **`spec_step_idx` 的层选择**：`spec_step_idx % num_mtp_layers`（上游同款），不是"每步都跑第 0 层"；
3. **与上游 `Qwen3NextMTP` 的数值对照**：同一份胶水权重、两侧都把 decoder 层换成"直通"之后，
   forward 输出与 `compute_logits` 的 logits 逐值比。

第 3 条为什么要换掉 decoder 层：上游 Qwen3-Next 的 MTP 块是**混合注意力**（GatedDeltaNet + full
attention）+ MoE，本仓库只有稠密 Qwen3 —— 块本身没法逐值比（也不该假装能比）。换成直通之后，
比的就正好是"**MTP 特有的那部分**"：两侧归一化 → 拼接（顺序）→ `fc` → 最终归一化 → 输出头。
"""

import json

import pytest
import torch
from torch import nn

from minivllm.models.qwen3_mtp import Qwen3MTP, Qwen3MultiTokenPredictor
from minivllm.testing.tiny_models import tiny_qwen3_config

HIDDEN = 32
VOCAB = 11


class IdentityBlock(nn.Module):
    """把 MTP 的 decoder 层换成直通（只比胶水；签名与 `Qwen3DecoderLayer`/上游同款）。

    `use_attn_reduce_scatter_for_moe` 是上游 `Qwen3NextMultiTokenPredictor.forward` 会读的属性
    （TP/EP 的那条分支）：直通块上补一个 False，与「单卡、不做 reduce-scatter」一致。

    返回值必须是 `(hidden, residual_tensor)`：两侧的 forward 都写
    `hidden_states, _ = self.norm(hidden_states, residual)`，而 `RMSNorm` 在
    `residual is None` 时返回**单个**张量（融合残差那条分支才返回 tuple）。真实的
    decoder 层第一层会把 `residual` 设成输入，所以给个全零残差就与「没有残差」逐位等价
    （`x + 0 == x`），又不破坏两侧共同的调用约定。
    """

    use_attn_reduce_scatter_for_moe = False

    def forward(self, positions, hidden_states, residual):
        return hidden_states, torch.zeros_like(hidden_states)


class NoAttention(nn.Module):
    """把 block 里的注意力换成直通（单测里不想为了 attention 去搭 forward 上下文/KV 元数据）。"""

    def forward(self, positions, hidden_states):
        return hidden_states


def mtp_config(*, num_mtp_layers: int = 1, **overrides) -> dict:
    config = dict(tiny_qwen3_config("tiny_gqa"))
    config["num_nextn_predict_layers"] = num_mtp_layers
    config.update(overrides)
    return config


def build_model(*, num_mtp_layers: int = 1, seed: int = 0) -> Qwen3MTP:
    """建模型并把参数填成**有限值**（默认是 `torch.empty`，NaN 会让"输出变了没"失去意义）。"""
    torch.manual_seed(seed)
    model = Qwen3MTP(mtp_config(num_mtp_layers=num_mtp_layers)).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.05)
    return model


def stub_attention(model: Qwen3MTP) -> None:
    """把每层的注意力换成直通（保留当归一化与 MLP，层权重不同仍能区分选中的是哪一层）。"""
    for layer in model.model.layers:
        layer.self_attn = NoAttention()


# ---------------------------------------------------------------- 1. 胶水顺序


def test_forward_follows_the_source_op_order():
    """逐项复算：embed → pre_fc_norm_embedding；hidden → pre_fc_norm_hidden；cat → fc → block → norm。"""
    model = build_model()
    model.model.layers = nn.ModuleList([IdentityBlock()])          # 只留胶水
    input_ids = torch.tensor([1, 2, 3])
    positions = torch.tensor([0, 1, 2])
    hidden = torch.randn(3, HIDDEN)

    with torch.inference_mode():
        out = model(input_ids, positions, hidden)
        reference = model.model.fc(torch.cat(
            [model.model.pre_fc_norm_embedding(model.model.embed_tokens(input_ids)),
             model.model.pre_fc_norm_hidden(hidden)], dim=-1))
        reference = model.model.norm(reference)          # residual=None → 单个张量
    assert torch.equal(out, reference)


def test_concat_order_and_norms_are_load_bearing():
    """反证：拼接顺序反了、少归一化、少最终归一化——三处任何一处都会让输出变。"""
    model = build_model()
    model.model.layers = nn.ModuleList([IdentityBlock()])
    input_ids = torch.tensor([1, 2, 3])
    positions = torch.tensor([0, 1, 2])
    hidden = torch.randn(3, HIDDEN)
    with torch.inference_mode():
        out = model(input_ids, positions, hidden)
        embeds = model.model.pre_fc_norm_embedding(model.model.embed_tokens(input_ids))
        hidden_normed = model.model.pre_fc_norm_hidden(hidden)
        swapped = model.model.norm(model.model.fc(torch.cat([hidden_normed, embeds], dim=-1)))
        no_norm = model.model.norm(model.model.fc(torch.cat(
            [model.model.embed_tokens(input_ids), hidden], dim=-1)))
        hidden_only = model.model.norm(model.model.fc(torch.cat(
            [torch.zeros_like(embeds), hidden_normed], dim=-1)))
    for name, other in (("拼接顺序反了", swapped), ("两侧不归一化", no_norm),
                        ("embedding 半边为零", hidden_only)):
        assert not torch.allclose(out, other, atol=1e-6), f"{name} 居然没改变输出（假接线）"


def test_misaligned_hidden_changes_the_output():
    """反证：把 hidden **错位一行**（上一枚的 hidden 配错 token）必须改变输出。

    这是 MTP 最容易接错的地方——第一遍的 `(h_i, t_{i+1}) → t_{i+2}` 配错一行不会报错，
    只会让草稿静默变差。
    """
    model = build_model()
    model.model.layers = nn.ModuleList([IdentityBlock()])
    input_ids = torch.tensor([1, 2, 3, 4])
    positions = torch.tensor([0, 1, 2, 3])
    hidden = torch.randn(4, HIDDEN)
    shifted = torch.roll(hidden, shifts=1, dims=0)
    with torch.inference_mode():
        out = model(input_ids, positions, hidden)
        out_shifted = model(input_ids, positions, shifted)
    assert not torch.allclose(out, out_shifted, atol=1e-6)
    assert float((out - out_shifted).abs().max()) > 1e-2, "错位的差异应该很明显"


# ---------------------------------------------------------------- 2. spec_step_idx


def test_spec_step_idx_selects_the_mtp_layer():
    """`spec_step_idx % num_mtp_layers` 选层（上游同款），不是"每步都用第 0 层"。"""
    model = build_model(num_mtp_layers=2, seed=3)
    stub_attention(model)
    for index, layer in enumerate(model.model.layers):
        # 让两层的权重不同，才能看出选的是哪一层
        with torch.no_grad():
            layer.mlp.down_proj.weight.add_(float(index + 1))
    input_ids = torch.tensor([1, 2])
    positions = torch.tensor([0, 1])
    hidden = torch.randn(2, HIDDEN)
    with torch.inference_mode():
        step0 = model(input_ids, positions, hidden, spec_step_idx=0)
        step1 = model(input_ids, positions, hidden, spec_step_idx=1)
        step2 = model(input_ids, positions, hidden, spec_step_idx=2)     # 2 % 2 == 0
    assert not torch.allclose(step0, step1)
    assert torch.equal(step0, step2), "spec_step_idx 要对 num_mtp_layers 取模"


def test_compute_logits_is_lm_head():
    """`compute_logits(hidden, spec_step_idx)` = 词表头（上游在它外面还有 LogitsProcessor）。"""
    model = build_model()
    hidden = torch.randn(3, HIDDEN)
    with torch.inference_mode():
        logits = model.compute_logits(hidden)
        assert logits.shape == (3, VOCAB)
        assert torch.equal(logits, model.lm_head(hidden))
        assert torch.equal(model.compute_logits(hidden, spec_step_idx=1), logits)


# ---------------------------------------------------------------- 3. 与上游对照


UPSTREAM_HIDDEN = 64      # = num_attention_heads * head_dim（上游的 Qwen3-Next 配置要自洽）
# 词表取 64：vLLM 会把词表**补齐到 64 的倍数**（embed/lm_head 的行数是补齐后的），
# 给个不是倍数的值会让两边形状对不上——那是"补齐规则"的差异，不是 MTP 的差异。
UPSTREAM_VOCAB = 64


def upstream_qwen3_next_config(*, num_mtp_layers: int = 1) -> dict:
    """上游认得出的 Qwen3-Next 配置（`Qwen3NextMTP.__init__` 要 MoE 参数与混合层类型）。

    `hidden_size` 必须等于 `num_attention_heads * head_dim`：上游解析配置时会按这个关系推导，
    对不上就会得到 (vocab, 64) 的 lm_head，与我们的 (vocab, 32) 复制不上（实测踩过）。
    """
    config = mtp_config(num_mtp_layers=num_mtp_layers)
    config.update({
        "architectures": ["Qwen3NextForCausalLM"], "model_type": "qwen3_next",
        "hidden_size": UPSTREAM_HIDDEN, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 64,
        "vocab_size": UPSTREAM_VOCAB, "max_position_embeddings": 64, "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False, "hidden_act": "silu", "attention_bias": False,
        "attention_dropout": 0.0, "initializer_range": 0.02, "dtype": "float32",
        "bos_token_id": 0, "eos_token_id": 4, "pad_token_id": None, "use_cache": True,
        # 混合层（GatedDeltaNet）参数
        "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16,
        "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4,
        "full_attention_interval": 2, "layer_types": ["linear_attention", "full_attention"],
        "partial_rotary_factor": 0.25,
        # 稠密（`num_experts=0`）：块会被换成直通，MoE 参数在本次对照里没有作用，而带 MoE
        # 会让上游去要 **EP 组**（同进程里 EP 组由先跑的用例的配置决定 → 执行顺序一变就炸）
        "num_experts": 0, "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16, "decoder_sparse_step": 1,
        "norm_topk_prob": True, "n_group": 1, "topk_group": 1, "num_shared_experts": 1,
    })
    config.pop("rope_parameters", None)
    return config


@pytest.fixture(scope="module")
def glue_pair(tmp_path_factory):
    """`(ours, upstream, state)`：同一份胶水权重、两侧的 decoder 层都换成直通。"""
    from safetensors.torch import save_file

    from vllm.config import (CacheConfig, DeviceConfig, ModelConfig, ParallelConfig,
                             SpeculativeConfig, VllmConfig, set_current_vllm_config)
    from vllm.config.compilation import CompilationConfig, CompilationMode
    from vllm.config.scheduler import SchedulerConfig as UpSchedulerConfig
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed.parallel_state import model_parallel_is_initialized
    from vllm.model_executor.models.qwen3_next_mtp import Qwen3NextMTP

    config = upstream_qwen3_next_config()
    model_dir = tmp_path_factory.mktemp("tiny_qwen3_next_mtp")
    (model_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file({"dummy": torch.zeros(1)}, str(model_dir / "model.safetensors"))

    ours = Qwen3MTP({**config, "architectures": ["Qwen3MTPModel"]}).eval()
    ours.model.layers = nn.ModuleList([IdentityBlock()])

    # 两侧加载**同一份**胶水权重（decoder 层被换成直通，不参与比较）。
    # 形状取自我们模型的参数：两边对不上就说明配置解析出了分歧（那本身是对照要报的错）。
    generator = torch.Generator().manual_seed(7)

    def rand(*shape):
        return torch.empty(*shape).normal_(0.0, 0.05, generator=generator)

    glue_names = ["model.embed_tokens.weight", "model.fc.weight",
                  "model.pre_fc_norm_embedding.weight", "model.pre_fc_norm_hidden.weight",
                  "model.norm.weight", "lm_head.weight"]
    ours_params = dict(ours.named_parameters())
    state = {}
    for name in glue_names:
        shape = tuple(ours_params[name].shape)
        tensor = rand(*shape)
        if "norm" in name:
            tensor = tensor + 1.0
        state[name] = tensor

    target = ModelConfig(model=str(model_dir), dtype="float32", max_model_len=64)
    draft = ModelConfig(model=str(model_dir), dtype="float32", max_model_len=64)
    parallel = ParallelConfig()
    spec = SpeculativeConfig(model=str(model_dir), method="mtp", num_speculative_tokens=1,
                             target_model_config=target, draft_model_config=draft,
                             target_parallel_config=parallel)
    vllm_config = VllmConfig(
        model_config=target, speculative_config=spec,
        cache_config=CacheConfig(block_size=16, gpu_memory_utilization=0.1),
        parallel_config=parallel, device_config=DeviceConfig("cuda"),
        scheduler_config=UpSchedulerConfig(max_num_seqs=1, max_num_batched_tokens=64,
                                           max_model_len=64, is_encoder_decoder=False),
        # 比的是**eager 的数学**：关掉 torch.compile（否则 dynamo 会去 trace 我们的直通块）
        compilation_config=CompilationConfig(mode=CompilationMode.NONE))
    with set_current_vllm_config(vllm_config):
        # 同一个 pytest 进程里可能已经有别的用例初始化过（例如 tests/step63 的上游对照），
        # 而 vLLM 的 `initialize_model_parallel()` 会 assert "已经初始化过" —— 这里做成幂等，
        # 免得"一起跑就报错、单独跑就通过"（实测踩过）。
        if not model_parallel_is_initialized():
            init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                         distributed_init_method="tcp://127.0.0.1:29791",
                                         backend="gloo")
            initialize_model_parallel(tensor_model_parallel_size=1,
                                      pipeline_model_parallel_size=1)
        # `set_moe_parameters()` 只做 MoE 记账（本对照把块换成直通，用不到），而它要求
        # "model.layers 里必须有 MoE 层" —— 稠密配置下会直接报错。跳过它让这次对照与
        # "上游模型带不带 MoE"解耦。
        original_set_moe = Qwen3NextMTP.set_moe_parameters
        Qwen3NextMTP.set_moe_parameters = lambda self: None
        try:
            upstream = Qwen3NextMTP(vllm_config=vllm_config, prefix="mtp")
        finally:
            Qwen3NextMTP.set_moe_parameters = original_set_moe

    upstream = upstream.float().eval()
    upstream.model.layers = nn.ModuleList([IdentityBlock()])

    for model in (ours, upstream):
        params = dict(model.named_parameters())
        for name, tensor in state.items():
            assert name in params, f"{name} 不在 {type(model).__name__} 的参数里"
            assert tuple(params[name].shape) == tuple(tensor.shape), (
                f"{name} 的形状两边不一致：我们 {tuple(tensor.shape)} vs "
                f"{type(model).__name__} {tuple(params[name].shape)}")
            # **归一化的"口味"差异**：上游 Qwen3-Next 用 `GemmaRMSNorm`（`x * (1 + w)`，
            # weight 初始为 0），本仓库是 Qwen3 式 `x * w`。这是**家族差异**、不是 MTP 胶水差异，
            # 所以把权重按两种语义对齐（让两边实际乘的系数相同），比的仍然是胶水的顺序与结构。
            value = tensor - 1.0 if ("norm" in name and model is upstream) else tensor
            with torch.no_grad():
                params[name].copy_(value)
    return ours, upstream, state


def test_upstream_glue_forward_matches(glue_pair):
    """forward（胶水部分）与上游 `Qwen3NextMTP` 逐值一致（fp32）。"""
    ours, upstream, state = glue_pair
    torch.manual_seed(11)
    input_ids = torch.tensor([1, 2, 3, 4, 5])
    positions = torch.tensor([0, 1, 2, 10, 11])
    hidden = torch.randn(5, state["model.norm.weight"].shape[0])
    with torch.inference_mode():
        ours_out = ours(input_ids, positions, hidden)
        upstream_out = upstream(input_ids, positions, hidden)
    assert ours_out.shape == upstream_out.shape == (5, state["model.norm.weight"].shape[0])
    max_delta = float((ours_out - upstream_out).abs().max())
    # 同权重同顺序：差异只可能来自归一化/线性的内核实现（fp32 累加顺序与"先乘后转精度"），
    # 量级 1e-5；**结构错**（拼接顺序/归一化位置/层选错）会差 1e-1 以上
    print(f"胶水 forward vs 上游：max|Δ| = {max_delta:.3e}")
    assert max_delta < 1e-3, f"与上游的胶水 forward 差太多：max|Δ|={max_delta}"


def test_upstream_compute_logits_matches(glue_pair):
    """`compute_logits` 与上游逐值一致（同一份 lm_head 权重）。"""
    ours, upstream, state = glue_pair
    torch.manual_seed(13)
    hidden = torch.randn(4, state["model.norm.weight"].shape[0])
    with torch.inference_mode():
        ours_logits = ours.compute_logits(hidden)
        upstream_logits = upstream.compute_logits(hidden, spec_step_idx=0)
    max_delta = float((ours_logits - upstream_logits).abs().max())
    assert ours_logits.shape == upstream_logits.shape == (4, state["lm_head.weight"].shape[0])
    print(f"compute_logits vs 上游：max|Δ| = {max_delta:.3e}")
    assert max_delta < 1e-3, f"compute_logits 与上游差太多：max|Δ|={max_delta}"


def test_predictor_forward_signature_and_return():
    """`Qwen3MultiTokenPredictor.forward` 的签名与上游一致（含 `spec_step_idx` 的默认值）。"""
    import inspect

    ours = inspect.signature(Qwen3MultiTokenPredictor.forward)
    assert list(ours.parameters)[:4] == ["self", "input_ids", "positions", "hidden_states"]
    assert ours.parameters["spec_step_idx"].default == 0
    upstream = None
    try:
        from vllm.model_executor.models.qwen3_next_mtp import Qwen3NextMultiTokenPredictor

        upstream = inspect.signature(Qwen3NextMultiTokenPredictor.forward)
    except Exception:                                    # noqa: BLE001 —— 上游缺失就只比我们的
        pass
    if upstream is not None:
        assert list(upstream.parameters)[:4] == list(ours.parameters)[:4]
