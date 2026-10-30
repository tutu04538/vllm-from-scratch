"""需求 080E.E2：Gemma4 target + assistant 的真实接入路径。

这里区分两类证据，避免把"机制跑通"说成"真实 checkpoint 已验证"：

* `test_target_logits_match_hf_across_groups`：tiny target 与 HF 逐值对数（数值正确性）；
* `test_real_assistant_*`：**本机真实 assistant checkpoint** + 同配置的合成 target
  （本机没有真实 target 权重，E2 的完整真实 checkpoint 仍标"集成待验"）。测试检查
  真实 assistant 的加载/宽度/centroid/masked embedder、embedding 共享、Q-only 源层只读、
  多 KV group 映射，以及生成能跑完；
* V1/V2 各自独立构图/输入，比较同一 workload 的 greedy 输出；
* V2 的 eager 与整段 CUDA Graph 比较同一 workload。
"""

from __future__ import annotations

import copy
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

import pytest
import torch
from safetensors.torch import load_file, save_file

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig,
                      SamplingParams, SchedulerConfig, SpeculativeConfig,
                      UniProcExecutor, VllmConfig, Worker)
from minivllm.attention.layer import Attention
from minivllm.config import CUDAGraphMode, CompilationConfig
from minivllm.models.gemma4_mtp import Gemma4MTP

ROOT = Path(__file__).resolve().parents[2]
ASSISTANT = ROOT / "models" / "gemma-4-E2B-it-assistant"
REAL_TARGET = ROOT / "models" / "gemma-4-E2B-it"
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Gemma4 E2 集成测试需要 CUDA")
requires_assistant = pytest.mark.skipif(
    not (ASSISTANT / "model.safetensors").is_file(),
    reason="本机没有真实 Gemma4 assistant checkpoint")
requires_real_target = pytest.mark.skipif(
    not (REAL_TARGET / "model.safetensors").is_file(),
    reason="本机没有真实 Gemma4 target checkpoint（models/gemma-4-E2B-it）")


# ---------------------------------------------------------------------------
# 固定配置：tiny 用于逐值/双路径；full-width 用于真实 assistant 加载
# ---------------------------------------------------------------------------

def _tiny_target_text_config() -> dict:
    return dict(
        vocab_size=128, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, global_head_dim=16,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=64, rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        layer_types=["sliding_attention", "full_attention",
                     "sliding_attention", "full_attention"],
        num_kv_shared_layers=2, sliding_window=4,
        use_double_wide_mlp=True,
        hidden_size_per_layer_input=4, vocab_size_per_layer_input=128,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional",
                               "rope_theta": 1000000.0,
                               "partial_rotary_factor": 0.25},
        },
    )


def _tiny_assistant_text_config() -> dict:
    return dict(
        vocab_size=128, hidden_size=8, intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, global_head_dim=16,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=64, rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        layer_types=["sliding_attention", "full_attention"],
        num_kv_shared_layers=2, sliding_window=4,
        use_double_wide_mlp=False,
        hidden_size_per_layer_input=0, vocab_size_per_layer_input=0,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional",
                               "rope_theta": 1000000.0,
                               "partial_rotary_factor": 0.25},
        },
    )


def _fullwidth_target_text_config() -> dict:
    return dict(
        vocab_size=262144, hidden_size=1536, intermediate_size=512,
        num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=1,
        head_dim=256, global_head_dim=512,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=128, rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        layer_types=["sliding_attention", "full_attention",
                     "sliding_attention", "full_attention"],
        num_kv_shared_layers=2, sliding_window=512,
        use_double_wide_mlp=True,
        hidden_size_per_layer_input=8, vocab_size_per_layer_input=128,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional",
                               "rope_theta": 1000000.0,
                               "partial_rotary_factor": 0.25},
        },
    )


def _target_config_dict(text: dict) -> dict:
    return {"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4",
            "text_config": text, "tie_word_embeddings": True}


def _assistant_config_dict(text: dict) -> dict:
    return {"architectures": ["Gemma4AssistantForCausalLM"],
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": 16 if text["hidden_size"] == 8 else 1536,
            "use_ordered_embeddings": False, "tie_word_embeddings": True,
            "text_config": text}


def _save_target(directory: Path, text: dict, seed: int) -> dict:
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    cfg = _target_config_dict(text)
    torch.manual_seed(seed)
    hf = Gemma4ForCausalLM(Gemma4TextConfig(**text)).eval()
    sd = {name: tensor.to(torch.bfloat16)
          for name, tensor in hf.state_dict().items() if name != "lm_head.weight"}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(cfg))
    save_file(sd, str(directory / "model.safetensors"))
    return cfg


def _save_assistant(directory: Path, text: dict, seed: int) -> dict:
    cfg = _assistant_config_dict(text)
    torch.manual_seed(seed)
    model = Gemma4MTP(cfg).eval()
    sd = {name: tensor for name, tensor in model.state_dict().items()
          if name != "lm_head.weight"}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(cfg))
    save_file(sd, str(directory / "model.safetensors"))
    return cfg


@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory):
    root = tmp_path_factory.mktemp("gemma4_e2_tiny")
    target = _save_target(root / "target", _tiny_target_text_config(), seed=11)
    assistant = _save_assistant(
        root / "assistant", _tiny_assistant_text_config(), seed=12)
    return root / "target", target, root / "assistant", assistant


@pytest.fixture(scope="module")
def fullwidth_models(tmp_path_factory):
    root = tmp_path_factory.mktemp("gemma4_e2_fullwidth")
    target = _save_target(root / "target", _fullwidth_target_text_config(), seed=13)
    assistant = json.loads((ASSISTANT / "config.json").read_text())
    return root / "target", target, ASSISTANT, assistant


# ---------------------------------------------------------------------------
# 建引擎
# ---------------------------------------------------------------------------

def _make_engine(target_dir, target_hf, draft_dir, draft_hf, *, v2: bool,
                 k: int = 2, mode: str = "none", capture_sizes=None,
                 max_tokens: int = 4, attention_backend: str = "torch",
                 dtype: str | None = None):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if v2 else "0"
    if dtype is None:
        dtype = ("bfloat16"
                 if int(target_hf["text_config"]["hidden_size"]) == 1536 else "float32")
    target = ModelConfig(model=str(target_dir), dtype=dtype,
                         max_model_len=64, hf_config=target_hf)
    draft = ModelConfig(model=str(draft_dir), dtype=dtype,
                        max_model_len=64, hf_config=draft_hf)
    spec = (None if k == 0 else
            SpeculativeConfig(method=None, num_speculative_tokens=k,
                              draft_model_config=draft))
    cfg = VllmConfig(
        model_config=target,
        cache_config=CacheConfig(
            block_size=16 if dtype == "bfloat16" else 4,
            num_gpu_blocks=32 if dtype == "bfloat16" else 64,
            enable_prefix_caching=False),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=32),
        device_config=DeviceConfig(device="cuda"),
        speculative_config=spec,
        compilation_config=CompilationConfig(
            cudagraph_mode=mode,
            cudagraph_capture_sizes=capture_sizes,
            max_cudagraph_capture_size=(
                max(capture_sizes) if capture_sizes else 0)),
        attention_backend=attention_backend,
    )
    engine = LLMEngine(cfg, UniProcExecutor(cfg, Worker(cfg)), tokenizer=None)
    runner = (engine.engine_core.engine_core.model_executor
              .driver_worker.model_runner)
    return engine, runner, max_tokens


def _run(engine, prompts, max_tokens: int):
    for req_id, prompt in prompts:
        engine.add_request(
            req_id, list(prompt),
            SamplingParams(max_tokens=max_tokens, temperature=0.0,
                           eos_token_id=127))
    out = {}
    for _ in range(50):
        if not engine.has_unfinished_requests():
            break
        for item in engine.step():
            out[item.request_id] = list(item.token_ids)
    return out


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

def test_real_assistant_config_is_normalized_without_touching_user_copy():
    raw = json.loads((ASSISTANT / "config.json").read_text())
    draft = ModelConfig(model=str(ASSISTANT), dtype="bfloat16",
                        max_model_len=64, hf_config=raw)
    spec = SpeculativeConfig(method=None, num_speculative_tokens=2,
                             draft_model_config=draft)
    assert spec.method == "mtp"
    assert spec.use_gemma4_mtp() is True
    hf = spec.draft_model_config.hf_config
    assert hf["model_type"] == "gemma4_mtp"
    assert hf["architectures"] == ["Gemma4MTPModel"]
    assert hf["n_predict"] == 1
    assert hf["text_config"]["num_kv_shared_layers"] == 0
    # 用户传进来的对象不被原地改写
    assert raw["model_type"] == "gemma4_assistant"
    assert raw["text_config"]["num_kv_shared_layers"] == 4



def test_conditional_target_checkpoint_layout_loads_text_stack(tiny_models, tmp_path):
    """真实 target checkpoint 的包装命名（`model.language_model.*`）必须能加载。"""
    from minivllm.model_loader import get_model

    _target_dir, cfg, _draft_dir, _draft_cfg = tiny_models
    text = copy.deepcopy(cfg["text_config"])
    wrapped = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": text,
        "tie_word_embeddings": True,
    }
    src = load_file(str(Path(_target_dir) / "model.safetensors"))
    renamed = {
        ("model.language_model." + name[len("model."):]
         if name.startswith("model.") else name): tensor
        for name, tensor in src.items()
    }
    directory = tmp_path / "wrapped"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(wrapped))
    save_file(renamed, str(directory / "model.safetensors"))
    model_config = ModelConfig(
        model=str(directory), dtype="float32", max_model_len=64,
        hf_config=wrapped)
    model = get_model(model_config, "cpu")
    assert model.model.embed_tokens.weight.shape == (128, 16)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    # 首层 MLP 的 gate/up 已被打包，形状与真实 checkpoint 的 gate_proj 不同但数值一致。
    assert model.model.layers[0].mlp.gate_up_proj.weight.shape == (64, 16)


@requires_cuda
def test_request_id_reuse_does_not_keep_stale_gemma4_spec_state(tiny_models):
    target_dir, cfg, draft_dir, draft_cfg = tiny_models
    engine, _, max_tokens = _make_engine(
        target_dir, cfg, draft_dir, draft_cfg, v2=True, k=2)
    _run(engine, [("same", [1, 2, 3])], max_tokens=1)
    reused = _run(engine, [("same", [4, 5, 6, 7])], max_tokens=2)

    fresh, _, max_tokens = _make_engine(
        target_dir, cfg, draft_dir, draft_cfg, v2=True, k=2)
    expected = _run(fresh, [("same", [4, 5, 6, 7])], max_tokens=2)
    assert reused == expected


@requires_cuda
def test_target_logits_match_hf_across_sliding_and_block_boundaries(tiny_models):
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    target_dir, cfg, _, _ = tiny_models
    text = cfg["text_config"]
    engine, runner, max_tokens = _make_engine(
        target_dir, cfg, target_dir, cfg, v2=True, k=0)
    captured = []
    original = runner.sample

    def sample(hidden_states, input_batch, grammar_output):
        h = hidden_states[input_batch.logits_indices]
        captured.append(runner.model.compute_logits(h).detach().clone())
        return original(hidden_states, input_batch, grammar_output)

    runner.sample = sample
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    _run(engine, [("a", prompt)], max_tokens=1)
    assert len(captured) == 1
    ours = captured[0][0]

    hf = Gemma4ForCausalLM(Gemma4TextConfig(**text)).eval()
    sd = load_file(str(Path(target_dir) / "model.safetensors"))
    hf.load_state_dict(sd, strict=False)
    with torch.no_grad():
        ref = hf(torch.tensor([prompt])).logits[0, -1]
    diff = (ours.cpu() - ref).abs().max().item()
    assert diff < 1e-4, f"Gemma4 target logits 与 HF 不一致：max diff={diff}"
    assert ours.argmax().item() == ref.argmax().item()
    assert torch.equal(torch.topk(ours, 5).indices.cpu(),
                       torch.topk(ref, 5).indices)


@requires_cuda
def test_target_groups_and_q_only_sources(tiny_models):
    target_dir, cfg, _, _ = tiny_models
    engine, runner, _ = _make_engine(target_dir, cfg, target_dir, cfg, v2=True, k=0)
    groups = runner.kv_cache_config.kv_cache_groups
    assert len(groups) == 2
    assert runner.packed_backing is not None
    assert runner.packed_backing.numel() == (
        runner.packed_block_stride * runner.kv_cache_config.num_gpu_blocks)
    assert len({cache.untyped_storage().data_ptr()
                for cache in runner.kv_caches.values()}) == 1
    assert groups[0].kv_cache_spec.head_size == 8
    assert groups[0].kv_cache_spec.sliding_window == 4
    assert groups[1].kv_cache_spec.head_size == 16
    assert not hasattr(groups[1].kv_cache_spec, "sliding_window")
    layers = runner._attention_layers()
    # 目标模型自己的 YOCO：2/3 层 Q-only，分别读 0/1 层
    for idx, source_idx in ((2, 0), (3, 1)):
        layer = layers[f"model.layers.{idx}.self_attn.attn"]
        source = layers[f"model.layers.{source_idx}.self_attn.attn"]
        assert layer.q_only is True
        assert layer.kv_cache is source.kv_cache
        assert layer.kv_cache_group_id == source.kv_cache_group_id
        assert layer.kv_sharing_target_layer_name == source.layer_name


@requires_cuda
@requires_assistant
def test_real_assistant_loads_shares_embedding_and_never_writes_source_kv(
        fullwidth_models):
    target_dir, target_hf, assistant_dir, assistant_hf = fullwidth_models
    engine, runner, _ = _make_engine(
        target_dir, target_hf, assistant_dir, assistant_hf, v2=True, k=2)
    speculator = runner.speculator
    model = speculator.model

    # 真实 assistant 的宽度/centroid 特征
    assert model.model.backbone_hidden_size == 1536
    assert model.model.hidden_size == 256
    assert model.model.pre_projection.weight.shape == (256, 3072)
    assert model.model.post_projection.weight.shape == (1536, 256)
    assert model.lm_head.weight.shape == (262144, 256)
    assert model.masked_embedding is not None
    assert model.use_ordered_embeddings is True
    # 只把 target 的 backbone embedding 接过来，lm_head 保持 draft 宽
    assert model.model.embed_tokens is runner.model.model.embed_tokens
    assert model.lm_head.weight is not model.model.embed_tokens.weight

    groups = runner.kv_cache_config.kv_cache_groups
    assert runner.packed_backing is not None
    assert len({cache.untyped_storage().data_ptr()
                for cache in runner.kv_caches.values()}) == 1
    assert [g.kv_cache_spec.head_size for g in groups] == [256, 512]
    assert groups[0].kv_cache_spec.sliding_window == 512
    assert getattr(groups[1].kv_cache_spec, "sliding_window", None) is None

    draft_layers = [
        layer.self_attn.attn
        for layer in model.model.layers
    ]
    # 三个 sliding assistant 层读 target 的 sliding 源层，full 层读 full 源层。
    assert [layer.kv_cache_group_id for layer in draft_layers] == [0, 0, 0, 1]
    for layer in draft_layers:
        source_name = layer.kv_sharing_target_layer_name
        assert source_name is not None
        source = runner._attention_layers()[source_name]
        assert layer.kv_cache is source.kv_cache
        assert layer.head_size == source.head_size

    called = {"n": 0}
    original_propose = speculator.propose

    def propose(*args, **kwargs):
        called["n"] += 1
        # target 本轮已经写完它的 KV；从这里到 propose 返回，assistant 只能读。
        before = {name: cache.clone()
                  for name, cache in runner.kv_caches.items()}
        result = original_propose(*args, **kwargs)
        for name, expected in before.items():
            assert torch.equal(runner.kv_caches[name], expected), (
                f"assistant 的 Q-only 层写脏了 target cache {name!r}")
        return result

    speculator.propose = propose
    out = _run(engine, [("r", [1, 2, 3])], max_tokens=2)
    assert called["n"] > 0
    assert len(out["r"]) == 2


@requires_cuda
def test_v1_and_v2_greedy_outputs_match_on_same_tiny_models(tiny_models):
    target_dir, cfg, draft_dir, draft_cfg = tiny_models
    prompts = [("a", [1, 2, 3, 4, 5]), ("b", [6, 7, 8])]

    engine_v2, _, max_tokens = _make_engine(
        target_dir, cfg, draft_dir, draft_cfg, v2=True, k=2)
    out_v2 = _run(engine_v2, prompts, max_tokens)

    engine_v1, _, max_tokens = _make_engine(
        target_dir, cfg, draft_dir, draft_cfg, v2=False, k=2)
    out_v1 = _run(engine_v1, prompts, max_tokens)

    assert out_v1 == out_v2, (
        f"V1 与 V2 的 Gemma4 生成不一致：V1={out_v1}，V2={out_v2}")


@requires_cuda
def test_v2_eager_and_full_decode_graph_match(tiny_models):
    target_dir, cfg, draft_dir, draft_cfg = tiny_models
    prompts = [("a", [1, 2, 3, 4, 5]), ("b", [6, 7, 8])]
    eager, _, max_tokens = _make_engine(
        target_dir, cfg, draft_dir, draft_cfg, v2=True, k=2)
    out_eager = _run(eager, prompts, max_tokens)

    sizes = [1, 2, 3, 4, 6, 8, 9, 12]
    graph, runner, max_tokens = _make_engine(
        target_dir, cfg, draft_dir, draft_cfg, v2=True, k=2,
        mode="full_decode_only", capture_sizes=sizes)
    out_graph = _run(graph, prompts, max_tokens)
    assert out_graph == out_eager
    stats = runner.cudagraph_capture_stats
    assert stats["graphs"]["prefill"]["captured"] > 0
    assert stats["graphs"]["decode"]["captured"] > 0
    assert stats["target"]["captured"] > 0


@requires_cuda
def test_v1_multi_group_graph_is_rejected_instead_of_using_group0(tiny_models):
    target_dir, cfg, draft_dir, draft_cfg = tiny_models
    with pytest.raises(NotImplementedError, match="V1 的 CUDA Graph"):
        _make_engine(
            target_dir, cfg, draft_dir, draft_cfg, v2=False, k=2,
            mode="full_decode_only", capture_sizes=[3, 6])


@requires_cuda
def test_non_torch_backend_supports_q_only(tiny_models):
    """086：Q-only 共享读取也能走 TRITON_ATTN，而不是静默回 Torch。"""
    target_dir, cfg, draft_dir, draft_cfg = tiny_models
    torch_engine, _torch_runner, max_tokens = _make_engine(
        target_dir, cfg, draft_dir, draft_cfg, v2=True, k=2)
    want = _run(torch_engine, [("r", [1, 2, 3])], max_tokens)
    engine, runner, _ = _make_engine(
        target_dir, cfg, draft_dir, draft_cfg, v2=True, k=2,
        attention_backend="triton")
    got = _run(engine, [("r", [1, 2, 3])], max_tokens)
    assert got == want
    assert runner.attention_backend.get_name() == "TRITON_ATTN"
    shared = [m for _, m in runner.model.named_modules()
              if getattr(m, "q_only", False)]
    assert shared, "tiny Gemma4 应该有 Q-only 共享层"
    assert all(type(m.impl).__name__ == "TritonAttentionImpl" for m in shared)
    assert all(type(m.impl).__name__ == "TritonAttentionImpl"
               for _, m in runner.speculator.model.named_modules()
               if hasattr(m, "impl") and getattr(m, "q_only", False))


@requires_cuda
@requires_assistant
def test_v1_real_assistant_uses_centroid_greedy_graphs(fullwidth_models):
    target_dir, target_hf, assistant_dir, assistant_hf = fullwidth_models
    engine, runner, _ = _make_engine(
        target_dir, target_hf, assistant_dir, assistant_hf, v2=False, k=2)
    proposer = runner.proposer
    assert proposer._centroid_graphs, "真实 assistant 应捕获 centroid get_top_tokens 小图"
    assert 1 in proposer._centroid_graphs and 64 in proposer._centroid_graphs
    out = _run(engine, [("r", [1, 2, 3])], max_tokens=2)
    assert len(out["r"]) == 2


@requires_cuda
@requires_assistant
@requires_real_target
def test_real_target_and_assistant_greedy_graph_matches_target_only():
    """真实 target + 真实 assistant 的最小完整生成对照。

    target-only eager（无投机）与 target+assistant 的 speculative greedy **图**必须逐
    token 相同；这一步同时验证真实 35 层 target 的 loading、sliding/full 两个 group、
    YOCO 20 个 Q-only 尾层和 assistant 的源层映射。
    """
    target_hf = json.loads((REAL_TARGET / "config.json").read_text())
    assistant_hf = json.loads((ASSISTANT / "config.json").read_text())
    prompt = [2, 818, 5279, 529, 7001, 13]

    # 真实权重是 bf16；这里用 float32 做"逐 token 完全相等"的 greedy 对照，避免 bf16
    # 在 near-tie（实测两个 token 的 logits 完全相等）上把 graph/eager 的微小规约差
    # 解释成逻辑错。bf16 的生成路径另有 `requires_assistant` 用例覆盖。
    engine, _runner, _ = _make_engine(
        REAL_TARGET, target_hf, ASSISTANT, assistant_hf, v2=True, k=0,
        dtype="float32")
    target_only = _run(engine, [("r", prompt)], max_tokens=8)
    del engine, _runner
    gc.collect()
    torch.cuda.empty_cache()

    sizes = [1, 3, 6, 9, 12]
    engine, runner, _ = _make_engine(
        REAL_TARGET, target_hf, ASSISTANT, assistant_hf, v2=True, k=2,
        mode="full_decode_only", capture_sizes=sizes, dtype="float32")
    groups = runner.kv_cache_config.kv_cache_groups
    assert [len(g.layer_names) for g in groups] == [12, 3]
    assert groups[0].kv_cache_spec.head_size == 256
    assert groups[1].kv_cache_spec.head_size == 512
    assert groups[0].kv_cache_spec.sliding_window == 512
    layers = runner._attention_layers()
    assert len(layers) == 35
    q_only = [name for name, layer in layers.items() if layer.q_only]
    assert len(q_only) == 20
    for idx in range(15, 35):
        source_idx = 13 if idx % 5 != 4 else 14
        layer = layers[f"model.layers.{idx}.self_attn.attn"]
        source = layers[f"model.layers.{source_idx}.self_attn.attn"]
        assert layer.kv_cache is source.kv_cache
        assert layer.kv_cache_group_id == source.kv_cache_group_id
    draft_layers = [layer.self_attn.attn for layer in runner.speculator.model.model.layers]
    assert [layer.kv_sharing_target_layer_name for layer in draft_layers] == [
        "model.layers.13.self_attn.attn",
        "model.layers.13.self_attn.attn",
        "model.layers.13.self_attn.attn",
        "model.layers.14.self_attn.attn",
    ]

    called = {"n": 0}
    original = runner.speculator.propose

    def propose(*args, **kwargs):
        called["n"] += 1
        before = {name: cache.clone() for name, cache in runner.kv_caches.items()}
        out = original(*args, **kwargs)
        for name, expected in before.items():
            assert torch.equal(runner.kv_caches[name], expected)
        return out

    runner.speculator.propose = propose
    speculative = _run(engine, [("r", prompt)], max_tokens=8)
    assert called["n"] > 0
    assert speculative == target_only
    stats = runner.cudagraph_capture_stats
    assert stats["graphs"]["prefill"]["captured"] > 0
    assert stats["graphs"]["decode"]["captured"] > 0
    assert stats["target"]["captured"] > 0
