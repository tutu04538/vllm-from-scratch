"""80F：Step3p5 MTP 模型层与冻结上游的逐值差分。

固定 tiny 配置：
  target 4 层（full/sliding 混合）
  MTP 2 层（绝对层号 4/5）
  hidden=32, heads=4/2, head_dim=8, partial RoPE 0.5/1.0, head-wise gate

本文件的模型层测试只检查形状/权重路由与真实端到端；真实 cache 读取由 V1/V2
端到端和 80D 契约覆盖。冻结逐层 hidden/logits 差分不在本文件用替身伪造。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from minivllm.models.step3p5 import Step3p5ForCausalLM, Step3p5MTP

CFG = {
    "architectures": ["Step3p5ForCausalLM"], "model_type": "step3p5",
    "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
    "num_hidden_layers": 4, "num_attention_heads": 4, "num_attention_groups": 2,
    "head_dim": 8, "max_position_embeddings": 64, "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "layer_types": ["full_attention", "sliding_attention", "sliding_attention",
                    "full_attention", "sliding_attention", "full_attention"],
    "sliding_window": 8,
    "partial_rotary_factors": [0.5, 1.0, 1.0, 0.5, 1.0, 1.0],
    "use_head_wise_attn_gate": True, "use_rope_layers": [],
    "yarn_only_types": ["full_attention"], "attention_other_setting": None,
    "use_moe": False, "moe_layers_enum": "",
    "moe_num_experts": 4, "moe_top_k": 2, "moe_intermediate_size": 16,
    "share_expert_dim": 16, "moe_router_activation": "sigmoid",
    "moe_router_scaling_factor": 1.0, "norm_expert_weight": True,
    "swiglu_limits": [0] * 6, "swiglu_limits_shared": [0] * 6,
    "num_nextn_predict_layers": 2, "tie_word_embeddings": False,
    "use_qk_norm": True,
}


def _split_qkv(name, weight):
    q, k, v = weight.split([4 * 8, 2 * 8, 2 * 8], dim=0)
    base = name.replace(".qkv_proj.weight", "")
    return {base + ".q_proj.weight": q, base + ".k_proj.weight": k,
            base + ".v_proj.weight": v}


def _split_gu(name, weight):
    gate, up = weight.split([64, 64], dim=0)
    base = name.replace(".gate_up_proj.weight", "")
    return {base + ".gate_proj.weight": gate, base + ".up_proj.weight": up}


def _make_state() -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    target = Step3p5ForCausalLM(CFG)
    mtp = Step3p5MTP(CFG)
    for model in (target, mtp):
        for parameter in model.parameters():
            nn.init.normal_(parameter, mean=0.0, std=0.02)
    state: dict[str, torch.Tensor] = {}
    for name, parameter in target.named_parameters():
        if name.endswith(".qkv_proj.weight"):
            state.update(_split_qkv(name, parameter.detach()))
        elif name.endswith(".gate_up_proj.weight"):
            state.update(_split_gu(name, parameter.detach()))
        else:
            state[name] = parameter.detach().clone()
    for name, parameter in mtp.named_parameters():
        if name == "model.embed_tokens.weight":
            continue
        if name.endswith(".qkv_proj.weight"):
            for key, value in _split_qkv(name, parameter.detach()).items():
                state[key.replace(".mtp_block.", ".")] = value
        elif name.endswith(".gate_up_proj.weight"):
            for key, value in _split_gu(name, parameter.detach()).items():
                state[key.replace(".mtp_block.", ".")] = value
        elif ".mtp_block." in name:
            state[name.replace(".mtp_block.", ".")] = parameter.detach().clone()
        elif ".shared_head.head." in name:
            state[name.replace(".shared_head.head.",
                               ".transformer.shared_head.output.")] = parameter.detach().clone()
        elif ".shared_head.norm." in name:
            state[name.replace(".shared_head.norm.",
                               ".transformer.shared_head.norm.")] = parameter.detach().clone()
        else:
            state[name] = parameter.detach().clone()
    return state


def _make_ours() -> Step3p5MTP:
    state = _make_state()
    ours = Step3p5MTP(CFG)
    ours.load_weights(iter(state.items()))
    ours.eval()
    return ours


def test_parameter_shapes_and_step_heads():
    ours = _make_ours()
    assert list(ours.model.layers.keys()) == ["4", "5"]
    layer = ours.model.layers["4"]
    assert layer.mtp_block.self_attn.num_heads == 4
    assert layer.mtp_block.self_attn.num_kv_heads == 2
    assert layer.mtp_block.self_attn.head_dim == 8
    assert layer.mtp_block.self_attn.g_proj is not None
    hidden = torch.randn(3, 32)
    logits0 = ours.compute_logits(hidden, spec_step_idx=0)
    logits1 = ours.compute_logits(hidden, spec_step_idx=1)
    assert logits0.shape == (3, 64) and logits1.shape == (3, 64)
    assert not torch.allclose(logits0, logits1)


def test_weight_loader_routes_real_checkpoint_names():
    state = _make_state()
    ours = _make_ours()
    layer = ours.model.layers["4"]
    # q/k/v 打包进 qkv_proj 的三段
    torch.testing.assert_close(
        layer.mtp_block.self_attn.qkv_proj.weight[:32],
        state["model.layers.4.self_attn.q_proj.weight"])
    torch.testing.assert_close(
        layer.mtp_block.self_attn.qkv_proj.weight[32:48],
        state["model.layers.4.self_attn.k_proj.weight"])
    # gate/up 打包
    torch.testing.assert_close(
        layer.mtp_block.mlp.gate_up_proj.weight[:64],
        state["model.layers.4.mlp.gate_proj.weight"])
    # shared_head.output -> shared_head.head、.transformer. 去掉
    torch.testing.assert_close(
        layer.shared_head.head.weight,
        state["model.layers.4.transformer.shared_head.output.weight"])
    torch.testing.assert_close(
        layer.enorm.weight, state["model.layers.4.enorm.weight"])


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="V1 拒绝采样走 Triton，需要 CUDA")
def test_v1_step3p5_spec_matches_plain_on_tiny_checkpoint(tmp_path, monkeypatch):
    """V1 端到端：tiny Step3p5 target + 2 层 MTP，投机 greedy == 非投机。"""
    import json
    from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig,
                          SamplingParams, SchedulerConfig, SpeculativeConfig,
                          UniProcExecutor, VllmConfig, Worker)
    from minivllm.config import CompilationConfig
    from minivllm.models.step3p5 import Step3p5MTP as _MTP  # noqa: F401

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    state = _make_state()
    save_file(state, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(CFG, indent=2))

    def run(spec):
        config = VllmConfig(
            model_config=ModelConfig(model=str(tmp_path), dtype="float32",
                                     max_model_len=64, hf_config=CFG),
            cache_config=CacheConfig(block_size=4, num_gpu_blocks=32),
            scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=64),
            device_config=DeviceConfig(device="cuda"),
            speculative_config=spec,
            compilation_config=CompilationConfig(cudagraph_mode="none"))
        engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
        runner = engine.engine_core.engine_core.model_executor.driver_worker.model_runner
        engine.add_request("a", [1, 2, 3, 4, 5],
                           SamplingParams(max_tokens=8, temperature=0.0,
                                          eos_token_id=999))
        tokens = []
        for _ in range(30):
            if not engine.has_unfinished_requests():
                break
            output = engine.step()
            if output:
                tokens = output[0].token_ids
        return tokens, runner

    plain, _ = run(None)
    for k in (1, 2, 4):
        speculative, runner = run(SpeculativeConfig(
            method="mtp", num_speculative_tokens=k, draft_model_config=None))
        assert speculative == plain, f"K={k} 投机输出偏离非投机"
        assert runner.proposer.num_drafts_proposed > 0
        # MTP 层 4=sliding、5=full，必须映射到两个不同的 target group。
        assert set(runner.proposer._draft_layer_groups.values()) == {0, 1}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="V2 需要 CUDA + UVA")
def test_v2_step3p5_spec_matches_plain_and_graph(tmp_path, monkeypatch):
    """V2：K=1 走 MTPSpeculator，K=2/4 走 MultiModuleMTPSpeculator；eager/graph 均无损。"""
    import json
    from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig,
                          SamplingParams, SchedulerConfig, SpeculativeConfig,
                          UniProcExecutor, VllmConfig, Worker)
    from minivllm.config import CompilationConfig

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
    state = _make_state()
    save_file(state, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(CFG, indent=2))

    def run(k: int, graph: bool):
        target = ModelConfig(model=str(tmp_path), dtype="float32",
                             max_model_len=64, hf_config=CFG)
        if k == 0:
            spec = None
        else:
            draft = SpeculativeConfig(
                method="mtp", num_speculative_tokens=k,
                draft_model_config=None).derive_mtp_draft_config(target)
            spec = SpeculativeConfig(
                method="mtp", num_speculative_tokens=k, draft_model_config=draft)
        config = VllmConfig(
            model_config=target,
            cache_config=CacheConfig(block_size=4, num_gpu_blocks=32),
            scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=64),
            device_config=DeviceConfig(device="cuda"),
            speculative_config=spec,
            compilation_config=CompilationConfig(
                cudagraph_mode="full_decode_only" if graph else "none"))
        engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
        runner = engine.engine_core.engine_core.model_executor.driver_worker.model_runner
        seen_steps = []
        if spec is not None:
            original_logits = runner.speculator.model.compute_logits

            def logits_spy(hidden_states, *args, **kwargs):
                seen_steps.append(int(kwargs.get("spec_step_idx", 0)))
                return original_logits(hidden_states, *args, **kwargs)

            runner.speculator.model.compute_logits = logits_spy
        engine.add_request("a", [1, 2, 3, 4, 5],
                           SamplingParams(max_tokens=8, temperature=0.0,
                                          eos_token_id=999))
        tokens = []
        for _ in range(30):
            if not engine.has_unfinished_requests():
                break
            output = engine.step()
            if output:
                tokens = output[0].token_ids
        return tokens, runner, seen_steps

    plain, _, _ = run(0, graph=False)
    for k, expected_type in ((1, "MTPSpeculator"),
                             (2, "MultiModuleMTPSpeculator"),
                             (4, "MultiModuleMTPSpeculator")):
        tokens, runner, seen = run(k, graph=False)
        assert tokens == plain, f"V2 K={k} eager 偏离 plain"
        assert type(runner.speculator).__name__ == expected_type
        assert set(range(k)) <= set(seen), (
            f"V2 K={k} 采样没有按 spec_step_idx 选 head：seen={sorted(set(seen))}")
    graph_tokens, graph_runner, _ = run(2, graph=True)
    assert graph_tokens == plain, "V2 K=2 graph 偏离 plain"
    assert graph_runner.speculator.cudagraph_mode.name == "FULL_DECODE_ONLY"
