"""88：真实 Gemma4 target/assistant 在拒绝发生后的逐轮差分。

83 只记录了 assistant 首次拒绝后的输入，没有把它和另一条执行路径逐轮比较。
这里固定真实 target/assistant revision、随机请求和 seed，分别跑 eager 与 FULL
decode Graph，比较：

    - target 每个验证步的 hidden / logits（Graph 静态缓冲在 sample 入口 clone）；
    - assistant 每轮 propose 的输入 hidden / positions / input ids；
    - 每轮交出的 draft_tokens 与 num_rejected。

测试要求至少有 1 次拒绝；不把“最终 token 一样”当作 assistant 路径等价的唯一证据。
没有 CUDA 或真实权重时整组 skip。
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import pytest
import torch

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig,
                      SamplingParams, SchedulerConfig, SpeculativeConfig,
                      UniProcExecutor, VllmConfig, Worker)
from minivllm.config import CompilationConfig

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "models" / "gemma-4-E2B-it"
ASSISTANT = ROOT / "models" / "gemma-4-E2B-it-assistant"
PROMPT = [2, 818, 5279, 529, 7001, 13]
requires_real = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not (TARGET / "model.safetensors").is_file()
    or not (ASSISTANT / "model.safetensors").is_file(),
    reason="需要本机真实 Gemma4 target/assistant 权重与 CUDA")


def _make_engine(*, mode: str, capture_sizes=(1, 2, 3, 4, 8)):
    os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    target_hf = json.loads((TARGET / "config.json").read_text())
    assistant_hf = json.loads((ASSISTANT / "config.json").read_text())
    cfg = VllmConfig(
        model_config=ModelConfig(
            model=str(TARGET), dtype="bfloat16", max_model_len=2048,
            hf_config=target_hf),
        cache_config=CacheConfig(
            block_size=16, num_gpu_blocks=1024, enable_prefix_caching=False),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1, max_num_batched_tokens=128),
        device_config=DeviceConfig(device="cuda"),
        speculative_config=SpeculativeConfig(
            method=None, num_speculative_tokens=2,
            draft_model_config=ModelConfig(
                model=str(ASSISTANT), dtype="bfloat16",
                max_model_len=2048, hf_config=assistant_hf)),
        compilation_config=CompilationConfig(
            cudagraph_mode=mode,
            cudagraph_capture_sizes=(list(capture_sizes)
                                     if mode != "none" else None),
            max_cudagraph_capture_size=(
                max(capture_sizes) if mode != "none" else 0)),
        attention_backend="triton",
    )
    engine = LLMEngine(
        cfg, UniProcExecutor(cfg, Worker(cfg)), tokenizer=None)
    runner = (engine.engine_core.engine_core.model_executor
              .driver_worker.model_runner)
    return engine, runner


class CaptureTargetSamples:
    """在 sample 入口抓 target 本轮 hidden/logits。"""

    def __init__(self, runner):
        self.runner = runner
        self.records = []
        self._original = runner.sample

        def sample(hidden_states, input_batch, grammar_output):
            n = int(input_batch.num_tokens)
            hidden = hidden_states[:n].detach().clone().cpu()
            sample_hidden = hidden_states[input_batch.logits_indices]
            logits = runner.model.compute_logits(sample_hidden).detach().clone().cpu()
            self.records.append({"hidden": hidden, "logits": logits})
            return self._original(hidden_states, input_batch, grammar_output)

        runner.sample = sample


class CaptureAssistantRounds:
    """记录 assistant 每轮 propose 的输入与输出；Graph 路径也走这里。"""

    def __init__(self, runner):
        self.speculator = runner.speculator
        self.records = []
        self._original = self.speculator.propose

        def propose(input_batch, attn_metadata, slot_mappings, *args, **kwargs):
            num_tokens = int(input_batch.num_tokens)
            num_reqs = int(input_batch.num_reqs)
            last_hidden = args[0] if len(args) > 0 else kwargs.get(
                "last_hidden_states")
            num_rejected = args[3] if len(args) > 3 else kwargs.get(
                "num_rejected")
            input_ids = getattr(input_batch, "input_ids", None)
            positions = getattr(input_batch, "positions", None)
            before = {
                "target_step": len(self.records),
                "num_tokens": num_tokens,
                "num_reqs": num_reqs,
                "input_ids": (input_ids[:num_tokens].detach().to(torch.int64).cpu()
                              if isinstance(input_ids, torch.Tensor) else None),
                "positions": (positions[:num_tokens].detach().to(torch.int64).cpu()
                              if isinstance(positions, torch.Tensor) else None),
                "input_hidden": (last_hidden[:num_tokens].detach().float().cpu()
                                 if isinstance(last_hidden, torch.Tensor) else None),
                "num_rejected": (
                    num_rejected.detach().to(torch.int64).cpu().tolist()
                    if isinstance(num_rejected, torch.Tensor) else num_rejected),
            }
            result = self._original(
                input_batch, attn_metadata, slot_mappings, *args, **kwargs)
            before["draft_tokens"] = (
                result[:num_reqs].detach().to(torch.int64).cpu()
                if isinstance(result, torch.Tensor) else None)
            before["current_draft_step"] = int(
                self.speculator.current_draft_step.item())
            self.records.append(before)
            return result

        self.speculator.propose = propose


def _run(engine, *, max_tokens=8):
    engine.add_request(
        "r", list(PROMPT),
        SamplingParams(max_tokens=max_tokens, temperature=1.0, seed=0,
                       ignore_eos=True))
    tokens = []
    for _ in range(400):
        if not engine.has_unfinished_requests():
            break
        for output in engine.step():
            tokens = list(output.token_ids)
    return tokens


def _close(engine):
    engine.shutdown()
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def _assert_target_close(eager, graph):
    assert len(eager.records) == len(graph.records), (
        f"target 采样步数不同：{len(eager.records)} vs {len(graph.records)}")
    for step, (left, right) in enumerate(zip(eager.records, graph.records)):
        assert left["hidden"].shape == right["hidden"].shape
        torch.testing.assert_close(
            left["hidden"], right["hidden"], rtol=2e-2, atol=2e-2)
        assert left["logits"].shape == right["logits"].shape
        torch.testing.assert_close(
            left["logits"], right["logits"], rtol=2e-2, atol=2e-2)
        assert (left["logits"].argmax(-1)
                == right["logits"].argmax(-1)).all(), f"step {step} argmax 不同"


def _assert_assistant_close(eager, graph):
    assert len(eager.records) == len(graph.records), (
        f"assistant 轮数不同：{len(eager.records)} vs {len(graph.records)}")
    rejected = 0
    for step, (left, right) in enumerate(zip(eager.records, graph.records)):
        assert left["num_tokens"] == right["num_tokens"], f"step {step} 行数"
        assert left["num_reqs"] == right["num_reqs"]
        if left["input_ids"] is not None:
            assert torch.equal(left["input_ids"], right["input_ids"]), (
                f"step {step} assistant input ids 不同")
        if left["positions"] is not None:
            assert torch.equal(left["positions"], right["positions"]), (
                f"step {step} assistant positions 不同")
        if left["input_hidden"] is not None:
            torch.testing.assert_close(
                left["input_hidden"], right["input_hidden"],
                rtol=2e-2, atol=2e-2)
        assert left["num_rejected"] == right["num_rejected"], (
            f"step {step} num_rejected 不同："
            f"{left['num_rejected']} vs {right['num_rejected']}")
        if left["draft_tokens"] is not None:
            assert torch.equal(left["draft_tokens"], right["draft_tokens"]), (
                f"step {step} draft_tokens 不同")
        rejected += sum(left["num_rejected"] or [])
    assert rejected > 0, "随机请求没有触发拒绝，测试没有覆盖拒绝后路径"


@requires_real
def test_real_assistant_rounds_after_rejection_match_eager():
    eager, eager_runner = _make_engine(mode="none")
    eager_target = CaptureTargetSamples(eager_runner)
    eager_assistant = CaptureAssistantRounds(eager_runner)
    eager_tokens = _run(eager)
    eager_rounds = list(eager_assistant.records)
    _close(eager)

    graph, graph_runner = _make_engine(mode="full")
    graph_target = CaptureTargetSamples(graph_runner)
    graph_assistant = CaptureAssistantRounds(graph_runner)
    graph_tokens = _run(graph)
    graph_rounds = list(graph_assistant.records)
    try:
        assert graph_tokens == eager_tokens, (
            f"随机输出不同：eager={eager_tokens} graph={graph_tokens}")
        assert eager_rounds and graph_rounds
        _assert_target_close(eager_target, graph_target)
        _assert_assistant_close(eager_assistant, graph_assistant)
        assert graph_runner.target_cudagraph_manager.num_replays > 0
        speculator = graph_runner.speculator
        assert speculator.prefill_cudagraph_manager.num_replays > 0
        assert speculator.decode_cudagraph_manager.num_replays > 0
    finally:
        _close(graph)
