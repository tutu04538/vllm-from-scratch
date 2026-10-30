"""087 测试辅助：tiny 图引擎、真实 Gemma4 图引擎、逐轮 hidden/logits 记录。

真实模型路径与 086 保持一致（固定 revision 的 target/assistant）；tiny 用项目自带的
Qwen3/EAGLE3 夹具，只验证 target 的 Graph/metadata 边界，不把 EAGLE 草稿层说成
已经在 Triton 上。
"""

from __future__ import annotations

import gc
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path

os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minivllm import (  # noqa: E402
    CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,
    SchedulerConfig, SpeculativeConfig, UniProcExecutor, VllmConfig, Worker,
)
from minivllm.config import CompilationConfig  # noqa: E402
from minivllm.testing.tiny_models import (  # noqa: E402
    tiny_eagle3_dir, tiny_qwen3_config, tiny_qwen3_dir,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Graph/metadata 集成测试需要 CUDA")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET = ROOT / "models" / "gemma-4-E2B-it"
ASSISTANT = ROOT / "models" / "gemma-4-E2B-it-assistant"
REAL_PROMPT = [2, 818, 5279, 529, 7001, 13]
REAL_SHORT_PROMPT = [2, 818, 5279]

requires_models = pytest.mark.skipif(
    not (TARGET / "model.safetensors").is_file()
    or not (ASSISTANT / "model.safetensors").is_file(),
    reason="需要本机真实 Gemma4 target/assistant 权重")

# tiny Qwen3 夹具（与 86 的 tiny 路径相同）
TINY = tiny_qwen3_dir("tiny_gqa")
HF = tiny_qwen3_config("tiny_gqa")
DRAFT = tiny_eagle3_dir("tiny_gqa")
DRAFT_HF = json.loads((Path(DRAFT) / "config.json").read_text())
TINY_VOCAB = int(HF["vocab_size"])
TINY_EOS = TINY_VOCAB + 7


def make_tiny_engine(*, mode="none", spec_k=2, capture_sizes=(1, 2, 3, 4, 8, 16),
                     max_num_seqs=2, blocks=64, budget=64,
                     max_model_len=64):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    spec = None
    if spec_k is not None:
        spec = SpeculativeConfig(
            method="eagle3", num_speculative_tokens=spec_k,
            draft_sample_method="greedy",
            draft_model_config=ModelConfig(
                model=DRAFT, dtype="float32", max_model_len=max_model_len,
                hf_config=DRAFT_HF))
    cfg = VllmConfig(
        model_config=ModelConfig(
            model=TINY, dtype="float32", max_model_len=max_model_len,
            hf_config=HF),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=blocks),
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs, max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=DEVICE),
        speculative_config=spec,
        compilation_config=CompilationConfig(
            cudagraph_mode=mode,
            cudagraph_capture_sizes=list(capture_sizes) if capture_sizes else None,
            max_cudagraph_capture_size=(
                max(capture_sizes) if capture_sizes else 0)),
        attention_backend="triton",
    )
    engine = LLMEngine(
        cfg, UniProcExecutor(cfg, Worker(cfg)), tokenizer=None)
    core = engine.engine_core.engine_core
    runner = core.model_executor.driver_worker.model_runner
    return engine, runner


def make_real_engine(*, k=2, mode="none",
                     capture_sizes=(1, 2, 3, 4, 8),
                     max_num_seqs=1, blocks=1024, max_model_len=2048):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    target_hf = json.loads((TARGET / "config.json").read_text())
    assistant_hf = json.loads((ASSISTANT / "config.json").read_text())
    target = ModelConfig(
        model=str(TARGET), dtype="bfloat16", max_model_len=max_model_len,
        hf_config=target_hf)
    spec = None
    if k:
        draft = ModelConfig(
            model=str(ASSISTANT), dtype="bfloat16", max_model_len=max_model_len,
            hf_config=assistant_hf)
        spec = SpeculativeConfig(
            method=None, num_speculative_tokens=k, draft_model_config=draft)
    cfg = VllmConfig(
        model_config=target,
        cache_config=CacheConfig(
            block_size=16, num_gpu_blocks=blocks, enable_prefix_caching=False),
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs, max_num_batched_tokens=128),
        device_config=DeviceConfig(device="cuda"),
        speculative_config=spec,
        compilation_config=CompilationConfig(
            cudagraph_mode=mode,
            cudagraph_capture_sizes=list(capture_sizes) if capture_sizes else None,
            max_cudagraph_capture_size=(
                max(capture_sizes) if capture_sizes else 0)),
        attention_backend="triton",
    )
    engine = LLMEngine(
        cfg, UniProcExecutor(cfg, Worker(cfg)), tokenizer=None)
    core = engine.engine_core.engine_core
    runner = core.model_executor.driver_worker.model_runner
    return engine, runner


def run_engine(engine, prompts: Iterable[tuple[str, list[int]]], *,
               max_tokens=4, temperature=0.0, seed=None):
    for req_id, prompt in prompts:
        engine.add_request(
            req_id, list(prompt),
            SamplingParams(max_tokens=max_tokens, temperature=temperature,
                           seed=seed, ignore_eos=True))
    outputs: dict[str, list[int]] = {}
    for _ in range(200):
        if not engine.has_unfinished_requests():
            break
        for item in engine.step():
            outputs[item.request_id] = list(item.token_ids)
    return outputs


def close_engine(engine) -> None:
    engine.shutdown()
    del engine
    gc.collect()
    torch.cuda.empty_cache()


class CaptureTargetHiddens:
    """记录 target `sample()` 收到的 hidden 与它算出的 logits。

    记录点必须在 `sample()` 入口（`execute_model_state` 清空之前）：Graph 重放写的是
    静态 hidden 缓冲，只有在这一刻 clone 才能拿到本轮内容。只记录**真实 token 行**，
    不把图补齐行算进差分。
    """

    def __init__(self, runner) -> None:
        self.runner = runner
        self.records: list[dict[str, torch.Tensor]] = []
        self._original = runner.sample

        def sample(hidden_states, input_batch, grammar_output):
            n = int(input_batch.num_tokens)
            hidden = hidden_states[:n].detach().clone().cpu()
            sample_hidden = hidden_states[input_batch.logits_indices]
            logits = runner.model.compute_logits(
                sample_hidden).detach().clone().cpu()
            self.records.append({"hidden": hidden, "logits": logits})
            return self._original(hidden_states, input_batch, grammar_output)

        runner.sample = sample


class DraftRoundRecorder:
    """记录每轮 `propose()` 后的草稿状态，用于 eager/Graph 逐轮差分。

    这里不复制 propose 的算法，只在外部包一层：被比较的是生产路径真实交出的
    `draft_tokens`、下一步 positions/seq_lens、最终 slot_mapping 和 draft step。
    """

    def __init__(self, runner) -> None:
        self.runner = runner
        self.speculator = runner.speculator
        self.rounds: list[dict[str, torch.Tensor | int]] = []
        self._original = self.speculator.propose

        def propose(*args, **kwargs):
            input_batch = args[0]
            prefill_manager = self.speculator.prefill_cudagraph_manager
            decode_manager = self.speculator.decode_cudagraph_manager
            before_prefill = prefill_manager.num_replays if prefill_manager else 0
            before_decode = decode_manager.num_replays if decode_manager else 0
            out = self._original(*args, **kwargs)
            n = input_batch.num_reqs
            self.rounds.append({
                "draft_tokens": out[:n].detach().clone().cpu(),
                "positions": self.speculator.input_buffers.positions[:n]
                .detach().clone().cpu(),
                "seq_lens": self.speculator.input_buffers.seq_lens[:n]
                .detach().clone().cpu(),
                "slot_mappings": self.runner.block_tables.slot_mappings[:, :n]
                .detach().clone().cpu(),
                "current_step": int(self.speculator.current_draft_step.item()),
                "has_prefill": bool(input_batch.has_prefill),
                "prefill_replays": (
                    prefill_manager.num_replays - before_prefill
                    if prefill_manager else 0),
                "decode_replays": (
                    decode_manager.num_replays - before_decode
                    if decode_manager else 0),
            })
            return out

        self.speculator.propose = propose
