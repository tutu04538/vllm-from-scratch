"""真实 tiny V2 引擎强制每种接受前缀，验证下轮与非投机输出一致。"""
import json
from pathlib import Path

import pytest
import torch

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,
                      SchedulerConfig, SpeculativeConfig, VllmConfig)
from minivllm.config import CompilationConfig
from minivllm.executor.uniproc_executor import UniProcExecutor
from minivllm.testing.tiny_models import tiny_eagle3_dir, tiny_qwen3_config, tiny_qwen3_dir


def make_engine(monkeypatch, k):
    monkeypatch.setenv('VLLM_USE_V2_MODEL_RUNNER', '1')
    target = tiny_qwen3_dir('tiny_gqa'); hf = tiny_qwen3_config('tiny_gqa')
    spec = None
    if k:
        draft = Path(tiny_eagle3_dir('tiny_gqa'))
        spec = SpeculativeConfig(method='eagle3', num_speculative_tokens=k,
            draft_model_config=ModelConfig(str(draft), dtype='float32', max_model_len=64,
                                           hf_config=json.loads((draft/'config.json').read_text())))
    cfg = VllmConfig(model_config=ModelConfig(target, dtype='float32', max_model_len=64, hf_config=hf),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=64),
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=64),
        device_config=DeviceConfig(device='cuda'), compilation_config=CompilationConfig(cudagraph_mode='none'),
        speculative_config=spec)
    executor = UniProcExecutor(cfg)
    return LLMEngine(cfg, executor, tokenizer=None), executor.driver_worker.model_runner


def generate(engine):
    engine.add_request('A', [1, 2, 3, 4, 5, 6],
                       SamplingParams(max_tokens=16, temperature=0, ignore_eos=True))
    final = None
    for _ in range(100):
        if not engine.has_unfinished_requests(): break
        for output in engine.step(): final = list(output.token_ids)
    assert not engine.has_unfinished_requests()
    assert final is not None and len(final) == 16
    return final


@pytest.mark.parametrize('k,r', [(k, r) for k in (1, 2, 4) for r in range(k + 1)])
def test_forced_reject_full_tiny_engine(monkeypatch, k, r):
    assert torch.cuda.is_available(), '需要 CUDA 验收'
    baseline, _ = make_engine(monkeypatch, None)
    try:
        target = generate(baseline)
    finally:
        baseline.shutdown()
    engine, runner = make_engine(monkeypatch, k)
    original_propose = runner.speculator.propose
    original_sample = runner.sample
    sampled = []
    calls = 0

    def forced(*args, **kwargs):
        nonlocal calls
        output = original_propose(*args, **kwargs)
        if calls == 0:
            candidates = list(target[1:1+k])
            if r < k: candidates[r] = (candidates[r] + 1) % 11
            output[0, :k] = torch.tensor(candidates, dtype=output.dtype, device=output.device)
        calls += 1
        return output

    def sample(*args, **kwargs):
        output = original_sample(*args, **kwargs)
        batch = args[1]
        sampled.append((output[1].tolist(), output[2].tolist(),
                        batch.positions[:batch.num_tokens].tolist()))
        return output

    runner.speculator.propose = forced
    runner.sample = sample
    try:
        actual = generate(engine)
    finally:
        engine.shutdown()
    assert calls >= 2 and len(sampled) >= 2
    assert sampled[1][:2] == ([r + 1], [k - r])
    assert len(sampled) >= 3
    assert sampled[2][2][0] == sampled[1][2][0] + r + 1
    assert actual == target
