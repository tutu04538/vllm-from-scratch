import pytest
import torch
from minivllm.config import CacheConfig, CompilationConfig, DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig
from minivllm.engine.llm_engine import LLMEngine
from minivllm.executor import UniProcExecutor
from minivllm.sampling_params import SamplingParams
from minivllm.testing.tiny_models import tiny_qwen3_config, write_tiny_qwen3


@pytest.mark.parametrize('v2', [False, True])
def test_factory_real_gpu_engine(tmp_path, monkeypatch, v2):
    assert torch.cuda.device_count() >= 1
    monkeypatch.setenv('VLLM_USE_V2_MODEL_RUNNER', '1' if v2 else '0')
    path = write_tiny_qwen3('tiny_gqa', tmp_path)
    cfg = VllmConfig(ModelConfig(str(path), hf_config=tiny_qwen3_config('tiny_gqa'), max_model_len=64),
                     cache_config=CacheConfig(block_size=16, num_gpu_blocks=8),
                     scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=16),
                     device_config=DeviceConfig('cuda'),
                     compilation_config=CompilationConfig(cudagraph_mode='none'))
    engine = LLMEngine(cfg)
    try:
        executor = engine.engine_core.engine_core.model_executor
        assert type(executor) is UniProcExecutor
        assert next(executor.driver_worker.model_runner.model.parameters()).device.type == 'cuda'
        engine.add_request('gpu', [1, 2, 3], SamplingParams(max_tokens=2, temperature=0.0))
        outputs = []
        while engine.has_unfinished_requests():
            outputs.extend(engine.step())
        assert len(outputs[-1].token_ids) == 2
    finally:
        engine.shutdown()


def test_real_visible_device_budget_rejected_before_worker():
    from minivllm.config import ParallelConfig
    count = torch.cuda.device_count()
    assert 0 < count < 8
    parallel = ParallelConfig(tensor_parallel_size=4, data_parallel_size=2)
    with pytest.raises(ValueError, match='需要 8 张 GPU'):
        VllmConfig(ModelConfig('unused', hf_config={
            'num_attention_heads': 32, 'num_key_value_heads': 8, 'num_hidden_layers': 36}),
            device_config=DeviceConfig('cuda'), parallel_config=parallel)
