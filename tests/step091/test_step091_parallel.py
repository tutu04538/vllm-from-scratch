import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from minivllm.config import ModelConfig, ParallelConfig, SpeculativeConfig, VllmConfig
from minivllm.executor import UniProcExecutor, create_executor, get_executor_class

HF = {"num_attention_heads": 32, "num_key_value_heads": 8, "num_hidden_layers": 36}
MODEL = ModelConfig(model="unused", hf_config=HF)


@pytest.mark.parametrize("tp,pp,dp,world,total", [(1,1,1,1,1),(2,1,2,2,4),(2,2,1,4,4),(4,1,2,4,8)])
def test_layout(tp, pp, dp, world, total):
    p = ParallelConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp, data_parallel_size=dp)
    assert (p.world_size, p.total_gpu_count) == (world, total)
    p.validate_model(MODEL)
    json.dumps(asdict(p))
    if total > 4:
        with pytest.raises(ValueError, match="需要 8 张"):
            p.validate_devices(4)
    else:
        p.validate_devices(4)


@pytest.mark.parametrize("name", ["tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_bad_sizes(name, value):
    with pytest.raises(ValueError):
        ParallelConfig(**{name: value})


@pytest.mark.parametrize("kwargs", [dict(rank=1), dict(local_rank=-1), dict(data_parallel_rank=True), dict(device_ids=[0,0], tensor_parallel_size=2), dict(device_ids=[-1]), dict(device_ids=[True])])
def test_bad_ranks_and_devices(kwargs):
    with pytest.raises(ValueError):
        ParallelConfig(**kwargs)


def test_device_out_of_range():
    with pytest.raises(ValueError, match="超出"):
        ParallelConfig(device_ids=(4,)).validate_devices(4)


@pytest.mark.parametrize("hf,tp,pp", [({**HF, "num_attention_heads": 31},2,1), ({**HF, "num_key_value_heads": 3},2,1), (HF,1,37), ({},2,1)])
def test_bad_model_layout(hf, tp, pp):
    with pytest.raises(ValueError):
        ParallelConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp).validate_model(ModelConfig("x", hf_config=hf))


def test_kv_replication():
    ParallelConfig(tensor_parallel_size=4).validate_model(ModelConfig("x", hf_config={**HF, "num_key_value_heads": 2}))


@pytest.mark.parametrize("kwargs", [dict(prefill_context_parallel_size=2), dict(decode_context_parallel_size=2), dict(nnodes=2), dict(enable_expert_parallel=True), dict(enable_dbo=True), dict(distributed_executor_backend="ray")])
def test_unsupported_config(kwargs):
    with pytest.raises(NotImplementedError):
        ParallelConfig(**kwargs)


@pytest.mark.parametrize("tp,pp,dp", [(2,1,1),(1,2,1),(1,1,2),(2,2,1)])
def test_multi_rejected_before_worker(tp, pp, dp, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    with pytest.raises(NotImplementedError, match="多卡"):
        VllmConfig(MODEL, parallel_config=ParallelConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp, data_parallel_size=dp))


def test_eagle_pp_rejected_before_model_or_worker(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    spec = SpeculativeConfig(method="eagle3", num_speculative_tokens=1, draft_model_config=MODEL)
    with pytest.raises(NotImplementedError, match="EAGLE3.*PP"):
        VllmConfig(MODEL, speculative_config=spec, parallel_config=ParallelConfig(pipeline_parallel_size=2))


def test_external_pp_and_draft_tp():
    with pytest.raises(NotImplementedError, match="external_launcher.*PP"):
        VllmConfig(MODEL, parallel_config=ParallelConfig(pipeline_parallel_size=2, distributed_executor_backend="external_launcher"))
    spec = SpeculativeConfig(method="ngram", num_speculative_tokens=1, draft_tensor_parallel_size=2)
    with pytest.raises(NotImplementedError, match="draft TP"):
        VllmConfig(MODEL, speculative_config=spec)


def test_real_factory_class_and_instance(monkeypatch):
    class Worker:
        def __init__(self, cfg): self.cfg = cfg
        def init_device(self): pass
        def load_model(self): pass
        def shutdown(self): pass
    monkeypatch.setattr("minivllm.executor.uniproc_executor.Worker", Worker)
    cfg = VllmConfig(MODEL)
    assert get_executor_class(cfg) is UniProcExecutor
    executor = create_executor(cfg)
    assert type(executor) is UniProcExecutor
    executor.shutdown()


def test_frozen_source_matches_installed():
    import vllm.config.parallel as upstream
    path = Path("learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/vllm_0_28_0/config/parallel.py")
    assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(Path(upstream.__file__).read_bytes()).digest()
    # 单副本配置不启用 DP 端口分配，避免改变上游进程环境。
    for tp, pp in [(1, 1), (2, 1), (2, 2)]:
        oracle = upstream.ParallelConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp, distributed_executor_backend="mp")
        ours = ParallelConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp)
        assert ours.world_size == oracle.world_size


def test_frozen_draft_tp_oracle():
    from types import SimpleNamespace
    from vllm.config.speculative import SpeculativeConfig as UpstreamSpec
    import vllm.config.speculative as module
    base = Path("learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/vllm_0_28_0")
    assert hashlib.sha256((base / "config/speculative.py").read_bytes()).digest() == hashlib.sha256(Path(module.__file__).read_bytes()).digest()
    upstream_parallel = SimpleNamespace(tensor_parallel_size=4)
    draft = SimpleNamespace(model_type="llama")
    verify = UpstreamSpec._verify_and_get_draft_tp
    assert verify(upstream_parallel, None, draft) == 4
    assert verify(upstream_parallel, 1, draft) == 1
    assert verify(upstream_parallel, 4, draft) == 4
    with pytest.raises(ValueError):
        verify(upstream_parallel, 2, draft)


def test_external_launcher_world_size_matches_frozen_semantics():
    p = ParallelConfig(tensor_parallel_size=2, data_parallel_size=2,
                       distributed_executor_backend="external_launcher")
    assert p.world_size == 4
    assert p.total_gpu_count == 4
    with pytest.raises(NotImplementedError):
        VllmConfig(MODEL, parallel_config=p)


def test_device_selection_not_silently_ignored():
    from minivllm.config import DeviceConfig
    with pytest.raises(ValueError, match='不一致'):
        VllmConfig(MODEL, device_config=DeviceConfig('cuda:1'),
                   parallel_config=ParallelConfig(device_ids=(0,)))
    with pytest.raises(ValueError, match='CUDA'):
        VllmConfig(MODEL, parallel_config=ParallelConfig(device_ids=(0,)))


def test_default_draft_tp_follows_target():
    spec = SpeculativeConfig(method='ngram', num_speculative_tokens=1)
    assert VllmConfig(MODEL, speculative_config=spec).effective_draft_tensor_parallel_size == 1


def test_remaining_frozen_sources_match_installed():
    import vllm.config.vllm as upstream_config
    import vllm.v1.executor.abstract as upstream_executor
    base = Path('learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/vllm_0_28_0')
    for relative, module in [('config/vllm.py', upstream_config),
                             ('v1/executor/abstract.py', upstream_executor)]:
        assert hashlib.sha256((base / relative).read_bytes()).digest() == hashlib.sha256(Path(module.__file__).read_bytes()).digest()


def test_factory_rejects_device_index_before_worker(monkeypatch):
    from minivllm.config import DeviceConfig
    import torch
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    cfg = VllmConfig(MODEL, device_config=DeviceConfig('cuda:1'))
    with pytest.raises(ValueError, match='设备编号'):
        create_executor(cfg)
