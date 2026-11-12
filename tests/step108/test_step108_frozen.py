"""The installed differential oracle must be the recorded vLLM source."""
import hashlib
from pathlib import Path


def test_frozen_speculation_sources():
    from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler, rejection_sampler_utils
    from vllm.v1.worker.gpu import model_runner
    from vllm.v1.core.sched import scheduler
    root = Path('learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/vllm_0_28_0')
    files = ((speculator, 'v1/worker/gpu/spec_decode/autoregressive/speculator.py'),
             (rejection_sampler, 'v1/worker/gpu/spec_decode/rejection_sampler.py'),
             (rejection_sampler_utils, 'v1/worker/gpu/spec_decode/rejection_sampler_utils.py'),
             (model_runner, 'v1/worker/gpu/model_runner.py'),
             (scheduler, 'v1/core/sched/scheduler.py'))
    for module, relative in files:
        assert hashlib.sha256(Path(module.__file__).read_bytes()).digest() == hashlib.sha256((root/relative).read_bytes()).digest()
