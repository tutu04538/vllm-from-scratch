"""step58 测试的共享夹具（tiny 模型现场生成，不依赖 benchmarks/ 下的脚本）。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, Request,  # noqa: E402
                      SamplingParams, SchedulerConfig, SpeculativeConfig, UniProcExecutor,
                      VllmConfig, Worker)
from minivllm.core.kv_cache_manager import KVCacheManager  # noqa: E402
from minivllm.core.sched.scheduler import Scheduler  # noqa: E402

# 59 关起：投机**验证**走 Triton 内核（上游同样只有 GPU 路径），所以跑真引擎的用例要上 GPU。
# CPU 上的算法语义由 `minivllm/testing/torch_rejection_sampler.py` 覆盖（见 tests/step59）。
import torch  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_config(*, tiny_dir, hf_config, spec_k=3, budget=16, blocks=32, prefix=False,
                max_model_len=64, max_num_seqs=2, device=DEVICE, draft_dir=None,
                draft_config=None):
    model = ModelConfig(model=tiny_dir, dtype="float32", max_model_len=max_model_len,
                        hf_config=hf_config)
    spec = None if spec_k is None else SpeculativeConfig(
        method="draft_model", num_speculative_tokens=spec_k,
        draft_model_config=ModelConfig(model=draft_dir or tiny_dir,
                                       dtype="float32", max_model_len=max_model_len,
                                       hf_config=draft_config or hf_config))
    return VllmConfig(
        model_config=model,
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=blocks,
                                 enable_prefix_caching=prefix),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=device), speculative_config=spec)


def make_engine(**kwargs):
    """返回 (engine, core, runner)。调用方负责 engine.shutdown()。"""
    config = make_config(**kwargs)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner


def run_to_end(engine, limit=400):
    final = {}
    for _ in range(limit):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            final[out.request_id] = list(out.token_ids)
    return final


def greedy_outputs(*, spec_k, prompts=(("r", [1, 2, 3, 4, 5, 6]),), max_tokens=6, **kwargs):
    engine, _core, _runner = make_engine(spec_k=spec_k, **kwargs)
    for req_id, prompt in prompts:
        engine.add_request(req_id, list(prompt),
                           SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                          eos_token_id=999))
    outputs = run_to_end(engine)
    engine.shutdown()
    return outputs


class WorkspaceTrace:
    """记录每轮 draft 第一遍的工作区切片（**必须 clone**：视图会被下一次覆盖）。"""

    def __init__(self, runner):
        self.runner = runner
        self.rows = None
        self.forwards = []
        proposer = runner.proposer
        self._propose = proposer.propose
        self._forward = proposer._forward

        def propose(rows, all_token_ids, input_batch, reset_req_ids=None):
            self.rows = list(rows)
            return self._propose(rows, all_token_ids, input_batch, reset_req_ids)

        def forward(num_tokens, num_reqs):
            self.forwards.append({
                "rows": self.rows, "num_tokens": num_tokens, "num_reqs": num_reqs,
                "input_ids": proposer.input_ids_cpu[:num_tokens].clone().tolist(),
                "positions": proposer.positions_cpu[:num_tokens].clone().tolist(),
                "rejected": proposer.is_rejected_token_mask_cpu[:num_tokens].clone().tolist(),
                "slots": proposer.slot_mapping_cpu[:num_tokens].clone().tolist()})
            return self._forward(num_tokens, num_reqs)

        proposer.propose = propose
        proposer._forward = forward

    def restore(self):
        self.runner.proposer.propose = self._propose
        self.runner.proposer._forward = self._forward

    def first_passes(self):
        """每轮第一次 forward = 第一遍（后面的 forward 是自回归步骤）。"""
        out, seen = [], set()
        for entry in self.forwards:
            if id(entry["rows"]) in seen:
                continue
            seen.add(id(entry["rows"]))
            out.append(entry)
        return out


def make_scheduler(*, max_num_seqs=2, max_num_batched_tokens=8, num_gpu_blocks=8,
                   block_size=4, max_model_len=64, policy="fcfs", spec_method="draft_model",
                   k=3):
    spec = None if spec_method is None else SpeculativeConfig(
        method=spec_method, num_speculative_tokens=k,
        draft_model_config=(ModelConfig(model="dummy", max_model_len=max_model_len)
                            if spec_method == "draft_model" else None))
    scheduler_config = SchedulerConfig(max_num_seqs=max_num_seqs,
                                       max_num_batched_tokens=max_num_batched_tokens,
                                       policy=policy)
    kv_cache_manager = KVCacheManager(
        CacheConfig(block_size=block_size, num_gpu_blocks=num_gpu_blocks),
        max_model_len=max_model_len)
    return Scheduler(scheduler_config, kv_cache_manager, max_model_len=max_model_len,
                     speculative_config=spec)


def add_request(scheduler, request_id, prompt_len, priority=0, max_tokens=8):
    request = Request(request_id, list(range(prompt_len)),
                      SamplingParams(max_tokens=max_tokens, eos_token_id=999),
                      arrival_time=1.0, priority=priority)
    scheduler.add_request(request)
    return request
