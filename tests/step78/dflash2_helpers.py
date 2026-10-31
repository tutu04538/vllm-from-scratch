"""step78 的测试夹具：DFlash2 的模型、提议者、候选 walk 与引擎。

三块：

**A. 模型/提议者**（`tiny_pair` / `make_speculator`）  同一份 tiny target + 一份 tiny DFlash2
   草稿（进程内缓存）。提议者只走 `__init__ → load_draft_model()`，不建引擎，于是测试可以
   **逐位摆布**候选/边分/采样缓冲。

**B. walk 夹具**（`set_walk_inputs`）  直接填 `sample_pos` / `sample_idx_mapping` /
   `temperature` / `seeds`（这些就是内核 `_prepare_dflash_inputs_kernel` 的输出）。

**C. 引擎**（`make_engine` / `run_prompts`）  真正的 V2 通路（`VLLM_USE_V2_MODEL_RUNNER=1`）。

为什么 tiny 目录要缓存：`tiny_dflash2_dir()` 每次写新目录，而"同一份权重"是差分测试的前提。
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig,  # noqa: E402
                      SamplingParams, SchedulerConfig, SpeculativeConfig,
                      UniProcExecutor, VllmConfig, Worker)
from minivllm.testing.tiny_models import (tiny_dflash2_dir, tiny_qwen3_config,  # noqa: E402
                                          tiny_qwen3_dir)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_PAIR: dict[tuple, tuple[str, dict, str, dict]] = {}


@contextlib.contextmanager
def v2_env():
    """只在建配置/建引擎期间打开 V2 入口（DFlash2 只有 V2）。

    不能在模块顶层写死环境变量：一次 pytest 跑多个目录时 import 是**同一个进程**，顶层改动
    会把 step58~76 的 V1 用例全部切到 V2。
    """
    previous = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
        else:
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = previous


def tiny_pair(*, k: int = 3, num_layers: int = 2, causal: bool = False,
              selector_top_k: int = 3, selector_rank: int = 3,
              conv_kernel_size: int = 3, conv_group_size: int = 8,
              input_embedding_scale: float = 1.0,
              final_logit_softcapping: float | None = None,
              output_multiplier: float = 1.0
              ) -> tuple[str, dict, str, dict]:
    """(target 目录, target hf, draft 目录, draft hf)——进程内缓存同一份权重。"""
    key = (k, num_layers, causal, selector_top_k, selector_rank, conv_kernel_size,
           conv_group_size, input_embedding_scale, final_logit_softcapping,
           output_multiplier)
    if key not in _PAIR:
        target_dir = tiny_qwen3_dir("tiny_gqa")
        target_hf = tiny_qwen3_config("tiny_gqa")
        draft_dir = tiny_dflash2_dir(
            "tiny_gqa", num_layers=num_layers, causal=causal,
            selector_top_k=selector_top_k, selector_rank=selector_rank,
            conv_kernel_size=conv_kernel_size, conv_group_size=conv_group_size,
            input_embedding_scale=input_embedding_scale,
            final_logit_softcapping=final_logit_softcapping,
            output_multiplier=output_multiplier, num_speculative_tokens=k)
        draft_hf = json.loads((Path(draft_dir) / "config.json").read_text())
        _PAIR[key] = (target_dir, target_hf, draft_dir, draft_hf)
    return _PAIR[key]


def model_configs(**kwargs) -> tuple[ModelConfig, ModelConfig]:
    k = kwargs.get("k", 3)
    target_dir, target_hf, draft_dir, draft_hf = tiny_pair(**kwargs)
    # 直接构造模型时还没有 SpeculativeConfig 写回 K；这里按同一口径补上（`block_size=1+K`）。
    draft_hf = dict(draft_hf)
    draft_hf["num_speculative_tokens"] = int(k)
    return (
        ModelConfig(model=target_dir, dtype="float32", max_model_len=64, hf_config=target_hf),
        ModelConfig(model=draft_dir, dtype="float32", max_model_len=64, hf_config=draft_hf),
    )


def make_spec_config(*, k: int = 3, draft_sample_method: str = "probabilistic",
                     **kwargs) -> SpeculativeConfig:
    _, draft_config = model_configs(k=k, **kwargs)
    return SpeculativeConfig(method="dflash", num_speculative_tokens=k,
                             draft_model_config=draft_config,
                             draft_sample_method=draft_sample_method)


def make_vllm_config(spec: SpeculativeConfig, *, max_num_seqs: int = 2,
                     budget: int = 64, blocks: int = 64, mode: str = "none",
                     capture_sizes=(6, 12),
                     attention_backend: str = "torch") -> VllmConfig:
    from minivllm.config import CompilationConfig

    target_config, _ = model_configs()
    with v2_env():
        return VllmConfig(
            model_config=target_config,
            cache_config=CacheConfig(block_size=4, num_gpu_blocks=blocks),
            scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                             max_num_batched_tokens=budget),
            device_config=DeviceConfig(device=DEVICE), speculative_config=spec,
            compilation_config=CompilationConfig(
                cudagraph_mode=mode, cudagraph_capture_sizes=list(capture_sizes)),
            attention_backend=attention_backend,
        )


def make_speculator(*, k: int = 3, draft_sample_method: str = "probabilistic",
                    max_num_seqs: int = 2, **kwargs):
    """不建引擎的提议者：`__init__` + `load_draft_model()` 都是真的。"""
    from minivllm.model_loader import get_model
    from minivllm.worker.gpu.spec_decode import init_speculator

    spec = make_spec_config(k=k, draft_sample_method=draft_sample_method, **kwargs)
    config = make_vllm_config(spec, max_num_seqs=max_num_seqs)
    with v2_env():
        speculator = init_speculator(config, torch.device(DEVICE))
        target_model = get_model(config.model_config, torch.device(DEVICE))
        speculator.load_model(target_model)
    return speculator, target_model


def set_walk_inputs(speculator, anchors, *, positions=None, temperature=0.0, seeds=0,
                    slot_ids=None):
    """摆好 walk 依赖的常驻缓冲（= `prepare_dflash_inputs` 内核输出的那一半）。

    `sample_pos` 是"被预测 token 的位置 Q"；`sample_idx_mapping` 是请求 slot（-1 = 补齐行）。
    """
    num_reqs = len(anchors)
    n_spec = speculator.num_speculative_steps
    n = num_reqs * n_spec
    speculator.sample_indices[:n] = torch.arange(n, device=speculator.device)
    if positions is None:
        base = torch.arange(num_reqs, device=speculator.device, dtype=torch.int64) * 20 + 10
        steps = torch.arange(n_spec, device=speculator.device, dtype=torch.int64)
        positions = (base[:, None] + steps[None, :] + 1).reshape(-1)
    speculator.sample_pos[:n] = torch.as_tensor(positions, dtype=torch.int64,
                                                device=speculator.device)
    if slot_ids is None:
        slot_ids = torch.arange(num_reqs, device=speculator.device, dtype=torch.int32)
    mapping = torch.as_tensor(slot_ids, dtype=torch.int32,
                              device=speculator.device).repeat_interleave(n_spec)
    speculator.sample_idx_mapping[:n] = mapping
    speculator.temperature.fill_(0.0)
    speculator.temperature[torch.as_tensor(slot_ids, device=speculator.device)] = float(
        temperature)
    speculator.seeds.fill_(int(seeds))
    anchor_idx = speculator._anchor_indices[:num_reqs]
    speculator.input_buffers.input_ids[anchor_idx] = torch.as_tensor(
        anchors, dtype=torch.int32, device=speculator.device)
    return n


def walk_case(*, batch: int = 2, k: int = 3, top_k: int = 3, vocab: int = 11,
              seed: int = 0, unique: bool = True):
    """随机但确定的候选表与边分。"""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    candidates = []
    for _ in range(batch * k):
        if unique:
            candidates.append(torch.randperm(vocab, device=DEVICE, generator=generator)[
                :top_k])
        else:
            candidates.append(torch.randint(0, vocab, (top_k,), device=DEVICE,
                                            generator=generator))
    candidate_ids = torch.stack(candidates).view(batch, k, top_k)
    scores = torch.randn(batch, k, top_k, top_k, device=DEVICE, generator=generator)
    return candidate_ids, scores


# ---------------- 上游模块（只在测试里实例化） ----------------

_UPSTREAM_CONFIG = None


def _upstream_vllm_config():
    """构造一份最小上游 `VllmConfig`（只为了让 `initialize_model_parallel` 有配置可读）。

    **不建引擎、不跑模型**；`DFlashGroupedConv` / `CandidateSelector` 的构造只需要
    "TP 组已初始化"与一个当前配置上下文。
    """
    global _UPSTREAM_CONFIG
    if _UPSTREAM_CONFIG is None:
        from vllm.config import (DeviceConfig as UpDeviceConfig,
                                 ModelConfig as UpModelConfig,
                                 ParallelConfig, VllmConfig,
                                 set_current_vllm_config)  # noqa: F401
        from vllm.config.cache import CacheConfig as UpCacheConfig
        from vllm.config.scheduler import SchedulerConfig as UpSchedulerConfig

        target_dir = tiny_qwen3_dir("tiny_gqa")
        target = UpModelConfig(model=target_dir, dtype="float32", max_model_len=64)
        parallel = ParallelConfig()
        _UPSTREAM_CONFIG = VllmConfig(
            model_config=target,
            cache_config=UpCacheConfig(block_size=4, gpu_memory_utilization=0.1),
            parallel_config=parallel,
            device_config=UpDeviceConfig("cuda" if torch.cuda.is_available() else "cpu"),
            scheduler_config=UpSchedulerConfig(
                max_num_seqs=1, max_num_batched_tokens=64, max_model_len=64,
                is_encoder_decoder=False),
        )
    return _UPSTREAM_CONFIG


@contextlib.contextmanager
def upstream_context():
    """进入上游的当前配置上下文；首次调用时初始化单进程 TP 组（幂等）。"""
    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(_upstream_vllm_config()):
        from vllm.distributed import (init_distributed_environment,
                                      initialize_model_parallel)
        from vllm.distributed.parallel_state import model_parallel_is_initialized

        if not model_parallel_is_initialized():
            init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                         distributed_init_method="tcp://127.0.0.1:29811",
                                         backend="gloo")
            initialize_model_parallel(tensor_model_parallel_size=1,
                                      pipeline_model_parallel_size=1)
        yield


def make_upstream_conv(*, hidden_size: int, taps: int, group_size: int,
                       block_size: int):
    from vllm.model_executor.models.qwen3_dflash2 import DFlashGroupedConv

    with upstream_context():
        return DFlashGroupedConv(hidden_size=hidden_size, taps=taps,
                                 group_size=group_size, block_size=block_size,
                                 params_dtype=torch.float32, prefix="")


def make_upstream_selector(*, hidden_size: int, vocab_size: int, rank: int,
                           top_k: int):
    from vllm.model_executor.models.qwen3_dflash2 import CandidateSelector

    with upstream_context():
        return CandidateSelector(hidden_size=hidden_size, vocab_size=vocab_size,
                                 rank=rank, top_k=top_k, params_dtype=torch.float32,
                                 prefix="")


# ---------------- 引擎 ----------------

def make_engine(*, k: int = 3, draft_sample_method: str = "greedy", mode: str = "none",
                capture_sizes=(6, 12), max_num_seqs: int = 2,
                attention_backend: str = "torch", **kwargs):
    spec = make_spec_config(k=k, draft_sample_method=draft_sample_method, **kwargs)
    config = make_vllm_config(spec, max_num_seqs=max_num_seqs, mode=mode,
                              capture_sizes=capture_sizes,
                              attention_backend=attention_backend)
    with v2_env():
        engine = LLMEngine(config, UniProcExecutor(config, Worker(config)), tokenizer=None)
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner


def make_baseline_engine(*, max_num_seqs: int = 2):
    """同 target、**不投机**的 V2 引擎（greedy 无损性的对照）。"""
    from minivllm.config import CompilationConfig

    target_dir, target_hf = tiny_qwen3_dir("tiny_gqa"), tiny_qwen3_config("tiny_gqa")
    with v2_env():
        config = VllmConfig(
            model_config=ModelConfig(model=target_dir, dtype="float32",
                                     max_model_len=64, hf_config=target_hf),
            cache_config=CacheConfig(block_size=4, num_gpu_blocks=64),
            scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                             max_num_batched_tokens=64),
            device_config=DeviceConfig(device=DEVICE),
            compilation_config=CompilationConfig(cudagraph_mode="none"),
        )
        engine = LLMEngine(config, UniProcExecutor(config, Worker(config)), tokenizer=None)
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner


def run_prompts(engine, prompts, *, max_tokens=6, temperature=0.0, seed=None,
                shutdown=True):
    for req_id, prompt in prompts:
        engine.add_request(req_id, list(prompt),
                           SamplingParams(max_tokens=max_tokens, temperature=temperature,
                                          seed=seed, eos_token_id=99))
    outputs: dict[str, list[int]] = {}
    for _ in range(200):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            outputs[out.request_id] = list(out.token_ids)
    if shutdown:
        engine.shutdown()
    return outputs
