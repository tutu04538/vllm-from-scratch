"""step77 的测试夹具：DSpark 的配置/布局/采样三条线。

三块内容：

**A. 模型对**（`tiny_pair`）  同一份 tiny target + 一份 tiny DSpark 草稿（进程内缓存），
   参数化出 anchor/fill-in 布局、缩减词表、top-k、confidence 头、logit_scale。

**B. 轻量提议者**（`make_speculator` + `set_sampling`）  只走 `DSparkSpeculator.__init__`
   → `load_draft_model()` → `_sample_sequential()`，不建引擎、不捕获图。于是可以**逐位控制**
   base logits 与 anchor token，验证"条件分布随前一个 token 变化""缓存的 q == 实际用的 q"。

**C. 引擎**（`make_engine` / `run_prompts`）  真正的 V2 通路（`VLLM_USE_V2_MODEL_RUNNER=1`），
   用来跑端到端、图/eager 对照与草稿 token 快照。

为什么 A 要缓存：`tiny_dspark_dir()` 每次都写新目录，而"同一份权重"是差分测试的前提。
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# DSpark 与 DFlash 共用 `_prepare_dflash_inputs_kernel`，布局差分直接复用 step76 的
# 上游/本仓库内核夹具（它已经把 12 个输出对好了）。`tests/step76` 不一定在 sys.path 上
# （只跑 step77 时就不在），所以这里显式补。
STEP76 = Path(__file__).resolve().parents[1] / "step76"
if str(STEP76) not in sys.path:
    sys.path.insert(0, str(STEP76))

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig,  # noqa: E402
                      SamplingParams, SchedulerConfig, SpeculativeConfig,
                      UniProcExecutor, VllmConfig, Worker)
from minivllm.testing.tiny_models import (tiny_dspark_dir, tiny_qwen3_config,  # noqa: E402
                                          tiny_qwen3_dir)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_PAIR: dict[tuple, tuple[str, dict, str, dict]] = {}


@contextlib.contextmanager
def v2_env():
    """只在**建配置/建引擎**期间打开 V2 入口（DSpark 只有 V2）。

    为什么不能在模块顶层写死 `os.environ[...] = "1"`：一次 pytest 跑多个目录时，测试模块的
    import 是**同一个进程**——顶层改环境变量会把 step58~72 的 V1 用例全部切到 V2（实测 201 项
    失败）。所以这里按需打开、退出即还原。
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


def tiny_pair(*, anchor: bool = True, draft_vocab_size: int | None = None,
              draft_topk: int | None = None, confidence_head: bool = False,
              logit_scale: float = 1.0, omit_embed_tokens: bool = False,
              omit_lm_head: bool = False, causal: bool = False,
              markov_rank: int = 3, d2t_offset: int = 0) -> tuple[str, dict, str, dict]:
    """(target 目录, target hf, draft 目录, draft hf)——进程内缓存同一份权重。"""
    key = (anchor, draft_vocab_size, draft_topk, confidence_head, logit_scale,
           omit_embed_tokens, omit_lm_head, causal, markov_rank, d2t_offset)
    if key not in _PAIR:
        target_dir = tiny_qwen3_dir("tiny_gqa")
        target_hf = tiny_qwen3_config("tiny_gqa")
        draft_dir = tiny_dspark_dir(
            "tiny_gqa", sample_from_anchor=anchor, draft_vocab_size=draft_vocab_size,
            draft_topk=draft_topk, confidence_head=confidence_head,
            logit_scale=logit_scale, omit_embed_tokens=omit_embed_tokens,
            omit_lm_head=omit_lm_head, causal=causal, markov_rank=markov_rank,
            d2t_offset=d2t_offset)
        draft_hf = json.loads((Path(draft_dir) / "config.json").read_text())
        _PAIR[key] = (target_dir, target_hf, draft_dir, draft_hf)
    return _PAIR[key]


def model_configs(**kwargs) -> tuple[ModelConfig, ModelConfig]:
    target_dir, target_hf, draft_dir, draft_hf = tiny_pair(**kwargs)
    return (
        ModelConfig(model=target_dir, dtype="float32", max_model_len=64, hf_config=target_hf),
        ModelConfig(model=draft_dir, dtype="float32", max_model_len=64, hf_config=draft_hf),
    )


def make_spec_config(*, k: int = 3, draft_sample_method: str = "probabilistic",
                     **kwargs) -> SpeculativeConfig:
    _, draft_config = model_configs(**kwargs)
    return SpeculativeConfig(method="dspark", num_speculative_tokens=k,
                             draft_model_config=draft_config,
                             draft_sample_method=draft_sample_method)


def make_vllm_config(spec: SpeculativeConfig, *, max_num_seqs: int = 2,
                     budget: int = 64, blocks: int = 64, mode: str = "none",
                     capture_sizes=(6, 12), attention_backend: str = "torch", block_size: int = 4) -> VllmConfig:
    from minivllm.config import CompilationConfig

    target_config, _ = model_configs()
    with v2_env():
        return VllmConfig(
            model_config=target_config,
            cache_config=CacheConfig(block_size=block_size, num_gpu_blocks=blocks),
            scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                             max_num_batched_tokens=budget),
            device_config=DeviceConfig(device=DEVICE), speculative_config=spec,
            compilation_config=CompilationConfig(
                cudagraph_mode=mode, cudagraph_capture_sizes=list(capture_sizes)),
            attention_backend=attention_backend,
        )


# ---------------- B. 轻量提议者（直接调 `_sample_sequential`） ----------------

def make_speculator(*, k: int = 3, draft_sample_method: str = "probabilistic",
                    confidence_head: bool = False, **kwargs):
    """不建引擎的提议者：`__init__` + `load_draft_model()` 都是真的，只是不跑引擎循环。"""
    from minivllm.model_loader import get_model
    from minivllm.worker.gpu.spec_decode import init_speculator

    spec = make_spec_config(k=k, draft_sample_method=draft_sample_method,
                            confidence_head=confidence_head, **kwargs)
    config = make_vllm_config(spec)
    with v2_env():
        speculator = init_speculator(config, torch.device(DEVICE))
        target_model = get_model(config.model_config, torch.device(DEVICE))
        speculator.load_model(target_model)
    return speculator, target_model


def set_sampling(speculator, anchors, *, positions=None, temperature=0.0, seeds=0):
    """摆好 `_sample_sequential` 依赖的常驻缓冲（= `prepare_dflash_inputs` 内核输出的那一半）。

    `sample_indices` 指向 `head_hidden` 的行号，`sample_pos` 是"被预测 token 的位置 Q"
    （anchor-as-first：query 位置 + 1；fill-in：query 位置本身），`sample_idx_mapping` 是
    请求 slot。**这些值只影响 Gumbel 噪声的下标与 q 的落行**，所以测试可以直接给定。
    """
    num_reqs = len(anchors)
    n_spec = speculator.num_speculative_steps
    n = num_reqs * n_spec
    speculator.sample_indices[:n] = torch.arange(n, device=speculator.device)
    if positions is None:
        # 与内核同构：每请求 K 个位置从 10 开始，anchor 布局 +1、fill-in 原样。
        base = torch.arange(num_reqs, device=speculator.device, dtype=torch.int64) * 20 + 10
        steps = torch.arange(n_spec, device=speculator.device, dtype=torch.int64)
        positions = (base[:, None] + steps[None, :]
                     + (1 if speculator.sample_from_anchor else 0)).reshape(-1)
    speculator.sample_pos[:n] = torch.as_tensor(positions, dtype=torch.int64,
                                                device=speculator.device)
    speculator.sample_idx_mapping[:n] = torch.arange(
        num_reqs, device=speculator.device, dtype=torch.int32).repeat_interleave(n_spec)
    speculator.temperature.fill_(float(temperature))
    speculator.seeds.fill_(int(seeds))
    anchor_idx = speculator._anchor_idx[:num_reqs]
    speculator.input_buffers.input_ids[anchor_idx] = torch.as_tensor(
        anchors, dtype=torch.int32, device=speculator.device)
    return n


def patch_base_logits(speculator, base_logits: torch.Tensor):
    """把主干输出换成**给定的** base logits（`[num_reqs, K, V_draft]`）。

    这样"条件分布只由前一个 token 决定"这条性质就可以用固定的数字验证，不受 tiny 权重影响。
    返回一个还原函数（测试里用 try/finally 或 monkeypatch）。
    """
    flat = base_logits.reshape(-1, base_logits.shape[-1])
    original = speculator.model.compute_draft_logits
    speculator.model.compute_draft_logits = lambda hidden: flat
    return lambda: setattr(speculator.model, "compute_draft_logits", original)


# ---------------- C. 引擎（端到端 / 图对照） ----------------

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


# ---------------- D. 错误边界用的假配置 ----------------

def fake_pp_config(spec: SpeculativeConfig, pp_size: int):
    """只有 `load_dspark_model()` 会读的两个字段：PP 大小与投机配置。"""
    return types.SimpleNamespace(
        speculative_config=spec,
        device_config=DeviceConfig(device=DEVICE),
        parallel_config=types.SimpleNamespace(pipeline_parallel_size=pp_size),
    )
