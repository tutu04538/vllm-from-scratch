"""step76 的测试夹具：DFlash 的布局、模型与两条引擎路径。

三块内容：

**A. 布局夹具**（`DFlashRows` / `kernel_inputs` / `reference_expansion`）
   同一组数字既喂给上游 Triton 内核（V1 的 `copy_and_expand_dflash_inputs_kernel`、
   V2 的 `_prepare_dflash_inputs_kernel`），也喂给本仓库的 Python 实现——于是"逐值差分"
   说的是同一批输入、同一批输出。

**B. 模型夹具**（`bit_matched_model` / `bind_caches` / `run_query_forward`）
   加载 tiny DFlash、绑 KV 缓存、按"上下文 KV 预计算 + query 前向"两步跑；参考实现见
   `minivllm/testing/torch_dflash.py`。

**C. 引擎夹具**（`make_v1_engine` / `make_v2_engine` / `run_prompts`）
   只跑 tiny 模型（V2 需要 UVA，`VLLM_WSL2_ENABLE_PIN_MEMORY=1`；V1 不需要）。

**为什么 target/draft 的 tiny 目录要在这里统一取**：`tiny_dflash_dir()` 每次都写新目录，
两份加载必须吃**同一份权重**（否则差分没有意义），所以这里进程内缓存一份。
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig,  # noqa: E402
                      SamplingParams, SchedulerConfig, SpeculativeConfig,
                      UniProcExecutor, VllmConfig, Worker)
from minivllm.testing.tiny_models import tiny_dflash_dir, tiny_qwen3_config, tiny_qwen3_dir  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DFlash 的输入组装是 Triton 内核（上游同款边界），只在 CUDA 上可执行")

_PAIR: dict[str, tuple[str, dict, str, dict]] = {}


def tiny_dflash_pair(*, causal: bool = False, num_layers: int = 2,
                     naming: str = "layers", draft_vocab_size: int | None = None,
                     use_aux_hidden_state: bool = True) -> tuple[str, dict, str, dict]:
    """(target 目录, target hf, draft 目录, draft hf)——进程内缓存同一份权重。"""
    key = f"{causal}-{num_layers}-{naming}-{draft_vocab_size}-{use_aux_hidden_state}"
    if key not in _PAIR:
        target_dir = tiny_qwen3_dir("tiny_gqa")
        target_hf = tiny_qwen3_config("tiny_gqa")
        draft_dir = tiny_dflash_dir(
            "tiny_gqa", causal=causal, num_layers=num_layers, naming=naming,
            draft_vocab_size=draft_vocab_size, use_aux_hidden_state=use_aux_hidden_state)
        draft_hf = json.loads((Path(draft_dir) / "config.json").read_text())
        _PAIR[key] = (target_dir, target_hf, draft_dir, draft_hf)
    return _PAIR[key]


def model_configs(*, causal: bool = False, num_layers: int = 2, naming: str = "layers",
                  draft_vocab_size: int | None = None,
                  use_aux_hidden_state: bool = True) -> tuple[ModelConfig, ModelConfig]:
    target_dir, target_hf, draft_dir, draft_hf = tiny_dflash_pair(
        causal=causal, num_layers=num_layers, naming=naming,
        draft_vocab_size=draft_vocab_size, use_aux_hidden_state=use_aux_hidden_state)
    return (
        ModelConfig(model=target_dir, dtype="float32", max_model_len=64, hf_config=target_hf),
        ModelConfig(model=draft_dir, dtype="float32", max_model_len=64, hf_config=draft_hf),
    )


# ---------------- 布局夹具 ----------------

@dataclass
class LayoutCase:
    """一次"两请求、K=3"的布局事实（数字是手写的，便于在断言里直接引用）。"""

    target_positions: torch.Tensor
    target_query_start_loc: torch.Tensor
    next_token_ids: torch.Tensor
    num_rejected: torch.Tensor
    block_table: torch.Tensor
    block_size: int = 4
    num_query_per_req: int = 4
    num_speculative_steps: int = 3
    max_model_len: int = 64
    max_num_reqs: int = 8


def layout_case(device: str = "cpu") -> LayoutCase:
    """请求 0：4 行 target（位置 7..10，被拒 2 行）；请求 1：2 行 target（位置 20..21）。"""
    return LayoutCase(
        target_positions=torch.tensor([7, 8, 9, 10, 20, 21], dtype=torch.int64, device=device),
        target_query_start_loc=torch.tensor([0, 4, 6], dtype=torch.int32, device=device),
        next_token_ids=torch.tensor([11, 22], dtype=torch.int32, device=device),
        num_rejected=torch.tensor([2, 0], dtype=torch.int32, device=device),
        block_table=torch.stack([
            torch.arange(16, dtype=torch.int32, device=device),
            torch.arange(16, 32, dtype=torch.int32, device=device),
        ]),
    )


def fake_rows(case: LayoutCase):
    """把布局事实翻成 `TargetRows`（V1 提议者的输入契约）。"""
    from minivllm.spec_decode.utils import TargetRows

    rows = []
    for index, (start, end) in enumerate(zip(case.target_query_start_loc.tolist()[:-1],
                                             case.target_query_start_loc.tolist()[1:])):
        target_rows = end - start
        rejected = int(case.num_rejected[index])
        rows.append(TargetRows(
            req_id=f"r{index}", row=index, start=int(case.target_positions[start]),
            target_rows=target_rows, num_rejected=rejected,
            next_token_id=int(case.next_token_ids[index]), ready=True,
            history_end=int(case.target_positions[end - 1 - rejected]) + 1,
        ))
    return rows


# ---------------- 模型夹具 ----------------

def tiny_dflash_model(*, causal: bool = False, num_layers: int = 2,
                      naming: str = "layers", draft_vocab_size: int | None = None,
                      use_aux_hidden_state: bool = True, device: str = DEVICE):
    """加载 tiny DFlash 并把每层 KV 缓存绑上（大小为 32 块 × block_size 4）。"""
    from minivllm.model_loader import get_model

    _, draft_config = model_configs(
        causal=causal, num_layers=num_layers, naming=naming,
        draft_vocab_size=draft_vocab_size, use_aux_hidden_state=use_aux_hidden_state)
    model = get_model(draft_config, device)
    bind_caches(model, device=device)
    return model, draft_config.hf_config


def bind_caches(model, *, num_blocks: int = 32, block_size: int = 4, device: str = DEVICE):
    modules = dict(model.named_modules())
    for name in model.get_draft_kv_cache_layer_names():
        module = modules[name]
        module.kv_cache = torch.zeros(
            2, num_blocks, block_size, module.num_kv_heads, module.head_size, device=device)
    return model


def run_query_forward(model, layer_names, *, context_states, context_positions,
                      query_ids, query_positions, seq_len, causal: bool,
                      context_slots=None, query_slots=None):
    """生产路径的两步：写上下文 KV（可选）+ 用元数据跑 query 前向。"""
    from minivllm.attention.backends.torch_sdpa import TorchAttentionBackend
    from minivllm.forward_context import set_forward_context

    if context_slots is not None:
        model.precompute_and_store_context_kv(
            context_states, context_positions, context_slots)
    num_ctx = context_states.shape[0]
    num_query = query_ids.shape[0]
    builder = TorchAttentionBackend.get_builder_cls()(4)
    metadata = builder.build(
        query_start_loc=torch.tensor([0, num_query], device=query_ids.device),
        seq_lens=torch.tensor([seq_len], device=query_ids.device),
        block_table=torch.arange(8, device=query_ids.device).view(1, -1),
        slot_mapping=(query_slots if query_slots is not None
                      else torch.arange(num_ctx, num_ctx + num_query,
                                        device=query_ids.device)),
        num_reqs=1, causal=causal)
    with set_forward_context({name: metadata for name in layer_names}, num_tokens=num_query):
        return model(query_ids, query_positions)


# ---------------- 引擎夹具 ----------------

def make_v1_engine(*, spec_k: int = 3, budget: int = 64, blocks: int = 64,
                   max_num_seqs: int = 2, max_model_len: int = 64,
                   draft_sample_method: str = "greedy", causal: bool = False,
                   naming: str = "layers"):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    target_config, draft_config = model_configs(causal=causal, naming=naming)
    spec = SpeculativeConfig(
        method="dflash", num_speculative_tokens=spec_k,
        draft_model_config=draft_config, draft_sample_method=draft_sample_method)
    config = VllmConfig(
        model_config=target_config,
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=DEVICE), speculative_config=spec)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)), tokenizer=None)
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner


def make_v2_engine(*, spec_k: int = 3, budget: int = 64, blocks: int = 64,
                   max_num_seqs: int = 2, max_model_len: int = 64,
                   capture_sizes=(4, 8, 16), mode: str = "full_decode_only",
                   causal: bool = False, draft_sample_method: str = "greedy"):
    from minivllm.config import CompilationConfig

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    target_config, draft_config = model_configs(causal=causal)
    spec = SpeculativeConfig(
        method="dflash", num_speculative_tokens=spec_k,
        draft_model_config=draft_config, draft_sample_method=draft_sample_method)
    config = VllmConfig(
        model_config=target_config,
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=DEVICE), speculative_config=spec,
        compilation_config=CompilationConfig(
            cudagraph_mode=mode, cudagraph_capture_sizes=list(capture_sizes)),
    )
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)), tokenizer=None)
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner


def run_prompts(engine, prompts, *, max_tokens=6, temperature=0.0, seed=None):
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
    engine.shutdown()
    return outputs


# ---------------- 内核夹具（逐值差分用） ----------------

def _grid(case: LayoutCase):
    max_tokens_per_req = int(case.target_query_start_loc.diff().max()) \
        + case.num_query_per_req
    block = min(256, 1 << (max(1, max_tokens_per_req) - 1).bit_length())
    num_blocks = (max_tokens_per_req + block - 1) // block
    return (int(case.num_rejected.numel()), num_blocks), block


def run_upstream_v1_kernel(case: LayoutCase) -> dict[str, torch.Tensor]:
    """上游 `copy_and_expand_dflash_inputs_kernel` 的原始输出（V1 口径）。"""
    from vllm.v1.spec_decode.utils import copy_and_expand_dflash_inputs_kernel

    device = case.target_positions.device
    batch = int(case.num_rejected.numel())
    num_query = case.num_query_per_req
    num_ctx = case.target_positions.numel()
    outputs = {
        "input_ids": torch.zeros(batch * num_query, dtype=torch.int64, device=device),
        "context_positions": torch.zeros(num_ctx, dtype=torch.int64, device=device),
        "query_positions": torch.zeros(batch * num_query, dtype=torch.int64, device=device),
        "context_slot_mapping": torch.zeros(num_ctx, dtype=torch.int64, device=device),
        "query_slot_mapping": torch.zeros(batch * num_query, dtype=torch.int64, device=device),
        "sample_indices": torch.zeros(batch * case.num_speculative_steps,
                                      dtype=torch.int32, device=device),
    }
    grid, block = _grid(case)
    copy_and_expand_dflash_inputs_kernel[grid](
        case.next_token_ids, case.target_positions,
        outputs["input_ids"], outputs["context_positions"], outputs["query_positions"],
        outputs["context_slot_mapping"], outputs["query_slot_mapping"],
        outputs["sample_indices"],
        case.block_table, case.block_table.stride(0),
        case.target_query_start_loc, case.num_rejected,
        MASK_TOKEN_ID, case.block_size, num_query, case.num_speculative_steps,
        num_ctx, BLOCK_SIZE=block, HAS_NUM_REJECTED=True)
    return outputs


def run_our_v1_expansion(case: LayoutCase) -> dict[str, torch.Tensor]:
    """本仓库 V1 的展开（`expand_dflash_inputs` + 槽位查表），CPU/GPU 都能跑。"""
    from minivllm.spec_decode.utils import compute_slot_mapping_for_rows, expand_dflash_inputs

    device = case.target_positions.device
    rows = fake_rows(case)
    expanded = expand_dflash_inputs(
        [int(value) for value in case.target_positions.tolist()],
        rows, num_query_per_req=case.num_query_per_req, mask_token_id=MASK_TOKEN_ID)
    batch = len(rows)
    context_positions = torch.tensor(expanded.context_positions, dtype=torch.int64,
                                     device=device)
    context_req = torch.arange(batch, device=device).repeat_interleave(
        torch.tensor([row.target_rows for row in rows], device=device))
    query_positions = torch.tensor(expanded.positions, dtype=torch.int64, device=device)
    query_req = torch.arange(batch, device=device).repeat_interleave(
        case.num_query_per_req)
    return {
        "input_ids": torch.tensor(expanded.input_ids, dtype=torch.int64, device=device),
        "context_positions": context_positions,
        "query_positions": query_positions,
        "context_slot_mapping": compute_slot_mapping_for_rows(
            context_positions, context_req, case.block_table, case.block_size,
            case.max_model_len),
        "query_slot_mapping": compute_slot_mapping_for_rows(
            query_positions, query_req, case.block_table, case.block_size,
            case.max_model_len),
        "sample_indices": torch.tensor(expanded.token_indices_to_sample, dtype=torch.int32,
                                       device=device),
    }


def run_upstream_v2_kernel(case: LayoutCase, *, max_tokens: int = 64,
                           max_num_reqs: int = 8,
                           sample_from_anchor: bool = False) -> dict[str, torch.Tensor]:
    """上游 V2 `_prepare_dflash_inputs_kernel` 的原始输出。

    `sample_from_anchor=True` 时走 DSpark 的 anchor-as-first 布局（77 关）——上游同一个内核
    用 `SAMPLE_FROM_ANCHOR` 这个 constexpr 切两种布局，所以差异测试仍然比同一份内核。
    """
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        _prepare_dflash_inputs_kernel)

    return _run_v2_kernel(_prepare_dflash_inputs_kernel, case, upstream=True,
                          max_tokens=max_tokens, max_num_reqs=max_num_reqs,
                          sample_from_anchor=sample_from_anchor)


def run_our_v2_kernel(case: LayoutCase, *, max_tokens: int = 64,
                      max_num_reqs: int = 8,
                      sample_from_anchor: bool = False) -> dict[str, torch.Tensor]:
    from minivllm.worker.gpu.spec_decode.dflash.speculator import (
        _prepare_dflash_inputs_kernel)

    return _run_v2_kernel(_prepare_dflash_inputs_kernel, case, upstream=False,
                          max_tokens=max_tokens, max_num_reqs=max_num_reqs,
                          sample_from_anchor=sample_from_anchor)


MASK_TOKEN_ID = 7


def _run_v2_kernel(kernel, case: LayoutCase, *, upstream: bool, max_tokens: int,
                   max_num_reqs: int,
                   sample_from_anchor: bool = False) -> dict[str, torch.Tensor]:
    device = case.target_positions.device
    batch = int(case.num_rejected.numel())
    num_query = case.num_query_per_req
    num_steps = case.num_speculative_steps
    idx_mapping = torch.tensor([0, 5], dtype=torch.int32, device=device)[:batch]
    last_sampled = torch.tensor([11, 0, 0, 0, 0, 22, 0, 0], dtype=torch.int64,
                                device=device)[:max_num_reqs]
    next_prefill = torch.zeros(1, max_num_reqs, dtype=torch.int32, device=device)
    num_sampled = torch.ones(batch, dtype=torch.int32, device=device)
    temperature = torch.linspace(0.5, 1.0, max_num_reqs, dtype=torch.float32, device=device)
    seeds = torch.arange(max_num_reqs, dtype=torch.int64, device=device)
    outputs = {
        "input_ids": torch.zeros(batch * num_query, dtype=torch.int32, device=device),
        "query_positions": torch.zeros(batch * num_query, dtype=torch.int64, device=device),
        "query_start_loc": torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device),
        "seq_lens": torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
        "query_slot_mapping": torch.zeros(max_tokens, dtype=torch.int64, device=device),
        "context_positions": torch.zeros(max_tokens, dtype=torch.int64, device=device),
        "context_slot_mapping": torch.zeros(max_tokens, dtype=torch.int64, device=device),
        "sample_indices": torch.zeros(max_num_reqs * num_steps, dtype=torch.int64,
                                      device=device),
        "sample_pos": torch.zeros(max_num_reqs * num_steps, dtype=torch.int64, device=device),
        "sample_idx_mapping": torch.full((max_num_reqs * num_steps,), -1,
                                         dtype=torch.int32, device=device),
        "temperature": torch.zeros(max_num_reqs, dtype=torch.float32, device=device),
        "seeds": torch.zeros(max_num_reqs, dtype=torch.int64, device=device),
    }
    grid, block = _grid(case)
    args = (
        outputs["input_ids"], outputs["query_positions"], outputs["query_start_loc"],
        outputs["seq_lens"], outputs["query_slot_mapping"], outputs["context_positions"],
        outputs["context_slot_mapping"], outputs["sample_indices"], outputs["sample_pos"],
        outputs["sample_idx_mapping"], outputs["temperature"], outputs["seeds"],
        case.target_positions, case.target_query_start_loc, idx_mapping, last_sampled,
        next_prefill, num_sampled, case.num_rejected, temperature, seeds,
        case.block_table, case.block_table.stride(0),
        MASK_TOKEN_ID, case.block_size, num_query, num_steps, max_num_reqs, max_tokens,
        case.max_model_len,
    )
    if upstream:
        kernel[grid](*args, SAMPLE_FROM_ANCHOR=sample_from_anchor, PAD_SLOT_ID=-1,
                     BLOCK_SIZE=block)
    else:
        kernel[grid](*args, SAMPLE_FROM_ANCHOR=sample_from_anchor, PAD_SLOT_ID=-1,
                     BLOCK_SIZE=block)
    return outputs
