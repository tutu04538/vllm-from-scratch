"""需求 080B：多模块 MTP 的 eager 执行与拒绝后 KV 修复。

本文件的三层证据：

1. **调用链**：工厂返回 `MultiModuleMTPSpeculator`；K=2/4、模块数=2；每个 propose
   的 `spec_step_idx` 是 `0..K-1`；Graph 模式在 80C 接通（模式收敛表见 80C 测试）。
2. **端到端**：tiny 多模块 MTP 的 greedy 输出与不投机逐 token 相同；mixed/重排/
   partial-prefill/prefix/抢占场景不崩。
3. **逐模块 KV**：把生产 speculator 每轮的第一遍输入、每步输入、logits、每个模块的
   KV 快照下来，用**测试内独立重放**（不复用生产 `_prepare_inputs` / `_generate_drafts`）
   从同样的 target hidden + 上一轮 cache 重新构造 packed 窗口、跑同一份模型，再逐值比较。
   这不是“最终文本还对”的烟测；文本对照只放最后。

负向控制：
- 在 `propose` 前故意把上一轮 cache 清零 → 生产输入与独立重放必须不同；
- 故意把 step 0 的 hidden 当作后续 step 的反馈 → 后续输入/KV 必须不同。

这些负向用例**断言 mismatch**，用来证明比较器不是“永远相等”。
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")


import pytest
import torch


@pytest.fixture(autouse=True)
def _force_v2(monkeypatch):
    """只在本文件的测试内切 V2；测试结束恢复，避免污染同进程的 V1 用例。"""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig,
                      SamplingParams, SchedulerConfig, SpeculativeConfig,
                      UniProcExecutor, VllmConfig, Worker)
from minivllm.attention.backends.torch_sdpa import TorchAttentionMetadataBuilder
from minivllm.attention.layer import Attention
from minivllm.config import CUDAGraphMode, CompilationConfig
from minivllm.forward_context import BatchDescriptor, set_forward_context
from minivllm.model_loader import get_model
from minivllm.testing.tiny_models import tiny_mtp_dir
from minivllm.worker.gpu.spec_decode import init_speculator

DEVICE = "cuda"
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="V2 runner + Triton/注意力需要 CUDA",
)

EOS = 999
PROMPTS = [("a", [1, 2, 3, 4, 1, 2, 3, 4]), ("b", [5, 6, 5, 6])]


# ---------------------------------------------------------------------------
# 共同夹具
# ---------------------------------------------------------------------------


def _mtp_configs(*, k: int, device: str, max_num_seqs: int = 2,
                 budget: int = 64, blocks: int = 64,
                 prefix_caching: bool = False):
    directory = tiny_mtp_dir("tiny_gqa", num_mtp_layers=2)
    hf = json.loads((Path(directory) / "config.json").read_text())
    target = ModelConfig(
        model=directory, dtype="float32", max_model_len=64, hf_config=hf
    )
    draft_hf = {
        **hf,
        "n_predict": hf["num_nextn_predict_layers"],
        "architectures": ["Qwen3MTPModel"],
    }
    draft = ModelConfig(
        model=directory, dtype="float32", max_model_len=64, hf_config=draft_hf
    )
    spec = SpeculativeConfig(
        method="mtp",
        num_speculative_tokens=k,
        draft_model_config=draft,
        draft_sample_method="greedy",
    )
    cfg = VllmConfig(
        model_config=target,
        cache_config=CacheConfig(
            block_size=4,
            num_gpu_blocks=blocks,
            enable_prefix_caching=prefix_caching,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs, max_num_batched_tokens=budget
        ),
        device_config=DeviceConfig(device=device),
        speculative_config=spec,
        compilation_config=CompilationConfig(
            cudagraph_mode="none", cudagraph_capture_sizes=None
        ),
    )
    return cfg


def _make_engine(*, k: int, spec: bool = True, max_num_seqs: int = 2,
                 budget: int = 64, blocks: int = 64,
                 prefix_caching: bool = False):
    cfg = _mtp_configs(
        k=k, device=DEVICE, max_num_seqs=max_num_seqs, budget=budget,
        blocks=blocks, prefix_caching=prefix_caching,
    )
    if not spec:
        cfg = VllmConfig(
            model_config=cfg.model_config,
            cache_config=cfg.cache_config,
            scheduler_config=cfg.scheduler_config,
            device_config=cfg.device_config,
            speculative_config=None,
            compilation_config=cfg.compilation_config,
        )
    engine = LLMEngine(cfg, UniProcExecutor(cfg, Worker(cfg)), tokenizer=None)
    runner = engine.engine_core.engine_core.model_executor.driver_worker.model_runner
    return engine, runner


def _run(engine, prompts=PROMPTS, *, max_tokens: int = 6):
    for req_id, prompt in prompts:
        engine.add_request(
            req_id, list(prompt),
            SamplingParams(max_tokens=max_tokens, temperature=0.0, eos_token_id=EOS),
        )
    outputs = {}
    for _ in range(400):
        if not engine.has_unfinished_requests():
            break
        for item in engine.step():
            outputs[item.request_id] = list(item.token_ids)
    return outputs


def _factory_and_graph_modes():
    """工厂 + 冻结图模式收敛的纯 CPU 用例（不加载模型）。

    80C 之后 graph 不再被拒绝；这里只钉“请求模式 -> 实际 manager 模式”的表：
    带 PIECEWISE 的配置降级，FULL / FULL_DECODE_ONLY / NONE 原样保留。
    """
    cfg = _mtp_configs(k=2, device="cpu")
    speculator = init_speculator(cfg, torch.device("cpu"))
    assert type(speculator).__name__ == "MultiModuleMTPSpeculator"
    assert speculator.cached_draft_input_ids.shape == (2, 1)
    speculator.init_cudagraph_manager(CUDAGraphMode.NONE)
    assert speculator.cudagraph_mode == CUDAGraphMode.NONE
    speculator.init_cudagraph_manager(CUDAGraphMode.PIECEWISE)
    assert speculator.cudagraph_mode == CUDAGraphMode.NONE
    speculator.init_cudagraph_manager(CUDAGraphMode.FULL)
    assert speculator.cudagraph_mode == CUDAGraphMode.FULL
    speculator.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    assert speculator.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
    speculator.init_cudagraph_manager(CUDAGraphMode.FULL_AND_PIECEWISE)
    assert speculator.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
    # 无捕获档位时 capture 是空操作，不应再抛“属 80C”的 NotImplementedError。
    speculator.capture()


# ---------------------------------------------------------------------------
# 0. 工厂、Graph 边界、模块步进
# ---------------------------------------------------------------------------


def test_factory_selects_multi_module_and_graph_modes():
    _factory_and_graph_modes()


@requires_cuda
@pytest.mark.parametrize("k", [2, 4])
def test_eager_greedy_matches_target_and_steps_each_module(k):
    engine, runner = _make_engine(k=k)
    speculator = runner.speculator
    assert type(speculator).__name__ == "MultiModuleMTPSpeculator"
    assert speculator.model.model.num_mtp_layers == 2
    assert runner.req_states.next_prefill_tokens.shape[0] == k

    seen_steps = []
    original = speculator._run_model

    def spy(num_tokens, attn_metadata, slot_mappings, spec_module_idx):
        seen_steps.append(int(spec_module_idx))
        return original(num_tokens, attn_metadata, slot_mappings, spec_module_idx)

    speculator._run_model = spy
    try:
        got = _run(engine)
    finally:
        engine.shutdown()

    base_engine, _ = _make_engine(k=k, spec=False)
    try:
        want = _run(base_engine)
    finally:
        base_engine.shutdown()

    assert got == want, f"greedy output mismatch: {got} != {want}"
    assert seen_steps, "没有跑过任何 MTP 模块"
    chunks = [
        seen_steps[i : i + k] for i in range(0, len(seen_steps), k)
    ]
    assert all(chunk == list(range(k)) for chunk in chunks), (
        f"每轮 spec_step_idx 必须是 0..K-1，实际 {seen_steps}"
    )


# ---------------------------------------------------------------------------
# 1. 生产运行记录 + 独立重放
# ---------------------------------------------------------------------------


@dataclass
class StepCapture:
    input_ids: torch.Tensor
    positions: torch.Tensor
    hidden: torch.Tensor
    slot_mapping: torch.Tensor
    last_token_indices: torch.Tensor
    logits: torch.Tensor
    kv: dict[str, torch.Tensor]


@dataclass
class RoundCapture:
    num_tokens: int
    num_reqs: int
    target_ids: torch.Tensor
    target_positions: torch.Tensor
    target_hidden: torch.Tensor
    target_seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    idx_mapping: torch.Tensor
    num_sampled: torch.Tensor
    num_rejected: torch.Tensor
    last_sampled: torch.Tensor
    next_prefill_tokens: torch.Tensor
    cached_ids: torch.Tensor
    cached_hidden: torch.Tensor
    kv_before: dict[str, torch.Tensor]
    block_table: torch.Tensor
    steps: list[StepCapture] = field(default_factory=list)
    draft_tokens: torch.Tensor | None = None
    corrupted: bool = False


def _snapshot_kv(speculator) -> dict[str, torch.Tensor]:
    out = {}
    for name, module in speculator.model.named_modules():
        if isinstance(module, Attention) and module.kv_cache is not None:
            out[name] = module.kv_cache.clone()
    return out


class RoundRecorder:
    """记录生产 `propose()` 的每轮输入、每步输入/输出与 KV（只观测）。"""

    def __init__(self, runner, *, corrupt_cache: bool = False,
                 stale_hidden: bool = False):
        self.runner = runner
        self.speculator = runner.speculator
        self.corrupt_cache = corrupt_cache
        self.stale_hidden = stale_hidden
        self.records: list[RoundCapture] = []
        self._current: RoundCapture | None = None
        self._corrupted = False
        self._stale: torch.Tensor | None = None

        self._original_propose = self.speculator.propose
        self._original_run_model = self.speculator._run_model
        self.speculator.propose = self._propose
        self.speculator._run_model = self._run_model

    def _propose(self, input_batch, attn_metadata, slot_mappings,
                 last_hidden_states, aux_hidden_states, num_sampled,
                 num_rejected, last_sampled, next_prefill_tokens,
                 temperature, seeds, *args, **kwargs):
        num_reqs = int(input_batch.num_reqs)
        num_tokens = int(input_batch.num_tokens)
        rec = RoundCapture(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            target_ids=input_batch.input_ids[:num_tokens].clone(),
            target_positions=input_batch.positions[:num_tokens].clone(),
            target_hidden=last_hidden_states[:num_tokens].clone(),
            target_seq_lens=input_batch.seq_lens[:num_reqs].clone(),
            query_start_loc=input_batch.query_start_loc[: num_reqs + 1].clone(),
            idx_mapping=input_batch.idx_mapping[:num_reqs].clone(),
            num_sampled=num_sampled[:num_reqs].clone(),
            num_rejected=num_rejected[:num_reqs].clone(),
            last_sampled=last_sampled.clone(),
            next_prefill_tokens=next_prefill_tokens.clone(),
            cached_ids=self.speculator.cached_draft_input_ids.clone(),
            cached_hidden=self.speculator.cached_target_hidden_states.clone(),
            kv_before=_snapshot_kv(self.speculator),
            block_table=self.runner.block_tables.input_block_tables[0][:num_reqs].clone(),
        )
        assert int(rec.query_start_loc[-1]) == num_tokens

        self._current = rec
        self._stale = None
        try:
            if (
                self.corrupt_cache
                and not self._corrupted
                and int(rec.num_rejected.max().item()) > 1
            ):
                # 捕获完正确的 cached_* 后再破坏生产 cache：独立重放仍用 rec.cached_*。
                self.speculator.cached_draft_input_ids.zero_()
                self.speculator.cached_target_hidden_states.zero_()
                self._corrupted = True
                rec.corrupted = True

            result = self._original_propose(
                input_batch, attn_metadata, slot_mappings, last_hidden_states,
                aux_hidden_states, num_sampled, num_rejected, last_sampled,
                next_prefill_tokens, temperature, seeds, *args, **kwargs,
            )
            rec.draft_tokens = result.clone()
            self.records.append(rec)
            return result
        finally:
            self._current = None

    def _run_model(self, num_tokens, attn_metadata, slot_mappings,
                   spec_module_idx):
        rec = self._current
        assert rec is not None, "_run_model 只能在 propose 内被观测"
        last_indices = self.speculator.last_token_indices[: rec.num_reqs].clone()
        slot = (
            slot_mappings["slot_mapping"][0].clone()
            if slot_mappings is not None
            else torch.empty(0, dtype=torch.int64, device=self.speculator.device)
        )
        input_hidden = self.speculator.hidden_states[:num_tokens].clone()
        capture = StepCapture(
            input_ids=self.speculator.input_buffers.input_ids[:num_tokens].clone(),
            positions=self.speculator.input_buffers.positions[:num_tokens].clone(),
            hidden=input_hidden,
            slot_mapping=slot,
            last_token_indices=last_indices,
            logits=torch.empty(0),
            kv={},
        )
        last_hidden, hidden = self._original_run_model(
            num_tokens, attn_metadata, slot_mappings, spec_module_idx
        )
        with torch.inference_mode():
            capture.logits = self.speculator.model.compute_logits(
                last_hidden[last_indices]
            ).clone()
        capture.kv = _snapshot_kv(self.speculator)

        if self.stale_hidden:
            if self._stale is None:
                # 用 step 0 的**输入** hidden 冒充反馈：step 1 的输入会直接错。
                self._stale = input_hidden
            hidden = self._stale.clone()
        rec.steps.append(capture)
        return last_hidden, hidden


def _make_reference_model(speculator):
    model = get_model(
        speculator.speculative_config.draft_model_config,
        speculator.device,
    )
    model.load_state_dict(speculator.model.state_dict())
    model.eval()
    return model


def _valid_rows(rec: RoundCapture, last_indices: torch.Tensor) -> torch.Tensor:
    rows = []
    qsl = rec.query_start_loc.cpu().tolist()
    for req in range(rec.num_reqs):
        rows.extend(range(qsl[req], int(last_indices[req]) + 1))
    return torch.tensor(rows, dtype=torch.int64, device=rec.query_start_loc.device)


def _reference_round(rec: RoundCapture, speculator, ref_model,
                     cached_ids: torch.Tensor, cached_hidden: torch.Tensor):
    """测试内独立重放：不复用生产 `_prepare_inputs` / `_generate_drafts`。

    输入只来自 rec 里的 target hidden/token/position、上一轮 cache、verifier 的
    num_rejected/num_sampled/last_sampled/next_prefill；模块协议按源码重写。
    """
    K = speculator.num_speculative_steps
    num_reqs = rec.num_reqs
    num_tokens = rec.num_tokens
    hidden_size = speculator.hidden_size
    device = rec.query_start_loc.device
    qsl = rec.query_start_loc.cpu().tolist()
    idx = rec.idx_mapping.cpu().tolist()
    num_rejected = rec.num_rejected.cpu().tolist()
    num_sampled = rec.num_sampled.cpu().tolist()
    target_ids = rec.target_ids
    target_positions = rec.target_positions
    target_hidden = rec.target_hidden
    target_seq_lens = rec.target_seq_lens.cpu().tolist()
    block_size = speculator.vllm_config.cache_config.block_size

    ids = torch.zeros(num_tokens, dtype=torch.int64, device=device)
    positions = torch.zeros(num_tokens, dtype=torch.int64, device=device)
    hidden = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=device)
    last_indices = torch.zeros(num_reqs, dtype=torch.int64, device=device)
    overrides = torch.full((num_reqs, K - 1), -1, dtype=torch.int64, device=device)

    for req in range(num_reqs):
        qs, qe = qsl[req], qsl[req + 1]
        qlen = qe - qs
        r = num_rejected[req]
        nr = max(0, r - 1)
        ni = qlen - r
        last = qs + nr + ni - 1
        last_indices[req] = last
        slot = idx[req]

        # token ids：target 第 1..ni-1 行 → draft；gap 从上一轮 cache；右端 next token。
        for i in range(1, ni):
            ids[qs + nr - 1 + i] = target_ids[qs + i]
        for i in range(nr):
            ids[qs + i] = cached_ids[slot, K - 1 - nr + i]
        if num_sampled[req] > 0:
            ids[last] = rec.last_sampled[slot]
        else:
            ids[last] = rec.next_prefill_tokens[0, slot]

        # positions：target 第 0..ni-1 行 → draft；gap 在 first_position 之前。
        first_pos = int(target_positions[qs])
        for i in range(nr):
            positions[qs + i] = first_pos - nr + i
        for i in range(ni):
            positions[qs + nr + i] = target_positions[qs + i]

        # hidden：与 token 同一套 gap；target hidden 行 0..ni-1 复制到 nr 起。
        for i in range(nr):
            hidden[qs + i] = cached_hidden[slot, K - 1 - nr + i]
        for i in range(ni):
            hidden[qs + nr + i] = target_hidden[qs + i]

        for i in range(1, K):
            overrides[req, i - 1] = (
                rec.next_prefill_tokens[i, slot] if num_sampled[req] == 0 else -1
            )

    # 手动算 slot mapping（不使用生产 pad_trailing_draft_slots）。
    slot_mapping = torch.full((num_tokens,), -1, dtype=torch.int64, device=device)
    for req in range(num_reqs):
        qs = qsl[req]
        last = int(last_indices[req])
        bt = rec.block_table[req].to(torch.int64)
        for row in range(qs, last + 1):
            pos = int(positions[row])
            slot_mapping[row] = bt[pos // block_size] * block_size + pos % block_size

    seq_lens = torch.tensor(
        [int(target_seq_lens[req]) - max(0, num_rejected[req] - 1)
         for req in range(num_reqs)],
        dtype=torch.int32, device=device,
    )
    query_start_loc = rec.query_start_loc

    builder = TorchAttentionMetadataBuilder(block_size)
    metadata = builder.build(
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        block_table=rec.block_table,
        slot_mapping=slot_mapping,
        num_reqs=num_reqs,
        uniform_query_len=None,
    )
    attn_metadata = {name: metadata for name in speculator._draft_layer_names()}

    # 独立模型的 KV/cache 由 ReferenceReplay 跨轮维护；这里不读生产 kv_before。
    ref_model.eval()
    cached_ids, cached_hidden = _update_reference_cache(
        cached_ids, cached_hidden, ids, hidden, last_indices, idx, qsl, K
    )

    steps = []
    draft_tokens = []
    with torch.inference_mode():
        step_hidden = hidden
        for step in range(K):
            with set_forward_context(
                attn_metadata,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
                slot_mapping={"slot_mapping": slot_mapping.unsqueeze(0)},
            ):
                ret = ref_model(
                    input_ids=ids,
                    positions=positions,
                    hidden_states=step_hidden,
                    spec_step_idx=step,
                )
            if isinstance(ret, tuple):
                last_hidden, hidden_out = ret
            else:
                last_hidden = hidden_out = ret
            logits = ref_model.compute_logits(last_hidden[last_indices]).clone()
            draft_tokens.append(logits.argmax(dim=-1))
            steps.append(StepCapture(
                input_ids=ids.clone(),
                positions=positions.clone(),
                hidden=step_hidden.clone(),
                slot_mapping=slot_mapping.clone(),
                last_token_indices=last_indices.clone(),
                logits=logits,
                kv=_snapshot_kv_for_model(ref_model),
            ))
            if step < K - 1:
                step_hidden = hidden_out
                next_tokens = torch.where(
                    overrides[:, step] >= 0, overrides[:, step],
                    logits.argmax(dim=-1),
                )
                for req in range(num_reqs):
                    qs = qsl[req]
                    last = int(last_indices[req])
                    ids[qs:last] = ids[qs + 1 : last + 1].clone()
                    ids[last] = next_tokens[req]

    return steps, torch.stack(draft_tokens, dim=1), cached_ids, cached_hidden


def _update_reference_cache(cached_ids, cached_hidden, ids, hidden,
                            last_indices, idx, qsl, K):
    """测试内重写 80A 的 cache_inputs：按持久 slot 快照窗口尾部。"""
    window = K - 1
    for req in range(len(idx)):
        slot = int(idx[req])
        qs = int(qsl[req])
        last = int(last_indices[req])
        start = max(last - window + 1, qs)
        for row in range(start, last + 1):
            col = row - (last - window + 1)
            cached_ids[slot, col] = ids[row]
            cached_hidden[slot, col] = hidden[row]
    return cached_ids, cached_hidden


def _snapshot_kv_for_model(model) -> dict[str, torch.Tensor]:
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, Attention) and module.kv_cache is not None:
            out[name] = module.kv_cache.clone()
    return out


class ReferenceReplay:
    """跨轮维护独立模型的 KV 与 80A cache（不从生产快照恢复）。"""

    def __init__(self, speculator, first_rec: RoundCapture):
        self.speculator = speculator
        self.model = _make_reference_model(speculator)
        for name, module in self.model.named_modules():
            if isinstance(module, Attention):
                assert name in first_rec.kv_before, name
                module.kv_cache = torch.zeros_like(first_rec.kv_before[name])
        self.cached_ids = torch.zeros_like(first_rec.cached_ids)
        self.cached_hidden = torch.zeros_like(first_rec.cached_hidden)

    def assert_state(self, rec: RoundCapture) -> None:
        assert torch.equal(rec.cached_ids, self.cached_ids), "跨轮 cache ids 偏离独立重放"
        assert torch.equal(rec.cached_hidden, self.cached_hidden), "跨轮 cache hidden 偏离独立重放"
        for name, module in self.model.named_modules():
            if isinstance(module, Attention):
                assert torch.equal(rec.kv_before[name], module.kv_cache), (
                    f"跨轮 KV 偏离独立重放：{name}"
                )

    def run(self, rec: RoundCapture):
        steps, tokens, self.cached_ids, self.cached_hidden = _reference_round(
            rec, self.speculator, self.model, self.cached_ids, self.cached_hidden
        )
        return steps, tokens


def _assert_round_matches(rec: RoundCapture, ref_steps, ref_tokens):
    assert rec.draft_tokens is not None
    assert torch.equal(rec.draft_tokens, ref_tokens), (
        f"草稿 token 与独立重放不同：{rec.draft_tokens.tolist()} vs {ref_tokens.tolist()}"
    )
    assert len(rec.steps) == len(ref_steps)
    last_indices = rec.steps[0].last_token_indices
    rows = _valid_rows(rec, last_indices)
    for step, (prod, ref) in enumerate(zip(rec.steps, ref_steps)):
        _assert_tensor_equal(f"step{step} input_ids", prod.input_ids[rows], ref.input_ids[rows])
        _assert_tensor_equal(f"step{step} positions", prod.positions[rows], ref.positions[rows])
        _assert_tensor_equal(f"step{step} hidden", prod.hidden[rows], ref.hidden[rows])
        _assert_tensor_equal(f"step{step} slot", prod.slot_mapping[rows], ref.slot_mapping[rows])
        assert torch.allclose(prod.logits, ref.logits, atol=1e-6, rtol=1e-5), (
            f"step{step} logits 不同：max Δ="
            f"{float((prod.logits - ref.logits).abs().max())}"
        )
        prod_slots = prod.slot_mapping[rows]
        for name in prod.kv:
            assert name in ref.kv
            _assert_tensor_equal(
                f"step{step} {name} K", prod.kv[name][0, prod_slots], ref.kv[name][0, prod_slots]
            )
            _assert_tensor_equal(
                f"step{step} {name} V", prod.kv[name][1, prod_slots], ref.kv[name][1, prod_slots]
            )


def _assert_tensor_equal(label: str, got: torch.Tensor, want: torch.Tensor) -> None:
    assert torch.equal(got, want), (
        f"{label} 不同：max Δ={float((got.float() - want.float()).abs().max())}"
    )


# ---------------------------------------------------------------------------
# 2. 逐模块 KV / logits / 输入
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("k", [2, 4])
def test_per_module_inputs_kv_and_logits_match_independent_replay(k):
    engine, runner = _make_engine(k=k)
    recorder = RoundRecorder(runner)
    try:
        outputs = _run(engine, max_tokens=8)
    finally:
        engine.shutdown()

    assert outputs
    assert recorder.records, "没有捕获到 propose 轮次"
    assert any(
        int(rec.num_rejected.max().item()) > 1 for rec in recorder.records
    ), "测试没有覆盖“拒绝后重排”的轮次"
    assert any(
        int(rec.num_rejected.max().item()) == 0 for rec in recorder.records
    ), "测试没有覆盖全接受/第一遍轮次"
    assert any(
        0 < int(value) < k
        for rec in recorder.records
        for value in rec.num_rejected.tolist()
    ), "测试没有覆盖部分拒绝（0 < num_rejected < K）的下一轮"

    replay = ReferenceReplay(runner.speculator, recorder.records[0])
    for rec in recorder.records:
        replay.assert_state(rec)
        ref_steps, ref_tokens = replay.run(rec)
        _assert_round_matches(rec, ref_steps, ref_tokens)


# ---------------------------------------------------------------------------
# 3. 负向控制：错误 cache / 错误 hidden 必须被比较器抓到
# ---------------------------------------------------------------------------


@requires_cuda
def test_negative_control_corrupted_cache_is_detected():
    engine, runner = _make_engine(k=2)
    recorder = RoundRecorder(runner, corrupt_cache=True)
    try:
        _run(engine)
    finally:
        engine.shutdown()

    assert recorder._corrupted, "没有遇到 num_rejected>1 的轮次，无法验证 cache 负向控制"
    rec = next(rec for rec in recorder.records if rec.corrupted)
    replay = ReferenceReplay(runner.speculator, recorder.records[0])
    for prior in recorder.records:
        ref_steps, _ = replay.run(prior)
        if prior is rec:
            break
    rows = _valid_rows(rec, rec.steps[0].last_token_indices)
    # 生产用了被清零的 cache；独立重放用 rec.cached_*（正确值）。
    assert not torch.equal(
        rec.steps[0].input_ids[rows], ref_steps[0].input_ids[rows]
    ), "cache 被破坏后输入仍然相同：比较器没有抓到"


@requires_cuda
def test_negative_control_stale_hidden_is_detected():
    engine, runner = _make_engine(k=2)
    recorder = RoundRecorder(runner, stale_hidden=True)
    try:
        _run(engine)
    finally:
        engine.shutdown()

    rec = next((r for r in recorder.records if len(r.steps) > 1), None)
    assert rec is not None, "没有捕获到多步轮次"
    replay = ReferenceReplay(runner.speculator, recorder.records[0])
    for prior in recorder.records:
        ref_steps, _ = replay.run(prior)
        if prior is rec:
            break
    rows = _valid_rows(rec, rec.steps[0].last_token_indices)
    # step 0 的 hidden 被当成 step 1 的反馈；第二步骤输入必须与正确重放不同。
    assert not torch.equal(
        rec.steps[1].hidden[rows], ref_steps[1].hidden[rows]
    ), "错误 hidden 没有被比较器抓到"


# ---------------------------------------------------------------------------
# 4. 混合 partial prefill/decode + 两请求重排 + prefix + 抢占恢复
# ---------------------------------------------------------------------------


@requires_cuda
def test_mixed_partial_prefill_reorder_prefix_and_preemption_smoke():
    """不追求吞吐，只要求这些调度形态下 80B eager 仍能产出正确 greedy 文本。"""
    prompts = [("a", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
               ("b", [1, 2, 3, 4, 5, 6, 7]),
               ("c", [5, 6, 7, 8, 9, 10])]
    engine, runner = _make_engine(
        k=2, max_num_seqs=3, budget=8, blocks=6, prefix_caching=True
    )
    core = engine.engine_core.engine_core
    try:
        got = _run(engine, prompts, max_tokens=4)
        preemptions = core.scheduler.num_preemptions
        trace = list(core.scheduler.trace)
    finally:
        engine.shutdown()

    base_engine, _ = _make_engine(
        k=2, spec=False, max_num_seqs=3, budget=8, blocks=6,
        prefix_caching=True,
    )
    try:
        want = _run(base_engine, prompts, max_tokens=4)
    finally:
        base_engine.shutdown()
    assert got == want, f"mixed/prefix/抢占场景输出不同：{got} != {want}"
    assert preemptions > 0, "blocks=6 下三条请求没有触发抢占恢复，用例失去意义"
    assert any(
        sum(entry["hits"].values()) > 0 for entry in trace
    ), "prefix caching 开着，但 trace 里没有任何前缀命中"
    assert any(
        0 < n < 10
        for entry in trace
        for n in entry["scheduled"].values()
    ), "budget=8 下 prompt=10 的请求没有出现 partial prefill"
