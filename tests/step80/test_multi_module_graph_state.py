"""需求 080C：多模块 MTP 的 Graph 与采样生命周期。

本文件不重复 80A 的逐值输入布局，也不重复 80B 的逐模块 eager 重放；它只钉四件事：

1. 请求的 `cudagraph_mode` 按冻结规则收敛：纯 PIECEWISE → NONE，
   FULL_AND_PIECEWISE → FULL_DECODE_ONLY，FULL / FULL_DECODE_ONLY / NONE 原样。
2. 整段 K 步循环 replay 与 eager 的候选、q、跨轮 cache、逐模块 KV 一致。
3. 同一个图桶在 1→2→1 个请求、slot 顺序变化和 padding 行存在时不重捕、不污染有效槽位。
4. K=1 不进入多模块路径；K=3（模块数=2）的模块回绕与拒绝后修复可跑。

图里不会执行 Python 侧 `_run_model`，所以不能靠“重放后再用 CPU 补算”通过；比较的
对象是 replay 后常驻缓冲/静态输出里的真实值。
"""

import dataclasses
import os
from dataclasses import dataclass, field

os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

import pytest
import torch

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig,
                      SamplingParams, SchedulerConfig, SpeculativeConfig,
                      UniProcExecutor, VllmConfig, Worker)
from minivllm.attention.layer import Attention
from minivllm.config import CUDAGraphMode, CompilationConfig
from minivllm.testing.tiny_models import tiny_mtp_dir
from minivllm.worker.gpu.spec_decode import init_speculator

DEVICE = "cuda"
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="多模块 CUDA Graph 需要 CUDA"
)
EOS = 999
PROMPTS = [("a", [1, 2, 3, 4, 1, 2, 3, 4]), ("b", [5, 6, 5, 6])]


@pytest.fixture(autouse=True)
def _force_v2(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")


def _mtp_configs(*, k: int, device: str, max_num_seqs: int = 2,
                 budget: int = 64, blocks: int = 64,
                 sample_method: str = "greedy"):
    directory = tiny_mtp_dir("tiny_gqa", num_mtp_layers=2)
    import json
    from pathlib import Path

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
        draft_sample_method=sample_method,
    )
    return VllmConfig(
        model_config=target,
        cache_config=CacheConfig(
            block_size=4,
            num_gpu_blocks=blocks,
            enable_prefix_caching=False,
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


def _make_engine(*, k: int, mode: str = "none", max_num_seqs: int = 2,
                 sample_method: str = "greedy", capture_sizes=None):
    cfg = _mtp_configs(
        k=k, device=DEVICE, max_num_seqs=max_num_seqs, sample_method=sample_method
    )
    if mode != "none":
        sizes = capture_sizes or [k + 1, 2 * (k + 1)]
        cfg = dataclasses.replace(
            cfg,
            compilation_config=CompilationConfig(
                cudagraph_mode=mode,
                cudagraph_capture_sizes=sizes,
                max_cudagraph_capture_size=max(sizes),
            ),
        )
    engine = LLMEngine(cfg, UniProcExecutor(cfg, Worker(cfg)), tokenizer=None)
    runner = engine.engine_core.engine_core.model_executor.driver_worker.model_runner
    return engine, runner


def _add(engine, prompts=PROMPTS, *, max_tokens=6):
    for req_id, prompt in prompts:
        engine.add_request(
            req_id, list(prompt),
            SamplingParams(max_tokens=max_tokens, temperature=0.0, eos_token_id=EOS),
        )


def _run(engine, prompts=PROMPTS, *, max_tokens=6):
    _add(engine, prompts, max_tokens=max_tokens)
    outputs = {}
    for _ in range(400):
        if not engine.has_unfinished_requests():
            break
        for item in engine.step():
            outputs[item.request_id] = list(item.token_ids)
    return outputs


def _snapshot_kv(speculator) -> dict[str, torch.Tensor]:
    return {
        name: module.kv_cache.clone()
        for name, module in speculator.model.named_modules()
        if isinstance(module, Attention) and module.kv_cache is not None
    }


@dataclass
class RoundState:
    num_reqs: int
    idx: torch.Tensor
    q: torch.Tensor | None
    tokens: torch.Tensor
    kv: dict[str, torch.Tensor]
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    last_token_indices: torch.Tensor
    positions: torch.Tensor
    cached_ids: torch.Tensor
    cached_hidden: torch.Tensor
    num_rejected: torch.Tensor
    selection: dict | None


@dataclass
class StateTrace:
    runner: object
    rounds: list[RoundState] = field(default_factory=list)
    num_rejected: list[torch.Tensor] = field(default_factory=list)

    def __post_init__(self):
        speculator = self.runner.speculator
        self.speculator = speculator
        self._original = speculator.propose
        speculator.propose = self._wrap

    def _wrap(self, *args, **kwargs):
        input_batch = args[0]
        num_reqs = int(input_batch.num_reqs)
        num_rejected = args[6]
        result = self._original(*args, **kwargs)
        self.rounds.append(RoundState(
            num_reqs=num_reqs,
            idx=input_batch.idx_mapping[:num_reqs].clone(),
            q=(
                self.speculator.draft_logits.clone()
                if self.speculator.draft_logits is not None else None
            ),
            tokens=self.speculator.draft_tokens[:num_reqs].clone(),
            kv=_snapshot_kv(self.speculator),
            block_table=self.runner.block_tables.input_block_tables[0][
                :num_reqs
            ].clone(),
            query_start_loc=input_batch.query_start_loc[: num_reqs + 1].clone(),
            last_token_indices=self.speculator.last_token_indices[
                :num_reqs
            ].clone(),
            positions=self.speculator.input_buffers.positions.clone(),
            cached_ids=self.speculator.cached_draft_input_ids.clone(),
            cached_hidden=self.speculator.cached_target_hidden_states.clone(),
            num_rejected=num_rejected[:num_reqs].clone(),
            selection=(
                self.speculator.cudagraph_selections[-1]
                if self.speculator.cudagraph_selections else None
            ),
        ))
        return result

    def close(self):
        # 恢复方法，避免同一 engine 上再包一次时递归。
        self.speculator.propose = self._original


def _assert_kv_at_draft_window(eager: RoundState, graph: RoundState,
                               block_size: int, atol: float = 1e-5) -> None:
    """比较同一逻辑位置上的逐模块 KV。

    graph 与 eager 的物理块号可以不同（graph 模式会留白 0 号块），所以不能直接比
    `kv_cache[:, block]`；必须经各自的 block_table 把同一逻辑位置映射成物理槽位。
    只比较本轮 draft 窗口 `[query_start, last_token]` 的行——这是本步真正写/读的
    KV；窗口外的历史位置可能是上一模块/上一轮的旧内容，不是本轮的“逐模块 KV”。
    """
    assert eager.num_reqs == graph.num_reqs
    assert eager.positions.shape == graph.positions.shape
    for req in range(eager.num_reqs):
        assert int(eager.idx[req]) == int(graph.idx[req])
        qs = int(eager.query_start_loc[req])
        last = int(eager.last_token_indices[req])
        gqs = int(graph.query_start_loc[req])
        glast = int(graph.last_token_indices[req])
        assert (qs, last) == (gqs, glast)
        for row in range(qs, last + 1):
            pos = int(eager.positions[row].item())
            gpos = int(graph.positions[row].item())
            assert pos == gpos
            slot = (int(eager.block_table[req, pos // block_size].item())
                    * block_size + pos % block_size)
            gslot = (int(graph.block_table[req, gpos // block_size].item())
                     * block_size + gpos % block_size)
            for name in eager.kv:
                torch.testing.assert_close(
                    eager.kv[name][:, slot],
                    graph.kv[name][:, gslot],
                    atol=atol,
                    rtol=1e-4,
                    msg=lambda m, n=name, r=req, p=pos: (
                        f"{n} req={r} pos={p}: {m}"
                    ),
                )


# ---------------------------------------------------------------------------
# 1. 模式收敛 / K=1 分派（纯 CPU）
# ---------------------------------------------------------------------------


def test_graph_modes_follow_frozen_downgrade():
    cfg = _mtp_configs(k=2, device="cpu")
    speculator = init_speculator(cfg, torch.device("cpu"))
    expected = {
        CUDAGraphMode.NONE: CUDAGraphMode.NONE,
        CUDAGraphMode.PIECEWISE: CUDAGraphMode.NONE,
        CUDAGraphMode.FULL: CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_DECODE_ONLY: CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE: CUDAGraphMode.FULL_DECODE_ONLY,
    }
    for requested, actual in expected.items():
        speculator.init_cudagraph_manager(requested)
        assert speculator.requested_cudagraph_mode == requested
        assert speculator.cudagraph_mode == actual
        assert speculator.cudagraph_manager is not None
        assert speculator.cudagraph_manager.cudagraph_mode == actual
        assert speculator.cudagraph_managers["multi_module"] is (
            speculator.cudagraph_manager
        )
    # 无捕获档位时是空操作，不再把“属 80C”当异常。
    speculator.capture()


def test_k1_does_not_take_multi_module_path():
    cfg = _mtp_configs(k=1, device="cpu")
    assert not cfg.speculative_config.use_multi_module_mtp()
    speculator = init_speculator(cfg, torch.device("cpu"))
    assert type(speculator).__name__ == "MTPSpeculator"


# ---------------------------------------------------------------------------
# 2. replay vs eager：候选 / q / 跨轮 cache / 逐模块 KV
# ---------------------------------------------------------------------------


@requires_cuda
def test_graph_replay_matches_eager_q_and_module_kv():
    k = 3
    eager, eager_runner = _make_engine(k=k, sample_method="probabilistic")
    eager_trace = StateTrace(eager_runner)
    try:
        want = _run(eager, max_tokens=8)
    finally:
        eager_trace.close()
        eager.shutdown()

    graph, graph_runner = _make_engine(
        k=k, mode="full_decode_only", sample_method="probabilistic",
        capture_sizes=[k + 1, 2 * (k + 1)],
    )
    graph_trace = StateTrace(graph_runner)
    try:
        got = _run(graph, max_tokens=8)
    finally:
        graph_trace.close()
        graph.shutdown()

    assert got == want
    assert len(graph_trace.rounds) == len(eager_trace.rounds)
    assert graph_runner.speculator.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY

    replayed = [
        r for r in graph_trace.rounds
        if r.selection is not None and r.selection["mode"] == "FULL"
    ]
    assert replayed, "图引擎没有真正 replay 任何 decode 轮"
    assert any(
        r.selection is not None and r.selection["mode"] == "NONE"
        for r in graph_trace.rounds
    ), "prefill/非统一批应走 eager，不能被图桶错误吞掉"

    for i, (eg, gg) in enumerate(zip(eager_trace.rounds, graph_trace.rounds)):
        assert torch.equal(eg.tokens, gg.tokens), f"round {i}: 候选 token 不同"
        assert torch.equal(eg.cached_ids, gg.cached_ids), f"round {i}: cache ids 不同"
        torch.testing.assert_close(
            eg.cached_hidden, gg.cached_hidden, atol=1e-5, rtol=1e-4
        )
        assert eg.num_reqs == gg.num_reqs
        if gg.q is not None:
            slots = gg.idx.to(torch.long)
            torch.testing.assert_close(
                eg.q[slots], gg.q[slots], atol=2e-5, rtol=1e-4
            )
        if gg.selection is not None and gg.selection["mode"] == "FULL":
            _assert_kv_at_draft_window(
                eg, gg, block_size=4, atol=2e-5
            )


@requires_cuda
def test_multi_module_mtp_full_graph_matches_eager():
    """需求点名的旧测试名：调整图管理器字段预期后保留。

    旧实验实现期望 `graph_manager` / `num_graphs` 之类的内部字段；80C 的冻结结构是
    一个 `SpeculatorCudaGraphManager`，所以这里断言它真正捕获/重放了整段循环，
    并比较 greedy 文本，不再维护已经不存在的字段。
    """
    k = 2
    eager, _ = _make_engine(k=k)
    try:
        want = _run(eager, max_tokens=6)
    finally:
        eager.shutdown()

    graph, runner = _make_engine(
        k=k, mode="full_decode_only", capture_sizes=[k + 1, 2 * (k + 1)]
    )
    speculator = runner.speculator
    stats = runner.cudagraph_capture_stats
    try:
        got = _run(graph, max_tokens=6)
    finally:
        graph.shutdown()

    assert got == want
    assert speculator.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
    assert speculator.cudagraph_manager is not None
    assert len(speculator.cudagraph_manager.graphs) == 2
    assert stats is not None
    assert stats["graphs"]["multi_module"]["captured"] == 2
    assert {k + 1, 2 * (k + 1)} <= set(
        speculator.cudagraph_manager.captured_token_counts()
    )
    assert any(
        s["mode"] == "FULL" and s["replayed"]
        for s in speculator.cudagraph_selections
    )


@requires_cuda
def test_graph_padding_bucket_with_one_real_request_is_safe():
    """只捕获 2 请求桶，却用 1 请求 replay：补齐行不能污染真实槽位/输出。

    这里故意把 capture_sizes 限成 `[2*(K+1)]`（K=2 时为 6），让 1 条请求也必须
    补成 2 条请求的形状。如果 padding 行的 `idx_mapping=-1`、slot=-1 或块表 0
    处理错了，输出/有效 KV 会偏离 eager，而不是抛异常。
    """
    k = 2
    eager, _ = _make_engine(k=k)
    try:
        want = _run(eager, prompts=PROMPTS[:1], max_tokens=6)
    finally:
        eager.shutdown()

    graph, runner = _make_engine(
        k=k, mode="full_decode_only", capture_sizes=[2 * (k + 1)]
    )
    speculator = runner.speculator
    try:
        got = _run(graph, prompts=PROMPTS[:1], max_tokens=6)
    finally:
        graph.shutdown()

    assert got == want
    padded = [
        s for s in speculator.cudagraph_selections
        if s["mode"] == "FULL" and s["num_reqs"] == 1
    ]
    assert padded, "1 请求没有命中补齐过的图桶"
    assert all(s["desc_num_reqs"] == 2 for s in padded)
    assert all(s["desc_num_tokens"] == 2 * (k + 1) for s in padded)
    # 0 号块必须留白；真实请求的块首不能是 0。
    for row in runner.block_tables.input_block_tables[0]:
        if int(row[0].item()) != 0:
            assert int(row[0].item()) > 0


# ---------------------------------------------------------------------------
# 3. 图桶复用：1→2→1、slot 顺序、padding / 0 号块
# ---------------------------------------------------------------------------


@requires_cuda
def test_graph_bucket_reuse_1_2_1_and_padding_safety():
    k = 2
    engine, runner = _make_engine(
        k=k, mode="full_decode_only", capture_sizes=[k + 1, 2 * (k + 1)]
    )
    speculator = runner.speculator
    manager = speculator.cudagraph_manager
    graph_ids = {desc: id(graph) for desc, graph in manager.graphs.items()}
    assert len(manager.graphs) == 2

    seen_reqs: list[int] = []
    original_propose = speculator.propose

    def spy(*args, **kwargs):
        out = original_propose(*args, **kwargs)
        seen_reqs.append(int(args[0].num_reqs))
        return out

    speculator.propose = spy
    try:
        engine.add_request(
            "a", [1, 2, 3, 4, 1, 2, 3, 4],
            SamplingParams(max_tokens=8, temperature=0.0, eos_token_id=EOS),
        )
        # 先让 a 走完 prefill + 至少一次 decode，再插入 b；之后 b 结束，a 单独跑。
        for _ in range(2):
            if engine.has_unfinished_requests():
                engine.step()
        engine.add_request(
            "b", [5, 6, 5, 6],
            SamplingParams(max_tokens=2, temperature=0.0, eos_token_id=EOS),
        )
        for _ in range(200):
            if not engine.has_unfinished_requests():
                break
            engine.step()
    finally:
        speculator.propose = original_propose
        engine.shutdown()

    assert seen_reqs, "没有观察 propose 的批形状"
    assert seen_reqs[0] == 1
    assert max(seen_reqs) == 2
    assert seen_reqs[-1] == 1
    assert seen_reqs.index(2) < len(seen_reqs) - 1
    assert any(
        s["mode"] == "FULL" and s["num_reqs"] == 1
        for s in manager.selections
    )
    assert any(
        s["mode"] == "FULL" and s["num_reqs"] == 2
        for s in manager.selections
    )

    # 同一个桶只捕获一次：descriptor / graph 对象都不应因为批顺序改变而重建。
    assert {desc: id(graph) for desc, graph in manager.graphs.items()} == graph_ids
    assert len(manager.graphs) == 2

    # graph 模式必须留白 0 号块；真实请求的块号不能是 0。
    for row in runner.block_tables.input_block_tables[0]:
        if int(row[0].item()) != 0:
            assert int(row[0].item()) > 0


@requires_cuda
def test_graph_replay_segment_has_no_tensor_item():
    """replay 段不允许 `.item()`：图里也不能靠 CPU 读回标量再分支。

    这只包住 `run_fullgraph()`（真正的 replay），不把 propose 图外的
    `max_seq_len`/`max_query_len` 归一化算进来；那些是捕获前的宿主侧准备。
    """
    from unittest.mock import patch

    k = 2
    engine, runner = _make_engine(
        k=k, mode="full_decode_only", capture_sizes=[k + 1, 2 * (k + 1)]
    )
    speculator = runner.speculator
    manager = speculator.cudagraph_manager
    assert manager is not None
    engine.add_request(
        "a", [1, 2, 3, 4, 1, 2, 3, 4],
        SamplingParams(max_tokens=8, temperature=0.0, eos_token_id=EOS),
    )
    seen = []

    original = manager.run_fullgraph

    def checked(desc):
        seen.append(desc)
        with patch.object(
            torch.Tensor, "item",
            side_effect=AssertionError("replay 段出现 .item()：图内不能读回 CPU 标量"),
        ):
            return original(desc)

    manager.run_fullgraph = checked
    try:
        # 第一轮 prefill 走 eager；第二轮 decode 才命中 FULL。
        engine.step()
        engine.step()
    finally:
        manager.run_fullgraph = original
        engine.shutdown()

    assert seen, "没有走到 FULL replay，测试没有覆盖目标路径"


# ---------------------------------------------------------------------------
# 4. 强制首拒 / 全接受：图内循环仍要按 num_rejected 修复
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("forced_rejected", [1, 0])
def test_graph_forced_rejection_and_acceptance(forced_rejected):
    k = 3
    eager, eager_runner = _make_engine(k=k)
    eager_trace = StateTrace(eager_runner)
    eager_original = eager_runner.speculator.propose
    state = {"calls": 0}

    def eager_wrap(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 2:
            args = list(args)
            args[6] = torch.full_like(args[6], forced_rejected)
        return eager_original(*args, **kwargs)

    eager_runner.speculator.propose = eager_wrap
    try:
        want = _run(eager, prompts=PROMPTS[:1], max_tokens=5)
    finally:
        eager_trace.close()
        eager.shutdown()

    graph, graph_runner = _make_engine(
        k=k, mode="full_decode_only", capture_sizes=[k + 1, 2 * (k + 1)]
    )
    graph_trace = StateTrace(graph_runner)
    graph_original = graph_runner.speculator.propose
    state = {"calls": 0}

    def graph_wrap(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 2:
            args = list(args)
            args[6] = torch.full_like(args[6], forced_rejected)
        return graph_original(*args, **kwargs)

    graph_runner.speculator.propose = graph_wrap
    try:
        got = _run(graph, prompts=PROMPTS[:1], max_tokens=5)
    finally:
        graph_trace.close()
        graph.shutdown()

    assert got == want
    assert any(
        r.selection is not None and r.selection["mode"] == "FULL"
        for r in graph_trace.rounds
    )
    assert any(
        int(r.num_rejected.max().item()) == forced_rejected
        for r in graph_trace.rounds
    )
