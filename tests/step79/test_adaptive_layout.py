"""79 关：V2 DSpark 端到端的预算/布局接线。

单元测试已经在 `test_adaptive_budget.py` 里手算过 manager；这里只回答三个问题：

1. 预算真的接到了 target 的 `num_tokens` / `query_start_loc` / `cu_num_logits` 上吗？
2. CPU 的"均分上界"有没有被误当成 GPU 真实前缀？
3. 固定枚数（adaptive=False）与满预算（adaptive=True）是否给出同一批 token？
"""

from __future__ import annotations

import pytest
import torch

from adaptive_helpers import (DEVICE, RecordBatch, make_adaptive_engine,
                              run_adaptive)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="V2 DSpark 需要 CUDA")

PROMPTS = [("a", [1, 2, 3, 4]), ("b", [5, 6, 7])]


def _set_full_budget(runner):
    m = runner.adaptive_verification
    m.set_cost_curves([(1, 0.0), (4, 0.0)],
                      [(n, 1.0) for n in range(0, 65)])


def _set_zero_budget(runner):
    m = runner.adaptive_verification
    m.set_cost_curves([(1, 100.0), (4, 100.0)],
                      [(n, 1000.0 * n) for n in range(0, 65)])


def _set_intermediate_budget(runner):
    m = runner.adaptive_verification
    m.set_cost_curves([(1, 10.0), (4, 10.0)],
                      [(0, 1.0), (5, 1.0), (64, 1000.0)])


@requires_cuda
def test_full_budget_matches_fixed_verification():
    """满预算 = 每请求验证全部 K 枚，greedy 输出必须与 adaptive 关闭时逐 token 相同。"""
    off_engine, _, _ = make_adaptive_engine(adaptive=False)
    off = run_adaptive(off_engine, PROMPTS, max_tokens=4)

    on_engine, _, runner = make_adaptive_engine(adaptive=True)
    _set_full_budget(runner)
    rec = RecordBatch(runner)
    try:
        on = run_adaptive(on_engine, PROMPTS, max_tokens=4)
    finally:
        rec.stop()
    assert on == off
    decode_rows = [r for r in rec.rows if sum(r["scheduled_drafts"]) > 0]
    assert decode_rows
    assert all(r["num_draft_tokens"] == sum(r["scheduled_drafts"])
               and r["cu_num_logits_np_exact"] for r in decode_rows)


@requires_cuda
def test_zero_budget_verifies_only_bonus_rows():
    """预算 0：真实 `cu_num_logits` 每请求只剩 1 行，且 GPU offsets 是精确值。"""
    engine, _, runner = make_adaptive_engine(adaptive=True)
    _set_zero_budget(runner)
    rec = RecordBatch(runner)
    try:
        out = run_adaptive(engine, PROMPTS, max_tokens=3)
    finally:
        rec.stop()
    assert out["a"] and out["b"]
    decode_rows = [r for r in rec.rows if sum(r["scheduled_drafts"]) > 0]
    assert decode_rows
    # 第一步 decode 的 stale 置信度还是初始化值 1.0（上游同款：没有上一轮可读），
    # 在"每多验一行更贵"的曲线下它仍可能选满；从第二步拿到真实置信度后必须掉到 0。
    assert any(r["num_draft_tokens"] == 0 for r in decode_rows)
    for row in decode_rows[1:]:
        n = row["num_reqs"]
        assert row["num_draft_tokens"] == 0
        assert row["per_req_drafts"] == row["scheduled_drafts"]
        assert row["cu_num_logits"] == list(range(n + 1))
        assert row["cu_num_logits_np_exact"] is True
        # 每条 decode 请求一个非草稿行（bonus），所以 query_start_loc 也是 0..n。
        assert row["query_start_loc"] == list(range(n + 1))


@requires_cuda
def test_intermediate_budget_uses_gpu_prefixes_not_cpu_upper_bound():
    """CPU 侧均分只是上界；真实 capacities 之和 == 预算，且每请求是连续前缀。"""
    engine, _, runner = make_adaptive_engine(adaptive=True)
    _set_intermediate_budget(runner)
    rec = RecordBatch(runner)
    try:
        out = run_adaptive(engine, PROMPTS, max_tokens=4)
    finally:
        rec.stop()
    assert out["a"] and out["b"]
    rows = [r for r in rec.rows if r["num_draft_tokens"] == 3]
    assert rows
    for row in rows:
        n = row["num_reqs"]
        scheduled = row["scheduled_drafts"]
        # CPU 上界仍按"原调度 + 均分"算，不能直接当真实 offsets 使用。
        assert row["cu_num_logits_np"] != row["cu_num_logits"] or n == 1
        assert row["cu_num_logits_np_exact"] is False
        caps = []
        for i in range(n):
            query_len = row["query_start_loc"][i + 1] - row["query_start_loc"][i]
            logits_len = row["cu_num_logits"][i + 1] - row["cu_num_logits"][i]
            cap = logits_len - 1
            caps.append(cap)
            assert 0 <= cap <= scheduled[i]
            assert query_len == cap + 1, (row, caps)
        assert sum(caps) == 3
        assert row["num_tokens"] == sum(caps) + n


@requires_cuda
def test_gpu_current_confidence_decides_prefixes_not_cpu_stale():
    """强制 a 的置信度全 1、b 全 0：预算 3 全给 a，CPU 的均分 [2,1] 不能生效。"""
    engine, _, runner = make_adaptive_engine(adaptive=True)
    _set_intermediate_budget(runner)
    original = runner.speculator.propose

    def propose_with_fixed_confidence(*args, **kwargs):
        out = original(*args, **kwargs)
        input_batch = args[0]
        for i, req_id in enumerate(input_batch.req_ids):
            value = 1.0 if req_id == "a" else 0.0
            runner.speculator.draft_token_confidence_probs[i].fill_(value)
        return out

    runner.speculator.propose = propose_with_fixed_confidence
    rec = RecordBatch(runner)
    try:
        out = run_adaptive(engine, PROMPTS, max_tokens=4)
    finally:
        rec.stop()
    assert out["a"] and out["b"]
    rows = [r for r in rec.rows if r["num_draft_tokens"] == 3]
    assert rows
    # 两个请求都带草稿，顺序按调度器输入（a、b）。a 拿 3 枚、b 拿 0 枚。
    row = next(r for r in rows if r["req_ids"] == ["a", "b"])
    caps = [row["cu_num_logits"][i + 1] - row["cu_num_logits"][i] - 1
            for i in range(row["num_reqs"])]
    assert caps == [3, 0]


@requires_cuda
def test_profiling_installs_real_cost_tables_once():
    """`capture_model()` 里的真实 GPU event profiling 必须生成非空、有限的成本表。"""
    engine, _, runner = make_adaptive_engine(adaptive=True)
    try:
        manager = runner.adaptive_verification
        assert manager.cost_tables is not None
        draft, verify = manager.cost_curves
        assert draft and verify
        assert all(v >= 0 for _, v in draft)
        assert all(v > 0 for _, v in verify)
        assert manager.cost_tables[0].shape == (runner.max_num_reqs + 1,)
        assert manager.cost_tables[1].shape == (runner.max_num_tokens + 1,)
        again = runner.profile_adaptive_verification()
        assert again["cached"] is True
    finally:
        engine.shutdown()


@requires_cuda
def test_adaptive_survives_full_draft_cudagraph():
    """草稿侧 FULL 图 + target eager 变长批可以共存；预算/前缀仍然正确。"""
    engine, _, runner = make_adaptive_engine(adaptive=True, mode="full",
                                             capture_sizes=(6, 12))
    _set_intermediate_budget(runner)
    rec = RecordBatch(runner)
    try:
        out = run_adaptive(engine, PROMPTS, max_tokens=4)
    finally:
        rec.stop()
    assert out["a"] and out["b"]
    rows = [r for r in rec.rows if r["num_draft_tokens"] == 3]
    assert rows
    assert all(r["query_start_loc"][-1] == r["num_tokens"] for r in rec.rows)


@requires_cuda
def test_abort_and_slot_reuse_do_not_consume_stale_confidence():
    """取消一条请求、新请求复用 slot：不能拿旧请求的置信度选预算/前缀。"""
    from minivllm import SamplingParams

    engine, _, runner = make_adaptive_engine(adaptive=True, max_num_seqs=2)
    _set_intermediate_budget(runner)
    rec = RecordBatch(runner)
    try:
        engine.add_request("a", [1, 2, 3], SamplingParams(
            max_tokens=8, temperature=0.0, eos_token_id=99))
        engine.add_request("b", [4, 5, 6], SamplingParams(
            max_tokens=8, temperature=0.0, eos_token_id=99))
        for _ in range(2):
            engine.step()
        engine.abort_request(["a"])
        engine.add_request("c", [7, 8, 9], SamplingParams(
            max_tokens=8, temperature=0.0, eos_token_id=99))
        outputs = {}
        for _ in range(50):
            if not engine.has_unfinished_requests():
                break
            for out in engine.step():
                outputs[out.request_id] = list(out.token_ids)
    finally:
        rec.stop()
    assert outputs["c"], outputs
    for row in rec.rows:
        if not row["scheduled_drafts"]:
            continue
        assert 0 <= row["num_draft_tokens"] <= sum(row["scheduled_drafts"])
        for i in range(row["num_reqs"]):
            cap = row["cu_num_logits"][i + 1] - row["cu_num_logits"][i] - 1
            assert 0 <= cap <= row["per_req_drafts"][i]


@requires_cuda
def test_probabilistic_drafts_are_trimmed_safely():
    """概率草稿（需要 q）裁剪后仍走拒绝采样；不能因行数变短而拿错 q 列。"""
    engine, _, runner = make_adaptive_engine(
        adaptive=True, draft_sample_method="probabilistic")
    _set_intermediate_budget(runner)
    rec = RecordBatch(runner)
    try:
        out = run_adaptive(engine, PROMPTS, max_tokens=4)
    finally:
        rec.stop()
    assert out["a"] and out["b"]
    rows = [r for r in rec.rows if r["num_draft_tokens"] == 3]
    assert rows
    for row in rows:
        # 真实 logits 行数 == 预算 + 每个请求一个 bonus；draft_logits 的列按同一顺序取。
        assert row["cu_num_logits"][-1] == 3 + row["num_reqs"]


@requires_cuda
def test_missing_confidence_head_is_rejected_at_engine_init():
    """配置打开 adaptive 但 checkpoint 没有 confidence 头 → 加载期明确报错。"""
    with pytest.raises(ValueError, match="confidence"):
        make_adaptive_engine(adaptive=True, confidence_head=False)


@requires_cuda
def test_backend_without_device_cpu_mismatch_support_is_rejected(monkeypatch):
    """能力检查失败必须在 Runner 初始化时炸，不能静默按 CPU 上界跑。"""
    from minivllm.attention.backends.torch_sdpa import TorchAttentionBackend

    monkeypatch.setattr(
        TorchAttentionBackend, "supports_device_cpu_query_lens_mismatch",
        staticmethod(lambda: False))
    with pytest.raises(ValueError, match="does not support"):
        make_adaptive_engine(adaptive=True)


@requires_cuda
def test_intermediate_budget_with_logprobs_uses_exact_offsets():
    """中间预算 + 用户要 logprobs：每个输出 token 必须有且只有一条 logprob。

    这条用例钉住 `InputBatch.cu_num_logits_np_exact=False` 时拒绝采样器的 D2H 修正：
    拿 CPU 均分上界切行会把后一条请求的 logits 算到前一条头上（不报错、数字错）。
    """
    from minivllm import SamplingParams

    engine, _, runner = make_adaptive_engine(adaptive=True)
    _set_intermediate_budget(runner)
    rec = RecordBatch(runner)
    try:
        for rid, prompt in PROMPTS:
            engine.add_request(rid, list(prompt), SamplingParams(
                max_tokens=4, temperature=0.0, eos_token_id=99, logprobs=2))
        outputs = {}
        for _ in range(50):
            if not engine.has_unfinished_requests():
                break
            for out in engine.step():
                outputs[out.request_id] = (list(out.token_ids), out.logprobs)
    finally:
        rec.stop()
    assert outputs["a"][0] and outputs["b"][0]
    for token_ids, logprobs in outputs.values():
        assert len(logprobs) == len(token_ids)
        # logprobs=2 时若采样 token 不在 top-2，会额外带一条（rank>2）。
        assert all(2 <= len(row) <= 3 for row in logprobs)
    assert any(r["cu_num_logits_np_exact"] is False for r in rec.rows)


@requires_cuda
@pytest.mark.parametrize("method,kwargs", [
    ("block", {}),
    ("synthetic", {"synthetic_acceptance_rates": [0.8, 0.4, 0.2]}),
])
def test_adaptive_trimming_works_with_block_and_synthetic_verification(method, kwargs):
    """验证方式不是 standard 时，裁剪后的 `cu_num_logits` 仍被验证内核正确消费。"""
    engine, _, runner = make_adaptive_engine(
        adaptive=True, spec_kwargs={"rejection_sample_method": method, **kwargs})
    _set_intermediate_budget(runner)
    rec = RecordBatch(runner)
    try:
        out = run_adaptive(engine, PROMPTS, max_tokens=4)
    finally:
        rec.stop()
    assert out["a"] and out["b"]
    rows = [r for r in rec.rows if r["num_draft_tokens"] == 3]
    assert rows
    assert all(r["cu_num_logits"][-1] == 3 + r["num_reqs"] for r in rows)


@requires_cuda
def test_adaptive_target_varlen_graph_matches_eager():
    """Phase 2：自适应裁剪后的 1..1+K 行 decode 批命中 target varlen FULL 图。"""
    prompts = [("a", [1, 2, 3, 4]), ("b", [5, 6, 7])]
    eager_engine, _, eager_runner = make_adaptive_engine(
        adaptive=True, mode="none")
    _set_intermediate_budget(eager_runner)
    eager_out = run_adaptive(eager_engine, prompts, max_tokens=4)

    graph_engine, _, graph_runner = make_adaptive_engine(
        adaptive=True, mode="full", capture_sizes=(4, 8), max_num_seqs=2)
    _set_intermediate_budget(graph_runner)
    rec = RecordBatch(graph_runner)
    try:
        graph_out = run_adaptive(graph_engine, prompts, max_tokens=4)
    finally:
        rec.stop()
    manager = graph_runner.target_cudagraph_manager
    assert manager is not None and manager.varlen_decode is True
    assert graph_out == eager_out
    full = [s for s in manager.selections if s["mode"] == "FULL"]
    assert full
    assert all(s["desc_num_tokens"] >= s["num_tokens"] for s in full)
    assert all(s["desc_num_reqs"] == 2 for s in full)
    assert graph_runner.adaptive_verification._cudagraph_limit == 8
    # 真实 5 行命中捕获 8 行；图内 varlen 不按补齐后的 4 行/请求算。
    assert any(r["num_reqs"] == 2 and r["num_tokens"] == 5
               for r in rec.rows if r["num_draft_tokens"] == 3)


@requires_cuda
def test_adaptive_triton_backend_is_really_always():
    """Triton paged 后端声明 ALWAYS：adaptive 走 target FULL（含混合批），不再 bypass。"""
    from minivllm.config import CUDAGraphMode

    eager_engine, _, eager_runner = make_adaptive_engine(
        adaptive=True, mode="none", attention_backend="torch")
    _set_intermediate_budget(eager_runner)
    eager_out = run_adaptive(eager_engine, PROMPTS, max_tokens=4)

    graph_engine, _, graph_runner = make_adaptive_engine(
        adaptive=True, mode="full", capture_sizes=(4, 8), max_num_seqs=2,
        attention_backend="triton")
    _set_intermediate_budget(graph_runner)
    assert graph_runner.attention_backend_name == "triton"
    assert graph_runner.cudagraph_mode == CUDAGraphMode.FULL
    assert graph_runner._target_varlen_cudagraphs_enabled() is True
    graph_out = run_adaptive(graph_engine, PROMPTS, max_tokens=4)

    assert graph_out == eager_out
    assert graph_runner.adaptive_verification._cudagraph_limit == 8
    manager = graph_runner.target_cudagraph_manager
    assert manager is not None and manager.graphs
    assert manager.varlen_decode is True
    assert any(s["mode"] == "FULL" and s["replayed"]
               for s in manager.selections)


@requires_cuda
def test_adaptive_triton_attn_unified_block16_matches_eager():
    """block16 时走 vendored 上游 unified_attention，而不是本地逐行 fallback。"""
    from minivllm.config import CUDAGraphMode

    eager_engine, _, eager_runner = make_adaptive_engine(
        adaptive=True, mode="none", attention_backend="torch", block_size=16)
    _set_intermediate_budget(eager_runner)
    eager_out = run_adaptive(eager_engine, PROMPTS, max_tokens=4)

    graph_engine, _, graph_runner = make_adaptive_engine(
        adaptive=True, mode="full", capture_sizes=(4, 8), max_num_seqs=2,
        attention_backend="triton_attn", block_size=16)
    _set_intermediate_budget(graph_runner)
    assert graph_runner.attention_backend_name == "triton_attn"
    assert graph_runner.attention_backend.get_name() == "TRITON_ATTN"
    assert graph_runner.cudagraph_mode == CUDAGraphMode.FULL
    assert graph_runner._target_varlen_cudagraphs_enabled() is True
    graph_out = run_adaptive(graph_engine, PROMPTS, max_tokens=4)
    assert graph_out == eager_out
    assert graph_runner.adaptive_verification._cudagraph_limit == 8
