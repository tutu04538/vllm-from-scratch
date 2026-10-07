"""step65：MTP 的**端到端**行为（需求 065 §3.4/§3.5/§4）。

要钉住的四件事：

1. **greedy 与非投机逐 token 相同**（草稿只是候选，验证由 target 做）；
2. **草稿真的被提出来**，且走的是**通用迭代提议**（`EagleProposer`，不是另写一个 MTP 主循环）；
3. **自回归步之间真的回灌了上一枚的 hidden**——不回灌（一直用第一遍那份）不会报错，
   只会让第 2 枚起不再条件于第 1 枚；用例用"冻结回灌"的反证把它抓出来；
4. **KV 的有效范围**：首轮/被拒后/抢占恢复下 draft 的 KV 都不会越界，也不会丢掉重算。
"""

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,  # noqa: E402
                      SchedulerConfig, SpeculativeConfig, UniProcExecutor, VllmConfig, Worker)
from minivllm.spec_decode.eagle import EagleProposer  # noqa: E402
from minivllm.testing.tiny_models import tiny_mtp_dir  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPTS = [("a", [1, 2, 3, 4, 1, 2, 3, 4]), ("b", [5, 6, 5, 6])]
NAMINGS = ["mtp", "absolute"]


def make_engine(*, naming="mtp", k=1, with_spec=True, max_num_seqs=2, budget=64,
                max_model_len=64, num_gpu_blocks=32):
    directory = tiny_mtp_dir("tiny_gqa", naming=naming)
    hf_config = json.loads((Path(directory) / "config.json").read_text())
    spec = None
    if with_spec:
        spec = SpeculativeConfig(method="mtp", num_speculative_tokens=k)
    config = VllmConfig(
        model_config=ModelConfig(model=directory, dtype="float32",
                                 max_model_len=max_model_len, hf_config=hf_config),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=num_gpu_blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=DEVICE), speculative_config=spec)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner


def run(engine, core, prompts, *, max_tokens=6):
    for req_id, prompt in prompts:
        engine.add_request(req_id, list(prompt),
                           SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                          eos_token_id=999))
    outputs, stats = {}, []
    for _ in range(400):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            outputs[out.request_id] = list(out.token_ids)
        if core.scheduler.spec_decoding_stats is not None:
            stats.append(core.scheduler.spec_decoding_stats)
    return outputs, stats


def collect_drafts(runner):
    """把每轮提给调度器的草稿记下来（`take_draft_token_ids()` 是协议出口）。"""
    drafts_log = []
    original = runner.take_draft_token_ids

    def spy():
        drafts = original()
        if drafts is not None:
            drafts_log.append([list(tokens) for tokens in drafts.draft_token_ids])
        return drafts

    runner.take_draft_token_ids = spy
    return drafts_log


def install_forward_spy(runner):
    """记录每次 draft 前向的输入特征（第一遍与自回归步都在这里过）。"""
    proposer = runner.proposer
    calls = []
    original = proposer._forward

    def spy(num_tokens, num_reqs, hidden_states=None):
        features = proposer.hidden_states_cpu[:num_tokens].clone()
        result = original(num_tokens, num_reqs, hidden_states)
        calls.append({"num_tokens": num_tokens, "features": features})
        return result

    proposer._forward = spy
    return calls


def install_logits_spy(runner):
    """记录每轮草稿采样时**真正用的 logits**（比 argmax 更能反映"有没有吃到输入"）。"""
    proposer = runner.proposer
    logits_log = []
    original = proposer._sample_draft_tokens

    def spy(hidden, row_refs, input_batch, drafts, probs):
        rows = torch.tensor([row for _, row in row_refs], dtype=torch.int64,
                            device=hidden.device)
        with torch.inference_mode():
            logits_log.append(proposer.model.compute_logits(hidden[rows]).float().clone())
        return original(hidden, row_refs, input_batch, drafts, probs)

    proposer._sample_draft_tokens = spy
    return logits_log


# ---------------------------------------------------------------- 1/2. 端到端与草稿


@pytest.mark.parametrize("naming", NAMINGS)
def test_greedy_matches_non_speculative(naming):
    """greedy：开 MTP 与不开投机逐 token 相同，且草稿真的进过调度器。"""
    engine, core, runner = make_engine(naming=naming, k=1)
    try:
        assert isinstance(runner.proposer, EagleProposer), \
            "MTP 必须复用通用迭代提议（上游 use_eagle() 把 mtp 算进来）"
        assert runner.proposer.model_returns_tuple() is False, \
            "Qwen3 家族的 MTP 只返回一个 hidden（上游 Qwen3NextMTP 同款）"
        spec_outputs, stats = run(engine, core, PROMPTS)
    finally:
        engine.shutdown()
    base_engine, base_core, _ = make_engine(naming=naming, with_spec=False)
    try:
        base_outputs, _ = run(base_engine, base_core, PROMPTS)
    finally:
        base_engine.shutdown()
    assert spec_outputs == base_outputs, f"{spec_outputs} != {base_outputs}"
    assert stats and sum(s.num_draft_tokens for s in stats) > 0, "草稿必须真的被调度过"


@pytest.mark.parametrize("k", [1, 2])
def test_greedy_matches_for_several_k(k):
    """K=1 与 K>1 都要与非投机一致（K>1 会走自回归补足那一段）。"""
    engine, core, _ = make_engine(k=k)
    try:
        spec_outputs, stats = run(engine, core, PROMPTS)
    finally:
        engine.shutdown()
    base_engine, base_core, _ = make_engine(with_spec=False)
    try:
        base_outputs, _ = run(base_engine, base_core, PROMPTS)
    finally:
        base_engine.shutdown()
    assert spec_outputs == base_outputs
    assert any(s.num_draft_tokens >= k for s in stats), \
        f"K={k} 时每轮至少该提 {k} 枚（实际 {[s.num_draft_tokens for s in stats]}）"


def test_draft_tokens_are_in_vocab_and_variable_length():
    """草稿是合法词表 id，长度不超过 K（被拒/预算不够时会少提）。"""
    engine, core, runner = make_engine(k=2)
    drafts_log = collect_drafts(runner)
    try:
        run(engine, core, PROMPTS, max_tokens=6)
    finally:
        engine.shutdown()
    hf_config = json.loads((Path(tiny_mtp_dir("tiny_gqa")) / "config.json").read_text())
    vocab = hf_config["vocab_size"]
    assert drafts_log
    for round_drafts in drafts_log:
        for tokens in round_drafts:
            assert len(tokens) <= 2
            assert all(0 <= token < vocab for token in tokens), tokens


# ---------------------------------------------------------------- 3. hidden 回灌


def test_ar_steps_feed_back_the_previous_hidden():
    """自回归步的输入特征**必须**来自上一步的输出（不是第一遍那份 frozen 特征）。"""
    engine, core, runner = make_engine(k=3, max_num_seqs=1)
    calls = install_forward_spy(runner)
    try:
        run(engine, core, [PROMPTS[0]], max_tokens=4)
    finally:
        engine.shutdown()
    assert len(calls) >= 2, f"K=3 时一轮里应该有第一遍 + 自回归步，实际 {len(calls)} 次前向"
    first, second = calls[0], calls[1]
    assert first["num_tokens"] != second["num_tokens"] or \
        not torch.equal(first["features"], second["features"]), \
        "自回归步用的是第一遍那份特征（没有回灌上一枚的 hidden）——静默错，草稿质量会掉"


def test_freezing_the_hidden_feedback_changes_the_drafts():
    """反证：把回灌"冻"在第一遍那份特征上，草稿 logits 必须变（否则这条链路根本没接上）。

    只比 argmax 不够：tiny 权重的 argmax 可能很稳（实测错位一整行都未必翻 argmax），
    所以这里比**采样用的 logits**——"完全没吃到输入"会让 logits 逐位相同。
    """
    first_pass_features = {}

    def run_with(freeze: bool):
        engine, core, runner = make_engine(k=3, max_num_seqs=1)
        logits_log = install_logits_spy(runner)
        proposer = runner.proposer
        original_forward = proposer._forward
        original_ar = proposer._set_autoregressive_inputs

        def forward_spy(num_tokens, num_reqs, hidden_states=None):
            if not first_pass_features and num_tokens > 0:
                first_pass_features["rows"] = proposer.hidden_states_cpu[:num_tokens].clone()
            return original_forward(num_tokens, num_reqs, hidden_states)

        proposer._forward = forward_spy

        def ar_spy(pending, drafts, input_batch):
            original_ar(pending, drafts, input_batch)
            if freeze:
                # 把工作区里的特征换回第一遍那份（= 忘记回灌上一枚的 hidden）
                frozen = first_pass_features["rows"][:len(pending)]
                proposer.hidden_states_cpu[:len(pending)] = frozen

        proposer._set_autoregressive_inputs = ar_spy
        try:
            run(engine, core, [PROMPTS[0]], max_tokens=6)
        finally:
            engine.shutdown()
        return logits_log

    correct = run_with(freeze=False)
    frozen = run_with(freeze=True)
    assert correct and frozen and len(correct) == len(frozen)
    deltas = [float((a - b).abs().max()) for a, b in zip(correct, frozen)]
    assert max(deltas) > 1e-3, f"冻结回灌后 logits 几乎没变：{deltas}"


def test_misaligned_target_hidden_changes_the_drafts():
    """反证：把喂给 MTP 的 target hidden **错位一行**，草稿 logits 必须变（抓"假接线"）。

    同样比 logits 而不是 argmax：错位一整行在 tiny 权重下未必翻 argmax，但一定会改变分布。
    另外顺手验证补丁真的生效（提议者收到的特征与正确那份不同）。
    """
    seen_features = {}

    def run_with(shift: bool):
        engine, core, runner = make_engine(k=2, max_num_seqs=1)
        logits_log = install_logits_spy(runner)
        forward_calls = install_forward_spy(runner)
        if shift:
            original = runner._target_hidden_states_by_req

            def spy(scheduler_output, num_reqs, **kwargs):   # 70 关：多传 non_block
                by_req = original(scheduler_output, num_reqs)
                if by_req is None:
                    return None
                # 每个请求内部的 hidden 整体错开一行（第 0 行补 0）
                shifted = {}
                for req_id, rows in by_req.items():
                    moved = torch.zeros_like(rows)
                    if rows.shape[0] > 1:
                        moved[1:] = rows[:-1]
                    shifted[req_id] = moved
                return shifted

            runner._target_hidden_states_by_req = spy
        try:
            run(engine, core, [PROMPTS[0]], max_tokens=6)
        finally:
            engine.shutdown()
        seen_features["shifted" if shift else "correct"] = forward_calls[0]["features"]
        return logits_log

    correct = run_with(shift=False)
    shifted = run_with(shift=True)
    assert correct and shifted and len(correct) == len(shifted)
    assert not torch.equal(seen_features["correct"], seen_features["shifted"]),         "错位补丁没生效（提议者收到的特征没变）"
    deltas = [float((a - b).abs().max()) for a, b in zip(correct, shifted)]
    assert max(deltas) > 1e-3, f"错位 hidden 后 logits 几乎没变：{deltas}"


# ---------------------------------------------------------------- 4. KV 有效范围


def test_rejected_positions_are_recomputed_next_round():
    """被拒位置下一轮必被重算（63 关的不变量，MTP 的 KV 同样依赖它）。"""
    engine, core, runner = make_engine(k=2, max_num_seqs=1)
    rounds = []
    original = runner.execute_model

    def spy(scheduler_output, **kwargs):      # 70 关：executor 会多传 non_block
        if scheduler_output.total_num_scheduled_tokens > 0:
            starts = {data.req_id: data.num_computed_tokens
                      for data in scheduler_output.scheduled_new_reqs}
            for index, req_id in enumerate(scheduler_output.scheduled_cached_reqs.req_ids):
                starts[req_id] = \
                    scheduler_output.scheduled_cached_reqs.num_computed_tokens[index]
            rounds.append({"start": starts.get("a"),
                           "scheduled": dict(scheduler_output.num_scheduled_tokens),
                           "spec": {k: list(v) for k, v in
                                    scheduler_output.scheduled_spec_decode_tokens.items()}})
        return original(scheduler_output)

    runner.execute_model = spy
    try:
        run(engine, core, [PROMPTS[0]], max_tokens=6)
    finally:
        engine.shutdown()
    # 至少有一轮带了草稿，且下一轮的起点 = 本轮起点 + 有效行数（= 采用数 - 被拒数 + 1）
    with_drafts = [r for r in rounds if r["spec"]]
    assert with_drafts, "必须出现过带草稿的轮次"
    checked = 0
    for entry, next_entry in zip(rounds, rounds[1:]):
        if not entry["spec"]:
            continue
        adopted = len(entry["spec"].get("a", []))
        scheduled = entry["scheduled"].get("a", 0)
        # 下一轮的起点落在 [start + scheduled - adopted, start + scheduled]
        assert next_entry["start"] is not None
        assert entry["start"] + scheduled - adopted <= next_entry["start"] \
            <= entry["start"] + scheduled
        checked += 1
    assert checked > 0


def test_preemption_and_resume_keep_greedy_output():
    """块很少 → 触发抢占/恢复：draft 的进度重置后 greedy 仍与非投机一致。"""
    engine, core, runner = make_engine(k=2, max_num_seqs=2, num_gpu_blocks=6)
    try:
        spec_outputs, _ = run(engine, core, PROMPTS, max_tokens=6)
        preempted = core.scheduler.num_preemptions if hasattr(
            core.scheduler, "num_preemptions") else None
    finally:
        engine.shutdown()
    base_engine, base_core, _ = make_engine(with_spec=False, num_gpu_blocks=6)
    try:
        base_outputs, _ = run(base_engine, base_core, PROMPTS, max_tokens=6)
    finally:
        base_engine.shutdown()
    assert spec_outputs == base_outputs, (
        f"抢占/恢复后输出不一致：{spec_outputs} != {base_outputs}（preemptions={preempted}）")


def test_draft_kv_never_writes_beyond_max_model_len():
    """上下文快满时少提几枚（draft 的位置必须留在 max_model_len 内）。"""
    engine, core, runner = make_engine(k=3, max_num_seqs=1, max_model_len=12, budget=64)
    errors = []
    original = runner.proposer._set_autoregressive_inputs

    def spy(pending, drafts, input_batch):
        for target, position in pending:
            if position >= runner.proposer.max_model_len:
                errors.append((target.req_id, position))
        return original(pending, drafts, input_batch)

    runner.proposer._set_autoregressive_inputs = spy
    try:
        run(engine, core, [("a", [1, 2, 3, 4, 1, 2, 3, 4])], max_tokens=8)
    finally:
        engine.shutdown()
    assert not errors, f"自回归步写了越界位置：{errors}"
