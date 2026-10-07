"""58：draft 第一遍输入的逐值检查（需求 §8.B）。

纯函数算例 + 真实跑出来的轨迹（prefill / 首拒绝 / 全接受 / prefix 命中 / chunked prefill）。
"""

import pytest
import torch

from helpers import WorkspaceTrace, greedy_outputs, make_engine, run_to_end
from minivllm import SamplingParams
from minivllm.spec_decode.utils import (PADDING_SLOT_ID, DraftInputRows,
                                        compute_new_slot_mapping, expand_draft_inputs,
                                        extend_all_queries_by_N)


# ---------------------------------------------------------------- 纯函数逐值
def test_expand_draft_inputs_layout():
    ids, positions, rejected, sample = expand_draft_inputs([
        DraftInputRows(valid_token_ids=[5, 6], start=6, next_token_id=99, num_rejected=2)])
    assert ids == [5, 6, 99, 0, 0]                  # 有效行 + 扩容行 + 被拒行(token=padding)
    assert positions == [6, 7, 8, 0, 0]             # 被拒行 position=0
    assert rejected == [0, 0, 0, 1, 1]
    assert sample == [2]                            # 扩容行的全局行号


def test_compute_new_slot_mapping_sentinels():
    block_table = torch.tensor([[0, 1, 2], [4, 5, 6]], dtype=torch.int64)
    slots = compute_new_slot_mapping(
        block_table, query_lens=[3, 3],
        new_positions=torch.tensor([6, 7, 8, 0, 8, 9, 10, 11]),
        is_rejected_token_mask=torch.tensor([0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.bool),
        block_size=4, num_new_tokens=1, max_model_len=10)
    assert PADDING_SLOT_ID == -1
    assert slots.tolist() == [6, 7, 8, -1, 24, 25, -1, -1]   # 被拒行与越界行都是哨兵


def test_extend_all_queries_by_N():
    qsl, seq = extend_all_queries_by_N([0, 4, 7], [10, 13], num_new_tokens=1)
    assert qsl == [0, 5, 9]
    assert seq == [11, 14]


# ---------------------------------------------------------------- 真实轨迹
def test_prefill_first_pass_layout(tiny_dir, hf_config):
    engine, _core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=3)
    trace = WorkspaceTrace(runner)
    engine.add_request("r", [1, 2, 3, 4, 5, 6], SamplingParams(max_tokens=4, temperature=0.0,
                                                               eos_token_id=999))
    run_to_end(engine)
    trace.restore()
    entry = trace.first_passes()[0]
    target = entry["rows"][0]
    assert entry["num_tokens"] == target.target_rows + 1        # target 行数 + 1
    assert entry["positions"] == list(range(target.target_rows + 1))
    assert entry["rejected"] == [0] * entry["num_tokens"]
    assert entry["input_ids"][:target.target_rows] == [1, 2, 3, 4, 5, 6]
    # 槽位 = 这条请求块表里**第一个物理块**的偏移展开。69 关起第一个块不再是 0 号块：
    # 开了 CUDA Graph 时 0 号块留白当垃圾桶（padding 行的 slot 会被 clamp 到它），
    # 所以这里断言的是"按块表算出来的槽位"，而不是写死的 range（断言强度不变）。
    first_block = int(runner.input_batch.block_table.cpu[0, 0])
    block_size = runner.block_size
    assert first_block != 0, "0 号块必须留白（CUDA Graph 的 padding 垃圾桶）"
    assert entry["slots"] == [first_block * block_size + offset
                              for offset in range(entry["num_tokens"])]
    engine.shutdown()


def test_rejected_tail_is_masked(tiny_dir, hf_config):
    """强制提一个必被拒的草稿：被拒尾部物理存在但 mask=1 / slot=-1 / position=0。"""
    engine, _core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=1)
    trace = WorkspaceTrace(runner)
    original = runner.proposer._sample_draft_tokens

    def always_zero(hidden, row_refs, input_batch, drafts, probs):
        for req_id, _ in row_refs:
            drafts[req_id].append(0)
            probs[req_id].append(torch.ones(hf_config["vocab_size"]) / hf_config["vocab_size"])

    runner.proposer._sample_draft_tokens = always_zero
    engine.add_request("r", [1, 2, 3, 4, 5, 6], SamplingParams(max_tokens=4, temperature=0.0,
                                                               eos_token_id=999))
    run_to_end(engine)
    trace.restore()
    runner.proposer._sample_draft_tokens = original
    rejecting = [e for e in trace.first_passes() if e["rows"][0].num_rejected]
    assert rejecting, "没有捕获到被拒轮次"
    entry = rejecting[0]
    rejected_rows = [i for i, rej in enumerate(entry["rejected"]) if rej]
    assert len(rejected_rows) == entry["rows"][0].num_rejected
    assert all(entry["slots"][i] == PADDING_SLOT_ID for i in rejected_rows)
    assert all(entry["positions"][i] == 0 for i in rejected_rows)
    assert all(entry["input_ids"][i] == 0 for i in rejected_rows)
    assert entry["num_tokens"] == entry["rows"][0].target_rows + 1
    engine.shutdown()


def test_full_accept_has_no_rejected_rows(tiny_dir, hf_config):
    """draft 与 target 同模型 + 纯贪心 → 全接受，物理行数仍是 target 行数 + 1。"""
    engine, _core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=3)
    trace = WorkspaceTrace(runner)
    engine.add_request("r", [1, 2, 3, 4, 5, 6], SamplingParams(max_tokens=6, temperature=0.0,
                                                               eos_token_id=999))
    run_to_end(engine)
    trace.restore()
    first = trace.first_passes()
    assert all(entry["rejected"] == [0] * entry["num_tokens"] for entry in first)
    assert all(entry["num_tokens"] == entry["rows"][0].target_rows + 1 for entry in first)
    engine.shutdown()


def test_prefix_hit_does_not_recompute_hit_range(tiny_dir, hf_config):
    """第二个相同 prompt 的请求命中前缀：draft 第一遍从命中末尾开始。"""
    engine, core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=3,
                                       prefix=True)
    trace = WorkspaceTrace(runner)
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 3, 4, 5]
    engine.add_request("first", prompt, SamplingParams(max_tokens=2, temperature=0.0,
                                                       eos_token_id=999))
    run_to_end(engine)
    engine.add_request("reuse", prompt, SamplingParams(max_tokens=2, temperature=0.0,
                                                       eos_token_id=999))
    run_to_end(engine)
    trace.restore()
    reuse = [e for e in trace.first_passes() if e["rows"][0].req_id == "reuse"][0]
    hit = reuse["rows"][0].start
    assert hit > 0, "第二个请求没有命中前缀"
    assert min(reuse["positions"]) == hit
    assert len(reuse["positions"]) < len(prompt)          # 没有从 0 重算整段
    assert reuse["num_tokens"] == reuse["rows"][0].target_rows + 1
    engine.shutdown()


def test_chunked_prefill_syncs_only_scheduled_range(tiny_dir, hf_config):
    engine, _core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=1,
                                        budget=4)
    trace = WorkspaceTrace(runner)
    engine.add_request("r", [1, 2, 3, 4, 5, 6, 7, 8], SamplingParams(max_tokens=4,
                                                                     temperature=0.0,
                                                                     eos_token_id=999))
    run_to_end(engine)
    trace.restore()
    chunks = [e for e in trace.first_passes() if not e["rows"][0].ready]
    assert chunks
    for entry in chunks:
        target = entry["rows"][0]
        assert entry["num_tokens"] == target.target_rows + 1
        assert entry["positions"][:target.target_rows] == list(
            range(target.start, target.start + target.target_rows))
    engine.shutdown()


def test_rows_invariant_under_preemption(tiny_dir, hf_config):
    """抢占恢复后仍然满足 rows ≤ target_rows + 1，且输出与非投机一致。"""
    # blocks=4 而不是 3：69 关起 0 号块留白当垃圾桶（CUDA Graph 的 padding 落点），
    # 可用块数 = blocks - 1，所以想让"可用块数"仍然是 3 就必须多给一块。
    reference = greedy_outputs(spec_k=None, tiny_dir=tiny_dir, hf_config=hf_config,
                               prompts=(("A", [1, 2]), ("B", [3, 4])), max_tokens=8,
                               budget=4, blocks=4, max_num_seqs=2)
    engine, core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=3,
                                       budget=4, blocks=4, max_num_seqs=2)
    trace = WorkspaceTrace(runner)
    for req_id, prompt in (("A", [1, 2]), ("B", [3, 4])):
        engine.add_request(req_id, prompt, SamplingParams(max_tokens=8, temperature=0.0,
                                                          eos_token_id=999))
    outputs = run_to_end(engine)
    trace.restore()
    assert core.scheduler.num_preemptions > 0
    assert outputs == reference
    assert all(e["num_tokens"] <= e["rows"][0].target_rows + 1 for e in trace.first_passes())
    engine.shutdown()
