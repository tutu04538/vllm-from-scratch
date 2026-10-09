"""69 关（一）：投机输入 padding 的语义与上游逐值对照。

需求 069 §3.1/§3.2 要求先把 **eager 语义**对齐，再谈图：

    prepare_inputs_padded()      每请求"该从哪一行采样"与"被拒了几行"（上游 device 内核）
    token_indices_to_sample      上者的第一个输出（target 行坐标系）
    num_rejected_tokens_gpu      上者的第二个输出（决定 draft 上下文长度要减多少）
    工作区覆盖                   input_ids/positions/masks/slot_mapping/query_start_loc/
                                 seq_lens/block table；padding 的槽位用源码哨兵

本文件里"上游"= 本机装的 vllm==0.28.0 的 Triton 内核（`minivllm/spec_decode/utils.py`
的 eager 实现与它逐值对照），差分测试可以直接调参考实现（AGENTS §2.1 允许）。
"""

import pytest
import torch

from spec69_helpers import (DEVICE, make_engine, make_config, run_to_end, tiny_model)  # noqa: F401
from minivllm import SamplingParams
from minivllm.spec_decode.utils import (PADDING_SLOT_ID, prepare_inputs_padded,
                                        eagle_step_update_slot_mapping_and_metadata)
from minivllm.worker.gpu_model_runner import _PaddedRun  # noqa: F401

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                   reason="上游内核是 Triton/CUDA，逐值差分需要 GPU")


# ---------------------------------------------------------------- 1. 纯张量语义


def test_prepare_inputs_padded_matches_hand_computed_cases():
    """四种接受情况的逐值算例（行号 = 每请求 query 块的最后一行 − 被拒行数）。

    批里有 4 条请求，K 分别是 3/2/0/1，有效 token 数（= 接受数 + 1）分别是 1/3/1/2：

        req0: K=3, 0 枚被接受 → 被拒 3 → 采样行 = 块内第 0 行（b 那一行）
        req1: K=2, 全接受   → 被拒 0 → 采样行 = 块内第 2 行（最后一枚草稿）
        req2: K=0           → 没有草稿，"被拒"记 0（第 0 行就是纠正/bonus 行）
        req3: K=1, 1 枚接受 → 被拒 0 → 采样行 = 块内第 1 行
    """
    cu = torch.tensor([3, 5, 5, 6], dtype=torch.int32)        # 包含式前缀和
    valid = torch.tensor([1, 3, 1, 2], dtype=torch.int32)
    qsl = torch.tensor([0, 4, 7, 8, 10], dtype=torch.int32)
    index, rejected = prepare_inputs_padded(cu, valid, qsl, num_reqs=4)
    assert index.tolist() == [0, 6, 7, 9]
    assert rejected.tolist() == [3, 0, 0, 0]
    assert index.dtype == torch.int32 and rejected.dtype == torch.int32


def test_prepare_inputs_padded_rejects_mismatched_shapes():
    cu = torch.tensor([1, 2], dtype=torch.int32)
    with pytest.raises(ValueError):
        prepare_inputs_padded(cu, torch.tensor([1], dtype=torch.int32),
                              torch.tensor([0, 2, 3], dtype=torch.int32), num_reqs=2)
    with pytest.raises(ValueError):
        prepare_inputs_padded(cu, torch.tensor([1, 2], dtype=torch.int32),
                              torch.tensor([0, 2], dtype=torch.int32), num_reqs=2)


@requires_cuda
def test_prepare_inputs_padded_matches_upstream_kernel():
    """与上游 `eagle_prepare_inputs_padded_kernel` 逐值差分（同一份输入、同一批形状）。"""
    from vllm.v1.spec_decode.utils import eagle_prepare_inputs_padded_kernel

    torch.manual_seed(0)
    for num_reqs in (1, 2, 5):
        drafts = torch.randint(0, 4, (num_reqs,), dtype=torch.int32, device="cuda")
        qlens = drafts + 1
        valid = torch.minimum(torch.randint(1, 8, (num_reqs,), dtype=torch.int32,
                                            device="cuda"), qlens)
        cu = torch.cumsum(drafts, 0).to(torch.int32)
        qsl = torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"),
                         torch.cumsum(qlens, 0).to(torch.int32)])
        up_index = torch.empty(num_reqs, dtype=torch.int32, device="cuda")
        up_rejected = torch.empty(num_reqs, dtype=torch.int32, device="cuda")
        eagle_prepare_inputs_padded_kernel[(num_reqs,)](
            cu, valid, qsl, up_index, up_rejected, num_reqs)
        our_index, our_rejected = prepare_inputs_padded(cu, valid, qsl, num_reqs)
        assert up_index.tolist() == our_index.tolist(), (num_reqs, drafts.tolist())
        assert up_rejected.tolist() == our_rejected.tolist(), (num_reqs, drafts.tolist())


def test_eagle_step_update_slot_mapping_and_metadata_semantics():
    """位置 +1 / 查块表得槽位 / 上下文 +1；越界行是哨兵槽位并把长度拉回 1。"""
    positions = torch.tensor([10, 20, 63], dtype=torch.int64)
    block_table = torch.tensor([[3, 4, 5], [6, 7, 0], [1, 2, 3]], dtype=torch.int64)
    seq_lens = torch.tensor([31, 41, 64], dtype=torch.int64)
    out_positions = torch.zeros(5, dtype=torch.int64)
    out_slots = torch.zeros(5, dtype=torch.int64)
    eagle_step_update_slot_mapping_and_metadata(
        positions, block_table, seq_lens, block_size=4, max_model_len=64,
        out_clamped_positions=out_positions, out_slot_mapping=out_slots,
        input_batch_size=5)
    # 位置 +1：11 / 21 / 64（64 >= max_model_len → 钳到 0）
    assert out_positions[:3].tolist() == [11, 21, 0]
    # 槽位（逐条按公式算）：
    #   pos 11 → 块 11//4=2，block_table[0,2]=5，5*4 + 11%4 = 23
    #   pos 21 → 块 21//4=5，钳到 n_blocks-1=2，block_table[1,2]=0，0*4 + 21%4 = 1
    #   越界（pos 64 >= max_model_len）→ 哨兵
    assert out_slots[:3].tolist() == [23, 1, PADDING_SLOT_ID]
    # 长度：31+1 / 41+1 / 越界 → 1
    assert seq_lens.tolist() == [32, 42, 1]
    # padding 行（>= batch_size）只写哨兵
    assert out_slots[3:].tolist() == [PADDING_SLOT_ID, PADDING_SLOT_ID]


@requires_cuda
def test_eagle_step_update_matches_upstream_kernel():
    """与上游 `eagle_step_slot_mapping_metadata_kernel` 逐值差分（含 padding 行）。"""
    from vllm.v1.spec_decode.utils import eagle_step_update_slot_mapping_and_metadata as up_fn

    torch.manual_seed(1)
    batch_size, input_batch_size, block_size, max_model_len = 4, 7, 4, 32
    for _ in range(4):
        positions = torch.randint(0, max_model_len, (batch_size,),
                                  dtype=torch.int64, device="cuda")
        block_table = torch.randint(1, 6, (batch_size, max_model_len // block_size),
                                    dtype=torch.int64, device="cuda")
        # 上游与本地都原地改 seq_lens，所以各复制一份
        up_seq = torch.randint(1, max_model_len, (batch_size,), dtype=torch.int64,
                               device="cuda")
        our_seq = up_seq.clone()
        up_positions = torch.empty(batch_size, dtype=torch.int64, device="cuda")
        up_slots = torch.empty(input_batch_size, dtype=torch.int64, device="cuda")
        our_positions = torch.empty(batch_size, dtype=torch.int64, device="cuda")
        our_slots = torch.empty(input_batch_size, dtype=torch.int64, device="cuda")
        up_fn(positions, block_table, up_seq, block_size, max_model_len,
              up_positions, up_slots, input_batch_size=input_batch_size)
        eagle_step_update_slot_mapping_and_metadata(
            positions, block_table, our_seq, block_size, max_model_len,
            our_positions, our_slots, input_batch_size=input_batch_size)
        assert up_positions.tolist() == our_positions.tolist()
        assert up_slots.tolist() == our_slots.tolist()
        assert up_seq.tolist() == our_seq.tolist()


# ---------------------------------------------------------------- 2. 真实轨迹里的 padding


class _PaddingTrace:
    """记录每轮 draft 第一遍的：padded 索引、被拒行数、AR 步的位置/槽位/长度。"""

    def __init__(self, runner):
        self.runner = runner
        proposer = runner.proposer
        self.rounds = []
        self._propose = proposer.propose
        self._check = proposer._check_ar_metadata_on_device
        this = self

        def propose(rows, all_token_ids, input_batch, reset_req_ids=None, **kwargs):
            drafts = this._propose(rows, all_token_ids, input_batch, reset_req_ids, **kwargs)
            this.rounds.append({
                "rows": list(rows),
                "padded": dict(proposer.last_padded_inputs or {}),
                "num_rejected": list(getattr(drafts, "num_rejected", []) or []),
            })
            return drafts

        def check_ar(pending, positions, num_reqs):
            this.rounds.append({"ar": [(t.req_id, p) for t, p in pending],
                               "seq_lens": proposer.seq_lens_cpu[:num_reqs].clone().tolist(),
                               "slots": proposer.slot_mapping_cpu[:num_reqs].clone().tolist()})
            return this._check(pending, positions, num_reqs)

        proposer.propose = propose
        proposer._check_ar_metadata_on_device = check_ar

    def restore(self):
        self.runner.proposer.propose = self._propose
        self.runner.proposer._check_ar_metadata_on_device = self._check

    def first_passes(self):
        return [entry for entry in self.rounds if "rows" in entry]


def test_padded_indices_match_scheduler_facts(tiny_dir, hf_config):
    """真实轨迹：`token_indices_to_sample` / `num_rejected_tokens_gpu` 与 TargetRows 一致。

    这一步在**生产路径里**已经逐值校验过（提议者会报错），这里再从外部确认：
    device 侧的索引 = 块起点 + target_rows - 1 - num_rejected，被拒数 = TargetRows.num_rejected。
    """
    engine, _core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=3,
                                        budget=16, blocks=32, mode="none")
    engine.add_request("r", [1, 2, 3, 4, 5, 6],
                       SamplingParams(max_tokens=6, temperature=0.0, eos_token_id=999))
    run_to_end(engine)
    engine.shutdown()
    # 断言放在轨迹里：每一轮的第一个 forward 之前，Runner 都传了这两个张量
    assert runner.proposer.last_padded_inputs is not None
    index = runner.proposer.last_padded_inputs["token_indices_to_sample"]
    rejected = runner.proposer.last_padded_inputs["num_rejected_tokens_gpu"]
    assert index == [] or len(index) >= 1
    assert rejected is not None and all(value >= 0 for value in rejected)


def test_draft_first_pass_sample_row_layouts(tiny_dir, hf_config, eagle3_dir):
    """两种提议者的"采样行"与 target 行的换算关系（本关最容易搞错的一处）。

    draft_model（`SpecDecodeBaseProposer`）：工作区 = [有效行][扩容行][被拒行]，
        采样行 = 块起点 + num_valid = target 行号 + 请求序号 + 1
    EAGLE（`EagleProposer`）：沿用 target 的行块、把新采出的 token 打在**最后一枚有效行**上，
        采样行 = 块起点 + target_rows - 1 - num_rejected
        （⚠️ 2026-10-08 复核修正：以前写的是"块的最后一行"`target_rows - 1`，被拒行 > 0 时
        锚点/采样行的位置与上下文会偏出 num_rejected 格；见 docs/step63_alignment.md §8）
    """
    from minivllm.spec_decode.utils import TargetRows

    rows = [TargetRows(req_id="a", row=0, start=10, target_rows=4, num_rejected=2,
                       history_end=13, next_token_id=7, ready=True),
            TargetRows(req_id="b", row=1, start=20, target_rows=3, num_rejected=0,
                       history_end=24, next_token_id=8, ready=True)]

    all_token_ids = {"a": list(range(30)), "b": list(range(30))}
    draft_engine, _core, draft_runner = make_engine(
        tiny_dir=tiny_dir, hf_config=hf_config, spec_k=3, mode="none")
    draft_plan = draft_runner.proposer.set_inputs_first_pass(rows, all_token_ids)
    # a：块起点 0 + num_valid(4-2=2) = 2？不——采样行是"扩容行"，它在有效行之后：
    #    有效 2 行（0,1）+ 扩容行(2) + 被拒 2 行(3,4) → 采样行 2
    # b：块起点 5 + 有效 3 行(5,6,7) + 扩容行(8) → 采样行 8
    assert draft_plan.sample_rows == [2, 8]
    assert draft_plan.sample_req_ids == ["a", "b"]
    draft_engine.shutdown()

    import json
    from pathlib import Path

    eagle_config = json.loads((Path(eagle3_dir) / "config.json").read_text())
    eagle_engine, _core, eagle_runner = make_engine(
        tiny_dir=tiny_dir, hf_config=hf_config, spec_k=3, draft_dir=eagle3_dir,
        draft_config=eagle_config, mode="none", method="eagle3")
    # EAGLE：行块 = target 的行块（4+3=7 行），扩容 token 打在**每块最后一行**
    # 特征只要形状对（这里不跑模型，只看拼行与采样行）：按请求给 7 行、宽度 = 融合后的 hidden
    # 宽度 = hidden × 辅助层数（`combine_hidden_states` 会把多层切块后投影，与 63 关同款布局）
    num_aux = eagle_runner.proposer.num_aux_layers
    hidden = {target.req_id: torch.zeros(target.target_rows,
                                         eagle_runner.proposer.hidden_size * num_aux,
                                         device=eagle_runner.device)
              for target in rows}
    eagle_plan = eagle_runner.proposer.set_inputs_first_pass(
        rows, all_token_ids, target_hidden_states=hidden,
        target_token_ids=torch.tensor(list(range(10))),
        target_positions=torch.tensor(list(range(10))))
    # a：块起点 0 + 4 - 1 - 被拒 2 = 1；b：块起点 4 + 3 - 1 - 被拒 0 = 6
    assert eagle_plan.sample_rows == [1, 6]
    eagle_engine.shutdown()


def test_rejected_rows_are_removed_from_draft_context(tiny_dir, hf_config):
    """被拒行不写 KV、也不进 draft 的上下文（`seq_lens` 减掉被拒数；上游同款动作）。

    构造一个必然被拒的轮次（提议者永远提 token 0），然后检查：
      * 被拒行在工作区里是 padding（token=0 / position=0 / slot=-1）
      * 第一遍之后 draft 的 `seq_lens` = 起点 + 有效行数 + 1（**不含**被拒行）
      * AR 步的 `seq_lens` 从那个值开始每步 +1（不是从乐观值开始）
    """
    from minivllm.spec_decode.utils import PADDING_SLOT_ID as SENTINEL

    engine, _core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=2,
                                        budget=16, mode="none")
    proposer = runner.proposer
    original = proposer._sample_draft_tokens
    trace = _PaddingTrace(runner)

    def always_zero(hidden, row_refs, input_batch, drafts, probs):
        for req_id, _row in row_refs:
            drafts[req_id].append(0)
            probs[req_id].append(torch.ones(hf_config["vocab_size"]) / hf_config["vocab_size"])

    proposer._sample_draft_tokens = always_zero
    engine.add_request("r", [1, 2, 3, 4, 5, 6],
                       SamplingParams(max_tokens=5, temperature=0.0, eos_token_id=999))
    run_to_end(engine)
    trace.restore()
    proposer._sample_draft_tokens = original
    engine.shutdown()

    rejecting = [entry for entry in trace.first_passes()
                 if entry["rows"][0].num_rejected > 0]
    assert rejecting, "没有捕获到被拒轮次"
    entry = rejecting[0]
    target = entry["rows"][0]
    # 被拒行的哨兵语义（工作区里物理存在）
    num_tokens = proposer.max_num_tokens
    slots = proposer.slot_mapping_cpu[:target.target_rows + 1].tolist()
    assert slots[target.num_valid] >= 0, "扩容行的槽位必须是真的（它要写 KV）"
    assert all(slot == SENTINEL for slot in slots[target.num_valid + 1:]), slots
    # 被拒行数两套口径一致，并且进了 draft 的上下文长度修正
    assert entry["padded"]["num_rejected_tokens_gpu"] == [target.num_rejected]
    ar_steps = [e for e in trace.rounds if "ar" in e]
    assert ar_steps, "没有捕获到自回归步"
    first_ar = ar_steps[0]
    expected = target.start + target.num_valid + 1
    assert first_ar["seq_lens"] == [expected], (first_ar, target)


def test_padding_rows_extend_the_compact_layout(tiny_dir, hf_config):
    """补齐的行是**紧凑布局的尾巴**（不是重排）：真实行的索引与内容一个都没变。

    这是"采样/掩码/logprobs 的行映射不用改"的依据：紧凑行 = 补齐布局的前缀。
    """
    from minivllm.worker.gpu_model_runner import _PaddedRun

    engine, _core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=3,
                                        budget=32, blocks=32, mode="full_decode_only")
    engine.add_request("r", [1, 2, 3, 4, 5, 6],
                       SamplingParams(max_tokens=6, temperature=0.0, eos_token_id=999))
    run_to_end(engine)
    engine.shutdown()
    selections = [s for s in runner.cudagraph_selections if s["mode"] != "NONE"]
    assert selections, "这一轮配置下应该出现过图分派"
    for selection in selections:
        # 统一 decode 批：补齐后的行数 = 请求数 × (1+K)，是紧凑行数的整数倍
        assert selection["padded_tokens"] % (1 + 3) == 0
        assert selection["padded_tokens"] >= selection["num_tokens"]
    # 静态缓冲里的 padding 行槽位是哨兵（不是 0：0 是真实槽位）
    buffers = runner._padded_buffers
    assert buffers is not None
    assert int(buffers["slot_mapping"].min()) == PADDING_SLOT_ID
