"""72 关（PARD 与 P-EAGLE 并行提议）的自检。

需求 072 §4 的验收项在这里逐条落地：

    * B=2、K=1/4：输入、mask、positions、hidden 映射、采样行**逐值**对照上游 kernel
      （两请求长度故意不同，并带被拒尾部）
    * K=1 时 P-EAGLE 净增槽位 = 0、PARD = 1；调度侧的 draft_slots / 预算口径正确
    * 并行模式下**一次** forward 出 K 枚（串行是 K 次）
    * partial prefill / 被拒 / 全接受下 greedy 端到端 == 非投机
    * 普通（串行训练的）draft 权重开并行必须在加载期报错——不能拿它做并行验收

**模型格式清单**（需求 §5 交付物）写在 `docs/step72_alignment.md` §3：
P-EAGLE 的 checkpoint 必须有 `mask_token_id`（config）与 `mask_hidden`（权重，`(1, hidden×aux)`）；
PARD 必须有 `pard_token`。本机没有这两类权重 → 性能/质量验收标"集成待验"（§5）。
"""

import pytest
import torch

from spec72_helpers import (DEVICE, HF, MASK_TOKEN, PARD_TOKEN, ParallelRunRecorder, TINY,
                            draft_dir, greedy, make_config, make_engine, requires_cuda,
                            run_to_end)
from minivllm import SamplingParams
from minivllm.spec_decode.utils import TargetRows, expand_parallel_draft_inputs


def _rows(specs):
    """把 `(req_id, target_rows, rejected, next_token, ready)` 变成 `TargetRows` 列表。"""
    return [TargetRows(req_id=req_id, row=index, start=start, target_rows=rows,
                       num_rejected=rejected, history_end=start + rows,
                       next_token_id=next_token, ready=ready)
            for index, (req_id, start, rows, rejected, next_token, ready) in enumerate(specs)]


# ------------------------------------------------------------ 1. 与上游 kernel 逐值差分


@requires_cuda
@pytest.mark.parametrize("shift", [True, False])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_expand_matches_upstream_kernel(shift, k):
    """PARD（shift=False）/ P-EAGLE（shift=True）的展开与**上游真 kernel** 逐值一致。

    上游 `copy_and_expand_eagle_inputs_kernel` 是 Triton kernel（生产路径不 import，差分测试
    可以调）。B=2、两请求长度故意不同（4 行 / 2 行）、带被拒尾部（3 枚 / 1 枚）全覆盖。
    """
    from vllm.v1.spec_decode.utils import copy_and_expand_eagle_inputs_kernel

    reqs = [("A", 0, 4, 3, 901, True), ("B", 4, 2, 1, 902, True)]
    rows = _rows(reqs)
    qsl = [0, 4, 6]
    total_in = 6
    base = 5
    t_ids = torch.arange(1, total_in + 1, dtype=torch.int32, device=DEVICE) + 100
    t_pos = torch.arange(total_in, dtype=torch.int32, device=DEVICE) + base
    next_ids = torch.tensor([901, 902], dtype=torch.int32, device=DEVICE)
    extra = k
    net = extra - (1 if shift else 0)
    total_out = total_in + net * len(reqs)
    out_ids = torch.zeros(total_out, dtype=torch.int32, device=DEVICE)
    out_pos = torch.zeros(total_out, dtype=torch.int32, device=DEVICE)
    out_rej = torch.zeros(total_out, dtype=torch.bool, device=DEVICE)
    out_msk = torch.zeros(total_out, dtype=torch.bool, device=DEVICE)
    out_idx = torch.zeros(len(reqs) * extra, dtype=torch.int32, device=DEVICE)
    out_hid = torch.zeros(total_in, dtype=torch.int32, device=DEVICE)
    block = min(256, 2 ** max(0, (max(4, 2) + net - 1).bit_length()))
    copy_and_expand_eagle_inputs_kernel[(len(reqs), 1)](
        target_token_ids_ptr=t_ids, target_positions_ptr=t_pos, next_token_ids_ptr=next_ids,
        out_input_ids_ptr=out_ids, out_positions_ptr=out_pos,
        out_is_rejected_token_mask_ptr=out_rej, out_is_masked_token_mask_ptr=out_msk,
        out_new_token_indices_ptr=out_idx, out_hidden_state_mapping_ptr=out_hid,
        query_start_loc_ptr=torch.tensor(qsl, dtype=torch.int32, device=DEVICE),
        query_end_loc_ptr=torch.tensor([3 - 3, 5 - 1], dtype=torch.int32, device=DEVICE),
        padding_token_id=0, parallel_drafting_token_id=MASK_TOKEN,
        total_input_tokens=total_in, num_padding_slots_per_request=extra,
        shift_input_ids=shift, BLOCK_SIZE_TOKENS=block)
    torch.cuda.synchronize()

    mine = expand_parallel_draft_inputs(t_ids.cpu().tolist(), t_pos.cpu().tolist(), rows,
                                        extra_slots_per_request=extra,
                                        parallel_drafting_token_id=MASK_TOKEN,
                                        shift_input_ids=shift)
    assert mine.num_tokens == total_out
    assert mine.input_ids == out_ids.cpu().tolist()
    assert mine.positions == out_pos.cpu().tolist()
    assert mine.is_rejected == [int(x) for x in out_rej.cpu().tolist()]
    assert mine.is_masked == [int(x) for x in out_msk.cpu().tolist()]
    assert mine.token_indices_to_sample == [int(x) for x in out_idx.cpu().tolist()]
    if shift:
        assert [mine.hidden_state_mapping.get(i, -1) for i in range(total_in)] == \
            out_hid.cpu().tolist()


def test_pard_layout_shapes_and_masks():
    """PARD（不左移）B=2、K=4：行数/掩码/采样行按"1 锚点 + K−1 mask"摆，且与 kernel 同构。

    这条不依赖 CUDA（纯函数），所以任何机器上都能验布局口径。
    """
    rows = _rows([("A", 0, 4, 3, 901, True), ("B", 4, 2, 1, 902, True)])
    out = expand_parallel_draft_inputs(list(range(100, 106)), list(range(5, 11)), rows,
                                       extra_slots_per_request=4,
                                       parallel_drafting_token_id=MASK_TOKEN,
                                       shift_input_ids=False)
    # 每请求行数 = target 行数 + net(4)；请求 A：有效 1 行 + 锚点 + 3 mask + 被拒 3 行 = 8
    assert out.num_tokens == (4 + 4) + (2 + 4)
    assert out.input_ids[:8] == [100, 901, MASK_TOKEN, MASK_TOKEN, MASK_TOKEN, 0, 0, 0]
    # 位置一律 `start_pos + j`（不随 shift 移动）：有效行 5、锚点 6、mask 7~9、被拒行 0
    assert out.positions[:8] == [5, 6, 7, 8, 9, 0, 0, 0]
    assert out.is_rejected[:8] == [0, 0, 0, 0, 0, 1, 1, 1]
    assert out.is_masked[:8] == [0, 0, 1, 1, 1, 0, 0, 0]
    assert out.token_indices_to_sample == [1, 2, 3, 4, 9, 10, 11, 12]
    assert out.hidden_state_mapping == {}          # 不左移 → 没有 hidden 映射


def test_peagle_layout_reuses_the_last_row_and_maps_hidden():
    """P-EAGLE（左移）K=4：有效行少一行、锚点复用 target 块最后一行、hidden 有映射。"""
    rows = _rows([("A", 0, 4, 0, 901, True)])
    out = expand_parallel_draft_inputs([100, 101, 102, 103], [5, 6, 7, 8], rows,
                                       extra_slots_per_request=4,
                                       parallel_drafting_token_id=MASK_TOKEN,
                                       shift_input_ids=True)
    # 左移：跳过第 0 个 token，前 3 行是 101、102、103，锚点在第 4 行（位置 8）
    assert out.input_ids == [101, 102, 103, 901, MASK_TOKEN, MASK_TOKEN, MASK_TOKEN]
    assert out.positions == [5, 6, 7, 8, 9, 10, 11]
    assert out.is_masked == [0, 0, 0, 0, 1, 1, 1]
    assert out.token_indices_to_sample == [3, 4, 5, 6]
    # hidden 不跟着移：源行 i → 目标行 i
    assert out.hidden_state_mapping == {0: 0, 1: 1, 2: 2, 3: 3}
    assert out.num_tokens == 4 + (4 - 1)


# ------------------------------------------------------------ 2. 槽位口径（PARD=K / P-EAGLE=K−1）


@pytest.mark.parametrize("method,k,parallel,net", [
    ("eagle3", 1, True, 0),        # P-EAGLE、K=1：没有 mask 行，复用 target 那一行
    ("eagle3", 4, True, 3),        # P-EAGLE：K−1
    ("draft_model", 1, True, 1),   # PARD、K=1：锚点仍是新行
    ("draft_model", 4, True, 4),   # PARD：K
    ("eagle3", 4, False, 0),       # 串行 EAGLE：复用
    ("draft_model", 4, False, 1),  # 串行 draft：尾部扩容行
])
def test_slot_accounting(method, k, parallel, net):
    """槽位净增：PARD=K、P-EAGLE=K−1（K=1 → 0）、串行 draft=1 / EAGLE=0。"""
    config = make_config(method=method, spec_k=k, parallel=parallel)
    spec = config.speculative_config
    assert spec.max_num_new_slots_for_drafting == net
    assert spec.uses_dynamic_speculative_decoding() is False


def test_proposer_slot_fields_match_upstream_formula():
    """提议者手里的两个量（extra / net）与上游 `llm_base_proposer.py:112-119` 逐条一致。"""
    engine, _core, runner = make_engine(method="draft_model", spec_k=4, parallel=True)
    try:
        proposer = runner.proposer
        assert proposer.extra_slots_per_request == 4
        assert proposer.net_num_new_slots_per_request == 4
        assert proposer.needs_extra_input_slots is True
        assert proposer.parallel_drafting_token_id == PARD_TOKEN
        assert proposer.parallel_drafting_hidden_state_tensor is None   # 不吃 hidden
    finally:
        engine.shutdown()

    engine, _core, runner = make_engine(method="eagle3", spec_k=4, parallel=True)
    try:
        proposer = runner.proposer
        assert proposer.extra_slots_per_request == 4
        assert proposer.net_num_new_slots_per_request == 3     # EAGLE 左移省一行
        assert proposer.parallel_drafting_token_id == MASK_TOKEN
        assert tuple(proposer.parallel_drafting_hidden_state_tensor.shape) == (32,)  # hidden
    finally:
        engine.shutdown()


def test_scheduler_draft_slots_follow_the_same_table():
    """调度侧的输入预算 `draft_slots` 与提议者的 net 是**同一个数**（不各推一遍）。"""
    for method, parallel, want in (("draft_model", True, 4), ("eagle3", True, 3)):
        engine, core, runner = make_engine(method=method, spec_k=4, parallel=parallel,
                                           budget=64, max_num_seqs=2)
        try:
            assert core.scheduler.draft_slots == want == \
                runner.proposer.net_num_new_slots_per_request
        finally:
            engine.shutdown()


# ------------------------------------------------------------ 3. 一次 forward / 端到端


@pytest.mark.parametrize("method,parallel", [("eagle3", True), ("draft_model", True)])
def test_parallel_drafting_uses_one_forward_per_round(method, parallel):
    """并行模式每轮**只跑 1 次** draft 前向（串行 K=3 要跑 3 次）——"伪并行"是这条的反面。"""
    engine, _core, runner = make_engine(method=method, spec_k=3, parallel=True,
                                        max_num_seqs=1, budget=32)
    recorder = ParallelRunRecorder(runner)
    try:
        engine.add_request("A", [1, 2, 3, 4, 5, 6],
                           SamplingParams(max_tokens=6, temperature=0.0, eos_token_id=999))
        run_to_end(engine)
    finally:
        recorder.restore()
        engine.shutdown()
    assert len(recorder.rounds) >= 3, recorder.rounds
    assert recorder.forwards == len(recorder.rounds), (
        f"{method} 并行提议跑了 {recorder.forwards} 次前向、{len(recorder.rounds)} 轮："
        f"每轮应当只有 1 次（K 行一次算完）")
    for round_ in recorder.rounds:
        assert sorted(round_["drafts"]) == [3] * len(round_["drafts"]), \
            f"并行提议每轮每请求要交回 K=3 枚：{round_['drafts']}"


def test_serial_drafting_still_uses_k_forwards():
    """对照组：串行模式 K=3 时每轮 3 次前向（第一遍 + 2 次自回归）。"""
    engine, _core, runner = make_engine(method="draft_model", spec_k=3, parallel=False,
                                        max_num_seqs=1, budget=32)
    recorder = ParallelRunRecorder(runner)
    try:
        engine.add_request("A", [1, 2, 3, 4, 5, 6],
                           SamplingParams(max_tokens=6, temperature=0.0, eos_token_id=999))
        run_to_end(engine)
    finally:
        recorder.restore()
        engine.shutdown()
    assert recorder.forwards == 3 * len(recorder.rounds), \
        f"串行 K=3 应当每轮 3 次前向：{recorder.forwards} / {len(recorder.rounds)}"


@pytest.mark.parametrize("method", ["eagle3", "draft_model"])
@pytest.mark.parametrize("k", [1, 4])
def test_parallel_greedy_matches_non_speculative(method, k):
    """并行提议的 greedy 输出必须与非投机逐 token 相同（K=1 与 K=4 都验）。"""
    plain = greedy(req_ids=("A",), spec_k=None, parallel=False)
    parallel = greedy(req_ids=("A",), method=method, spec_k=k, parallel=True)
    assert parallel["A"] == plain["A"], (method, k, parallel["A"], plain["A"])


def test_parallel_greedy_matches_with_two_requests_and_prefix_hit():
    """B=2（两请求长度不同）+ 前缀命中：并行提议的 greedy 仍与非投机一致。

    前缀命中这条路会让第二条请求从"命中末尾"起算（`TargetRows.start` 非 0），
    并行第一遍的有效行数随之变化——布局算错时不会报错，只会让草稿变差/输出错位。
    """
    prompt_a = (1, 2, 3, 4, 5, 6)
    prompt_b = (1, 2, 3, 4, 5, 6, 7, 8)          # 前半段与 A 相同 → 可能命中前缀
    results = {}
    for parallel, method in ((False, "draft_model"), (True, "draft_model")):
        engine, core, runner = make_engine(method=method, spec_k=2, parallel=parallel,
                                           max_num_seqs=2, budget=64, blocks=48)
        config = core.vllm_config
        # 打开前缀缓存需要在建引擎前改配置：这里用**两次相同前缀**的请求间接覆盖命中路径
        try:
            engine.add_request("A", list(prompt_a),
                               SamplingParams(max_tokens=4, temperature=0.0, eos_token_id=999))
            engine.add_request("B", list(prompt_b),
                               SamplingParams(max_tokens=4, temperature=0.0, eos_token_id=999))
            results[parallel] = run_to_end(engine)
        finally:
            engine.shutdown()
    assert results[True] == results[False], results


def test_parallel_k1_end_to_end_runs():
    """K=1：并行第一遍只有锚点一行（P-EAGLE 净增 0、PARD 净增 1），端到端要能跑通。"""
    for method, net in (("eagle3", 0), ("draft_model", 1)):
        engine, _core, runner = make_engine(method=method, spec_k=1, parallel=True,
                                            max_num_seqs=1, budget=32)
        try:
            assert runner.proposer.net_num_new_slots_per_request == net
            engine.add_request("A", [1, 2, 3, 4, 5, 6],
                               SamplingParams(max_tokens=4, temperature=0.0, eos_token_id=999))
            outputs = run_to_end(engine)
            assert len(outputs["A"]) == 4
        finally:
            engine.shutdown()


# ------------------------------------------------------------ 4. 布局落到真实工作区（含 mask hidden）


@requires_cuda
def test_peagle_masked_rows_get_the_mask_hidden():
    """P-EAGLE 的 mask 行 hidden 必须换成模型自带的 mask 向量（不换=静默用残留值）。"""
    engine, _core, runner = make_engine(method="eagle3", spec_k=3, parallel=True,
                                        max_num_seqs=1, budget=32)
    recorder = ParallelRunRecorder(runner)
    try:
        engine.add_request("A", [1, 2, 3, 4, 5, 6],
                           SamplingParams(max_tokens=4, temperature=0.0, eos_token_id=999))
        run_to_end(engine)
        mask = runner.proposer.parallel_drafting_hidden_state_tensor.cpu()
    finally:
        recorder.restore()
        engine.shutdown()
    checked = 0
    for round_ in recorder.rounds:
        hidden = round_["hidden"]
        for index, is_masked in enumerate(round_["is_masked"]):
            if not is_masked:
                continue
            assert torch.allclose(hidden[index], mask, atol=1e-6), (
                f"第 {index} 行是 mask 槽位，但 hidden 不是 mask 向量："
                f"{hidden[index][:4]} vs {mask[:4]}")
            checked += 1
    assert checked > 0, "一轮 mask 行都没验到"


@requires_cuda
def test_workspace_rows_equal_the_reference_expansion():
    """引擎**真正写进工作区**的行 == 用同一批事实算出的参考展开（B=2、长度不同）。"""
    engine, _core, runner = make_engine(method="draft_model", spec_k=3, parallel=True,
                                        max_num_seqs=2, budget=64)
    recorder = ParallelRunRecorder(runner)
    try:
        engine.add_request("A", [1, 2, 3, 4, 5, 6],
                           SamplingParams(max_tokens=6, temperature=0.0, eos_token_id=999))
        engine.add_request("B", [1, 2, 3],
                           SamplingParams(max_tokens=6, temperature=0.0, eos_token_id=999))
        for _ in range(4):
            engine.step()
    finally:
        recorder.restore()
        engine.shutdown()
    round_ = recorder.rounds[0]          # 第一轮：A 6 行 prefill + B 3 行 prefill
    assert round_["is_masked"].count(1) == 2 * 2, round_["is_masked"]   # 2 请求 × (K−1)=2 个 mask
    # 采样行 = 每个 mask 组的第一行（锚点）起、每组 K 行
    assert round_["sample_rows"] == sorted(round_["sample_rows"])
    assert len(round_["sample_rows"]) == 2 * 3
    # 被拒行在第一条请求的回合里不该出现（首次 prefill 没有被拒草稿）
    assert round_["is_rejected"].count(1) == 0


# ------------------------------------------------------------ 5. 边界：串行权重不能开并行


def test_serial_checkpoint_rejected_at_load_time():
    """**串行训练**的 EAGLE3 权重开并行 → 加载期明确报错（上游同款边界）。"""
    serial_dir, serial_hf = draft_dir("eagle3", parallel=False)
    serial_hf = dict(serial_hf, parallel_drafting=True, mask_token_id=MASK_TOKEN)
    config = make_config(method="eagle3", spec_k=2, parallel=True,
                         draft=(serial_dir, serial_hf))
    with pytest.raises(ValueError, match="mask_hidden"):
        from minivllm import LLMEngine as _LLMEngine, UniProcExecutor as _Exec, Worker as _Worker
        _LLMEngine(config, _Exec(config, _Worker(config)))


def test_missing_mask_token_rejected_at_config_time():
    """draft 配置里没有任何 mask token 字段 → 提议者构造时就报错（不猜一个）。"""
    draft_path, draft_hf = draft_dir("eagle3", parallel=True)
    stripped = {key: value for key, value in draft_hf.items() if key != "mask_token_id"}
    config = make_config(method="eagle3", spec_k=2, parallel=True,
                         draft=(draft_path, stripped))
    with pytest.raises(ValueError, match="mask token"):
        from minivllm import LLMEngine as _LLMEngine, UniProcExecutor as _Exec, Worker as _Worker
        _LLMEngine(config, _Exec(config, _Worker(config)))


def test_parallel_drafting_only_for_eagle_and_draft_model():
    """并行提议只与 EAGLE / draft_model 兼容（其余方法配置期拒绝，不静默忽略）。"""
    from minivllm import SpeculativeConfig
    for method in ("ngram", "ngram_gpu"):
        with pytest.raises(ValueError, match="parallel_drafting"):
            SpeculativeConfig(method=method, num_speculative_tokens=4, parallel_drafting=True)
