"""72 关验收探针：PARD / P-EAGLE 并行提议（输入协议 + 槽位口径 + 一次 forward + 端到端）。

跑法：`python benchmarks/check_step72_parallel_draft.py`（退出码非 0 = 失败）。

分段（对应需求 072 §4）：

    A 槽位口径      PARD=K、P-EAGLE=K−1、K=1 时 P-EAGLE=0；串行对照；方法边界配置期拒绝
    B 输入协议      本地展开与**上游 Triton kernel** 逐值一致（两种 shift × K=1/2/4，B=2 长度不同）
    C 一次 forward   并行每轮 1 次（串行 K 次）；每轮每请求交回 K 枚
    D 端到端         greedy == 非投机（K=1/4、两种方法、两请求）；mask 行 hidden 落地
    E 模型格式       串行权重开并行 → 加载期报错；缺 mask token → 构造期报错

⚠️ 本机没有**按并行草稿训练**的 PARD/P-EAGLE 权重（见 `docs/step72_alignment.md` §3 的清单），
所以本探针验的是**协议与正确性**（tiny 权重 + 上游内核逐值差分），**不是**草稿质量/加速比
——那两条标"集成待验"，不在这里假装通过（需求 072 §4 最后一条）。
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step72"))

from spec72_helpers import (DEVICE, MASK_TOKEN, PARD_TOKEN, ParallelRunRecorder,  # noqa: E402
                            draft_dir, greedy, make_config, make_engine, run_to_end)
from minivllm import LLMEngine, SamplingParams, SpeculativeConfig, UniProcExecutor, Worker  # noqa: E402
from minivllm.spec_decode.utils import TargetRows, expand_parallel_draft_inputs  # noqa: E402

PASSED, FAILED, TRACE = 0, [], []


def check(name, ok, detail=""):
    global PASSED
    if ok:
        PASSED += 1
        print(f"PASS  {name}  {detail}")
    else:
        FAILED.append(name)
        print(f"FAIL  {name}  {detail}")
    TRACE.append({"item": name, "ok": bool(ok), "detail": str(detail)[:300]})


def rows_from(specs):
    return [TargetRows(req_id=req_id, row=index, start=start, target_rows=count,
                       num_rejected=rejected, history_end=start + count,
                       next_token_id=next_token, ready=True)
            for index, (req_id, start, count, rejected, next_token) in enumerate(specs)]


# ---------------------------------------------------------------- A 槽位口径

slot_ok = True
slot_detail = []
for method, k, parallel, want in (("eagle3", 1, True, 0), ("eagle3", 4, True, 3),
                                  ("draft_model", 1, True, 1), ("draft_model", 4, True, 4),
                                  ("eagle3", 4, False, 0), ("draft_model", 4, False, 1)):
    config = make_config(method=method, spec_k=k, parallel=parallel)
    got = config.speculative_config.max_num_new_slots_for_drafting
    slot_ok &= got == want
    slot_detail.append(f"{method}/par={parallel}/K={k}→{got}")
check("A1. 槽位净增：PARD=K、P-EAGLE=K−1（K=1→0）、串行 draft=1 / EAGLE=0",
      slot_ok, " ".join(slot_detail))

engine, core, runner = make_engine(method="draft_model", spec_k=4, parallel=True)
proposer = runner.proposer
check("A2. 提议者的 extra/net 与上游公式一致（PARD：extra=K=4、net=4、mask token 来自 config）",
      proposer.extra_slots_per_request == 4 and proposer.net_num_new_slots_per_request == 4
      and proposer.parallel_drafting_token_id == PARD_TOKEN
      and proposer.parallel_drafting_hidden_state_tensor is None,
      f"extra={proposer.extra_slots_per_request} net={proposer.net_num_new_slots_per_request}")
check("A3. 调度侧的 draft_slots 与提议者的 net 是同一个数",
      core.scheduler.draft_slots == proposer.net_num_new_slots_per_request == 4)
engine.shutdown()

engine, _core, runner = make_engine(method="eagle3", spec_k=4, parallel=True)
proposer = runner.proposer
check("A4. P-EAGLE：extra=K、net=K−1、带 mask hidden（宽度 = draft hidden）",
      proposer.extra_slots_per_request == 4 and proposer.net_num_new_slots_per_request == 3
      and proposer.parallel_drafting_token_id == MASK_TOKEN
      and tuple(proposer.parallel_drafting_hidden_state_tensor.shape) == (32,),
      f"net={proposer.net_num_new_slots_per_request}")
engine.shutdown()

rejected = []
for method in ("ngram", "ngram_gpu"):
    try:
        SpeculativeConfig(method=method, num_speculative_tokens=4, parallel_drafting=True)
    except ValueError:
        rejected.append(method)
check("A5. 方法边界：parallel_drafting 只与 EAGLE / draft_model 兼容（其余配置期拒绝）",
      len(rejected) == 2, f"拒绝 {rejected}")

# ---------------------------------------------------------------- B 输入协议（上游 kernel 逐值）


def compare_with_kernel(shift, k, reqs, qsl, base=5):
    from vllm.v1.spec_decode.utils import copy_and_expand_eagle_inputs_kernel

    rows = rows_from(reqs)
    total_in = qsl[-1]
    extra = k
    net = extra - (1 if shift else 0)
    total_out = total_in + net * len(reqs)
    t_ids = torch.arange(1, total_in + 1, dtype=torch.int32, device=DEVICE) + 100
    t_pos = torch.arange(total_in, dtype=torch.int32, device=DEVICE) + base
    next_ids = torch.tensor([r[4] for r in reqs], dtype=torch.int32, device=DEVICE)
    out_ids = torch.zeros(total_out, dtype=torch.int32, device=DEVICE)
    out_pos = torch.zeros(total_out, dtype=torch.int32, device=DEVICE)
    out_rej = torch.zeros(total_out, dtype=torch.bool, device=DEVICE)
    out_msk = torch.zeros(total_out, dtype=torch.bool, device=DEVICE)
    out_idx = torch.zeros(len(reqs) * extra, dtype=torch.int32, device=DEVICE)
    out_hid = torch.zeros(total_in, dtype=torch.int32, device=DEVICE)
    ends = [qsl[i + 1] - 1 - reqs[i][3] for i in range(len(reqs))]
    block = min(256, 2 ** max(0, (max(r[2] for r in reqs) + net - 1).bit_length()))
    copy_and_expand_eagle_inputs_kernel[(len(reqs), 1)](
        target_token_ids_ptr=t_ids, target_positions_ptr=t_pos, next_token_ids_ptr=next_ids,
        out_input_ids_ptr=out_ids, out_positions_ptr=out_pos,
        out_is_rejected_token_mask_ptr=out_rej, out_is_masked_token_mask_ptr=out_msk,
        out_new_token_indices_ptr=out_idx, out_hidden_state_mapping_ptr=out_hid,
        query_start_loc_ptr=torch.tensor(qsl, dtype=torch.int32, device=DEVICE),
        query_end_loc_ptr=torch.tensor(ends, dtype=torch.int32, device=DEVICE),
        padding_token_id=0, parallel_drafting_token_id=MASK_TOKEN,
        total_input_tokens=total_in, num_padding_slots_per_request=extra,
        shift_input_ids=shift, BLOCK_SIZE_TOKENS=block)
    torch.cuda.synchronize()
    mine = expand_parallel_draft_inputs(t_ids.cpu().tolist(), t_pos.cpu().tolist(), rows,
                                        extra_slots_per_request=extra,
                                        parallel_drafting_token_id=MASK_TOKEN,
                                        shift_input_ids=shift)
    same = (mine.num_tokens == total_out
            and mine.input_ids == out_ids.cpu().tolist()
            and mine.positions == out_pos.cpu().tolist()
            and mine.is_rejected == [int(x) for x in out_rej.cpu().tolist()]
            and mine.is_masked == [int(x) for x in out_msk.cpu().tolist()]
            and mine.token_indices_to_sample == [int(x) for x in out_idx.cpu().tolist()])
    if shift:
        same &= [mine.hidden_state_mapping.get(i, -1) for i in range(total_in)] == \
            out_hid.cpu().tolist()
    return same


if torch.cuda.is_available():
    cases = [(True, 1), (True, 2), (True, 4), (False, 1), (False, 2), (False, 4)]
    ok_all, bad = True, []
    for shift, k in cases:
        if not compare_with_kernel(shift, k, [("A", 0, 4, 3, 901), ("B", 4, 2, 1, 902)],
                                   [0, 4, 6]):
            ok_all, bad = False, bad + [(shift, k)]
    check("B1. B=2（长度 4/2、被拒 3/1）× shift ∈ {True,False} × K ∈ {1,2,4}："
          "input_ids/positions/两个 mask/采样行/hidden 映射与上游 kernel 逐值一致",
          ok_all, f"不一致 {bad}" if bad else "6 组全一致")

    ok2 = compare_with_kernel(True, 3, [("A", 0, 3, 2, 901), ("B", 3, 1, 0, 902),
                                        ("C", 4, 5, 4, 903)], [0, 3, 4, 9], base=17)
    check("B2. B=3、长度 3/1/5、被拒 2/0/4、base=17：整体仍逐值一致", ok2)
else:
    check("B1. 与上游 kernel 的逐值差分：待验（本机没有 CUDA）", False)
    check("B2. B=3 混合被拒：待验（本机没有 CUDA）", False)

pard = expand_parallel_draft_inputs(list(range(100, 106)), list(range(5, 11)),
                                    rows_from([("A", 0, 4, 3, 901), ("B", 4, 2, 1, 902)]),
                                    extra_slots_per_request=4,
                                    parallel_drafting_token_id=MASK_TOKEN,
                                    shift_input_ids=False)
check("B3. PARD 布局（纯函数，任何机器可跑）：每请求 = target 行数 + K，"
      "锚点 1 行 + mask K−1 行 + 被拒尾部掩码",
      pard.num_tokens == 8 + 6
      and pard.input_ids[:5] == [100, 901, MASK_TOKEN, MASK_TOKEN, MASK_TOKEN]
      and pard.positions[:5] == [5, 6, 7, 8, 9]
      and pard.is_masked[:5] == [0, 0, 1, 1, 1]
      and pard.is_rejected[5:8] == [1, 1, 1]
      and pard.token_indices_to_sample == [1, 2, 3, 4, 9, 10, 11, 12], pard.input_ids[:8])

# ---------------------------------------------------------------- C 一次 forward


def forwards_per_round(method, k, parallel):
    engine, _core, runner = make_engine(method=method, spec_k=k, parallel=parallel,
                                        max_num_seqs=1, budget=32)
    recorder = ParallelRunRecorder(runner)
    try:
        engine.add_request("A", [1, 2, 3, 4, 5, 6],
                           SamplingParams(max_tokens=6, temperature=0.0, eos_token_id=999))
        run_to_end(engine)
    finally:
        recorder.restore()
        engine.shutdown()
    return recorder


rec = forwards_per_round("draft_model", 3, True)
check("C1. PARD：每轮只跑 1 次 draft 前向（K=3）",
      rec.forwards == len(rec.rounds) and len(rec.rounds) >= 3,
      f"{rec.forwards} 次 / {len(rec.rounds)} 轮")
check("C2. 并行每轮每请求交回 K 枚草稿", all(sorted(r["drafts"]) == [3] * len(r["drafts"])
                                             for r in rec.rounds),
      f"{[r['drafts'] for r in rec.rounds]}")

rec = forwards_per_round("draft_model", 3, False)
check("C3. 对照：串行 K=3 每轮 3 次前向（第一遍 + 2 次自回归）",
      rec.forwards == 3 * len(rec.rounds), f"{rec.forwards} 次 / {len(rec.rounds)} 轮")

rec = forwards_per_round("eagle3", 3, True)
check("C4. P-EAGLE：每轮 1 次前向（复用 target 那一行 + K−1 个 mask）",
      rec.forwards == len(rec.rounds) and len(rec.rounds) >= 3,
      f"{rec.forwards} 次 / {len(rec.rounds)} 轮")

# ---------------------------------------------------------------- D 端到端

plain = greedy(req_ids=("A",), spec_k=None)
ok_all, detail = True, []
for method in ("eagle3", "draft_model"):
    for k in (1, 4):
        out = greedy(req_ids=("A",), method=method, spec_k=k, parallel=True)
        ok_all &= out["A"] == plain["A"]
        detail.append(f"{method}/K={k}")
check("D1. 并行提议的 greedy 输出 == 非投机（两种方法 × K=1/4）", ok_all, " ".join(detail))

results = {}
for parallel in (False, True):
    engine, _core, _runner = make_engine(method="draft_model", spec_k=2, parallel=parallel,
                                         max_num_seqs=2, budget=64, blocks=48)
    try:
        engine.add_request("A", [1, 2, 3, 4, 5, 6],
                           SamplingParams(max_tokens=4, temperature=0.0, eos_token_id=999))
        engine.add_request("B", [1, 2, 3, 4, 5, 6, 7, 8],
                           SamplingParams(max_tokens=4, temperature=0.0, eos_token_id=999))
        results[parallel] = run_to_end(engine)
    finally:
        engine.shutdown()
check("D2. B=2（两请求长度不同、混批 prefill）下并行与串行输出逐 token 相同",
      results[True] == results[False], str(results[True]))

if torch.cuda.is_available():
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
    good = True
    for round_ in recorder.rounds:
        for index, flag in enumerate(round_["is_masked"]):
            if flag:
                good &= bool(torch.allclose(round_["hidden"][index], mask, atol=1e-6))
                checked += 1
    check("D3. P-EAGLE 的 mask 行 hidden 换成模型自带的 mask 向量（含被拒行与 mask 区重叠的情形）",
          good and checked > 0, f"验了 {checked} 行")

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
    first = recorder.rounds[0]
    check("D4. 引擎真正写进工作区的行：2 请求 × (K−1)=2 个 mask、采样行 2×K、"
          "首轮没有被拒行",
          first["is_masked"].count(1) == 4 and len(first["sample_rows"]) == 6
          and first["is_rejected"].count(1) == 0,
          f"mask={first['is_masked'].count(1)} 采样行={len(first['sample_rows'])} "
          f"rejected={first['is_rejected'].count(1)}")
else:
    check("D3. mask 行 hidden：待验（本机没有 CUDA）", False)
    check("D4. 工作区行数：待验（本机没有 CUDA）", False)

# ---------------------------------------------------------------- E 模型格式边界

serial_dir, serial_hf = draft_dir("eagle3", parallel=False)
serial_hf = dict(serial_hf, parallel_drafting=True, mask_token_id=MASK_TOKEN)
try:
    config = make_config(method="eagle3", spec_k=2, parallel=True, draft=(serial_dir, serial_hf))
    LLMEngine(config, UniProcExecutor(config, Worker(config)))
    rejected_serial = False
except ValueError as exc:
    rejected_serial = "mask_hidden" in str(exc)
check("E1. 串行训练的 EAGLE3 权重开并行 → 加载期明确报错（不能拿它做并行验收）",
      rejected_serial)

pe_dir, pe_hf = draft_dir("eagle3", parallel=True)
stripped = {key: value for key, value in pe_hf.items() if key != "mask_token_id"}
try:
    config = make_config(method="eagle3", spec_k=2, parallel=True, draft=(pe_dir, stripped))
    LLMEngine(config, UniProcExecutor(config, Worker(config)))
    rejected_token = False
except ValueError as exc:
    rejected_token = "mask token" in str(exc)
check("E2. draft 配置里没有 mask token 字段 → 构造期报错（不猜一个 token）", rejected_token)

print(f"\n{'全部通过' if not FAILED else '失败: ' + ', '.join(FAILED)}  （{PASSED} 项通过）")
sys.exit(1 if FAILED else 0)
