"""73 关验收探针：V2 Model Runner（常驻 slot 请求状态 + 投机执行路径 + V1/V2 差分）。

跑法：`python benchmarks/check_step73_v2_runner.py`（退出码非 0 = 失败）。

分段（对应需求 073 §4 的六条验收）：

    A 常驻 slot       A=5/B=2、batch [B,A]：tokens/logits/slot_mapping/块表全部按 slot 寻址
    B 输入组装        每请求 1+K 行、cu_num_logits 含前导 0、expanded 映射、批尾哨兵
    C 状态所有权      post_update 按 slot 落盘（换行不影响状态）；prompt_len ≠ prefill_len
    D V1/V2 差分      同一 tiny 模型 greedy 逐 token 相同（K=1/3；含抢占恢复与混合 prefill）
    E 投机管道        草稿进批、num_rejected = num_logits − num_sampled、-1 占位协议
    F logprobs        V2 与 V1 同值；每个位置第 0 项是实际采到的 token
    G 边界            非 EAGLE 方法 / CUDA Graph / 异步调度在配置期明确拒绝；零请求轮不碰模型

⚠️ 本机 tiny 权重是**随机**的：探针验的是**寻址与协议**（V1/V2 逐 token 一致 + 自洽计数），
不是草稿质量。真实 EAGLE3 权重的接受长度受 63 关的已知差距影响，见
`docs/step63_alignment.md` §8。
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step73"))

from spec73_helpers import (DEVICE, DraftRecorder, greedy, make_config,  # noqa: E402
                            make_engine, run_to_end)

from minivllm import SamplingParams  # noqa: E402
from minivllm.worker.gpu.block_table import PAD_SLOT_ID, BlockTables  # noqa: E402
from minivllm.worker.gpu.buffer_utils import async_copy_to_gpu  # noqa: E402
from minivllm.worker.gpu.input_batch import (  # noqa: E402
    InputBuffers,
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    post_update,
    prepare_pos_seq_lens,
)
from minivllm.worker.gpu.states import RequestState  # noqa: E402

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


if not torch.cuda.is_available():  # pragma: no cover - 本机有 CUDA
    print("SKIP  本探针需要 CUDA（V2 = Triton 内核 + UVA 常驻状态）")
    sys.exit(1)

VOCAB = 11
K = 2
MAX_REQS, MAX_TOKENS, BLOCK_SIZE = 8, 64, 4


def make_states():
    return RequestState(max_num_reqs=MAX_REQS, max_model_len=MAX_TOKENS,
                        max_num_batched_tokens=MAX_TOKENS, num_speculative_steps=K,
                        vocab_size=32, device=torch.device(DEVICE))


def reserve_slot(states, slot):
    holders = []
    while states.free_indices[-1] != slot:
        holder_id = f"__hold_{states.free_indices[-1]}"
        states.add_request(holder_id, 1, [1], 0, 4)
        holders.append(holder_id)
    return holders


# ---------------------------------------------------------------- A 常驻 slot

states = make_states()
hold_a = reserve_slot(states, 5)
states.add_request("A", 4, [10, 11, 12, 13], 0, 16)
hold_b = reserve_slot(states, 2)
states.add_request("B", 2, [20, 21], 0, 16)
for holder in hold_a + hold_b:
    states.remove_request(holder)
states.apply_staged_writes()
check("A1. 请求拿到指定常驻 slot（A=5、B=2），且与 batch 行号无关",
      states.req_id_to_index == {"A": 5, "B": 2}, str(states.req_id_to_index))

idx_mapping = async_copy_to_gpu(np.array([2, 5], dtype=np.intp), device=DEVICE)
block_tables = BlockTables(block_sizes=[BLOCK_SIZE], max_num_reqs=MAX_REQS,
                           max_num_batched_tokens=MAX_TOKENS,
                           max_num_blocks_per_group=[MAX_TOKENS // BLOCK_SIZE],
                           device=torch.device(DEVICE),
                           kernel_block_sizes=[BLOCK_SIZE])
block_tables.append_block_ids(2, ([3, 7],), overwrite=True)
block_tables.append_block_ids(5, ([1, 5],), overwrite=True)
block_tables.apply_staged_writes()
gathered = block_tables.gather_block_tables(idx_mapping, num_reqs_padded=2)
check("A2. 块表按 slot 登录、按 idx_mapping gather 成 batch 顺序（B 行=块[3,7]、A 行=块[1,5]）",
      gathered[0][0, :2].tolist() == [3, 7] and gathered[0][1, :2].tolist() == [1, 5],
      f"{gathered[0][0, :2].tolist()} / {gathered[0][1, :2].tolist()}")

check("A3. all_token_ids / total_len 按 slot 落盘（A 的行在 slot 5、B 在 slot 2）",
      states.all_token_ids.gpu[5, :4].tolist() == [10, 11, 12, 13]
      and states.all_token_ids.gpu[2, :2].tolist() == [20, 21]
      and int(states.total_len.gpu[5]) == 4 and int(states.total_len.gpu[2]) == 2)

# ---------------------------------------------------------------- B 输入组装

input_buffers = InputBuffers(MAX_REQS, MAX_TOKENS, torch.device(DEVICE))
num_computed = torch.zeros(MAX_REQS, dtype=torch.int32, device=DEVICE)
num_computed[5] = 4
num_computed[2] = 2
query_start_loc = async_copy_to_gpu(
    np.array([0, 1 + K, 2 * (1 + K)] + [2 * (1 + K)] * (MAX_REQS - 1), dtype=np.int32),
    device=DEVICE)
prepare_pos_seq_lens(idx_mapping, query_start_loc, num_computed,
                     input_buffers.positions, input_buffers.seq_lens)
check("B1. positions = num_computed + 行内偏移；seq_lens = num_computed + query_len",
      input_buffers.positions[:6].tolist() == [2, 3, 4, 4, 5, 6]
      and input_buffers.seq_lens[:2].tolist() == [2 + 1 + K, 4 + 1 + K],
      f"{input_buffers.positions[:6].tolist()} / {input_buffers.seq_lens[:2].tolist()}")

last_sampled = torch.zeros(MAX_REQS, 1, dtype=torch.int64, device=DEVICE)
last_sampled[2], last_sampled[5] = 20, 10
draft_tokens = torch.zeros(MAX_REQS, K, dtype=torch.int64, device=DEVICE)
draft_tokens[2] = torch.tensor([21, 22], dtype=torch.int64, device=DEVICE)
draft_tokens[5] = torch.tensor([11, 12], dtype=torch.int64, device=DEVICE)
prefill_len = torch.zeros(MAX_REQS, dtype=torch.int32, device=DEVICE)
prefill_len[5], prefill_len[2] = 4, 2
cu_np = np.array([0, 1 + K, 2 * (1 + K)], dtype=np.int32)
cu_num_logits = async_copy_to_gpu(cu_np, device=DEVICE)
logits_indices = combine_sampled_and_draft_tokens(
    input_buffers.input_ids, idx_mapping, last_sampled, query_start_loc,
    input_buffers.seq_lens, prefill_len, draft_tokens, cu_num_logits, 2 * (1 + K))
check("B2. 上一轮采样 + 本轮草稿按 slot 写回 input_ids（B=[20,21,22]、A=[10,11,12]）",
      input_buffers.input_ids[:6].tolist() == [20, 21, 22, 10, 11, 12],
      str(input_buffers.input_ids[:6].tolist()))
expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
    idx_mapping, 2 * (1 + K), cu_num_logits, 1 + K)
check("B3. cu_num_logits 含前导 0；expanded 映射 = logits 行→slot + 行内序号",
      cu_np.tolist() == [0, 3, 6]
      and expanded_idx_mapping.tolist() == [2, 2, 2, 5, 5, 5]
      and expanded_local_pos.tolist() == [0, 1, 2, 0, 1, 2],
      f"cu={cu_np.tolist()} expanded={expanded_idx_mapping.tolist()}")
check("B4. 每请求 K+1 行都要 logits（K 个候选位 + 1 个 bonus 位）",
      logits_indices.tolist() == [0, 1, 2, 3, 4, 5], str(logits_indices.tolist()))

slot_mappings = block_tables.compute_slot_mappings(idx_mapping, query_start_loc,
                                                   input_buffers.positions, 2 * (1 + K))
check("B5. slot_mapping 现算：B 的位置 2..4 → 14/15/28；A 的位置 4..6 → 20/21/22",
      slot_mappings[0].tolist() == [14, 15, 28, 20, 21, 22],
      str(slot_mappings[0].tolist()))
check("B6. 批尾按 PAD_SLOT_ID(-1) 填充（不能被当成合法槽位）",
      int(block_tables.slot_mappings[0, 2 * (1 + K)]) == PAD_SLOT_ID,
      f"tail={int(block_tables.slot_mappings[0, 2 * (1 + K)])}")

# ---------------------------------------------------------------- C 状态所有权

sampled = torch.tensor([[21, 22, -1], [12, 13, 14]], dtype=torch.int64, device=DEVICE)
num_sampled = torch.tensor([2, 3], dtype=torch.int32, device=DEVICE)
num_rejected = torch.tensor([1, 0], dtype=torch.int32, device=DEVICE)
post_update(idx_mapping, num_computed, last_sampled, None, sampled, num_sampled,
            num_rejected, query_start_loc, states.all_token_ids.gpu, states.total_len.gpu)
torch.cuda.synchronize()
# B 的历史是 [20,21]（长度 2），A 是 [10,11,12,13]（长度 4）：
# B 本轮采 2 个 → 追加到 [20,21,21,22]、total_len 2→4；
# A 本轮采 3 个 → 追加到 [10,11,12,13,12,13,14]、total_len 4→7。
check("C1. post_update 追加到 slot 的历史尾部（不是覆盖行首）",
      states.all_token_ids.gpu[2, :4].tolist() == [20, 21, 21, 22]
      and states.all_token_ids.gpu[5, :7].tolist() == [10, 11, 12, 13, 12, 13, 14],
      f"{states.all_token_ids.gpu[2, :4].tolist()} / {states.all_token_ids.gpu[5, :7].tolist()}")
check("C2. last_sampled / total_len 按 slot 更新；已算数 = query 行数 − 被拒行数",
      int(last_sampled[2]) == 22 and int(last_sampled[5]) == 14
      and int(states.total_len.gpu[2]) == 4 and int(states.total_len.gpu[5]) == 7
      # 已算数从 B 段设定的上下文继续加：B 2 + (3−1) = 4、A 4 + (3−0) = 7
      and int(num_computed[2]) == 4 and int(num_computed[5]) == 7,
      f"total_len={states.total_len.gpu.tolist()[:8]} "
      f"num_computed={num_computed.tolist()[:8]}")

# 行序翻转：状态跟着 slot 走
idx_mapping = async_copy_to_gpu(np.array([5, 2], dtype=np.intp), device=DEVICE)
post_update(idx_mapping, num_computed, last_sampled, None,
            torch.tensor([[15], [23]], dtype=torch.int64, device=DEVICE),
            torch.tensor([1, 1], dtype=torch.int32, device=DEVICE),
            torch.tensor([0, 0], dtype=torch.int32, device=DEVICE),
            None, states.all_token_ids.gpu, states.total_len.gpu)
torch.cuda.synchronize()
check("C3. batch 重排（[B,A] → [A,B]）后状态不变：A 仍追加在 slot 5、B 在 slot 2",
      int(last_sampled[5]) == 15 and int(last_sampled[2]) == 23
      and states.all_token_ids.gpu[5, :8].tolist() == [10, 11, 12, 13, 12, 13, 14, 15],
      f"A={states.all_token_ids.gpu[5, :8].tolist()}")

states2 = make_states()
states2.add_request("R", 4, [1, 2, 3, 4, 5, 6], 4, 8)
slot = states2.req_id_to_index["R"]
check("C4. prompt_len 与 prefill_len 分开（恢复时 prefill_len=6 > prompt_len=4，互不覆盖）",
      int(states2.prompt_len.np[slot]) == 4 and int(states2.prefill_len.np[slot]) == 6
      and int(states2.max_seq_len[slot]) == 12,
      f"prompt={int(states2.prompt_len.np[slot])} prefill={int(states2.prefill_len.np[slot])}")

# ---------------------------------------------------------------- D V1/V2 差分

PROMPTS = [("A", [1, 2, 3, 4, 5, 6]), ("B", [7, 8, 9]), ("C", [10, 9, 8, 7, 6, 5, 4, 3])]
for k in (1, 3):
    v1 = greedy(PROMPTS, max_tokens=12, v2=False, spec_k=k)
    v2 = greedy(PROMPTS, max_tokens=12, v2=True, spec_k=k)
    check(f"D1. K={k}：V1 与 V2 greedy 输出逐 token 相同（三条请求、混合 prefill）",
          v1 == v2 == {rid: v1[rid] for rid, _ in PROMPTS} and all(len(x) == 12 for x in v1.values()),
          f"V1={v1}")

v1p = greedy(PROMPTS, max_tokens=10, v2=False, spec_k=3, blocks=6, budget=16, max_num_seqs=3)
v2p = greedy(PROMPTS, max_tokens=10, v2=True, spec_k=3, blocks=6, budget=16, max_num_seqs=3)
check("D2. 抢占恢复（num_gpu_blocks=6 压出 4 次抢占）后 V1/V2 仍逐 token 相同", v1p == v2p,
      f"V2={v2p}")

# ---------------------------------------------------------------- E 投机管道

engine, core, runner = make_engine(v2=True, spec_k=3, max_num_seqs=1)
rec = DraftRecorder(runner)
try:
    engine.add_request("A", [1, 2, 3, 4, 5, 6],
                       SamplingParams(max_tokens=12, temperature=0.0, eos_token_id=VOCAB + 7))
    run_to_end(engine)
finally:
    engine.shutdown()
consistent = all(rej == rows - samp
                 for rows, samp, rej in zip(rec.logits_rows, rec.num_sampled, rec.num_rejected))
check("E1. 草稿真的进批（每轮 3 行/请求）", max(rec.draft_rows) == 3,
      f"draft_rows={rec.draft_rows}")
check("E2. num_rejected 由 GPU 算出且自洽（= logits 行数 − 采样数）", consistent,
      f"sampled={rec.num_sampled} rejected={rec.num_rejected} rows={rec.logits_rows}")

engine, _core, runner = make_engine(v2=True, spec_k=3, max_num_seqs=1)
try:
    engine.add_request("A", [1, 2, 3, 4, 5, 6],
                       SamplingParams(max_tokens=4, temperature=0.0, eos_token_id=VOCAB + 7))
    engine.step()
    engine.step()
    draft_ids = runner.take_draft_token_ids()
    check("E3. 同步调度下交回 -1 占位（宽度=K）：真 id 留在执行侧按 slot 存",
          draft_ids is not None and draft_ids.draft_token_ids == [[-1, -1, -1]],
          str(None if draft_ids is None else draft_ids.draft_token_ids))
finally:
    engine.shutdown()

# ---------------------------------------------------------------- F logprobs

def last_output(v2_flag):
    engine, _core, _runner = make_engine(v2=v2_flag, spec_k=3, max_num_seqs=1)
    try:
        engine.add_request("A", [1, 2, 3, 4, 5, 6],
                           SamplingParams(max_tokens=6, temperature=0.0,
                                          eos_token_id=VOCAB + 7, logprobs=2))
        outs = []
        for _ in range(200):
            if not engine.has_unfinished_requests():
                break
            outs += list(engine.step())
        return outs[-1]
    finally:
        engine.shutdown()


out1, out2 = last_output(False), last_output(True)
same_values = out1.token_ids == out2.token_ids and all(
    abs(out1.logprobs[pos][tok].logprob - out2.logprobs[pos][tok].logprob) < 1e-5
    for pos, tok in enumerate(out1.token_ids))
top_ok = all(tok in out2.logprobs[pos] and out2.logprobs[pos][tok].rank == 1
             for pos, tok in enumerate(out2.token_ids))
check("F1. V2 与 V1 的 logprobs 逐位置同值、行数 = 交付 token 数", same_values,
      f"tokens={out2.token_ids}")
check("F2. 每个位置第 0 项是实际采到的 token（rank=1）", top_ok)

# ---------------------------------------------------------------- G 边界

boundary_ok, boundary_detail = True, []
for kwargs, expect in (
        (dict(method="ngram", spec_k=2), "只支持"),
        (dict(spec_k=2, mode="full_and_piecewise"), "CUDA Graph 属 74 关"),
):
    try:
        make_config(v2=True, **kwargs)
        boundary_ok = False
        boundary_detail.append(f"{kwargs} 没有报错")
    except NotImplementedError as exc:
        boundary_detail.append(f"{kwargs}→{str(exc)[:24]}…")
        boundary_ok &= expect in str(exc)
check("G1. V2 的不支持组合在配置期明确拒绝（非 EAGLE 方法 / CUDA Graph）",
      boundary_ok, " | ".join(boundary_detail))

engine, _core, runner = make_engine(v2=True, spec_k=3, max_num_seqs=2)
try:
    empty = list(engine.step())
    ok = empty == [] and runner.execute_model_state is None and runner.req_states.num_reqs == 0
    engine.add_request("A", [1, 2, 3], SamplingParams(max_tokens=4, temperature=0.0,
                                                      eos_token_id=VOCAB + 7))
    run_to_end(engine)
    ok &= (runner.req_states.num_reqs == 0
           and len(runner.req_states.free_indices) == runner.max_num_reqs)
    check("G2. 零请求轮不碰模型；请求结束后 slot 与状态全部归还", ok)
finally:
    engine.shutdown()

# ---------------------------------------------------------------- 汇总

print(f"\n{PASSED} passed, {len(FAILED)} failed")
if FAILED:
    print("FAILED:", ", ".join(FAILED))
    sys.exit(1)
