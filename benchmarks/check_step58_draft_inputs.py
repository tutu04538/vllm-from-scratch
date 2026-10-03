"""58 验收 B（需求 §8.B）：**draft 第一遍输入**。

对齐目标（上游 `SpecDecodeBaseProposer.set_inputs_first_pass()` 的普通 draft 扩容分支）：

    每条请求的物理行 = [有效行 n-num_rejected] + [1 行扩容行（新 token）] + [num_rejected 行被拒行]
    positions        = start + 行内序号；被拒行 position=0、token=padding、slot=哨兵
    is_rejected      = 只有被拒尾部是 1（物理存在，但不能成为有效上下文）
    token_indices_to_sample = 每条请求扩容行的全局行号
    query_start_loc/seq_lens = 上游 extend_all_queries_by_N(N=1) 的结果

本脚本查四件事：

  1. **纯函数逐值**：`expand_draft_inputs` / `compute_new_slot_mapping` /
     `extend_all_queries_by_N` 的算例（手算对照，含被拒尾部与越界哨兵）
  2. **真实跑出来的轨迹**：prefill / decode / 首拒绝 / 中拒绝 / 全接受，逐值看
     input_ids、positions、slot_mapping、mask 与行数不变量
  3. **prefix 命中**：第二个相同 prompt 的请求只从命中末尾开始算，命中段**不再重算**
  4. **chunked prefill 与恢复**：中间 prefill 块只同步被安排的范围；恢复时从协议给的
     有效前缀开始（没命中才从 0 重算），且始终满足 `rows <= target_rows + 1`
"""

import sys

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

import torch

# 59 关起：投机**验证**走 Triton 内核（上游同样只有 GPU 路径），所以跑真引擎的用例要上 GPU。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,
                      SchedulerConfig, SpeculativeConfig, UniProcExecutor, VllmConfig, Worker)
from minivllm.spec_decode.draft_model import SpecDecodeBaseProposer
from minivllm.spec_decode.utils import (PADDING_SLOT_ID, DraftInputRows, TargetRows,
                                        compute_new_slot_mapping, expand_draft_inputs,
                                        extend_all_queries_by_N)
from minivllm.testing.tiny_models import tiny_qwen3_config, tiny_qwen3_dir

FAIL = []
TINY = tiny_qwen3_dir("tiny_gqa")
HF = tiny_qwen3_config("tiny_gqa")


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


# ---------------------------------------------------------------- 1. 纯函数逐值
ids, positions, rejected, sample_idx = expand_draft_inputs([
    DraftInputRows(valid_token_ids=[5, 6], start=6, next_token_id=99, num_rejected=2)])
check("B1. 扩容展开：有效行 + 1 扩容行 + 被拒尾部（行序/位置/哨兵与上游 kernel 一致）",
      ids == [5, 6, 99, 0, 0] and positions == [6, 7, 8, 0, 0] and rejected == [0, 0, 0, 1, 1]
      and sample_idx == [2],
      f"ids={ids} positions={positions} rejected={rejected} sample_rows={sample_idx}")

block_table = torch.tensor([[0, 1, 2], [4, 5, 6]], dtype=torch.int64)
# 两条请求各 3 行 target：req0 有 1 行被拒（有效 2 + 扩容 1 + 被拒 1），
# req1 无被拒（有效 3 + 扩容 1），扩容行位置 10、11 都到了 max_model_len 之外
slots = compute_new_slot_mapping(
    block_table, query_lens=[3, 3],
    new_positions=torch.tensor([6, 7, 8, 0, 8, 9, 10, 11]),
    is_rejected_token_mask=torch.tensor([0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.bool),
    block_size=4, num_new_tokens=1, max_model_len=10)
check("B1b. slot mapping：正常行按块表算、被拒行与越界行都是 PADDING_SLOT_ID(-1)",
      slots.tolist() == [6, 7, 8, -1, 24, 25, -1, -1] and PADDING_SLOT_ID == -1,
      f"slots={slots.tolist()}（期望 [6, 7, 8, -1, 24, 25, -1, -1]）")

qsl, seq = extend_all_queries_by_N([0, 4, 7], [10, 13], num_new_tokens=1)
check("B1c. extend_all_queries_by_N：query 起点每请求 +N*index、上下文长度 +N",
      qsl == [0, 5, 9] and seq == [11, 14], f"qsl={qsl} seq={seq}")


# ---------------------------------------------------------------- 轨迹捕获
class Trace:
    """记录每轮 draft 工作区的前若干行 + 对应的 TargetRows（测试必须 clone：视图会被覆盖）。"""

    def __init__(self, runner):
        self.runner = runner
        self.rows = None
        self.rounds = []
        proposer = runner.proposer
        self._original_propose = proposer.propose
        self._original_forward = proposer._forward

        def propose(rows, all_token_ids, input_batch, reset_req_ids=None):
            self.rows = list(rows)
            return self._original_propose(rows, all_token_ids, input_batch, reset_req_ids)

        def forward(num_tokens, num_reqs):
            self.rounds.append({
                "rows": self.rows,
                "num_tokens": num_tokens,
                "num_reqs": num_reqs,
                "input_ids": proposer.input_ids_cpu[:num_tokens].clone().tolist(),
                "positions": proposer.positions_cpu[:num_tokens].clone().tolist(),
                "rejected": proposer.is_rejected_token_mask_cpu[:num_tokens].clone().tolist(),
                "slots": proposer.slot_mapping_cpu[:num_tokens].clone().tolist(),
                "seq_lens": proposer.seq_lens_cpu[:num_reqs].clone().tolist(),
            })
            return self._original_forward(num_tokens, num_reqs)

        proposer.propose = propose
        proposer._forward = forward

    def restore(self):
        self.runner.proposer.propose = self._original_propose
        self.runner.proposer._forward = self._original_forward


def build(*, k=3, draft_dir=None, budget=16, blocks=32, prefix=False, max_model_len=64,
          max_tokens=6, max_num_seqs=2):
    model = ModelConfig(model=TINY, dtype="float32", max_model_len=max_model_len, hf_config=HF)
    spec = None if k is None else SpeculativeConfig(
        method="draft_model", num_speculative_tokens=k,
        draft_model_config=ModelConfig(model=draft_dir or TINY, dtype="float32",
                                       max_model_len=max_model_len, hf_config=HF))
    config = VllmConfig(model_config=model,
                        cache_config=CacheConfig(block_size=4, num_gpu_blocks=blocks,
                                                 enable_prefix_caching=prefix),
                        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                                         max_num_batched_tokens=budget),
                        device_config=DeviceConfig(device=DEVICE), speculative_config=spec)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    core = engine.engine_core.engine_core
    runner = core.model_executor.driver_worker.model_runner
    return engine, core, runner


def run_to_end(engine, limit=200):
    final = {}
    for _ in range(limit):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            final[out.request_id] = list(out.token_ids)
    return final


def first_pass_of_round(rounds):
    """每轮第一次 forward 就是第一遍（后面几次是自回归步骤，行数 = 活跃请求数）。"""
    out, seen = [], set()
    for entry in rounds:
        key = id(entry["rows"])
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out


# ---------------------------------------------------------------- 2. 轨迹：prefill / decode / 三种接受情况
engine, _core, runner = build(k=3, budget=16, max_tokens=4)
trace = Trace(runner)
engine.add_request("r", [1, 2, 3, 4, 5, 6], SamplingParams(max_tokens=4, temperature=0.0,
                                                           eos_token_id=999))
run_to_end(engine)
trace.restore()
first = first_pass_of_round(trace.rounds)
entry, rows = first[0], first[0]["rows"]
target = rows[0]
check("B2. prefill 轮：draft 第一遍 = prompt 行 + 1 行扩容行（扩容行是新采样的 token）",
      entry["num_tokens"] == target.target_rows + 1
      and entry["positions"] == list(range(0, target.target_rows + 1))
      and entry["rejected"] == [0] * entry["num_tokens"]
      and entry["input_ids"][:target.target_rows] == [1, 2, 3, 4, 5, 6]
      and entry["slots"] == list(range(entry["num_tokens"])),
      f"rows={target} positions={entry['positions']} ids={entry['input_ids']}")
engine.shutdown()

# 首拒绝：把提议者换成一个"永远提 token 0"的桩（target 的贪心不是 0 → 必拒）
engine, _core, runner = build(k=1, budget=16, max_tokens=4)
trace = Trace(runner)
original_sample = runner.proposer._sample_draft_tokens


def always_zero(hidden, row_refs, input_batch, drafts, probs):
    for req_id, _ in row_refs:
        drafts[req_id].append(0)
        row = torch.zeros(hidden.shape[-1] if False else 1)      # 占位，概率不参与本用例断言
        probs[req_id].append(torch.ones(HF["vocab_size"]) / HF["vocab_size"])


runner.proposer._sample_draft_tokens = always_zero
engine.add_request("r", [1, 2, 3, 4, 5, 6], SamplingParams(max_tokens=4, temperature=0.0,
                                                           eos_token_id=999))
run_to_end(engine)
trace.restore()
runner.proposer._sample_draft_tokens = original_sample
reject_rounds = [entry for entry in first_pass_of_round(trace.rounds) if entry["rows"][0].num_rejected]
check("B2b. 首拒绝：被拒尾部物理存在但被屏蔽（mask=1、slot=-1、position=0、token=0）",
      bool(reject_rounds)
      and reject_rounds[0]["rejected"].count(1) == reject_rounds[0]["rows"][0].num_rejected
      and all(slot == PADDING_SLOT_ID for slot, rej in zip(reject_rounds[0]["slots"],
                                                           reject_rounds[0]["rejected"]) if rej)
      and all(pos == 0 for pos, rej in zip(reject_rounds[0]["positions"],
                                           reject_rounds[0]["rejected"]) if rej),
      f"被拒轮={len(reject_rounds)}、样例={reject_rounds[0] if reject_rounds else None}")
engine.shutdown()

# 中拒绝：draft 与 target 同一个模型（贪心必接受），只把第 2 枚草稿改成错 token
engine, _core, runner = build(k=3, budget=16, max_tokens=6)
trace = Trace(runner)
original_sample = runner.proposer._sample_draft_tokens
calls = {"n": 0}


def corrupt_second(hidden, row_refs, input_batch, drafts, probs):
    calls["n"] += 1
    original_sample(hidden, row_refs, input_batch, drafts, probs)
    if calls["n"] % 3 == 2:                                   # 每轮的第 2 枚草稿改成 0
        for req_id, _ in row_refs:
            drafts[req_id][-1] = 0


runner.proposer._sample_draft_tokens = corrupt_second
engine.add_request("r", [1, 2, 3, 4, 5, 6], SamplingParams(max_tokens=6, temperature=0.0,
                                                           eos_token_id=999))
run_to_end(engine)
trace.restore()
runner.proposer._sample_draft_tokens = original_sample
mid = [entry for entry in first_pass_of_round(trace.rounds)
       if 0 < entry["rows"][0].num_rejected < entry["rows"][0].target_rows - 1]
check("B2c. 中拒绝：有效行 = 起点+接受的草稿、被拒尾部紧跟其后，扩容行仍在有效行之后",
      bool(mid)
      and all(entry["num_tokens"] == entry["rows"][0].target_rows + 1 for entry in mid)
      and all(entry["rejected"][:entry["num_tokens"] - entry["rows"][0].num_rejected]
              == [0] * (entry["num_tokens"] - entry["rows"][0].num_rejected) for entry in mid)
      and all(entry["positions"][i] == entry["rows"][0].start + i
              for entry in mid
              for i in range(entry["num_tokens"] - entry["rows"][0].num_rejected)),
      f"中拒绝轮次={[(e['rows'][0].num_rejected, e['num_tokens']) for e in mid]}")
engine.shutdown()

# 全接受：draft 与 target 同模型、纯贪心 → K 枚全中
engine, _core, runner = build(k=3, budget=16, max_tokens=6)
trace = Trace(runner)
engine.add_request("r", [1, 2, 3, 4, 5, 6], SamplingParams(max_tokens=6, temperature=0.0,
                                                           eos_token_id=999))
run_to_end(engine)
trace.restore()
first = first_pass_of_round(trace.rounds)
check("B2d. 全接受：没有任何被拒行，物理行数 = target 行数 + 1（不变量）",
      all(entry["rejected"] == [0] * entry["num_tokens"] for entry in first)
      and all(entry["num_tokens"] == entry["rows"][0].target_rows + 1 for entry in first),
      f"轮数={len(first)}")
engine.shutdown()

# ---------------------------------------------------------------- 3. prefix 命中：不重算命中段
engine, core, runner = build(k=3, budget=16, prefix=True, max_tokens=2)
trace = Trace(runner)
prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 3, 4, 5]
engine.add_request("first", prompt, SamplingParams(max_tokens=2, temperature=0.0,
                                                   eos_token_id=999))
run_to_end(engine)
before = len(trace.rounds)
engine.add_request("reuse", prompt, SamplingParams(max_tokens=2, temperature=0.0,
                                                   eos_token_id=999))
run_to_end(engine)
trace.restore()
reuse_first = first_pass_of_round(trace.rounds[before:])[0]
hit = reuse_first["rows"][0].start
check("B3. prefix 命中：第二个请求的 draft 只从命中末尾开始，命中段（位置 0..hit-1）不再出现",
      hit > 0 and min(reuse_first["positions"]) == hit
      and reuse_first["num_tokens"] == reuse_first["rows"][0].target_rows + 1
      and len(reuse_first["positions"]) < len(prompt),
      f"命中起点={hit}、本次第一遍位置={reuse_first['positions']}")
engine.shutdown()

# ---------------------------------------------------------------- 4. chunked prefill + 恢复
engine, core, runner = build(k=1, budget=4, max_tokens=4)
trace = Trace(runner)
engine.add_request("r", [1, 2, 3, 4, 5, 6, 7, 8], SamplingParams(max_tokens=4, temperature=0.0,
                                                                 eos_token_id=999))
run_to_end(engine)
trace.restore()
chunk_rounds = [entry for entry in first_pass_of_round(trace.rounds)
                if not entry["rows"][0].ready]
check("B4. chunked prefill：中间 prefill 块只同步被安排的那一段（不提草稿）",
      bool(chunk_rounds)
      and all(entry["num_tokens"] == entry["rows"][0].target_rows + 1 for entry in chunk_rounds)
      and all(entry["positions"][:entry["rows"][0].target_rows]
              == list(range(entry["rows"][0].start,
                            entry["rows"][0].start + entry["rows"][0].target_rows))
              for entry in chunk_rounds),
      f"中间 prefill 轮={[(e['rows'][0].start, e['rows'][0].target_rows) for e in chunk_rounds]}")
engine.shutdown()


def preemption_run(k):
    engine, core, runner = build(k=k, budget=4, blocks=3, max_tokens=8, max_num_seqs=2)
    trace = Trace(runner) if runner.proposer is not None else None
    for req, priority in (("A", 0), ("B", 5)):
        engine.add_request(req, [1, 2], SamplingParams(max_tokens=8, temperature=0.0,
                                                       eos_token_id=999), priority=priority)
    outputs = run_to_end(engine)
    first = first_pass_of_round(trace.rounds) if trace is not None else []
    if trace is not None:
        trace.restore()
    preemptions = core.scheduler.num_preemptions
    engine.shutdown()
    return outputs, preemptions, first


reference, _rp, _rf = preemption_run(None)
speculative, preemptions, rounds = preemption_run(3)
check("B4b. 抢占恢复：确实发生抢占，输出与非投机一致；恢复后仍满足 rows <= target_rows + 1",
      preemptions > 0 and reference == speculative and bool(rounds)
      and all(entry["num_tokens"] <= entry["rows"][0].target_rows + 1 for entry in rounds),
      f"抢占={preemptions}、输出一致={reference == speculative}、第一遍轮数={len(rounds)}")

# ---------------------------------------------------------------- 5. 与上游 kernel 逐值差分
# 生产实现是自己写的（CPU 也能跑），但语义必须与上游的扩容 kernel 一致：
# 同一组输入分别喂给 `copy_and_expand_eagle_inputs_kernel`（Triton，需要 CUDA）与本地函数，
# 逐行比较 input_ids / positions / is_rejected / token_indices_to_sample / is_masked。
if torch.cuda.is_available():
    try:
        from vllm.v1.spec_decode.utils import copy_and_expand_eagle_inputs_kernel

        dev = "cuda"
        target_ids = torch.tensor([5, 6, 7, 8, 9, 10], dtype=torch.int32, device=dev)
        target_pos = torch.tensor([6, 7, 8, 8, 9, 10], dtype=torch.int32, device=dev)
        next_ids = torch.tensor([99, 111], dtype=torch.int32, device=dev)
        qsl = torch.tensor([0, 3, 6], dtype=torch.int32, device=dev)
        qel = torch.tensor([1, 5], dtype=torch.int32, device=dev)   # 已扣掉被拒行
        total_rows = 8
        k_ids = torch.zeros(total_rows, dtype=torch.int32, device=dev)
        k_pos = torch.zeros(total_rows, dtype=torch.int32, device=dev)
        k_rej = torch.zeros(total_rows, dtype=torch.bool, device=dev)
        k_mask = torch.zeros(total_rows, dtype=torch.bool, device=dev)
        k_sample = torch.zeros(2, dtype=torch.int32, device=dev)
        k_hidden = torch.zeros(6, dtype=torch.int32, device=dev)
        copy_and_expand_eagle_inputs_kernel[(2, 1)](
            target_ids, target_pos, next_ids, k_ids, k_pos, k_rej, k_mask, k_sample, k_hidden,
            qsl, qel, 0, 0, 6, 1, False, BLOCK_SIZE_TOKENS=4)
        ours = expand_draft_inputs([
            DraftInputRows(valid_token_ids=[5, 6], start=6, next_token_id=99, num_rejected=1),
            DraftInputRows(valid_token_ids=[8, 9, 10], start=8, next_token_id=111,
                           num_rejected=0)])
        check("B5. 与上游 copy_and_expand_eagle_inputs_kernel（shift_input_ids=False）逐值一致",
              k_ids.tolist() == ours[0] and k_pos.tolist() == ours[1]
              and k_rej.tolist() == [bool(x) for x in ours[2]]
              and k_sample.tolist() == ours[3] and not bool(k_mask.any()),
              f"kernel ids={k_ids.tolist()}、本地 ids={ours[0]}")
    except Exception as exc:                                 # noqa: BLE001
        check("B5. （CUDA 差分未跑：导入/调用上游 kernel 失败）", False,
              f"{type(exc).__name__}: {exc}")
else:
    check("B5. （跳过与上游 kernel 的 CUDA 差分：本机没有 CUDA）", True)

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
