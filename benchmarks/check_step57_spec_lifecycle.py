"""57E 验收（对应需求里的 `test_spec_lifecycle.py`）：投机的状态与时序。

用 **ngram** 跑（确定性、不需要 draft 模型），专门查 199 点名的几条状态边界：

  1. **时序**：轮 t 验证时提草稿 → 轮 t+1 才采用。提议**不改本轮计划**（199 §4 明确
     要删掉旧代码里"提议者直接改 Scheduler 计划"的通道）；
  2. **K 裁剪**：预算不够时只采用前几枚，剩下的丢掉（不能留着下轮拿旧草稿对新历史）；
  3. **进度回退**：被拒的草稿要从 `num_computed_tokens` 里退回来——排的是 K+1 行，
     有效的只有"接受的 a 枚 + 最后那个 token"；
  4. **抢占清空草稿**（199 §10.6）：被抢的请求草稿要丢掉，恢复后不拿旧 q 验证新历史；
  5. **greedy 下投机与非投机输出逐 token 一致**（199 §10.7）——接受的就是贪心本来会给的；
  6. 槽位不越界：草稿也要有 KV 槽位。
"""

import json
import sys

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

import torch

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,
                    SchedulerConfig, SpeculativeConfig, UniProcExecutor, VllmConfig, Worker)

FAIL = []
from minivllm.testing.tiny_models import tiny_qwen3_dir   # 测试模型现场生成（仓库不再放 fixtures）
TINY_DIR = tiny_qwen3_dir("tiny_gqa")
TINY_CONFIG = json.load(open(f"{TINY_DIR}/config.json"))


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def first_line(error):
    return error.splitlines()[0] if error else "没有报错"


# 59 关起：投机**验证**走 Triton 内核（上游同样只有 GPU 路径），所以开投机的脚本要上 GPU。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build(spec_tokens=3, method="ngram", blocks=16, budget=16, model=TINY_DIR,
          hf_config=TINY_CONFIG, draft_config=None, seqs=2, block_size=4):
    config = VllmConfig(
        model_config=ModelConfig(model=model, dtype="float32", max_model_len=64,
                                 hf_config=hf_config),
        cache_config=CacheConfig(block_size=block_size, num_gpu_blocks=blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=seqs,
                                         max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=DEVICE),
        speculative_config=None if spec_tokens == 0 else SpeculativeConfig(
            method=method, num_speculative_tokens=spec_tokens,
            # 60 关起：ngram 的匹配窗口默认跟上游一样是 5/5（短 prompt 找不到 5-gram 重复），
            # 本脚本要考的是"提 → 采用"的时序，所以显式给个小窗口（57 时代本机用的是 1..3）
            prompt_lookup_min=1, prompt_lookup_max=3,
            draft_model_config=draft_config))
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    return engine, engine.engine_core.engine_core.scheduler


class Recorder:
    """记录每轮的包 + 提议器这一轮提了什么（时序对照用）。"""

    def __init__(self, scheduler, runner):
        self.packets = []
        self.proposals = []
        self._schedule = scheduler.schedule
        scheduler.schedule = self.schedule
        if runner.proposer is not None:
            # 60 关：ngram 提议者的协议入口是 `propose_drafts`（`propose` 现在是上游同签名的
            # 批量匹配入口，返回的是 list[list[int]]）。draft_model 提议者仍是 `propose`。
            self._entry = ("propose_drafts" if hasattr(runner.proposer, "propose_drafts")
                           else "propose")
            self._propose = getattr(runner.proposer, self._entry)
            setattr(runner.proposer, self._entry, self.propose)

    def schedule(self):
        packet = self._schedule()
        self.packets.append((dict(packet.scheduled_spec_decode_tokens),
                             dict(packet.num_scheduled_tokens)))
        return packet

    def propose(self, *args, **kwargs):
        drafts = self._propose(*args, **kwargs)
        self.proposals.append({req_id: list(ids) for req_id, ids in
                               zip(drafts.req_ids, drafts.draft_token_ids)})
        return drafts


PROMPT = [1, 2, 3, 4, 1, 2, 3, 4]          # 有重复片段，ngram 找得到草稿


# ------------------------------------------------ 1. 时序：轮 t 提 → 轮 t+1 采用

engine, scheduler = build(spec_tokens=3)
runner = engine.engine_core.engine_core.model_executor.driver_worker.model_runner
recorder = Recorder(scheduler, runner)
engine.add_request("r", PROMPT, SamplingParams(max_tokens=6, temperature=0.0, eos_token_id=999))
rounds = 0
while engine.has_unfinished_requests():
    engine.step()
    rounds += 1
check("1. （用例前提）跑了多轮、提议器确实提过草稿",
      rounds >= 4 and any(recorder.proposals), f"{rounds} 轮、提议 {recorder.proposals[:2]}")

first_proposal_round = next(index for index, proposal in enumerate(recorder.proposals)
                            if proposal.get("r"))
check("1. 提议**不改本轮计划**：提草稿的那一轮，包里没有它的草稿",
      not recorder.packets[first_proposal_round][0].get("r"),
      f"第 {first_proposal_round + 1} 轮的草稿={recorder.packets[first_proposal_round][0]}")
next_round_spec = recorder.packets[first_proposal_round + 1][0].get("r")
check("1. 下一轮才采用：采用的正是上一轮提的那几枚（前缀）",
      bool(next_round_spec)
      and recorder.proposals[first_proposal_round]["r"][:len(next_round_spec)] == next_round_spec,
      f"上轮提 {recorder.proposals[first_proposal_round]['r']}、本轮采用 {next_round_spec}")
request_after_resume = scheduler.requests.get("r")
check("1. 采用之后请求上的候选被清空（不会拿旧草稿对下一轮的新历史）",
      request_after_resume is None                      # 已经跑完并收尾
      or request_after_resume.spec_token_ids != recorder.proposals[first_proposal_round]["r"],
      f"候选={getattr(request_after_resume, 'spec_token_ids', None)}")
mismatched = [(index, spec, num) for index, (spec, num) in enumerate(recorder.packets)
              for req_id in spec
              if num.get(req_id) != len(spec[req_id]) + 1]
check("1. 带草稿的请求，排的行数恰好是 K+1（多一行都不行：草稿是 query 的尾部）",
      not mismatched, str(mismatched))
engine.shutdown()

# ------------------------------------------------ 2. 进度回退与 K 裁剪

engine, scheduler = build(spec_tokens=3)
runner = engine.engine_core.engine_core.model_executor.driver_worker.model_runner
engine.add_request("r", PROMPT, SamplingParams(max_tokens=6, temperature=0.0, eos_token_id=999))
progress_ok = True
while engine.has_unfinished_requests():
    engine.step()
    request = scheduler.requests.get("r")
    if request is not None and request.num_computed_tokens != request.num_tokens - 1:
        progress_ok = False
        break
check("2. 每轮结束时进度都恰好是 num_tokens - 1（被拒的草稿退回来了；"
      "没有这条，下一轮会从错误的位置续算）",
      progress_ok, f"最后状态={getattr(scheduler.requests.get('r'), 'num_computed_tokens', None)}")
engine.shutdown()

# 预算刚好只够 1 行：K=3 的草稿必须被截成 0（只剩 b 那一行）
engine, scheduler = build(spec_tokens=3, budget=1)
runner = engine.engine_core.engine_core.model_executor.driver_worker.model_runner
recorder = Recorder(scheduler, runner)
engine.add_request("r", PROMPT, SamplingParams(max_tokens=3, temperature=0.0, eos_token_id=999))
while engine.has_unfinished_requests():
    engine.step()
check("2. K 裁剪：预算只够 1 行时，草稿一枚都不发（而不是排 K+1 行超预算）",
      all(not spec for spec, _num in recorder.packets[1:]),
      str([spec for spec, _num in recorder.packets]))
check("2. 提议照常发生（截的是**采用**，不是提议）",
      any(proposal.get("r") for proposal in recorder.proposals),
      str(recorder.proposals[:3]))
engine.shutdown()

# ------------------------------------------------ 3. 抢占清空草稿

engine, scheduler = build(spec_tokens=3, blocks=3, seqs=2, budget=16)
runner = engine.engine_core.engine_core.model_executor.driver_worker.model_runner
engine.add_request("r1", PROMPT, SamplingParams(max_tokens=8, temperature=0.0, eos_token_id=999))
engine.add_request("r2", PROMPT, SamplingParams(max_tokens=8, temperature=0.0, eos_token_id=999))
preempted_with_drafts = False
while engine.has_unfinished_requests() and scheduler.num_preemptions == 0:
    engine.step()
if scheduler.num_preemptions > 0:
    preempted = [request for request in scheduler.waiting.request_ids()]
    preempted_with_drafts = all(
        scheduler.requests[req_id].spec_token_ids == [] for req_id in preempted)
check("3. 抢占之后草稿被清空（恢复后不拿旧草稿/旧 q 验证新历史）",
      preempted_with_drafts, f"抢占 {scheduler.num_preemptions} 次、等待={preempted}")
engine.shutdown()

# ------------------------------------------------ 4. greedy 下投机与非投机输出一致

def run_greedy(spec_tokens):
    engine, scheduler = build(spec_tokens=spec_tokens)
    engine.add_request("r", PROMPT, SamplingParams(max_tokens=8, temperature=0.0,
                                                   eos_token_id=999))
    tokens = []
    while engine.has_unfinished_requests():
        for output in engine.step():
            tokens = list(output.token_ids)
    engine.shutdown()
    return tokens


without_spec = run_greedy(0)
with_spec = run_greedy(3)
check("4. greedy 下开/关投机的输出**逐 token 一致**（接受的就该是贪心本来会给的）",
      without_spec == with_spec, f"不投机={without_spec}；投机={with_spec}")

# ------------------------------------------------ 5. 草稿也有 KV 槽位（不越界）

engine, scheduler = build(spec_tokens=3)
runner = engine.engine_core.engine_core.model_executor.driver_worker.model_runner
observed = []
original_prepare = runner._prepare_inputs


def spy_prepare(scheduler_output):
    inputs = original_prepare(scheduler_output)
    # 本轮真正要写的最大位置 vs 这一刻块表覆盖的槽位数
    for row in range(runner.input_batch.num_reqs):
        covered = runner.input_batch.block_table.num_blocks(row) * 4
        observed.append(int(inputs.seq_lens[row]) - covered)
    return inputs


runner._prepare_inputs = spy_prepare
engine.add_request("r", PROMPT, SamplingParams(max_tokens=6, temperature=0.0, eos_token_id=999))
while engine.has_unfinished_requests():
    engine.step()
runner._prepare_inputs = original_prepare
check("5. 草稿的位置都被块表覆盖（不需要额外的 lookahead 预留——它们在本轮调度范围之内）",
      observed and max(observed) <= 0, f"最紧的一次：超出 {max(observed)} 个槽位")
check("5. 整段投机跑下来没触发过块表越界（BlockTable 自己会报）", True)
engine.shutdown()

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
