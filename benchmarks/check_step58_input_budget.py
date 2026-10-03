"""58 验收 A（需求 §8.A）：**双预算** —— target 的 token 预算 + draft 的输入预算。

为什么需要第二份预算：draft 第一遍吃的是"target 本轮 query 的有效行 + 1 行扩容行"，
所以每条**被调度的请求**在输入工作区里多占 `draft_slots` 行（普通 draft = 1，ngram = 0）。
只按 target 的 token 预算排请求，工作区就会放不下（需求 §2/§3 的最小例子）。

本脚本是**纯 Scheduler 单测**（不跑模型、不碰 GPU）：直接看 `Scheduler` 的账——
本轮排了谁几个 token、两份预算的余额、被抢占时返还了多少。

    1. M=8、两条请求各要 4 行：排不出 target 8 / draft 10，计划必须同时满足两份预算
    2. running 与 waiting 用同一个口径；ngram 的额外槽是 0
    3. 本轮已排入计划的 victim 被抢占：两份预算都返还、草稿计划同步删除
    4. 极小预算的非法配置初始化失败（不空转）；普通非投机行为不变
"""

import sys

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from minivllm import (CacheConfig, ModelConfig, Request, RequestStatus, SamplingParams,
                      SchedulerConfig, SpeculativeConfig, VllmConfig)
from minivllm.core.kv_cache_manager import KVCacheManager
from minivllm.core.sched.scheduler import Scheduler

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def spec_config(method="draft_model", k=3):
    if method is None:
        return None
    model = ModelConfig(model="dummy", max_model_len=64)
    return SpeculativeConfig(method=method, num_speculative_tokens=k,
                             draft_model_config=model if method == "draft_model" else None)


def make_scheduler(*, max_num_seqs=2, max_num_batched_tokens=8, num_gpu_blocks=8,
                   block_size=4, max_model_len=64, policy="fcfs", spec=None):
    scheduler_config = SchedulerConfig(max_num_seqs=max_num_seqs,
                                       max_num_batched_tokens=max_num_batched_tokens,
                                       policy=policy)
    kv_cache_manager = KVCacheManager(
        CacheConfig(block_size=block_size, num_gpu_blocks=num_gpu_blocks),
        max_model_len=max_model_len)
    return Scheduler(scheduler_config, kv_cache_manager, max_model_len=max_model_len,
                     speculative_config=spec)


def add(scheduler, request_id, prompt_len, priority=0, max_tokens=8):
    request = Request(request_id, list(range(prompt_len)),
                      SamplingParams(max_tokens=max_tokens, eos_token_id=999),
                      arrival_time=1.0, priority=priority)
    scheduler.add_request(request)
    return request


# ---------------------------------------------------------------- 1. 两份预算一起约束
scheduler = make_scheduler(max_num_batched_tokens=8, spec=spec_config())
assert scheduler.draft_slots == 1 and scheduler.max_num_batched_tokens == 8
for req_id in ("A", "B"):
    add(scheduler, req_id, prompt_len=4)
out = scheduler.schedule()
plan = dict(out.num_scheduled_tokens)
target_total = sum(plan.values())
check("A1. M=8、两条请求各要 4 行：排不出 target 8 / draft 10（输入预算真的在拦）",
      target_total == 6 and plan == {"A": 4, "B": 2},
      f"计划={plan}；target 合计={target_total}（若无输入预算会是 8）")
check("A1b. 计划同时满足两份预算：Σtarget ≤ token 预算、Σ(target+draft_slots) ≤ 工作区",
      target_total <= 8 and target_total + scheduler.draft_slots * len(plan) <= 8
      and scheduler.last_token_budget == 8 - target_total
      and scheduler.last_input_budget == 8 - target_total - len(plan),
      f"token 余额={scheduler.last_token_budget}、input 余额={scheduler.last_input_budget}")

# ---------------------------------------------------------------- 2. 同一口径 + ngram 0 槽
# 同一条 waiting 循环里：先接纳的 A 把 8 行输入预算吃成 0（7 行 target + 1 行额外），
# 轮到 B 时 `input_budget <= draft_slots`，于是 B 这一轮不进来——不是只看 token 预算
scheduler = make_scheduler(max_num_batched_tokens=8, spec=spec_config())
add(scheduler, "A", prompt_len=8)
add(scheduler, "B", prompt_len=4)
out = scheduler.schedule()
plan = dict(out.num_scheduled_tokens)
check("A2. waiting 与 running 用同一个 input_budget 口径（B 因为只剩 0 行而没被接纳）",
      plan == {"A": 7} and scheduler.last_input_budget == 0
      and scheduler.last_token_budget == 1,
      f"计划={plan}、token 余额={scheduler.last_token_budget}、"
      f"input 余额={scheduler.last_input_budget}")

scheduler = make_scheduler(max_num_batched_tokens=8, spec=spec_config(method="ngram", k=3))
assert scheduler.draft_slots == 0
for req_id in ("A", "B"):
    add(scheduler, req_id, prompt_len=4)
out = scheduler.schedule()
plan = dict(out.num_scheduled_tokens)
check("A2b. ngram 不跑模型、不写 KV → max_num_new_slots_for_drafting=0，不占额外输入",
      plan == {"A": 4, "B": 4} and scheduler.last_input_budget == 0,
      f"计划={plan}、input 余额={scheduler.last_input_budget}")

# 输入预算连"额外行"都放不下时：本轮一条都排不了（上游 running/waiting 两条循环的守卫）
scheduler = make_scheduler(max_num_batched_tokens=2, spec=spec_config())
add(scheduler, "A", prompt_len=8)
out = scheduler.schedule()
check("A2c. 输入预算 <= draft_slots 时守卫生效：本轮不排（target 剩 2 行也只排 1 行）",
      out.num_scheduled_tokens == {"A": 1} and scheduler.last_input_budget == 0,
      f"计划={out.num_scheduled_tokens}")

# ---------------------------------------------------------------- 3. 抢占返还两份预算
# 要走到 `input_budget += restored + draft_slots` 这条分支，victim 必须**已经在本轮计划里**，
# 所以它得排在失败请求前面（running 顺序 = 接纳顺序），而且是优先级最低的那个：
#   round1: A 单独进场（prompt 2，占 2 块）
#   round2: A 带着 3 枚待验证草稿再排一次（n=3），B（prompt 4）也进来 —— running=[A,B]，池子占满
#   round3: A 再带草稿（n=3，已排入计划）；B 带草稿后要长到第 3 个块 → 分配失败 →
#           victim = 优先级最低的 A（把它的 priority 调大即可，priority 是调度期读的字段）
#           → A 的记录被撤销、两份预算返还 → B 重试成功
scheduler = make_scheduler(max_num_batched_tokens=8, num_gpu_blocks=4,
                           spec=spec_config(), policy="priority")
a = add(scheduler, "A", prompt_len=2)
scheduler.schedule()
a.spec_token_ids = [7, 7, 7]
b = add(scheduler, "B", prompt_len=4)
scheduler.schedule()
a.spec_token_ids = [7, 7, 7]
b.spec_token_ids = [7, 7, 7]
a.priority = 5                                        # A 变成优先级最低的 victim
out = scheduler.schedule()
plan = dict(out.num_scheduled_tokens)
check("A3. 块不够时抢占**本轮已排入计划**的 victim：撤销它的记录、两份预算都返还",
      plan == {"B": 4} and a.status == RequestStatus.PREEMPTED
      and scheduler.last_token_budget == 8 - 4
      and scheduler.last_input_budget == 8 - 4 - 1,
      f"计划={plan}、A 状态={a.status}、token 余额={scheduler.last_token_budget}、"
      f"input 余额={scheduler.last_input_budget}")
check("A3b. 被抢占的请求不会留在 scheduled_spec_decode_tokens 里（草稿计划同步删除）",
      set(out.scheduled_spec_decode_tokens) == {"B"},
      f"草稿计划={dict(out.scheduled_spec_decode_tokens)}")

# ---------------------------------------------------------------- 4. 非法配置 + 非投机不变
try:
    make_scheduler(max_num_batched_tokens=1, spec=spec_config())
    error = None
except ValueError as exc:
    error = str(exc)
check("A4. 工作区装不下「1 个 target token + 1 行 draft 额外输入」→ 初始化直接拒绝",
      error is not None and "max_num_batched_tokens" in error, error or "没有报错")

try:
    make_scheduler(max_num_batched_tokens=2, spec=None)     # 非投机：2 行够
    error = None
except ValueError as exc:
    error = str(exc)
check("A4b. 非投机没有额外输入槽：同样的小预算不该被这条校验拦住",
      error is None and make_scheduler(max_num_batched_tokens=2, spec=None).draft_slots == 0,
      error or "正常")

scheduler = make_scheduler(max_num_batched_tokens=8, spec=None)
for req_id in ("A", "B"):
    add(scheduler, req_id, prompt_len=4)
out = scheduler.schedule()
check("A4c. 非投机的行为不变：没有额外槽位，输入预算就等于 target 行数（8 行全排满）",
      out.num_scheduled_tokens == {"A": 4, "B": 4} and scheduler.last_input_budget == 0
      and scheduler.last_token_budget == 0,
      f"计划={out.num_scheduled_tokens}、input 余额={scheduler.last_input_budget}")

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
