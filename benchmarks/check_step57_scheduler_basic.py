"""57A 验收（对应需求里的 `test_scheduler_basic.py`）：统一预算调度与结果提交。

查的是"调度器自己的账"：

  1. **统一预算**：`num_new_tokens = min(num_tokens_with_spec - num_computed_tokens, 预算, ...)`；
  2. **先 running 后 waiting**；预算被前面的吃光时后面的**等待**（不强求"每条都分一点"）；
  3. `max_num_seqs` 与**块容量**两个上限：块不够时本轮不调度它，且不留半个成功的块表；
  4. 进度快照顺序：包里的 `num_computed_tokens` 是**调度前**的值，推进发生在打包之后；
  5. `all_token_ids` 只在这条请求"上一轮没被调度"时才带上（不是每轮复制全部历史）；
  6. 结果按 ID 更新（含"本轮没被执行的请求"）；
  7. 结束：`_free_request` 还块 + 登记清理消息，**只送一次**；
  8. `abort` 路径：running / waiting 都能摘掉；
  9. 排不出任何 token 时不无限空转（连续两轮就明确报错）。
"""

import sys

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from step57 import (CacheConfig, FinishReason, ModelConfig, Request, RequestStatus,
                    SamplingParams, SchedulerConfig, VllmConfig)
from step57.core.kv_cache_manager import KVCacheManager
from step57.core.sched.scheduler import Scheduler

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def make_scheduler(max_num_seqs=4, max_num_batched_tokens=16, num_gpu_blocks=8,
                   block_size=4, max_model_len=64, policy="fcfs"):
    scheduler_config = SchedulerConfig(max_num_seqs=max_num_seqs,
                                       max_num_batched_tokens=max_num_batched_tokens,
                                       policy=policy)
    cache_config = CacheConfig(block_size=block_size, num_gpu_blocks=num_gpu_blocks)
    kv_cache_manager = KVCacheManager(cache_config)
    return Scheduler(scheduler_config, kv_cache_manager, max_model_len=max_model_len)


def add(scheduler, request_id, prompt_len, **sampling):
    params = SamplingParams(**{"max_tokens": 8, "eos_token_id": 999, **sampling})
    request = Request(request_id, list(range(prompt_len)), params, arrival_time=1.0)
    scheduler.add_request(request)
    return request


def model_output(scheduler_output, tokens_by_req=None):
    """按包里的请求顺序造一份 ModelRunnerOutput（不经过 FakeRunner，专测调度器自己的账）。"""
    from step57.outputs import ModelRunnerOutput

    tokens_by_req = tokens_by_req or {}
    req_ids = list(scheduler_output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
        sampled_token_ids=[list(tokens_by_req.get(req_id, [])) for req_id in req_ids],
    )


# ------------------------------------------------ 1. 统一预算公式

scheduler = make_scheduler(max_num_batched_tokens=2, max_num_seqs=2)
add(scheduler, "r1", 3)                                  # prompt [0,1,2]，预算 2
output = scheduler.schedule()
check("统一预算：第一轮只排 2 个 token（预算限制），差值为 3",
      output.num_scheduled_tokens == {"r1": 2} and output.total_num_scheduled_tokens == 2,
      str(output.num_scheduled_tokens))
check("统一预算：包里的进度是**调度前**的旧值 0",
      output.scheduled_cached_reqs.num_computed_tokens == []      # 首次执行走 new_reqs
      and output.scheduled_new_reqs[0].num_computed_tokens == 0)
check("统一预算：调度之后才推进（2），此时还没算完 prompt（3）",
      scheduler.requests["r1"].num_computed_tokens == 2
      and scheduler.requests["r1"].is_prefill_chunk)
first_blocks = output.scheduled_new_reqs[0].block_ids
check("统一预算：新请求发出的是 NewRequestData（带 prompt 与块表）",
      [data.req_id for data in output.scheduled_new_reqs] == ["r1"]
      and list(output.scheduled_new_reqs[0].prompt_token_ids) == [0, 1, 2]
      and len(first_blocks) == 1 and len(first_blocks[0]) == 1
      and 0 <= first_blocks[0][0] < 8,
      f"块表={first_blocks}（按 group 分组；具体块号取决于分配顺序）")

output = scheduler.schedule()                            # 第二轮：差值 1、预算 2 → 排 1
check("统一预算：第二轮排剩下的 1 个 token，差值归零（不再是 prefill 块）",
      output.num_scheduled_tokens == {"r1": 1}
      and not scheduler.requests["r1"].is_prefill_chunk
      and scheduler.requests["r1"].num_tokens == 3
      and scheduler.requests["r1"].num_computed_tokens == 3)
check("统一预算：这一轮走 CachedRequestData（只发增量，不再发 prompt）",
      output.scheduled_new_reqs == []
      and output.scheduled_cached_reqs.req_ids == ["r1"]
      and output.scheduled_cached_reqs.new_block_ids == [None],   # 复用已有块 → 没有新增块
      f"new_block_ids={output.scheduled_cached_reqs.new_block_ids}")

# ------------------------------------------------ 2. 预算被吃光 → 后面等待（196 §9.2）

scheduler = make_scheduler(max_num_batched_tokens=3, max_num_seqs=4)
add(scheduler, "first", 6)
add(scheduler, "second", 6)
output = scheduler.schedule()
check("预算优先给先来的：第一条吃满 3 个 token，第二条这一轮一个都不排（留在 waiting）",
      output.num_scheduled_tokens == {"first": 3}
      and [r.request_id for r in scheduler.running] == ["first"]
      and [r.request_id for r in scheduler.waiting] == ["second"],
      f"排了 {output.num_scheduled_tokens}；running={[r.request_id for r in scheduler.running]}")

# ------------------------------------------------ 3. 两个上限：max_num_seqs 与块容量

scheduler = make_scheduler(max_num_seqs=1, max_num_batched_tokens=16)
add(scheduler, "a", 2)
add(scheduler, "b", 2)
output = scheduler.schedule()
check("max_num_seqs 是硬上限：并发满了就不再接纳 waiting",
      output.num_scheduled_tokens == {"a": 2} and [r.request_id for r in scheduler.waiting] == ["b"],
      str(output.num_scheduled_tokens))

# 块容量不足：块大小 4、总块数 1 → 一条 8 token 的 prompt 需要 2 块，第一条占住后第二条排不了
scheduler = make_scheduler(num_gpu_blocks=2, block_size=4, max_num_batched_tokens=8,
                           max_num_seqs=2)
add(scheduler, "big", 8)          # 需要 2 块
add(scheduler, "small", 4)        # 需要 1 块
output = scheduler.schedule()
check("块容量不足时：排得下的先排（big 拿满 2 块），排不下的留在 waiting",
      output.num_scheduled_tokens == {"big": 8}
      and list(scheduler.waiting)[0].request_id == "small"
      and scheduler.kv_cache_manager.num_free_blocks() == 0,
      f"排了 {output.num_scheduled_tokens}、空闲块 {scheduler.kv_cache_manager.num_free_blocks()}")

# 分配失败**不留半个块表**：先让一条请求申请失败，再看它的块表
scheduler = make_scheduler(num_gpu_blocks=1, block_size=4, max_num_batched_tokens=8)
request = add(scheduler, "r", 8)
before_free = scheduler.kv_cache_manager.num_free_blocks()
scheduler.schedule()
check("分配失败原子性：申请不到的请求，块表仍然为空（没有半个成功）",
      scheduler.kv_cache_manager.get_blocks("r").blocks[0] == []
      and scheduler.kv_cache_manager.num_free_blocks() == before_free
      and request.num_computed_tokens == 0,
      f"块表={scheduler.kv_cache_manager.get_blocks('r').blocks[0]}、"
      f"空闲块={scheduler.kv_cache_manager.num_free_blocks()}")

# ------------------------------------------------ 4. all_token_ids 只在"上轮没被调度"时带上

scheduler = make_scheduler(max_num_batched_tokens=1, max_num_seqs=1)
add(scheduler, "r1", 3)
scheduler.schedule()                       # 第 1 轮：新请求，走 NewRequestData
output = scheduler.schedule()              # 第 2 轮：上一轮调度过 → 不带完整历史
check("all_token_ids 的发送规则：上一轮调度过的请求不带（省掉每轮复制整段历史）",
      output.scheduled_cached_reqs.all_token_ids == {},
      str(output.scheduled_cached_reqs.all_token_ids))

scheduler = make_scheduler(max_num_batched_tokens=4, max_num_seqs=2)
add(scheduler, "r1", 2)
add(scheduler, "r2", 2)
scheduler.schedule()                       # 只排得下 r1（预算 4，r1 拿 2，r2 也拿 2）
check("（用例前提）两条都排上了（并发上限 2 允许两条）",
      [r.request_id for r in scheduler.running] == ["r1", "r2"],
      f"running={[r.request_id for r in scheduler.running]}")

# ------------------------------------------------ 5. 结果按 ID 更新（含本轮未被执行的请求）

scheduler = make_scheduler(max_num_batched_tokens=2, max_num_seqs=2)
add(scheduler, "r1", 2)                    # 一轮就能算完
add(scheduler, "r2", 4)                    # 需要两轮
output = scheduler.schedule()
check("（用例前提）本轮只排了 r1", output.num_scheduled_tokens == {"r1": 2})
core_outputs = scheduler.update_from_output(output, model_output(output, {"r1": [101]}))
check("结果按 ID 更新：本轮排到的请求提交了 token",
      [o.request_id for o in core_outputs.outputs] == ["r1"]
      and list(scheduler.requests["r1"].output_token_ids) == [101])
check("本轮没被执行的请求不受影响（进度与输出都没动）",
      scheduler.requests["r2"].num_computed_tokens == 0
      and scheduler.requests["r2"].num_output_tokens == 0)

# ------------------------------------------------ 6. 结束：还块 + 清理消息只送一次

scheduler = make_scheduler(max_num_batched_tokens=4, max_num_seqs=2)
add(scheduler, "r1", 2, max_tokens=1)
output = scheduler.schedule()
blocks_used_before = scheduler.kv_cache_manager.num_allocated_blocks
core_outputs = scheduler.update_from_output(output, model_output(output, {"r1": [7]}))
check("结束：输出带 finish_reason，块全部归还",
      core_outputs.outputs[0].finish_reason == FinishReason.LENGTH
      and blocks_used_before == 1
      and scheduler.kv_cache_manager.num_allocated_blocks == 0,
      f"结束前占用 {blocks_used_before} 块、结束后 "
      f"{scheduler.kv_cache_manager.num_allocated_blocks} 块")
check("结束：从活动索引里摘掉，但清理消息留在 finished_req_ids 里等下一轮送",
      "r1" not in scheduler.requests and scheduler.finished_req_ids == {"r1"}
      and scheduler.has_finished_requests() and not scheduler.has_unfinished_requests())
check("结束：has_requests() 仍为真（清理消息没送出去之前引擎不能停）",
      scheduler.has_requests())

output = scheduler.schedule()              # 清理轮：0 token，但带走 finished_req_ids
check("清理轮：包是空的（0 token、没有请求），但带上了要清理的 ID",
      output.total_num_scheduled_tokens == 0 and output.num_scheduled_tokens == {}
      and output.finished_req_ids == {"r1"})
check("清理轮之后：finished_req_ids 已重置（同一条 ID 不会再送第二遍）",
      scheduler.finished_req_ids == set() and not scheduler.has_requests())

# ------------------------------------------------ 7. abort

scheduler = make_scheduler(max_num_batched_tokens=2, max_num_seqs=2)
add(scheduler, "run1", 2)
add(scheduler, "wait1", 2)
output = scheduler.schedule()
check("（用例前提）一条 running、一条 waiting",
      [r.request_id for r in scheduler.running] == ["run1"]
      and [r.request_id for r in scheduler.waiting] == ["wait1"])
finished = scheduler.finish_requests(["run1", "wait1"], RequestStatus.FINISHED_ABORTED)
check("abort：running 与 waiting 都能摘掉，状态是 FINISHED_ABORTED，块归还",
      {r.request_id for r in finished} == {"run1", "wait1"}
      and all(r.status == RequestStatus.FINISHED_ABORTED for r in finished)
      and not scheduler.running and not scheduler.waiting
      and scheduler.kv_cache_manager.num_allocated_blocks == 0,
      f"块占用={scheduler.kv_cache_manager.num_allocated_blocks}")
check("abort：结束原因映射为 ABORT",
      all(r.get_finished_reason() == FinishReason.ABORT for r in finished))
check("abort：已经结束的 ID 再 abort 一次不报错、也不重复结束",
      scheduler.finish_requests(["run1"], RequestStatus.FINISHED_ABORTED) == [])

# ------------------------------------------------ 8. 不无限空转

scheduler = make_scheduler(max_num_batched_tokens=4, max_model_len=4)
add(scheduler, "r1", 4, max_tokens=4)     # prompt 4 + 要生成 4 > max_model_len 4 → 排不出
try:
    for _ in range(4):
        scheduler.schedule()
    error = None
except RuntimeError as exc:
    error = str(exc)
check("排不出任何 token 且仍有未完成请求 → 连续两轮后明确报错（不无限空转）",
      error is not None and "排不出" in error and "空闲块" in error,
      (error or "").splitlines()[0] if error else "没有报错")

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
