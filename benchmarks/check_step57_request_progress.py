"""57A 验收（对应需求里的 `test_request_progress.py`）：`Request` 的进度语义与停止判定。

这一层回答"请求进度谁维护、谁能改"：

  * `prompt_token_ids` 在提交时**复制**（用户之后改自己那份无影响）；
  * `output_token_ids` / `all_token_ids` 是**只读视图**，唯一写入点是 `append_output_token_ids()`；
  * `num_tokens` / `num_tokens_with_spec` / `num_output_tokens` 是派生量（不再手动加减）；
  * `num_computed_tokens` 是**计划进度**（Scheduler 写），与"显存里写到哪"不是一回事；
  * `status` 与 `stop_reason` 分开；`check_stop` 六个分支各有一条用例。
"""

import sys

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from minivllm import (EngineCoreRequest, FinishReason, Request, RequestStatus, SamplingParams)
from minivllm.core.sched.request_queue import (FCFSRequestQueue, PriorityRequestQueue,
                                             create_request_queue)
from minivllm.core.sched.utils import check_stop

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def make(request_id="r1", prompt=(1, 2, 3), **sampling):
    params = SamplingParams(**{"max_tokens": 4, "eos_token_id": 99, **sampling})
    return Request(request_id, list(prompt), params, arrival_time=1.0)


# ------------------------------------------------ 1. prompt 复制与只读视图

prompt = [10, 11, 12]
request = make(prompt=prompt)
prompt.append(999)                      # 用户提交之后继续改自己那份
check("prompt 在提交时被复制：用户改自己的列表不影响请求",
      list(request.prompt_token_ids) == [10, 11, 12] and request.num_prompt_tokens == 3,
      str(request.prompt_token_ids))

for name in ("append", "extend", "insert", "pop"):
    try:
        getattr(request.output_token_ids, name)(1)
        readonly = False
    except AttributeError:
        readonly = True
    if not readonly:
        break
check("output_token_ids 是只读视图（没有 append / extend / insert / pop）", readonly)
check("只读视图仍然像 list：索引、切片、len、迭代、与 list 相加都可用",
      request.all_token_ids[0] == 10 and list(request.all_token_ids[1:]) == [11, 12]
      and len(request.all_token_ids) == 3 and list(request.all_token_ids) == [10, 11, 12]
      and request.all_token_ids + [7] == [10, 11, 12, 7]
      and request.all_token_ids == [10, 11, 12])

# ------------------------------------------------ 2. 唯一写入点同步两份列表

request.append_output_token_ids(20)
check("append_output_token_ids：输出列表与完整历史同时更新",
      list(request.output_token_ids) == [20] and list(request.all_token_ids) == [10, 11, 12, 20])
check("派生量跟着走：num_tokens / num_output_tokens",
      request.num_tokens == 4 and request.num_output_tokens == 1)
request.append_output_token_ids([21, 22])
check("append_output_token_ids 也接受整批 token",
      list(request.output_token_ids) == [20, 21, 22] and request.num_output_tokens == 3)
check("num_tokens_with_spec 在没草稿时等于 num_tokens",
      request.num_tokens_with_spec == request.num_tokens)
request.spec_token_ids = [31, 32]
check("num_tokens_with_spec = 已确定 token + 草稿（草稿**尚未提交**，不在 all_token_ids 里）",
      request.num_tokens_with_spec == request.num_tokens + 2
      and 31 not in request.all_token_ids)

# ------------------------------------------------ 3. 分配谁改：num_computed_tokens 的语义

request = make()
check("新建请求：num_computed_tokens 从 0 开始、状态是 WAITING",
      request.num_computed_tokens == 0 and request.status == RequestStatus.WAITING)
request.num_computed_tokens = 2          # 这是 Scheduler 的动作（这里手动模拟）
check("num_computed_tokens 只表示'已安排计算'的进度，与输出数无关",
      request.num_computed_tokens == 2 and request.num_output_tokens == 0)

# ------------------------------------------------ 4. 状态与结束原因

request = make()
check("is_finished：WAITING/RUNNING/PREEMPTED 都不算结束",
      not any(RequestStatus.is_finished(status) for status in
              (RequestStatus.WAITING, RequestStatus.RUNNING, RequestStatus.PREEMPTED)))
check("is_finished：FINISHED_* 都算结束（约定是'PREEMPTED 之后都算'）",
      all(RequestStatus.is_finished(status) for status in
          (RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED,
           RequestStatus.FINISHED_ABORTED, RequestStatus.FINISHED_ERROR)))
request.status = RequestStatus.FINISHED_STOPPED
check("状态 → 结束原因 的映射",
      request.is_finished() and request.get_finished_reason() == FinishReason.STOP)

# ------------------------------------------------ 5. 排序：priority → 到达时间 → ID

a = Request("a", [1], SamplingParams(max_tokens=1), arrival_time=5.0, priority=1)
b = Request("b", [1], SamplingParams(max_tokens=1), arrival_time=1.0, priority=1)
c = Request("c", [1], SamplingParams(max_tokens=1), arrival_time=9.0, priority=0)
check("排序：数值小的 priority 更优先",
      sorted([a, b, c])[0].request_id == "c", str([r.request_id for r in sorted([a, b, c])]))
check("排序：同优先级按到达时间（先到先排）", a > b and sorted([a, b]) == [b, a])
d = Request("d", [1], SamplingParams(max_tokens=1), arrival_time=1.0, priority=1)
check("排序：同优先级同到达时间按 request_id 兜底（确定性）", sorted([b, d]) == [b, d])

fcfs = create_request_queue("fcfs")
prio = create_request_queue("priority")
for request in (a, b, c):
    fcfs.add_request(request)
    prio.add_request(request)
check("FCFS 队列按到达顺序（c 的优先级更高也不会插队）",
      [fcfs.pop_request().request_id for _ in range(3)] == ["a", "b", "c"])
check("priority 队列按 (priority, arrival_time)：先出 c，再出 b（同优先级先到）",
      [prio.pop_request().request_id for _ in range(3)] == ["c", "b", "a"])
check("两种队列都能前插（恢复用）与按 ID 移除",
      (lambda q: (q.add_request(a), q.prepend_request(b), q.pop_request().request_id == "b",
                  q.remove_request(a), not q))(FCFSRequestQueue()) == (None, None, True, None, True))

# priority 队列的懒惰删除：移除后堆里的旧条目不能再冒出来
prio = PriorityRequestQueue()
for request in (a, b, c):
    prio.add_request(request)
prio.remove_request(c)
check("priority 队列：移除的请求不会再被取出（堆里的旧条目被惰性丢弃）",
      [prio.pop_request().request_id for _ in range(2)] == ["b", "a"] and not prio)

# ------------------------------------------------ 6. check_stop 的六条分支

def stop_with(output_tokens, max_model_len=64, **sampling):
    request = make(**sampling)
    request.append_output_token_ids(list(output_tokens))
    stopped = check_stop(request, max_model_len)
    return stopped, request


stopped, _ = stop_with([1], min_tokens=3)
check("check_stop：生成数没到 min_tokens 时**不查任何条件**（哪怕已到 max_tokens 也不行）",
      stopped is False)
stopped, request = stop_with([1, 99], min_tokens=1)
check("check_stop：遇到 eos → FINISHED_STOPPED，原因 STOP",
      stopped and request.status == RequestStatus.FINISHED_STOPPED
      and request.get_finished_reason() == FinishReason.STOP)
stopped, _ = stop_with([1, 99], ignore_eos=True, max_tokens=4)
check("check_stop：ignore_eos=True 时 eos 不停（继续按长度判）", stopped is False)
stopped, request = stop_with([1, 7], stop_token_ids=[7])
check("check_stop：显式 stop token → 停止并记下 stop_reason",
      stopped and request.status == RequestStatus.FINISHED_STOPPED and request.stop_reason == 7)
stopped, request = stop_with([1, 2, 3, 4], max_tokens=4)
check("check_stop：生成数够 max_tokens → FINISHED_LENGTH_CAPPED",
      stopped and request.status == RequestStatus.FINISHED_LENGTH_CAPPED
      and request.get_finished_reason() == FinishReason.LENGTH)
stopped, request = stop_with([1, 2], max_model_len=5)
check("check_stop：上下文到顶（num_tokens >= max_model_len）→ FINISHED_LENGTH_CAPPED",
      stopped and request.status == RequestStatus.FINISHED_LENGTH_CAPPED
      and request.num_tokens == 5)
stopped, _ = stop_with([1, 2], max_model_len=64)
check("check_stop：都不满足时不停止", stopped is False)

# ------------------------------------------------ 7. 从 EngineCoreRequest 构造

params = SamplingParams(max_tokens=2)
core_request = EngineCoreRequest(request_id="x", prompt_token_ids=[5, 6],
                                 sampling_params=params, arrival_time=3.0, priority=7)
request = Request.from_engine_core_request(core_request)
check("from_engine_core_request：字段逐个带过来，prompt 是副本",
      request.request_id == "x" and list(request.prompt_token_ids) == [5, 6]
      and request.priority == 7 and request.arrival_time == 3.0
      and request.max_tokens == 2 and request.max_tokens == params.max_tokens)
check("from_engine_core_request：是副本而不是引用",
      (core_request.prompt_token_ids.append(77),
       list(request.prompt_token_ids) == [5, 6])[1])

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
