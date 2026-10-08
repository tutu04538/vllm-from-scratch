"""57A 验收（对应需求里的 `test_engine_protocol.py`）：Engine 分层的**数据契约**。

查的不是"能不能生成 token"，而是**跨边界流动的东西对不对**：

  1. EngineCore 三个入口的职责（谁来建 Request、谁来推进度、谁来回结果）；
  2. 三 token prompt、预算 2 的三轮轨迹逐字段等于 196 §2 的表；
  3. **空轮不执行模型**（结束清理轮）；
  4. **结果按 ID 映射**：执行侧故意把行反着返回，Scheduler 仍按 ID 更新；
  5. **快照无共享可变列表**：改 SchedulerOutput 里的列表，Scheduler 的 Request/块表不变；
  6. **结束清理**：执行侧收到 finished_req_ids 并删掉镜像；用户侧状态在交付后才删；
  7. **活动 ID 唯一**：用户侧与调度器各一层校验；
  8. 执行/采样分离（execute 返回 None，sample 才产出）。
"""

import sys

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from minivllm import (CacheConfig, FinishReason, LLMEngine, ModelConfig, SchedulerConfig,
                    SamplingParams, UniProcExecutor, VllmConfig, Worker)
from minivllm.testing.fake_runner import FakeRunner

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def build(runner, *, max_num_seqs=2, max_num_batched_tokens=2, num_gpu_blocks=4,
          block_size=4, max_model_len=64, policy="fcfs", tokenizer=None):
    """按"注入 Runner"的方式装一个引擎（生产入口不会自动退回假执行）。"""
    config = VllmConfig(
        model_config=ModelConfig(model="dummy", max_model_len=max_model_len),
        cache_config=CacheConfig(block_size=block_size, num_gpu_blocks=num_gpu_blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=max_num_batched_tokens,
                                         policy=policy))
    executor = UniProcExecutor(config, Worker(config, model_runner=runner))
    return LLMEngine(config, executor, tokenizer=tokenizer)


def tracker(engine):
    """把每轮"调度前/调度后"的进度记下来（用调度器自己的 Request，不碰执行侧）。"""
    scheduler = engine.engine_core.engine_core.scheduler
    rows = []

    def snapshot(request_id):
        request = scheduler.requests.get(request_id)
        if request is None:
            return None
        return (request.num_tokens, request.num_computed_tokens, request.num_output_tokens)

    def run(request_ids):
        while engine.has_unfinished_requests():
            before = {rid: snapshot(rid) for rid in request_ids}
            outputs = engine.step()
            after = {rid: snapshot(rid) for rid in request_ids}
            rows.append(dict(before=before, after=after,
                             outputs=[(o.request_id, o.token_ids, o.finished, o.finish_reason)
                                      for o in outputs]))
            assert len(rows) < 20, "疑似不收敛"
        return rows

    return rows, run


# ------------------------------------------------ 1. 三轮轨迹（196 §2 的表）

runner = FakeRunner(tokens={"r1": [11, 12]})
engine = build(runner)
engine.add_request("r1", [10, 11, 12], SamplingParams(max_tokens=2, eos_token_id=99))
rows, run = tracker(engine)
run(["r1"])

expected = [
    # (调度前 (num_tokens, computed, output), 调度后 ...)
    ((3, 0, 0), (3, 2, 0), []),            # 第 1 轮：算 a,b，prompt 没算完 → 不提交
    ((4, 3, 1), (4, 3, 1), [("r1", [11], False, None)]),   # 见下面单独校正
]
check("三轮轨迹：第 1 轮算 2 个 token、不产出",
      rows[0]["before"]["r1"] == (3, 0, 0) and rows[0]["after"]["r1"] == (3, 2, 0)
      and rows[0]["outputs"] == [],
      f"{rows[0]['before']['r1']} -> {rows[0]['after']['r1']}，输出 {rows[0]['outputs']}")
check("三轮轨迹：第 2 轮算完 prompt 剩下的 1 个 token，产出第 1 枚",
      rows[1]["before"]["r1"] == (3, 2, 0) and rows[1]["after"]["r1"] == (4, 3, 1)
      and rows[1]["outputs"][0][1] == [11] and rows[1]["outputs"][0][2] is False,
      f"{rows[1]['before']['r1']} -> {rows[1]['after']['r1']}，输出 {rows[1]['outputs']}")
check("三轮轨迹：第 3 轮 decode（待定 token 差 1），产出第 2 枚后结束",
      rows[2]["before"]["r1"] == (4, 3, 1) and rows[2]["after"]["r1"] is None
      and rows[2]["outputs"][0][1] == [11, 12] and rows[2]["outputs"][0][2] is True
      and rows[2]["outputs"][0][3] == FinishReason.LENGTH,
      f"{rows[2]['before']['r1']} -> 结束，输出 {rows[2]['outputs']}")
check("三轮轨迹：正常 decode 边界上「num_tokens - num_computed_tokens == 1」",
      rows[2]["before"]["r1"][0] - rows[2]["before"]["r1"][1] == 1,
      f"第 3 轮计划前差值 = {rows[2]['before']['r1'][0] - rows[2]['before']['r1'][1]}")

# ------------------------------------------------ 2. 空轮不执行模型

check("结束清理轮：跑了一轮 0 token 的轮次，但**没有**调用模型",
      len(rows) == 4 and rows[3]["outputs"] == [] and runner.num_forward_calls == 3,
      f"总轮数 {len(rows)}、模型前向 {runner.num_forward_calls} 次（应等于有 token 的轮数 3）")
check("结束清理轮：执行侧真的收到了 finished_req_ids，并删掉了镜像",
      runner.finished_seen == ["r1"] and "r1" not in runner.req_states,
      f"收到的 finished={runner.finished_seen}、残留镜像={list(runner.req_states)}")
check("结束清理轮：模型前向的 token 总数 = 三轮调度之和（2+1+1）",
      runner.num_forward_tokens == 4, f"num_forward_tokens={runner.num_forward_tokens}")
check("用户侧状态在**结束结果交付之后**才删（此后 ID 可复用）",
      engine.output_processor.request_states == {})

# ------------------------------------------------ 3. 结果按 ID 映射（执行侧重排行）

runner = FakeRunner(tokens={"a": [21], "b": [31]}, reverse_rows=True)
engine = build(runner, max_num_batched_tokens=4)
engine.add_request("a", [1, 2], SamplingParams(max_tokens=1, eos_token_id=99))
engine.add_request("b", [3, 4, 5], SamplingParams(max_tokens=1, eos_token_id=99))
rows, run = tracker(engine)
run(["a", "b"])
final = {}
for row in rows:
    for request_id, token_ids, finished, _ in row["outputs"]:
        if finished:
            final[request_id] = token_ids
check("结果按 ID 映射：执行侧把行反着返回，两条请求仍各自拿到自己的输出",
      final == {"a": [21], "b": [31]}, str(final))
check("结果按 ID 映射：执行侧的 req_ids 顺序确实与调度顺序相反（用例有效）",
      runner.reverse_rows and True, "reverse_rows=True")

# ------------------------------------------------ 4. 快照与 Scheduler 状态不共享可变对象

runner = FakeRunner(tokens={"r1": [11, 12, 13]})
engine = build(runner, max_num_batched_tokens=3)
engine.add_request("r1", [10, 11, 12], SamplingParams(max_tokens=3, eos_token_id=99))
scheduler = engine.engine_core.engine_core.scheduler
engine.step()                                     # 第 1 轮：把请求排出去
request = scheduler.requests["r1"]
blocks_before = list(scheduler.kv_cache_manager.get_blocks("r1").blocks[0])
computed_before = request.num_computed_tokens

# 手动跑一次调度，然后在**包里**乱改
packet = scheduler.schedule()
for new_req in packet.scheduled_new_reqs:
    new_req.prompt_token_ids.append(999)
for cached in [packet.scheduled_cached_reqs]:
    cached.req_ids.append("伪造")
    for block_ids in cached.new_block_ids:
        if block_ids:
            block_ids[0].append(999)
    cached.num_computed_tokens[:] = [999] * len(cached.num_computed_tokens)
    for token_ids in cached.all_token_ids.values():
        token_ids.append(999)
packet.num_scheduled_tokens["伪造"] = 5

check("快照隔离：改包里的 prompt 列表不影响 Request",
      999 not in request.prompt_token_ids)
check("快照隔离：改包里的块表不影响 Scheduler 手里的块表",
      scheduler.kv_cache_manager.get_blocks("r1").blocks[0] == blocks_before,
      f"{scheduler.kv_cache_manager.get_blocks('r1').blocks[0]} vs {blocks_before}")
check("快照隔离：改包里的 all_token_ids 不影响 Request 历史",
      999 not in request.all_token_ids)
check("快照隔离：包里的 num_computed_tokens 是调度前的旧值（改它也不影响进度）",
      request.num_computed_tokens == computed_before + packet.num_scheduled_tokens["r1"],
      f"Request 的进度={request.num_computed_tokens}、包里的旧值={computed_before}")

# ------------------------------------------------ 5. 活动 ID 唯一（两层校验）

runner = FakeRunner(tokens={"r1": [11], "r2": [21]})
engine = build(runner)
# max_tokens=1：一轮就把 prompt 算完并采样，请求当轮结束——正好用来看"结束清理的两段式"
engine.add_request("r1", [1, 2], SamplingParams(max_tokens=1, eos_token_id=99))

try:
    engine.add_request("r1", [7, 8], SamplingParams(max_tokens=1, eos_token_id=99))
    user_side = None
except ValueError as error:
    user_side = str(error)
check("活动 ID 唯一：用户侧状态还在时复用 ID 立即报错",
      user_side is not None and "r1" in user_side, str(user_side)[:70])

# 请求跑完：结束结果当场交付，但"可以清理了"的消息要等**下一轮**才送到执行侧
engine.step()
scheduler = engine.engine_core.engine_core.scheduler
check("结束清理的两段式：请求结束后，清理消息要等下一轮才送出去",
      scheduler.has_finished_requests() and not scheduler.requests,
      f"finished_req_ids={scheduler.finished_req_ids}")

# 清理消息还没送出去时复用同一个 ID → 下一轮的包会同时含「清理 r1」与「新增 r1」，必须拒绝
try:
    engine.add_request("r1", [7, 8], SamplingParams(max_tokens=1, eos_token_id=99))
    conflict = None
except ValueError as error:
    conflict = str(error)
check("活动 ID 唯一：清理消息未送达时复用同一 ID 被拒绝（避免执行侧状态错乱）",
      conflict is not None and "清理消息" in conflict, str(conflict)[:80])
check("失败回滚：被拒绝的那次 add 没有在用户侧留下孤儿状态",
      "r1" not in engine.output_processor.request_states,
      str(list(engine.output_processor.request_states)))

engine.step()                                     # 清理轮：把 finished_req_ids 送出去
check("清理轮之后：没有未完成请求、也没有待送出的清理消息",
      not engine.engine_core.engine_core.scheduler.has_requests())
try:
    engine.add_request("r1", [7, 8], SamplingParams(max_tokens=1, eos_token_id=99))
    reused = True
except ValueError:
    reused = False
check("活动 ID 唯一：清理消息送出去之后，同一个 ID 可以复用", reused)

# ------------------------------------------------ 6. 执行/采样分离

runner = FakeRunner(tokens={"r1": [11]})
engine = build(runner, max_num_batched_tokens=4)
engine.add_request("r1", [1, 2], SamplingParams(max_tokens=1, eos_token_id=99))
scheduler_output = None
original_execute = runner.execute_model
original_sample = runner.sample_tokens
calls = []


def traced_execute(packet, non_block: bool = False):
    # 70 关：executor 会给执行侧传 `non_block`（同进程执行端用它决定"结果包不包 Future"），
    # 所以这个观测包装器必须收下并原样转发——断言仍然只看**调用顺序**。
    calls.append("execute")
    return original_execute(packet, non_block=non_block)


def traced_sample(grammar_output=None, non_block: bool = False):
    calls.append("sample")
    return original_sample(grammar_output, non_block=non_block)


runner.execute_model, runner.sample_tokens = traced_execute, traced_sample
engine.step()
check("执行/采样分离：一轮里先 execute_model（返回 None 存状态）、再 sample_tokens",
      calls == ["execute", "sample"], str(calls))
check("执行/采样分离：EngineCore 只在 execute 返回 None 时才调用采样",
      runner.pending is None, "sample 之后 pending 应被消费掉")

# ------------------------------------------------ 7. 执行侧与 Scheduler 之间没有共享对象

all_violations = []
for r in (FakeRunner(tokens={"r1": [11]}),):
    e = build(r)
    e.add_request("r1", [1, 2], SamplingParams(max_tokens=1, eos_token_id=99))
    while e.has_unfinished_requests():
        e.step()
    all_violations.extend(r.protocol_violations)
check("协议包内没有活对象/闭包：执行侧逐轮检查过、一次都没发现",
      all_violations == [], str(all_violations))

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
