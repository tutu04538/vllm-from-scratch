"""57C 验收（对应需求里的 `test_preemption.py`）：抢占与恢复。

抢占在 196 §6 的定义很短：**释放块 → 状态改 PREEMPTED → 进度归零 → 放回 waiting 队首**，
保留 prompt、已提交输出、优先级、采样配置与块 hash 链。用例按"谁被抢 → 计划怎么撤 →
怎么回来"分四段，最后一段用真实模型端到端验证"抢占过的那一批，输出与没被抢占时一模一样"。

  1. victim 选择（FCFS 取 running 尾部 / priority 取数值最大者）与抢占动作；
  2. **计划撤销**：victim 可能已经在本轮计划里——要从 `num_scheduled_tokens` 里删掉它、
     把预算退回来，否则"已经释放的块"会被发到执行侧（只有 priority 策略下才会出现
     "victim 排在当前请求前面"，FCFS 的 victim 永远是队尾）；
  3. 本轮发生过抢占就**不接纳 waiting**（否则刚释放的块立刻被新请求吃掉，被抢的永远回不来）；
  4. 恢复：包里带 `resumed_req_ids` + **整张**块表；执行侧整表替换；恢复可以命中仍在池子里的
     完整块，所以不必从位置 0 全部重算；
  5. 端到端：小池子逼出抢占，输出与大池子（不抢占）逐 token 一致。

**测试手法上的一条纪律**：想看某一轮的包，不能"绕过 execute 直接调 `scheduler.schedule()`"——
那样 Scheduler 的进度前进了、执行侧却没算，之后两边就对不上（我的第一版就是这样：两条请求
卡在 computed == num_tokens 上，最后被空转保护报出来）。这里用 `PacketRecorder` 代理
`schedule()`：正常跑引擎，顺手把包记下来。
"""

import json
import sys

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, RequestStatus,
                    SamplingParams, SchedulerConfig, UniProcExecutor, VllmConfig, Worker)
from minivllm.core.kv_cache_utils import init_none_hash
from minivllm.testing.fake_runner import FakeRunner

FAIL = []
TINY_DIR = "fixtures/step30_qwen3/tiny_gqa"
TINY_CONFIG = json.load(open(f"{TINY_DIR}/config.json"))
init_none_hash("check_step57_preemption")


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


class PacketRecorder:
    """代理 `Scheduler.schedule()`：包照常返回，同时记下来（不打断执行链）。"""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.packets = []
        self._original = scheduler.schedule
        scheduler.schedule = self.schedule

    def schedule(self):
        packet = self._original()
        self.packets.append(packet)
        return packet

    def restore(self):
        self.scheduler.schedule = self._original


def make_config(blocks=4, block_size=4, seqs=4, budget=8, policy="fcfs", prefix=True,
                model="dummy", hf_config=None, max_model_len=64):
    return VllmConfig(
        model_config=ModelConfig(model=model, max_model_len=max_model_len, hf_config=hf_config),
        cache_config=CacheConfig(block_size=block_size, num_gpu_blocks=blocks,
                                 enable_prefix_caching=prefix),
        scheduler_config=SchedulerConfig(max_num_seqs=seqs, max_num_batched_tokens=budget,
                                        policy=policy),
        device_config=DeviceConfig(device="cpu"))


def build(config, tokens, prompts, max_tokens=4, priority=None, model_runner=None):
    runner = model_runner or FakeRunner(tokens=tokens)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config, model_runner=runner)))
    scheduler = engine.engine_core.engine_core.scheduler
    priority = priority or {}
    for req_id, prompt in prompts.items():
        engine.add_request(req_id, list(prompt),
                           SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                          eos_token_id=999),
                           priority=priority.get(req_id, 0))
    return engine, scheduler


def run_until(engine, scheduler, predicate):
    while engine.has_unfinished_requests() and not predicate():
        engine.step()


# ------------------------------------------------ 1. victim 选择与抢占动作

# 池子 2 块：两条 prompt 各 4 个 token 各占 1 块；下一轮各自长大就需要第 2 块 → 必然抢占
config = make_config(blocks=2, budget=8)
engine, scheduler = build(config, {"A": [11] * 8, "B": [22] * 8},
                          {"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}, max_tokens=6)
recorder = PacketRecorder(scheduler)
run_until(engine, scheduler, lambda: scheduler.num_preemptions > 0)
victim = scheduler.requests["B"]
check("1. FCFS：牺牲者是 running **尾部**（B），正在被调度的 A 不受影响",
      victim.status == RequestStatus.PREEMPTED
      and [r.request_id for r in scheduler.running] == ["A"],
      f"B 状态={victim.status}、running={[r.request_id for r in scheduler.running]}")
check("1. 抢占动作：进度归零、status=PREEMPTED、抢占计数 +1",
      victim.num_computed_tokens == 0 and victim.num_preemptions == 1
      and scheduler.num_preemptions == 1)
check("1. 放回 waiting **队首**（它已经有历史，该比新来的先得到机会）",
      scheduler.waiting.peek_request().request_id == "B",
      f"waiting={scheduler.waiting.request_ids()}")
check("1. 保留历史与配置：prompt、已提交输出、优先级、采样参数、块 hash 链都在",
      victim.prompt_token_ids == [5, 6, 7, 8]
      and victim.sampling_params.max_tokens == 6 and victim.priority == 0
      and victim.num_tokens == 5 and len(victim.block_hashes) == 1,
      f"prompt={victim.prompt_token_ids}、num_tokens={victim.num_tokens}、"
      f"hash 数={len(victim.block_hashes)}")
check("1. 抢占释放了它的块：B 的块表清空、块被让给了 A（这正是抢占的目的）",
      scheduler.kv_cache_manager.get_blocks("B").blocks[0] == []
      and scheduler.kv_cache_manager.num_free_blocks() == 0
      and scheduler.kv_cache_manager.num_allocated_blocks == 2,
      f"B 块表={scheduler.kv_cache_manager.get_blocks('B').blocks[0]}、"
      f"空闲={scheduler.kv_cache_manager.num_free_blocks()}")
check("1. 抢占那一轮的包：B 不在 scheduled 里（它的计划被撤了）",
      recorder.packets[-1].num_scheduled_tokens == {"A": 1}
      and scheduler.trace[-1]["preempted"] == ["B"],
      f"包={recorder.packets[-1].num_scheduled_tokens}、"
      f"trace={scheduler.trace[-1]['preempted']}")
recorder.restore()
engine.shutdown()

# priority：牺牲者是 (priority, arrival_time) 最大的那条
config = make_config(blocks=2, budget=8, policy="priority")
engine, scheduler = build(config, {"A": [11] * 8, "B": [22] * 8},
                          {"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}, max_tokens=6,
                          priority={"A": 0, "B": 5})   # 数值大 = 优先级低
run_until(engine, scheduler, lambda: scheduler.num_preemptions > 0)
check("1. priority：牺牲者是优先级**最低**的那条（数值最大的 B），不是队尾",
      scheduler.requests["B"].status == RequestStatus.PREEMPTED
      and [r.request_id for r in scheduler.running] == ["A"],
      f"B 状态={scheduler.requests['B'].status}")
engine.shutdown()

# ------------------------------------------------ 2. 计划撤销与预算退回

# priority 下 victim 可能**排在当前请求前面**（FCFS 永远取队尾，所以这条路径只有 priority 会走到）：
# V 优先级最低、排在队首先被排上；随后 X 分配失败 → 抢 V → 撤回 V 的计划再重试分配。
config = make_config(blocks=3, budget=8, policy="priority", seqs=4)
engine, scheduler = build(config, {"V": [11] * 8, "X": [22] * 8},
                          {"V": [1, 2, 3, 4], "X": [5, 6, 7, 8]}, max_tokens=6,
                          priority={"V": 9, "X": 0})
recorder = PacketRecorder(scheduler)
run_until(engine, scheduler, lambda: scheduler.num_preemptions > 0)
packet = recorder.packets[-1]
preempt_round = scheduler.trace[-1]
check("2. （用例前提）被抢的是 V",
      preempt_round["preempted"] == ["V"],
      f"trace={preempt_round['preempted']}、本轮包={packet.num_scheduled_tokens}")
check("2. 撤销彻底：`num_scheduled_tokens` 里没有 V，X 还在（它才是这一轮真正要跑的）",
      "V" not in packet.num_scheduled_tokens and "X" in packet.num_scheduled_tokens,
      str(packet.num_scheduled_tokens))
check("2. 它也不出现在 CachedRequestData 里（否则执行侧会收到已经释放的块）",
      "V" not in packet.scheduled_cached_reqs.req_ids,
      str(packet.scheduled_cached_reqs.req_ids))
check("2. 预算退回之后本轮排的 token 数不超过预算（撤回的份额没被重复使用）",
      packet.total_num_scheduled_tokens <= 8
      and packet.total_num_scheduled_tokens == sum(packet.num_scheduled_tokens.values()),
      f"本轮共排 {packet.total_num_scheduled_tokens} 个 token（预算 8）")
check("2. 本轮发生过抢占 → **不接纳 waiting**（即便预算还有剩）",
      not packet.scheduled_new_reqs and bool(scheduler.waiting),
      f"new={[d.req_id for d in packet.scheduled_new_reqs]}、"
      f"waiting={scheduler.waiting.request_ids()}")
check("2. V 被放回 waiting 等着重来（不是被丢掉）",
      scheduler.requests["V"].status == RequestStatus.PREEMPTED
      and "V" in scheduler.waiting.request_ids())
recorder.restore()
engine.shutdown()

# ------------------------------------------------ 3. 恢复

config = make_config(blocks=4, budget=8, prefix=False)   # 关缓存：恢复必须从位置 0 重算
engine, scheduler = build(config, {"A": [11] * 12, "B": [22] * 12},
                          {"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}, max_tokens=8)
recorder = PacketRecorder(scheduler)
run_until(engine, scheduler, lambda: scheduler.num_preemptions > 0)
tokens_before_preempt = scheduler.requests["B"].num_tokens
check("3. （用例前提）B 被抢，块表已清空（旧物理编号不再属于它）",
      scheduler.kv_cache_manager.get_blocks("B").blocks[0] == [],
      str(scheduler.kv_cache_manager.get_blocks("B").blocks[0]))
run_until(engine, scheduler, lambda: scheduler.requests["B"].status == RequestStatus.RUNNING
          and scheduler.requests["B"].num_preemptions == 1)
resume_packet = recorder.packets[-1]
cached = resume_packet.scheduled_cached_reqs
check("3. 恢复的请求走 CachedRequestData，并且被标记为 **resumed**（不是新请求）",
      "B" in cached.req_ids and "B" in cached.resumed_req_ids
      and not any(data.req_id == "B" for data in resume_packet.scheduled_new_reqs),
      f"cached={cached.req_ids}、resumed={cached.resumed_req_ids}")
index_b = cached.req_ids.index("B")
new_block_ids = cached.new_block_ids[index_b]
check("3. 恢复发的是**整张**块表（resumed 的语义是替换，不是追加）",
      new_block_ids is not None
      and len(new_block_ids[0]) == len(scheduler.kv_cache_manager.get_blocks("B").blocks[0]),
      f"包里的块表={new_block_ids}、管理器的块表长度="
      f"{len(scheduler.kv_cache_manager.get_blocks('B').blocks[0])}")
check("3. 关掉缓存时恢复从位置 0 重算（进度归零是抢占动作的一部分）",
      cached.num_computed_tokens[index_b] == 0,
      f"起点={cached.num_computed_tokens[index_b]}")
check("3. 已提交输出与 prompt 一个都没丢（重算的是 KV，不是结果）：抢占前后 num_tokens 不变",
      scheduler.requests["B"].num_tokens == tokens_before_preempt
      and scheduler.requests["B"].prompt_token_ids == [5, 6, 7, 8],
      f"抢占前 {tokens_before_preempt} → 恢复时 {scheduler.requests['B'].num_tokens}")
recorder.restore()
engine.shutdown()

# 恢复时命中池子里仍留着的完整块：让 victim 有较长的历史（发布的块多），
# 且它被抢之后那些块没有被新请求用掉（池子留有余量）
config = make_config(blocks=5, budget=16, policy="priority", seqs=4, prefix=True)
engine, scheduler = build(config, {"r1": [11] * 8, "r2": [22] * 8},
                          # r1 先到但优先级高；r2 后到、历史长，是 FCFS/priority 下的牺牲者
                          {"r1": [1, 2, 3, 4], "r2": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
                          max_tokens=6, priority={"r1": 0, "r2": 9})
recorder = PacketRecorder(scheduler)
run_until(engine, scheduler, lambda: any(row["hits"] for row in scheduler.trace)
          or scheduler.num_preemptions > 0)
run_until(engine, scheduler, lambda: any(row["hits"] for row in scheduler.trace)
          or not engine.has_unfinished_requests())
hits = [row["hits"] for row in scheduler.trace if row["hits"]]
check("3. 开缓存时，被抢占的请求回来时能命中自己留在池子里的完整块（不必从 0 重算）",
      any("r2" in hit for hit in hits),
      f"抢占 {scheduler.num_preemptions} 次、命中记录={hits}")
recorder.restore()
engine.shutdown()

# ------------------------------------------------ 4. 端到端：真模型 + 小池子

def run_real(blocks):
    config = make_config(blocks=blocks, block_size=8, budget=24, seqs=4, prefix=True,
                         model=TINY_DIR, hf_config=TINY_CONFIG, max_model_len=64)
    # 不注入 Runner：走真实路径（Worker 自己建 GPUModelRunner 并读权重）
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    scheduler = engine.engine_core.engine_core.scheduler
    # tiny_gqa 的 vocab 只有 11（token id 必须 < 11）；两条共享后 8 个 token 的前缀，
    # 这样"抢占后命中缓存块"这条路径也有机会走到
    for req_id, prompt in (("r1", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1]),
                           ("r2", [9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])):
        engine.add_request(req_id, list(prompt),
                           SamplingParams(max_tokens=6, temperature=0.0, eos_token_id=999))
    runner = engine.engine_core.engine_core.model_executor.driver_worker.model_runner
    tokens = {}
    while engine.has_unfinished_requests():
        for output in engine.step():
            tokens[output.request_id] = list(output.token_ids)
    engine.shutdown()
    return tokens, scheduler, runner


big, scheduler_big, _ = run_real(blocks=16)
# 4 块 × 8 槽 = 32 槽：两条 12 token 的 prompt 先各自占 2 块（正好占满），
# 之后任一条长到第 17 个 token 都需要第 3 块 → 分配失败 → 抢占队尾
small, scheduler_small, runner_small = run_real(blocks=4)
check("4. 小池子真的逼出了抢占（否则下面的一致性对照没有意义）",
      scheduler_small.num_preemptions > 0,
      f"抢占 {scheduler_small.num_preemptions} 次")
check("4. 抢占过的那一批，输出与大池子（不抢占）**逐 token 一致**（重算不改变结果）",
      big == small, f"大池子={big}；小池子={small}")
check("4. 结束之后引用账目干净：没有活引用、全部块回到空闲队列",
      scheduler_small.kv_cache_manager.num_allocated_blocks == 0
      and scheduler_small.kv_cache_manager.num_free_blocks() == 4,
      f"占用={scheduler_small.kv_cache_manager.num_allocated_blocks}、"
      f"空闲={scheduler_small.kv_cache_manager.num_free_blocks()}")

printed = scheduler_small.format_trace()
check("4. `format_trace()` 能打印出可读的调度轨迹（要求里的 scheduler_trace 交付物）",
      "preempted" in printed and "free" in printed and printed.count("\n") >= 3,
      printed.splitlines()[0])

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
