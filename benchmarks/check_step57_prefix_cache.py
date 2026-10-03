"""57C 验收（对应需求里的 `test_prefix_cache.py`）：前缀缓存的 hash、发布与命中。

197 把这件事拆成两步，用例也按这两步组织：

  **第一步：确定 token 后算 hash**（`BlockHasher`）——增量、链式、稳定、带 salt；
  未凑满的块**不参与**（半块的内容随时会被覆盖）。

  **第二步：有有效 KV 后才能复用**（发布）——只有完整块进索引；发布上限
  `floor(min(num_computed, num_tokens) / block_size)`；abort 不发布；失败的 forward 不发布。

然后是命中侧的四条硬要求（197 §5-§7）：

  - 命中只可能是**前缀**（链式 hash，中间不会断）；
  - 命中上限留一个 token（`num_tokens - 1`）——最后那个 token 的 logits 是下一个 token 的来源，
    不能靠"全命中"凭空得到；
  - 零引用的命中块要被 `touch` 摘出空闲队列，所以**容量检查必须把它算进去**（否则检查说够、
    分配时不够）；
  - 共享的完整块**不能被覆写**：新算的 token 落在新块上（按块表逐 token 验一遍）。

最后一条是"开/关前缀缓存输出一致"：同一批请求跑两遍，生成的 token 必须一样——缓存是加速，
不是改变语义。
"""

import sys

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

import torch

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, Request, SamplingParams,
                    SchedulerConfig, UniProcExecutor, VllmConfig, Worker)
from minivllm.core.kv_cache_manager import KVCacheManager
from minivllm.core.kv_cache_utils import BlockHasher, init_none_hash
from minivllm.testing.fake_runner import FakeRunner
from minivllm.worker.block_table import BlockTable

FAIL = []
init_none_hash("check_step57_prefix_cache")


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def first_line(error):
    return error.splitlines()[0] if error else "没有报错"


def make_config(prefix=True, blocks=8, block_size=4, seqs=4, budget=16, max_model_len=64):
    return VllmConfig(
        model_config=ModelConfig(model="dummy", max_model_len=max_model_len),
        cache_config=CacheConfig(block_size=block_size, num_gpu_blocks=blocks,
                                 enable_prefix_caching=prefix),
        scheduler_config=SchedulerConfig(max_num_seqs=seqs, max_num_batched_tokens=budget),
        device_config=DeviceConfig(device="cpu"))


def make_request(request_id, prompt, max_tokens=4, hasher=None):
    request = Request(request_id, list(prompt),
                      SamplingParams(max_tokens=max_tokens, temperature=0.0, eos_token_id=999),
                      arrival_time=1.0, block_hasher=hasher)
    return request


def run_engine(config, prompts, max_tokens=4, tokens=None):
    """跑一批请求到底，返回 {req_id: [输出 token]}。"""
    tokens = tokens or {req_id: [7] * (max_tokens + 4) for req_id in prompts}
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config, model_runner=FakeRunner(tokens=tokens))))
    outputs = {}
    for req_id, prompt in prompts.items():
        engine.add_request(req_id, list(prompt), SamplingParams(max_tokens=max_tokens,
                                                               temperature=0.0, eos_token_id=999))
    scheduler = engine.engine_core.engine_core.scheduler
    while engine.has_unfinished_requests():
        for output in engine.step():
            outputs[output.request_id] = list(output.token_ids)
    engine.shutdown()
    return outputs, scheduler


# ------------------------------------------------ 1. hash 链

hasher = BlockHasher(block_size=4)
request = make_request("h", [1, 2, 3, 4, 5, 6], hasher=hasher)
check("1. 未凑满的块不算 hash：6 个 token / 块大小 4 → 只有 1 个 hash",
      len(request.block_hashes) == 1, f"{len(request.block_hashes)} 个")

request.append_output_token_ids([7, 8, 9])
check("1. 增量：追加到 9 个 token → 第 2 个块凑满，hash 变 2 个（前面的不动）",
      len(request.block_hashes) == 2)

base = make_request("base", [1, 2, 3, 4, 5, 6, 7, 8], hasher=BlockHasher(4))
changed_head = make_request("ch", [1, 2, 3, 9, 5, 6, 7, 8], hasher=BlockHasher(4))
changed_tail = make_request("ct", [1, 2, 3, 4, 9, 6, 7, 8], hasher=BlockHasher(4))
check("1. 链式：第 1 个块变了 → 它自己与**后面所有块**的 hash 都变",
      changed_head.block_hashes[0] != base.block_hashes[0]
      and changed_head.block_hashes[1] != base.block_hashes[1])
check("1. 链式（另一半）：只有第 2 个块变了 → 第 1 个块 hash 不变、第 2 个变",
      changed_tail.block_hashes[0] == base.block_hashes[0]
      and changed_tail.block_hashes[1] != base.block_hashes[1])

same = make_request("h3", [1, 2, 3, 4, 5, 6], hasher=BlockHasher(4))
check("1. 稳定：同样的 token 前缀算出同样的 hash（sha256 + 自定编码，不用 hash()）",
      same.block_hashes[0] == request.block_hashes[0])

salted = make_request("h4", [1, 2, 3, 4, 5, 6], hasher=BlockHasher(4, extra_keys=("tenant-a",)))
check("1. salt 不同 → hash 不同（同一段 token 不能跨租户命中）",
      salted.block_hashes[0] != request.block_hashes[0])

plain = make_request("h5", [1, 2, 3, 4, 5, 6])
check("1. 没挂计算器（关了前缀缓存）→ 不产生 hash", plain.block_hashes == [])

# ------------------------------------------------ 2. 发布边界

manager = KVCacheManager(CacheConfig(block_size=4, num_gpu_blocks=8, enable_prefix_caching=True),
                         max_model_len=64)
hasher = BlockHasher(4)
request = make_request("p1", [1, 2, 3, 4, 5, 6], hasher=hasher)
blocks = manager.allocate_slots(request, num_new_tokens=6)
check("2. 分配：6 个 token 要 2 个块（第二个只用了半块）",
      len(blocks.blocks[0]) == 2 and manager.num_allocated_blocks == 2)
request.num_computed_tokens = 6
manager.cache_blocks(request, request.num_computed_tokens)
check("2. 发布只进**完整块**：6 个 token → 只发布 1 个块（半块不进索引）",
      manager.num_cached_blocks() == 1 and manager.coordinator.manager.num_cached_block["p1"] == 1)

request2 = make_request("p2", [1, 2, 3, 4, 5, 6, 7, 8], hasher=BlockHasher(4))
manager.allocate_slots(request2, num_new_tokens=8)
request2.num_computed_tokens = 8
manager.cache_blocks(request2, request2.num_computed_tokens)
check("2. 继续凑满 → 发布第 2 个块（增量发布，不重复登记前一个）",
      manager.coordinator.manager.num_cached_block["p2"] == 2
      and manager.coordinator.block_pool.blocks[
          manager.coordinator.manager.get_blocks("p2")[1].block_id].block_hash is not None)

# 上限：num_computed 可能比 num_tokens 大（本轮算完、采样结果还没提交回来）
overshoot = make_request("p3", [1, 2, 3, 4, 5, 6, 7, 8, 9], hasher=BlockHasher(4))
manager.allocate_slots(overshoot, num_new_tokens=9)
overshoot.num_computed_tokens = 12                     # 假装算过头了
manager.cache_blocks(overshoot, overshoot.num_computed_tokens)
check("2. 发布上限夹到 min(num_computed, num_tokens)：算过头也不会发布不存在的块",
      manager.coordinator.manager.num_cached_block["p3"] == 2,
      f"发布了 {manager.coordinator.manager.num_cached_block['p3']} 个块（9 个 token → 2 个完整块）")

# ------------------------------------------------ 3. 命中：前缀、粒度、共享

config = make_config(blocks=8, budget=8)
engine = LLMEngine(config, UniProcExecutor(config, Worker(config, model_runner=FakeRunner(
    tokens={"A": [7] * 8, "B": [9] * 8}))))
scheduler = engine.engine_core.engine_core.scheduler
pool = scheduler.kv_cache_manager
params = SamplingParams(max_tokens=6, temperature=0.0, eos_token_id=999)
engine.add_request("A", [1, 2, 3, 4, 5, 6, 7, 8], params)
while not scheduler.has_finished_requests() and scheduler.num_steps < 1:
    engine.step()
check("3. A 跑完 prompt 之后：它的完整块留在池子里等复用（此时 A 还在生成）",
      pool.num_cached_blocks() >= 2 and [r.request_id for r in scheduler.running] == ["A"],
      f"cached={pool.num_cached_blocks()}、running={[r.request_id for r in scheduler.running]}")

hit_blocks, hit_length = pool.get_computed_blocks(
    make_request("probe", [1, 2, 3, 4, 5, 6, 7, 8], hasher=BlockHasher(4)))
check("3. 命中上限留一个 token：8 个 token / 块大小 4 → 只命中 1 个块（4 个 token）",
      hit_length == 4 and len(hit_blocks.blocks[0]) == 1,
      f"命中 {hit_length} 个 token、{len(hit_blocks.blocks[0])} 个块")

engine.add_request("B", [1, 2, 3, 4, 5, 6, 7, 8], params)
packet = scheduler.schedule()
new_b = [data for data in packet.scheduled_new_reqs if data.req_id == "B"]
check("3. 同前缀的请求在**首次接纳**就命中（不必先算一遍）", bool(new_b),
      f"packet={packet.num_scheduled_tokens}")
if new_b:
    data = new_b[0]
    check("3. 命中之后起点不是 0：包里 num_computed_tokens = 命中长度（执行侧据此跳过这段计算）",
          data.num_computed_tokens == 4, f"起点 {data.num_computed_tokens}")
    check("3. B 的块表 = 命中块 + 新块（整张表一起发）",
          len(data.block_ids[0]) == 2, f"块表 {data.block_ids[0]}")
    shared_id = data.block_ids[0][0]
    shared = pool.coordinator.block_pool.blocks[shared_id]
    check("3. 命中块被 touch：引用计数 2（A 还在跑）+ 有 hash（可复用身份没丢）",
          shared.ref_cnt == 2 and shared.block_hash is not None,
          f"ref_cnt={shared.ref_cnt}")

    # 共享块不可覆写：B 这轮要算的位置都落在**新块**上
    table = BlockTable(4, 1, 8)
    table.add_row(0, list(data.block_ids[0]))
    positions = torch.arange(data.num_computed_tokens, data.num_computed_tokens
                             + packet.num_scheduled_tokens["B"])
    req_indices = torch.zeros(positions.shape[0], dtype=torch.int64)
    slots = table.compute_slot_mapping(1, positions, req_indices)
    touched_blocks = {int(slot) // 4 for slot in slots.tolist()}
    check("3. 共享块不被覆写：B 本轮写的槽位全部落在**新块**上（命中块一个都不碰）",
          shared_id not in touched_blocks and touched_blocks == {int(data.block_ids[0][1])},
          f"写入块 {sorted(touched_blocks)}、命中块 {shared_id}")
engine.shutdown()

# ------------------------------------------------ 4. 容量检查必须算上"会被摘走的命中块"

manager = KVCacheManager(CacheConfig(block_size=4, num_gpu_blocks=2, enable_prefix_caching=True),
                         max_model_len=64)
producer = make_request("producer", [1, 2, 3, 4], hasher=BlockHasher(4))
manager.allocate_slots(producer, num_new_tokens=4)
producer.num_computed_tokens = 4
manager.cache_blocks(producer, producer.num_computed_tokens)
manager.free(producer)
check("4. 前置：池子 2 块全空闲，其中 1 块带着可命中的 hash",
      manager.num_free_blocks() == 2 and manager.num_cached_blocks() == 1)

consumer = make_request("consumer", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], hasher=BlockHasher(4))
computed_blocks, num_hit = manager.get_computed_blocks(consumer)
before = (manager.num_free_blocks(), manager.num_allocated_blocks)
result = manager.allocate_slots(consumer, num_new_tokens=12, num_new_computed_tokens=num_hit,
                                new_computed_blocks=computed_blocks)
check("4. 命中块 ref_cnt=0（在空闲队列里）：需求 = 新增 + 会被摘走的命中块，"
      "所以 2 空闲块装不下 3 个块 → 返回 None（不是先通过、再在分配时报错）",
      result is None and (manager.num_free_blocks(), manager.num_allocated_blocks) == before,
      f"命中 {num_hit} 个 token、结果={result}、状态未变={(manager.num_free_blocks(), manager.num_allocated_blocks) == before}")

# ------------------------------------------------ 5. 逐出：命中块被分配走就不再命中

manager = KVCacheManager(CacheConfig(block_size=4, num_gpu_blocks=2, enable_prefix_caching=True),
                         max_model_len=64)
producer = make_request("p", [1, 2, 3, 4], hasher=BlockHasher(4))
manager.allocate_slots(producer, num_new_tokens=4)
producer.num_computed_tokens = 4
manager.cache_blocks(producer, producer.num_computed_tokens)
manager.free(producer)
probe = [1, 2, 3, 4, 5, 6, 7, 8]
_, before_hit = manager.get_computed_blocks(make_request("q", probe, hasher=BlockHasher(4)))
eater = make_request("eater", [9, 9, 9, 9, 9, 9, 9, 9], hasher=BlockHasher(4))
manager.allocate_slots(eater, num_new_tokens=8)          # 2 个新块 → 把缓存块也吃掉了
_, after_hit = manager.get_computed_blocks(make_request("q", probe, hasher=BlockHasher(4)))
check("5. 缓存块被分配出去覆写后，它的登记被摘掉（下一次不再命中）",
      before_hit == 4 and after_hit == 0 and manager.num_cached_blocks() == 0,
      f"命中 {before_hit} → {after_hit}、cached={manager.num_cached_blocks()}")

# ------------------------------------------------ 6. 开关前缀缓存：输出一致

prompts = {"r1": [1, 2, 3, 4, 5, 6, 7, 8], "r2": [1, 2, 3, 4, 5, 6, 7, 8], "r3": [9, 9, 9, 9, 9]}
# 预算 8：一轮只装得下一个 prompt，r2/r3 必须等 r1 算完并发布之后才被接纳——命中才有机会发生
with_cache, scheduler_on = run_engine(make_config(prefix=True, blocks=8, budget=8), prompts)
without_cache, scheduler_off = run_engine(make_config(prefix=False, blocks=8, budget=8), prompts)
check("6. 开/关前缀缓存：同一批请求生成的 token **完全一致**（缓存只该改变速度）",
      with_cache == without_cache, f"{with_cache} vs {without_cache}")
check("6. 关掉时真的没走缓存路径：没有 hash 计算器、没有发布、没有命中",
      scheduler_off.block_hasher is None
      and scheduler_off.kv_cache_manager.num_cached_blocks() == 0
      and not any(record["hits"] for record in scheduler_off.trace),
      f"cached={scheduler_off.kv_cache_manager.num_cached_blocks()}")
check("6. 开缓存时确实命中了（r2/r3 与 r1 共享前缀）",
      any(record["hits"] for record in scheduler_on.trace),
      str([record["hits"] for record in scheduler_on.trace if record["hits"]]))

# ------------------------------------------------ 7. abort 不发布

config = make_config(prefix=True, blocks=4)
engine = LLMEngine(config, UniProcExecutor(config, Worker(config, model_runner=FakeRunner(
    tokens={"z": [5] * 8}))))
engine.add_request("z", [1, 2, 3, 4], SamplingParams(max_tokens=4, temperature=0.0,
                                                    eos_token_id=999))
scheduler = engine.engine_core.engine_core.scheduler
scheduler.schedule()
engine.abort_request(["z"])
check("7. abort 的请求不把自己的历史发布进缓存（半截状态不该被别人命中）",
      scheduler.kv_cache_manager.num_cached_blocks() == 0
      and scheduler.kv_cache_manager.num_free_blocks() == 4,
      f"cached={scheduler.kv_cache_manager.num_cached_blocks()}")
engine.shutdown()

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
