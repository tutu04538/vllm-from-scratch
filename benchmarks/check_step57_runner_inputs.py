"""57B 验收（对应需求里的 `test_runner_inputs.py`）：Runner 的输入打包与批状态。

查的是"执行侧自己的账"，不碰模型数学（那是 `check_step57_model_logits.py`）：

  1. **198 §4 的逐值对照**：两条请求（A 续跑 2 个 token、B 新来 3 个）时
     `input_ids / positions / query_start_loc / seq_lens / slot_mapping / logits_indices`
     必须与需求里的数字完全一致；
  2. 只有 **ready 行**才采样：中间 prefill 块返回 `[]`，同一批里另一条照常出 token；
  3. `_update_states` 七步：删结束、移出未调度（保留镜像）、建新镜像、校正进度与输出长度、
     resumed 整表替换、用 `all_token_ids` 重建、入批并压实；
  4. **行号不是身份**：中间行被删掉之后 `condense` 会把后面的行搬过来，token 缓冲、
     块表、generator 都必须跟着**请求**走；
  5. 两步协议：`execute_model` 未消费就再来一轮 → 报错；没有 execute 就 sample → 报错；
     模型抛异常后 Runner 停摆，不把旧 logits 留给下一次采样；
  6. 块表越界（协议给少了块）当场报错，不悄悄写到 0 号块；
  7. **Scheduler 不碰 GPU 张量**：整条端到端跑一遍，检查每个 `SchedulerOutput` 里没有
     `torch.Tensor`、没有 Request / Scheduler / KVCacheManager 这类活对象。
"""

import json
import sys

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

import torch

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,
                    SchedulerConfig, UniProcExecutor, VllmConfig, Worker)
from minivllm.core.kv_cache_manager import KVCacheManager
from minivllm.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from minivllm.core.sched.scheduler import Scheduler
from minivllm.request import Request
from minivllm.worker import CachedRequestState, GPUModelRunner

FAIL = []
MODEL_DIR = "fixtures/step30_qwen3/tiny_gqa"
TINY_CONFIG = json.load(open(f"{MODEL_DIR}/config.json"))


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def make_config(model_dir=MODEL_DIR, block_size=4, num_gpu_blocks=16, max_num_seqs=4,
                max_num_batched_tokens=16, max_model_len=64, hf_config=None):
    return VllmConfig(
        model_config=ModelConfig(model=model_dir, dtype="float32", max_model_len=max_model_len,
                                 hf_config=hf_config if hf_config is not None else TINY_CONFIG),
        cache_config=CacheConfig(block_size=block_size, num_gpu_blocks=num_gpu_blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                        max_num_batched_tokens=max_num_batched_tokens),
        device_config=DeviceConfig(device="cpu"))


def build_runner(vllm_config=None, with_model=True, with_kv=True):
    runner = GPUModelRunner(vllm_config or make_config(), "cpu")
    if with_model:
        runner.load_model()
        if with_kv:
            runner.initialize_kv_cache((vllm_config or make_config()).cache_config)
    return runner


def add_state(runner, req_id, prompt, block_ids, num_computed=0, output=(), params=None):
    """直接往执行端镜像里放一条请求（真实路径由上一轮的包建出来）。"""
    state = CachedRequestState(req_id, list(prompt), params or SamplingParams(temperature=0.0),
                               None, [list(block_ids)], num_computed_tokens=num_computed,
                               output_token_ids=list(output))
    runner.requests[req_id] = state
    runner.input_batch.add_request(state)
    return state


def make_packet(new=(), cached_req_ids=(), num_scheduled=None, finished=(), **cached_kwargs):
    num_scheduled = num_scheduled or {}
    cached = CachedRequestData.make_empty()
    if cached_req_ids:
        cached = CachedRequestData(
            req_ids=list(cached_req_ids),
            resumed_req_ids=set(cached_kwargs.get("resumed_req_ids", ())),
            new_block_ids=list(cached_kwargs.get("new_block_ids", [None] * len(cached_req_ids))),
            num_computed_tokens=list(cached_kwargs.get("num_computed_tokens",
                                                       [0] * len(cached_req_ids))),
            num_output_tokens=list(cached_kwargs.get("num_output_tokens",
                                                     [0] * len(cached_req_ids))),
            all_token_ids=dict(cached_kwargs.get("all_token_ids", {})))
    return SchedulerOutput(
        scheduled_new_reqs=list(new), scheduled_cached_reqs=cached,
        num_scheduled_tokens=dict(num_scheduled),
        total_num_scheduled_tokens=sum(num_scheduled.values()),
        scheduled_spec_decode_tokens={}, finished_req_ids=set(finished))


def new_request(req_id, prompt, block_ids, num_computed=0, params=None):
    return NewRequestData(req_id=req_id, prompt_token_ids=list(prompt),
                          sampling_params=params or SamplingParams(temperature=0.0),
                          block_ids=([list(block_ids)],), num_computed_tokens=num_computed)


# ------------------------------------------------ 1. 198 §4 的逐值对照

runner = build_runner()
# A：prompt 5 个 token（chunked prefill 的第二块，前 3 个已算过），块表 [7,2]
# B：新请求，prompt 3 个 token，块表 [9]
#   → A[3]=4、A[4]=5、B[0..2]=6,7,8；位置 3、4 与 0、1、2
add_state(runner, "A", [1, 2, 3, 4, 5], [7, 2], num_computed=3)
packet = make_packet(new=[new_request("B", [6, 7, 8], [9])], cached_req_ids=["A"],
                     num_scheduled={"A": 2, "B": 3}, num_computed_tokens=[3],
                     num_output_tokens=[0])
runner._update_states(packet)
inputs = runner._prepare_inputs(packet)
check("input_ids 与 198 §4 一致（A[3], A[4], B[0], B[1], B[2]）",
      inputs.input_ids.tolist() == [4, 5, 6, 7, 8], str(inputs.input_ids.tolist()))
check("positions 是**绝对位置**（A 从 3 起、B 从 0 起）",
      inputs.positions.tolist() == [3, 4, 0, 1, 2], str(inputs.positions.tolist()))
check("query_start_loc / seq_lens",
      inputs.query_start_loc.tolist() == [0, 2, 5] and inputs.seq_lens.tolist() == [5, 3],
      f"{inputs.query_start_loc.tolist()} / {inputs.seq_lens.tolist()}")
check("slot_mapping = block_table[pos // bs] * bs + pos % bs（块 7、2、9）",
      inputs.slot_mapping.tolist() == [31, 8, 36, 37, 38], str(inputs.slot_mapping.tolist()))
check("logits_indices = query_start_loc[1:] - 1",
      inputs.logits_indices.tolist() == [1, 4], str(inputs.logits_indices.tolist()))
check("两条请求这一轮都算完已知历史 → 两行都可采样",
      inputs.sample_rows == [0, 1], str(inputs.sample_rows))

# ------------------------------------------------ 2. 未完成的 prefill 不产出

runner = build_runner()
# C 的 prompt 有 7 个 token，这一轮只算到第 5 个（中间 prefill 块）→ 不该产出 token
add_state(runner, "C", [1, 2, 3, 4, 5, 6, 7], [3, 4], num_computed=3)
packet = make_packet(new=[new_request("D", [8, 9, 0], [5])], cached_req_ids=["C"],
                     num_scheduled={"C": 2, "D": 3}, num_computed_tokens=[3],
                     num_output_tokens=[0])
runner.execute_model(packet)
state = runner.execute_model_state
check("只对 ready 行算 logits：3 个 query 行里只有 D 的末行进了采样行",
      state.sample_rows == [1] and state.logits.shape[0] == 1,
      f"sample_rows={state.sample_rows}、logits={tuple(state.logits.shape)}")
output = runner.sample_tokens()
check("未 ready 的请求本轮返回空列表（不提交采样结果）",
      output.sampled_token_ids[output.req_id_to_index["C"]] == []
      and len(output.sampled_token_ids[output.req_id_to_index["D"]]) == 1,
      str(dict(zip(output.req_ids, output.sampled_token_ids))))
check("未 ready 的请求镜像也没有变（进度与输出都没动）",
      runner.requests["C"].output_token_ids == [])

# ------------------------------------------------ 3. _update_states 的七步

runner = build_runner()
runner.execute_model(make_packet(new=[new_request("n1", [1, 2, 3], [0])],
                                 num_scheduled={"n1": 3}))
runner.sample_tokens()
check("3. 新请求：建镜像 + 入批（行 0），进度来自协议",
      list(runner.requests) == ["n1"] and runner.input_batch.req_id_to_index == {"n1": 0}
      and runner.requests["n1"].num_computed_tokens == 0)

# 结束清理：finished_req_ids 里的请求，镜像与批行都要删
runner.execute_model(make_packet(finished=["n1"]))
check("3. 结束清理：empty 轮的包里没有 token，但 finished_req_ids 仍然把镜像删干净",
      runner.requests == {} and runner.input_batch.req_id_to_index == {}
      and runner.input_batch.num_reqs == 0)

# 移出未调度：两条在批里，这一轮只排一条，另一条要被移出批但保留镜像
runner = build_runner()
runner.execute_model(make_packet(new=[new_request("x", [1, 2], [0]),
                                      new_request("y", [3, 4], [1])],
                                 num_scheduled={"x": 2, "y": 2}))
runner.sample_tokens()
runner.execute_model(make_packet(cached_req_ids=["x"], num_scheduled={"x": 1},
                                 num_computed_tokens=[2], num_output_tokens=[1]))
runner.sample_tokens()
check("3. 未调度（≠结束）：移出批行、**保留**镜像",
      "y" not in runner.input_batch.req_id_to_index and "y" in runner.requests
      and runner.input_batch.num_reqs == 1)

# 重建镜像：上一轮没被调度的请求回来了，协议必须带 all_token_ids
# （x 上一轮调度过 → 不带；y 没调度过 → 带。这正是"不是每轮都复制全部历史"的落点）
y_token = runner.requests["y"].output_token_ids[0]
runner.execute_model(make_packet(cached_req_ids=["x", "y"], num_scheduled={"x": 1, "y": 1},
                                 num_computed_tokens=[3, 2], num_output_tokens=[2, 1],
                                 all_token_ids={"y": [3, 4, y_token]}))
check("3. 重建镜像：用协议里的 all_token_ids 补回历史，行号重新分配",
      runner.requests["y"].output_token_ids == [y_token]
      and runner.input_batch.req_id_to_index["y"] == 1
      and runner.input_batch.token_ids_cpu[1, :3].tolist() == [3, 4, y_token],
      f"y 的输出={runner.requests['y'].output_token_ids}、"
      f"缓冲={runner.input_batch.token_ids_cpu[1, :3].tolist()}")

# 输出长度校正：控制端说只提交了 0 个（未提交的尾部要丢）
runner = build_runner()
runner.execute_model(make_packet(new=[new_request("z", [1, 2], [0])], num_scheduled={"z": 2}))
runner.sample_tokens()
runner.requests["z"].output_token_ids.append(999)          # 伪造一个"执行侧多算出来的"
runner.input_batch.num_tokens_no_spec[0] = 3
runner.execute_model(make_packet(cached_req_ids=["z"], num_scheduled={"z": 1},
                                 num_computed_tokens=[2], num_output_tokens=[0]))
check("3. 输出长度以协议为准：未提交的尾部被丢掉，缓冲也跟着退回",
      runner.requests["z"].output_token_ids == []
      and int(runner.input_batch.num_tokens_no_spec[0]) == 2,
      f"输出={runner.requests['z'].output_token_ids}、"
      f"缓冲长度={int(runner.input_batch.num_tokens_no_spec[0])}")

# resumed：整表替换，旧物理编号一个都不留
runner = build_runner()
runner.execute_model(make_packet(new=[new_request("r", [1, 2, 3], [11])], num_scheduled={"r": 3}))
runner.sample_tokens()
runner.execute_model(make_packet(cached_req_ids=["r"], num_scheduled={"r": 1},
                                 num_computed_tokens=[2], num_output_tokens=[1],
                                 new_block_ids=[([3, 4],)], resumed_req_ids={"r"}))
check("3. resumed：**整表替换**（不是接在 [11] 后面）",
      runner.requests["r"].block_ids == [[3, 4]]
      and runner.input_batch.block_table.cpu[0, :3].tolist() == [3, 4, 0],
      str(runner.requests["r"].block_ids))

# all_token_ids 缺失 → 协议违约，当场报错
runner = build_runner()
runner.execute_model(make_packet(new=[new_request("m", [1], [0])], num_scheduled={"m": 1}))
runner.sample_tokens()
runner.input_batch.remove_request("m")
try:
    runner.execute_model(make_packet(cached_req_ids=["m"], num_scheduled={"m": 1}))
    error = None
except RuntimeError as exc:
    error = str(exc)
check("3. 需要重建镜像但协议没带 all_token_ids → 明确报错（不静默算错）",
      error is not None and "all_token_ids" in error, (error or "").splitlines()[0])

# ------------------------------------------------ 4. 行号不是身份

runner = build_runner()
params_seeded = SamplingParams(temperature=1.0, seed=7)
states = [add_state(runner, f"q{i}", [i, i + 1], [i], params=params_seeded) for i in range(3)]
runner._make_generator_for_test = None  # noqa: B018 （仅说明 generator 由协议路径建）
for index, state in enumerate(states):
    state.generator = runner._make_generator(state.sampling_params)
    runner.input_batch.generators[index] = state.generator
moved_generator = runner.input_batch.generators[2]      # q2 的 generator
runner.input_batch.remove_request("q1")                    # 删中间那行
old_tokens = runner.input_batch.token_ids_cpu[2, :2].clone()
runner.input_batch.condense()
check("4. condense 之后行 [0, num_reqs) 全是有效行（保序压实）",
      runner.input_batch.req_ids == ["q0", "q2"] and runner.input_batch.num_reqs == 2,
      str(runner.input_batch.req_ids))
check("4. 缓冲与 generator 跟着**请求**搬家（不是跟着行号）",
      runner.input_batch.token_ids_cpu[1, :2].tolist() == old_tokens.tolist()
      and runner.input_batch.generators[1] is moved_generator
      and runner.input_batch.req_id_to_index["q2"] == 1,
      f"q2 的行={runner.input_batch.req_id_to_index['q2']}、"
      f"缓冲={runner.input_batch.token_ids_cpu[1, :2].tolist()}")
check("4. 块表也跟着搬（q2 的块号 2 搬到行 1）",
      runner.input_batch.block_table.cpu[1, 0].item() == 2
      and runner.input_batch.block_table.num_blocks(0) == 1,
      f"行1 块表={runner.input_batch.block_table.cpu[1].tolist()}")

# ------------------------------------------------ 5. 两步协议与失败停摆

runner = build_runner()
packet = make_packet(new=[new_request("s", [1, 2], [0])], num_scheduled={"s": 2})
runner.execute_model(packet)
try:
    runner.execute_model(packet)
    error = None
except RuntimeError as exc:
    error = str(exc)
check("5. 上一轮的 execute 结果没被消费就再来一轮 → 报错",
      error is not None and "sample_tokens" in error, (error or "").splitlines()[0])
runner.sample_tokens()

runner = build_runner()
try:
    runner.sample_tokens()
    error = None
except RuntimeError as exc:
    error = str(exc)
check("5. 没有 execute 就 sample → 报错（不返回上一轮的 logits）",
      error is not None and "execute" in error, (error or "").splitlines()[0])

runner = build_runner()
original_forward = runner.model.forward
runner.model.forward = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("模拟前向失败"))
try:
    runner.execute_model(make_packet(new=[new_request("f", [1, 2], [0])], num_scheduled={"f": 2}))
    error = None
except RuntimeError as exc:
    error = str(exc)
runner.model.forward = original_forward
try:
    runner.execute_model(make_packet(finished=[]))
    later = None
except RuntimeError as exc:
    later = str(exc)
check("5. 前向失败 → 这一轮的 logits 不留下，Runner 停摆（不再接受新的一轮）",
      error is not None and "模拟前向失败" in error
      and runner.execute_model_state is None and runner.failure is not None
      and later is not None and "已经失败" in later,
      f"failure={runner.failure}")

# ------------------------------------------------ 6. 块表越界

runner = build_runner()
# prompt 5 个 token、只有 1 个块（4 个槽位）→ 第 5 个 token 没有槽位
add_state(runner, "o", [1, 2, 3, 4, 5], [6], num_computed=0)
try:
    runner._prepare_inputs(make_packet(cached_req_ids=["o"], num_scheduled={"o": 5},
                                       num_computed_tokens=[0], num_output_tokens=[0]))
    error = None
except IndexError as exc:
    error = str(exc)
check("6. 块不够（协议给少了块 / positions 算错）→ 当场报错，不写到 0 号块",
      error is not None and "不在块表里" in error, (error or "").splitlines()[0])

# ------------------------------------------------ 7. Scheduler 不碰 GPU 张量

LIVE_OBJECTS = (Request, Scheduler, KVCacheManager, GPUModelRunner, CachedRequestState)


def scan_packet(value, path, problems):
    """递归找活对象与张量：协议包里两样都不该有。"""
    if isinstance(value, (torch.Tensor,)):
        problems.append(f"{path} 是 torch.Tensor")
        return
    if isinstance(value, LIVE_OBJECTS):
        problems.append(f"{path} 是活对象 {type(value).__name__}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            scan_packet(item, f"{path}.{key}", problems)
    elif isinstance(value, (list, tuple, set)):
        for index, item in enumerate(value):
            scan_packet(item, f"{path}[{index}]", problems)


class PacketSpy:
    """把 Scheduler 包一层，检查每个发出的快照。"""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.problems = []
        self.packets = 0

    def __getattr__(self, name):
        return getattr(self.scheduler, name)

    def schedule(self):
        packet = self.scheduler.schedule()
        scan_packet(packet, "packet", self.problems)
        self.packets += 1
        return packet


config = make_config(max_num_seqs=2, max_num_batched_tokens=8)
engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
spy = PacketSpy(engine.engine_core.engine_core.scheduler)
engine.engine_core.engine_core.scheduler = spy
for req_id, prompt in (("p1", [1, 2, 3]), ("p2", [4, 5, 6])):
    engine.add_request(req_id, prompt, SamplingParams(max_tokens=3, temperature=0.0))
texts = []
while engine.has_unfinished_requests():
    for output in engine.step():
        texts.append((output.request_id, len(output.token_ids), output.finished))
tensor_attrs = [name for name, value in vars(spy.scheduler).items()
                if isinstance(value, torch.Tensor)]
check("7. 端到端：真实模型 + 真 KV 走完整条链路，两条请求都跑完",
      sorted(req_id for req_id, _, finished in texts if finished) == ["p1", "p2"]
      and spy.packets >= 3,
      f"{texts}、调度 {spy.packets} 轮")
check("7. 每个 SchedulerOutput 里没有 torch.Tensor、没有 Request/Scheduler/KV 管理器这类活对象",
      not spy.problems, "；".join(spy.problems[:3]))
check("7. Scheduler 实例上也没有张量属性（GPU 缓冲都在执行侧）",
      not tensor_attrs, str(tensor_attrs))
# 推理边界：`execute_model`/`sample_tokens` 上必须有 `torch.inference_mode()`，
# 否则 KV 写入（index_copy_）会把反向图挂在缓存上，并且**一步一步累积**
kv_tensors = list(runner.kv_caches.values())
check("7. KV 缓存上没有 autograd 图（推理边界；否则显存随步数单调上涨）",
      all(not tensor.requires_grad and tensor.grad_fn is None for tensor in kv_tensors),
      f"{[tensor.grad_fn for tensor in kv_tensors][:2]}")
engine.shutdown()

# ------------------------------------------------ 8. 入口边界（prompt 装不下）

config = make_config(max_model_len=6, max_num_seqs=2, max_num_batched_tokens=8)
engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
try:
    engine.add_request("too-long", list(range(6)), SamplingParams(max_tokens=2, temperature=0.0))
    error = None
except ValueError as exc:
    error = str(exc)
check("8. prompt 长度 ≥ max_model_len 在入口就拒绝（不是等到调度器空转报错）",
      error is not None and "max_model_len" in error and "生成不出来" in error,
      (error or "").splitlines()[0])
check("8. 拒绝之后没有留下孤儿状态（同一个 ID 还能再用）",
      engine.output_processor.request_states == {},
      str(list(engine.output_processor.request_states)))

engine.add_request("fits", list(range(4)), SamplingParams(max_tokens=100, temperature=0.0))
capped = engine.engine_core.engine_core.scheduler.requests["fits"]
check("8. max_tokens 超出剩余上下文时截到装得下的量（4 + 2 = 6），不拒绝",
      capped.max_tokens == 2, f"max_tokens={capped.max_tokens}")
engine.shutdown()

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
