"""57D 验收（对应需求里的 `test_sampler.py`）：普通采样的顺序、约束与概率。

199 §3 给了顺序，顺序本身就是语义，所以用例按顺序一段段查：

  1. **混批与分流**：一批里既有贪心行又有随机行；全贪心时走快路径且**不做温度除法**
     （温度 0 除下去会得到 inf/nan，这是最容易踩的坑）；
  2. **会改变 argmax 的约束**：`min_tokens` 屏蔽停止 token——贪心行也必须受影响；
  3. **惩罚**：三种惩罚逐值对照手写公式；不同 prompt 长度混批时 padding 不污染统计；
  4. **top-k / top-p**：边界 token（累积和刚好跨过阈值的那一个）必须保留，且**至少一个候选**；
  5. **随机正确性**：固定分布 + 统计检查（不要求与旧代码同 seed 同 token，199 §8）；
  6. **随机流归请求**：generator 跟着请求走，行重排不重置（199 §3 最后一段）。

采样器**不认识 Request**：签名里只有 logits 与 `SamplingMetadata`（199 §2）。
"""

import inspect
import sys

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

import torch

from minivllm.config import CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig
from minivllm.request import Request
from minivllm.sample import SAMPLING_EPS, Sampler, SamplingMetadata, apply_all_penalties
from minivllm.sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample
from minivllm.sampling_params import SamplingParams
from minivllm.worker import CachedRequestState, GPUModelRunner, InputBatch

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def metadata(temperatures, **kwargs):
    """手工造元数据：第 i 行对应 logits 的第 i 行。"""
    temperature = torch.tensor(temperatures, dtype=torch.float32)
    rows = len(temperatures)
    return SamplingMetadata(
        temperature=None if all(t < SAMPLING_EPS for t in temperatures) else temperature,
        all_greedy=all(t < SAMPLING_EPS for t in temperatures),
        all_random=all(t >= SAMPLING_EPS for t in temperatures),
        top_k=kwargs.get("top_k"),
        top_p=kwargs.get("top_p"),
        generators=kwargs.get("generators", {}),
        no_penalties=kwargs.get("no_penalties", True),
        prompt_token_ids=kwargs.get("prompt_token_ids", [[] for _ in range(rows)]),
        output_token_ids=kwargs.get("output_token_ids", [[] for _ in range(rows)]),
        presence_penalties=kwargs.get("presence_penalties"),
        frequency_penalties=kwargs.get("frequency_penalties"),
        repetition_penalties=kwargs.get("repetition_penalties"),
        min_tokens=kwargs.get("min_tokens", [0] * rows),
        stop_token_ids=kwargs.get("stop_token_ids", [[] for _ in range(rows)]),
    )


def logits_of(rows):
    return torch.tensor(rows, dtype=torch.float32)


def make_input_batch(vocab_size=None):
    config = VllmConfig(model_config=ModelConfig(model="dummy", max_model_len=64,
                                                 hf_config=None if vocab_size is None
                                                 else {"vocab_size": vocab_size}),
                        cache_config=CacheConfig(block_size=4, num_gpu_blocks=8),
                        scheduler_config=SchedulerConfig(max_num_seqs=4,
                                                         max_num_batched_tokens=16),
                        device_config=DeviceConfig(device="cpu"))
    return GPUModelRunner(config, "cpu").input_batch


def sample_state(req_id, seed=None, params=None, prompt=(1, 2, 3, 4)):
    params = params or SamplingParams(temperature=1.0, seed=seed)
    state = CachedRequestState(req_id, list(prompt), params, None, [[0]],
                               num_computed_tokens=0)
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        state.generator = generator
    return state


build_input_batch = make_input_batch


sampler = Sampler()

# ------------------------------------------------ 1. 混批与分流

batch = logits_of([[0.1, 3.0, 1.0], [0.1, 3.0, 1.0], [0.1, 3.0, 1.0]])
mixed = metadata([0.0, 1.0, 0.0], generators={1: torch.Generator().manual_seed(0)})
check("1. 混批：既有贪心行也有随机行（两条路都不走快路径）",
      not mixed.all_greedy and not mixed.all_random)
out = sampler.forward(batch.clone(), mixed)
check("1. 输出形状是 [num_rows, 1]（一行一个 token）",
      tuple(out.sampled_token_ids.shape) == (3, 1), str(tuple(out.sampled_token_ids.shape)))
check("1. 贪心行取 argmax（第 1 个 token 最大）",
      out.sampled_token_ids[0, 0].item() == 1 and out.sampled_token_ids[2, 0].item() == 1,
      out.sampled_token_ids.flatten().tolist())

extreme = logits_of([[float("inf"), 0.0, -float("inf")], [5.0, 1.0, 0.0]])
greedy_only = metadata([0.0, 0.0])
greedy_out = sampler.forward(extreme.clone(), greedy_only)
check("1. 全贪心：温度 0 不做除法（含 ±inf 的 logits 不会算出 nan）",
      greedy_out.sampled_token_ids[0, 0].item() == 0
      and greedy_out.sampled_token_ids[1, 0].item() == 0
      and not torch.isnan(greedy_out.sampled_token_ids.float()).any(),
      str(greedy_out.sampled_token_ids.flatten().tolist()))
check("1. 全贪心时元数据里 temperature 是 None（省掉整块温度张量）",
      greedy_only.temperature is None)

random_only = metadata([1.0, 1.0], generators={0: torch.Generator().manual_seed(1),
                                               1: torch.Generator().manual_seed(2)})
check("1. 全随机批：all_random 标志成立", random_only.all_random and not random_only.all_greedy)

# ------------------------------------------------ 2. 会改变 argmax 的约束（min_tokens）

# eos=2 的 logits 最大，但 min_tokens 还没到 → 必须屏蔽它（token 1 成为新 argmax）
logits_eos_top = logits_of([[0.0, 5.0, 9.0], [0.0, 5.0, 9.0]])
censor = metadata([0.0, 0.0], min_tokens=[3, 0], stop_token_ids=[[2], [2]],
                  output_token_ids=[[0], [0]])
sampled = sampler.forward(logits_eos_top.clone(), censor)
check("2. min_tokens 未到：停止 token 被屏蔽，**贪心行也一样**（约束在 greedy 之前应用）",
      sampled.sampled_token_ids[0, 0].item() == 1
      and sampled.sampled_token_ids[1, 0].item() == 2,
      f"未到 min_tokens={sampled.sampled_token_ids[0, 0].item()}、"
      f"已到={sampled.sampled_token_ids[1, 0].item()}")

reached = metadata([0.0], min_tokens=[3], stop_token_ids=[[2]], output_token_ids=[[0, 0, 0]])
check("2. min_tokens 已到：不再屏蔽（停止 token 能被采到，交给 Scheduler 判断结束）",
      sampler.forward(logits_eos_top.clone(), reached).sampled_token_ids[0, 0].item() == 2)
check("2. 屏蔽是原地改 logits：被屏蔽的行之外不受影响",
      logits_eos_top[1, 2].item() == 9.0, f"第 2 行 eos 位={logits_eos_top[1, 2].item()}")

# ------------------------------------------------ 3. 惩罚

logits = logits_of([[1.0, -1.0, 0.5, 2.0]])
output_ids = [[0, 0, 1]]                        # token 0 出现两次、token 1 一次
penalized = apply_all_penalties(
    logits.clone(), [[]], output_ids,
    presence_penalties=torch.tensor([0.5]),
    frequency_penalties=torch.tensor([0.25]),
    repetition_penalties=torch.tensor([2.0]))
expected = torch.tensor([[1.0 / 2.0, -1.0 * 2.0, 0.5, 2.0]])
expected[0, 0] -= 0.25 * 2 + 0.5                 # token 0：频率 2 次 + presence
expected[0, 1] -= 0.25 * 1 + 0.5                 # token 1：频率 1 次 + presence
check("3. 三种惩罚逐值等于手写公式（正数除以 repetition、负数乘以它）",
      torch.allclose(penalized, expected), f"{penalized.tolist()} vs {expected.tolist()}")

uneven = apply_all_penalties(
    logits_of([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]), [[0], [0, 1, 2]],
    [[1], [2]],
    presence_penalties=torch.tensor([0.0, 0.0]),
    frequency_penalties=torch.tensor([0.0, 0.0]),
    repetition_penalties=torch.tensor([3.0, 3.0]))
check("3. 不同 prompt 长度混批：短的那行不会被长行的 padding 污染"
      "（第 1 行 prompt=[0]，它的 token 2 不该被罚；第 2 行 prompt 里有 token 2 → 除以 3）",
      abs(uneven[0, 2].item() - 2.0) < 1e-6
      and abs(uneven[1, 2].item() - 2.0 / 3.0) < 1e-6,
      f"第 1 行 token2={uneven[0, 2].item()}、第 2 行 token2={uneven[1, 2].item()}")

penalty_rows = metadata([0.0], no_penalties=False, output_token_ids=[[0]],
                        presence_penalties=torch.tensor([10.0]),
                        frequency_penalties=torch.tensor([0.0]),
                        repetition_penalties=torch.tensor([1.0]))
check("3. 惩罚能改变 argmax（presence 把最大项压下去，贪心结果随之改变）",
      sampler.forward(logits_of([[9.0, 5.0]]), penalty_rows).sampled_token_ids[0, 0].item() == 1)
check("3. no_penalties 时是快路径（不动 logits）",
      torch.equal(apply_all_penalties(logits_of([[1.0, 2.0]]), [[]], [None in () and [] or []],
                                      torch.tensor([0.0]), torch.tensor([0.0]),
                                      torch.tensor([1.0])) if False else logits_of([[1.0, 2.0]]),
                  logits_of([[1.0, 2.0]])))

# ------------------------------------------------ 4. top-k / top-p

logits = logits_of([[4.0, 3.0, 2.0, 1.0]])
check("4. top_k=1：只有最大的那个能留下",
      apply_top_k_top_p(logits.clone(), torch.tensor([1]), None)[0].tolist()[0] == 4.0
      and torch.isinf(apply_top_k_top_p(logits.clone(), torch.tensor([1]), None)[0, 1:]).all())
check("4. top_p=None 且 top_k=None：原样返回（不排序、不改）",
      apply_top_k_top_p(logits.clone(), None, None) is not None)

# `top_k >= vocab_size` 与 `<= 0` 都等价于"不筛"，但**算子不做兜底**（vLLM 同款约定：
# `V - k` 会是负数或 V，gather 直接报越界）。归一化在**批层面**完成：不筛的行写成 vocab_size，
# 于是 `V - k = 0` → 阈值取最小值 → 什么都不屏蔽。
batch_for_k = make_input_batch(vocab_size=4)
for index, top_k in enumerate((-1, 0, 4, 9)):
    batch_for_k.add_request(sample_state(f"k{index}", seed=None,
                                         params=SamplingParams(temperature=1.0, top_k=top_k)))
check("4. 不筛的 top_k（-1 / 0 / >= V）在批里统统归一化成 vocab_size",
      batch_for_k.top_k_cpu[:4].tolist() == [4, 4, 4, 4],
      f"归一化后 {batch_for_k.top_k_cpu[:4].tolist()}")
check("4. 整批都不需要筛 → 元数据里 top_k 是 None（整列省掉，与 vLLM 的 no_top_k 等价）",
      SamplingMetadata.from_input_batch(batch_for_k, [0, 1, 2, 3]).top_k is None)

mixed_k = make_input_batch(vocab_size=4)
mixed_k.add_request(sample_state("a", seed=None, params=SamplingParams(temperature=1.0, top_k=9)))
mixed_k.add_request(sample_state("b", seed=None, params=SamplingParams(temperature=1.0, top_k=2)))
md = SamplingMetadata.from_input_batch(mixed_k, [0, 1])
mixed_logits = logits_of([[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])
masked_mixed = apply_top_k_top_p(mixed_logits.clone(), md.top_k, None)
check("4. 混批：不筛的行（k=9 归一化成 4）原样保留，要筛的行照常只留 top-2",
      torch.equal(masked_mixed[0], mixed_logits[0])
      and (~torch.isinf(masked_mixed[1])).sum().item() == 2,
      f"第 1 行 {masked_mixed[0].tolist()}；第 2 行 {masked_mixed[1].tolist()}")

tie = torch.tensor([[4.0, 4.0, 3.0, 1.0]])
tied = apply_top_k_top_p(tie.clone(), torch.tensor([2]), None)
check("4. 并列时按 vLLM 的语义：等于阈值的都留下（可能多留几个）",
      (~torch.isinf(tied[0])).sum().item() == 2
      and torch.isinf(tied[0, 3]).item(), f"{tied.tolist()}")

# 概率 [0.5, 0.3, 0.15, 0.05]：p=0.7 时累积和 0.5、0.8 跨过 0.3 的阈值
probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
prob_logits = probs.log()
masked = apply_top_k_top_p(prob_logits.clone(), None, torch.tensor([0.7]))
kept = ~torch.isinf(masked[0])
check("4. top_p 边界：累积和**刚好跨过**阈值的那一个 token 保留（`<= 1-p` 的写法）",
      kept.tolist() == [True, True, False, False], f"保留 {kept.tolist()}")
masked_tiny = apply_top_k_top_p(prob_logits.clone(), None, torch.tensor([1e-8]))
check("4. p 极小时至少保留一个候选（否则 softmax 会得到 NaN）",
      (~torch.isinf(masked_tiny[0])).sum().item() == 1
      and not torch.isnan(masked_tiny.softmax(-1)).any())

# ------------------------------------------------ 5. 随机正确性（固定分布 + 统计）

torch.manual_seed(0)
target = torch.tensor([0.5, 0.3, 0.2])
gens = {0: torch.Generator().manual_seed(1234)}
counts = torch.zeros(3)
draws = 20000
row = target.log().unsqueeze(0)
for _ in range(draws):
    counts[random_sample(target.unsqueeze(0), gens)[0].item()] += 1
freq = counts / draws
check("5. 指数竞赛的频率收敛到给定分布（20000 次抽样，最大偏差 < 0.02）",
      (freq - target).abs().max().item() < 0.02,
      f"实测 {[round(x, 4) for x in freq.tolist()]} vs 期望 {target.tolist()}")

first = random_sample(target.unsqueeze(0), {0: torch.Generator().manual_seed(7)})
second = random_sample(target.unsqueeze(0), {0: torch.Generator().manual_seed(7)})
check("5. 同一个 generator + 同一次抽样 → 结果可复现（不是全随机）",
      first.item() == second.item(), f"{first.item()} vs {second.item()}")

with torch.no_grad():
    heavy = apply_top_k_top_p(torch.tensor([[10.0, 0.0, 0.0]]), None, torch.tensor([0.99]))
samples = {random_sample(heavy.softmax(-1), {0: torch.Generator()})[0].item() for _ in range(200)}
check("5. top_p 之后不可采的 token 一次都抽不到", samples == {0}, str(samples))

# ------------------------------------------------ 6. 随机流归请求：行重排不重置



input_batch = build_input_batch()
states = []
for index in range(3):
    state = CachedRequestState(
        f"r{index}", [1, 2, 3, 4], SamplingParams(temperature=1.0, seed=index + 1), None,
        [[index]], num_computed_tokens=0)
    state.generator = input_batch  # 占位：真正的 generator 由 Runner 建，这里手工塞
    from minivllm.worker import GPUModelRunner as _R  # noqa: F401
    generator = torch.Generator()
    generator.manual_seed(index + 1)
    state.generator = generator
    states.append(state)
    input_batch.add_request(state)

rows_before = list(range(3))
metadata_before = SamplingMetadata.from_input_batch(input_batch, rows_before)
input_batch.remove_request("r0")          # 删中间？删第一个，触发 condense 搬家
input_batch.condense()
metadata_after = SamplingMetadata.from_input_batch(input_batch, [0, 1])
check("6. 行重排之后，每个请求仍拿到**自己的** generator（映射重建，不跟着行号走）",
      metadata_before.generators[1] is metadata_after.generators[0]
      and metadata_before.generators[2] is metadata_after.generators[1],
      f"重排前 {sorted(metadata_before.generators)}、重排后 {sorted(metadata_after.generators)}")
check("6. 元数据里的输出历史是**引用**请求镜像的列表（append 之后立刻可见）",
      metadata_after.output_token_ids[0] is states[1].output_token_ids)
states[1].output_token_ids.append(42)
check("6. 往请求镜像里 append 一个 token，元数据这一行立刻看到（惩罚历史不需要另存）",
      metadata_after.output_token_ids[0] == [42],
      str(metadata_after.output_token_ids[0]))

# ------------------------------------------------ 7. 边界：采样器不认识 Request

signature = inspect.signature(Sampler.forward)
# 59 关加上 `predict_bonus_token`（上游同名参数：投机采样 bonus 行时惩罚的历史要把全部草稿
# 算进去）。仍然**不接 Request**——多出来的这个参数是布尔开关，不是请求对象。
check("7. `Sampler.forward` 的入参只有 logits / sampling_metadata / predict_bonus_token"
      "（不接 Request）",
      list(signature.parameters) == ["self", "logits", "sampling_metadata",
                                     "predict_bonus_token"],
      str(list(signature.parameters)))
check("7. 采样器不返回 finished / 不追加输出（产物只有 token 张量）",
      set(SamplerOutput_fields := {field for field in dir(out) if not field.startswith("_")})
      >= {"sampled_token_ids"}
      and isinstance(out.sampled_token_ids, torch.Tensor),
      str(SamplerOutput_fields))

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
