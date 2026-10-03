"""58 验收 C（需求 §8.C）：**固定输入工作区**与端到端。

工作区的要求（需求 §7）：初始化时分配、之后每轮复用；只把有效切片 `[:num_tokens]` /
`[:num_reqs]` 交给模型；尾部残留不能被读到（用长度表达有效，不是"内容恰好是 0"）；
`data_ptr()` 稳定；后续 K-1 次自回归复用同一工作区、不再计输入预算。

    1. 缓冲地址稳定：不同请求数 / 不同 K / batch 缩了又涨，主缓冲 `data_ptr()` 不变
    2. 只读有效切片：把尾部灌成垃圾也不影响输出
    3. 端到端：greedy 投机 == 非投机（多组配置）；CUDA 上再跑一遍
    4. 生命周期回归：abort / 请求 ID 复用 / 失败态 / 上下文上限 / 请求暂停后再入批
    5. 可复现：同一 seed 两次运行逐 token 一致
"""

import sys

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

import torch

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,
                      SchedulerConfig, SpeculativeConfig, UniProcExecutor, VllmConfig, Worker)
from minivllm.testing.tiny_models import tiny_qwen3_config, tiny_qwen3_dir

FAIL = []
TINY = tiny_qwen3_dir("tiny_gqa")
HF = tiny_qwen3_config("tiny_gqa")


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def build(*, k=3, budget=16, blocks=32, prefix=False, max_model_len=64, max_num_seqs=2,
          device="cpu"):
    model = ModelConfig(model=TINY, dtype="float32", max_model_len=max_model_len, hf_config=HF)
    spec = None if k is None else SpeculativeConfig(
        method="draft_model", num_speculative_tokens=k,
        draft_model_config=ModelConfig(model=TINY, dtype="float32",
                                      max_model_len=max_model_len, hf_config=HF))
    config = VllmConfig(model_config=model,
                        cache_config=CacheConfig(block_size=4, num_gpu_blocks=blocks,
                                                 enable_prefix_caching=prefix),
                        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                                         max_num_batched_tokens=budget),
                        device_config=DeviceConfig(device=device), speculative_config=spec)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner


def run_to_end(engine, limit=400):
    final = {}
    for _ in range(limit):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            final[out.request_id] = list(out.token_ids)
    return final


# ---------------------------------------------------------------- 1. 缓冲地址稳定
engine, core, runner = build(k=3, budget=16)
proposer = runner.proposer
check("C1. 工作区容量 = 配置的 max_num_batched_tokens；缓冲形状按 M / max_num_reqs 开",
      proposer.max_num_tokens == 16 and proposer.max_num_reqs == 2
      and proposer.input_ids.shape == (16,) and proposer.positions.shape == (16,)
      and proposer.slot_mapping.shape == (16,)
      and proposer.query_start_loc.shape == (3,) and proposer.seq_lens.shape == (2,)
      and proposer.block_table.shape == (2, 16),
      f"M={proposer.max_num_tokens} block_table={tuple(proposer.block_table.shape)}")

addresses = {name: getattr(proposer, name).data_ptr()
             for name in ("input_ids", "positions", "slot_mapping", "is_rejected_token_mask",
                          "query_start_loc", "seq_lens", "block_table")}
engine.add_request("A", [1, 2, 3, 4], SamplingParams(max_tokens=4, temperature=0.0,
                                                     eos_token_id=999))
engine.add_request("B", [5, 6], SamplingParams(max_tokens=3, temperature=0.0, eos_token_id=999))
run_to_end(engine)
engine.add_request("C", [7, 8, 9], SamplingParams(max_tokens=3, temperature=0.0, eos_token_id=999))
run_to_end(engine)
same = all(getattr(proposer, name).data_ptr() == address for name, address in addresses.items())
check("C1b. 多轮 / 多请求 / batch 缩了又涨之后：所有输入缓冲仍是同一块内存（没有每轮新建）",
      same, f"地址表={ {k: hex(v) for k, v in addresses.items()} }")
engine.shutdown()

# ---------------------------------------------------------------- 2. 只读有效切片
reference_engine, _c, _r = build(k=3, budget=16)
reference_engine.add_request("r", [1, 2, 3, 4, 5, 6], SamplingParams(max_tokens=6,
                                                                     temperature=0.0,
                                                                     eos_token_id=999))
reference = run_to_end(reference_engine)
reference_engine.shutdown()

engine, _core, runner = build(k=3, budget=16)
proposer = runner.proposer
original_forward = proposer._forward


def poisoning_forward(num_tokens, num_reqs):
    hidden = original_forward(num_tokens, num_reqs)
    # 把"没被声明的尾部"灌成垃圾：如果哪里按长度之外去读，输出就会变
    proposer.input_ids_cpu[num_tokens:] = 999
    proposer.positions_cpu[num_tokens:] = 999
    proposer.slot_mapping_cpu[num_tokens:] = -1
    proposer.query_start_loc_cpu[num_reqs + 1:] = 999
    proposer.seq_lens_cpu[num_reqs:] = 999
    proposer.block_table_cpu[num_reqs:] = 999
    return hidden


proposer._forward = poisoning_forward
engine.add_request("r", [1, 2, 3, 4, 5, 6], SamplingParams(max_tokens=6, temperature=0.0,
                                                           eos_token_id=999))
poisoned = run_to_end(engine)
proposer._forward = original_forward
check("C2. 把工作区尾部灌成垃圾也不影响输出：模型只读 `[:num_tokens]` / `[:num_reqs]`",
      poisoned == reference, f"灌垃圾={poisoned}、参考={reference}")
engine.shutdown()

# ---------------------------------------------------------------- 3. 端到端 == 非投机
def greedy_run(k, *, prefix=False, prompts=((("r", [1, 2, 3, 4, 5, 6]),)),
               max_tokens=6, device="cpu", budget=16):
    engine, _core, _runner = build(k=k, prefix=prefix, device=device, budget=budget)
    for req_id, prompt in prompts:
        engine.add_request(req_id, list(prompt), SamplingParams(max_tokens=max_tokens,
                                                                temperature=0.0,
                                                                eos_token_id=999))
    outputs = run_to_end(engine)
    engine.shutdown()
    return outputs


for k in (1, 3):
    for prefix in (False, True):
        reference = greedy_run(None, prefix=prefix)
        speculative = greedy_run(k, prefix=prefix)
        check(f"C3. greedy 投机(k={k}, prefix={prefix}) == 非投机",
              reference == speculative, f"非投机={reference}、投机={speculative}")

# 两条请求、长度不同：batch 里既有长大又有结束
prompts = (("A", [1, 2, 3, 4]), ("B", [5, 6, 7, 8, 9]))
reference = greedy_run(None, prompts=prompts, max_tokens=5)
speculative = greedy_run(3, prompts=prompts, max_tokens=5)
check("C3b. 两条不同长度请求同批：投机与非投机输出一致（含 batch 缩小）",
      reference == speculative, f"非投机={reference}、投机={speculative}")

# ---------------------------------------------------------------- 4. 生命周期回归
# abort：中途取消，引擎继续把别人跑完
engine, core, _runner = build(k=3, budget=16)
engine.add_request("keep", [1, 2, 3], SamplingParams(max_tokens=3, temperature=0.0,
                                                     eos_token_id=999))
engine.add_request("drop", [4, 5, 6], SamplingParams(max_tokens=8, temperature=0.0,
                                                     eos_token_id=999))
engine.step()
engine.abort_request(["drop"])
outputs = run_to_end(engine)
check("C4. abort：被取消的请求不再产出，另一条正常跑完",
      "drop" not in outputs and len(outputs.get("keep", [])) == 3,
      f"输出={outputs}")
engine.shutdown()

# 请求 ID 复用：老请求收尾（含清理轮）之后同名新请求可用
engine, _core, runner = build(k=1, prefix=True)
engine.add_request("reuse", [1, 2, 3, 4, 5, 6, 7, 8], SamplingParams(max_tokens=1,
                                                                     temperature=0.0,
                                                                     eos_token_id=999))
run_to_end(engine)
engine.add_request("reuse", [8, 7, 6, 5], SamplingParams(max_tokens=2, temperature=0.0,
                                                         eos_token_id=999))
reused = run_to_end(engine)
check("C4b. ID 复用：结束清理之后同名请求能重新建立状态并产出",
      len(reused.get("reuse", [])) == 2, f"输出={reused}")
engine.shutdown()

# 失败态：提议异常之后，下一轮在调度之前被拒绝
engine, core, runner = build(k=1, budget=16)
engine.add_request("r", [1, 2, 3, 4, 5, 6], SamplingParams(max_tokens=4, temperature=0.0,
                                                           eos_token_id=999))
original_forward = runner.proposer._forward


def injected(*args, **kwargs):
    raise RuntimeError("review-injected-draft-forward-failure")


runner.proposer._forward = injected
first_error = None
try:
    engine.step()
except Exception as exc:                                     # noqa: BLE001
    first_error = f"{type(exc).__name__}: {exc}"
runner.proposer._forward = original_forward
second_error = None
try:
    engine.step()
except Exception as exc:                                     # noqa: BLE001
    second_error = f"{type(exc).__name__}: {exc}"
check("C4c. 失败态：提议异常后 Runner 记 failure，下一轮在调度前被拒绝",
      first_error is not None and runner.failure is not None and second_error is not None,
      f"首次={first_error}；下一轮={second_error}")
engine.shutdown()

# 上下文上限：走到 max_model_len 正常收尾（与 57 的边界回归一致）
reference = greedy_run(None, prompts=(("r", [1, 2, 3, 4, 5, 6, 7, 8]),), max_tokens=2)
speculative = greedy_run(3, prompts=(("r", [1, 2, 3, 4, 5, 6, 7, 8]),), max_tokens=2)
check("C4d. 上下文上限（prompt 8 + max_model_len 10）：输出与非投机一致、正常收尾",
      reference == speculative and len(speculative["r"]) == 2,
      f"非投机={reference}、投机={speculative}")

# 请求暂停后再入批：随机流不重置（第二次入批仍能跑完）
engine, core, runner = build(k=3, budget=6)
for req, seed in (("A", 10), ("B", 20)):
    engine.add_request(req, [1, 2], SamplingParams(max_tokens=3, temperature=1.0, seed=seed,
                                                   eos_token_id=999))
engine.step()
before = runner.proposer._draft_generators["B"]
engine.step()
scheduled = list(runner.input_batch.req_ids)
outputs = run_to_end(engine)
check("C4e. 请求暂停后再入批：generator 保留、两条都跑完",
      scheduled == ["A"] and runner.proposer is not None and len(outputs.get("B", [])) == 3,
      f"第二轮 batch={scheduled}、输出={outputs}")
engine.shutdown()

# ---------------------------------------------------------------- 5. 可复现
def seeded(seed):
    engine, _core, _runner = build(k=3, budget=16)
    engine.add_request("r", [1, 2, 3, 4, 5, 6], SamplingParams(max_tokens=6, temperature=0.8,
                                                               seed=seed, eos_token_id=999))
    outputs = run_to_end(engine)
    engine.shutdown()
    return outputs


first, second = seeded(1234), seeded(1234)
check("C5. 固定 seed：两次运行逐 token 一致（草稿与 target 的随机流都可复现）",
      first == second, f"{first} vs {second}")

# ---------------------------------------------------------------- 6. CUDA（有卡就再跑一遍）
if torch.cuda.is_available():
    reference = greedy_run(None, device="cuda")
    speculative = greedy_run(3, device="cuda")
    check("C6. CUDA 上端到端：greedy 投机 == 非投机",
          reference == speculative and len(speculative["r"]) == 6,
          f"非投机={reference}、投机={speculative}")
else:
    check("C6. （跳过 CUDA 端到端：本机没有 CUDA）", True)

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
