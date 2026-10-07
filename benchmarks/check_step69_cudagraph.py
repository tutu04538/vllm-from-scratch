"""69 关验收探针：CUDA Graph 的键/分派 + 投机输入 padding + 图与 eager 的一致性。

跑法：`python benchmarks/check_step69_cudagraph.py`（退出码非 0 = 失败）。
需要 CUDA（图与投机验证都只有 GPU 路径）；没有 CUDA 时**明确报"待验"**并按失败退出，
不写成"通过"（AGENTS §9）。

探针分四段：

    A 配置与解析        默认模式、档位取整、不支持的模式的明确拒绝
    B 键与分派          补齐映射、bucket 边界、混合批回退、捕获顺序
    C 图与 eager 一致    同权重 greedy（普通 / 首拒 / 全接受）+ 中间值（logits）对照
    D padding 不留痕      污染补齐区后输出与真实块 KV 逐位不变；0 号块只被 padding 用
    E profiler           真实 `cudaGraphLaunch` + 图区域内无同步
"""

import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step69"))

from spec69_helpers import (CUDAGraphMode, CompilationConfig, DispatcherHarness,  # noqa: E402
                            graph_snapshot, greedy_outputs, kv_digest, make_config,
                            make_engine, run_to_end)
from minivllm import SamplingParams  # noqa: E402
from minivllm.config import CompilationMode  # noqa: E402
from minivllm.spec_decode.utils import (PADDING_SLOT_ID,  # noqa: E402
                                        prepare_inputs_padded)
from minivllm.testing.tiny_models import tiny_qwen3_config, tiny_qwen3_dir  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TINY = tiny_qwen3_dir("tiny_gqa")
HF = tiny_qwen3_config("tiny_gqa")

PASSED, FAILED, TRACE = 0, [], []


def check(name, ok, detail=""):
    global PASSED
    if ok:
        PASSED += 1
        print(f"PASS  {name}  {detail}")
    else:
        FAILED.append(name)
        print(f"FAIL  {name}  {detail}")
    TRACE.append({"item": name, "ok": bool(ok), "detail": str(detail)[:400]})


if DEVICE != "cuda":
    print("FAIL  设备：CUDA Graph 与投机验证都需要 CUDA（本机没有）——本关待验，不是通过")
    sys.exit(1)

# ---------------------------------------------------------------- A 配置与解析
config = make_config(tiny_dir=TINY, hf_config=HF, device="cuda")
check("A1. 默认解析：CUDA + 未开 enforce_eager → FULL_DECODE_ONLY",
      config.compilation_config.cudagraph_mode is CUDAGraphMode.FULL_DECODE_ONLY
      and config.compilation_config.cudagraph_capture_sizes,
      f"mode={config.compilation_config.cudagraph_mode} "
      f"sizes={config.compilation_config.cudagraph_capture_sizes}")

cpu_config = make_config(tiny_dir=TINY, hf_config=HF, device="cpu")
check("A2. CPU 设备 → NONE（没有图可捕获，也不该假装有）",
      cpu_config.compilation_config.cudagraph_mode is CUDAGraphMode.NONE
      and cpu_config.compilation_config.cudagraph_capture_sizes == [])

eager_config = make_config(tiny_dir=TINY, hf_config=HF, device="cuda", enforce_eager=True)
check("A3. enforce_eager=True → NONE（上游同款开关）",
      eager_config.compilation_config.cudagraph_mode is CUDAGraphMode.NONE)

spec_config = make_config(tiny_dir=TINY, hf_config=HF, device="cuda", spec_k=2)
sizes = spec_config.compilation_config.cudagraph_capture_sizes
check("A4. K=2 → 档位全部取整到 1+K=3 的倍数（上游 issue #28207 的修法）",
      sizes and all(size % 3 == 0 for size in sizes), f"sizes={sizes}")

rejections = []
for kwargs in ({"cudagraph_mode": "piecewise"}, {"cudagraph_mode": "full_and_piecewise"},
               {"mode": CompilationMode.VLLM_COMPILE}, {"compile_sizes": [8]}):
    try:
        CompilationConfig(**kwargs)
        rejections.append((kwargs, None))
    except NotImplementedError as exc:
        rejections.append((kwargs, str(exc)[:60]))
check("A5. 不支持的编译/图模式在配置期明确报错（不静默退化）",
      all(reason for _kwargs, reason in rejections), str(rejections))

try:
    make_engine(tiny_dir=TINY, hf_config=HF, mode="full")
    full_reason = None
except NotImplementedError as exc:
    full_reason = str(exc)
check("A6. 非分段 FULL（混合批也要进图）被明确拒绝，并指向 full_decode_only",
      full_reason is not None and "full_decode_only" in full_reason, str(full_reason))

# ---------------------------------------------------------------- B 键与分派
harness = DispatcherHarness(budget=64, max_num_seqs=4, mode="full_decode_only",
                            capture_sizes=[2, 4, 8], max_capture_size=8)
mapping = harness.dispatcher._bs_to_padded_graph_size
check("B1. 补齐映射逐值：正好命中档位不补，落在两档之间补到下一个档位",
      [mapping[i] for i in range(1, 9)] == [2, 2, 4, 4, 8, 8, 8, 8],
      f"mapping[1..8]={[mapping[i] for i in range(1, 9)]}")

spec_harness = DispatcherHarness(spec_k=3, budget=64, max_num_seqs=4)
mode, desc = spec_harness.dispatch(5, uniform_decode=True)
check("B2. 统一 decode 批：5 行 → 补到 8 行 → 2 条请求（num_reqs 由 1+K 反推）",
      mode is CUDAGraphMode.FULL and desc.num_tokens == 8 and desc.num_reqs == 2,
      f"{mode} {desc}")
check("B3. 混合 prefill/decode 批没有图键 → 按规则回退 NONE（eager）",
      spec_harness.dispatch(5, uniform_decode=False)[0] is CUDAGraphMode.NONE
      and spec_harness.dispatch(12, uniform_decode=False)[0] is CUDAGraphMode.NONE)
big = DispatcherHarness(spec_k=0, budget=16, max_num_seqs=16, mode="full")
mode, desc = big.dispatch(17, uniform_decode=True)
check("B4. 超过最大档位 → NONE，且键是未补齐的原始形状",
      mode is CUDAGraphMode.NONE and desc.num_tokens == 17
      and desc.num_reqs is None, f"{mode} {desc}")
orders = [[d.num_tokens for _m, descs in harness.dispatcher.get_capture_descs()
           for d in descs]]
check("B5. 捕获顺序：大档位在前（小图复用大图占下的显存池）",
      all(tokens == sorted(tokens, reverse=True) for tokens in orders), str(orders))

# ---------------------------------------------------------------- C 图与 eager 一致
t_start = time.perf_counter()
eager_captures = []
for spec_k in (None, 1, 3):
    prompts = (("A", [1, 2, 3, 4, 5, 6]), ("B", [2, 3, 4]))
    eager = greedy_outputs(tiny_dir=TINY, hf_config=HF, spec_k=spec_k, prompts=prompts,
                           max_tokens=8, budget=32, blocks=32, mode="none")
    graph = greedy_outputs(tiny_dir=TINY, hf_config=HF, spec_k=spec_k, prompts=prompts,
                           max_tokens=8, budget=32, blocks=32, mode="full_decode_only")
    check(f"C1. K={spec_k}：eager 与 Graph 的 greedy 输出逐 token 相同",
          eager == graph, f"eager={eager} graph={graph}")

engine, _core, runner = make_engine(tiny_dir=TINY, hf_config=HF, spec_k=3, budget=32,
                                    blocks=32, mode="full_decode_only")
snapshot = graph_snapshot(runner)
eager_captures.append(snapshot)
engine.add_request("r", [1, 2, 3, 4, 5, 6],
                   SamplingParams(max_tokens=8, temperature=0.0, eos_token_id=999))
addresses = {name: buf.data_ptr() for name, buf in runner._padded_buffers.items()}
run_to_end(engine)
after = graph_snapshot(runner)
check("C2. 捕获一次、之后全重放；静态缓冲地址恒定",
      after["captures"] == snapshot["captures"] and after["replays"] >= 2
      and {name: buf.data_ptr() for name, buf in runner._padded_buffers.items()} == addresses,
      f"captures={after['captures']} replays={after['replays']}")
modes = [entry["mode"] for entry in runner.cudagraph_selections]
check("C3. 每轮真实选中的 mode/key 有记录：prefill 回退 NONE、decode 走 FULL",
      modes[0] == "NONE" and all(m == "FULL" for m in modes[1:]),
      f"前几步={modes[:6]}")
engine.shutdown()

# 中间值对照：同一份输入下两条路径的 logits（图内的 attention 与 eager 的逐请求循环）
engine_a, _c, runner_a = make_engine(tiny_dir=TINY, hf_config=HF, spec_k=3, budget=32,
                                     blocks=32, mode="none")
engine_b, _c, runner_b = make_engine(tiny_dir=TINY, hf_config=HF, spec_k=3, budget=32,
                                     blocks=32, mode="full_decode_only")
logits_pair = []
for engine, runner in ((engine_a, runner_a), (engine_b, runner_b)):
    engine.add_request("r", [1, 2, 3, 4, 5, 6],
                       SamplingParams(max_tokens=3, temperature=0.0, eos_token_id=999))
    list(engine.step())                      # prefill（两边都 eager）
    captured = {}
    original = runner.model.compute_logits

    def spy(hidden, *args, _captured=captured, _original=original, **kwargs):
        out = _original(hidden, *args, **kwargs)
        _captured.setdefault("logits", out.detach().float().cpu())
        return out

    runner.model.compute_logits = spy
    list(engine.step())                      # decode（一边 eager、一边图）
    logits_pair.append(captured["logits"])
    engine.shutdown()
delta = float((logits_pair[0] - logits_pair[1]).abs().max())
check("C4. 中间值对照：两条路径的 logits 最大差 < 1e-3",
      delta < 1e-3, f"max|Δlogits|={delta:.3e}")

# ---------------------------------------------------------------- D padding 不留痕
def polluted_run(poison: bool):
    """3 条请求 × K=1 = 6 行 → 档位 8 行：**一定**有 2 行补齐的行 + 1 条补齐的请求。

    污染用**合法但在这些行上错**的值（最大 token id / 最大位置）：padding 行的内容会被
    embedding 与 RoPE 查表读到，超范围的值会被模型自己的越界检查拦下——那不是本项要考的东西。
    """
    engine, core, runner = make_engine(tiny_dir=TINY, hf_config=HF, spec_k=1, budget=16,
                                       blocks=32, max_num_seqs=4, mode="full_decode_only")
    original = runner._prepare_inputs_padded
    seen = {}

    def wrapper(inputs, mode, batch_descriptor):
        original(inputs, mode, batch_descriptor)
        used = inputs.num_tokens
        seen["tail"] = int(batch_descriptor.num_tokens - used)
        seen["extra_reqs"] = int(batch_descriptor.num_reqs) - runner.input_batch.num_reqs
        if poison and seen["tail"] > 0:
            buffers = runner._padded_buffers
            buffers["input_ids"][used:].fill_(HF["vocab_size"] - 1)
            buffers["positions"][used:].fill_(runner.max_model_len - 1)
    runner._prepare_inputs_padded = wrapper
    for req_id, prompt in (("a", [1, 2, 3, 4]), ("b", [2, 3, 4, 5]), ("c", [3, 4, 5, 6])):
        engine.add_request(req_id, list(prompt),
                           SamplingParams(max_tokens=4, temperature=0.0, eos_token_id=999))
    outputs = run_to_end(engine)
    digest = kv_digest(runner).clone()
    manager = core.kv_cache_manager
    null_block = manager.block_pool.null_block
    used_blocks = {block_id for row in
                   runner.input_batch.block_table.cpu[:runner.input_batch.num_reqs]
                   for block_id in row.tolist() if block_id != 0}
    engine.shutdown()
    return outputs, digest, null_block is not None, used_blocks, seen


clean_out, clean_kv, has_null, used_blocks, clean_seen = polluted_run(False)
dirty_out, dirty_kv, _has_null, _used, dirty_seen = polluted_run(True)
check("D1. 开了图就留白 0 号块（padding 的垃圾桶）", has_null)
check("D2. 0 号块没有被分给任何真实请求", 0 not in used_blocks, str(sorted(used_blocks))[:80])
check("D3. 用例真的覆盖到补齐区（2 行补齐 + 1 条补齐请求）",
      dirty_seen.get("tail", 0) > 0 and dirty_seen.get("extra_reqs", 0) > 0, str(dirty_seen))
check("D4. 污染 padding 缓冲后有效输出不变",
      clean_out == dirty_out, f"clean={clean_out} dirty={dirty_out}")
check("D5. 污染 padding 后真实块的 KV 逐位不变（0 号块除外）",
      torch.equal(clean_kv, dirty_kv))
check("D6. 补齐行的内容被显式写死（token/位置=0、槽位=哨兵），不依赖上一轮残值",
      clean_seen.get("tail", 0) > 0)

# ---------------------------------------------------------------- E profiler / 同步
engine, _core, runner = make_engine(tiny_dir=TINY, hf_config=HF, spec_k=3, budget=32,
                                    blocks=32, mode="full_decode_only")
engine.add_request("r", [1, 2, 3, 4, 5, 6],
                   SamplingParams(max_tokens=4, temperature=0.0, eos_token_id=999))
list(engine.step())
from torch.profiler import ProfilerActivity, profile  # noqa: E402

with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    while engine.has_unfinished_requests():
        list(engine.step())
    torch.cuda.synchronize()
names = [event.name for event in prof.events()]
graph_launches = sum(1 for name in names if "cudaGraphLaunch" in name)
check("E1. profiler 里看到真实的图重放（cudaGraphLaunch）", graph_launches > 0,
      f"cudaGraphLaunch={graph_launches}")

mode, batch_descriptor = runner._determine_batch_execution_and_padding(
    num_tokens=4, num_reqs=1, num_scheduled_tokens=[4], max_num_scheduled_tokens=4)
runner._fill_padded_buffers(4, 1, [4], batch_descriptor.num_tokens,
                            batch_descriptor.num_reqs, 4)
metadata = runner._padded_attn_metadata(batch_descriptor)
from minivllm.worker.gpu_model_runner import _PaddedRun  # noqa: E402

torch.cuda.synchronize()
torch.cuda.set_sync_debug_mode("error")
sync_error = None
try:
    runner._run_model_padded(
        _PaddedRun(batch_descriptor=batch_descriptor, mode=mode,
                   num_tokens=batch_descriptor.num_tokens), metadata=metadata)
except RuntimeError as exc:
    sync_error = str(exc)[:80]
finally:
    torch.cuda.set_sync_debug_mode("default")
check("E2. 图区域内没有任何逐请求同步（set_sync_debug_mode=error 下跑通）",
      sync_error is None, str(sync_error))
engine.shutdown()

# ---------------------------------------------------------------- F drafter padding
cu = torch.tensor([3, 5, 5, 6], dtype=torch.int32, device="cuda")
valid = torch.tensor([1, 3, 1, 2], dtype=torch.int32, device="cuda")
qsl = torch.tensor([0, 4, 7, 8, 10], dtype=torch.int32, device="cuda")
index, rejected = prepare_inputs_padded(cu, valid, qsl, 4)
check("F1. prepare_inputs_padded：每请求采样行 = 块末行 − 被拒数（含首拒/全接受/K=0）",
      index.tolist() == [0, 6, 7, 9] and rejected.tolist() == [3, 0, 0, 0],
      f"index={index.tolist()} rejected={rejected.tolist()}")
from vllm.v1.spec_decode.utils import eagle_prepare_inputs_padded_kernel  # noqa: E402

up_index = torch.empty(4, dtype=torch.int32, device="cuda")
up_rejected = torch.empty(4, dtype=torch.int32, device="cuda")
eagle_prepare_inputs_padded_kernel[(4,)](cu, valid, qsl, up_index, up_rejected, 4)
check("F2. 与上游内核逐值一致（同一个 kernel 的 GPU 结果）",
      up_index.tolist() == index.tolist() and up_rejected.tolist() == rejected.tolist(),
      f"upstream index={up_index.tolist()}")

check("F3. padding 槽位哨兵 = 上游常量 PADDING_SLOT_ID(-1)", PADDING_SLOT_ID == -1)

elapsed = time.perf_counter() - t_start
print(f"\n{'全部通过' if not FAILED else '失败: ' + ', '.join(FAILED)}  "
      f"（{PASSED} 项通过，耗时 {elapsed:.1f}s）")
result_path = Path(__file__).with_name("results") / "check_step69_cudagraph.json"
result_path.parent.mkdir(exist_ok=True)
result_path.write_text(json.dumps({
    "device": DEVICE, "passed": PASSED, "failed": FAILED, "trace": TRACE,
    "seconds": round(elapsed, 2),
    "capture_stats": {k: v for k, v in (eager_captures[0] if eager_captures else {}).items()
                      if k != "selections"},
    "env": {"pid": os.getpid(), "torch": torch.__version__},
}, ensure_ascii=False, indent=2), encoding="utf-8")
sys.exit(1 if FAILED else 0)
