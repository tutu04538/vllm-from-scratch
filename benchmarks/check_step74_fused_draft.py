"""74 关验收探针：V2 自回归 Speculator 与多步融合 Graph。

跑法：`python benchmarks/check_step74_fused_draft.py`（退出码非 0 = 失败；26 项 + 1 段 profiler）。

分段（对应需求 074 §3/§4）：

    A 管理器        候选表取整 / 兼容性两档 / 分派回退 / 捕获期假输入的安全值
    B capability    Torch 后端声明支持且空操作；"不支持"的替身被调用就报错；AttentionGroup 转发
    C 融合门        K=1/2/4 × 支持/不支持 → 选融合还是逐步（**实际执行**取证，不只看开关）
    D 两条路等价    逐轮草稿 token / positions / seq_lens / slot_mapping / 草稿 KV 全等
    E 图里真的有多步 一轮 1 次重放 + 0 次 Python forward + K 列草稿全写出来
    F 端到端        V2 图 == V2 eager == V1（EAGLE3 与 MTP）；批重排 / 分块 prefill / 首拒 / 长度边界
    G 边界          纯 PIECEWISE 与 V2+mtp 缺 draft 配置在配置期/初始化期明确拒绝

⚠️ 本机 tiny 权重是**随机**的：探针验的是**寻址、协议与两条路的等价性**，
不是草稿质量（接受长度受 63 关已知差距影响，见 `docs/step63_alignment.md` §8）。
profiler 那一段只比较"融合 vs 逐步"在同一次运行里的 CPU 派发数量与草稿耗时，
**不给虚构加速倍数**（数字随机器/驱动变化，只作为方向性证据）。
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step74"))

from spec74_helpers import (DEFAULT_SIZES, DraftForwardCounter,  # noqa: E402
                            DraftSampleCounter, RoundRecorder,
                            draft_attn_groups, force_draft_metadata_capability,
                            make_config, make_engine, run_steps)

from minivllm import SamplingParams  # noqa: E402
from minivllm.attention.backends.torch_sdpa import (  # noqa: E402
    TorchAttentionBackend,
    TorchAttentionMetadataBuilder,
)
from minivllm.attention.metadata import AttentionMetadata  # noqa: E402
from minivllm.config import CUDAGraphMode  # noqa: E402
from minivllm.testing.fake_attn_backend import NoDraftMetadataUpdateBuilder  # noqa: E402
from minivllm.worker.gpu.block_table import PAD_SLOT_ID, BlockTables  # noqa: E402
from minivllm.worker.gpu.cudagraph_utils import (  # noqa: E402
    BatchExecutionDescriptor,
    CudaGraphManager,
    _is_compatible,
)
from minivllm.worker.gpu.input_batch import InputBatch, InputBuffers  # noqa: E402
from minivllm.worker.utils import (  # noqa: E402
    AttentionGroup,
    get_uniform_decode_token_count,
)

PASSED, FAILED, TRACE = 0, [], []


def check(name, ok, detail=""):
    global PASSED
    if ok:
        PASSED += 1
        print(f"PASS  {name}  {detail}")
    else:
        FAILED.append(name)
        print(f"FAIL  {name}  {detail}")
    TRACE.append({"item": name, "ok": bool(ok), "detail": str(detail)[:300]})


class _MonkeyPatch:
    """探针里的最小 monkeypatch（不引 pytest）。"""

    def __init__(self):
        self.undo = []

    def setattr(self, obj, name, value):
        self.undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def revert(self):
        for obj, name, old in reversed(self.undo):
            setattr(obj, name, old)
        self.undo.clear()


if not torch.cuda.is_available():  # pragma: no cover - 本机有 CUDA
    print("本探针需要 CUDA（V2 的输入组装/采样/验证是 Triton 内核 + UVA 常驻状态）")
    sys.exit(1)

PROMPTS = [("A", [1, 2, 3, 4, 5, 6]), ("B", [7, 8, 9]), ("C", [10, 9, 8, 7])]

# ================================================================ A 管理器

config = make_config(spec_k=3, mode="full_decode_only", capture_sizes=[1, 2, 3, 4, 8])
prefill_mgr = CudaGraphManager(config, torch.device("cpu"), CUDAGraphMode.FULL_DECODE_ONLY,
                              decode_query_len=4, allow_shape_padding=False)
decode_mgr = CudaGraphManager(config, torch.device("cpu"), CUDAGraphMode.FULL_DECODE_ONLY,
                              decode_query_len=1, allow_shape_padding=True)
prefill_shapes = sorted((d.num_tokens, d.num_reqs) for d in
                        prefill_mgr._capture_descs[CUDAGraphMode.FULL])
decode_shapes = sorted((d.num_tokens, d.num_reqs) for d in
                       decode_mgr._capture_descs[CUDAGraphMode.FULL])
check("A1. 草稿 prefill 图的档位按 1+K 取整（1/2/3/4→4 行 1 请求、8→2 请求）",
      prefill_shapes == [(4, 1), (8, 2)], f"{prefill_shapes}")
check("A2. 草稿 decode 图每请求 1 行，且丢掉请求数超上限的档位",
      decode_shapes == [(1, 1), (2, 2), (3, 3)], f"{decode_shapes}")

desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=8, num_reqs=2,
                                uniform_token_count=4)
compat = {
    "exact/pad": _is_compatible(desc, 2, 8, 4, 0, None, True),
    "exact/nopad": _is_compatible(desc, 2, 8, 4, 0, None, False),
    "smaller/pad": _is_compatible(desc, 1, 4, 4, 0, None, True),
    "smaller/nopad": _is_compatible(desc, 1, 4, 4, 0, None, False),
    "wrong-uniform": _is_compatible(desc, 2, 8, 1, 0, None, True),
}
check("A3. 兼容性两档：精确命中的图两边都收，补齐的图只给'自己拥有那份缓冲'的 decode 图",
      compat == {"exact/pad": True, "exact/nopad": True, "smaller/pad": True,
                 "smaller/nopad": False, "wrong-uniform": False}, f"{compat}")

fallback = prefill_mgr.dispatch(1, 4, 4)
prefill_mgr._graphs_captured = True
no_shape = prefill_mgr.dispatch(3, 12, 4)
hit = prefill_mgr.dispatch(1, 4, 4)
check("A4. 分派：没捕获完/没这个档位 → NONE（带未补齐数字）；命中 → 返回图键",
      fallback.cg_mode == CUDAGraphMode.NONE and fallback.num_tokens == 4
      and no_shape.cg_mode == CUDAGraphMode.NONE
      and hit.cg_mode == CUDAGraphMode.FULL and hit.num_tokens == 4,
      f"fallback={fallback.cg_mode} no_shape={no_shape.cg_mode} hit={hit.cg_mode}")

buffers = InputBuffers(max_num_reqs=4, max_num_tokens=32, device="cuda")
buffers.input_ids[:8] = 7
dummy = InputBatch.make_dummy(2, 8, buffers)
safe = (buffers.input_ids[:8].tolist() == [0] * 8
        and buffers.positions[:8].tolist() == [0] * 8
        and buffers.seq_lens[2:].tolist() == [0, 0]
        and buffers.query_start_loc[:3].tolist() == [0, 4, 8]
        and dummy.uniform_query_len == 4
        and dummy.input_ids.data_ptr() == buffers.input_ids.data_ptr())
check("A5. 捕获期假输入只写常驻缓冲；补齐行 token/position 清零、seq_len=0、query_start_loc 非递减",
      safe, f"uniform_query_len={dummy.uniform_query_len}")

tables = BlockTables(block_sizes=[4], max_num_reqs=4, max_num_batched_tokens=32,
                     max_num_blocks_per_group=[4], device="cuda", kernel_block_sizes=[4])
tables.input_block_tables[0][:2] = 9
tables.get_dummy_block_tables(2)
tables.slot_mappings.fill_(3)
slot_view = tables.get_dummy_slot_mappings(5)
check("A6. 假块表补零行（0 号垃圾桶）、假槽位全哨兵，且都是常驻张量的视图",
      tables.input_block_tables[0][:2].abs().sum().item() == 0
      and (tables.slot_mappings == PAD_SLOT_ID).all()
      and slot_view.data_ptr() == tables.slot_mappings.data_ptr())

# ================================================================ B capability

prod_group = AttentionGroup(backend=TorchAttentionBackend, layer_names=["l0"],
                            metadata_builders=[TorchAttentionMetadataBuilder(4)])
fake_group = AttentionGroup(backend=TorchAttentionBackend, layer_names=["l0"],
                            metadata_builders=[NoDraftMetadataUpdateBuilder(4)])
check("B1. capability 挂在构造器上并由 AttentionGroup 转发（生产 True / 替身 False）",
      prod_group.supports_draft_decode_metadata_update is True
      and fake_group.supports_draft_decode_metadata_update is False)

metadata = AttentionMetadata(
    query_start_loc=torch.zeros(2, dtype=torch.int32), seq_lens=torch.zeros(1, dtype=torch.int32),
    block_table=torch.zeros(1, 1, dtype=torch.int32),
    slot_mapping=torch.zeros(1, dtype=torch.int64), block_size=4, num_reqs=1,
    uniform_query_len=1)
prod_noop = TorchAttentionMetadataBuilder(4).update_draft_decode_metadata(metadata) is None
try:
    NoDraftMetadataUpdateBuilder(4).update_draft_decode_metadata(metadata)
    fake_raises = False
except NotImplementedError:
    fake_raises = True
check("B2. 生产后端是空操作（步相关字段就是常驻缓冲的切片）；声明不支持的后端被调用就报错",
      prod_noop and fake_raises)

check("B3. 统一 decode 判据：形状相同**且**没有请求在 prefill 才算统一批",
      get_uniform_decode_token_count(2, 8, 4, False) == 4
      and get_uniform_decode_token_count(2, 8, 4, True) is None)

# ================================================================ C/D/E 融合 vs 逐步

monkeypatch = _MonkeyPatch()
gate_ok, gate_detail = True, []
for spec_k in (1, 2, 4):
    for supports in (True, False):
        force_draft_metadata_capability(monkeypatch, supports_update=supports)
        engine, _core, runner = make_engine(spec_k=spec_k, mode="full_decode_only")
        try:
            expected = supports and spec_k > 1
            actual = runner.speculator.use_fused_multi_step_decode
            gate_ok &= actual is expected
            if spec_k == 1:
                gate_ok &= runner.speculator.decode_cudagraph_manager.graphs == {}
            gate_detail.append(f"K={spec_k}/sup={int(supports)}→fused={int(actual)}")
        finally:
            engine.shutdown()
check("C1. 融合门：K=1 恒不融合；K>1 完全由后端 capability 决定（两个后端各跑一遍）",
      gate_ok, " ".join(gate_detail))

rounds_by_capability, replays_ok = {}, True
for supports in (True, False):
    force_draft_metadata_capability(monkeypatch, supports_update=supports)
    engine, _core, runner = make_engine(spec_k=3, mode="full_decode_only")
    rec = RoundRecorder(runner)
    try:
        for req_id, prompt in PROMPTS:
            engine.add_request(req_id, list(prompt),
                               SamplingParams(max_tokens=10, temperature=0.0,
                                              eos_token_id=18))
        run_steps(engine, runner)
    finally:
        engine.shutdown()
    rounds_by_capability[supports] = rec.rounds
    decode_rounds = [r for r in rec.rounds if r["num_reqs"] > 0]
    expected = 1 if supports else 2
    replays_ok &= bool(decode_rounds) and all(r["decode_replays"] == expected
                                              for r in decode_rounds)
check("C2. 一轮里草稿 decode 图重放次数：融合 1 次（走完 K−1 步）/ 逐步 K−1 次",
      replays_ok,
      f"fused={[r['decode_replays'] for r in rounds_by_capability[True]][:4]} "
      f"fallback={[r['decode_replays'] for r in rounds_by_capability[False]][:4]}")

fused, fallback = rounds_by_capability[True], rounds_by_capability[False]
same_len = len(fused) == len(fallback) > 1
equal = same_len and all(
    torch.equal(a["draft_tokens"], b["draft_tokens"])
    and torch.equal(a["positions"], b["positions"])
    and torch.equal(a["seq_lens"], b["seq_lens"])
    and torch.equal(a["slot_mappings"], b["slot_mappings"])
    and a["kv_checksum"] == b["kv_checksum"]
    for a, b in zip(fused, fallback))
check("D1. 两条路**逐轮**草稿 token / positions / seq_lens / slot_mapping / 草稿 KV 全等",
      equal, f"轮数={len(fused)}")

counts = {}
for supports in (True, False):
    force_draft_metadata_capability(monkeypatch, supports_update=supports)
    engine, _core, runner = make_engine(spec_k=4, mode="full_decode_only")
    forward_counter = DraftForwardCounter(runner.speculator)
    sample_counter = DraftSampleCounter(runner.speculator)
    rec = RoundRecorder(runner)
    try:
        engine.add_request("A", [1, 2, 3, 4, 5, 6],
                           SamplingParams(max_tokens=8, temperature=0.0, eos_token_id=18))
        engine.step()
        forward_counter.count = sample_counter.count = 0
        rec.rounds.clear()
        run_steps(engine, runner)
    finally:
        engine.shutdown()
    decode_rounds = [r for r in rec.rounds if r["num_reqs"] > 0]
    counts[supports] = {
        "replays": [r["decode_replays"] for r in decode_rounds],
        "forwards": forward_counter.count,
        "samples": sample_counter.count,
        "step": decode_rounds[-1]["current_draft_step"] if decode_rounds else None,
        "columns": bool(decode_rounds and (decode_rounds[-1]["draft_tokens"] > 0).all()),
    }
check("E1. 融合：一轮重放 1 次、Python 前向 0 次、采样 0 次，而 current_draft_step 到 K−1、"
      "K 列草稿全写出来（多步**真的在图里**）",
      set(counts[True]["replays"]) == {1} and counts[True]["forwards"] == 0
      and counts[True]["samples"] == 0 and counts[True]["step"] == 3
      and counts[True]["columns"],
      f"{counts[True]}")
check("E2. 逐步（对照组）：同样 0 次 Python 前向，但每轮重放 K−1=3 次",
      set(counts[False]["replays"]) == {3} and counts[False]["forwards"] == 0,
      f"{counts[False]}")

# ================================================================ F 端到端


def greedy_pair(*, spec_k=3, **kwargs):
    results = {}
    for v2, mode in ((False, "none"), (True, "none"), (True, "full_decode_only")):
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if v2 else "0"
        config = make_config(spec_k=spec_k, mode=mode, **kwargs)
        from minivllm import LLMEngine, UniProcExecutor, Worker
        engine = LLMEngine(config, UniProcExecutor(config, Worker(config)), tokenizer=None)
        try:
            for req_id, prompt in PROMPTS:
                engine.add_request(req_id, list(prompt),
                                   SamplingParams(max_tokens=10, temperature=0.0,
                                                  eos_token_id=18))
            final = {}
            for _ in range(400):
                if not engine.has_unfinished_requests():
                    break
                for out in engine.step():
                    final[out.request_id] = list(out.token_ids)
        finally:
            engine.shutdown()
        results[(v2, mode)] = final
    return results


def _same(runs) -> bool:
    """三条路的结果是否逐 token 相同（把 dict 规范化后比，别比 `str(dict)` 的键序）。"""
    canonical = {tuple(sorted((req, tuple(tokens)) for req, tokens in run.items()))
                 for run in runs.values()}
    return len(canonical) == 1


monkeypatch.revert()
results = greedy_pair()
check("F1. EAGLE3：V2 图 == V2 eager == V1 eager（3 条请求、批重排、greedy 逐 token）",
      _same(results) and all(len(tokens) == 10
                             for run in results.values() for tokens in run.values()),
      f"{list(results.values())[0]}")

chunked = greedy_pair(budget=8, max_num_seqs=2)
check("F2. 分块 prefill（budget=8 < 提示词长度）下三条路仍然逐 token 相同", _same(chunked))

preempt = greedy_pair(blocks=6, budget=16, max_num_seqs=3)
check("F3. 抢占恢复（块不够）下三条路仍然逐 token 相同", _same(preempt))

os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
length_results = {}
for mode in ("none", "full_decode_only"):
    config = make_config(spec_k=3, mode=mode, max_model_len=12,
                         capture_sizes=DEFAULT_SIZES)
    from minivllm import LLMEngine, UniProcExecutor, Worker
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)), tokenizer=None)
    try:
        engine.add_request("A", [1, 2, 3, 4, 5, 6],
                           SamplingParams(max_tokens=12, temperature=0.0, eos_token_id=18))
        final = {}
        for _ in range(200):
            if not engine.has_unfinished_requests():
                break
            for out in engine.step():
                final[out.request_id] = list(out.token_ids)
    finally:
        engine.shutdown()
    length_results[mode] = final
check("F4. 模型长度边界（max_model_len=12）下图路径与 eager 相同且请求被截断",
      length_results["none"] == length_results["full_decode_only"]
      and 1 <= len(length_results["none"]["A"]) <= 7,
      f"tokens={len(length_results['none']['A'])}")

# MTP：V2（eager 与图）与 V1 逐 token 相同
sys.path.insert(0, str(ROOT / "tests" / "step74"))
import test_mtp_speculator as mtp  # noqa: E402

mtp_runs = {}
for label, v2, mode in (("v1", False, "none"), ("v2-eager", True, "none"),
                        ("v2-graph", True, "full_decode_only")):
    mtp_runs[label], _runner = mtp.run_mtp(v2=v2, mode=mode, k=2)
check("F5. MTP（原生 MTP 提议者）：V2 图 == V2 eager == V1（K=2、含 decode 图重放）",
      len(set(map(str, mtp_runs.values()))) == 1,
      f"{mtp_runs['v2-graph']}")

# ================================================================ G 边界

boundary_ok, boundary_detail = True, []
try:
    make_config(spec_k=2, mode="piecewise")
    boundary_ok = False
    boundary_detail.append("pure-piecewise 没报错")
except NotImplementedError as exc:
    boundary_detail.append(f"piecewise→{str(exc)[:18]}…")

from minivllm.worker.gpu.spec_decode import init_speculator  # noqa: E402
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
try:
    init_speculator(mtp.mtp_configs(k=2, explicit_draft=False), torch.device("cpu"))
    boundary_ok = False
    boundary_detail.append("mtp-without-draft-config 没报错")
except NotImplementedError as exc:
    boundary_detail.append(f"mtp→{str(exc)[:18]}…")
check("G1. 纯 PIECEWISE 与 V2+mtp 缺 draft 配置都在配置期/初始化期明确拒绝（不静默降级）",
      boundary_ok, " | ".join(boundary_detail))

# ================================================================ Profiler

print("\n---- profiler：融合 vs 逐步（同一 workload，各跑 12 轮 decode）----")


def profile_run(*, supports_update: bool):
    force_draft_metadata_capability(monkeypatch, supports_update=supports_update)
    engine, _core, runner = make_engine(spec_k=4, mode="full_decode_only",
                                        max_num_seqs=1)
    speculator = runner.speculator
    draft_seconds = [0.0]
    original_propose = speculator.propose

    def timed(*args, **kwargs):
        start = time.perf_counter()
        out = original_propose(*args, **kwargs)
        draft_seconds[0] += time.perf_counter() - start
        return out

    speculator.propose = timed
    try:
        engine.add_request("A", [1, 2, 3, 4, 5, 6],
                           SamplingParams(max_tokens=24, temperature=0.0, eos_token_id=18))
        engine.step()          # prefill 轮：不算进稳态
        draft_seconds[0] = 0.0
        from torch.profiler import ProfilerActivity, profile
        steps = 0
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(400):
                if not engine.has_unfinished_requests() or steps >= 12:
                    break
                # 每"轮"包含一次 target 前向 + 一次草稿提议（草稿提议在 sample_tokens 里）
                engine.step()
                steps += 1
        cpu_ops = sum(e.count for e in prof.key_averages() if e.device_type.name == "CPU")
        cuda_ops = sum(e.count for e in prof.key_averages() if e.device_type.name == "CUDA")
    finally:
        engine.shutdown()
    return {"steps": steps, "replays": speculator.decode_cudagraph_manager.num_replays,
            "draft_seconds": draft_seconds[0], "cpu_ops": cpu_ops, "cuda_ops": cuda_ops,
            "fused": speculator.use_fused_multi_step_decode}


fused_prof = profile_run(supports_update=True)
serial_prof = profile_run(supports_update=False)
print(f"  fused  : {fused_prof}")
print(f"  serial : {serial_prof}")
per_step = lambda r: r["draft_seconds"] / max(1, r["steps"])       # noqa: E731
check("P1. profiler：融合路径每轮草稿耗时 ≤ 逐步路径（同 workload、同 seed）",
      per_step(fused_prof) <= per_step(serial_prof) * 1.05,
      f"fused={per_step(fused_prof) * 1e3:.2f} ms/轮  serial={per_step(serial_prof) * 1e3:.2f} ms/轮")
check("P2. profiler：融合路径的草稿图重放次数 ≈ 轮数（逐步是 K−1 倍）",
      abs(fused_prof["replays"] - fused_prof["steps"]) <= 1
      and serial_prof["replays"] >= 2 * max(1, serial_prof["steps"] - 1),
      f"fused={fused_prof['replays']}/{fused_prof['steps']} 轮  "
      f"serial={serial_prof['replays']}/{serial_prof['steps']} 轮")
check("P3. profiler：融合路径的 CPU 侧算子派发数少于逐步路径（草稿步整体在图里提交）",
      fused_prof["cpu_ops"] < serial_prof["cpu_ops"],
      f"cpu_ops fused={fused_prof['cpu_ops']} serial={serial_prof['cpu_ops']}；"
      f"cuda 事件 fused={fused_prof['cuda_ops']} serial={serial_prof['cuda_ops']}")
TRACE.append({"item": "profiler", "ok": True,
              "detail": {"fused": fused_prof, "serial": serial_prof}})

# ================================================================ 汇总

print(f"\n{PASSED} passed, {len(FAILED)} failed")
results_path = Path(__file__).with_name("results") / "step74_probe.json"
results_path.parent.mkdir(exist_ok=True)
import json  # noqa: E402

results_path.write_text(json.dumps({"passed": PASSED, "failed": FAILED, "trace": TRACE},
                                   ensure_ascii=False, indent=2))
if FAILED:
    print("FAILED:", ", ".join(FAILED))
    sys.exit(1)
