"""71 关验收探针：动态投机长度（batch-size → K 表）与调度 / Graph / 异步的兼容面。

跑法：`python benchmarks/check_step71_dynamic_sd.py`（退出码非 0 = 失败）。

分段（对应需求 071 §4 的验收项）：

    A 查找表       与上游工具函数逐值差分；空隙 / 尾部 / 超最大 K / 六类坏表
    B 配置改写     full graph → PIECEWISE 的矩阵；DP>1 关表；方法边界；最大 K 必须 > 0
    C 调度器       K 按**实际被调度的请求数**查；空轮不查表；两轮时序（K 变但不重解释旧候选）；
                   异步占位宽度用本轮 K
    D 端到端       K 跨 4 → 0 → 2；K=0 仍跑 draft 第一遍；greedy 输出与不开投机逐 token 相同；
                   只命中 PIECEWISE；工作区地址不随 K 重建；ngram 旧列不被消费；q 宽度随轮次

CUDA 相关的两项在无 CUDA 的机器上记 **待验**（不算通过，与仓库其余探针同一口径）。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step71"))

import torch  # noqa: E402

from spec71_helpers import (DOC_TABLE, HF, TINY, RoundRecorder, greedy, make_config,  # noqa: E402
                            make_engine, make_prompt_requests, run_to_end, table_lookup)
from minivllm import (CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig,  # noqa: E402
                      SpeculativeConfig, VllmConfig)
from minivllm.core.sched.async_scheduler import AsyncScheduler  # noqa: E402
from minivllm.spec_decode.dynamic.utils import (  # noqa: E402
    build_dynamic_sd_schedule_lookup, validate_and_normalize_dynamic_sd_schedule)

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


def _spec(**kwargs):
    draft_hf = kwargs.pop("draft_hf", None) or HF
    defaults = dict(method="draft_model", num_speculative_tokens=4,
                    draft_model_config=ModelConfig(model=TINY, dtype="float32",
                                                   max_model_len=64, hf_config=draft_hf))
    defaults.update(kwargs)
    return SpeculativeConfig(**defaults)


# ---------------------------------------------------------------- A 查找表


TABLES = [DOC_TABLE, [(1, 16, 3), (32, 128, 2)], [(1, 1, 0), (2, 64, 3)], [(1, 4, 5)],
          [(1, 8, 0), (9, 16, 2), (20, 20, 7)], [(2, 4, 2)], [(1, 2, 4), (2, 3, 1)],
          [(1, 2, -1)], []]

from vllm.v1.spec_decode.dynamic.utils import (  # noqa: E402
    build_dynamic_sd_schedule_lookup as upstream_lookup)

mismatch = None
for table in TABLES:
    for max_batch in (1, 2, 4, 8, 16, 64, 200):
        for max_k in (1, 2, 3, 4, 8):
            try:
                want = ("ok", upstream_lookup(table, max_batch, max_k))
            except ValueError:
                want = ("error", None)
            try:
                got = ("ok", build_dynamic_sd_schedule_lookup(table, max_batch, max_k))
            except ValueError:
                got = ("error", None)
            if want != got:
                mismatch = (table, max_batch, max_k, want, got)
                break
check("A1. 查找表与上游同名函数逐值一致（9 张表 × 7 个上界 × 5 个最大 K）",
      mismatch is None, "" if mismatch is None else str(mismatch))

dense = build_dynamic_sd_schedule_lookup(DOC_TABLE, 10, 4)
check("A2. 需求 §3 的例子：[(1,2,4),(5,8,1)] → B=3/4 继承 4、B>8 延续 1、索引 0 不用",
      dense == [0, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1], dense)

clipped = build_dynamic_sd_schedule_lookup([(1, 4, 5)], 4, 3)
check("A3. 表里的 K 被全局最大 K 裁剪（表只能被裁小，不能放大容量）",
      clipped[1:] == [3, 3, 3, 3], clipped)

normalized = validate_and_normalize_dynamic_sd_schedule([(5, 8, "1"), (1, 2, 4)])
check("A4. 归一化 = 排序 + 逐项 int()（上游同一行为），结果写回配置",
      normalized == [(1, 2, 4), (5, 8, 1)]
      and _spec(num_speculative_tokens_per_batch_size=[(5, 8, 1), (1, 2, 4)])
      .num_speculative_tokens_per_batch_size == [(1, 2, 4), (5, 8, 1)], normalized)

bad_tables = [(1, 2, 4), [], [(1, 2)], [(0, 2, 1)], [(4, 2, 1)], [(1, 2, -1)],
              [(1, 2, 4), (2, 3, 1)], [(2, 4, 2)]]
rejected = []
for table in bad_tables:
    try:
        _spec(num_speculative_tokens_per_batch_size=table)
    except ValueError:
        rejected.append(table)
check("A5. 八类坏表在**配置期**报错（非 list / 空 / 非三元组 / 端点非正 / 起点>终点 / K<0 / 重叠 / 首段非 1）",
      len(rejected) == len(bad_tables), f"{len(rejected)}/{len(bad_tables)}")

try:
    _spec(num_speculative_tokens=0, num_speculative_tokens_per_batch_size=DOC_TABLE)
    ok = False
except ValueError:
    ok = True
check("A6. 最大 K 必须 > 0（表里的 K 被它裁剪；工作区/掩码按它开）", ok)

# ---------------------------------------------------------------- B 配置改写


MODES = [(None, "PIECEWISE"), ("full", "PIECEWISE"), ("full_decode_only", "PIECEWISE"),
         ("full_and_piecewise", "PIECEWISE"), ("piecewise", "PIECEWISE"), ("none", "NONE")]
wrong = []
for requested, expected in MODES:
    config = make_config(table=DOC_TABLE, mode=requested, spec_k=4)
    got = str(config.compilation_config.cudagraph_mode)
    if got != expected:
        wrong.append((requested, expected, got))
check("B1. 含 full graph 的模式一律降级成 PIECEWISE；纯 PIECEWISE / NONE 不动",
      not wrong, wrong or "6 种配置全部符合")

sizes = make_config(table=DOC_TABLE, mode=None, spec_k=4).compilation_config.cudagraph_capture_sizes
static_sizes = make_config(table=None, mode="full_decode_only",
                           spec_k=4).compilation_config.cudagraph_capture_sizes
check("B2. 降级之后档位表按 PIECEWISE 算：不再做「取整到 1+K 的倍数」"
      "（动态：[1,2,4,8,16]；静态 full：全是 5 的倍数）",
      any(size % 5 for size in sizes) and 1 in sizes
      and all(size % 5 == 0 for size in static_sizes),
      f"dynamic={sizes} static={static_sizes}")

static_mode = str(make_config(table=None, mode="full_decode_only", spec_k=4)
                  .compilation_config.cudagraph_mode)
check("B3. 对照组：不开动态表时 `full_decode_only` 不被这条规则改写",
      static_mode in ("FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"), static_mode)


class _Parallel:
    data_parallel_size = 2


dp_config = make_config(table=DOC_TABLE, mode="none", spec_k=4)
object.__setattr__(dp_config, "parallel_config", _Parallel())
dp_config._maybe_disable_dynamic_sd_for_data_parallel()
check("B4. DP>1 → 清空动态表并退回固定 K（上游规则；本仓库没有 DP 轴，用替身钉住分支）",
      dp_config.speculative_config.num_speculative_tokens_per_batch_size is None
      and not dp_config.speculative_config.uses_dynamic_speculative_decoding()
      and dp_config.speculative_config.num_speculative_tokens == 4)

kept = make_config(table=DOC_TABLE, mode="none", spec_k=2)
kept._maybe_disable_dynamic_sd_for_data_parallel()
check("B5. DP=1（本仓库正常路径）不动配置",
      kept.speculative_config.uses_dynamic_speculative_decoding() is True)

supported, unsupported_errors = [], []
for method in ("draft_model", "ngram"):
    spec = SpeculativeConfig(method=method, num_speculative_tokens=4,
                             num_speculative_tokens_per_batch_size=DOC_TABLE)
    supported.append(spec.uses_dynamic_speculative_decoding())
medusa_hf = dict(HF, architectures=["MedusaModel"])
extract_hf = dict(HF, eagle_aux_hidden_state_layer_ids=[0, 1])
for method, extra in [("ngram_gpu", {}), ("suffix", {}), ("medusa", {"draft_hf": medusa_hf}),
                      ("extract_hidden_states", {"draft_hf": extract_hf}),
                      ("custom_class", {"model": "examples.custom_proposer.RepeatLastTokenProposer"})]:
    draft_hf = extra.pop("draft_hf", None)
    try:
        _spec(method=method, num_speculative_tokens=1,
              num_speculative_tokens_per_batch_size=DOC_TABLE,
              draft_hf=draft_hf, **extra)
    except ValueError as exc:
        unsupported_errors.append((method, "num_speculative_tokens_per_batch_size" in str(exc)))
check("B6. 方法边界：draft_model / ngram 支持；ngram_gpu / suffix / medusa / extract / "
      "custom_class 在配置期明确拒绝（不删上游断言、不静默忽略）",
      all(supported) and len(unsupported_errors) == 5 and all(ok for _, ok in unsupported_errors),
      f"支持={supported} 拒绝={unsupported_errors}")

# ---------------------------------------------------------------- C 调度器


def _staggered(table):
    engine, core, runner = make_engine(table=table, spec_k=4, max_num_seqs=4, budget=32)
    recorder = RoundRecorder(runner, core)
    make_prompt_requests(engine, ("A",), max_tokens=20)
    for _ in range(2):
        engine.step()
    make_prompt_requests(engine, ("B",), max_tokens=8)
    engine.step()
    make_prompt_requests(engine, ("C",), max_tokens=8)
    run_to_end(engine)
    return recorder, engine, core


TABLE = [(1, 1, 4), (2, 2, 0), (3, 8, 2)]
recorder, engine, core = _staggered(TABLE)
non_empty = [round_ for round_ in recorder.rounds if round_["num_reqs"] > 0]
check("C1. K 按**实际被调度的请求数**查表（B=1→4、B=2→0、B=3→2）",
      [round_["num_reqs"] for round_ in non_empty[:4]] == [1, 1, 2, 3]
      and [round_["k"] for round_ in non_empty[:4]] == [4, 4, 0, 2],
      [(round_["num_reqs"], round_["k"]) for round_ in recorder.rounds])

check("C2. 每一轮的 SchedulerOutput 值都等于查找表在该轮请求数下的取值（空轮保持最大 K）",
      all((round_["k"] == 4 if round_["num_reqs"] == 0
           else round_["k"] == table_lookup(TABLE, 4, 4, round_["num_reqs"]))
          for round_ in recorder.rounds),
      recorder.ks)

zero_index = next(index for index, round_ in enumerate(recorder.rounds) if round_["k"] == 0)
previous = recorder.rounds[zero_index - 1]
check("C3. 两轮时序：K=0 的那一轮仍然验证**上一轮（K=4）提的旧候选**，且逐值相同",
      previous["k"] == 4 and bool(recorder.rounds[zero_index]["adopted"])
      and all(adopted == previous["proposed"].get(req_id, [])
              for req_id, adopted in recorder.rounds[zero_index]["adopted"].items()),
      f"上一轮提={ {k: len(v) for k, v in previous['proposed'].items()} } "
      f"本轮验={ {k: len(v) for k, v in recorder.rounds[zero_index]['adopted'].items()} }")

check("C4. 本轮提的草稿长度 ≤ 本轮 K，且 K=0 的轮一枚都不提",
      all(len(drafts) <= round_["k"] for round_ in recorder.rounds
          for drafts in round_["proposed"].values())
      and all(not drafts for round_ in recorder.rounds if round_["k"] == 0
              for drafts in round_["proposed"].values()),
      recorder.ks)
recorder.restore()
engine.shutdown()

# 异步：占位宽度必须是**本轮 K**（不是配置的最大 K）
engine, core, _runner = make_engine(table=[(1, 1, 0), (2, 8, 2)], spec_k=2, max_num_seqs=2)
scheduler = AsyncScheduler(core.vllm_config.scheduler_config, core.kv_cache_manager,
                           max_model_len=64,
                           speculative_config=core.vllm_config.speculative_config,
                           structured_output_manager=core.structured_output_manager)
make_prompt_requests(engine, ("A",))
scheduler.add_request(core.scheduler.requests["A"])
first = scheduler.schedule()
make_prompt_requests(engine, ("B",))
scheduler.add_request(core.scheduler.requests["B"])
second = scheduler.schedule()
check("C5. AsyncScheduler 的占位草稿宽度 = 本轮 K（B=1→空、B=2→[-1,-1]）",
      first.num_spec_tokens_to_schedule == 0
      and scheduler.requests["A"].spec_token_ids == [-1, -1]
      and second.num_spec_tokens_to_schedule == 2,
      f"first K={first.num_spec_tokens_to_schedule} second K={second.num_spec_tokens_to_schedule}")
engine.shutdown()

try:
    make_engine(table=DOC_TABLE, spec_k=4, async_scheduling=True)
    async_rejected = False
except NotImplementedError:
    async_rejected = True
check("C6. 显式 async 组合：端到端仍按 70 关的边界明确拒绝（动态表不放开它）", async_rejected)

# ---------------------------------------------------------------- D 端到端


dyn = greedy(req_ids=("A",), spec_k=3, table=[(1, 1, 0), (2, 8, 3)])
eager = greedy(req_ids=("A",), spec_k=None, table=None)
static = greedy(req_ids=("A",), spec_k=3, table=None)
check("D1. K 在 0 与 3 之间切换的 greedy 输出与**不开投机**、与固定 K=3 逐 token 相同",
      dyn["A"] == eager["A"] == static["A"], f"{dyn['A']} / {eager['A']} / {static['A']}")

engine, core, runner = make_engine(table=[(1, 1, 0), (2, 8, 3)], spec_k=3, max_num_seqs=2,
                                   budget=32)
recorder = RoundRecorder(runner, core)
make_prompt_requests(engine, ("A",), max_tokens=6)
for _ in range(3):
    engine.step()
zero_rounds = [round_ for round_ in recorder.rounds if round_["k"] == 0]
publish_ok = all(core.scheduler.publish_bound(request)
                 <= runner.proposer._draft_computed.get(req_id, 0)
                 for req_id, request in core.scheduler.requests.items())
check("D2. K=0 的轮**仍然跑 draft 第一遍**（次数 ≥1），draft 进度追平已提交历史，"
      "且发布上界不超过 draft 已同步的位置（199 §9）",
      bool(zero_rounds) and all(round_["first_passes"] >= 1 for round_ in zero_rounds)
      and runner.proposer._draft_computed.get("A") == runner.requests["A"].num_tokens
      and publish_ok,
      f"K 序列={recorder.ks} 第一遍次数={[r['first_passes'] for r in recorder.rounds]}")

pointers = {(runner.proposer.input_ids.data_ptr(), runner.proposer.positions.data_ptr(),
             runner.proposer.slot_mapping.data_ptr())}
make_prompt_requests(engine, ("B",), max_tokens=3)
run_to_end(engine)
pointers.add((runner.proposer.input_ids.data_ptr(), runner.proposer.positions.data_ptr(),
              runner.proposer.slot_mapping.data_ptr()))
check("D3. 工作区（input_ids / positions / slot_mapping）地址不随逐轮 K 变化重建",
      len(pointers) == 1 and 0 in recorder.ks and 3 in recorder.ks,
      f"{len(pointers)} 组地址；K 序列={recorder.ks}")
recorder.restore()
engine.shutdown()

# ngram：定宽草稿缓冲的"旧无效列"不许被交出去
from minivllm.spec_decode.ngram_proposer import NgramProposer  # noqa: E402
from minivllm.spec_decode.utils import TargetRows  # noqa: E402

ngram_spec = _spec(method="ngram", num_speculative_tokens=4, prompt_lookup_min=2,
                   prompt_lookup_max=2)
ngram_config = VllmConfig(
    model_config=ModelConfig(model=TINY, dtype="float32", max_model_len=64, hf_config=HF),
    cache_config=CacheConfig(block_size=4, num_gpu_blocks=32),
    scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=32),
    device_config=DeviceConfig("cpu"), speculative_config=ngram_spec)
proposer = NgramProposer(ngram_config)
history = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
rows = [TargetRows(req_id="A", row=0, start=0, target_rows=len(history), num_rejected=0,
                   history_end=len(history), next_token_id=6, ready=True)]
full = proposer.propose_drafts(rows, {"A": history}, input_batch=None,
                               num_speculative_tokens=4).draft_token_ids[0]
short = proposer.propose_drafts(rows, {"A": history}, input_batch=None,
                                num_speculative_tokens=1).draft_token_ids[0]
none = proposer.propose_drafts(rows, {"A": history}, input_batch=None,
                               num_speculative_tokens=0).draft_token_ids[0]
try:
    proposer.propose_drafts(rows, {"A": history}, input_batch=None,
                            num_speculative_tokens=5)
    over = False
except ValueError:
    over = True
check("D4. ngram 的定宽草稿缓冲：K=4→[1,2,3,4]、K=1→只回第一枚、K=0→空、超上界报错",
      full == [1, 2, 3, 4] and short == [1] and none == [] and over,
      f"{full} / {short} / {none} / 超界报错={over}")

if torch.cuda.is_available():
    engine, core, runner = make_engine(table=[(1, 1, 0), (2, 8, 2)], spec_k=2,
                                       max_num_seqs=2, budget=32)
    ok_piecewise = str(runner.compilation_config.cudagraph_mode) == "PIECEWISE"
    make_prompt_requests(engine, ("A", "B"), max_tokens=4)
    run_to_end(engine)
    modes = {selection["mode"] for selection in runner.cudagraph_selections}
    engine.shutdown()
    check("D5. 真实引擎上动态 K 只命中 PIECEWISE（配置改写真的落到运行时）",
          ok_piecewise and "PIECEWISE" in modes and modes <= {"PIECEWISE", "NONE"}, modes)

    engine, core, runner = make_engine(table=[(1, 1, 0), (2, 8, 2)], spec_k=2, max_num_seqs=2,
                                       budget=32, draft_sample_method="probabilistic")
    recorder = RoundRecorder(runner, core)
    make_prompt_requests(engine, ("A",), max_tokens=4)
    for _ in range(2):
        engine.step()
    make_prompt_requests(engine, ("B",), max_tokens=4)
    run_to_end(engine)
    shapes = [(index, round_["drafts_gpu"],
               sum(len(drafts) for drafts in round_["proposed"].values()))
              for index, round_ in enumerate(recorder.rounds)
              if round_["drafts_gpu"] is not None]
    # 每项是 (轮号, q 的形状, 本轮实际提议的草稿数)：q 的行数必须等于后者
    check("D6. 概率草稿：q 的行数 = 本轮实际提议的草稿数（逐轮变宽时不掺旧轮）",
          len(shapes) >= 2 and all(shape[1][0] == shape[2] for shape in shapes)
          and 0 in recorder.ks and 2 in recorder.ks,
          f"{shapes} K 序列={recorder.ks}")
    recorder.restore()
    engine.shutdown()
else:
    check("D5. 真实引擎上动态 K 只命中 PIECEWISE：待验（本机没有 CUDA）", False)
    check("D6. 概率草稿的 q 行数与提议宽度一致：待验（需要 Triton 拒绝采样）", False)

print(f"\n{'全部通过' if not FAILED else '失败: ' + ', '.join(FAILED)}  （{PASSED} 项通过）")
sys.exit(1 if FAILED else 0)
