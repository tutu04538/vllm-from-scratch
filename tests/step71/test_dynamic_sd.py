"""71 关（动态投机长度与调度 Graph 兼容）的自检。

需求 071 §4 的验收项在这里逐条落地：

    1. 所有边界配置、空隙、尾部、超最大 K —— 与上游工具函数**逐值**比较
    2. B 跨区间变化（含 K=4→0→2）—— 检查 SchedulerOutput、实际 proposal、验证长度、q 的轮次
    3. K=0 经数轮后恢复 K>0 —— draft KV 不缺历史，greedy 输出与不开投机逐 token 相同
    4. Graph 配置转换（full → PIECEWISE）、DP fallback、显式 async 组合与冻结 config 一致
    5. 稳定工作区不随 K 重新分配，旧无效列不被消费

"固定 K 的提议者不支持动态表"这条边界单独用一组用例列出**真实限制与报错**（需求 071 §3.6），
不宣称全方法支持。
"""

import pytest
import torch

from spec71_helpers import (DEVICE, DOC_TABLE, HF, TINY, RoundRecorder, greedy, make_config,
                            make_engine, make_prompt_requests, requires_cuda, run_to_end,
                            table_lookup)
from minivllm import (CacheConfig, DeviceConfig, ModelConfig, SamplingParams, SchedulerConfig,
                      SpeculativeConfig, VllmConfig)
from minivllm.config import CUDAGraphMode, CompilationConfig
from minivllm.core.sched.async_scheduler import AsyncScheduler
from minivllm.spec_decode.dynamic.utils import (
    build_dynamic_sd_schedule_lookup, validate_and_normalize_dynamic_sd_schedule)


def _spec(**kwargs):
    """只造一个 SpeculativeConfig（不需要引擎）。`draft_hf` 用来改 draft 侧的 hf 配置。"""
    draft_hf = kwargs.pop("draft_hf", None) or HF
    defaults = dict(method="draft_model", num_speculative_tokens=4,
                    draft_model_config=ModelConfig(model=TINY, dtype="float32",
                                                   max_model_len=64, hf_config=draft_hf))
    defaults.update(kwargs)
    return SpeculativeConfig(**defaults)


def _config(table=None, mode=None, spec_k=4, **kwargs):
    return make_config(table=table, mode=mode, spec_k=spec_k, **kwargs)


# ------------------------------------------------------------ 1. 查找表（与上游逐值差分）


def test_lookup_matches_upstream_reference():
    """边界配置 / 空隙 / 尾部 / 超最大 K 全都要与上游同名函数**逐值**一致。

    上游在本机是 `vllm.v1.spec_decode.dynamic.utils`（差分测试允许 import 参考实现，
    见 AGENTS §2.1）；两边都跑同一批输入，输出与报错种类都必须一致。
    """
    from vllm.v1.spec_decode.dynamic.utils import (
        build_dynamic_sd_schedule_lookup as upstream_lookup)

    tables = [
        DOC_TABLE,                                  # 需求 §3 的原例
        [(1, 16, 3), (32, 128, 2)],                 # 段间有 17~31 的空隙 → 沿用前段
        [(1, 1, 0), (2, 64, 3)],                    # 首段就是 K=0
        [(1, 4, 5)],                                # 单段、K 超上限 → 被裁剪
        [(1, 8, 0), (9, 16, 2), (20, 20, 7)],       # 尾部 + 超上限一起
        [(2, 4, 2)],                                # 首段不从 1 开始 → 报错
        [(1, 2, 4), (2, 3, 1)],                     # 重叠 → 报错
        [(1, 2, -1)],                               # K 为负 → 报错
        [],                                         # 空表 → 报错
    ]
    for table in tables:
        for max_batch in (1, 2, 4, 8, 16, 64, 200):
            for max_k in (1, 2, 3, 4, 8):
                try:
                    want = upstream_lookup(table, max_batch, max_k)
                except ValueError as exc:
                    want, want_error = None, str(exc)
                else:
                    want_error = None
                if want_error is None:
                    got = build_dynamic_sd_schedule_lookup(table, max_batch, max_k)
                    assert got == want, (table, max_batch, max_k, got, want)
                else:
                    with pytest.raises(ValueError):
                        build_dynamic_sd_schedule_lookup(table, max_batch, max_k)


def test_lookup_fills_gaps_tail_and_clips_to_max_k():
    """不靠差分也把规则钉死一遍（差分用例挂了要能一眼看出是哪条规则错）。"""
    # 需求 071 §3 的例子：最大 K=4、表 [(1,2,4),(5,8,1)] → B=3/4 继承 K=4、B>8 延续 K=1
    dense = build_dynamic_sd_schedule_lookup(DOC_TABLE, vllm_max_batch_size=10,
                                             vllm_num_speculative_tokens=4)
    assert dense[0] == 0, "索引 0 故意不用"
    assert dense[1] == dense[2] == 4
    assert dense[3] == dense[4] == 4, "段间空隙沿用前一段的 K"
    assert dense[5] == dense[8] == 1
    assert dense[9] == dense[10] == 1, "尾部延续最后一段的 K"
    # 超最大 K：表里写 5，而配置的最大 K 是 3 → 实际 3（表只能被裁剪、不能放大容量）
    clipped = build_dynamic_sd_schedule_lookup([(1, 4, 5)], vllm_max_batch_size=4,
                                               vllm_num_speculative_tokens=3)
    assert clipped[1:] == [3, 3, 3, 3]
    # 规范化：排序 + int 转换（上游 `int(entry[i])` 的行为）
    assert validate_and_normalize_dynamic_sd_schedule(
        [(5, 8, "1"), (1, 2, 4)]) == [(1, 2, 4), (5, 8, 1)]


def test_invalid_schedules_rejected_at_config_time():
    """坏表在**配置期**就报错（上游是懒校验，本仓库提前；差异记账本 §6）。"""
    bad_tables = [
        (1, 2, 4),                      # 不是 list
        [],                             # 空表
        [(1, 2)],                       # 不是三元组
        [(0, 2, 1)],                    # 端点非正
        [(4, 2, 1)],                    # 起点 > 终点
        [(1, 2, -1)],                   # K 为负
        [(1, 2, 4), (2, 3, 1)],         # 重叠
        [(2, 4, 2)],                    # 首段不从 1 开始
    ]
    for table in bad_tables:
        with pytest.raises(ValueError):
            _spec(num_speculative_tokens_per_batch_size=table)
    # 合法表：排序 + 归一之后写回配置（后面所有读者都用这一份）
    spec = _spec(num_speculative_tokens_per_batch_size=[(5, 8, 1), (1, 2, 4)])
    assert spec.num_speculative_tokens_per_batch_size == [(1, 2, 4), (5, 8, 1)]
    assert spec.uses_dynamic_speculative_decoding() is True


def test_dynamic_table_requires_positive_max_k():
    """表里的 K 会被最大 K 裁剪，所以最大 K 必须 > 0（上游 lookup 的同一句检查提前）。"""
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        _spec(num_speculative_tokens=0, num_speculative_tokens_per_batch_size=DOC_TABLE)


# ------------------------------------------------------------ 2. 方法边界（真实限制）


def test_only_variable_k_proposers_accept_dynamic_table():
    """只有"提议者真的收逐轮 K"的方法能开动态表，其余**明确报错**（需求 071 §3.6）。"""
    # 支持：llm_base_proposer（draft_model / eagle / eagle3 / mtp）与 CPU ngram
    for method in ("draft_model", "ngram"):
        spec = SpeculativeConfig(method=method, num_speculative_tokens=4,
                                 num_speculative_tokens_per_batch_size=DOC_TABLE)
        assert spec.uses_dynamic_speculative_decoding() is True
    # 不支持：各自的上游提议者里写死了 K（断言/固定宽度），custom_class 则收不到 K。
    # 每个组合都要**先满足该方法自己原有的配置要求**（否则报的是另一个错，掩盖了本关的边界）。
    medusa_hf = dict(HF, architectures=["MedusaModel"])
    extract_hf = dict(HF, eagle_aux_hidden_state_layer_ids=[0, 1])
    unsupported = [
        ("ngram_gpu", {}, "assert num_speculative_tokens == self.k"),
        ("suffix", {}, "suffix"),
        ("medusa", {"draft_hf": medusa_hf}, "medusa"),
        ("extract_hidden_states", {"draft_hf": extract_hf}, "extract_hidden_states"),
        ("custom_class", {"model": "examples.custom_proposer.RepeatLastTokenProposer"},
         "custom_class"),
    ]
    for method, extra, hint in unsupported:
        # 每个方法都要**先满足它自己原有的 K 约束**（extract 恒 K=1），否则报的是另一个错
        extra = dict(extra, num_speculative_tokens=1)
        with pytest.raises(ValueError) as excinfo:
            _spec(method=method, num_speculative_tokens_per_batch_size=DOC_TABLE, **extra)
        message = str(excinfo.value)
        assert "num_speculative_tokens_per_batch_size" in message, (method, message)
        assert hint in message, (method, message)


# ------------------------------------------------------------ 3. 配置改写（Graph / DP / async）


@pytest.mark.parametrize("requested,expected", [
    (None, "PIECEWISE"),                    # 默认（FULL_AND_PIECEWISE）→ 降级
    ("full", "PIECEWISE"),
    ("full_decode_only", "PIECEWISE"),
    ("full_and_piecewise", "PIECEWISE"),
    ("piecewise", "PIECEWISE"),             # 本来就是分段图 → 不动
    ("none", "NONE"),                       # 已经关图 → 不动
])
def test_dynamic_sd_downgrades_full_graph_to_piecewise(requested, expected):
    """动态 K 逐轮改验证长度，full graph 冻结不了形状 → 降级成 PIECEWISE（上游同款）。"""
    config = _config(table=DOC_TABLE, mode=requested, spec_k=4)
    assert str(config.compilation_config.cudagraph_mode) == expected
    # 降级之后档位表按 PIECEWISE 的规则算：**不再**做"取整到 1+K 的倍数"
    sizes = config.compilation_config.cudagraph_capture_sizes
    if expected != "NONE":
        assert 1 in sizes and any(size % 2 for size in sizes), \
            f"PIECEWISE 的档位表不该被 1+K 取整：{sizes}"


def test_static_k_keeps_the_requested_mode():
    """对照组：不开动态表时 `full` 不会被这条规则改写（69b 的解析规则原样保留）。"""
    config = _config(table=None, mode="full_decode_only", spec_k=4)
    assert str(config.compilation_config.cudagraph_mode) in ("FULL_DECODE_ONLY",
                                                             "FULL_AND_PIECEWISE")


def test_data_parallel_fallback_disables_the_table():
    """DP>1 时上游关掉动态表、退回固定 K（各 rank 选不同 K 会分歧/死锁）。

    本仓库没有 DP 轴（单进程单卡），所以规则用**带 `data_parallel_size` 的替身**钉住：
    分支是可达的，不是"写了不跑"。
    """
    config = _config(table=DOC_TABLE, mode="none", spec_k=4)
    assert config.speculative_config.uses_dynamic_speculative_decoding() is True

    class _Parallel:
        data_parallel_size = 2

    # `VllmConfig` 是 frozen dataclass（配置一旦建好就不许随手改），测试用 `object.__setattr__`
    # 挂一个**替身**并调用同一条规则——分支真的会跑，不是"写了不跑"的死代码。
    object.__setattr__(config, "parallel_config", _Parallel())
    config._maybe_disable_dynamic_sd_for_data_parallel()
    assert config.speculative_config.uses_dynamic_speculative_decoding() is False
    assert config.speculative_config.num_speculative_tokens_per_batch_size is None
    assert config.speculative_config.num_speculative_tokens == 4, "退回固定 K，K 本身不变"

    # DP=1（本仓库的正常路径）：什么都不动
    other = _config(table=DOC_TABLE, mode="none", spec_k=4)
    other._maybe_disable_dynamic_sd_for_data_parallel()
    assert other.speculative_config.uses_dynamic_speculative_decoding() is True


def test_dynamic_sd_with_explicit_async_scheduling():
    """显式 async 组合：调度器的占位宽度按**本轮 K**，端到端仍按 70 关的边界明确拒绝。

    为什么两件事都要断言：占位宽度是动态 K 与异步调度的接缝（用最大 K 会让下一轮的预算/
    占位/验证长度同时错位）；而"端到端异步"在本仓库是**冻结的拒绝**，不能因为 71 关
    动了 SchedulerOutput 就悄悄放开（AGENTS §1）。
    """
    engine, core, _runner = make_engine(table=[(1, 1, 0), (2, 8, 2)], spec_k=2, max_num_seqs=2)
    try:
        scheduler = AsyncScheduler(core.vllm_config.scheduler_config, core.kv_cache_manager,
                                   max_model_len=64,
                                   speculative_config=core.vllm_config.speculative_config,
                                   structured_output_manager=core.structured_output_manager)
        make_prompt_requests(engine, ("A",))
        # 借引擎自己的 Request 对象（同一套构造路径），只把调度器换成异步的那个
        scheduler.add_request(core.scheduler.requests["A"])
        first = scheduler.schedule()
        assert first.num_spec_tokens_to_schedule == 0, "B=1 → 表里 K=0"
        assert scheduler.requests["A"].spec_token_ids == [], "K=0 的占位宽度是 0"

        make_prompt_requests(engine, ("B",))
        scheduler.add_request(core.scheduler.requests["B"])
        second = scheduler.schedule()
        assert second.num_spec_tokens_to_schedule == 2, "B=2 → 表里 K=2"
        assert scheduler.requests["A"].spec_token_ids == [-1, -1], \
            "占位草稿的宽度必须是**本轮选出的 K**（不是配置的最大 K）"
    finally:
        engine.shutdown()

    # 端到端 `async_scheduling=True` 仍是 70 关的明确拒绝（动态表不改变这条边界）
    with pytest.raises(NotImplementedError):
        make_engine(table=DOC_TABLE, spec_k=4, async_scheduling=True)


# ------------------------------------------------------------ 4. 调度器：K 随批大小变


def _staggered_dynamic(table, *, spec_k=4, max_num_seqs=4, budget=32):
    """造一个"批大小按 1 → 2 → 3 递进"的场景，返回 `(recorder, engine)`。

    时序是**确定**的（不靠 sleep/抢占）：A 先跑两轮（B=1 → 表里 K=4）；放进 B（B=2 → K=0）；
    再放进 C（B=3 → K=2）。于是 K 真的跨过 **4 → 0 → 2**，而 B=2 那一轮还要**验证 A 在上一轮
    提的 4 枚旧候选**——这正是"改 K 不许重新解释旧候选"的现场。
    """
    engine, core, runner = make_engine(table=table, spec_k=spec_k, max_num_seqs=max_num_seqs,
                                       budget=budget)
    recorder = RoundRecorder(runner, core)
    make_prompt_requests(engine, ("A",), max_tokens=20)
    for _ in range(2):
        engine.step()
    make_prompt_requests(engine, ("B",), max_tokens=8)
    engine.step()
    make_prompt_requests(engine, ("C",), max_tokens=8)
    run_to_end(engine)
    return recorder, engine


def test_scheduler_picks_k_by_scheduled_request_count():
    """K 按**实际被调度的请求数**查表（需求 071 §3.2），并把值放进 SchedulerOutput。"""
    table = [(1, 1, 4), (2, 2, 0), (3, 8, 2)]
    recorder, engine = _staggered_dynamic(table)
    try:
        ks = recorder.ks
        non_empty = [round_ for round_ in recorder.rounds if round_["num_reqs"] > 0]
        assert [round_["num_reqs"] for round_ in non_empty[:4]] == [1, 1, 2, 3], \
            f"场景没按预期递进：{[(r['num_reqs'], r['k']) for r in recorder.rounds]}"
        assert non_empty[0]["k"] == 4 and non_empty[2]["k"] == 0 and non_empty[3]["k"] == 2, \
            f"K 必须跨过 4 → 0 → 2：{ks}"
        for index, round_ in enumerate(recorder.rounds):
            if round_["num_reqs"] == 0:
                # 空轮（结束清理）：没有请求被调度 → 不查表，字段保持默认的最大 K
                # （上游同一句：`if self.dynamic_sd_lookup is not None and
                # len(num_scheduled_tokens) > 0`）
                assert round_["k"] == 4
                continue
            want = table_lookup(table, 4, 4, round_["num_reqs"])
            assert round_["k"] == want, (index, round_, want)
    finally:
        recorder.restore()
        engine.shutdown()


def test_proposal_width_is_this_round_k_and_verification_is_previous_round():
    """两轮时序（需求 071 §3.3）：本轮 K 控制**本轮提**的草稿，本轮验证的是**上一轮**提的候选。

    判据（两条都要求，且是不同来源）：
      * 本轮提出的草稿长度 ≤ 本轮 `SchedulerOutput.num_spec_tokens_to_schedule`
        （K=0 必须一枚都不提；"空闲的列"不许被消费）
      * 本轮验证的旧候选 == **上一轮**提出的那份（逐值相等）——K 变了也不许重新解释它
    """
    table = [(1, 1, 4), (2, 2, 0), (3, 8, 2)]
    recorder, engine = _staggered_dynamic(table)
    try:
        assert 4 in recorder.ks and 0 in recorder.ks and 2 in recorder.ks, recorder.ks
        zero_rounds = [round_ for round_ in recorder.rounds if round_["k"] == 0]
        assert zero_rounds, "场景里必须有 K=0 的轮"
        for round_ in zero_rounds:
            assert all(not drafts for drafts in round_["proposed"].values()), \
                f"K=0 轮不许提草稿：{round_['proposed']}"
        # K=0 那一轮验证的仍然是上一轮（K=4）提的旧候选，且逐值相同
        zero_index = recorder.rounds.index(zero_rounds[0])
        previous = recorder.rounds[zero_index - 1]
        assert previous["k"] == 4
        assert zero_rounds[0]["adopted"], "上一轮提了 4 枚，这一轮就该验证它们"
        for req_id, adopted in zero_rounds[0]["adopted"].items():
            assert adopted == previous["proposed"].get(req_id, []), \
                (f"K 从 4 改成 0 时，{req_id} 的旧候选被重新解释了："
                 f"{adopted} != {previous['proposed'].get(req_id)}")
        for index, round_ in enumerate(recorder.rounds):
            for req_id, drafts in round_["proposed"].items():
                assert len(drafts) <= round_["k"], \
                    f"第 {index} 轮为 {req_id} 提了 {len(drafts)} 枚，超过本轮 K={round_['k']}"
            if index == 0:
                continue
            for req_id, adopted in round_["adopted"].items():
                assert adopted == recorder.rounds[index - 1]["proposed"].get(req_id, []), \
                    (f"第 {index} 轮验证的 {req_id} 候选与第 {index - 1} 轮提的不是同一份："
                     f"{adopted} != {recorder.rounds[index - 1]['proposed'].get(req_id)}")
    finally:
        recorder.restore()
        engine.shutdown()


def test_workspace_addresses_do_not_change_with_k():
    """工作区/KV 预留按最大 K 开一次，**不随逐轮 K 重建**（需求 071 §4 最后一条）。"""
    table = [(1, 1, 0), (2, 8, 3)]
    engine, core, runner = make_engine(table=table, spec_k=3, max_num_seqs=2, budget=32)
    proposer = runner.proposer
    recorder = RoundRecorder(runner, core)
    pointers = []
    try:
        make_prompt_requests(engine, ("A",), max_tokens=6)
        for _ in range(3):
            engine.step()
            pointers.append((proposer.input_ids.data_ptr(), proposer.positions.data_ptr(),
                             proposer.slot_mapping.data_ptr(),
                             proposer.query_start_loc.data_ptr()))
        make_prompt_requests(engine, ("B",), max_tokens=3)
        run_to_end(engine)
    finally:
        recorder.restore()
        engine.shutdown()

    assert 0 in recorder.ks and 3 in recorder.ks, recorder.ks
    assert len(set(pointers)) == 1, f"K 变化时工作区被重建了：{pointers}"


# ------------------------------------------------------------ 5. 端到端：K=0 ↔ K>0


def test_k0_rounds_still_sync_draft_kv_and_output_stays_identical():
    """K=0 的数轮之后恢复 K>0：draft KV 不缺历史，greedy 输出与不开投机**逐 token 相同**。

    三个引擎跑同一组 prompt：不开投机（参考）/ 固定 K=3 / 动态表（B=1 时 K=0、B≥2 时 K=3）。
    "K=0 仍跑第一遍"的证据不看输出（greedy 输出与草稿质量无关），而看
    `first_passes >= 1` 且 `_draft_computed[req] == history_end`：跳过第一遍就不会有这两个数。
    """
    prompts = {"A": (1, 2, 3, 4, 5, 6), "B": (6, 5, 4, 3, 2, 1)}
    eager = greedy(req_ids=("A",), spec_k=None, table=None)
    static = greedy(req_ids=("A",), spec_k=3, table=None)

    table = [(1, 1, 0), (2, 8, 3)]
    engine, core, runner = make_engine(table=table, spec_k=3, max_num_seqs=2, budget=32)
    recorder = RoundRecorder(runner, core)
    try:
        engine.add_request("A", list(prompts["A"]),
                           SamplingParams(max_tokens=6, temperature=0.0, eos_token_id=999))
        for _ in range(3):
            engine.step()
        # 前几轮 B=1 → K=0：第一遍必须跑过，草稿必须是空的
        zero_rounds = [round_ for round_ in recorder.rounds if round_["k"] == 0]
        assert zero_rounds, f"这组表在 B=1 时应当选出 K=0：{recorder.ks}"
        for round_ in zero_rounds:
            assert round_["first_passes"] >= 1, "K=0 也必须跑 draft 第一遍（同步 KV）"
            assert all(not drafts for drafts in round_["proposed"].values()), \
                "K=0 不许交回草稿"
        for _ in range(2):
            engine.step()
        assert runner.proposer._draft_computed.get("A") == runner.requests["A"].num_tokens, \
            "K=0 轮结束后 draft 侧进度必须与 target 的已提交历史一致（第一遍真的跑了）"
        # 199 §9 的不变量：**能发布的进度不能超过 draft 已经写过的位置**（draft 缺历史
        # 时发布出去的块在 draft 那几层是空的，别的请求命中就读到垃圾）。
        for req_id, request in core.scheduler.requests.items():
            assert core.scheduler.publish_bound(request) <= \
                runner.proposer._draft_computed.get(req_id, 0), \
                f"{req_id} 的发布上界超过了 draft 已同步的进度（K=0 轮漏了第一遍）"

        # 放第二条请求进来：B=2 → K=3，恢复猜 3 枚
        engine.add_request("B", list(prompts["B"]),
                           SamplingParams(max_tokens=4, temperature=0.0, eos_token_id=999))
        outputs = run_to_end(engine)
    finally:
        recorder.restore()
        engine.shutdown()

    assert 3 in recorder.ks, f"恢复 K>0 的那几轮没出现：{recorder.ks}"
    assert outputs["A"] == eager["A"] == static["A"], \
        (f"动态 K 的 greedy 输出必须与不开投机逐 token 相同："
         f"{outputs['A']} vs {eager['A']} vs {static['A']}")


@requires_cuda
def test_dynamic_sd_actually_runs_piecewise_graphs():
    """真实引擎上：动态表 + 默认模式 → 只有 PIECEWISE 命中，绝不命中 FULL（配置改写的落点）。"""
    engine, core, runner = make_engine(table=[(1, 1, 0), (2, 8, 2)], spec_k=2,
                                       max_num_seqs=2, budget=32)
    try:
        assert str(runner.compilation_config.cudagraph_mode) == "PIECEWISE"
        make_prompt_requests(engine, ("A", "B"), max_tokens=4)
        run_to_end(engine)
        modes = {selection["mode"] for selection in runner.cudagraph_selections}
    finally:
        engine.shutdown()
    assert modes <= {"PIECEWISE", "NONE"}, f"动态投机长度下不该出现 FULL 图：{modes}"
    assert "PIECEWISE" in modes, f"混合/不等宽的批应当走分段图：{modes}"


@requires_cuda
def test_q_rows_follow_this_round_proposal_width():
    """概率草稿：`q` 的行数 = 本轮**实际提议**的草稿数（逐轮变宽时也不能掺旧轮的行）。

    为什么单独测：`_get_spec_decode_draft_probs()` 是按"上一轮的提议顺序 + 本轮采用的前缀"
    重拼 q 的；K 逐轮变时若按最大 K 或按旧矩阵前 P 行取，**不会报错**，只会把别人的概率
    配到这条请求的草稿上。
    """
    engine, core, runner = make_engine(table=[(1, 1, 0), (2, 8, 2)], spec_k=2, max_num_seqs=2,
                                       budget=32, draft_sample_method="probabilistic")
    recorder = RoundRecorder(runner, core)
    try:
        # 先只放 A（B=1 → K=0），再放 B（B=2 → K=2）：q 的宽度必须随轮次变
        make_prompt_requests(engine, ("A",), max_tokens=4)
        for _ in range(2):
            engine.step()
        make_prompt_requests(engine, ("B",), max_tokens=4)
        run_to_end(engine)
    finally:
        recorder.restore()
        engine.shutdown()

    checked = 0
    for index, round_ in enumerate(recorder.rounds):
        shape = round_["drafts_gpu"]
        if shape is None:
            continue
        num_proposed = sum(len(drafts) for drafts in round_["proposed"].values())
        assert num_proposed > 0
        assert shape[0] == num_proposed, (index, shape, round_["proposed"])
        if index + 1 < len(recorder.rounds):
            # 下一轮采用的草稿数不会超过这一轮提的（q 也就不会缺行）
            num_adopted = sum(len(v) for v in recorder.rounds[index + 1]["adopted"].values())
            assert num_adopted <= num_proposed
        checked += 1
    assert checked >= 2, "至少要有两轮带 q 的提议才算测到（含宽度变化）"
    assert 0 in recorder.ks and 2 in recorder.ks, recorder.ks


def test_ngram_dynamic_k_does_not_consume_stale_columns():
    """ngram 的草稿缓冲按**最大 K** 开：逐轮变小后，旧的宽列不许被当成有效草稿交出去。

    这是需求 071 §4 最后一条（"旧无效列不被消费"）在 ngram 上的落点：它的 `valid_ngram_draft`
    是定宽 `[max_num_seqs, k]` 的预分配缓冲，只有 `valid_ngram_num_drafts[i]` 说了"这行有效几枚"。
    走 CPU 路径（不需要 GPU 内核），所以任何机器上都能跑。
    """
    from minivllm.spec_decode.ngram_proposer import NgramProposer
    from minivllm.spec_decode.utils import TargetRows

    spec = _spec(method="ngram", num_speculative_tokens=4, prompt_lookup_min=2,
                 prompt_lookup_max=2)
    config = VllmConfig(
        model_config=ModelConfig(model=TINY, dtype="float32", max_model_len=64, hf_config=HF),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=32),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=32),
        device_config=DeviceConfig("cpu"), speculative_config=spec)
    proposer = NgramProposer(config)
    # 历史里放一段重复的 6-gram：匹配点后面能抄满 4 枚（K=4 时正好 [1,2,3,4]）
    history = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
    rows = [TargetRows(req_id="A", row=0, start=0, target_rows=len(history),
                       num_rejected=0, history_end=len(history), next_token_id=6, ready=True)]

    full = proposer.propose_drafts(rows, {"A": history}, input_batch=None,
                                   num_speculative_tokens=4)
    assert full.draft_token_ids[0] == [1, 2, 3, 4], full.draft_token_ids

    short = proposer.propose_drafts(rows, {"A": history}, input_batch=None,
                                    num_speculative_tokens=1)
    assert short.draft_token_ids[0] == full.draft_token_ids[0][:1], \
        f"K=1 时只能交回第一枚，不能把上一轮的宽列漏出来：{short.draft_token_ids}"

    empty = proposer.propose_drafts(rows, {"A": history}, input_batch=None,
                                    num_speculative_tokens=0)
    assert empty.draft_token_ids == [[]], empty.draft_token_ids

    with pytest.raises(ValueError, match="超出"):
        proposer.propose_drafts(rows, {"A": history}, input_batch=None,
                                num_speculative_tokens=5)
