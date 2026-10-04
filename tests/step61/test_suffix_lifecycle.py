"""step61：请求内 / 跨请求历史（需求 §3 调用顺序、§4 验收）——与冻结的上游提议者逐步比对。

需求 §4 要求"逐事件 trace 对照冻结 proposer 和依赖"：本文件把同一串事件（请求进入/离开批、
采样、行序变化、容量变化）喂给我们那份 `SuffixDecodingProposer` 与**上游那份**，逐步比对
候选与缓存状态；再单独用 spy 断言 §3 的六步调用顺序与参数。

上游的 `SuffixDecodingProposer` 不吃 Runner、只吃 config 的 6 个字段 + InputBatch 的 5 个只读
属性，所以能原样实例化当基准（见 `suffix_helpers.upstream_proposer`）。

**依赖包的匹配语义**（读 `csrc/suffix_decoding/suffix_tree.cc:593-622` + 实测，写在这里免得
误读测试期望）：
    - `speculate(context)` 从 `match_len = 1` **递增**地取 context 的**后缀**去树里找，
      某个长度找不到就**停**（后缀树性质：更长的也一定找不到）；因此最终用的是"能匹配上的
      最长后缀"，但**全长 context 永不参与匹配**（循环上界是 `context.size()`），最后那个
      token 只当锚点。
    - 每个匹配的得分是路径上概率之和，`_speculate_path` 逐 token 乘"子节点频次/父节点频次"，
      低于 `min_token_prob` 就停；token 数上限 = `min(K, match_len * factor + offset)`。
    - 所以"上下文正好等于历史末尾"时候选常为空（末尾那次出现没有后继）；要出候选，
      最近几个 token 必须在更早的位置**带着同样的后继**出现过——这正是重复性 prompt /
      跨请求共享模式能命中的原因。
"""

import pytest

from suffix_helpers import (our_proposer, run_segments_on, run_trace, run_trace_on)

K = 8
# 单请求就命中（"1 2 3 4" 在 prompt 里出现过两次，锚点 token=1）
REPEAT_PROMPT = {"r0": [1, 2, 3, 4, 1, 2, 3, 4]}
# 跨请求：A 教会全局树 "…1 2 3 → 4 5"，B 的结尾也是 "1 2 3"
CROSS_PROMPTS = {"A": [1, 2, 3, 9, 9, 1, 2, 3], "B": [7, 8, 1, 2, 3]}
CROSS_STEPS = [{"order": ["A"], "sampled": {"A": [4]}},
               {"order": ["A"], "sampled": {"A": [5]}},
               {"order": ["B"], "sampled": {"B": [4]}}]


def _both(prompts, steps, *, k=K, max_model_len=64, **config):
    """同一串事件跑两个实现，返回 `(ours, upstream)`。"""
    ours = run_trace_on("ours", max_model_len=max_model_len, prompts=prompts, steps=steps,
                        num_speculative_tokens=k, **config)
    upstream = run_trace_on("upstream", max_model_len=max_model_len, prompts=prompts, steps=steps,
                            num_speculative_tokens=k, **config)
    return ours, upstream


def _assert_same(ours, upstream):
    """两个实现的候选与缓存状态逐步相同（本关的主断言）。"""
    assert ours["drafts"] == upstream["drafts"]
    assert ours["states"] == upstream["states"]


# ---------------------------------------------------------------------------
# 1. 单请求：建树 → 追加输出 → 逐步提议（长度动态、可多可少）
# ---------------------------------------------------------------------------


def test_single_request_five_steps_matches_upstream():
    """单请求 5 步，每步候选都非空且逐步与上游相同（不是"两边都空"蒙过去）。"""
    steps = [{"order": ["r0"], "sampled": {"r0": [token]}} for token in (1, 2, 3, 4, 1)]
    ours, upstream = _both(REPEAT_PROMPT, steps)
    _assert_same(ours, upstream)
    assert ours["drafts"] == [[[2, 3, 4, 1]], [[3, 4, 1, 2]], [[4, 1, 2, 3]],
                             [[1, 2, 3, 4]], [[2, 3, 4, 1, 2]]]
    assert ours["states"][-1] == ({"r0"}, {"r0"})


def test_num_speculative_tokens_caps_draft_length():
    """`num_speculative_tokens` 是**上限**：K=1/2/8 依次给出 1/2/4 枚候选。"""
    steps = [{"order": ["r0"], "sampled": {"r0": [1]}}]
    lengths = {}
    for k in (1, 2, 8):
        ours, upstream = _both(REPEAT_PROMPT, steps, k=k)
        _assert_same(ours, upstream)
        lengths[k] = ours["drafts"][0][0]
    assert lengths[1] == [2] and lengths[2] == [2, 3] and lengths[8] == [2, 3, 4, 1]


def test_max_model_len_caps_draft_length():
    """`max_spec_tokens = min(K, max_model_len - num_tokens - 1)`：贴着上限时只猜得动 1 枚。"""
    steps = [{"order": ["r0"], "sampled": {"r0": [1]}}]  # 记账后 num_tokens = 9
    ours, upstream = _both(REPEAT_PROMPT, steps, max_model_len=11)
    _assert_same(ours, upstream)
    assert ours["drafts"][0][0] == [2], ours["drafts"]


@pytest.mark.parametrize("factor,expected", [(1.0, [2, 3, 4, 1]), (0.5, [2, 3]), (0.1, [])])
def test_max_spec_factor_is_applied(factor, expected):
    """`max_spec_factor` 按匹配长度限长（`match_len * factor`）；0.1 时直接一枚都不猜。"""
    steps = [{"order": ["r0"], "sampled": {"r0": [1]}}]
    ours, upstream = _both(REPEAT_PROMPT, steps, max_spec_factor=factor)
    _assert_same(ours, upstream)
    assert ours["drafts"][0][0] == expected, ours["drafts"]


@pytest.mark.parametrize("prob,expected", [(0.1, [5]), (0.5, [5]), (0.51, []), (1.0, [])])
def test_min_token_prob_is_applied(prob, expected):
    """`min_token_prob` 按频次估计的概率过滤：该分支概率 0.5，阈值 0.51 就把它挡掉。"""
    # "1 2 3" 后面跟过 4(1 次) 和 5(2 次)，锚点 token=3 让匹配落在 "1 2 3" 上
    prompts = {"r0": [7, 7, 1, 2, 3, 1, 2, 3, 5, 1, 2, 3, 5]}
    steps = [{"order": ["r0"], "sampled": {"r0": [3]}}]
    ours, upstream = _both(prompts, steps, min_token_prob=prob)
    _assert_same(ours, upstream)
    assert ours["drafts"][0][0] == expected, ours["drafts"]


@pytest.mark.parametrize("depth,expected", [(24, [2, 3, 4, 1]), (3, [2])])
def test_max_tree_depth_changes_context_and_match(depth, expected):
    """`max_tree_depth` 只管"取最近多少 token 当上下文"（树上限同值），截短后匹配也变短。"""
    prompts = {"r0": [7, 7, 1, 2, 3, 4, 1, 2, 3, 4]}
    steps = [{"order": ["r0"], "sampled": {"r0": [1]}}]
    ours, upstream = _both(prompts, steps, max_tree_depth=depth)
    _assert_same(ours, upstream)
    assert ours["drafts"][0][0] == expected, ours["drafts"]


# ---------------------------------------------------------------------------
# 2. 跨请求复用（本关的痛点）：全局树
# ---------------------------------------------------------------------------


def test_global_tree_gives_cross_request_drafts():
    """A 的输出进全局树 → B 只凭自己历史猜不出来，却能从 A 那里拿到 [5]。"""
    ours, upstream = _both(CROSS_PROMPTS, CROSS_STEPS)
    _assert_same(ours, upstream)
    assert ours["drafts"][0] == [[]] and ours["drafts"][1] == [[]]
    assert ours["drafts"][2] == [[5]], ours["drafts"]
    assert ours["states"][2][1] == {"A", "B"}, "两条请求都应在全局缓存里"


def test_disabling_global_tree_kills_cross_request_drafts_but_keeps_prompt_tree():
    """`max_cached_requests=0`：跨请求候选消失（B 拿不到 [5]），prompt 树照旧工作。"""
    ours, upstream = _both(CROSS_PROMPTS, CROSS_STEPS, max_cached_requests=0)
    _assert_same(ours, upstream)
    assert ours["drafts"][2] == [[]], ours["drafts"]
    assert ours["states"][2] == ({"B"}, set()), "全局缓存必须为空"

    # 同一个开关下，自己的 prompt 树仍然能猜（"1 2 3 4" 在 prompt 里出现两次）
    local, upstream_local = _both(REPEAT_PROMPT, [{"order": ["r0"], "sampled": {"r0": [1]}}],
                                 max_cached_requests=0)
    _assert_same(local, upstream_local)
    assert local["drafts"][0][0] == [2, 3, 4, 1]


@pytest.mark.parametrize("capacity,expected_c", [((2), []), ((3), [5]), ((10000), [5])])
def test_fifo_eviction_by_capacity(capacity, expected_c):
    """容量 2 时 C 进来先淘汰最早的 A（FIFO）→ C 拿不到 [5]；容量 3 时还在 → 拿得到。"""
    prompts = {"A": [1, 2, 3, 9, 9, 1, 2, 3], "B": [7, 8, 1, 2, 3], "C": [9, 9, 1, 2, 3]}
    steps = CROSS_STEPS + [{"order": ["C"], "sampled": {"C": [4]}}]
    ours, upstream = _both(prompts, steps, max_cached_requests=capacity)
    _assert_same(ours, upstream)
    assert ours["drafts"][2] == [[5]], "B 依赖 A（此时 A 还在缓存里）"
    assert ours["drafts"][3] == [expected_c], ours["drafts"]
    assert ours["states"][3][1] == ({"B", "C"} if capacity == 2 else {"A", "B", "C"})


@pytest.mark.parametrize("max_cached_requests", [0, 1, 2, 10000])
def test_capacity_paths_match_upstream(max_cached_requests):
    """容量 0/1/2/大 四条路径逐步与上游一致。"""
    ours, upstream = _both(CROSS_PROMPTS, CROSS_STEPS,
                           max_cached_requests=max_cached_requests)
    _assert_same(ours, upstream)


def test_negative_capacity_rejected_like_upstream():
    """上游 `_validate_suffix_decoding` 只接受 `cached_requests >= 0`，-1 要报错（不能照抄包里的"负数=不限"）。"""
    from minivllm.config import SpeculativeConfig

    with pytest.raises(ValueError, match="suffix_decoding_max_cached_requests"):
        SpeculativeConfig(method="suffix", suffix_decoding_max_cached_requests=-1)


# ---------------------------------------------------------------------------
# 3. 批的变化：行序、缺席、中间 prefill、到达长度上限
# ---------------------------------------------------------------------------


def test_row_reordering_and_absence_match_upstream():
    """行序反转、某条请求中途缺席一步（被抢占/预算不够）、再回来。"""
    prompts = {"rA": [1, 2, 3, 4, 1, 2, 3, 4], "rB": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]}
    steps = [{"order": ["rA", "rB"], "sampled": {"rA": [1], "rB": [1]}},
             {"order": ["rB", "rA"], "sampled": {"rB": [2], "rA": [2]}},
             {"order": ["rA"], "sampled": {"rA": [3]}},                 # rB 缺席
             {"order": ["rB", "rA"], "sampled": {"rB": [4], "rA": [4]}},
             {"order": ["rA"], "sampled": {"rA": [1]}}]
    ours, upstream = _both(prompts, steps)
    _assert_same(ours, upstream)
    assert all(draft for step in ours["drafts"] for draft in step), "每步都该有候选"
    # rB 中途缺席 → "活跃但不在批里" → 上游末尾那次扫描把它停掉（局部树丢弃，全局缓存保留）
    assert ours["states"][2] == ({"rA"}, {"rA", "rB"}), ours["states"][2]
    assert ours["states"][3][0] == {"rB", "rA"}, "rB 回来会重新建 prompt 树"


def test_partial_prefill_row_is_skipped_without_side_effects():
    """中间 prefill 块（采样为空）不建树、不追加、不提议；输出第一枚后才开始猜。"""
    steps = [{"order": ["r0"], "sampled": {}, "visible": {"r0": 4}},
             {"order": ["r0"], "sampled": {}, "visible": {"r0": 8}},
             {"order": ["r0"], "sampled": {"r0": [1]}}]
    ours, upstream = _both(REPEAT_PROMPT, steps)
    _assert_same(ours, upstream)
    assert ours["drafts"][0] == [[]] and ours["drafts"][1] == [[]]
    assert ours["states"][0] == (set(), set()) and ours["states"][1] == (set(), set())
    assert ours["drafts"][2] == [[2, 3, 4, 1]], ours["drafts"]
    assert ours["states"][2] == ({"r0"}, {"r0"})


def test_max_model_len_row_returns_empty_and_is_not_started():
    """到达上限的行：候选为空，且**连树都不建**（上游把判断放在建树之前）。

    rA 的 prompt 长 5 = `max_model_len - 1`：它的**第一步（prefill）就已经把 num_tokens
    顶到 6**，所以整条请求从未进过树；rB 还没到上限，照常出候选。
    """
    prompts = {"rA": [1, 2, 3, 4, 1], "rB": [1, 2, 3]}
    steps = [{"order": ["rA", "rB"], "sampled": {"rA": [9], "rB": [1]}}]
    ours, upstream = _both(prompts, steps, max_model_len=6)
    _assert_same(ours, upstream)
    assert ours["drafts"][0][0] == [], ours["drafts"]
    assert ours["drafts"][0][1] == [2], ours["drafts"]
    assert ours["states"][0] == ({"rB"}, {"rB"}), "rA 到上限 → 连 prompt 树都不该建"


# ---------------------------------------------------------------------------
# 4. 生命周期收尾：请求结束、同 ID 重用、两条清理路径等价
# ---------------------------------------------------------------------------


def test_id_reuse_evicts_previous_response_from_global_tree():
    """同 ID 重用：新请求只看得到**自己**那条序列；不做 evict 就会继承上一轮的响应。

    第一条请求 P1 教会全局树 "…4 → 9 5"；它结束后，同 ID 的第二条请求换成 P2（不同的
    prompt），第一步也采出 9。evict 生效时第二条请求的候选是空的（它自己的 prompt 里没有
    可匹配的重复模式）；把 `evict_cached_response` 变成空操作，旧序列仍在全局树里，
    第二条请求就会拿到沿 **上一条请求** 响应继续的候选 [5]（draft 只是猜测、不影响最终
    正确性，但候选"从哪来"完全不同——这正是上游要在同 ID 重用时先 evict 的原因）。

    依赖包的 `start_request` 自己也会 evict 已缓存的同 ID 响应，所以"不 evict"要用
    monkeypatch 构造，属于反证用例；正常路径逐步与上游比对。
    """
    p1, p2 = [1, 2, 3, 4, 1, 2, 3, 4], [7, 7, 1, 2, 3, 4]
    segments = [{"prompts": {"r0": p1},
                 "steps": [{"order": ["r0"], "sampled": {"r0": [9]}},
                           {"order": ["r0"], "sampled": {"r0": [5]}}],
                 "stop": ["r0"]},
                {"prompts": {"r0": p2},
                 "steps": [{"order": ["r0"], "sampled": {"r0": [9]}}]}]
    ours = run_segments_on("ours", max_model_len=64, segments=segments, num_speculative_tokens=K)
    upstream = run_segments_on("upstream", max_model_len=64, segments=segments,
                               num_speculative_tokens=K)
    _assert_same(ours, upstream)
    assert ours["drafts"] == [[[]], [[]], [[]]], ours["drafts"]
    assert ours["states"][-1] == ({"r0"}, {"r0"}), "重建后活跃与全局缓存都只有这条新请求"

    proposer = our_proposer(num_speculative_tokens=K, max_model_len=64)
    run_trace(proposer, max_model_len=64, prompts={"r0": p1}, steps=segments[0]["steps"],
              num_speculative_tokens=K)
    proposer.suffix_cache.evict_cached_response = lambda req_id: None  # 不 evict
    proposer.suffix_cache.stop_request("r0")
    stale = run_trace(proposer, max_model_len=64, prompts={"r0": p2},
                      steps=segments[1]["steps"], num_speculative_tokens=K)
    assert stale["drafts"] == [[[5]]], stale["drafts"]


def test_remove_requests_matches_empty_batch_sweep():
    """两条清理路径等价：Runner 的 `remove_requests`（结束通知）与批里缺席时的末尾扫描。"""
    prompts = {"r0": [1, 2, 3, 4, 1, 2, 3, 4]}
    after_first = {"r0": [1, 2, 3, 4, 1, 2, 3, 4, 1]}
    first = {"order": ["r0"], "sampled": {"r0": [1]}}
    third = {"order": ["r0"], "sampled": {"r0": [2]}}
    via_sweep = run_trace_on("ours", max_model_len=64, prompts=prompts,
                             steps=[first, {"order": [], "sampled": {}}, third],
                             num_speculative_tokens=K)

    proposer = our_proposer(num_speculative_tokens=K, max_model_len=64)
    run_trace(proposer, max_model_len=64, prompts=prompts, steps=[first],
              num_speculative_tokens=K)
    assert proposer.suffix_cache.active_requests == {"r0"}
    proposer.remove_requests(["r0"])
    assert proposer.suffix_cache.active_requests == set(), "结束通知必须摘掉活跃标记"
    assert proposer.suffix_cache.cached_requests == {"r0"}, "全局缓存不能被结束通知删掉"
    proposer.remove_requests(["r0"])  # 重复调用（已停/已结束）必须是安全的空操作
    via_hook = run_trace(proposer, max_model_len=64, prompts=prompts, history=after_first,
                         steps=[third], num_speculative_tokens=K)

    assert via_hook["drafts"] == [via_sweep["drafts"][2]]
    assert via_hook["states"] == [via_sweep["states"][2]]


# ---------------------------------------------------------------------------
# 5. 调用顺序（需求 §3 的六步）与候选归属（§4）
# ---------------------------------------------------------------------------


def _record_cache_calls(proposer):
    """把依赖包上的 5 个方法换成记录器：返回 `(calls, restore)`。

    记录形式 `(方法名, 位置参数元组, kwargs)`；`start_request` 内部对"同 ID 已缓存"的
    evict 也会被记到（依赖包自己会调 `self.evict_cached_response`，实例属性替换后同样生效）。
    """
    cache = proposer.suffix_cache
    names = ("start_request", "evict_cached_response", "add_active_response",
             "speculate", "stop_request")
    originals = {name: getattr(cache, name) for name in names}
    calls: list[tuple] = []

    def wrap(name):
        original = originals[name]

        def recorded(*args, **kwargs):
            calls.append((name, args, kwargs))
            return original(*args, **kwargs)

        return recorded

    for name in names:
        setattr(cache, name, wrap(name))

    def restore():
        for name, original in originals.items():
            setattr(cache, name, original)

    return calls, restore


def test_call_order_and_arguments_follow_requirement_section_3():
    """§3 六步顺序：先 start → 再 add（只加本轮采样）→ 再 speculate；离批的活跃请求末尾 stop。"""
    proposer = our_proposer(num_speculative_tokens=K, max_model_len=64)
    calls, restore = _record_cache_calls(proposer)
    try:
        run_trace(proposer, max_model_len=64, prompts=REPEAT_PROMPT,
                  steps=[{"order": ["r0"], "sampled": {"r0": [1]}}], num_speculative_tokens=K)
        first_round = list(calls)
        calls.clear()
        run_trace(proposer, max_model_len=64, prompts=REPEAT_PROMPT,
                  steps=[{"order": [], "sampled": {}}], num_speculative_tokens=K)
        absent_round = list(calls)
        calls.clear()
        # 同 ID 新请求：先 evict 旧响应，再 start，再 add，再 speculate
        run_trace(proposer, max_model_len=64, prompts=REPEAT_PROMPT,
                  steps=[{"order": ["r0"], "sampled": {"r0": [1]}}], num_speculative_tokens=K)
        reuse_round = list(calls)
    finally:
        restore()

    assert [name for name, _args, _kwargs in first_round] == [
        "start_request", "add_active_response", "speculate"], first_round
    # 建树用的是 prompt；追加的**只有**本轮采样出来的 token（不含草稿）
    start_args = first_round[0][1]
    assert start_args[0] == "r0"
    assert start_args[1].tolist() == REPEAT_PROMPT["r0"], start_args[1]
    add_args = first_round[1][1]
    assert add_args[0] == "r0" and list(add_args[1]) == [1], add_args
    # speculate 的上下文是"最近 max_tree_depth 个 token"的尾部 pattern，且带上限与阈值
    spec_args, spec_kwargs = first_round[2][1], first_round[2][2]
    assert spec_args[0] == "r0"
    assert spec_args[1].tolist() == [1, 2, 3, 4, 1, 2, 3, 4, 1], spec_args[1]
    assert spec_kwargs["max_spec_tokens"] == min(K, 64 - 9 - 1) == 8
    assert spec_kwargs["max_spec_factor"] == 1.0 and spec_kwargs["min_token_prob"] == 0.1
    # 空采样：什么都不做；不在批里的活跃请求在末尾被停掉（§3 第 6 步）
    assert [name for name, _args, _kwargs in absent_round] == ["stop_request"]
    assert absent_round[0][1][0] == "r0"
    # 同 ID 重用：evict 在 start 之前（§3 第 3 步）；随后流程与首次一致
    assert [name for name, _args, _kwargs in reuse_round] == [
        "evict_cached_response", "start_request", "add_active_response", "speculate"], reuse_round


def test_partial_prefill_does_not_call_the_cache_at_all():
    """§3 第 1 步：空采样（中间 prefill）直接跳过——依赖包上一个方法都不该被调用。"""
    proposer = our_proposer(num_speculative_tokens=K, max_model_len=64)
    calls, restore = _record_cache_calls(proposer)
    try:
        run_trace(proposer, max_model_len=64, prompts=REPEAT_PROMPT,
                  steps=[{"order": ["r0"], "sampled": {}, "visible": {"r0": 4}}],
                  num_speculative_tokens=K)
    finally:
        restore()
    assert calls == [], calls


def test_drafts_follow_request_identity_not_row_order():
    """不同请求的候选长度不同，且跟着**请求身份**走：换行序后候选跟着换。"""
    prompts = {"rA": [1, 2, 3, 4, 1, 2, 3, 4], "rB": [9, 8]}
    forward = {"order": ["rA", "rB"], "sampled": {"rA": [1], "rB": [7]}}
    backward = {"order": ["rB", "rA"], "sampled": {"rB": [7], "rA": [1]}}
    ours_f, up_f = _both(prompts, [forward])
    ours_b, up_b = _both(prompts, [backward])
    _assert_same(ours_f, up_f)
    _assert_same(ours_b, up_b)
    assert ours_f["drafts"][0] == [[2, 3, 4, 1], []], ours_f["drafts"]
    assert ours_b["drafts"][0] == [[], [2, 3, 4, 1]], ours_b["drafts"]
