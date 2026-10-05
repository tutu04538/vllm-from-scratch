"""step62：自定义 Proposer 的接入与配置分派边界。

四段：
  A. 方法推断只在**配置期**发生一次（含 `_is_custom_proposer_path` 的边界判定）
  B. `create_custom_proposer` 的接口与**五类错误分类**（含异常链）
  C. Runner 接线：合法实例就是 Runner 持有的对象；调用参数与冻结的上游分支一致；插件拿不到运行时状态
  D. 行为：空候选 / 全错候选 / 变长候选都不改变 greedy 最终答案（含中间 prefill 行）

本模块里定义的几个类会被**按点号路径导入**（`model="test_custom_proposer.XxxProposer"`），
所以它们必须写在这个文件里、并且不带构造参数依赖（除了 `vllm_config`）。
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step59"))

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,
                      SchedulerConfig, SpeculativeConfig, UniProcExecutor, VllmConfig, Worker)
from minivllm.spec_decode.custom_class_proposer import create_custom_proposer
from minivllm.testing.tiny_models import tiny_qwen3_config, tiny_qwen3_dir

EXAMPLE_PATH = "examples.custom_proposer.RepeatLastTokenProposer"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# 测试用的提议者类（会被按点号路径导入）
# ---------------------------------------------------------------------------


class EmptyProposer:
    """永远给空候选（等价于"这一轮不投机"，但走的是插件路径）。"""

    calls = 0

    def __init__(self, vllm_config):
        self.num_speculative_tokens = vllm_config.speculative_config.num_speculative_tokens

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None):
        EmptyProposer.calls += 1
        return [[] for _ in sampled_token_ids]


class AllWrongProposer:
    """永远猜 token 0：候选基本全被拒，用来验证"错误候选不改答案"。"""

    def __init__(self, vllm_config):
        self.k = vllm_config.speculative_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None):
        drafts = []
        for row, sampled in enumerate(sampled_token_ids):
            if not sampled:
                drafts.append([])
                continue
            budget = min(self.k, self.max_model_len - int(num_tokens_no_spec[row]) - 1)
            drafts.append([0] * max(budget, 0))
        return drafts


class VariableLengthProposer:
    """每行给的长度不同（行号越大越长），用来验证变长候选的归属正确。"""

    def __init__(self, vllm_config):
        self.k = vllm_config.speculative_config.num_speculative_tokens

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None):
        drafts = []
        for row, sampled in enumerate(sampled_token_ids):
            if not sampled:
                drafts.append([])
                continue
            length = min(self.k, row + 1)
            drafts.append([int(sampled[-1])] * length)
        return drafts


class RecordingProposer:
    """记录构造参数与每次调用参数（测试用）。"""

    constructed: list = []
    calls: list = []

    def __init__(self, *args, **kwargs):
        RecordingProposer.constructed.append((args, kwargs))
        self.vllm_config = args[0] if args else None
        spec = self.vllm_config.speculative_config
        self.k = spec.num_speculative_tokens
        self.max_model_len = self.vllm_config.model_config.max_model_len

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None):
        RecordingProposer.calls.append(
            {"rows": [list(row) for row in sampled_token_ids],
             "num_tokens": [int(x) for x in num_tokens_no_spec],
             "token_ids_shape": tuple(token_ids_cpu.shape),
             "slot_mappings": slot_mappings})
        drafts = []
        for row, sampled in enumerate(sampled_token_ids):
            if not sampled:
                drafts.append([])
                continue
            budget = min(self.k, self.max_model_len - int(num_tokens_no_spec[row]) - 1)
            drafts.append([int(sampled[-1])] * max(budget, 0))
        return drafts


class BoomOnInitProposer:
    def __init__(self, vllm_config):
        raise RuntimeError("构造器故意炸（测试异常链）")


class NotCallableProposeProposer:
    def __init__(self, vllm_config):
        self.propose = 123


class NoProposeProposer:
    def __init__(self, vllm_config):
        self.something = 1


# ---------------------------------------------------------------------------
# A. 方法推断边界（配置期一次判定）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", [
    "examples.custom_proposer.RepeatLastTokenProposer",
    "my_module.MyCustomProposer",
    "pkg.sub.Class",
])
def test_dotted_path_infers_custom_class(model):
    """点号路径 → method 自动推成 custom_class（不用显式指定）。"""
    config = SpeculativeConfig(model=model, num_speculative_tokens=4)
    assert config.method == "custom_class"
    assert config.model == model


@pytest.mark.parametrize("model,expected", [
    ("ngram", "ngram"),
    ("[ngram]", "ngram"),
    ("Qwen/Qwen3-0.6B", "draft_model"),          # 带 `/` = HF 模型名 → draft_model
    ("https://huggingface.co/a.b", "draft_model"),  # URL 不是类路径
    ("file:///tmp/m.Class", "draft_model"),
    ("single", "draft_model"),                   # 没有点号
    ("my-module.Class", "draft_model"),          # 段不是合法标识符
    ("a.1b", "draft_model"),
    (None, "draft_model"),                       # 什么都没给
])
def test_non_custom_paths_keep_upstream_defaults(model, expected):
    """其余情况按上游默认走（`model="ngram"` → ngram，其它 → draft_model）。"""
    assert SpeculativeConfig(model=model).method == expected


def test_explicit_method_wins_over_inference():
    """显式给的 method 优先：`method="ngram"` + 像类路径的 model 仍然是 ngram。"""
    assert SpeculativeConfig(method="ngram", model="a.b.C").method == "ngram"
    assert SpeculativeConfig(method="custom_class", model="a.b.C").method == "custom_class"


def test_custom_class_requires_model_path():
    """显式 custom_class 但没给 model → 配置期 ValueError（照抄上游文案）。"""
    with pytest.raises(ValueError, match="requires 'model' to contain the custom proposer"):
        SpeculativeConfig(method="custom_class")


def test_unknown_method_rejected_at_config_time():
    """未支持的方法在**配置期**拒绝，不静默回退成 ngram。

    （63 关起 `eagle`/`eagle3` 已被支持，所以这条用例改用仍未实现的 `medusa`；
    断言强度不变——仍然是"配置期 ValueError + 不回退"。）
    """
    with pytest.raises(ValueError, match="只支持 method="):
        SpeculativeConfig(method="medusa", num_speculative_tokens=4)


def test_config_derived_quantities_for_custom_class():
    """派生量与 ngram 同档：不写 KV → lookahead 0；不占输入槽位 → draft_slots 0。"""
    config = SpeculativeConfig(method="custom_class", model="a.b.C", num_speculative_tokens=4)
    assert config.uses_draft_model() is False
    assert config.use_ngram_gpu() is False
    assert config.max_num_new_slots_for_drafting == 0


def test_runner_does_not_re_guess_the_method():
    """分派边界：Runner 只按 `config.method` 分派，不再自己判一遍点号路径。

    源码级检查（比"跑一遍看行为"更直接）：`_is_custom_proposer_path` 只在 config.py 里出现。
    """
    runner_source = (ROOT / "minivllm" / "worker" / "gpu_model_runner.py").read_text()
    config_source = (ROOT / "minivllm" / "config.py").read_text()
    assert "_is_custom_proposer_path" in config_source
    assert "_is_custom_proposer_path" not in runner_source
    assert 'config.method == "custom_class"' in runner_source


# ---------------------------------------------------------------------------
# B. create_custom_proposer：接口与五类错误
# ---------------------------------------------------------------------------


def _vllm_config_for(path, k=4, max_model_len=64):
    return VllmConfig(
        model_config=ModelConfig(model="fake/tiny", dtype="float32", max_model_len=max_model_len),
        speculative_config=SpeculativeConfig(method="custom_class", model=path,
                                             num_speculative_tokens=k))


def test_happy_path_returns_instance_directly():
    """合法类：返回的就是那个类的实例，`propose` 可调用（**不加套壳 Adapter**）。"""
    from examples.custom_proposer import RepeatLastTokenProposer

    instance = create_custom_proposer(_vllm_config_for(EXAMPLE_PATH))
    assert type(instance) is RepeatLastTokenProposer
    assert callable(instance.propose)
    assert instance.num_speculative_tokens == 4


def test_constructor_receives_only_vllm_config():
    """插件只拿到 `VllmConfig`，拿不到任何运行时对象（构造参数逐个核对）。"""
    RecordingProposer.constructed.clear()
    create_custom_proposer(_vllm_config_for("test_custom_proposer.RecordingProposer"))
    assert len(RecordingProposer.constructed) == 1
    args, kwargs = RecordingProposer.constructed[0]
    assert kwargs == {}
    assert len(args) == 1 and isinstance(args[0], VllmConfig)
    # 配置对象上没有任何"活对象"入口 → 插件无法越权改 Scheduler/KV/批状态
    handed = args[0]
    for name in ("scheduler", "kv_cache_manager", "requests", "input_batch", "worker"):
        assert not hasattr(handed, name), f"VllmConfig 不该暴露 {name}"


def test_error_no_dot_in_path():
    with pytest.raises(ValueError, match="full module path"):
        create_custom_proposer(_vllm_config_for("NoDots"))


def test_error_module_not_found():
    with pytest.raises(ImportError, match="Cannot import module 'no_such_module_xyz'") as info:
        create_custom_proposer(_vllm_config_for("no_such_module_xyz.Klass"))
    assert isinstance(info.value.__cause__, ImportError), "必须保留原始异常链"


def test_error_class_not_found():
    with pytest.raises(AttributeError, match="has no attribute 'NoSuchClass'"):
        create_custom_proposer(_vllm_config_for("examples.custom_proposer.NoSuchClass"))


def test_error_constructor_failure():
    with pytest.raises(RuntimeError, match="must accept VllmConfig") as info:
        create_custom_proposer(_vllm_config_for("test_custom_proposer.BoomOnInitProposer"))
    assert isinstance(info.value.__cause__, RuntimeError)
    assert "构造器故意炸" in str(info.value.__cause__)


def test_error_propose_not_callable():
    with pytest.raises(AttributeError, match="not callable"):
        create_custom_proposer(_vllm_config_for("test_custom_proposer.NotCallableProposeProposer"))


def test_error_propose_missing():
    with pytest.raises(AttributeError, match="must have a 'propose' method"):
        create_custom_proposer(_vllm_config_for("test_custom_proposer.NoProposeProposer"))


# ---------------------------------------------------------------------------
# C. Runner 接线
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny():
    return tiny_qwen3_dir("tiny_gqa"), tiny_qwen3_config("tiny_gqa")


def make_engine(tiny, *, spec, max_num_seqs=2, budget=64, max_model_len=64):
    tiny_dir, hf_config = tiny
    config = VllmConfig(
        model_config=ModelConfig(model=tiny_dir, dtype="float32", max_model_len=max_model_len,
                                 hf_config=hf_config),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=32),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=DEVICE),
        speculative_config=spec)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner


def run(engine, core, requests, *, max_tokens=6):
    for req_id, prompt in requests:
        engine.add_request(req_id, list(prompt),
                           SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                          eos_token_id=999))
    outputs, stats = {}, []
    for _ in range(300):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            outputs[out.request_id] = list(out.token_ids)
        stats.append(core.scheduler.spec_decoding_stats)
    return outputs, stats


def test_runner_holds_the_plugin_instance(tiny):
    """合法类实例就是 Runner 持有的对象（不是包装类）。"""
    from examples.custom_proposer import RepeatLastTokenProposer

    engine, core, runner = make_engine(
        tiny, spec=SpeculativeConfig(model=EXAMPLE_PATH, num_speculative_tokens=4))
    try:
        assert type(runner.proposer) is RepeatLastTokenProposer
        assert isinstance(runner.proposer, RepeatLastTokenProposer)
    finally:
        engine.shutdown()


def test_call_args_match_frozen_runner_branch(tiny):
    """调用参数与冻结的上游 `custom_class` 分支一致：3 个位置参数 + `slot_mappings=`。

    三个对象就是 InputBatch 自己的缓冲（**身份相同**，不做拷贝/换格式），返回的是逐行 list。
    """
    RecordingProposer.calls.clear()
    engine, core, runner = make_engine(
        tiny, spec=SpeculativeConfig(method="custom_class",
                                     model="test_custom_proposer.RecordingProposer",
                                     num_speculative_tokens=4))
    # 在**调用当刻**记录批行数，并核对传进去的就是 InputBatch 自己的那两个缓冲
    rows_at_call, identity_at_call = [], []
    original_propose = runner.proposer.propose

    def spy(sampled, num_tokens_no_spec, token_ids_cpu, slot_mappings=None):
        rows_at_call.append(len(runner.input_batch.req_ids))
        identity_at_call.append((num_tokens_no_spec is runner.input_batch.num_tokens_no_spec,
                                 token_ids_cpu is runner.input_batch.token_ids_cpu))
        return original_propose(sampled, num_tokens_no_spec, token_ids_cpu, slot_mappings)

    runner.proposer.propose = spy
    try:
        outputs, stats = run(engine, core, [("a", [1, 2, 3, 4, 1, 2, 3, 4])])
        assert outputs["a"], "必须正常产出"
        assert RecordingProposer.calls, "插件至少要被调用一次"
        sample = RecordingProposer.calls[0]
        assert sample["token_ids_shape"] == runner.input_batch.token_ids_cpu.shape
        assert sample["slot_mappings"] is None
        # 行对齐：`sampled_token_ids` 的行数 == 当时的批行数；
        # 两个缓冲是**定长**的（长度 = max_num_reqs），只有前 N 行有效（上游同款）
        assert len(rows_at_call) == len(RecordingProposer.calls)
        for rows, call in zip(rows_at_call, RecordingProposer.calls):
            assert len(call["rows"]) == rows, (rows, call)
            assert len(call["num_tokens"]) == runner.input_batch.max_num_reqs
        # 对象身份：不拷贝、不换格式
        assert all(identity_at_call), identity_at_call
        assert sample["rows"][0] != []
    finally:
        engine.shutdown()

    # 冻结源码对照：上游那一段的参数顺序与我们调用的一致
    upstream = (Path("/home/user/anaconda3/envs/vllm-omni-dev/lib/python3.12/site-packages/vllm")
                / "v1" / "worker" / "gpu_model_runner.py").read_text()
    marker = 'elif spec_config.method == "custom_class":'
    snippet = upstream[upstream.index(marker):upstream.index(marker) + 400]
    for expected in ("propose(", "sampled_token_ids,", "num_tokens_no_spec,",
                     "token_ids_cpu,", "slot_mappings=slot_mappings,"):
        assert expected in snippet, f"上游分支里没有 {expected!r}"


def test_plugin_state_is_not_leaked_between_calls(tiny):
    """插件自己有没有状态是它的事；Runner 只负责按行喂、按行收（这里跑两轮验证稳定）。"""
    RecordingProposer.calls.clear()
    engine, core, runner = make_engine(
        tiny, spec=SpeculativeConfig(method="custom_class",
                                     model="test_custom_proposer.RecordingProposer",
                                     num_speculative_tokens=4))
    try:
        run(engine, core, [("a", [1, 2, 3, 4, 1, 2, 3, 4])], max_tokens=4)
        first = len(RecordingProposer.calls)
        run(engine, core, [("b", [5, 6, 5, 6])], max_tokens=4)
        assert len(RecordingProposer.calls) > first
    finally:
        engine.shutdown()


def test_finished_requests_without_remove_requests_hook(tiny):
    """插件**不需要**实现 `remove_requests`：请求结束那一轮不能因为缺少钩子而崩。

    （`create_custom_proposer` 只承诺 `propose`；Runner 侧对可选钩子按"有才调"处理。）
    """
    engine, core, runner = make_engine(
        tiny, spec=SpeculativeConfig(model=EXAMPLE_PATH, num_speculative_tokens=4))
    try:
        assert not hasattr(runner.proposer, "remove_requests")
        outputs, _ = run(engine, core, [("a", [1, 2, 3, 4, 1, 2, 3, 4])], max_tokens=4)
        assert outputs["a"]
        list(engine.step())          # 空跑一轮：这一轮才会处理 finished_req_ids
    finally:
        engine.shutdown()


class BufferShapeProposer:
    """记录两个缓冲的形状与"有效前缀"（验证定长缓冲契约）。"""

    seen: list = []

    def __init__(self, vllm_config):
        self.max_model_len = vllm_config.model_config.max_model_len

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None):
        rows = len(sampled_token_ids)
        BufferShapeProposer.seen.append({
            "rows": rows,
            "num_tokens_len": len(num_tokens_no_spec),
            "token_ids_shape": tuple(token_ids_cpu.shape),
            "valid_prefix": [int(x) for x in num_tokens_no_spec[:rows]],
            "stale_tail": [int(x) for x in num_tokens_no_spec[rows:]],
        })
        return [[] for _ in sampled_token_ids]


class TooFewRowsProposer:
    """少返回一行（协议违约）。"""

    def __init__(self, vllm_config):
        pass

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None):
        return [[] for _ in sampled_token_ids][:-1]


class TooManyRowsProposer:
    """多返回一行（协议违约）。"""

    def __init__(self, vllm_config):
        pass

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None):
        return [[] for _ in sampled_token_ids] + [[]]


def test_buffers_are_fixed_size_and_only_front_rows_are_valid(tiny):
    """插件契约：两个缓冲是定长的（`max_num_reqs`），**只有前 len(sampled) 行有效**。

    这条很重要：如果插件按 `range(len(num_tokens_no_spec))` 遍历，就会读到上一轮留下的脏行
    （上游 vLLM 的 `InputBatch` 同样是定长 `max_num_reqs` 缓冲，行为一致）。
    """
    BufferShapeProposer.seen.clear()
    engine, core, runner = make_engine(
        tiny, spec=SpeculativeConfig(method="custom_class",
                                     model="test_custom_proposer.BufferShapeProposer",
                                     num_speculative_tokens=2),
        max_num_seqs=1)
    try:
        run(engine, core, [("a", [1, 2, 3, 4])], max_tokens=3)
    finally:
        engine.shutdown()
    assert BufferShapeProposer.seen
    for seen in BufferShapeProposer.seen:
        assert seen["num_tokens_len"] == runner.input_batch.max_num_reqs
        assert seen["token_ids_shape"] == (runner.input_batch.max_num_reqs, 64)
        assert seen["rows"] <= seen["num_tokens_len"]
        assert all(value > 0 for value in seen["valid_prefix"]), seen


@pytest.mark.parametrize("path", ["test_custom_proposer.TooFewRowsProposer",
                                 "test_custom_proposer.TooManyRowsProposer"])
def test_row_count_mismatch_fails_loudly(tiny, path):
    """返回行数 != 批行数 → 当场 RuntimeError（不静默少给/丢掉几行）。"""
    engine, core, runner = make_engine(
        tiny, spec=SpeculativeConfig(method="custom_class", model=path,
                                     num_speculative_tokens=2))
    try:
        with pytest.raises(RuntimeError, match="必须按批行逐行返回"):
            run(engine, core, [("a", [1, 2, 3, 4, 1, 2, 3, 4])], max_tokens=3)
    finally:
        engine.shutdown()


# ---------------------------------------------------------------------------
# D. 行为：候选质量再差也不改答案
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path,expect_draft_tokens", [
    (EXAMPLE_PATH, True),                                  # 常规：会猜
    ("test_custom_proposer.EmptyProposer", False),          # 全空候选
    ("test_custom_proposer.AllWrongProposer", True),        # 全错候选
    ("test_custom_proposer.VariableLengthProposer", True),  # 变长候选
])
def test_candidates_do_not_change_greedy_output(tiny, path, expect_draft_tokens):
    """四类插件下：greedy 输出与"不开投机"逐 token 相同；候选确实走过链路。"""
    prompts = [("a", [1, 2, 3, 4, 1, 2, 3, 4]), ("b", [5, 6, 5, 6, 5, 6])]
    EmptyProposer.calls = 0

    engine, core, runner = make_engine(
        tiny, spec=SpeculativeConfig(method="custom_class", model=path,
                                     num_speculative_tokens=3))
    try:
        outputs, stats = run(engine, core, prompts)
    finally:
        engine.shutdown()

    baseline_engine, baseline_core, _ = make_engine(tiny, spec=None)
    try:
        baseline, _ = run(baseline_engine, baseline_core, prompts)
    finally:
        baseline_engine.shutdown()

    assert outputs == baseline, (outputs, baseline)
    hits = [entry for entry in stats if entry is not None]
    if expect_draft_tokens:
        assert sum(entry.num_drafts for entry in hits) > 0, "提案路径必须真的被调用"
        assert sum(entry.num_draft_tokens for entry in hits) > 0
    else:
        # 全空候选：Scheduler 那边根本没有草稿可记（`num_drafts` 保持 0），
        # 所以这里断言"插件确实被调用过，但没有一枚草稿进入链路"
        assert EmptyProposer.calls > 0, "插件必须被调用（只是返回空）"
        assert sum(entry.num_draft_tokens for entry in hits) == 0


def test_mid_prefill_row_is_skipped(tiny):
    """预算小 → 一条请求要分多轮 prefill：中间块那一行必须给空候选（采样为空）。"""
    RecordingProposer.calls.clear()
    engine, core, runner = make_engine(
        tiny, spec=SpeculativeConfig(method="custom_class",
                                     model="test_custom_proposer.RecordingProposer",
                                     num_speculative_tokens=2),
        budget=4, max_model_len=64)
    try:
        run(engine, core, [("a", [1, 2, 3, 4, 1, 2, 3, 4, 1, 2])], max_tokens=3)
    finally:
        engine.shutdown()
    # 至少有一次调用里第一行是空的（中间 prefill 块）
    assert any(call["rows"] and call["rows"][0] == [] for call in RecordingProposer.calls), \
        RecordingProposer.calls
    # 也有过非空的一行（最后一块 prefill 会采样出第一枚 token）
    assert any(call["rows"] and call["rows"][0] != [] for call in RecordingProposer.calls)
