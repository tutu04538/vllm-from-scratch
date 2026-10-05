"""step61：配置、依赖接入、适配层与"这关解决什么痛点"的直接证据。

差分（与上游提议者逐步比对）在 `test_suffix_differential.py`；真引擎在 `test_suffix_engine.py`。
"""

import numpy as np
import pytest
import torch

from suffix_helpers import FakeBatch, our_proposer, run_trace, run_trace_on

from minivllm.config import ModelConfig, SpeculativeConfig, VllmConfig, has_arctic_inference
from minivllm.outputs import DraftTokenIds
from minivllm.spec_decode.ngram_proposer import NgramProposer
from minivllm.spec_decode.suffix_decoding import SuffixDecodingProposer, _as_int32
from minivllm.spec_decode.utils import TargetRows

K = 8
REPEAT_PROMPT = {"r0": [1, 2, 3, 4, 1, 2, 3, 4]}


def _fake_vllm_config(**spec_kwargs):
    kwargs = dict(method="suffix", num_speculative_tokens=K)
    kwargs.update(spec_kwargs)
    return VllmConfig(
        model_config=ModelConfig(model="fake/tiny", dtype="float32", max_model_len=64),
        speculative_config=SpeculativeConfig(**kwargs))


# ---------------------------------------------------------------------------
# 配置：字段、默认值、校验（照抄上游 `_validate_suffix_decoding`）
# ---------------------------------------------------------------------------


def test_dependency_is_installed_and_detected():
    """本机装了 `arctic_inference`（需求 61 要求接入依赖实现，不自研后缀树）。"""
    import importlib.metadata

    assert has_arctic_inference() is True
    # 具体版本/来源/sha256 记在 docs/results.json → step61.dependencies（含 0.1.1 → 0.3.0 的偏差说明）
    assert importlib.metadata.version("arctic_inference") == "0.3.0"


def test_defaults_follow_upstream():
    """上游默认值：树深 24、全局缓存 10000 条、spec factor 1.0、min token prob 0.1。"""
    config = SpeculativeConfig(method="suffix")
    assert config.suffix_decoding_max_tree_depth == 24
    assert config.suffix_decoding_max_cached_requests == 10000
    assert config.suffix_decoding_max_spec_factor == 1.0
    assert config.suffix_decoding_min_token_prob == 0.1


def test_unspecified_num_speculative_tokens_defaults_to_tree_depth():
    """没给 `num_speculative_tokens`（本仓库用 0 表示）→ 取树深（上游 `is None` 同义）。

    suffix decoding 每步动态决定猜几枚，K 只是上限；显式给 K 时按给定的走。
    """
    assert SpeculativeConfig(method="suffix").num_speculative_tokens == 24
    assert SpeculativeConfig(method="suffix",
                             suffix_decoding_max_tree_depth=6).num_speculative_tokens == 6
    assert SpeculativeConfig(method="suffix", num_speculative_tokens=4).num_speculative_tokens == 4


@pytest.mark.parametrize("kwargs,pattern", [
    ({"suffix_decoding_max_tree_depth": 0}, "suffix_decoding_max_tree_depth"),
    ({"suffix_decoding_max_cached_requests": -1}, "suffix_decoding_max_cached_requests"),
    ({"suffix_decoding_max_spec_factor": -0.5}, "suffix_decoding_max_spec_factor"),
    ({"suffix_decoding_min_token_prob": 1.5}, "suffix_decoding_min_token_prob"),
    ({"suffix_decoding_min_token_prob": -0.1}, "suffix_decoding_min_token_prob"),
])
def test_invalid_values_are_rejected(kwargs, pattern):
    """取值校验与上游逐条对应（报错文案也照抄，方便对照源码行号）。"""
    with pytest.raises(ValueError, match=pattern):
        SpeculativeConfig(method="suffix", **kwargs)


def test_missing_dependency_fails_loudly(monkeypatch):
    """没装依赖时必须显式报错（不能悄悄退回别的算法，也不能静默关闭投机）。"""
    import minivllm.config as config_module

    monkeypatch.setattr(config_module, "has_arctic_inference", lambda: False)
    with pytest.raises(ImportError, match="arctic-inference==0.1.1"):
        SpeculativeConfig(method="suffix")


def test_unknown_method_still_rejected():
    """别的投机方法照旧明确报错（不静默降级）。

    （这条用例的"仍未实现的方法"换过两次：63 关的 `eagle`/`eagle3` → `medusa`；
    66 关实现了 `medusa`，现在改用仍未实现的 `dflash`（76–78 关）。断言强度不变：
    配置期 ValueError + 不静默回退。）
    """
    with pytest.raises(ValueError, match="本关只支持"):
        SpeculativeConfig(method="dflash")


# ---------------------------------------------------------------------------
# 依赖接入口径：缓冲类型转换
# ---------------------------------------------------------------------------


def test_as_int32_matches_upstream_buffer_type():
    """本仓库的 int64 torch 缓冲要转成**上游传给依赖的那种**：1 维、连续、int32。"""
    tensor = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
    array = _as_int32(tensor[0, 1:4])
    assert array.dtype == np.int32 and array.ndim == 1 and array.flags["C_CONTIGUOUS"]
    assert array.tolist() == [2, 3, 4]
    # 上游 InputBatch 里就是 int32 numpy 视图（vllm/v1/worker/gpu_input_batch.py:140）
    assert _as_int32([7, 8]).dtype == np.int32


def test_our_proposer_would_fail_on_non_contiguous_or_wrong_dtype():
    """依赖包对 ndarray 只接受 1 维/连续/int32（它的 `_validate_ndarray`）→ 我们必须先转。"""
    from arctic_inference.suffix_decoding import SuffixDecodingCache

    cache = SuffixDecodingCache()
    cache.start_request("r", np.array([1, 2, 3, 4], dtype=np.int32))
    with pytest.raises((ValueError, TypeError)):
        cache.speculate("r", np.array([[1, 2, 3, 4]], dtype=np.int32))  # 2 维
    with pytest.raises((ValueError, TypeError)):
        cache.speculate("r", np.array([1, 2, 3, 4], dtype=np.int64))  # 非 int32


# ---------------------------------------------------------------------------
# Runner 适配层：propose_drafts
# ---------------------------------------------------------------------------


def test_propose_drafts_wraps_upstream_protocol():
    """`propose_drafts` 把 `sampled_by_row` 摊成逐行 list，并按批行序返回 `DraftTokenIds`。"""
    proposer = our_proposer(num_speculative_tokens=K, max_model_len=64)
    batch = FakeBatch(2, 64)
    histories = {"rA": [1, 2, 3, 4, 1, 2, 3, 4, 1], "rB": [9, 9]}
    batch.layout(["rA", "rB"], histories, {"rA": 8, "rB": 2})
    rows = [TargetRows(req_id="rA", row=0, start=0, target_rows=1, num_rejected=0,
                       history_end=9, next_token_id=1, ready=True),
            TargetRows(req_id="rB", row=1, start=0, target_rows=1, num_rejected=0,
                       history_end=2, next_token_id=9, ready=False)]
    drafts = proposer.propose_drafts(rows, histories, batch, sampled_by_row={0: [1]},
                                     sample_rows=[0])
    assert isinstance(drafts, DraftTokenIds)
    assert drafts.req_ids == ["rA", "rB"]
    assert drafts.draft_token_ids == [[2, 3, 4, 1], []]
    assert drafts.draft_probs is None, "suffix decoding 是确定性提议（无 q）"
    # 与直接按上游签名调 `propose` 等价（换一个干净提议者，避免树状态被上一次调用改变）
    fresh_batch = FakeBatch(2, 64)
    fresh_batch.layout(["rA", "rB"], histories, {"rA": 8, "rB": 2})
    direct = our_proposer(num_speculative_tokens=K, max_model_len=64).propose(
        K, fresh_batch, [[1], []])
    assert drafts.draft_token_ids == direct
    # **一轮只能调一次**：再调一次会把同一批 token 当成新响应再追加一遍（树会变长）
    again = proposer.propose(proposer.num_speculative_tokens, batch, [[1], []])
    assert again[0] != drafts.draft_token_ids[0], again


def test_propose_drafts_requires_input_batch():
    """不给 `input_batch` 就明确报错，不自己拼一份会改变建树范围的缓冲。"""
    proposer = our_proposer(num_speculative_tokens=K, max_model_len=64)
    with pytest.raises(ValueError, match="input_batch"):
        proposer.propose_drafts([], {}, None)


def test_load_model_is_noop():
    """本关没有模型要装（上游同款空实现），Runner 调用它只是为了统一接口。"""
    proposer = our_proposer(num_speculative_tokens=K, max_model_len=64)
    assert proposer.load_model() is None


def test_max_model_len_comes_from_model_config():
    proposer = SuffixDecodingProposer(_fake_vllm_config())
    assert proposer.max_model_len == 64
    assert proposer.num_speculative_tokens == K
    assert proposer.max_tree_depth == 24 and proposer.max_spec_factor == 1.0
    assert proposer.min_token_prob == 0.1


# ---------------------------------------------------------------------------
# 痛点：ngram（60 关）看不见的跨请求模式，suffix decoding 能看见
# ---------------------------------------------------------------------------


def _ngram_drafts(history, *, prompt_len, k=K, minimum=3, maximum=3):
    """跑一遍 **60 关的 ngram 提议者**（它只有本请求历史，没有跨请求状态）。"""
    config = VllmConfig(
        model_config=ModelConfig(model="fake/tiny", dtype="float32", max_model_len=64),
        speculative_config=SpeculativeConfig(method="ngram", num_speculative_tokens=k,
                                             prompt_lookup_min=minimum,
                                             prompt_lookup_max=maximum))
    proposer = NgramProposer(config)
    batch = FakeBatch(1, 64)
    batch.layout(["r0"], {"r0": history}, {"r0": prompt_len})
    rows = [TargetRows(req_id="r0", row=0, start=0, target_rows=1, num_rejected=0,
                       history_end=len(history), next_token_id=history[-1], ready=True)]
    return proposer.propose_drafts(rows, {"r0": history}, batch).draft_token_ids


@pytest.mark.parametrize("minimum,maximum", [(3, 3), (5, 5)])
def test_ngram_cannot_reuse_another_requests_pattern(minimum, maximum):
    """B 的历史里 "1 2 3" 只出现一次 → ngram 匹配不到；A 的输出对它不可见。"""
    assert _ngram_drafts([7, 8, 1, 2, 3, 4], prompt_len=5, minimum=minimum,
                         maximum=maximum) == [[]]


def test_suffix_decoding_reuses_another_requests_pattern():
    """同一个 B：A 教过全局树 "1 2 3 → 4 5"，suffix decoding 就能给出 [5]。

    两边看的是**同一份历史**（prompt [7,8,1,2,3] + 本轮采样 [4]），差别只在
    "有没有跨请求的全局后缀树"——这正是需求 61 要解决的痛点。
    """
    ours = run_trace_on("ours", max_model_len=64,
                        prompts={"A": [1, 2, 3, 9, 9, 1, 2, 3], "B": [7, 8, 1, 2, 3]},
                        steps=[{"order": ["A"], "sampled": {"A": [4]}},
                               {"order": ["A"], "sampled": {"A": [5]}},
                               {"order": ["B"], "sampled": {"B": [4]}}],
                        num_speculative_tokens=K)
    assert ours["drafts"][2] == [[5]], ours["drafts"]
    # 关掉全局树（= 只剩 ngram 那种"只看自己"的视野）→ 候选消失
    closed = run_trace_on("ours", max_model_len=64,
                          prompts={"A": [1, 2, 3, 9, 9, 1, 2, 3], "B": [7, 8, 1, 2, 3]},
                          steps=[{"order": ["A"], "sampled": {"A": [4]}},
                                 {"order": ["A"], "sampled": {"A": [5]}},
                                 {"order": ["B"], "sampled": {"B": [4]}}],
                          num_speculative_tokens=K, max_cached_requests=0)
    assert closed["drafts"][2] == [[]], closed["drafts"]


def test_prompt_tree_avoids_the_cost_ngram_pays_for_repeats():
    """自己的 prompt 里有重复模式时，两者都能猜——但本关的候选长度是**动态**的。"""
    steps = [{"order": ["r0"], "sampled": {"r0": [1]}}]
    suffix = run_trace_on("ours", max_model_len=64, prompts=REPEAT_PROMPT, steps=steps,
                          num_speculative_tokens=K)
    ngram = _ngram_drafts([1, 2, 3, 4, 1, 2, 3, 4, 1], prompt_len=8, minimum=4, maximum=4)
    assert suffix["drafts"][0][0] == [2, 3, 4, 1]
    assert ngram == [[2, 3, 4, 1]], "同一份重复模式下 ngram 也能给出同样的候选"


def test_propose_is_stateless_between_calls():
    """同一串事件喂两个**独立**提议者，结果一致（没有隐藏的全局随机/顺序依赖）。"""
    steps = [{"order": ["r0"], "sampled": {"r0": [1]}}]
    first = run_trace(our_proposer(num_speculative_tokens=K, max_model_len=64),
                      max_model_len=64, prompts=REPEAT_PROMPT, steps=steps,
                      num_speculative_tokens=K)
    second = run_trace(our_proposer(num_speculative_tokens=K, max_model_len=64),
                       max_model_len=64, prompts=REPEAT_PROMPT, steps=steps,
                       num_speculative_tokens=K)
    assert first["drafts"] == second["drafts"] == [[[2, 3, 4, 1]]]
