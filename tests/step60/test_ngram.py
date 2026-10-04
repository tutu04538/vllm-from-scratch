"""step60：CPU ngram 提议者与上游冻结实现逐值一致（需求 060 §3.1/§4）。

对照对象是 **site-packages 里真的 vLLM**（`_find_longest_matched_ngram_and_propose_tokens`
与 `batch_propose_numba`）——生产路径不 import 它，只有测试这么干。

覆盖需求 §4 的前三组：无匹配 / 多个相同匹配 / 不同 n 长度竞争 / 历史不足 / 重复全同 token；
新采样长度 0/1/K+1；prompt 末尾与模型长度边界。
"""

import dataclasses

import numpy as np
import pytest
from ngram_helpers import FakeInputBatch, make_rows
from vllm.v1.spec_decode.ngram_proposer import (
    _find_longest_matched_ngram_and_propose_tokens as upstream_find)

from minivllm import SamplingParams, SpeculativeConfig
from minivllm.spec_decode.ngram_proposer import (
    NgramProposer, _find_longest_matched_ngram_and_propose_tokens as ours_find)

MAX_MODEL_LEN = 64


class _Config:
    """够用的配置替身（`NgramProposer` 只读这几项）。"""

    def __init__(self, k=4, min_n=2, max_n=4, max_model_len=MAX_MODEL_LEN, max_num_seqs=4):
        self.speculative_config = SpeculativeConfig(
            method="ngram", num_speculative_tokens=k,
            prompt_lookup_min=min_n, prompt_lookup_max=max_n)
        self.model_config = type("M", (), {"max_model_len": max_model_len})()
        self.scheduler_config = type("S", (), {"max_num_seqs": max_num_seqs})()


# ---------------------------------------------------------------------------
# 1. 逐值差分：单条序列的匹配/提取
# ---------------------------------------------------------------------------


CASES = [
    # (tokens, min_n, max_n, k) —— 名字里写清这条在考什么
    ([1, 2, 3, 1, 2, 9, 1, 2], 2, 2, 4),                     # 多处等长匹配：取最早那处
    ([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3], 3, 3, 2),           # 重复模式
    ([5, 5, 5, 5, 5, 5], 2, 5, 3),                           # 重复全同 token
    ([7, 8, 9, 1, 2, 3, 7, 8, 9], 3, 5, 2),                  # 不同 n 竞争：长的赢
    ([7, 8, 9, 1, 2, 3, 7, 8, 9], 5, 6, 2),                  # 历史不足 min_n → 不提
    ([1, 2], 5, 5, 3),                                       # 比 min_n 还短
    ([1, 2, 3], 1, 3, 4),                                    # 匹配点后面不够 k 个 → 少给
    ([9, 9, 9, 1, 9, 9, 9], 3, 3, 4),                        # 只匹配一次
    ([4, 1, 2, 3, 4, 1, 2], 1, 1, 3),                        # n=1（单 token 匹配）
]

# 随机补一批（固定 rng，失败可复现）
_rng = np.random.default_rng(0)
for _ in range(40):
    _length = int(_rng.integers(1, 40))
    _vocab = int(_rng.integers(2, 6))
    _tokens = _rng.integers(0, _vocab, size=_length).tolist()
    _min, _max = sorted((int(_rng.integers(1, 4)), int(_rng.integers(1, 6))))
    CASES.append((_tokens, _min, _max, int(_rng.integers(1, 6))))


@pytest.mark.parametrize("tokens,min_n,max_n,k", CASES)
def test_find_matches_upstream(tokens, min_n, max_n, k):
    """同一序列、同一参数，与上游冻结实现逐值一致（含 max_model_len 边界）。"""
    array = np.array(tokens, dtype=np.int32)
    for max_model_len in (10 ** 9, len(tokens) + 2, len(tokens)):
        assert (ours_find(array, min_n, max_n, max_model_len, k).tolist()
                == upstream_find(array, min_n, max_n, max_model_len, k).tolist()), (
            f"tokens={tokens} n=[{min_n},{max_n}] k={k} mml={max_model_len}")


def test_tie_takes_earliest_occurrence():
    """同长度多处匹配取**最早**那处（60 关之前本机取的是"最近一次"）。"""
    tokens = np.array([1, 2, 3, 1, 2, 9, 1, 2], dtype=np.int32)
    assert ours_find(tokens, 2, 2, 10 ** 9, 3).tolist() == [3, 1, 2]


def test_model_length_boundary_gives_nothing():
    """已经到 max_model_len：2..4 的 `k` 上限把草稿压成 0 个。"""
    tokens = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=np.int32)
    assert ours_find(tokens, 3, 3, len(tokens), 4).tolist() == []
    assert ours_find(tokens, 3, 3, len(tokens) + 2, 4).tolist() == [1, 2]


# ---------------------------------------------------------------------------
# 2. batch_propose / propose（批量与跳过规则）
# ---------------------------------------------------------------------------


def make_proposer(**kwargs) -> NgramProposer:
    return NgramProposer(_Config(**kwargs))


def test_batch_propose_matches_upstream_numba():
    """`batch_propose` 与上游 numba 版逐值一致（含"只算部分行"的用法）。"""
    from vllm.v1.spec_decode.ngram_proposer import batch_propose_numba

    rng = np.random.default_rng(7)
    batch, max_len, k = 6, 32, 3
    proposer = make_proposer(k=k, min_n=2, max_n=4, max_num_seqs=8)
    token_ids = np.zeros((batch, MAX_MODEL_LEN), dtype=np.int32)
    lengths = np.zeros(batch, dtype=np.int32)
    for i in range(batch):
        n = int(rng.integers(2, max_len))
        token_ids[i, :n] = rng.integers(0, 4, size=n)
        lengths[i] = n
    valid = [0, 1, 3, 5]

    theirs_draft = np.zeros((8, k), dtype=np.int32)
    theirs_num = np.zeros(8, dtype=np.int32)
    batch_propose_numba(valid, lengths, token_ids, proposer.min_n, proposer.max_n,
                        proposer.max_model_len, k, theirs_draft, theirs_num)
    theirs = [theirs_draft[i, :theirs_num[i]].tolist() if i in valid else []
              for i in range(batch)]
    assert proposer.batch_propose(batch, valid, lengths, token_ids, k) == theirs


def test_propose_skips_without_sampled_tokens_and_at_model_len():
    """两条跳过规则：本轮没采样 → 不提；已经到 max_model_len → 不提。"""
    histories = [[1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2]]
    batch = FakeInputBatch(histories, MAX_MODEL_LEN)
    lengths = batch.num_tokens_no_spec.numpy()
    proposer = make_proposer(k=2, min_n=2, max_n=2)
    token_ids = batch.token_ids_cpu.numpy()

    drafts = proposer.propose(2, [[7], [], [7]], lengths, token_ids)
    assert drafts[0] == [1, 2] and drafts[1] == [] and drafts[2] == [1, 2]

    lengths_full = lengths.copy()
    lengths_full[2] = MAX_MODEL_LEN
    assert proposer.propose(2, [[7], [7], [7]], lengths_full, token_ids)[2] == []


def test_propose_drafts_protocol_and_history_end():
    """统一协议入口：`history_end` 之后的内容（上一轮遗留的草稿区）不参与匹配。"""
    histories = [[1, 2, 1, 2, 1, 2], [3, 3, 3, 3, 3, 3]]
    all_token_ids = {f"r{i}": list(tokens) for i, tokens in enumerate(histories)}
    proposer = make_proposer(k=2, min_n=2, max_n=2)

    rows = make_rows(histories)
    drafts = proposer.propose_drafts(rows, all_token_ids)
    assert drafts.req_ids == ["r0", "r1"]
    assert drafts.draft_token_ids[0] == [1, 2] and drafts.draft_token_ids[1] == [3, 3]
    assert drafts.num_valid_draft_tokens is None          # CPU 提议者没有"占位"

    # 把 history_end 截短到 2：历史里的重复片段还没出现 → 不提
    short_rows = [dataclasses.replace(rows[0], history_end=2), rows[1]]
    short = proposer.propose_drafts(short_rows, all_token_ids)
    assert short.draft_token_ids[0] == []

    # 不在 ready 的中间 prefill 块：跳过（即使历史里有匹配）
    not_ready = [dataclasses.replace(rows[0], ready=False), rows[1]]
    assert proposer.propose_drafts(not_ready, all_token_ids).draft_token_ids[0] == []


def test_propose_drafts_uses_input_batch_buffer():
    """给了 `input_batch` 就用它的缓冲（切片是视图，不复制整段历史）。

    历史 `[1,2,1,2,1,2,9,9,9]` 的后缀是 `9 9`：它在第 6 位出现过一次，后面只剩一个 9
    （倒数第二个）→ 草稿就是 `[9]`（"给不满比给错好"）。
    """
    histories = [[1, 2, 1, 2, 1, 2, 9, 9, 9]]
    all_token_ids = {"r0": histories[0]}
    batch = FakeInputBatch(histories, MAX_MODEL_LEN)
    proposer = make_proposer(k=2, min_n=2, max_n=2)
    rows = make_rows(histories)
    assert proposer.propose_drafts(rows, all_token_ids, batch).draft_token_ids[0] == [9]
    # 把缓冲改成"后缀是 1 2"的历史，草稿随之改变（证明真的读了批缓冲）
    batch.set_row(0, [9, 9, 1, 2, 1, 2, 1, 2], req_id="r0")
    rows = make_rows([[9, 9, 1, 2, 1, 2, 1, 2]])
    assert proposer.propose_drafts(rows, all_token_ids, batch).draft_token_ids[0] == [1, 2]


# ---------------------------------------------------------------------------
# 3. 配置口径
# ---------------------------------------------------------------------------


def test_prompt_lookup_defaults_and_validation():
    """上游那套补全规则：都没给 5/5；只给一个 → 另一个跟随；min > max 报错。"""
    spec = SpeculativeConfig(method="ngram", num_speculative_tokens=3)
    assert (spec.prompt_lookup_min, spec.prompt_lookup_max) == (5, 5)
    spec = SpeculativeConfig(method="ngram", num_speculative_tokens=3, prompt_lookup_max=8)
    assert (spec.prompt_lookup_min, spec.prompt_lookup_max) == (8, 8)
    spec = SpeculativeConfig(method="ngram", num_speculative_tokens=3, prompt_lookup_min=3)
    assert (spec.prompt_lookup_min, spec.prompt_lookup_max) == (3, 3)
    with pytest.raises(ValueError, match="不能大于"):
        SpeculativeConfig(method="ngram", num_speculative_tokens=3,
                          prompt_lookup_min=7, prompt_lookup_max=2)


def test_ngram_gpu_method_flag():
    assert SpeculativeConfig(method="ngram_gpu", num_speculative_tokens=3).use_ngram_gpu()
    assert not SpeculativeConfig(method="ngram", num_speculative_tokens=3).use_ngram_gpu()
    # 两条 ngram 路径都不需要额外输入槽位（不跑 draft 模型、不写 KV）
    assert SpeculativeConfig(method="ngram_gpu",
                             num_speculative_tokens=3).max_num_new_slots_for_drafting == 0


# ---------------------------------------------------------------------------
# 4. 端到端（真 tiny 模型；投机验证走 Triton 内核，所以只在 CUDA 上跑）
# ---------------------------------------------------------------------------


def _greedy(tiny_dir, hf_config, device, *, method, spec_k, prompts, max_tokens=8,
            min_n=None, max_n=None):
    from ngram_helpers import make_config
    from minivllm import LLMEngine, UniProcExecutor, Worker

    config = make_config(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=spec_k,
                         method=method, device=device, max_num_seqs=2)
    if spec_k is not None and min_n is not None:
        spec = config.speculative_config
        object.__setattr__(spec, "prompt_lookup_min", min_n)
        object.__setattr__(spec, "prompt_lookup_max", max_n)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    core = engine.engine_core.engine_core
    for req_id, prompt in prompts:
        engine.add_request(req_id, list(prompt),
                           SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                          eos_token_id=999))
    outputs, stats = {}, []
    for _ in range(60):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            outputs[out.request_id] = list(out.token_ids)
        stats.append(core.scheduler.spec_decoding_stats)
    engine.shutdown()
    return outputs, [entry for entry in stats if entry is not None]


PROMPTS = (("A", [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]), ("B", [2, 3, 2, 3, 2, 3, 9, 9]))


def test_ngram_speculative_matches_non_speculative(cuda_device, tiny_dir, hf_config):
    """CPU ngram：greedy 投机 == 非投机，而且真的提出了多枚草稿（min_n=1）。"""
    plain, _ = _greedy(tiny_dir, hf_config, cuda_device, method="ngram", spec_k=None,
                       prompts=PROMPTS)
    spec, stats = _greedy(tiny_dir, hf_config, cuda_device, method="ngram", spec_k=3,
                          prompts=PROMPTS, min_n=1, max_n=3)
    assert spec == plain
    assert stats and sum(entry.num_draft_tokens for entry in stats) > 0


def test_ngram_gpu_speculative_matches_non_speculative(cuda_device, tiny_dir, hf_config):
    """GPU ngram：greedy 投机 == 非投机；统计只记"验证过的候选"。"""
    plain, _ = _greedy(tiny_dir, hf_config, cuda_device, method="ngram_gpu", spec_k=None,
                       prompts=PROMPTS)
    spec, stats = _greedy(tiny_dir, hf_config, cuda_device, method="ngram_gpu", spec_k=3,
                          prompts=PROMPTS, min_n=1, max_n=3)
    assert spec == plain
    assert stats and all(0 <= entry.num_accepted_tokens <= entry.num_draft_tokens
                         <= 3 * entry.num_drafts for entry in stats)


def test_ngram_cpu_and_gpu_agree_on_the_same_prompt(cuda_device, tiny_dir, hf_config):
    """同一模型、同一 prompt、同一窗口：两条提议路径的 greedy 输出一致。"""
    cpu, _ = _greedy(tiny_dir, hf_config, cuda_device, method="ngram", spec_k=3,
                     prompts=PROMPTS, min_n=1, max_n=3)
    gpu, _ = _greedy(tiny_dir, hf_config, cuda_device, method="ngram_gpu", spec_k=3,
                     prompts=PROMPTS, min_n=1, max_n=3)
    assert cpu == gpu
