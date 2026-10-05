"""step67：异构词表的 token 级对齐（需求 067 §2/§3/§4）。

分四层：

1. **构造层**（假 tokenizer）：同一个词在两套 id 空间里不同号、空格标记两族（▁ 与 Ġ）、
   规范化后重名取第一个、超出模型 `vocab_size` 的条目丢掉、交集为空/极小的行为；
2. **映射层**：两个方向的 map 精确（交集上往返一致）、不改写入参、`q`/logits 的约束
   （非交集列置 `-inf` → 永远选不到）；
3. **上游差分**：同一对假 tokenizer 喂给 site-packages 里真的
   `v1/spec_decode/vocab_mapping.py::VocabMapping`，逐张量/逐输出比对（"冻结实现"是基线）；
4. **集成层**：一对真正的 tiny 模型（同 KV 规格、vocab 11 vs 13、两套真 tokenizer 文件）跑
   greedy：开了 TLI 的输出与非投机逐 token 相同、交出去的草稿是 **target 空间**的 id、
   第一遍的输入**真的过了映射**（反"假接线"）、概率草稿被配置期拒绝。

真实 tokenizer 的规模（Qwen3-1.7B vs gpt2）单独记一条，缺缓存时跳过并写明原因。
"""

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,  # noqa: E402
                      SchedulerConfig, SpeculativeConfig, UniProcExecutor, VllmConfig, Worker)
from minivllm.spec_decode.vocab_mapping import (VocabMapping, _detect_space_prefix,  # noqa: E402
                                               _get_unk_token_id, _normalize_token,
                                               load_tokenizer)
from minivllm.testing.tiny_models import tiny_hetero_pair  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 2


# ---------------------------------------------------------------- 假 tokenizer


class FakeTokenizer:
    """只提供 `VocabMapping` 真正用到的接口（上游同样只用这几个）：

        get_vocab()            token 字符串 → id
        encode(text, ...)      `_detect_space_prefix` 用它探测" a" 怎么切
        convert_ids_to_tokens 把探测到的 id 换回 token 字符串
        unk_token_id / eos_token_id

    这样能在不碰真 tokenizer 的前提下造出"同词不同 id / 重复规范化 / 越界 id / 交集为空"这些形状。
    """

    def __init__(self, vocab: dict[str, int], *, space_marker: str | None = None,
                 unk_token_id: int | None = None, eos_token_id: int | None = None) -> None:
        self._vocab = dict(vocab)
        self._reverse = {index: token for token, index in self._vocab.items()}
        self.space_marker = space_marker
        self.unk_token_id = unk_token_id
        self.eos_token_id = eos_token_id

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # 只模拟 "_detect_space_prefix" 需要的那一次调用：把 " a" 切成"以 marker 开头、以 a 结尾"的 token。
        # 真实 tokenizer 切出来的具体是哪个 token 由词表决定，所以这里也允许"任意一个"这样的 token。
        if self.space_marker is not None and text == " a":
            exact = f"{self.space_marker}a"
            if exact in self._vocab:
                return [self._vocab[exact]]
            candidates = sorted((index, token) for token, index in self._vocab.items()
                               if token.startswith(self.space_marker) and token.endswith("a"))
            if candidates:
                return [candidates[0][0]]
        return [self._vocab.get(text, self.unk_token_id if self.unk_token_id is not None else 0)]

    def convert_ids_to_tokens(self, index: int) -> str:
        return self._reverse.get(index, "")


def space_marker_tokenizer(marker: str, pairs: list[tuple[str, int]], **kwargs) -> FakeTokenizer:
    """按"每个词给一个 id"造一个假 tokenizer（词自动带上 marker 前缀）。"""
    vocab = {f"{marker}{word}": index for word, index in pairs}
    return FakeTokenizer(vocab, space_marker=marker, **kwargs)


# ---------------------------------------------------------------- 1. 构造层


def test_same_text_different_ids_builds_intersection():
    """同一个词在两套 id 空间里不同号：交集按**字符串**建，两个方向都对上。"""
    target = space_marker_tokenizer("\u2581", [("hello", 3), ("kv", 5), ("onlyt", 7)],
                                    unk_token_id=0, eos_token_id=1)
    draft = space_marker_tokenizer("\u0120", [("kv", 9), ("hello", 2), ("onlyd", 4)],
                                   unk_token_id=0, eos_token_id=1)
    mapping = VocabMapping(target, draft, target_vocab_size=11, draft_vocab_size=13,
                           device="cpu")
    assert mapping.intersection_size == 2                       # hello / kv
    assert mapping.target_to_draft_ids[3] == 2                  # target ▁hello(id 3) → draft Ġhello(id 2)
    assert mapping.draft_to_target_ids[9] == 5                  # draft Ġkv(id 9) → target ▁kv(id 5)
    assert mapping.target_to_draft_ids[7] == -1                 # target 独有
    assert mapping.draft_to_target_ids[4] == -1                 # draft 独有
    # 交集上往返精确（需求 §4）
    ids = torch.tensor([3, 5], dtype=torch.int64)
    assert torch.equal(mapping.map_draft_to_target_ids(
        mapping.map_target_to_draft_ids(ids)), ids)


def test_different_space_markers_are_normalized():
    """两族空格标记（SentencePiece 的 ▁ / BPE 的 Ġ）归一化后能对上同一个词。"""
    # 词表里要有"以 marker 开头、以 a 结尾"的 token（真实 tokenizer 都有 " a" 这种切法），
    # `_detect_space_prefix()` 就是靠 encode(" a") 探出来的
    target = space_marker_tokenizer("\u2581", [("hello", 1), ("a", 4)], unk_token_id=0)
    draft = space_marker_tokenizer("\u0120", [("hello", 7), ("a", 2)], unk_token_id=0)
    assert _detect_space_prefix(target) == ("\u2581",)
    assert _detect_space_prefix(draft) == ("\u0120",)
    assert _normalize_token("\u2581hello", ("\u2581",)) == " hello"
    assert _normalize_token("\u0120hello", ("\u0120",)) == " hello"
    mapping = VocabMapping(target, draft, target_vocab_size=6, draft_vocab_size=8, device="cpu")
    assert mapping.target_to_draft_ids[1] == 7 and mapping.draft_to_target_ids[7] == 1
    assert mapping.target_to_draft_ids[4] == 2 and mapping.intersection_size == 2


def test_duplicate_normalized_tokens_keep_the_first_id():
    """同一个 tokenizer 里规范化后重名的，只留**字典里第一个**（上游同款规则）。"""
    # 同一个词表里同时有 ▁a 与 Ġa（混合家族），规范化后都是 " a"
    vocab = {"\u2581a": 4, "\u0120a": 9, "\u2581b": 5}
    target = FakeTokenizer(vocab, space_marker="\u2581", unk_token_id=0)
    draft = space_marker_tokenizer("\u0120", [("a", 2), ("b", 3)], unk_token_id=0)
    mapping = VocabMapping(target, draft, target_vocab_size=11, draft_vocab_size=5, device="cpu")
    # target id 4（▁a，dict 里第一个）映射到 draft id 2；target id 9（Ġa，重名被丢掉）不在表里
    assert mapping.target_to_draft_ids[4] == 2
    assert mapping.target_to_draft_ids[9] == -1
    assert mapping.draft_to_target_ids[2] == 4


def test_out_of_range_ids_are_skipped():
    """tokenizer 里超出**模型** vocab_size 的条目（多出来的 special token）不参与映射。"""
    target = space_marker_tokenizer("\u2581", [("hello", 3), ("inrange", 5), ("toobig", 20)],
                                    unk_token_id=0)
    draft = space_marker_tokenizer("\u0120", [("hello", 2), ("inrange", 1), ("toobig", 9)],
                                   unk_token_id=0)
    mapping = VocabMapping(target, draft, target_vocab_size=11, draft_vocab_size=10, device="cpu")
    assert mapping.intersection_size == 2                       # 只有 hello / inrange
    # 越界条目根本没进表：表长就是模型词表大小，20 / 9 这种 id 在表里没有位置
    assert tuple(mapping.target_to_draft_ids.shape) == (11,)
    assert tuple(mapping.draft_to_target_ids.shape) == (10,)
    assert mapping.target_to_draft_ids[3] == 2                  # ▁hello ↔ Ġhello
    assert mapping.target_to_draft_ids[5] == 1                  # ▁inrange ↔ Ġinrange
    assert mapping.intersection_size == int(mapping.intersection_mask_draft.sum())


def test_unk_zero_is_a_legal_id():
    """`unk_token_id = 0` 必须被当成合法 id（写成 `unk or eos` 会把它换成 eos）。"""
    target = FakeTokenizer({"\u2581a": 4}, space_marker="\u2581", unk_token_id=0, eos_token_id=5)
    draft = FakeTokenizer({"\u0120b": 2}, space_marker="\u0120", unk_token_id=0, eos_token_id=7)
    mapping = VocabMapping(target, draft, target_vocab_size=8, draft_vocab_size=8, device="cpu")
    assert mapping.target_unk_token_id == 0 and mapping.draft_unk_token_id == 0
    assert mapping.intersection_size == 0
    assert mapping.map_target_to_draft_ids(torch.tensor([4])).tolist() == [0]     # 不是 7
    assert mapping.map_draft_to_target_ids(torch.tensor([2])).tolist() == [0]     # 不是 5


def test_unk_missing_falls_back_to_eos_with_warning():
    """没有 unk 就退 eos，并且**告警**（上游同款顺序）。"""
    target = FakeTokenizer({"\u2581a": 4}, space_marker="\u2581", unk_token_id=None,
                           eos_token_id=3)
    with pytest.warns(UserWarning, match="eos_token_id=3"):
        assert _get_unk_token_id(target, "target tokenizer") == 3


def test_no_unk_no_eos_raises():
    """两者都没有 → 明确报错（不能猜）。"""
    target = FakeTokenizer({"\u2581a": 4}, space_marker="\u2581")
    with pytest.raises(ValueError, match="既没有 unk_token_id 也没有 eos_token_id"):
        _get_unk_token_id(target, "draft tokenizer")


def test_small_and_empty_intersection_are_recorded_not_hidden():
    """交集极小只告警、交集为空照样建表（都按上游行为记录，不假装支持任意 tokenizer）。"""
    target = space_marker_tokenizer("\u2581", [(f"t{i}", i) for i in range(4)], unk_token_id=0)
    draft = space_marker_tokenizer("\u0120", [(f"d{i}", i) for i in range(4)], unk_token_id=0)
    with pytest.warns(UserWarning, match="交集只有 0 个 token"):
        empty = VocabMapping(target, draft, target_vocab_size=4, draft_vocab_size=4,
                             device="cpu")
    assert empty.intersection_size == 0
    assert not bool(empty.intersection_mask_draft.any())
    # 空交集：所有草稿 logits 都被掩成 -inf（上游也没为这种情况加保护，行为照实记录）
    constrained = empty.constrain_draft_logits(torch.randn(2, 4))
    assert bool(torch.isinf(constrained).all())
    # 两个方向的映射全部回退 unk
    assert empty.map_target_to_draft_ids(torch.tensor([0, 1])).tolist() == [0, 0]
    assert empty.map_draft_to_target_ids(torch.tensor([2, 3])).tolist() == [0, 0]


# ---------------------------------------------------------------- 2. 映射层


def sample_mapping() -> VocabMapping:
    target = space_marker_tokenizer("\u2581", [("hello", 2), ("kv", 3), ("onlyt", 6)],
                                    unk_token_id=0, eos_token_id=1)
    draft = space_marker_tokenizer("\u0120", [("kv", 5), ("hello", 7), ("onlyd", 9)],
                                   unk_token_id=0, eos_token_id=1)
    return VocabMapping(target, draft, target_vocab_size=8, draft_vocab_size=11, device="cpu")


def test_map_methods_do_not_modify_their_inputs():
    """`map_*` / `constrain_draft_logits` **不改写入参**（需求 §4）。"""
    mapping = sample_mapping()
    target_ids = torch.tensor([2, 3, 6], dtype=torch.int64)
    draft_ids = torch.clone(target_ids)
    logits = torch.randn(3, 11)
    logits_copy = logits.clone()

    mapping.map_target_to_draft_ids(target_ids)
    mapping.map_draft_to_target_ids(draft_ids)
    mapping.constrain_draft_logits(logits)

    assert torch.equal(target_ids, torch.tensor([2, 3, 6], dtype=torch.int64))
    assert torch.equal(draft_ids, target_ids)
    assert torch.equal(logits, logits_copy)


def test_non_intersection_logits_can_never_be_selected():
    """草稿独有列即使 logits 最大也选不到（掩码 → argmax 永远落在交集里）。"""
    mapping = sample_mapping()
    logits = torch.full((2, 11), -5.0)
    logits[:, 9] = 100.0                                        # draft 独有列给极大值
    logits[:, 10] = 99.0
    constrained = mapping.constrain_draft_logits(logits)
    picked = constrained.argmax(dim=-1)
    assert not bool((picked == 9).any()) and not bool((picked == 10).any())
    assert bool(mapping.intersection_mask_draft[picked].all())
    # 掩掉的列恰好是 -inf，其余列数值不变
    assert bool((constrained[:, ~mapping.intersection_mask_draft] == float("-inf")).all())
    assert torch.equal(constrained[:, mapping.intersection_mask_draft],
                       logits[:, mapping.intersection_mask_draft])


def test_same_text_survives_the_round_trip_through_ids():
    """id 换了，但"念出来的字"没变：用两边的 tokenizer 解码同一串 id，文本相同。"""
    target_dir, draft_dir, info = tiny_hetero_pair()
    target_tokenizer = load_tokenizer(target_dir)
    draft_tokenizer = load_tokenizer(draft_dir)
    mapping = VocabMapping(target_tokenizer, draft_tokenizer,
                           target_vocab_size=info["target_vocab_size"],
                           draft_vocab_size=info["draft_vocab_size"], device="cpu")
    prompt = torch.tensor(info["prompt_token_ids"], dtype=torch.int64)
    mapped = mapping.map_target_to_draft_ids(prompt)
    assert mapped.shape == prompt.shape                          # **不重新分词**：长度不变
    assert not torch.equal(mapped, prompt)                       # id 真的变了
    # 逐个位置比"规范化后的 token 字符串"：换的是 id，不是文字。
    # （不比 `decode()` 出来的整段文本：两族 tokenizer 的 decoder 对"词首空格"的处理不同，
    #   Metaspace 解码时会吃掉开头那个空格、ByteLevel 会留一个——那是解码器的口味，不是映射的问题。）
    from minivllm.spec_decode.vocab_mapping import _detect_space_prefix, _normalize_token

    target_prefixes = _detect_space_prefix(target_tokenizer)
    draft_prefixes = _detect_space_prefix(draft_tokenizer)
    for original, converted in zip(prompt.tolist(), mapped.tolist()):
        target_token = _normalize_token(target_tokenizer.convert_ids_to_tokens(original),
                                        target_prefixes)
        draft_token = _normalize_token(draft_tokenizer.convert_ids_to_tokens(converted),
                                       draft_prefixes)
        assert target_token == draft_token


# ---------------------------------------------------------------- 3. 与上游差分


def test_upstream_vocab_mapping_differential():
    """同一对假 tokenizer 喂给**上游真的** `VocabMapping`，逐张量、逐输出比对。"""
    from vllm.v1.spec_decode.vocab_mapping import VocabMapping as UpstreamVocabMapping

    cases = [
        # (target vocab, draft vocab, target 模型词表, draft 模型词表)
        (space_marker_tokenizer("\u2581", [("hello", 2), ("kv", 3), ("onlyt", 6)],
                                unk_token_id=0, eos_token_id=1),
         space_marker_tokenizer("\u0120", [("kv", 5), ("hello", 7), ("onlyd", 9)],
                                unk_token_id=0, eos_token_id=1), 8, 11),
        # 混合标记 + 越界 id + 重复规范化
        (FakeTokenizer({"\u2581a": 4, "\u0120a": 9, "\u2581big": 20}, space_marker="\u2581",
                       unk_token_id=0),
         FakeTokenizer({"\u0120a": 2, "\u2581a": 6, "\u0120big": 3}, space_marker="\u0120",
                       unk_token_id=0), 11, 7),
    ]
    for target_tokenizer, draft_tokenizer, target_vocab_size, draft_vocab_size in cases:
        ours = VocabMapping(target_tokenizer, draft_tokenizer, target_vocab_size,
                            draft_vocab_size, device="cpu")
        theirs = UpstreamVocabMapping(target_tokenizer, draft_tokenizer, target_vocab_size,
                                      draft_vocab_size, device="cpu")
        assert ours.intersection_size == theirs.intersection_size
        assert torch.equal(ours.draft_to_target_ids, theirs.draft_to_target_ids)
        assert torch.equal(ours.target_to_draft_ids, theirs.target_to_draft_ids)
        assert torch.equal(ours.intersection_mask_draft, theirs.intersection_mask_draft)
        assert ours.target_unk_token_id == theirs.target_unk_token_id
        assert ours.draft_unk_token_id == theirs.draft_unk_token_id
        # 两个方向 + 约束，输出逐位相同
        target_ids = torch.arange(target_vocab_size, dtype=torch.int64)
        draft_ids = torch.arange(draft_vocab_size, dtype=torch.int64)
        assert torch.equal(ours.map_target_to_draft_ids(target_ids),
                           theirs.map_target_to_draft_ids(target_ids))
        assert torch.equal(ours.map_draft_to_target_ids(draft_ids),
                           theirs.map_draft_to_target_ids(draft_ids))
        logits = torch.randn(3, draft_vocab_size, dtype=torch.float32)
        assert torch.equal(ours.constrain_draft_logits(logits),
                           theirs.constrain_draft_logits(logits))


def test_real_tokenizer_intersection_is_recorded():
    """真实规模记一条：Qwen3-1.7B 的 tokenizer vs gpt2（缓存里没有 gpt2 时跳过并写明原因）。"""
    target_dir = ROOT / "models" / "Qwen3-1.7B"
    if not (target_dir / "tokenizer.json").is_file():
        pytest.skip("本机没有 models/Qwen3-1.7B 的 tokenizer：真实规模待验，不是通过")
    try:
        target_tokenizer = load_tokenizer(str(target_dir))
        draft_tokenizer = load_tokenizer("gpt2")                  # 需要 HF 缓存里有 gpt2
    except Exception as exc:                                      # noqa: BLE001
        pytest.skip(f"gpt2 tokenizer 不在本地缓存（local_files_only）：{exc}")
    mapping = VocabMapping(target_tokenizer, draft_tokenizer,
                           target_vocab_size=151936, draft_vocab_size=50257, device="cpu")
    assert mapping.intersection_size > 0
    # 记录（不是断言"必须是多少"：tokenizer 版本一变数字就会动）
    print(f"[step67] Qwen3-1.7B × gpt2 交集 = {mapping.stats}")


# ---------------------------------------------------------------- 4. 配置边界


def test_config_rejects_tli_outside_draft_model():
    with pytest.raises(ValueError, match="only works with method='draft_model'"):
        SpeculativeConfig(method="eagle3", num_speculative_tokens=1,
                          use_heterogeneous_vocab=True,
                          draft_model_config=ModelConfig(model="/x", hf_config={
                              "vocab_size": 11}))


def test_config_rejects_probabilistic_tli():
    """概率草稿的 TLI 明确拒绝（需求 067 §3.5：不能自行放开）。"""
    with pytest.raises(ValueError, match="only supports greedy draft sampling"):
        SpeculativeConfig(method="draft_model", num_speculative_tokens=1,
                          use_heterogeneous_vocab=True, draft_sample_method="probabilistic",
                          draft_model_config=ModelConfig(model="/x", hf_config={
                              "vocab_size": 11}))


def test_config_rejects_unknown_draft_sample_method():
    with pytest.raises(ValueError, match="draft_sample_method"):
        SpeculativeConfig(method="draft_model", num_speculative_tokens=1,
                          draft_sample_method="synthetic",
                          draft_model_config=ModelConfig(model="/x", hf_config={
                              "vocab_size": 11}))


def test_tokenizer_path_defaults_to_model():
    assert ModelConfig(model="/m").tokenizer_path == "/m"
    assert ModelConfig(model="/m", tokenizer="/t").tokenizer_path == "/t"


# ---------------------------------------------------------------- 5. 集成（真模型 + 真 tokenizer）


def build_engine(*, spec, target_dir, target_hf, num_gpu_blocks=32, budget=64):
    config = VllmConfig(
        model_config=ModelConfig(model=target_dir, dtype="float32", max_model_len=64,
                                 hf_config=target_hf),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=num_gpu_blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=DEVICE), speculative_config=spec)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner


def run(engine, prompts, *, max_tokens=6, temperature=0.0):
    for req_id, prompt in prompts:
        engine.add_request(req_id, list(prompt),
                           SamplingParams(max_tokens=max_tokens, temperature=temperature,
                                          eos_token_id=999))
    outputs = {}
    for _ in range(400):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            outputs[out.request_id] = list(out.token_ids)
    return outputs


def hetero_spec(*, k=K, **kwargs) -> tuple:
    target_dir, draft_dir, info = tiny_hetero_pair()
    draft_hf = json.loads((Path(draft_dir) / "config.json").read_text())
    spec = SpeculativeConfig(
        method="draft_model", num_speculative_tokens=k, use_heterogeneous_vocab=True,
        draft_model_config=ModelConfig(model=draft_dir, dtype="float32", max_model_len=64,
                                       hf_config=draft_hf), **kwargs)
    return target_dir, draft_dir, info, spec


@pytest.fixture(scope="module")
def hetero_engines():
    """一对引擎（非投机基线 + TLI 投机 K=2）跑同一组 prompt。"""
    target_dir, _, info, spec = hetero_spec()
    target_hf = json.loads((Path(target_dir) / "config.json").read_text())
    prompts = [("a", info["prompt_token_ids"]), ("b", info["prompt_token_ids"][:2])]

    plain_engine, _, _ = build_engine(spec=None, target_dir=target_dir, target_hf=target_hf)
    try:
        plain = run(plain_engine, prompts)
    finally:
        plain_engine.shutdown()

    engine, core, runner = build_engine(spec=spec, target_dir=target_dir, target_hf=target_hf)
    drafts = []
    original = runner.take_draft_token_ids

    def spy():
        result = original()
        if result is not None:
            drafts.append({"req_ids": list(result.req_ids),
                           "drafts": [list(row) for row in result.draft_token_ids],
                           "probs": result.draft_probs})
        return result

    runner.take_draft_token_ids = spy
    try:
        spec_outputs = run(engine, prompts)
        mapping = runner.proposer.vocab_mapping
        stats = dict(mapping.stats)
        target_to_draft = mapping.target_to_draft_ids.clone()
    finally:
        engine.shutdown()
    return {"plain": plain, "spec": spec_outputs, "drafts": drafts, "stats": stats,
            "target_to_draft": target_to_draft, "info": info}


def test_heterogeneous_greedy_matches_non_speculative(hetero_engines):
    """真实异构词表下 greedy 输出与非投机逐 token 相同（草稿只是候选）。"""
    assert hetero_engines["spec"] == hetero_engines["plain"]
    assert hetero_engines["drafts"], "一枚草稿都没提出来？这条集成就没验到 TLI 那条路"


def test_draft_ids_are_target_space_and_inside_intersection(hetero_engines):
    """交出去的草稿必须是 **target 空间**的 id，而且落在交集像里（draft 独有 token 永远不会出现）。"""
    info = hetero_engines["info"]
    allowed = {index for index, mapped in enumerate(hetero_engines["target_to_draft"].tolist())
               if mapped != -1}
    seen = set()
    for entry in hetero_engines["drafts"]:
        for tokens in entry["drafts"]:
            for token in tokens:
                assert token < info["target_vocab_size"], "草稿 id 超出 target 词表：id 空间串了"
                assert token in allowed, f"草稿 {token} 不在交集像里（draft 独有的 token 不该出门）"
                seen.add(token)
    assert seen


def test_tli_drafts_are_point_mass(hetero_engines):
    """TLI 的草稿是 argmax（点质量）：不带 `draft_probs`（draft 空间的 q 宽度与 target 不符）。"""
    for entry in hetero_engines["drafts"]:
        assert entry["probs"] is None
        assert all(len(tokens) <= K for tokens in entry["drafts"])


def test_first_pass_inputs_really_go_through_the_mapping():
    """反"假接线"：第一遍的历史行与扩容行**真的**过了 `map_target_to_draft_ids`。

    做法：把映射方法包一层，记录每次的入参/出参，然后断言
      (a) 被调用过（不是只建了表不用）；
      (b) 出参确实按表变了（入参里有不在交集的 token 时出参是 draft unk）；
      (c) 入参都是 **target 空间**的 id（< target 词表）。
    """
    target_dir, _, info, spec = hetero_spec()
    target_hf = json.loads((Path(target_dir) / "config.json").read_text())
    engine, _, runner = build_engine(spec=spec, target_dir=target_dir, target_hf=target_hf)
    calls = []
    mapping = runner.proposer.vocab_mapping
    original = mapping.map_target_to_draft_ids

    def spy(ids):
        result = original(ids)
        calls.append((ids.tolist(), result.tolist()))
        return result

    mapping.map_target_to_draft_ids = spy
    try:
        outputs = run(engine, [("a", info["prompt_token_ids"])])
    finally:
        engine.shutdown()
    assert outputs["a"]
    assert calls, "第一遍/自回归都没有调用映射：表建了但没接线"
    for inputs, outputs_ in calls:
        assert all(token < info["target_vocab_size"] for token in inputs)
    # 有一条调用里出现了 target 独有 token（prompt 里没有，但生成过程可能出现）或者映射结果与输入不同
    assert any(inputs != outputs_ for inputs, outputs_ in calls), "映射没有真正改变任何 id"


def test_non_greedy_request_still_uses_point_mass_drafts():
    """请求开温度时草稿仍是 argmax（TLI 只支持 greedy 草稿），端到端不报错、输出长度正常。"""
    target_dir, _, info, spec = hetero_spec()
    target_hf = json.loads((Path(target_dir) / "config.json").read_text())
    engine, _, runner = build_engine(spec=spec, target_dir=target_dir, target_hf=target_hf)
    drafts = []
    original = runner.take_draft_token_ids

    def spy():
        result = original()
        if result is not None:
            drafts.append(result)
        return result

    runner.take_draft_token_ids = spy
    try:
        outputs = run(engine, [("a", info["prompt_token_ids"])], max_tokens=6, temperature=0.8)
    finally:
        engine.shutdown()
    assert len(outputs["a"]) == 6
    assert drafts and all(entry.draft_probs is None for entry in drafts)


def test_vocab_mismatch_without_tli_still_raises():
    """不开 TLI 时词表不一致照旧明确报错（TLI 是**显式**开启的一项能力，不静默生效）。"""
    target_dir, draft_dir, _info, _ = hetero_spec()
    target_hf = json.loads((Path(target_dir) / "config.json").read_text())
    draft_hf = json.loads((Path(draft_dir) / "config.json").read_text())
    spec = SpeculativeConfig(method="draft_model", num_speculative_tokens=1,
                             draft_model_config=ModelConfig(model=draft_dir, dtype="float32",
                                                            max_model_len=64,
                                                            hf_config=draft_hf))
    with pytest.raises(ValueError, match="词表不一致"):
        build_engine(spec=spec, target_dir=target_dir, target_hf=target_hf)
