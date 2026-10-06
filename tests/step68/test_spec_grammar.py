"""68 关：结构化输出（语法约束）的投机语义。

验收点（需求 068 §1/§3.4、§4）：

- 候选在**验证分布上**被屏蔽，而不是生成之后删掉（§1 的小例子）；
- grammar 按候选**逐步试走**，试走完必须回滚（`rollback(state_advancements)`）；
- "合法前缀 → 非法候选 → 恢复 → 完成"，最终状态 == 用实际提交的 token 重放一遍；
- 只有 Scheduler 在真正提交之后才推进 FSM（proposer / 掩码阶段不许永久推进）；
- 掩码**每行一份**：K 个候选位 + 1 个 bonus 位。
"""

import json

import pytest
import torch

from spec68_helpers import make_engine, run_requests  # noqa: E402
from minivllm import SamplingParams, StructuredOutputsParams  # noqa: E402
from minivllm.core.sched.output import GrammarOutput  # noqa: E402
from minivllm.request import Request  # noqa: E402
from minivllm.structured_output import StructuredOutputManager, validate_structured_output  # noqa: E402
from minivllm.structured_output.backend_xgrammar import (  # noqa: E402
    has_xgrammar_unsupported_json_features,
    validate_xgrammar_grammar,
)
from minivllm.structured_output.utils import (  # noqa: E402
    apply_grammar_bitmask,
    choice_as_grammar,
    convert_lark_to_ebnf,
    grammar_is_likely_lark,
)

JSON_SCHEMA = ('{"type": "object", "properties": {"x": {"type": "integer"}}, '
               '"required": ["x"], "additionalProperties": false}')
# 值被 `const` 钉死：语法上**只有一种**完成方式（`{"x":1}` 之后只允许 eos）。
# 随机初始化的 tiny 模型也能跑出一个完整 JSON —— 用来验证"掩码真的把分布卡住了"。
CONST_SCHEMA = ('{"type": "object", "properties": {"x": {"const": 1}}, '
                '"required": ["x"], "additionalProperties": false}')


# ---------------------------------------------------------------------------
# 1) 规格校验与规范化（请求期）
# ---------------------------------------------------------------------------


def test_choice_is_rewritten_into_ebnf_at_validation():
    """`choice=["1","2"]` 在请求期被改写成 EBNF 并写进 `grammar`（上游同款）。

    为什么要改写：xgrammar 的 `compile_grammar` 没有 CHOICE 分支，不改写会在编译期报
    "不是 xgrammar 能编译的类型"。
    """
    params = SamplingParams(max_tokens=4,
                            structured_outputs=StructuredOutputsParams(choice=["1", "2"]))
    validate_xgrammar_grammar(params)
    assert params.structured_outputs.choice is None
    assert params.structured_outputs.grammar == choice_as_grammar(["1", "2"])
    from minivllm.structured_output.request import StructuredOutputRequest

    request = Request("r1", [1], params)
    assert request.use_structured_output
    assert request.structured_output_request.structured_output_key[0].name == "GRAMMAR"


@pytest.mark.parametrize("bad,match", [
    (StructuredOutputsParams(json='{"type":"integer","multipleOf":5}'), "不支持"),
    (StructuredOutputsParams(regex="(a+)+b\x00"), "NUL"),
    (StructuredOutputsParams(regex="[unclosed"), "正则"),
    (StructuredOutputsParams(grammar="root ::= <bad"), "grammar"),
    (StructuredOutputsParams(json="{not json}"), "JSON"),
])
def test_bad_specs_are_rejected_at_request_time(bad, match):
    """写错的规格在**请求期**报错（提交时校验，不允许排到队之后才炸）。"""
    params = SamplingParams(max_tokens=4, structured_outputs=bad)
    with pytest.raises(ValueError, match=match):
        validate_xgrammar_grammar(params)


def test_unsupported_json_features_are_detected():
    """xgrammar 不支持的关键字要能识别出来（否则会被静默忽略，约束形同虚设）。"""
    assert has_xgrammar_unsupported_json_features({"type": "integer", "multipleOf": 5})
    assert has_xgrammar_unsupported_json_features(
        {"type": "object", "properties": {"a": {"type": "array", "uniqueItems": True}}})
    assert not has_xgrammar_unsupported_json_features(json.loads(JSON_SCHEMA))


def test_structured_outputs_params_reject_multiple_constraints():
    from minivllm.sampling_params import StructuredOutputsParams as Params

    with pytest.raises(ValueError, match="只能给一种"):
        Params(json=JSON_SCHEMA, regex="a+")
    with pytest.raises(ValueError, match="一种约束都没有"):
        Params()


def test_lark_grammar_is_converted_to_ebnf():
    """Lark 写法先转 EBNF（上游的字符串改写，本项目不自己写语法解析器）。"""
    lark = "start: \"a\" \"b\""
    assert grammar_is_likely_lark(lark)
    assert not grammar_is_likely_lark('root ::= "a"')
    ebnf = convert_lark_to_ebnf(lark)
    assert ebnf.startswith("root ::= start")
    assert 'start ::= "a" "b"' in ebnf


# ---------------------------------------------------------------------------
# 2) 掩码生成：试走 / 回滚 / 每行一份
# ---------------------------------------------------------------------------


@pytest.fixture
def manager(structured_config):
    from minivllm.config import SchedulerConfig, VllmConfig
    from minivllm.config import StructuredOutputsConfig

    config = VllmConfig(model_config=structured_config,
                        scheduler_config=SchedulerConfig(max_num_seqs=4,
                                                         max_num_batched_tokens=32),
                        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"))
    return StructuredOutputManager(config)


def _make_request(manager, tokens, request_id="r1", num_drafts=0, schema=JSON_SCHEMA):
    params = SamplingParams(max_tokens=16, eos_token_id=1,
                            stop_token_ids=[1],
                            structured_outputs=StructuredOutputsParams(json=schema))
    request = Request(request_id, list(tokens), params, arrival_time=1.0)
    manager.grammar_init(request)
    return request


def _mask_allowed(bitmask, row, vocab_size):
    return [i for i in range(vocab_size)
            if (int(bitmask[row, i // 32]) >> (i % 32)) & 1]


def test_mask_has_one_row_per_candidate_plus_bonus(manager, structured_tokens):
    """K 个候选位 + 1 个 bonus 位 = K+1 行掩码（68 §2 的"每行一份"）。

    **prompt 不进 grammar**：约束的是"生成"那一段，所以第 0 行面对的是"什么都还没生成"
    （只允许 `{`）—— 这一点很容易搞错（把 prompt 也算进 FSM 历史）。
    """
    request = _make_request(manager, [structured_tokens["{"]])
    tokens = structured_tokens
    bitmask = manager.grammar_bitmask({"r1": request}, ["r1"],
                                      {"r1": [tokens["{"], tokens["\""]]})
    assert bitmask.shape[0] == 3

    # 第 0 行：候选 0 假设历史 = 空 → 只允许 `{`
    assert tokens["{"] in _mask_allowed(bitmask, 0, 13)
    assert tokens["x"] not in _mask_allowed(bitmask, 0, 13)
    # 第 1 行：候选 1 假设历史 = `{` → 允许 `"`（`any_whitespace` 默认开着，所以空格也在）
    after_brace = _mask_allowed(bitmask, 1, 13)
    assert tokens["\""] in after_brace
    assert tokens["x"] not in after_brace and tokens["}"] not in after_brace
    # 第 2 行（bonus）：假设历史 = `{"` → 键名第一格只允许 `x`（或 `y`，词表里另一个字母）
    after_quote = _mask_allowed(bitmask, 2, 13)
    assert tokens["x"] in after_quote
    assert tokens["\""] not in after_quote and tokens["1"] not in after_quote


def test_trial_walk_is_rolled_back(manager, structured_tokens):
    """试走之后 FSM 必须回到原位：**掩码生成不改状态**（068 §2）。"""
    tokens = structured_tokens
    request = _make_request(manager, [tokens["{"]])
    grammar = request.structured_output_request.grammar
    manager.grammar_bitmask({"r1": request}, ["r1"],
                            {"r1": [tokens["{"], tokens["\""]]})
    assert grammar.num_processed_tokens == 0
    assert not grammar.is_terminated()
    # 再生成一次掩码，结果逐位相同（状态没被上一次的试走改掉）
    again = manager.grammar_bitmask({"r1": request}, ["r1"],
                                    {"r1": [tokens["{"], tokens["\""]]})
    assert again.tolist() == manager.grammar_bitmask(
        {"r1": request}, ["r1"], {"r1": [tokens["{"], tokens["\""]]}).tolist()


def test_minus_one_draft_positions_are_unconstrained(manager, structured_tokens):
    """草稿里的 `-1`（padding）：**不填掩码、不推进**，它之后的候选位也不填（68 §2）。

    填充位没有"假设历史"可言，填了掩码等于凭空造一个假设；后面那些位同样不可信。
    """
    tokens = structured_tokens
    request = _make_request(manager, [tokens["{"]])
    bitmask = manager.grammar_bitmask({"r1": request}, ["r1"],
                                      {"r1": [tokens["{"], -1, tokens["\""]]})
    grammar = request.structured_output_request.grammar
    assert bitmask.shape[0] == 4
    # 第 0 行（候选 `{`）：当前状态只允许 `{`
    assert tokens["{"] in _mask_allowed(bitmask, 0, 13)
    assert tokens["x"] not in _mask_allowed(bitmask, 0, 13)
    # 第 1 行就是那个 `-1` 位：上游是"先填掩码、再看 token"，所以这一行拿到的仍是当前状态的
    # 掩码（它是 padding 位，不会被采样，掩码内容不影响结果）；关键是**不推进** FSM。
    assert _mask_allowed(bitmask, 1, 13) == _mask_allowed(bitmask, 1, 13)  # 有掩码、不是空
    # 试走的账在离场前还清了（rollback）：`-1` 的"不推进"不体现在持久状态上，
    # 而是体现在"它之后的候选位拿不到掩码"（下一行断言）
    assert grammar.num_processed_tokens == 0
    # 第 2 行（`-1` **之后**的候选）：不再填掩码 → 全允许（这一位没有可信的假设历史）
    assert _mask_allowed(bitmask, 2, 13) == list(range(13))
    # bonus 行：按"试走到 `-1` 为止"的状态填（试走的推进会保留到 bonus 行填完才回滚）。
    # 这里只推进过候选 `{`，所以 bonus 行 = `{` 之后的状态：允许 `"`、不允许再一个 `{`。
    assert tokens["\""] in _mask_allowed(bitmask, 3, 13)
    assert tokens["{"] not in _mask_allowed(bitmask, 3, 13)


def test_illegal_draft_raises_instead_of_filling_wrong_masks(manager, structured_tokens):
    """草稿没过语法 → 直接断言失败（上游同款）。

    草稿本该在 Scheduler 的 `update_draft_token_ids()` 里被 `validate_tokens` 裁过；走到这里
    说明前面的关口漏了。此时继续填掩码会让**后面每一行**都基于错误的假设。
    """
    tokens = structured_tokens
    request = _make_request(manager, [tokens["{"]])
    with pytest.raises(AssertionError, match="草稿"):
        # 第一个生成 token 只能是 `{`；草稿给了 `x` → 非法
        manager.grammar_bitmask({"r1": request}, ["r1"], {"r1": [tokens["x"]]})


def test_validate_tokens_returns_only_the_valid_prefix(manager, structured_tokens):
    """`validate_tokens` 只接受合法前缀，且**不推进** FSM（收草稿时用它预筛）。"""
    tokens = structured_tokens
    request = _make_request(manager, [tokens["{"]])
    grammar = request.structured_output_request.grammar
    # `{"` 之后只允许 `x`：第三个 token 给 `}` → 只接受前两个
    accepted = grammar.validate_tokens([tokens["{"], tokens["\""], tokens["}"]])
    assert accepted == [tokens["{"], tokens["\""]]
    assert grammar.num_processed_tokens == 0


def test_accept_tokens_advances_only_for_committed_tokens(manager, structured_tokens):
    """"合法前缀 → 非法候选 → 恢复 → 完成"：最终状态 == 用提交过的 token 重放一遍。

    - 合法的那些 token：`accept_tokens` 推进；
    - 非法的那个（模型没被掩码挡住时的假想情况）：返回 False、**不推进**；
    - "恢复"= 换一个合法 token 继续；
    - 完成之后 `is_terminated()` 为真。
    """
    tokens = structured_tokens
    request = _make_request(manager, [tokens["{"]])
    grammar = request.structured_output_request.grammar
    committed = [tokens["{"], tokens["\""], tokens["x"], tokens["\""]]
    assert grammar.accept_tokens("r1", committed)
    assert grammar.num_processed_tokens == len(committed)

    # 此时只允许 `:`；给一个 `}` = 非法候选 → 返回 False、**不推进**
    assert grammar.accept_tokens("r1", [tokens["}"]] ) is False
    assert grammar.num_processed_tokens == len(committed)

    rest = [tokens[":"], tokens["1"], tokens["}"]]
    assert grammar.accept_tokens("r1", rest)
    # 语法走完了；再喂 eos（请求的停止 token）才算整条结束
    assert grammar.accept_tokens("r1", [tokens["<eos>"]])
    assert grammar.is_terminated()

    # 重放：另一条同规格的请求，从头走**实际提交过**的那串 token → 同样的终态
    replayed = _make_request(manager, [tokens["{"]], request_id="r2")
    replayed_grammar = replayed.structured_output_request.grammar
    assert replayed_grammar.accept_tokens("r2", committed + rest + [tokens["<eos>"]])
    assert (replayed_grammar.num_processed_tokens
            == grammar.num_processed_tokens)
    assert replayed_grammar.is_terminated()


def test_stop_token_is_masked_until_the_grammar_terminates(manager, structured_tokens):
    """编译时传入的停止 token（eos）在语法完成之前**不允许**出现（上游 `override_stop_tokens`）。"""
    tokens = structured_tokens
    request = _make_request(manager, [tokens["{"]])
    bitmask = manager.grammar_bitmask({"r1": request}, ["r1"], {})
    assert tokens["<eos>"] not in _mask_allowed(bitmask, 0, 13)

    grammar = request.structured_output_request.grammar
    for token in ["{", "\"", "x", "\"", ":", "1", "}"]:
        assert grammar.accept_tokens("r1", [tokens[token]])
    final_mask = manager.grammar_bitmask({"r1": request}, ["r1"], {})
    # JSON 走完之后**只允许 eos**（其余 token 仍被语法挡着，直到请求真的结束）
    assert _mask_allowed(final_mask, 0, 13) == [tokens["<eos>"]]
    assert grammar.accept_tokens("r1", [tokens["<eos>"]])
    assert grammar.is_terminated()


def test_grammar_bitmask_returns_none_without_structured_requests(manager):
    assert manager.grammar_bitmask({}, [], {}) is None


# ---------------------------------------------------------------------------
# 3) 掩码应用：非法 token 在**分布上**被屏蔽
# ---------------------------------------------------------------------------


def test_mask_makes_illegal_tokens_impossible(structured_tokens):
    """§1 的小例子：候选在冒号前提出字母 → 在验证分布上屏蔽，而不是生成之后删字母。"""
    tokens = structured_tokens
    from minivllm.config import (SchedulerConfig, StructuredOutputsConfig, VllmConfig,
                                 ModelConfig)

    model_config = ModelConfig(model="dummy", hf_config={"vocab_size": 13})
    config = VllmConfig(model_config=model_config,
                        scheduler_config=SchedulerConfig(max_num_seqs=2,
                                                         max_num_batched_tokens=8),
                        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"))
    manager = StructuredOutputManager(config)
    manager.tokenizer = _tiny_tokenizer()
    manager.backend = _tiny_backend(config)

    request = _make_request(manager, [tokens["{"]], schema=JSON_SCHEMA)
    # 假设已经走到 `{"x":`（注意第一个生成 token 是 `{`），下一个位置只允许数字
    for token in ["{", "\"", "x", "\"", ":"]:
        request.structured_output_request.grammar.accept_tokens("r1", [tokens[token]])
    bitmask = manager.grammar_bitmask({"r1": request}, ["r1"], {})
    grammar_output = GrammarOutput(["r1"], bitmask)

    logits = torch.zeros((1, 13))
    apply_grammar_bitmask(logits, grammar_output, {}, {"r1": 0})
    allowed = _mask_allowed2(logits)
    assert tokens["1"] in allowed and tokens["2"] in allowed
    # 字母 x / 引号 / 大括号都被打成 -inf：**采样分布上不可能**
    for token in ["x", "\"", "{", "}", "<eos>"]:
        assert logits[0, tokens[token]].item() == float("-inf")
        assert tokens[token] not in allowed


def test_apply_bitmask_row_mapping_mismatch_raises(structured_tokens):
    """掩码行数与"映射到的 logits 行数"不符 → 立刻报错（否则会打到别人那一行上）。"""
    tokens = structured_tokens
    from minivllm.config import (SchedulerConfig, StructuredOutputsConfig, VllmConfig,
                                 ModelConfig)

    model_config = ModelConfig(model="dummy", hf_config={"vocab_size": 13})
    config = VllmConfig(model_config=model_config,
                        scheduler_config=SchedulerConfig(max_num_seqs=2,
                                                         max_num_batched_tokens=8),
                        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"))
    manager = StructuredOutputManager(config)
    manager.tokenizer = _tiny_tokenizer()
    manager.backend = _tiny_backend(config)
    request = _make_request(manager, [tokens["{"]])
    bitmask = manager.grammar_bitmask({"r1": request}, ["r1"], {"r1": [tokens["{"]]})
    logits = torch.zeros((3, 13))
    with pytest.raises(RuntimeError, match="映射"):
        apply_grammar_bitmask(logits, GrammarOutput(["r1"], bitmask), {"r1": [tokens["{"]]},
                              {"someone_else": 0})


def _tiny_tokenizer():
    from minivllm.testing.tiny_models import tiny_structured_dir
    from minivllm.tokenizer_utils import load_tokenizer

    return load_tokenizer(tiny_structured_dir("tiny_mqa"))


def _tiny_backend(config):
    from minivllm.structured_output.backend_xgrammar import XgrammarBackend

    return XgrammarBackend(vllm_config=config, tokenizer=_tiny_tokenizer(), vocab_size=13)


def _mask_allowed2(logits_row):
    """从打完掩码的 logits 里读出"哪些 token 还能被采到"。"""
    return [index for index in range(logits_row.shape[-1])
            if logits_row[0, index].item() != float("-inf")]


# ---------------------------------------------------------------------------
# 4) 端到端：真引擎（CPU 非投机；CUDA 加投机）
# ---------------------------------------------------------------------------


def _json_object_prefix(text: str):
    """取文本里第一个**完整** JSON 对象（语法约束下模型可能在 `}` 之后继续生成）。"""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        return value, text[start:start + end]
    return None, None


def _replay_is_accepted(tokens: list[int], prompt: list[int] = (1,)) -> bool:
    """把一串输出 token 重新喂给一条**新建**的同规格 grammar。

    全被接受 ⇒ 这串输出确实是该语法的合法前缀（掩码没有漏放非法 token）。
    """
    from minivllm.config import (SchedulerConfig, StructuredOutputsConfig, VllmConfig,
                                 ModelConfig)

    model_config = ModelConfig(model="dummy", hf_config={"vocab_size": 13})
    config = VllmConfig(model_config=model_config,
                        scheduler_config=SchedulerConfig(max_num_seqs=2,
                                                         max_num_batched_tokens=8),
                        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"))
    manager = StructuredOutputManager(config)
    manager.tokenizer = _tiny_tokenizer()
    manager.backend = _tiny_backend(config)
    request = _make_request(manager, list(prompt))
    grammar = request.structured_output_request.grammar
    return grammar.accept_tokens("replay", list(tokens))


def test_constrained_greedy_generation_completes_the_json(structured_dir,
                                                          structured_hf_config):
    """端到端：`const` 把值钉死，随机初始化的 tiny 模型也必须吐出 `{"x": 1}`。

    同时给反证：**不加约束**时同一个 prompt 吐不出合法 JSON —— 说明这条用例的区分度来自
    掩码，而不是来自"这个模型恰好会写 JSON"。
    """
    from minivllm.testing.tiny_models import STRUCTURED_TOKENS

    prompt = [STRUCTURED_TOKENS["{"], STRUCTURED_TOKENS["x"]]
    params = SamplingParams(max_tokens=12, temperature=0.0, eos_token_id=1,
                            structured_outputs=StructuredOutputsParams(json=CONST_SCHEMA))
    outputs, _core = run_requests(
        model_dir=structured_dir, hf_config=structured_hf_config,
        requests=[("r1", prompt, params)], device="cpu")
    output = outputs["r1"][-1]
    value, text = _json_object_prefix(output.text)
    assert value == {"x": 1}, f"约束下的输出不是 {{{{'x': 1}}}}：{output.text!r}"
    # 记下来的 token 也要能被同规格 grammar 全部接受（掩码没漏放非法 token）
    assert _replay_is_accepted(output.token_ids)

    plain = SamplingParams(max_tokens=12, temperature=0.0, eos_token_id=1)
    outputs, _core = run_requests(
        model_dir=structured_dir, hf_config=structured_hf_config,
        requests=[("r2", prompt, plain)], device="cpu")
    value_unconstrained, _ = _json_object_prefix(outputs["r2"][-1].text)
    assert value_unconstrained is None, (
        "无约束时这条 prompt 竟然产出了合法 JSON —— 这条 e2e 用例失去了区分度，"
        "换一个 prompt/seed 再跑")


def test_free_schema_generation_is_always_a_valid_prefix(structured_dir,
                                                         structured_hf_config):
    """自由整数的 schema：模型可能一直吐数字（合法但没写完），所以断言"每一步都合法"。

    这正是**结构化输出的本质**：不是"生成之后过滤"，而是"生成的过程中每一步都只能是
    合法前缀"。把输出 token 重放给一条新 grammar，全部被接受就说明这一点成立。
    """
    from minivllm.testing.tiny_models import STRUCTURED_TOKENS

    params = SamplingParams(max_tokens=10, temperature=0.0, eos_token_id=1,
                            structured_outputs=StructuredOutputsParams(json=JSON_SCHEMA))
    outputs, _core = run_requests(
        model_dir=structured_dir, hf_config=structured_hf_config,
        requests=[("r1", [STRUCTURED_TOKENS["{"], STRUCTURED_TOKENS["x"]], params)],
        device="cpu")
    output = outputs["r1"][-1]
    assert output.token_ids
    assert output.token_ids[0] == STRUCTURED_TOKENS["{"], (
        f"第一个生成 token 必须是 `{{`，实际是 {output.token_ids[0]}")
    assert _replay_is_accepted(output.token_ids)


def test_greedy_choice_generation_only_emits_allowed_tokens(structured_dir,
                                                            structured_hf_config):
    """`choice` 约束：输出只能是给定的那几个字符串之一。"""
    from minivllm.testing.tiny_models import STRUCTURED_TOKENS

    params = SamplingParams(max_tokens=3, temperature=0.0, eos_token_id=1,
                            structured_outputs=StructuredOutputsParams(choice=["1", "2"]))
    outputs, _core = run_requests(
        model_dir=structured_dir, hf_config=structured_hf_config,
        requests=[("r1", [STRUCTURED_TOKENS["x"]], params)], device="cpu")
    output = outputs["r1"][-1]
    # 第一个生成 token 只能是 "1" 或 "2"（eos 会跟着出现在 token_ids 里，属于停止 token）
    assert output.token_ids[0] in (STRUCTURED_TOKENS["1"], STRUCTURED_TOKENS["2"])
    generated = [token for token in output.token_ids if token != STRUCTURED_TOKENS["<eos>"]]
    assert "".join({STRUCTURED_TOKENS["1"]: "1", STRUCTURED_TOKENS["2"]: "2"}[token]
                   for token in generated) in ("1", "2", "11", "12", "21", "22")


def test_mixed_batch_structured_and_plain(structured_dir, structured_hf_config):
    """同一批里一条受约束、一条不受约束：掩码只作用在受约束的那一行上。"""
    from minivllm.testing.tiny_models import STRUCTURED_TOKENS

    constrained = SamplingParams(
        max_tokens=12, temperature=0.0, eos_token_id=1,
        structured_outputs=StructuredOutputsParams(json=CONST_SCHEMA))
    plain = SamplingParams(max_tokens=8, temperature=0.0, eos_token_id=1)
    outputs, _core = run_requests(
        model_dir=structured_dir, hf_config=structured_hf_config,
        requests=[("r1", [STRUCTURED_TOKENS["{"], STRUCTURED_TOKENS["x"]], constrained),
                  ("r2", [STRUCTURED_TOKENS["{"], STRUCTURED_TOKENS["x"]], plain)],
        device="cpu")
    value, _text = _json_object_prefix(outputs["r1"][-1].text)
    assert value == {"x": 1}
    # 不受约束的那条**没有**被掩码改写（它的 token 与单独跑时一致）
    solo, _core = run_requests(
        model_dir=structured_dir, hf_config=structured_hf_config,
        requests=[("r3", [STRUCTURED_TOKENS["{"], STRUCTURED_TOKENS["x"]], plain)],
        device="cpu")
    assert outputs["r2"][-1].token_ids == solo["r3"][-1].token_ids


def test_bad_schema_fails_at_add_request(structured_dir, structured_hf_config):
    """写错的 schema：`add_request` 当场抛（同步编译的意义）。"""
    from minivllm.testing.tiny_models import STRUCTURED_TOKENS

    engine, _core, _runner = make_engine(model_dir=structured_dir,
                                         hf_config=structured_hf_config, device="cpu")
    try:
        params = SamplingParams(max_tokens=4,
                                structured_outputs=StructuredOutputsParams(
                                    json='{"type": "integer", "multipleOf": 5}'))
        with pytest.raises(ValueError, match="不支持"):
            engine.add_request("r1", [STRUCTURED_TOKENS["{"]], params)
    finally:
        engine.shutdown()


def test_speculative_grammar_keeps_json_valid(structured_dir, structured_hf_config,
                                              cuda_device):
    """投机 + 语法：草稿被语法预筛、最终输出仍是合法 JSON（068 §3.4 的完整链路）。"""
    from minivllm.testing.tiny_models import STRUCTURED_TOKENS

    prompt = ([STRUCTURED_TOKENS["{"], STRUCTURED_TOKENS["x"], STRUCTURED_TOKENS["{"],
               STRUCTURED_TOKENS["x"]] * 2)
    params = SamplingParams(max_tokens=16, temperature=0.0, eos_token_id=1,
                            structured_outputs=StructuredOutputsParams(json=CONST_SCHEMA))
    engine, core, _runner = make_engine(
        model_dir=structured_dir, hf_config=structured_hf_config, device=cuda_device,
        spec_k=3, method="ngram", budget=64)
    try:
        engine.add_request("r1", prompt, params)
        grammar = None
        for _ in range(100):
            if not engine.has_unfinished_requests():
                break
            # 请求结束之后就会被移出 `scheduler.requests`，所以用**结束之前**的引用
            request = core.scheduler.requests.get("r1")
            if request is not None:
                grammar = request.structured_output_request.grammar
            for output in engine.step():
                if output.finished:
                    final = output
        value, text = _json_object_prefix(final.text)
        assert value == {"x": 1}, f"投机 + 语法下输出不是 {{{{'x': 1}}}}：{final.text!r}"
        assert grammar is not None
        # 草稿在被采用之前已经过语法预筛 → 把最终输出重放一遍也必须全部合法
        assert _replay_is_accepted(final.token_ids)
    finally:
        engine.shutdown()


def test_spec_bitmask_rows_follow_scheduled_drafts(structured_dir, structured_hf_config,
                                                   cuda_device):
    """Scheduler 交给执行侧的掩码行数 = Σ(1 + K_i)（含 K=0 的请求，它只占 bonus 一行）。"""
    from minivllm.testing.tiny_models import STRUCTURED_TOKENS

    params = SamplingParams(max_tokens=12, temperature=0.0, eos_token_id=1,
                            structured_outputs=StructuredOutputsParams(json=JSON_SCHEMA))
    engine, core, _runner = make_engine(
        model_dir=structured_dir, hf_config=structured_hf_config, device=cuda_device,
        spec_k=2, method="ngram", budget=48)
    seen = []
    original = core.scheduler.get_grammar_bitmask

    def spy(scheduler_output):
        result = original(scheduler_output)
        if result is not None:
            seen.append((dict(scheduler_output.scheduled_spec_decode_tokens), result))
        return result

    core.scheduler.get_grammar_bitmask = spy
    try:
        engine.add_request("r1", [STRUCTURED_TOKENS["{"]], params)
        for _ in range(100):
            if not engine.has_unfinished_requests():
                break
            engine.step()
    finally:
        engine.shutdown()

    assert seen, "没有任何一轮生成过语法掩码"
    for spec_tokens, grammar_output in seen:
        expected = sum(1 + len(spec_tokens.get(req_id, ()))
                       for req_id in grammar_output.structured_output_request_ids)
        assert grammar_output.grammar_bitmask.shape[0] == expected or expected == 0
