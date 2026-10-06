"""68 关验收脚本（B）：结构化输出（语法约束）的投机语义（需求 068 §2/§3.4/§5）。

脚本式 PASS/FAIL（供回归用），口径与 `tests/step68/` 一致：

  A. 规格层：choice→EBNF 改写、Lark→EBNF、xgrammar 不支持的关键字、坏规格请求期报错
  B. 掩码层：每请求 1+K 行、试走 + 回滚（掩码生成不改状态）、`-1` 语义、非法草稿断言
  C. 状态层：`accept_tokens` 只推进合法 token、非法候选不推进、终态 == 提交序列的重放
  D. 应用层：非法 token 被 -inf（**分布上**不可能）、行映射不符就报错
  E. 端到端（CPU）：`const` schema 强制完整 JSON、自由 schema 每步合法、混批互不影响
  F. 端到端（CUDA，有卡时）：投机 + 语法仍产出完整 JSON、掩码行数 == Σ(1+K)
  G. 边界：只接 xgrammar / auto，其余后端明确拒绝
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step68"))

import test_spec_grammar as g  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def _raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc as err:
        return True, str(err)[:70]
    except Exception as err:               # noqa: BLE001 —— 别的异常不算"预期失败"
        return False, f"{type(err).__name__}: {err}"
    return False, "没有抛异常"


from minivllm import SamplingParams, StructuredOutputsParams  # noqa: E402
from minivllm.config import (ModelConfig, SchedulerConfig, StructuredOutputsConfig,  # noqa: E402
                             VllmConfig)

TOKENS = g.__dict__.get("STRUCTURED_TOKENS") or {}
from minivllm.testing.tiny_models import STRUCTURED_TOKENS, tiny_structured_dir  # noqa: E402
from minivllm.tokenizer_utils import load_tokenizer  # noqa: E402

TOKENS = STRUCTURED_TOKENS
MODEL_DIR = tiny_structured_dir("tiny_mqa")
import json  # noqa: E402

HF_CONFIG = json.loads(Path(MODEL_DIR, "config.json").read_text())
C = 13


def make_manager(config=None, vocab_size=C):
    from minivllm.structured_output import StructuredOutputManager
    from minivllm.structured_output.backend_xgrammar import XgrammarBackend

    config = config or VllmConfig(
        model_config=ModelConfig(model=MODEL_DIR, hf_config=HF_CONFIG, dtype="float32",
                                 tokenizer=MODEL_DIR),
        scheduler_config=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=32),
        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"))
    manager = StructuredOutputManager(config)
    manager.tokenizer = load_tokenizer(MODEL_DIR)
    manager.backend = XgrammarBackend(vllm_config=config, tokenizer=manager.tokenizer,
                                      vocab_size=vocab_size)
    return manager


def make_request(manager, schema=g.CONST_SCHEMA, request_id="r1", prompt=(2,)):
    from minivllm.request import Request

    params = SamplingParams(max_tokens=16, eos_token_id=1, stop_token_ids=[1],
                            structured_outputs=StructuredOutputsParams(json=schema))
    request = Request(request_id, list(prompt), params, arrival_time=1.0)
    manager.grammar_init(request)
    return request


def allowed(bitmask, row, vocab_size=C):
    return [i for i in range(vocab_size)
            if (int(bitmask[row, i // 32]) >> (i % 32)) & 1]


# ---------------------------------------------------------------- A. 规格层
params = SamplingParams(max_tokens=4,
                        structured_outputs=StructuredOutputsParams(choice=["1", "2"]))
g.validate_xgrammar_grammar(params)
check("A1. choice 在请求期被改写成 EBNF（xgrammar 的 compile_grammar 没有 CHOICE 分支）",
      params.structured_outputs.choice is None
      and params.structured_outputs.grammar == g.choice_as_grammar(["1", "2"]))

lark = 'start: "a" "b"'
check("A2. Lark → EBNF 转换",
      g.grammar_is_likely_lark(lark) and 'start ::= "a" "b"' in g.convert_lark_to_ebnf(lark))

check("A3. xgrammar 不支持的关键字被识别（multipleOf / uniqueItems / 未知 format）",
      g.has_xgrammar_unsupported_json_features({"type": "integer", "multipleOf": 5})
      and g.has_xgrammar_unsupported_json_features(
          {"type": "array", "uniqueItems": True})
      and g.has_xgrammar_unsupported_json_features({"type": "string", "format": "no-such"})
      and not g.has_xgrammar_unsupported_json_features(json.loads(g.JSON_SCHEMA)))

ok, detail = _raises(ValueError, g.validate_xgrammar_grammar,
                     SamplingParams(max_tokens=4,
                                    structured_outputs=StructuredOutputsParams(
                                        json='{"type":"integer","multipleOf":5}')))
check("A4. 不支持的 schema 在请求期报错（不静默忽略关键字）", ok, detail)
ok, detail = _raises(ValueError, g.validate_xgrammar_grammar,
                     SamplingParams(max_tokens=4,
                                    structured_outputs=StructuredOutputsParams(regex="[a-")))
check("A5. 坏正则请求期报错", ok, detail)
ok, detail = _raises(ValueError, StructuredOutputsParams, json="{}", regex="a+")
check("A6. 一次只能给一种约束", ok, detail)

# ---------------------------------------------------------------- B. 掩码层
manager = make_manager()
request = make_request(manager)
bitmask = manager.grammar_bitmask({"r1": request}, ["r1"],
                                  {"r1": [TOKENS["{"], TOKENS['"']]})
check("B1. 每请求 1+K 份掩码（K=2 → 3 行）", bitmask.shape[0] == 3, f"shape={bitmask.shape}")
check("B2. 第 0 行 = 未生成任何 token 的状态（只允许 `{`）",
      TOKENS["{"] in allowed(bitmask, 0) and TOKENS["x"] not in allowed(bitmask, 0))
check("B3. 第 1 行 = 试走 `{` 之后的状态（允许 `\"`、不允许再 `{`）",
      TOKENS['"'] in allowed(bitmask, 1) and TOKENS["{"] not in allowed(bitmask, 1))
grammar = request.structured_output_request.grammar
check("B4. 试走之后 FSM 已回滚（掩码生成不改持久状态）",
      grammar.num_processed_tokens == 0 and not grammar.is_terminated())

request2 = make_request(manager, request_id="r2")
bitmask2 = manager.grammar_bitmask({"r2": request2}, ["r2"],
                                   {"r2": [TOKENS["{"], -1, TOKENS['"']]})
check("B5. `-1` 之后的候选位不填掩码（全允许）",
      allowed(bitmask2, 2) == list(range(C)) and allowed(bitmask2, 1) != list(range(C)),
      f"row1={allowed(bitmask2, 1)} row2={allowed(bitmask2, 2)}")

request3 = make_request(manager, request_id="r3")
ok, detail = _raises(AssertionError, manager.grammar_bitmask, {"r3": request3}, ["r3"],
                     {"r3": [TOKENS["x"]]})
check("B6. 草稿没过语法 → 断言失败（不许带着错误假设继续填掩码）", ok, detail)

request4 = make_request(manager, request_id="r4")
mask4 = manager.grammar_bitmask({"r4": request4}, ["r4"], {})
check("B7. 停止 token（eos）在语法走完之前不允许出现",
      TOKENS["<eos>"] not in allowed(mask4, 0))

# ---------------------------------------------------------------- C. 状态层
request5 = make_request(manager, request_id="r5")
grammar5 = request5.structured_output_request.grammar
accepted_prefix = grammar5.validate_tokens([TOKENS["{"], TOKENS['"'], TOKENS["}"]])
check("C1. validate_tokens 只接受合法前缀、且不推进",
      accepted_prefix == [TOKENS["{"], TOKENS['"']] and grammar5.num_processed_tokens == 0)

committed = [TOKENS["{"], TOKENS['"'], TOKENS["x"], TOKENS['"']]
forward = grammar5.accept_tokens("r5", committed)
advance_after_illegal = grammar5.accept_tokens("r5", [TOKENS["}"]])
rest = [TOKENS[":"], TOKENS["1"], TOKENS["}"]]
finish = grammar5.accept_tokens("r5", rest)
check("C2. 合法 token 推进；非法候选返回 False 且不推进",
      forward and advance_after_illegal is False
      and grammar5.num_processed_tokens == len(committed) + len(rest),
      f"processed={grammar5.num_processed_tokens}")

replay = make_request(manager, request_id="r6")
ok_replay = replay.structured_output_request.grammar.accept_tokens(
    "r6", committed + rest + [TOKENS["<eos>"]])
check("C3. 终态 == 用**实际提交过**的 token 重放一遍（含 eos 收尾）",
      ok_replay and replay.structured_output_request.grammar.is_terminated())

# ---------------------------------------------------------------- D. 应用层
from minivllm.core.sched.output import GrammarOutput  # noqa: E402
from minivllm.structured_output.utils import apply_grammar_bitmask  # noqa: E402

request7 = make_request(manager, request_id="r7")
for token in ["{", '"', "x", '"', ":"]:
    request7.structured_output_request.grammar.accept_tokens("r7", [TOKENS[token]])
mask7 = manager.grammar_bitmask({"r7": request7}, ["r7"], {})
logits = torch.zeros((1, C))
apply_grammar_bitmask(logits, GrammarOutput(["r7"], mask7), {}, {"r7": 0})
cols = [i for i in range(C) if logits[0, i].item() != float("-inf")]
check("D1. 掩码把非法 token 打成 -inf（`{` 之后只允许 `\"`）",
      cols and all(logits[0, i].item() == 0.0 for i in cols)
      and logits[0, TOKENS["x"]].item() == float("-inf"))

ok, detail = _raises(RuntimeError, apply_grammar_bitmask, torch.zeros((3, C)),
                     GrammarOutput(["r7"], mask7), {}, {"someone_else": 0})
check("D2. 请求 → logits 行的映射不符就报错（避免掩码打到别人那一行）", ok, detail)

# ---------------------------------------------------------------- E. 端到端（CPU）
from spec68_helpers import run_requests  # noqa: E402

outputs, _core = run_requests(
    model_dir=MODEL_DIR, hf_config=HF_CONFIG, device="cpu",
    requests=[("e1", [TOKENS["{"], TOKENS["x"]],
               SamplingParams(max_tokens=12, temperature=0.0, eos_token_id=1,
                              structured_outputs=StructuredOutputsParams(
                                  json=g.CONST_SCHEMA)))])
value, _text = g._json_object_prefix(outputs["e1"][-1].text)
check("E1. `const` schema 下随机初始化的 tiny 模型也吐出完整 JSON", value == {"x": 1},
      f"text={outputs['e1'][-1].text!r}")
check("E2. 输出 token 能被同规格 grammar 重放接受",
      g._replay_is_accepted(outputs["e1"][-1].token_ids))

outputs, _core = run_requests(
    model_dir=MODEL_DIR, hf_config=HF_CONFIG, device="cpu",
    requests=[("e3", [TOKENS["{"], TOKENS["x"]],
               SamplingParams(max_tokens=10, temperature=0.0, eos_token_id=1,
                              structured_outputs=StructuredOutputsParams(
                                  json=g.JSON_SCHEMA)))])
free_tokens = outputs["e3"][-1].token_ids
check("E3. 自由整数 schema：每一步都是合法前缀（可能没写完），首 token 必是 `{`",
      free_tokens and free_tokens[0] == TOKENS["{"] and g._replay_is_accepted(free_tokens),
      f"tokens={free_tokens}")

from spec68_helpers import make_engine  # noqa: E402

engine, _core, _runner = make_engine(model_dir=MODEL_DIR, hf_config=HF_CONFIG, device="cpu")
ok, detail = _raises(ValueError, engine.add_request, "e4", [TOKENS["{"]], SamplingParams(
    max_tokens=4, structured_outputs=StructuredOutputsParams(
        json='{"type": "integer", "multipleOf": 5}')))
engine.shutdown()
check("E4. 坏 schema 在 `add_request` 当场失败（同步编译的意义）", ok, detail)

# ---------------------------------------------------------------- F. 端到端（CUDA）
if torch.cuda.is_available():
    # 自由整数的 schema + 足够长的预算：模型会重复吐数字 → ngram 提议者能给出草稿，
    # 于是掩码生成真的会**逐步试走草稿**（这是本关最核心的一条链路）。
    engine, core, _runner = make_engine(
        model_dir=MODEL_DIR, hf_config=HF_CONFIG, device="cuda", spec_k=3, method="ngram",
        budget=64)
    seen = []
    original = core.scheduler.get_grammar_bitmask

    def spy(scheduler_output):
        result = original(scheduler_output)
        if result is not None:
            # 记下"这一轮开始时已提交的输出"：草稿的合法性是相对**那个状态**说的
            request = core.scheduler.requests.get("f1")
            committed = list(request.output_token_ids) if request is not None else None
            seen.append((dict(scheduler_output.scheduled_spec_decode_tokens), result,
                         committed))
        return result

    core.scheduler.get_grammar_bitmask = spy
    prompt = [TOKENS["{"], TOKENS["x"]] * 3
    engine.add_request("f1", prompt, SamplingParams(
        max_tokens=24, temperature=0.0, eos_token_id=1,
        structured_outputs=StructuredOutputsParams(json=g.JSON_SCHEMA)))
    final = None
    for _ in range(120):
        if not engine.has_unfinished_requests():
            break
        for output in engine.step():
            if output.finished:
                final = output
    engine.shutdown()

    rounds_with_drafts = [spec for spec, _r, _c in seen if spec]
    check("F1. 真走到了「带草稿」的轮（否则投机语义没被测到）", bool(rounds_with_drafts),
          f"{len(rounds_with_drafts)}/{len(seen)} 轮有草稿")
    rows_ok = all(
        result.grammar_bitmask.shape[0] == sum(
            1 + len(spec_tokens.get(req_id, ()))
            for req_id in result.structured_output_request_ids)
        for spec_tokens, result, _committed in seen)
    check("F2. 掩码行数 == Σ(1 + K_i)（含 K=0 的请求只占 bonus 一行）", rows_ok and bool(seen),
          f"{len(seen)} 轮")

    # 草稿的合法性是相对"这一轮开始时的已提交历史"说的：重建那个状态再试走一遍，
    # 被采用的草稿必须**整段**都能被语法接受（Scheduler 的 validate_tokens 预筛过）。
    drafts_ok = True
    checked = 0
    for spec_tokens, _result, committed in seen:
        drafts = spec_tokens.get("f1", ())
        if not drafts or committed is None:
            continue
        checked += 1
        # 用**同一份 schema**重建状态（用错 schema 会让这条检查自己失败）
        grammar = make_request(make_manager(), schema=g.JSON_SCHEMA,
                               request_id="f-check").structured_output_request.grammar
        if committed and not grammar.accept_tokens("f-check", list(committed)):
            drafts_ok = False
            break
        if grammar.validate_tokens(list(drafts)) != list(drafts):
            drafts_ok = False
            break
    check("F3. 每轮被采用的草稿都是「当时语法状态」的合法前缀（预筛生效）",
          drafts_ok and checked > 0, f"检查了 {checked} 轮")
    check("F4. 最终输出 token 全部是语法的合法前缀（掩码没有漏放非法 token）",
          g._replay_is_accepted(final.token_ids),
          f"tokens={final.token_ids}")
else:
    check("F1. 投机 + 语法（需要 CUDA 的拒绝采样内核）", False, "本机没有 CUDA：待验")

# ---------------------------------------------------------------- G. 边界
for backend in ("guidance", "outlines", "lm-format-enforcer"):
    from minivllm.structured_output import StructuredOutputManager
    from minivllm.structured_output import validate_structured_output

    ok, detail = _raises(NotImplementedError, validate_structured_output,
                         SamplingParams(max_tokens=4, structured_outputs=StructuredOutputsParams(
                             json_object=True)),
                         StructuredOutputsConfig(backend=backend))
    check(f"G1. 未接入的后端明确拒绝：{backend}", ok, detail)

ok, detail = _raises(ValueError, StructuredOutputsConfig, backend="no-such-backend")
check("G2. 未知后端在配置期就报错", ok, detail)

print()
print("口径说明（完整版见 docs/step68_alignment.md §3）：")
print("  · 掩码行序 = 每个结构化请求 `1 + K` 行（K 个候选位 + 1 个 bonus 位），与执行侧的")
print("    logits 行序逐请求对齐；行映射由 Runner 给出（本仓库只为要采样的行算 logits）。")
print("  · 试走只在**掩码生成**里发生，离去前统一 rollback；永久推进只发生在 Scheduler")
print("    真正提交 token 之后（068 §2：「不要让 proposer 永久推进 grammar」）。")
print("  · 上游行为照抄的一处细节：`-1` 那一位仍会按当前试走状态填掩码（填在判 `-1` 之前），")
print("    但它**之后**的候选位不再填、也不再推进。")
print(f"设备={'cuda' if torch.cuda.is_available() else 'cpu'}；tokenizer={MODEL_DIR}")
print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
