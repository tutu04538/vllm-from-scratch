"""67 关验收脚本：异构词表 TLI 与草案采样空间（需求 067 §2/§3/§4）。

脚本式 PASS/FAIL（供回归用），口径与 `tests/step67/` 一致：

  A. 构造：同词不同 id、两族空格标记（▁/Ġ）、重复规范化取第一个、越界条目丢掉、
     unk=0 是合法 id、无 unk 退 eos、两者皆无报错、空交集按源码行为记录
  B. 映射：两个方向精确（交集上往返一致）、不改写入参、非交集列永远选不到、**不重新分词**
  C. 上游差分：同一对假 tokenizer 喂给 site-packages 里真的 `VocabMapping`，逐张量/逐输出比对
  D. 配置边界：只支持 `draft_model`；概率草稿的 TLI 明确拒绝；`draft_sample_method` 取值校验
  E. 集成（真 tiny 模型 + 真 tokenizer 文件）：greedy 与非投机逐 token 相同、草稿是 target 空间
     id 且在交集像内、点质量（不带 q）、第一遍输入真的过了映射（反「假接线」）
  F. 真实 tokenizer 规模：Qwen3-1.7B × gpt2 的交集大小（记录用；缺缓存时说明原因）
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step67"))

import test_vocab_mapping as tli  # noqa: E402

DEVICE = tli.DEVICE
FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def _raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    except Exception:                     # noqa: BLE001 —— 别的异常不算"预期失败"
        return False
    return False


def _warns(fn, *args, **kwargs):
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return any(issubclass(item.category, UserWarning) for item in caught), result


# ---------------------------------------------------------------- A. 构造层
from minivllm.spec_decode.vocab_mapping import (VocabMapping, _detect_space_prefix,  # noqa: E402
                                               _get_unk_token_id, _normalize_token)

target = tli.space_marker_tokenizer("\u2581", [("hello", 3), ("kv", 5), ("onlyt", 7), ("a", 8)],
                                    unk_token_id=0, eos_token_id=1)
draft = tli.space_marker_tokenizer("\u0120", [("kv", 9), ("hello", 2), ("onlyd", 4), ("a", 6)],
                                   unk_token_id=0, eos_token_id=1)
mapping = VocabMapping(target, draft, target_vocab_size=11, draft_vocab_size=13, device="cpu")
ids = torch.tensor([3, 5], dtype=torch.int64)
check("A1. 同一个词在两套 id 空间里不同号，交集按 token 字符串建，交集上往返精确",
      mapping.intersection_size == 3 and mapping.target_to_draft_ids[3] == 2
      and mapping.draft_to_target_ids[9] == 5
      and torch.equal(mapping.map_draft_to_target_ids(mapping.map_target_to_draft_ids(ids)), ids),
      f"交集 {mapping.intersection_size} 个")

target_a = tli.space_marker_tokenizer("\u2581", [("hello", 1), ("a", 4)], unk_token_id=0)
draft_a = tli.space_marker_tokenizer("\u0120", [("hello", 7), ("a", 2)], unk_token_id=0)
markers = VocabMapping(target_a, draft_a, target_vocab_size=6, draft_vocab_size=8, device="cpu")
check("A2. 两族空格标记（SentencePiece ▁ / BPE Ġ）探测正确且归一化后能对上同一个词",
      _detect_space_prefix(target_a) == ("\u2581",)
      and _detect_space_prefix(draft_a) == ("\u0120",)
      and markers.target_to_draft_ids[1] == 7,
      f"▁hello(id 1) ↔ Ġhello(id 7)")

dup_target = tli.FakeTokenizer({"\u2581a": 4, "\u0120a": 9, "\u2581b": 5}, space_marker="\u2581",
                               unk_token_id=0)
dup_draft = tli.space_marker_tokenizer("\u0120", [("a", 2), ("b", 3)], unk_token_id=0)
dup = VocabMapping(dup_target, dup_draft, target_vocab_size=11, draft_vocab_size=5, device="cpu")
check("A3. 同一个 tokenizer 内规范化后重名的只留字典里第一个（上游同款）",
      dup.target_to_draft_ids[4] == 2 and dup.target_to_draft_ids[9] == -1)

big_target = tli.space_marker_tokenizer("\u2581", [("hello", 3), ("inrange", 5), ("toobig", 20)],
                                        unk_token_id=0)
big_draft = tli.space_marker_tokenizer("\u0120", [("hello", 2), ("inrange", 1), ("toobig", 9)],
                                       unk_token_id=0)
big = VocabMapping(big_target, big_draft, target_vocab_size=11, draft_vocab_size=10, device="cpu")
check("A4. 超出模型 vocab_size 的条目（tokenizer 多出来的 special token）不参与映射",
      big.intersection_size == 2 and tuple(big.target_to_draft_ids.shape) == (11,)
      and tuple(big.draft_to_target_ids.shape) == (10,),
      "越界 id 没有位置可放，直接不进表")

zero_target = tli.FakeTokenizer({"\u2581a": 4}, space_marker="\u2581", unk_token_id=0,
                                eos_token_id=5)
zero_draft = tli.FakeTokenizer({"\u0120b": 2}, space_marker="\u0120", unk_token_id=0,
                               eos_token_id=7)
zero = VocabMapping(zero_target, zero_draft, target_vocab_size=8, draft_vocab_size=8, device="cpu")
check("A5. unk_token_id = 0 被当成**合法 id**（不是 `unk or eos`）",
      zero.target_unk_token_id == 0
      and zero.map_target_to_draft_ids(torch.tensor([4])).tolist() == [0]
      and zero.map_draft_to_target_ids(torch.tensor([2])).tolist() == [0],
      "两边都填 0，而不是各自的 eos(5/7)")

warned, unk = _warns(_get_unk_token_id,
                     tli.FakeTokenizer({"\u2581a": 4}, space_marker="\u2581",
                                       unk_token_id=None, eos_token_id=3), "target tokenizer")
check("A6. 没有 unk 就退 eos，并且告警（上游同款顺序）", warned and unk == 3)
check("A7. 既没有 unk 也没有 eos → 明确报错（不猜）",
      _raises(ValueError, _get_unk_token_id,
              tli.FakeTokenizer({"\u2581a": 4}, space_marker="\u2581"), "draft tokenizer"))

empty_target = tli.space_marker_tokenizer("\u2581", [(f"t{i}", i) for i in range(4)],
                                          unk_token_id=0)
empty_draft = tli.space_marker_tokenizer("\u0120", [(f"d{i}", i) for i in range(4)],
                                         unk_token_id=0)
empty_warned, empty = _warns(VocabMapping, empty_target, empty_draft, 4, 4, "cpu")
check("A8. 交集为空/极小时只告警、照样建表（按源码行为记录，不假装支持任意 tokenizer）",
      empty_warned and empty.intersection_size == 0
      and bool(torch.isinf(empty.constrain_draft_logits(torch.randn(2, 4))).all()))

# ---------------------------------------------------------------- B. 映射层
sample = tli.sample_mapping()
target_ids = torch.tensor([2, 3, 6], dtype=torch.int64)
draft_ids = target_ids.clone()
logits = torch.randn(3, 11)
logits_copy = logits.clone()
sample.map_target_to_draft_ids(target_ids)
sample.map_draft_to_target_ids(draft_ids)
sample.constrain_draft_logits(logits)
check("B1. `map_*` / `constrain_draft_logits` 不改写入参（需求 §4）",
      torch.equal(target_ids, torch.tensor([2, 3, 6], dtype=torch.int64))
      and torch.equal(draft_ids, target_ids) and torch.equal(logits, logits_copy))

peak = torch.full((2, 11), -5.0)
peak[:, 9] = 100.0
peak[:, 10] = 99.0
constrained = sample.constrain_draft_logits(peak)
picked = constrained.argmax(dim=-1)
check("B2. 非交集列即使 logits 最大也永远选不到（argmax 必落在交集里）",
      not bool((picked == 9).any()) and not bool((picked == 10).any())
      and bool(sample.intersection_mask_draft[picked].all()),
      f"argmax={picked.tolist()}（draft 独有列 9/10 被掩）")

hetero_target_dir, hetero_draft_dir, hetero_info = tli.tiny_hetero_pair()
from minivllm.spec_decode.vocab_mapping import load_tokenizer  # noqa: E402

hetero_target_tok = load_tokenizer(hetero_target_dir)
hetero_draft_tok = load_tokenizer(hetero_draft_dir)
hetero_map = VocabMapping(hetero_target_tok, hetero_draft_tok,
                          hetero_info["target_vocab_size"], hetero_info["draft_vocab_size"],
                          device="cpu")
prompt = torch.tensor(hetero_info["prompt_token_ids"], dtype=torch.int64)
mapped_prompt = hetero_map.map_target_to_draft_ids(prompt)
same_text = all(
    _normalize_token(hetero_target_tok.convert_ids_to_tokens(a),
                     _detect_space_prefix(hetero_target_tok))
    == _normalize_token(hetero_draft_tok.convert_ids_to_tokens(b),
                        _detect_space_prefix(hetero_draft_tok))
    for a, b in zip(prompt.tolist(), mapped_prompt.tolist()))
check("B3. **不重新分词**：id 换了但位置数与 token 文字都不变（TLI 只换标号）",
      mapped_prompt.shape == prompt.shape and not torch.equal(mapped_prompt, prompt) and same_text,
      f"{prompt.tolist()} → {mapped_prompt.tolist()}")

# ---------------------------------------------------------------- C. 上游差分
from vllm.v1.spec_decode.vocab_mapping import VocabMapping as UpstreamVocabMapping  # noqa: E402

diff_ok, diff_detail = True, []
for target_tok, draft_tok, tv, dv in (
        (target, draft, 11, 13),
        (tli.FakeTokenizer({"\u2581a": 4, "\u0120a": 9, "\u2581big": 20}, space_marker="\u2581",
                           unk_token_id=0),
         tli.FakeTokenizer({"\u0120a": 2, "\u2581a": 6, "\u0120big": 3}, space_marker="\u0120",
                           unk_token_id=0), 11, 7)):
    ours = VocabMapping(target_tok, draft_tok, tv, dv, device="cpu")
    theirs = UpstreamVocabMapping(target_tok, draft_tok, tv, dv, device="cpu")
    same = (ours.intersection_size == theirs.intersection_size
            and torch.equal(ours.draft_to_target_ids, theirs.draft_to_target_ids)
            and torch.equal(ours.target_to_draft_ids, theirs.target_to_draft_ids)
            and torch.equal(ours.intersection_mask_draft, theirs.intersection_mask_draft)
            and ours.target_unk_token_id == theirs.target_unk_token_id
            and ours.draft_unk_token_id == theirs.draft_unk_token_id)
    t_ids = torch.arange(tv, dtype=torch.int64)
    d_ids = torch.arange(dv, dtype=torch.int64)
    logits_case = torch.randn(3, dv)
    same = (same
            and torch.equal(ours.map_target_to_draft_ids(t_ids),
                            theirs.map_target_to_draft_ids(t_ids))
            and torch.equal(ours.map_draft_to_target_ids(d_ids),
                            theirs.map_draft_to_target_ids(d_ids))
            and torch.equal(ours.constrain_draft_logits(logits_case),
                            theirs.constrain_draft_logits(logits_case)))
    diff_ok &= same
    diff_detail.append(f"{ours.intersection_size} 项交集")
check("C1. 与上游真 `VocabMapping` 差分：三张量表 + 两个 map + 约束输出逐位相同",
      diff_ok, "；".join(diff_detail))

# ---------------------------------------------------------------- D. 配置边界
from minivllm import ModelConfig, SpeculativeConfig  # noqa: E402

check("D1. TLI 只支持 method='draft_model'（其它方法明确拒绝）",
      _raises(ValueError, SpeculativeConfig, method="eagle3", num_speculative_tokens=1,
              use_heterogeneous_vocab=True,
              draft_model_config=ModelConfig(model="/x", hf_config={"vocab_size": 11})))
check("D2. 概率草稿的 TLI 明确拒绝（需求 §3.5：不自行放开上游未实现的那条路）",
      _raises(ValueError, SpeculativeConfig, method="draft_model", num_speculative_tokens=1,
              use_heterogeneous_vocab=True, draft_sample_method="probabilistic",
              draft_model_config=ModelConfig(model="/x", hf_config={"vocab_size": 11})))
check("D3. `draft_sample_method` 取值受校验（greedy / probabilistic）",
      _raises(ValueError, SpeculativeConfig, method="draft_model", num_speculative_tokens=1,
              draft_sample_method="synthetic",
              draft_model_config=ModelConfig(model="/x", hf_config={"vocab_size": 11}))
      and SpeculativeConfig(method="draft_model", num_speculative_tokens=1,
                            use_heterogeneous_vocab=True,
                            draft_model_config=ModelConfig(
                                model="/x", hf_config={"vocab_size": 11})
                            ).use_heterogeneous_vocab)
check("D4. `ModelConfig.tokenizer_path` 缺省回落到模型目录（上游默认同款）",
      ModelConfig(model="/m").tokenizer_path == "/m"
      and ModelConfig(model="/m", tokenizer="/t").tokenizer_path == "/t")

# ---------------------------------------------------------------- E. 集成
engines = tli.hetero_engines.__wrapped__()
check("E1. 真实异构词表下 greedy 输出与非投机逐 token 相同",
      engines["spec"] == engines["plain"] and bool(engines["drafts"]),
      f"{len(engines['drafts'])} 轮草稿")
allowed = {index for index, value in enumerate(engines["target_to_draft"].tolist())
           if value != -1}
tokens_seen = {token for entry in engines["drafts"] for row in entry["drafts"]
               for token in row}
check("E2. 交出去的草稿是 **target 空间** 的 id，且都在交集像里（draft 独有 token 不出门）",
      bool(tokens_seen) and tokens_seen <= allowed
      and max(tokens_seen) < engines["info"]["target_vocab_size"],
      f"出现过的草稿 id {sorted(tokens_seen)}")
check("E3. TLI 草稿是 **点质量**（不带 `draft_probs`：draft 空间的 q 宽度与 target 不符）",
      all(entry["probs"] is None for entry in engines["drafts"]))

calls_ok, calls_detail = True, ""
tli_engine, _, tli_runner = tli.build_engine(
    spec=tli.hetero_spec()[3], target_dir=hetero_target_dir,
    target_hf=json.loads((Path(hetero_target_dir) / "config.json").read_text()))
calls = []
tli_mapping = tli_runner.proposer.vocab_mapping
original_map = tli_mapping.map_target_to_draft_ids


def spy(ids):
    result = original_map(ids)
    calls.append((ids.tolist(), result.tolist()))
    return result


tli_mapping.map_target_to_draft_ids = spy
try:
    outputs = tli.run(tli_engine, [("a", hetero_info["prompt_token_ids"])])
finally:
    tli_engine.shutdown()
calls_ok = (bool(outputs["a"]) and bool(calls)
            and all(token < hetero_info["target_vocab_size"] for inputs, _ in calls
                    for token in inputs)
            and any(inputs != mapped for inputs, mapped in calls))
check("E4. 反「假接线」：第一遍/自回归的 target id **真的**过了映射（入参是 target 空间、出参变了）",
      calls_ok, f"{len(calls)} 次映射调用")
check("E5. 真实 tokenizer 规模（记录用）：Qwen3-1.7B × gpt2 的交集",
      True, json.dumps(tli.VocabMapping(load_tokenizer(str(ROOT / "models" / "Qwen3-1.7B")),
                                        load_tokenizer("gpt2"), 151936, 50257,
                                        device="cpu").stats, ensure_ascii=False)
      if (ROOT / "models" / "Qwen3-1.7B").is_dir() else "本机没有 Qwen3-1.7B：待验")

# ---------------------------------------------------------------- F. 差异账本
print()
print("口径说明（完整版见 docs/step67_alignment.md §3）：")
print("  · TLI 是 **token 级交集**：只换 id、不重新分词 → 行数/位置/KV 槽位一律不变；")
print("    交集外的历史 token 填 draft unk，交集外的草稿列被 -inf 掩掉（永远选不到）。")
print("  · 与 EAGLE3 的 d2t **不是一回事**：d2t 是「同一个 tokenizer、缩小词表」的偏移表，")
print("    TLI 是「两套 tokenizer 的交集」，前者在 compute_logits 里 scatter，后者要两个方向都搬。")
print("  · 概率草稿的 TLI 上游未实现（配置期拒绝）：把 q 从 draft 空间搬到 target 空间需要"
      "在 softmax 之前做掩码+搬运，上游留了 TODO，需求 067 §3.5 要求照抄这条边界。")
print(f"设备={DEVICE}；tiny 对：target vocab={hetero_info['target_vocab_size']}、"
      f"draft vocab={hetero_info['draft_vocab_size']}、交集={hetero_map.intersection_size}")
print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
