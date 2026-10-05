"""64 关验收脚本：`extract_hidden_states` 的 cache-only 执行路径（需求 064 §4）。

脚本式 PASS/FAIL（供回归用），口径与 `tests/step64/test_hidden_cache.py` 一致，但**全部走生产
路径**（tiny 引擎真跑）：不 import 上游的任何东西，也不自己造一套"参考缓存"。

检查项：

  A. 配置与缓存形状：L 当 head 数、H 当 head_size、层名带 target 层数
  B. 两请求 + 两辅助层：读**物理 slot** 验证层/请求/位置没有互换（含独立特征参考）
  C. 特殊协议：`sampled_token_ids[:, :1]`（宽度 >1 也只返回第 0 列）
  D. 拒绝尾部：被拒行下一轮重算 → 同一槽位被覆盖，最终内容 = 最后一次写入
  E. chunked prefill：分块算的 prompt，每个位置的特征都在（对照另一个引擎的一次性前向）
  F. prefix 命中：命中段不重算，但特征已经在复用的块里
  G. 请求释放/复用：新请求的槽位装的是它自己的特征
  H. 端到端 greedy：与非投机逐 token 相同，且草稿真的进过调度器
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step64"))

import test_hidden_cache as hc  # noqa: E402  只借夹具（起引擎/跑请求/读槽位/独立参考）

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FAIL = []


def _raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    return False


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


# ---------------------------------------------------------------- A. 配置与形状
engine, core, runner = hc.make_engine()
try:
    proposer = runner.proposer
    layer = proposer.model.cache_only_layers["2"]
    cache = hc.cache_of(runner)
    check("A1. 提议者是 ExtractHiddenStatesProposer，target 开了辅助层采集",
          type(proposer).__name__ == "ExtractHiddenStatesProposer"
          and runner.capture_aux_hidden_states
          and runner.eagle_aux_hidden_state_layers == tuple(hc.AUX_LAYERS))
    check("A2. 缓存形状 = [blocks, block_size, L, H]（L 当 head 数、H 当 head_size）",
          tuple(cache.shape) == (32, 4, len(hc.AUX_LAYERS), 32)
          and (layer.num_heads, layer.head_size) == (len(hc.AUX_LAYERS), 32),
          f"shape={tuple(cache.shape)} L={layer.num_heads} H={layer.head_size}")
    check("A3. 层名带 target 的层数（cache_only_layers.2）",
          proposer.attn_layer_names == ["cache_only_layers.2"])
    check("A4. 配置期校验：K != 1 或缺辅助层编号都当场报错",
          _raises(ValueError, hc.SpeculativeConfig, method="extract_hidden_states",
                  num_speculative_tokens=2,
                  draft_model_config=hc.ModelConfig(
                      model="/tmp",
                      hf_config={"eagle_aux_hidden_state_layer_ids": hc.AUX_LAYERS}))
          and _raises(ValueError, hc.SpeculativeConfig,
                      method="extract_hidden_states", num_speculative_tokens=1))
finally:
    engine.shutdown()

# ---------------------------------------------------------------- B/C/D/E/F/G/H
rounds_engine, rounds_core, rounds_runner = hc.make_engine()
calls = hc.install_spy(rounds_runner)
round_spy = hc.install_round_spy(rounds_runner)
try:
    hc.run(rounds_engine, rounds_core, hc.PROMPTS, max_tokens=4)
    first = calls[0]
    stacked, slots = first["stacked"], first["slot_mapping"]
    same_source = torch.equal(
        hc.reference_aux(rounds_runner, first["metadata"],
                         [token for _, prompt in hc.PROMPTS for token in prompt],
                         [position for _, prompt in hc.PROMPTS
                          for position in range(len(prompt))]),
        stacked)
    written_ok = all(torch.equal(hc.read_slot(rounds_runner, int(slots[index])), stacked[index])
                     for index in range(stacked.shape[0]))
    distinguishable = (not torch.equal(stacked[:, 0], stacked[:, 1])
                       and not torch.equal(stacked[0], stacked[8]))
    slot_formula_ok = [entry[2] for entry in first["positions"]] == slots.tolist()
    check("B1. 每一行特征都写到了它的物理槽位（直接读缓存）", written_ok,
          f"{stacked.shape[0]} 行")
    check("B2. 提议者拿到/写下的就是本轮 target 算出的特征（独立重跑对照）", same_source)
    check("B3. 有区分力：两层不同、两条请求不同（互换看得出来）", distinguishable)
    check("B4. 位置/请求 → 槽位（从元数据独立推出的三元组与写入顺序一致）", slot_formula_ok)

    protocol_ok = all(torch.equal(call["returned"], call["sampled"][:, :1])
                      and call["returned"].shape[1] == 1 for call in calls)
    check("C1. 返回的就是 sampled 的第 0 列（宽度 >1 也只取一列）", protocol_ok,
          f"{len(calls)} 轮")

    write_counts = {}
    for call in calls:
        for slot in call["slot_mapping"].tolist():
            write_counts[slot] = write_counts.get(slot, 0) + 1
    last_write = {}
    for call in calls:
        for index, slot in enumerate(call["slot_mapping"].tolist()):
            last_write[slot] = call["stacked"][index]
    overwritten = max(write_counts.values()) >= 2
    final_ok = all(torch.equal(hc.read_slot(rounds_runner, slot), expected)
                   for slot, expected in last_write.items())
    check("D1. 拒绝尾部被下一轮重算覆盖（同一槽位被重写）", overwritten,
          f"最多写了 {max(write_counts.values())} 次")
    check("D2. 槽位的最终内容 = 最后一次写入（无残留污染）", final_ok)
finally:
    rounds_engine.shutdown()

# E. chunked prefill：分块写 vs 另一个引擎一次写完
prompt = [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 5, 6]
chunk_engine, chunk_core, chunk_runner = hc.make_engine(budget=8, max_num_seqs=1)
chunk_rounds = hc.install_round_spy(chunk_runner)
try:
    hc.add_request(chunk_engine, "a", prompt, max_tokens=8)
    for _ in range(4):
        chunk_engine.step()
        if len(chunk_rounds) >= 2:
            break
    chunked = {position: hc.read_slot(chunk_runner,
                                      hc.slot_of(chunk_runner, "a", position)).clone()
               for position in range(len(prompt))}
    rounds_seen = len(chunk_rounds)
finally:
    chunk_engine.shutdown()
ref_engine, ref_core, ref_runner = hc.make_engine(budget=64, max_num_seqs=1)
ref_calls = hc.install_spy(ref_runner)
try:
    hc.add_request(ref_engine, "a", prompt, max_tokens=8)
    ref_engine.step()
    reference = ref_calls[0]["stacked"]
finally:
    ref_engine.shutdown()
chunk_ok = all(torch.allclose(chunked[position], reference[position], atol=1e-6, rtol=1e-5)
               for position in range(len(prompt)))
check("E1. chunked prefill：每个位置的特征都在、且与一次性前向一致", chunk_ok,
      f"{rounds_seen} 块算完 {len(prompt)} 个位置")

# F. prefix 命中
prefix_engine, prefix_core, prefix_runner = hc.make_engine(prefix_caching=True, max_num_seqs=1)
try:
    hc.run(prefix_engine, prefix_core, [hc.PROMPTS[0]], max_tokens=1)
    prefix_calls = hc.install_spy(prefix_runner)
    prefix_rounds = hc.install_round_spy(prefix_runner)
    prompt_b = hc.PROMPTS[0][1] + [7, 8, 9, 10]
    hc.add_request(prefix_engine, "b", prompt_b, max_tokens=8)
    prefix_engine.step()
    start = prefix_rounds[0]["starts"].get("b")
    scheduled = prefix_rounds[0]["num_scheduled"]["b"]
    hit_nonempty = all(int(torch.count_nonzero(hc.read_slot(
        prefix_runner, hc.slot_of(prefix_runner, "b", position)))) > 0
        for position in range(len(hc.PROMPTS[0][1])))
    reference_b = hc.reference_aux(prefix_runner, prefix_calls[0]["metadata"],
                                   list(prompt_b[8:]), list(range(8, len(prompt_b))))
    new_ok = all(torch.equal(hc.read_slot(prefix_runner, hc.slot_of(prefix_runner, "b", pos)),
                             reference_b[index])
                 for index, pos in enumerate(range(8, len(prompt_b))))
    check("F1. prefix 命中：命中段不重算（起点 = 命中长度、本轮只排新 token）",
          start == 8 and scheduled == len(prompt_b) - 8, f"start={start} 本轮排 {scheduled}")
    check("F2. 命中段的特征已经在复用的块里（非空）", hit_nonempty)
    check("F3. 本轮新算的那段写对了（独立参考）", new_ok)
finally:
    prefix_engine.shutdown()

# G. 释放 + 复用
reuse_engine, reuse_core, reuse_runner = hc.make_engine(budget=64, max_num_seqs=1)
try:
    hc.run(reuse_engine, reuse_core, [hc.PROMPTS[0]], max_tokens=4)
    reuse_calls = hc.install_spy(reuse_runner)
    hc.run(reuse_engine, reuse_core, [("c", [3, 1, 4, 1, 5])], max_tokens=2)
    reuse_ok = reuse_calls and all(
        torch.equal(hc.read_slot(reuse_runner, int(slot)), call["stacked"][index])
        for call in reuse_calls
        for index, slot in enumerate(call["slot_mapping"].tolist()))
    check("G1. 请求释放、块复用后：槽位装的是新请求的特征", reuse_ok,
          f"{sum(len(c['slot_mapping']) for c in reuse_calls)} 行")
finally:
    reuse_engine.shutdown()

# H. 端到端 greedy 对照
spec_engine, spec_core, spec_runner = hc.make_engine()
try:
    spec_outputs, spec_stats = hc.run(spec_engine, spec_core, hc.PROMPTS)
finally:
    spec_engine.shutdown()
base_engine, base_core, _ = hc.make_engine(with_spec=False)
try:
    base_outputs, _ = hc.run(base_engine, base_core, hc.PROMPTS)
finally:
    base_engine.shutdown()
check("H1. 端到端 greedy：extract 投机 == 非投机（逐 token）", spec_outputs == base_outputs,
      f"{spec_outputs}")
check("H2. 草稿真的进过调度器",
      bool(spec_stats) and sum(s.num_draft_tokens for s in spec_stats) > 0,
      f"drafts={sum(s.num_drafts for s in spec_stats)} tokens="
      f"{sum(s.num_draft_tokens for s in spec_stats)}")

print()
print(f"设备={DEVICE}")
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
