"""63 关验收脚本：EAGLE 第一遍输入与端到端（**只测生产路径**）。

历史上这里测的是两个"参考实现"（`eagle_first_pass_input_ids` / `expand_eagle_inputs_shifted`）；
按用户要求删掉它们之后，本脚本改为**直接驱动 tiny 引擎**，对着提议者写进工作区的内容做检查
（口径与 `tests/step63/test_eagle_e2e.py` 一致，但脚本式 PASS/FAIL，供回归用）：

  A. 上游不扩容分支的四行：行数 = 本轮 target 行数、整体左移一格、每请求最后一格换新 token
  B. positions 与特征**逐行原样**（第 i 行配第 i 行的特征）
  C. 采样行 = 每请求最后一行
  D. "被拒位置下一轮必被重算"（`next_start == start + num_valid`）——喂被拒草稿无害的前提
  E. greedy 端到端：投机输出 == 非投机输出，且真的提了草稿
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step63"))

import test_eagle_e2e as e2e  # noqa: E402  只借夹具（起引擎/跑请求）

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def collect_rounds(spec_k=3, max_tokens=10):
    """跑一轮生成，把提议者每次第一遍的入参/结果记下来。"""
    engine, core, runner = e2e.make_engine(spec_k=spec_k, max_num_seqs=1)
    rounds = []
    original = runner.proposer.set_inputs_first_pass

    def spy(rows, all_token_ids, target_hidden_states=None, target_token_ids=None,
            target_positions=None):
        plan = original(rows, all_token_ids, target_hidden_states, target_token_ids,
                        target_positions)
        row = rows[0] if len(rows) == 1 else None
        if row is not None:
            combined = runner.proposer.model.model.combine_hidden_states(
                target_hidden_states[row.req_id].to(runner.device)).cpu()
            rounds.append({
                "row": row, "plan": plan,
                "input_ids": runner.proposer.input_ids_cpu[:plan.num_tokens].tolist(),
                "positions": runner.proposer.positions_cpu[:plan.num_tokens].tolist(),
                "hidden": runner.proposer.hidden_states_cpu[:plan.num_tokens].clone(),
                "round_tokens": [int(t) for t in target_token_ids[:row.target_rows]],
                "round_positions": [int(p) for p in target_positions[:row.target_rows]],
                "combined": combined})
        return plan

    runner.proposer.set_inputs_first_pass = spy
    outputs = {}
    try:
        engine.add_request("a", [1, 2, 3, 4, 1, 2, 3, 4],
                           e2e.SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                              eos_token_id=999))
        stats = []
        for _ in range(200):
            if not engine.has_unfinished_requests():
                break
            for out in engine.step():
                outputs[out.request_id] = list(out.token_ids)
            stats.append(core.scheduler.spec_decoding_stats)
    finally:
        engine.shutdown()
    return rounds, outputs, stats


rounds, outputs, stats = collect_rounds()
shifted_ok = positions_ok = hidden_ok = sample_ok = True
for entry in rounds:
    row = entry["row"]
    if len(entry["input_ids"]) != row.target_rows:
        shifted_ok = False
    elif entry["input_ids"][:-1] != entry["round_tokens"][1:]:
        shifted_ok = False
    elif entry["input_ids"][-1] != row.next_token_id:
        shifted_ok = False
    if entry["positions"] != entry["round_positions"]:
        positions_ok = False
    if not all(torch.allclose(entry["hidden"][i].float(), entry["combined"][i].float())
               for i in range(row.target_rows)):
        hidden_ok = False
    if entry["plan"].sample_rows[0] != row.target_rows - 1:
        sample_ok = False

check("A. 行数 = 本轮 target 行数；整体左移一格；每请求最后一格 = 新 token",
      shifted_ok, f"{len(rounds)} 轮")
check("B1. positions 与 target 逐行相同（不被移位带走）", positions_ok)
check("B2. 特征逐行原样（第 i 行配第 i 行的特征）", hidden_ok)
check("C. 采样行 = 每请求最后一行", sample_ok)

recompute_ok = all(next_entry["row"].start == entry["row"].start + entry["row"].num_valid
                   for entry, next_entry in zip(rounds, rounds[1:]))
check("D. 被拒位置下一轮必被重算（next_start == start + num_valid）",
      recompute_ok and any(entry["row"].num_rejected > 0 for entry in rounds))

rejected_seen = sum(1 for entry in rounds if entry["row"].num_rejected > 0)
check("D2. 本轮确实出现过被拒行（否则上面那条没有区分力）", rejected_seen > 0,
      f"{rejected_seen}/{len(rounds)} 轮")

# 端到端：与非投机逐 token 一致
base_engine, base_core, _ = e2e.make_engine(with_spec=False)
try:
    baseline, _ = e2e.run(base_engine, base_core, e2e.PROMPTS)
finally:
    base_engine.shutdown()
engine, core, runner = e2e.make_engine(spec_k=2)
try:
    spec_outputs, spec_stats = e2e.run(engine, core, e2e.PROMPTS)
finally:
    engine.shutdown()
hits = [s for s in spec_stats if s is not None]
check("E1. greedy：EAGLE3 投机输出 == 非投机输出", spec_outputs == baseline, f"{spec_outputs}")
check("E2. 真的提了草稿", sum(s.num_draft_tokens for s in hits) > 0,
      f"drafts={sum(s.num_drafts for s in hits)} tokens={sum(s.num_draft_tokens for s in hits)}")

print()
print(f"设备={DEVICE}")
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
