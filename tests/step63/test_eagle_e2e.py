"""step63：EAGLE3 端到端（需求 §4）——tiny target + tiny EAGLE3 draft，greedy 对照。

验收口径与前面几关一致：**草稿只是猜测**，greedy 下"开 EAGLE3"与"不开投机"的输出必须逐 token 相同；
同时钉住 63 关特有的三件事：
  1. target 真的按辅助层编号输出了特征（`runner.eagle_aux_hidden_state_layers`）；
  2. 提议者确实是 `EagleProposer`（`pass_hidden_states_to_model=True`），且**真的提了草稿**；
  3. 第一遍输入遵守 EAGLE 对齐：token 逐请求错一格、扩容行 = 采样行的位置与特征。
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,  # noqa: E402
                      SchedulerConfig, SpeculativeConfig, UniProcExecutor, VllmConfig, Worker)
from minivllm.spec_decode.eagle import EagleProposer  # noqa: E402
from minivllm.testing.tiny_models import (tiny_eagle3_dir, tiny_qwen3_config,  # noqa: E402
                                          tiny_qwen3_dir)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUX_LAYERS = (0, 1)
PROMPTS = [("a", [1, 2, 3, 4, 1, 2, 3, 4]), ("b", [5, 6, 5, 6])]


def make_engine(*, spec_k=2, with_spec=True, max_num_seqs=2, budget=64, max_model_len=64,
                proposer=None):
    target_dir = tiny_qwen3_dir("tiny_gqa")
    target_hf = tiny_qwen3_config("tiny_gqa")
    spec = None
    if with_spec:
        draft_dir = tiny_eagle3_dir("tiny_gqa", num_aux_layers=len(AUX_LAYERS),
                                    aux_layers=AUX_LAYERS)
        import json

        draft_hf = json.loads((Path(draft_dir) / "config.json").read_text())
        spec = SpeculativeConfig(
            method="eagle3", num_speculative_tokens=spec_k,
            draft_model_config=ModelConfig(model=draft_dir, dtype="float32",
                                           max_model_len=max_model_len, hf_config=draft_hf))
    config = VllmConfig(
        model_config=ModelConfig(model=target_dir, dtype="float32",
                                 max_model_len=max_model_len, hf_config=target_hf),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=32),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=DEVICE), speculative_config=spec)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner


def run(engine, core, prompts, *, max_tokens=6):
    for req_id, prompt in prompts:
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


def test_eagle3_greedy_matches_non_speculative():
    """greedy：EAGLE3 投机 == 非投机（逐 token），且草稿真的走过链路。"""
    engine, core, runner = make_engine(spec_k=2)
    try:
        assert isinstance(runner.proposer, EagleProposer)
        assert runner.proposer.pass_hidden_states_to_model is True
        assert runner.eagle_aux_hidden_state_layers == AUX_LAYERS, "target 必须按 draft 配置输出辅助层"
        outputs, stats = run(engine, core, PROMPTS)
    finally:
        engine.shutdown()

    baseline_engine, baseline_core, _ = make_engine(with_spec=False)
    try:
        baseline, _ = run(baseline_engine, baseline_core, PROMPTS)
    finally:
        baseline_engine.shutdown()

    assert outputs == baseline, (outputs, baseline)
    hits = [entry for entry in stats if entry is not None]
    assert hits and sum(entry.num_drafts for entry in hits) > 0
    assert sum(entry.num_draft_tokens for entry in hits) > 0, "必须真的提了草稿"


@pytest.mark.parametrize("spec_k", [1, 2, 4])
def test_eagle3_greedy_matches_for_several_k(spec_k):
    """K=1/2/4 都要与非投机一致（K=1 不走自回归补足那段，K>1 才走）。"""
    engine, core, _ = make_engine(spec_k=spec_k)
    try:
        outputs, _ = run(engine, core, PROMPTS)
    finally:
        engine.shutdown()
    baseline_engine, baseline_core, _ = make_engine(with_spec=False)
    try:
        baseline, _ = run(baseline_engine, baseline_core, PROMPTS)
    finally:
        baseline_engine.shutdown()
    assert outputs == baseline


def test_first_pass_inputs_follow_eagle_alignment():
    """第一遍：token 逐请求错一格、扩容行位置 = 该请求最后一行的位置、特征逐行不动。

    直接看提议者写进工作区的内容（工作区是定长缓冲，只读有效前缀）。
    """
    engine, core, runner = make_engine(spec_k=3, budget=64, max_num_seqs=1)
    try:
        engine.add_request("a", [1, 2, 3, 4, 1, 2, 3, 4],
                           SamplingParams(max_tokens=8, temperature=0.0, eos_token_id=999))
        captured = []
        original = runner.proposer.set_inputs_first_pass

        def spy(rows, all_token_ids, target_hidden_states=None, target_token_ids=None,
                target_positions=None):
            plan = original(rows, all_token_ids, target_hidden_states, target_token_ids,
                            target_positions)
            captured.append({
                "rows": list(rows),
                "tokens": list(all_token_ids["a"]),
                "plan": plan,
                "input_ids": runner.proposer.input_ids_cpu[:plan.num_tokens].tolist(),
                "positions": runner.proposer.positions_cpu[:plan.num_tokens].tolist(),
                "hidden": runner.proposer.hidden_states_cpu[:plan.num_tokens].clone(),
                "hidden_src": (runner.proposer.model.model.combine_hidden_states(
                    target_hidden_states["a"].to(runner.device)).cpu()
                    if target_hidden_states else None),
                "round_tokens": [int(t) for t in target_token_ids] if target_token_ids is not None
                else None,
                "round_positions": [int(p) for p in target_positions]
                if target_positions is not None else None})
            return plan

        runner.proposer.set_inputs_first_pass = spy
        for _ in range(60):
            if not engine.has_unfinished_requests():
                break
            engine.step()
            # 挑一次**真的有被拒行**的调用：这时"最后一个有效行"与"最后一行"不是同一行，
            # 才能把 hidden 取错行的 bug 区分出来（无被拒时两者重合）
            if any(len(entry["rows"]) == 1 and entry["rows"][0].num_valid >= 2
                   and entry["rows"][0].num_rejected > 0 for entry in captured):
                break
    finally:
        engine.shutdown()

    # 找一次"有效行 ≥ 2"的调用：只有这时才看得出"错一格"与"扩容行位置"
    entry = next(item for item in captured
                 if len(item["rows"]) == 1 and item["rows"][0].num_valid >= 2
                 and item["rows"][0].num_rejected > 0)
    row = entry["rows"][0]
    tokens = entry["tokens"]
    input_ids, positions = entry["input_ids"], entry["positions"]
    # 单请求 → 工作区前 num_tokens 行都是这条请求的；**逐行照抄上游**：
    #   行数 = 本轮 target 的行数（含被拒行）；positions/特征**原样逐行**；
    #   token 整体左移一格，最后一格（每条请求的）换成新采出的 token。
    assert len(input_ids) == row.target_rows
    # 只取**这条请求的行块**：runner 交来的缓冲可能带补齐行（图路径的 padded 工作区），
    # 提议者自己按 Σ target_rows 切片，测试也要按同一口径切。
    round_tokens = entry["round_tokens"][:row.target_rows]
    bonus_row = row.target_rows - 1 - row.num_rejected
    # 整体左移一格 —— **锚点那一行除外**（它被换成新采出的 token），锚点之后仍是位移副本
    assert input_ids[:bonus_row] == round_tokens[1:1 + bonus_row], "左移一格（锚点之前）"
    assert input_ids[bonus_row + 1:] == round_tokens[bonus_row + 1:], "锚点之后仍是位移副本"
    # 新 token 打在**最后一枚有效 token 所在的那一行**（= `target_rows − 1 − num_rejected`），
    # 不是"块的最后一行"：上游 padded 通路的 `token_indices_to_sample` 就是这个索引
    # （`prepare_inputs_padded`，69 关已与上游内核逐值对过）。2026-10-08 复核修正，见
    # `docs/step63_alignment.md` §8：两个索引在没有被拒行时重合，只有 num_rejected > 0 才分得开
    # ——本用例挑的就是这种轮次。
    assert input_ids[bonus_row] == row.next_token_id, "新 token 打在最后一枚有效行上（锚点行）"
    # positions 与 target 逐行相同（不被移位带走）
    assert positions == entry["round_positions"], positions
    assert positions == list(range(row.start, row.start + row.target_rows)), positions
    # 特征逐行不动：第 i 行配第 i 行的特征（扩容行 = 最后一行的特征，天然对齐）
    source = entry["hidden_src"]
    for index in range(row.target_rows):
        assert torch.allclose(entry["hidden"][index].float(), source[index].float()), \
            f"第 {index} 行的特征必须取 target 同一行（上游就是逐行原样拷贝）"
    assert entry["plan"].sample_rows[0] == bonus_row, "采样行 = 最后一枚有效行（锚点行）"


def test_rejected_positions_are_recomputed_next_round():
    """“把本轮被拒草稿也喂进 draft”无害的**前提**：被拒位置下一轮一定会被重算。

    上游简单通路不屏蔽被拒行（照抄本轮输入），依赖的就是这条：轮次推进时
    `start` 只前进 `num_valid`（= 接受数 + 1 个纠正/新 token），所以上一轮写在被拒位置上的
    KV 下一轮会被 target 与 draft 一起覆盖——它没有机会被当成"正确上下文"读下去。
    实测（K=3，max_num_seqs=1）：轮1 `start=8, rows@[8,9,10,11], valid=1` → 轮2 `start=9, rows@[9,..]`。

    如果哪天这条不变量被破坏（比如 prefix 发布边界或抢占恢复让 `start` 跳过被拒位置），
    "喂被拒 token"就会从无害变成真错——所以在这里钉住。
    """
    engine, core, runner = make_engine(spec_k=3, max_num_seqs=1)
    seen = []
    original = runner.proposer.set_inputs_first_pass

    def spy(rows, all_token_ids, target_hidden_states=None, target_token_ids=None,
            target_positions=None):
        plan = original(rows, all_token_ids, target_hidden_states, target_token_ids,
                        target_positions)
        for target in rows:
            seen.append((target.req_id, target.start, target.num_valid, target.num_rejected))
        return plan

    runner.proposer.set_inputs_first_pass = spy
    try:
        engine.add_request("a", [1, 2, 3, 4, 1, 2, 3, 4],
                           SamplingParams(max_tokens=10, temperature=0.0, eos_token_id=999))
        for _ in range(200):
            if not engine.has_unfinished_requests():
                break
            engine.step()
    finally:
        engine.shutdown()

    rounds = [entry for entry in seen if entry[0] == "a"]
    assert len(rounds) >= 3, rounds
    assert any(num_rejected > 0 for _, _, _, num_rejected in rounds), "本例必须真的出现被拒"
    for (_, start, num_valid, _), (_, next_start, _, _) in zip(rounds, rounds[1:]):
        assert next_start == start + num_valid, (
            f"轮次推进必须只前进 num_valid（{start} + {num_valid} != {next_start}）："
            f"否则被拒位置上写的 KV 会被当成已计算上下文读下去")
