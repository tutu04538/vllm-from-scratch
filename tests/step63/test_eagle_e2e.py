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
    engine, core, runner = make_engine(spec_k=2, budget=64)
    try:
        request = {"a": [1, 2, 3, 4, 1, 2, 3, 4]}
        engine.add_request("a", request["a"],
                           SamplingParams(max_tokens=3, temperature=0.0, eos_token_id=999))
        captured = []
        original = runner.proposer.set_inputs_first_pass

        def spy(rows, all_token_ids, target_hidden_states=None):
            plan = original(rows, all_token_ids, target_hidden_states)
            captured.append({
                "rows": list(rows),
                "tokens": list(all_token_ids["a"]),
                "plan": plan,
                "input_ids": runner.proposer.input_ids_cpu[:plan.num_tokens].tolist(),
                "positions": runner.proposer.positions_cpu[:plan.num_tokens].tolist(),
                "hidden": runner.proposer.hidden_states_cpu[:plan.num_tokens].clone(),
                "hidden_src": (runner.proposer.model.model.combine_hidden_states(
                    target_hidden_states["a"].to(runner.device)).cpu()
                    if target_hidden_states else None)})
            return plan

        runner.proposer.set_inputs_first_pass = spy
        for _ in range(60):
            if not engine.has_unfinished_requests():
                break
            engine.step()
            if any(entry["rows"][0].num_valid >= 2 for entry in captured):
                break
    finally:
        engine.shutdown()

    # 找一次"有效行 ≥ 2"的调用：只有这时才看得出"错一格"与"扩容行位置"
    entry = next(item for item in captured if item["rows"][0].num_valid >= 2)
    row = entry["rows"][0]
    tokens = entry["tokens"]
    input_ids, positions = entry["input_ids"], entry["positions"]
    # 1) token：这条请求 [start+1, start+num_valid) 是旧的，最后一格 = 新 token
    assert input_ids[:-1] == [int(t) for t in tokens[row.start + 1:row.start + row.num_valid]]
    assert input_ids[-1] == row.next_token_id
    # 2) positions：与 target 逐行相同（不被移位带走），扩容行的位置 = 最后一行的位置
    assert positions == list(range(row.start, row.start + row.num_valid)), positions
    assert entry["plan"].sample_rows[0] == row.num_valid - 1
    # 3) 特征：有效行逐行不动；扩容行 = target **采样行**的特征
    source = entry["hidden_src"]
    for index in range(row.num_valid - 1):
        assert torch.allclose(entry["hidden"][index].float(), source[index].float())
    assert torch.allclose(entry["hidden"][row.num_valid - 1].float(),
                          source[row.target_rows - 1].float())
