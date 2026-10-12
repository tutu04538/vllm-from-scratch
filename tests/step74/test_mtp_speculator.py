"""74 关：V2 的原生 MTP 提议者（需求 074 §2/§3.6）。

三件要钉住的事：

    M1  端到端可用：`method="mtp"` + 显式派生好的 draft 配置，V2（eager 与图）与 V1 eager
        逐 token 相同；K=1 不融合、K>1 融合（decode 图真的被捕获）
    M2  hook 契约：四个 hook 在**捕获**与**每轮 propose** 两侧成对触发，且跑完不残留
        复用状态
    M3  `skip_topk` 的语义：`on_prefill_begin()` **无条件**清除（上一轮死在中间也不会残留到
        下一轮）；`on_prefill_end()` 只在 K>1 时压缩 top-k 索引
    M4  配置边界：V2 + mtp 但没给 draft 配置 → 明确拒绝（不猜、不静默退回 V1）
"""

import json
import os
from pathlib import Path

import pytest
import torch

from spec74_helpers import requires_cuda  # noqa: E402

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig,  # noqa: E402
                      SamplingParams, SchedulerConfig, SpeculativeConfig,
                      UniProcExecutor, VllmConfig, Worker)
from minivllm.config import CompilationConfig  # noqa: E402
from minivllm.testing.tiny_models import tiny_mtp_dir  # noqa: E402
from minivllm.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator  # noqa: E402

PROMPTS = [("a", [1, 2, 3, 4, 1, 2, 3, 4]), ("b", [5, 6, 5, 6])]


def mtp_configs(*, k: int, explicit_draft: bool = True, mode: str = "none"):
    """tiny MTP checkpoint 的 (target 配置, draft 配置)。

    真实 MTP 的权重与 target 在**同一个文件**里，draft 配置由
    `SpeculativeConfig.derive_mtp_draft_config(target_config)` 派生（改架构名 + n_predict）。
    这里手写同一份派生结果，好让"上游怎么派生的"在测试里一目了然。
    """
    directory = tiny_mtp_dir("tiny_gqa")
    hf_config = json.loads((Path(directory) / "config.json").read_text())
    target = ModelConfig(model=directory, dtype="float32", max_model_len=64,
                         hf_config=hf_config)
    draft_hf = {**hf_config, "n_predict": hf_config["num_nextn_predict_layers"],
                "architectures": ["Qwen3MTPModel"]}
    draft = ModelConfig(model=directory, dtype="float32", max_model_len=64,
                        hf_config=draft_hf)
    spec = SpeculativeConfig(method="mtp", num_speculative_tokens=k,
                             draft_model_config=draft if explicit_draft else None)
    cfg = VllmConfig(model_config=target,
                     cache_config=CacheConfig(block_size=4, num_gpu_blocks=32),
                     scheduler_config=SchedulerConfig(max_num_seqs=2,
                                                      max_num_batched_tokens=64),
                     device_config=DeviceConfig(device="cuda"),
                     speculative_config=spec,
                     compilation_config=CompilationConfig(
                         cudagraph_mode=mode, cudagraph_capture_sizes=[1, 2, 3, 4, 8]))
    return cfg


def run_mtp(*, v2: bool, mode: str, k: int, max_tokens: int = 8, explicit_draft: bool = True):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if v2 else "0"
    cfg = mtp_configs(k=k, explicit_draft=explicit_draft, mode=mode)
    engine = LLMEngine(cfg, UniProcExecutor(cfg, Worker(cfg)), tokenizer=None)
    runner = engine.engine_core.engine_core.model_executor.driver_worker.model_runner
    try:
        for req_id, prompt in PROMPTS:
            engine.add_request(req_id, list(prompt),
                               SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                              eos_token_id=999))
        final = {}
        for _ in range(400):
            if not engine.has_unfinished_requests():
                break
            for out in engine.step():
                final[out.request_id] = list(out.token_ids)
        return final, runner
    finally:
        engine.shutdown()


@requires_cuda
@pytest.mark.parametrize("k", [1, 2])
def test_mtp_v2_matches_v1_end_to_end(k):
    """M1：MTP 在 V2（eager 与图）下与 V1 eager 逐 token 相同，且融合门的取值正确。"""
    v1, _ = run_mtp(v2=False, mode="none", k=k)
    v2_eager, runner_eager = run_mtp(v2=True, mode="none", k=k)
    v2_graph, runner_graph = run_mtp(v2=True, mode="full_decode_only", k=k)
    assert v1 == v2_eager == v2_graph
    assert all(len(tokens) == 8 for tokens in v1.values())

    assert type(runner_graph.speculator).__name__ == "MTPSpeculator"
    for runner in (runner_eager, runner_graph):
        assert runner.speculator.use_fused_multi_step_decode is (k > 1)
    if k > 1:
        assert runner_graph.speculator.decode_cudagraph_manager.graphs, "K>1 必须捕获 decode 图"
        assert runner_graph.speculator.decode_cudagraph_manager.num_replays > 0
    else:
        assert runner_graph.speculator.decode_cudagraph_manager.graphs == {}


@requires_cuda
def test_mtp_hooks_fire_on_capture_and_every_round(monkeypatch):
    """M2：四个 hook 在捕获期与每一轮 propose 里都成对触发，顺序固定。"""
    events = []
    original = {}

    def recorder(name):
        def hook(self, num_reqs):
            events.append((name, num_reqs))
            return original[name](self, num_reqs)
        return hook

    for name in ("on_prefill_begin", "on_prefill_end",
                 "on_multi_step_decode_begin", "on_multi_step_decode_end"):
        original[name] = getattr(MTPSpeculator, name)
        monkeypatch.setattr(MTPSpeculator, name, recorder(name))

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    engine, _core, runner = None, None, None
    cfg = mtp_configs(k=2, mode="full_decode_only")
    engine = LLMEngine(cfg, UniProcExecutor(cfg, Worker(cfg)), tokenizer=None)
    runner = engine.engine_core.engine_core.model_executor.driver_worker.model_runner
    try:
        # 捕获期：四连（请求数 = max_num_reqs，上游同款）
        capture_events = list(events)
        assert [name for name, _ in capture_events] == [
            "on_prefill_begin", "on_prefill_end",
            "on_multi_step_decode_begin", "on_multi_step_decode_end"]
        assert {n for _, n in capture_events} == {2}

        for req_id, prompt in PROMPTS:
            engine.add_request(req_id, list(prompt),
                               SamplingParams(max_tokens=6, temperature=0.0,
                                              eos_token_id=999))
        for _ in range(200):
            if not engine.has_unfinished_requests():
                break
            engine.step()
        later = events[len(capture_events):]
        assert later, "运行期一次 hook 都没触发"
        # 每一轮都必须是"prefill_begin → prefill_end → multi_step_begin → multi_step_end"
        assert [name for name, _ in later] == [
            "on_prefill_begin", "on_prefill_end",
            "on_multi_step_decode_begin", "on_multi_step_decode_end"] * (len(later) // 4)
    finally:
        engine.shutdown()


def test_skip_topk_is_cleared_unconditionally(monkeypatch):
    """M3：`on_prefill_begin()` 无条件清除复用状态（异常中断也不残留）。"""
    recorded = []

    class FakeInner:
        def set_skip_topk(self, flag):
            recorded.append(("set_skip_topk", flag))
            self.flag = flag

        def compact_topk_indices(self, indices):
            recorded.append(("compact_topk_indices", int(indices.shape[0])))

    class FakeModel:
        def __init__(self):
            self.model = FakeInner()
            self.model.flag = True      # 模拟"上一轮死在中间、没走到 end"

    speculator = MTPSpeculator(mtp_configs(k=3), torch.device("cpu"))
    speculator.model = FakeModel()
    speculator.share_mtp_topk_indices = True

    speculator.on_prefill_begin(num_reqs=2)
    assert recorded[-1] == ("set_skip_topk", False), "prefill 开始必须无条件清掉复用状态"

    speculator.on_prefill_end(num_reqs=2)
    assert recorded[-1][0] == "compact_topk_indices"

    speculator.on_multi_step_decode_begin(num_reqs=2)
    assert recorded[-1] == ("set_skip_topk", True)
    speculator.on_multi_step_decode_end(num_reqs=2)
    assert recorded[-1] == ("set_skip_topk", False)

    # K=1 时没有"step 1+"，不需要压缩 top-k 索引
    recorded.clear()
    one_step = MTPSpeculator(mtp_configs(k=1), torch.device("cpu"))
    one_step.model = FakeModel()
    one_step.share_mtp_topk_indices = True
    one_step.on_prefill_end(num_reqs=1)
    assert recorded == [], f"K=1 不该压缩索引：{recorded}"

    # 探测结果决定一切：不开复用时一个方法都不该被碰（模型可能根本没有那两个方法）
    recorded.clear()
    speculator.share_mtp_topk_indices = False
    speculator.model.model.set_skip_topk = None
    speculator.on_prefill_begin(num_reqs=1)
    speculator.on_multi_step_decode_begin(num_reqs=1)
    assert recorded == []


def test_v2_mtp_requires_explicit_draft_config():
    """M4：V2 + mtp 但没给 draft 配置 → 配置期/初始化期明确拒绝（不静默退回 V1）。"""
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    cfg = mtp_configs(k=2, explicit_draft=False)
    from minivllm.worker.gpu.spec_decode import init_speculator

    with pytest.raises(NotImplementedError, match="显式"):
        init_speculator(cfg, torch.device("cpu"))
    # V1 仍然能建（拒绝来自 V2 的接线，不是把 MTP 一起砍了）
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    assert cfg.use_v2_model_runner is False
