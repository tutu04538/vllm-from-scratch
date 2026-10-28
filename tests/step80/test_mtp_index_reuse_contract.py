"""80G：MTP top-k 索引复用的四 hook 契约与压紧差分。

不实现真实 DeepSeek indexer/MLA；用 `minivllm/testing/mtp_sparse_index.py` 的
教学 buffer 验证：
  - prefill begin 无条件复位；
  - prefill end K>1 才压紧；
  - decode begin/end 切开关；
  - 压紧行等于压紧前采样行，旧尾部不被改写；
  - 两请求/重排行与冻结 DeepSeek 的 compact 方法逐值一致。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from minivllm import (CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig,
                      SpeculativeConfig, VllmConfig)
from minivllm.config import CompilationConfig
from minivllm.testing.mtp_sparse_index import (TeachingSparseMTPInner,
                                               TeachingSparseMTPOuter,
                                               TopKIndexBuffer)
from minivllm.testing.tiny_models import tiny_mtp_dir
from minivllm.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator


def _speculator(k: int = 3) -> MTPSpeculator:
    directory = tiny_mtp_dir("tiny_gqa")
    hf = json.loads((Path(directory) / "config.json").read_text())
    target = ModelConfig(model=directory, dtype="float32", max_model_len=64,
                         hf_config=hf)
    draft_hf = {**hf, "n_predict": hf["num_nextn_predict_layers"],
                "architectures": ["Qwen3MTPModel"]}
    draft = ModelConfig(model=directory, dtype="float32", max_model_len=64,
                        hf_config=draft_hf)
    cfg = VllmConfig(
        model_config=target,
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=16),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=32),
        device_config=DeviceConfig(device="cpu"),
        speculative_config=SpeculativeConfig(
            method="mtp", num_speculative_tokens=k, draft_model_config=draft),
        compilation_config=CompilationConfig(cudagraph_mode="none"))
    speculator = MTPSpeculator(cfg, torch.device("cpu"))
    speculator.share_mtp_topk_indices = True
    return speculator


def test_hook_order_and_compaction_contract():
    speculator = _speculator(k=3)
    inner = TeachingSparseMTPInner(max_tokens=16, topk=4, num_layers=2)
    speculator.model = TeachingSparseMTPOuter(inner)
    speculator.last_token_indices[:2] = torch.tensor([2, 5])
    buffer = inner.index_buffer

    speculator.on_prefill_begin(num_reqs=2)
    assert buffer.skip_topk is False
    # step 0：真正算一次索引
    inner.compute_indices(num_tokens=6)
    assert buffer.compute_calls == 1
    before_compact = buffer.buffers[0].clone()

    speculator.on_prefill_end(num_reqs=2)
    # 压紧后的第 0/1 行 = 压紧前的第 2/5 行；尾部不变
    torch.testing.assert_close(buffer.buffers[0][0], before_compact[2])
    torch.testing.assert_close(buffer.buffers[0][1], before_compact[5])
    torch.testing.assert_close(buffer.buffers[0][2:], before_compact[2:])

    speculator.on_multi_step_decode_begin(num_reqs=2)
    assert buffer.skip_topk is True
    inner.compute_indices(num_tokens=2)
    assert buffer.compute_calls == 1, "复用步不能再算 indexer"

    speculator.on_multi_step_decode_end(num_reqs=2)
    assert buffer.skip_topk is False


def test_k1_does_not_compact_and_missing_interface_is_noop():
    speculator = _speculator(k=1)
    inner = TeachingSparseMTPInner()
    speculator.model = TeachingSparseMTPOuter(inner)
    speculator.last_token_indices[:1] = torch.tensor([3])
    before = inner.index_buffer.buffers[0].clone()
    speculator.on_prefill_begin(1)
    speculator.on_prefill_end(1)
    torch.testing.assert_close(inner.index_buffer.buffers[0], before)

    # 关闭复用后，四个 hook 一个模型方法都不该碰。
    speculator.share_mtp_topk_indices = False
    inner.set_skip_topk = None
    inner.compact_topk_indices = None
    speculator.on_prefill_begin(1)
    speculator.on_prefill_end(1)
    speculator.on_multi_step_decode_begin(1)
    speculator.on_multi_step_decode_end(1)


def test_prefill_begin_clears_stale_skip_state():
    speculator = _speculator(k=3)
    inner = TeachingSparseMTPInner()
    speculator.model = TeachingSparseMTPOuter(inner)
    inner.set_skip_topk(True)  # 模拟上一轮死在 decode 中间
    speculator.on_prefill_begin(1)
    assert inner.index_buffer.skip_topk is False


def test_two_requests_reordered_rows_match_frozen_deepseek():
    from vllm.model_executor.models.deepseek_mtp import DeepSeekMultiTokenPredictor

    def run(implementation):
        buffer = torch.arange(6 * 4, dtype=torch.int32).reshape(6, 4)
        if implementation == "ours":
            cache = TopKIndexBuffer(6, 4)
            cache.buffers[0].copy_(buffer)
            cache.compact_topk_indices(torch.tensor([5, 2]))
            return cache.buffers[0]
        fake = SimpleNamespace(layers={
            "0": SimpleNamespace(mtp_block=SimpleNamespace(
                self_attn=SimpleNamespace(mla_attn=SimpleNamespace(
                    topk_indices_buffer=buffer))))})
        DeepSeekMultiTokenPredictor.compact_topk_indices(
            fake, torch.tensor([5, 2]))
        return buffer

    ours = run("ours")
    upstream = run("upstream")
    torch.testing.assert_close(ours, upstream)


def test_compact_out_of_range_is_rejected():
    cache = TopKIndexBuffer(8, 4)
    with pytest.raises(IndexError):
        cache.compact_topk_indices(torch.tensor([8]))


def test_interface_detection_requires_config_and_both_methods():
    from types import SimpleNamespace

    outer = TeachingSparseMTPOuter()
    assert MTPSpeculator._supports_index_sharing(
        {"index_share_for_mtp_iteration": True}, outer)
    assert not MTPSpeculator._supports_index_sharing(
        {"index_share_for_mtp_iteration": False}, outer)
    assert not MTPSpeculator._supports_index_sharing({}, outer)

    no_compact = SimpleNamespace(model=SimpleNamespace(
        set_skip_topk=lambda flag: None))
    assert not MTPSpeculator._supports_index_sharing(
        {"index_share_for_mtp_iteration": True}, no_compact)

    no_inner = SimpleNamespace()
    assert not MTPSpeculator._supports_index_sharing(
        {"index_share_for_mtp_iteration": True}, no_inner)
