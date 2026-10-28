"""81-C：已声明支持边界与交给 82 的 manifest 模板。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from minivllm.config import (CacheConfig, SchedulerConfig,
                             resolve_async_scheduling)
from minivllm.core.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                              KVCacheGroup, SlidingWindowSpec)
from spec81_helpers import make_v1_engine

ROOT = Path(__file__).resolve().parents[2]


def test_async_scheduling_true_is_rejected_not_silently_downgraded():
    cfg = SimpleNamespace(
        scheduler_config=SchedulerConfig(async_scheduling=True),
        speculative_config=None,
    )
    with pytest.raises(NotImplementedError, match="async_scheduling=True"):
        resolve_async_scheduling(cfg, executor_supports_async=True)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="V1 Runner 初始化需要 CUDA")
def test_execution_rejects_different_group_block_sizes():
    engine, core = make_v1_engine(spec_k=None, blocks=16)
    runner = core.model_executor.driver_worker.model_runner
    try:
        cfg = KVCacheConfig(
            num_gpu_blocks=16,
            kv_cache_groups=[
                KVCacheGroup(FullAttentionSpec(4, 1, 8, torch.float32), ["a"]),
                KVCacheGroup(SlidingWindowSpec(8, 1, 8, torch.float32,
                                               sliding_window=8), ["b"]),
            ],
        )
        with pytest.raises(NotImplementedError, match="block_size"):
            runner.initialize_kv_cache(cfg)
    finally:
        engine.shutdown()


def test_manifest_template_is_parseable_and_marks_unsupported():
    data = json.loads((ROOT / "docs/step81_experiment_manifest.json").read_text())
    assert data["runtime"]["runner"] == "v2"
    assert data["runtime"]["scheduling"] == "sync"
    assert data["runtime"]["attention_backend"] == "torch"
    assert data["model"]["text_only"] is True
    assert "async_scheduling" in data["unsupported_in_this_round"]
    assert "pd_kv_transfer" in data["unsupported_in_this_round"]
    # 模板只列已下载的固定 revision；不把 main 当实验输入。
    assert data["model"]["target_revision"] != "main"
    assert data["model"]["assistant_revision"] != "main"
