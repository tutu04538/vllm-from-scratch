"""88：vLLM worker 快照只读、按 storage 去重（CPU 可跑）。"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from benchmarks.step88_runtime import reset_peak_worker, worker_snapshot


class _Runner:
    def __init__(self, tensor):
        self.kv_caches = {"layer0": tensor, "layer1": tensor}
        self.kv_cache_config = SimpleNamespace(
            num_gpu_blocks=4, block_size=16, kv_cache_groups=[])
        self.cudagraph_manager = None
        self.speculator = None
        self.attn_groups = []


class _Worker:
    def __init__(self, runner):
        self.model_runner = runner


def test_snapshot_deduplicates_shared_kv_storage(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    tensor = torch.zeros(2, 4, 16, 1, 256, dtype=torch.bfloat16)
    snapshot = worker_snapshot(_Worker(_Runner(tensor)))
    assert snapshot["kv"]["bytes"] == tensor.numel() * tensor.element_size()
    assert snapshot["kv"]["unique_storages"] == 1
    assert snapshot["kv"]["kv_tensors"] == 2
    assert snapshot["kv"]["num_gpu_blocks"] == 4
    assert snapshot["cuda"] == {}
    assert reset_peak_worker(_Worker(_Runner(tensor))) is False
