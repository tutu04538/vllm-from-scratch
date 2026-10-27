"""80F：Step3p5 V1 proposer 的 per-group block table / slot mapping / metadata。

固定的两个 group 例子：
  G0 block7，position5 -> slot 7*4+1=29
  G1 block11，position5 -> slot 11*4+1=45

不能用 G0 的 slot/metadata 广播给 G1；否则不报错，只是草稿条件在错误的 K/V 上。
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from minivllm.forward_context import get_forward_context
from minivllm.spec_decode.step3p5 import Step3p5MTPProposer


def test_proposer_group_slot_mapping_uses_each_table():
    """80F 小例子：G0 block7 -> slot29，G1 block11 -> slot45。"""
    from types import SimpleNamespace
    from minivllm.spec_decode.step3p5 import Step3p5MTPProposer

    proposer = object.__new__(Step3p5MTPProposer)
    proposer.num_kv_cache_groups = 2
    proposer.max_num_tokens = 16
    proposer.block_size = 4
    proposer.max_blocks_per_req = 4
    proposer.positions_cpu = torch.tensor([5], dtype=torch.int64)
    proposer.is_rejected_token_mask_cpu = torch.tensor([False])
    proposer._group_block_tables_cpu = {
        0: torch.tensor([[0, 7, 0, 0]]),
        1: torch.tensor([[0, 11, 0, 0]]),
    }
    proposer._group_slot_mappings_cpu = {0: torch.zeros(16, dtype=torch.int64),
                                         1: torch.zeros(16, dtype=torch.int64)}
    row = SimpleNamespace(target_rows=1)
    proposer._prepare_group_slot_mappings([row], 1)
    assert int(proposer._group_slot_mappings_cpu[0][0]) == 29
    assert int(proposer._group_slot_mappings_cpu[1][0]) == 45


def test_proposer_forward_maps_each_group_metadata_separately():
    """不能把 group0 的 metadata 广播给 group1：两层拿到不同块表。"""
    from types import SimpleNamespace
    from minivllm.forward_context import get_forward_context
    from minivllm.spec_decode.step3p5 import Step3p5MTPProposer

    captured = {}

    class Builder:
        def build(self, *, block_table, **kwargs):
            return {"table": block_table.clone(), "num_reqs": kwargs["num_reqs"]}

    class FakeModel:
        def __call__(self, input_ids, positions, hidden, spec_step_idx):
            captured.update(get_forward_context().attn_metadata)
            return hidden

    proposer = object.__new__(Step3p5MTPProposer)
    proposer.device = "cpu"
    proposer.num_kv_cache_groups = 2
    proposer._draft_layer_groups = {"draft::a": 0, "draft::b": 1}
    proposer._group_block_tables = {
        0: torch.tensor([[7, 0, 0, 0]]),
        1: torch.tensor([[11, 0, 0, 0]]),
    }
    proposer._group_slot_mappings = {
        0: torch.tensor([29]), 1: torch.tensor([45])}
    proposer._group_block_tables_cpu = proposer._group_block_tables
    proposer._group_slot_mappings_cpu = proposer._group_slot_mappings
    proposer.query_start_loc = torch.tensor([0, 1])
    proposer.seq_lens = torch.tensor([1])
    proposer.metadata_builder = Builder()
    proposer.model = FakeModel()
    proposer.input_ids = torch.tensor([1])
    proposer.positions = torch.tensor([5])
    proposer.hidden_states = torch.zeros(1, 4)
    proposer._spec_step_idx = 0
    proposer._upload = lambda *a: None
    proposer.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(sliding_window=None)),
                         SimpleNamespace(kv_cache_spec=SimpleNamespace(sliding_window=4))])
    proposer._forward(1, 1)
    assert set(captured) == {"draft::a", "draft::b"}
    assert int(captured["draft::a"]["table"][0, 0]) == 7
    assert int(captured["draft::b"]["table"][0, 0]) == 11




def test_step3p5_draft_config_and_v2_boundary():
    from minivllm import (CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig,
                          SpeculativeConfig, VllmConfig)
    from minivllm.config import CompilationConfig
    from minivllm.worker.gpu.spec_decode import init_speculator

    target_cfg = {
        "architectures": ["Step3p5ForCausalLM"], "model_type": "step3p5",
        "vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_attention_groups": 2, "head_dim": 8,
        "max_position_embeddings": 32, "rms_norm_eps": 1e-6,
        "layer_types": ["full_attention", "sliding_attention"],
        "sliding_window": 8, "num_nextn_predict_layers": 2,
        "tie_word_embeddings": False,
    }
    target = ModelConfig(model="/tmp/step3p5-not-read", dtype="float32",
                         max_model_len=32, hf_config=target_cfg)
    spec = SpeculativeConfig(method="mtp", num_speculative_tokens=2,
                             draft_model_config=None)
    draft = spec.derive_mtp_draft_config(target)
    assert draft.hf_config["architectures"] == ["Step3p5MTP"]
    assert draft.hf_config["model_type"] == "step3p5_mtp"
    assert draft.hf_config["n_predict"] == 2

    vc = VllmConfig(
        model_config=target,
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=16),
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=16),
        device_config=DeviceConfig(device="cpu"),
        speculative_config=SpeculativeConfig(
            method="mtp", num_speculative_tokens=2, draft_model_config=draft),
        compilation_config=CompilationConfig(cudagraph_mode="none"))
    speculator = init_speculator(vc, torch.device("cpu"))
    assert type(speculator).__name__ == "MultiModuleMTPSpeculator"
    # K=1 时工厂按通用 MTP 路径选 MTPSpeculator（不是 multi-module）。
    k1 = SpeculativeConfig(
        method="mtp", num_speculative_tokens=1, draft_model_config=draft)
    vc1 = VllmConfig(
        model_config=target,
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=16),
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=16),
        device_config=DeviceConfig(device="cpu"),
        speculative_config=k1,
        compilation_config=CompilationConfig(cudagraph_mode="none"))
    assert type(init_speculator(vc1, torch.device("cpu"))).__name__ == "MTPSpeculator"
