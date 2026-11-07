"""Capacity negotiation and frozen upstream differential."""
import hashlib
from pathlib import Path
from types import SimpleNamespace as N

import pytest
import torch
from minivllm.core.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroup
from minivllm.core.kv_cache_utils import get_kv_cache_configs, generate_scheduler_kv_cache_config


def config():
    spec = FullAttentionSpec(16, 1, 16, torch.float32)
    return KVCacheConfig(8, [KVCacheGroup(spec, ['l0', 'l1'])])


def test_frozen_vllm_capacity_and_scheduler():
    import vllm.v1.core.kv_cache_utils as upstream
    from vllm.v1.kv_cache_interface import FullAttentionSpec as UpSpec
    root = Path(__file__).resolve().parents[2]
    frozen = root / 'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/vllm_0_28_0/v1/core/kv_cache_utils.py'
    assert hashlib.sha256(Path(upstream.__file__).read_bytes()).digest() == hashlib.sha256(frozen.read_bytes()).digest()
    us = UpSpec(16, num_kv_heads=1, head_size=16, dtype=torch.float32)
    uc = N(scheduler_config=N(disable_hybrid_kv_cache_manager=False), speculative_config=None,
           model_config=N(original_max_model_len=64, max_model_len=64),
           cache_config=N(num_gpu_blocks_override=None), kv_transfer_config=None,
           parallel_config=N(decode_context_parallel_size=1, prefill_context_parallel_size=1))
    budgets = [us.page_size_bytes*2*n for n in (8, 7, 6, 8)]
    expected = upstream.get_kv_cache_configs(uc, [{'l0':us,'l1':us} for _ in budgets], budgets)
    actual = get_kv_cache_configs([config() for _ in budgets], budgets, 64)
    assert [c.num_blocks for c in expected] == [c.num_gpu_blocks for c in actual] == [6]*4
    assert upstream.generate_scheduler_kv_cache_config(expected).num_blocks == generate_scheduler_kv_cache_config(actual).num_gpu_blocks
    assert all(t.size == us.page_size_bytes*6 for c in expected for t in c.kv_cache_tensors)


def test_capacity_boundaries():
    c=config(); byte=c.kv_cache_groups[0].kv_cache_spec.page_size_bytes*2
    with pytest.raises(ValueError, match='rank 1'):
        get_kv_cache_configs([c,c], [8*byte,3*byte],64)
    with pytest.raises(ValueError, match='所需的 5 块'):
        get_kv_cache_configs([c], [4*byte],64, reserve_null_block=True)
    assert get_kv_cache_configs([c], [100*byte],64)[0].num_gpu_blocks == 8
    other=config();other.kv_cache_groups[0].layer_names.append('different')
    with pytest.raises(ValueError, match='规格'):
        get_kv_cache_configs([c,other], [8*byte,8*byte],64)


def test_qwen3_8b_tp4_page_bytes_and_replication():
    page = FullAttentionSpec(16, 2, 128, torch.bfloat16)
    assert page.page_size_bytes == 16384
    # KV heads=2, TP4: each rank holds one of two heads, with two copies.
    replicated = FullAttentionSpec(16, 1, 128, torch.bfloat16)
    assert replicated.page_size_bytes == 8192
    assert replicated.page_size_bytes * 4 == page.page_size_bytes * 2
