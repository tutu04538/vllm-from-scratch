"""RoPE 表在构造设备生成，转换只转换已生成的值。"""
import pytest
import torch

from minivllm.layers import get_rope


@pytest.mark.skipif(not torch.cuda.is_available(), reason='需要 CUDA 验证构造设备')
def test_gpu_rope_cache_matches_upstream_and_preserves_conversion_origin():
    from vllm.config import VllmConfig
    from vllm.config.vllm import set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding as UpRotary

    params={'rope_theta': 1_000_000.0}
    with set_current_vllm_config(VllmConfig()), torch.device('cuda'):
        upstream=UpRotary(128,128,1024,1_000_000.0,True,torch.bfloat16)
        actual=get_rope(128,1024,rope_parameters=params,dtype=torch.bfloat16)
        sibling=get_rope(128,1024,rope_parameters=params,dtype=torch.bfloat16)
    assert actual.cos_sin_cache.device.type == 'cuda'
    assert torch.equal(actual.cos_sin_cache,upstream.cos_sin_cache)
    assert actual.cos_sin_cache.untyped_storage().data_ptr()==sibling.cos_sin_cache.untyped_storage().data_ptr()

    with torch.device('cpu'):
        cpu=get_rope(128,1024,rope_parameters=params,dtype=torch.bfloat16)
        cpu_sibling=get_rope(128,1024,rope_parameters=params,dtype=torch.bfloat16)
        original=cpu.cos_sin_cache.clone()
    cpu.to('cuda');cpu_sibling.to('cuda')
    assert torch.equal(cpu.cos_sin_cache.cpu(),original)
    assert cpu.cos_sin_cache.untyped_storage().data_ptr()==cpu_sibling.cos_sin_cache.untyped_storage().data_ptr()
    # 两个构造来源可能因 sin/cos 的设备实现产生不同舍入，不能共用目标设备缓存。
    assert cpu.cos_sin_cache.untyped_storage().data_ptr()!=actual.cos_sin_cache.untyped_storage().data_ptr()


def test_dtype_roundtrip_preserves_converted_values_without_retaining_old_table():
    import gc
    import weakref

    rope=get_rope(128,1024,rope_parameters={'rope_theta':1_000_000.0},dtype=torch.float32)
    original=rope.cos_sin_cache.clone()
    prior=weakref.ref(rope._rope_table)
    rope.to(torch.bfloat16)
    gc.collect()
    assert prior() is None
    rounded=original.to(torch.bfloat16).float()
    rope.to(torch.float32)
    assert torch.equal(rope.cos_sin_cache,rounded)
