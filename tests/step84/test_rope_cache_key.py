"""84-A：RoPE 表的 key、共享、紧凑布局与生命周期。"""

from __future__ import annotations

import copy
import gc

import pytest
import torch

from minivllm.layers import (Gemma4RotaryEmbedding, RotaryEmbedding,
                             get_rope, get_rope_cache_stats)
from minivllm.layers.rotary_embedding import _clear_rope_cache_for_tests

from spec84_helpers import compact_cache, rope_parameters


def _ptr(module) -> int:
    return module.cos_sin_cache.untyped_storage().data_ptr()


def test_same_key_shares_storage_and_math_config_changes_do_not():
    base = get_rope(16, max_position=64, rope_parameters=rope_parameters())
    same = get_rope(16, max_position=64, rope_parameters=rope_parameters())
    assert _ptr(base) == _ptr(same)

    # 只 shape 一样不够：base 不同必须拆表。
    different_base = get_rope(
        16, max_position=64, rope_parameters=rope_parameters(base=500000.0))
    # partial/rotary 不同、max_position 不同、旋转风格不同。
    different_partial = get_rope(
        16, max_position=64, rope_parameters=rope_parameters(rotary_dim=8))
    different_max = get_rope(16, max_position=32, rope_parameters=rope_parameters())
    different_style = get_rope(
        16, max_position=64, is_neox_style=False,
        rope_parameters=rope_parameters())
    different_dtype = get_rope(
        16, max_position=64, rope_parameters=rope_parameters(),
        dtype=torch.bfloat16)
    for other in (different_base, different_partial, different_max,
                  different_style, different_dtype):
        assert _ptr(base) != _ptr(other)
        assert base.rotary_dim == 16


def test_proportional_is_a_distinct_key_and_distinct_class():
    default = get_rope(
        16, max_position=64, rope_parameters=rope_parameters(rotary_dim=4))
    proportional = get_rope(
        16, max_position=64,
        rope_parameters=rope_parameters(proportional=True, rotary_dim=4))
    assert isinstance(default, RotaryEmbedding)
    assert not isinstance(default, Gemma4RotaryEmbedding)
    assert isinstance(proportional, Gemma4RotaryEmbedding)
    assert _ptr(default) != _ptr(proportional)
    # proportional 的公开 rotary_dim 是 apply_dim(head_size)，不是声明的 4。
    assert proportional.declared_rotary_dim == 4
    assert proportional.rotary_dim == 16
    assert proportional.rope_angles == 2
    assert proportional.nope_angles == 6
    assert proportional.cos_sin_cache.shape == (64, 16)


def test_rope_parameters_not_mutated_and_nested_key_is_stable():
    parameters = {
        "rope_theta": 10000.0,
        "rope_type": "default",
        "extra": {"items": [1, 2], "flag": True},
    }
    snapshot = copy.deepcopy(parameters)
    first = get_rope(16, max_position=64, rope_parameters=parameters)
    assert parameters == snapshot

    reordered = {
        "extra": {"flag": True, "items": [1, 2]},
        "rope_type": "default",
        "rope_theta": 10000.0,
    }
    second = get_rope(16, max_position=64, rope_parameters=reordered)
    assert _ptr(first) == _ptr(second)
    # list/tuple 在 cache key 里等价，但不会修改调用方。
    tuple_form = {
        "rope_theta": 10000.0,
        "rope_type": "default",
        "extra": {"items": (1, 2), "flag": True},
    }
    third = get_rope(16, max_position=64, rope_parameters=tuple_form)
    assert _ptr(first) == _ptr(third)
    assert parameters["extra"]["items"] == [1, 2]
    assert isinstance(reordered["extra"]["items"], list)


def test_compact_layout_shape_and_bytes():
    module = get_rope(16, max_position=64, rope_parameters=rope_parameters())
    assert module.cos_sin_cache.shape == (64, 16)
    assert module.cos_sin_cache.numel() * module.cos_sin_cache.element_size() == (
        64 * 16 * module.cos_sin_cache.element_size())
    # 旧实现是 2 * rotary_dim；新实现宽度就是 rotary_dim。
    assert module.cos_sin_cache.shape[1] == module.rotary_dim

    proportional = get_rope(
        16, max_position=64,
        rope_parameters=rope_parameters(proportional=True, rotary_dim=4))
    assert proportional.cos_sin_cache.shape == (64, 16)
    cos, sin = proportional.get_cos_sin(5)
    assert cos.shape == (5, 8) and sin.shape == (5, 8)
    # 未旋转的 nope 对：cos=1、sin=0，不是整行清零。
    assert torch.equal(cos[:, proportional.rope_angles:],
                       torch.ones_like(cos[:, proportional.rope_angles:]))
    assert torch.equal(sin[:, proportional.rope_angles:],
                       torch.zeros_like(sin[:, proportional.rope_angles:]))


def test_compact_cache_matches_independent_float32_formula():
    base = get_rope(16, max_position=64, rope_parameters=rope_parameters())
    expected = compact_cache(16, 16, 64)
    assert torch.equal(base.cos_sin_cache, expected)

    proportional = get_rope(
        16, max_position=64,
        rope_parameters=rope_parameters(proportional=True, rotary_dim=4))
    expected_prop = compact_cache(16, 4, 64, proportional=True)
    assert torch.equal(proportional.cos_sin_cache, expected_prop)


def test_apply_dtype_conversion_keeps_shared_storage():
    left = get_rope(16, max_position=32, rope_parameters=rope_parameters())
    right = get_rope(16, max_position=32, rope_parameters=rope_parameters())
    assert left.cos_sin_cache.dtype == torch.float32
    left.to(torch.bfloat16)
    right.to(torch.bfloat16)
    assert left.cos_sin_cache.dtype == torch.bfloat16
    assert _ptr(left) == _ptr(right)
    # 来回转换仍然共享，不能变成每层独立 storage。
    left.to(torch.float32)
    right.to(torch.float32)
    assert left.cos_sin_cache.dtype == torch.float32
    assert _ptr(left) == _ptr(right)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_device_is_part_of_key_and_to_cuda_keeps_sharing():
    cpu_left = get_rope(16, max_position=32, rope_parameters=rope_parameters())
    cpu_right = get_rope(16, max_position=32, rope_parameters=rope_parameters())
    assert cpu_left.cos_sin_cache.device.type == "cpu"
    cpu_left.to("cuda")
    cpu_right.to("cuda")
    assert cpu_left.cos_sin_cache.device.type == "cuda"
    assert _ptr(cpu_left) == _ptr(cpu_right)
    # CPU 表与 CUDA 表是不同 storage，即使 key 的数学部分一样。
    cpu_only = get_rope(16, max_position=32, rope_parameters=rope_parameters())
    assert cpu_only.cos_sin_cache.device.type == "cpu"
    assert _ptr(cpu_only) != _ptr(cpu_left)


def test_weak_cache_releases_after_last_user_without_keeping_modules_alive():
    gc.collect()
    before = get_rope_cache_stats()
    left = get_rope(16, max_position=32,
                    rope_parameters=rope_parameters(base=12345.0))
    right = get_rope(16, max_position=32,
                     rope_parameters=rope_parameters(base=12345.0))
    assert _ptr(left) == _ptr(right)
    left = None
    gc.collect()
    # 还有一个活使用者，表不能消失。
    assert get_rope(16, max_position=32,
                    rope_parameters=rope_parameters(base=12345.0)).cos_sin_cache.data_ptr() == (
        right.cos_sin_cache.data_ptr())
    right = None
    gc.collect()
    after = get_rope_cache_stats()
    assert after["entries"] <= before["entries"] + 1
    # 唯一字节回到基线附近；至少不能留下这张 12345 的表。
    assert after["bytes"] <= before["bytes"]


def test_lifetime_does_not_keep_engine_objects_alive():
    class Owner:
        def __init__(self):
            self.rope = get_rope(
                16, max_position=32,
                rope_parameters=rope_parameters(base=54321.0))

    gc.collect()
    before = get_rope_cache_stats()
    owner = Owner()
    ptr = owner.rope.cos_sin_cache.data_ptr()
    del owner
    gc.collect()
    after = get_rope_cache_stats()
    assert after["entries"] <= before["entries"]
    assert after["bytes"] <= before["bytes"]
    del ptr


@pytest.mark.parametrize("parameters", [
    {"rope_type": "llama3", "rope_theta": 10000.0},
    {"rope_type": "default", "partial_rotary_factor": 0.0},
    {"rope_type": "default", "partial_rotary_factor": 1.5},
    {"rope_type": "default", "rope_dim": 15},
    {"rope_type": "default", "rope_dim": 18},
])
def test_invalid_configs_fail_at_factory(parameters):
    with pytest.raises(ValueError):
        get_rope(16, max_position=32, rope_parameters=parameters)


def test_invalid_dtype_and_empty_cache_fail():
    with pytest.raises(ValueError):
        get_rope(16, max_position=32, rope_parameters=rope_parameters(),
                 dtype=torch.int8)
    with pytest.raises(ValueError):
        get_rope(16, max_position=0, rope_parameters=rope_parameters())
    with pytest.raises(ValueError):
        get_rope(16, max_position=32, is_neox_style=1,
                 rope_parameters=rope_parameters())


def test_forward_rejects_dtype_mismatch_instead_of_copying_shared_cache():
    module = get_rope(16, max_position=32, rope_parameters=rope_parameters())
    before = module.cos_sin_cache.data_ptr()
    q = torch.randn(3, 32, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="cos_sin_cache dtype"):
        module(torch.tensor([0, 1, 2]), q)
    assert module.cos_sin_cache.data_ptr() == before
    assert module.cos_sin_cache.dtype == torch.float32


def test_clear_cache_helper_only_drops_weak_index():
    module = get_rope(16, max_position=32,
                      rope_parameters=rope_parameters(base=777.0))
    ptr = module.cos_sin_cache.data_ptr()
    _clear_rope_cache_for_tests()
    # 活 module 的 storage 还在；helper 只清索引，不 free 正在用的表。
    assert module.cos_sin_cache.data_ptr() == ptr
    # 索引被清掉后，新调用会重新建一张表；旧 module 的 storage 仍合法。
    again = get_rope(16, max_position=32,
                     rope_parameters=rope_parameters(base=777.0))
    assert again.cos_sin_cache.data_ptr() != ptr
