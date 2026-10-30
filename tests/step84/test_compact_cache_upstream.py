"""84-A：紧凑表与冻结上游逐位置对照（CPU，不依赖 CUDA）。"""

from __future__ import annotations

import pytest
import torch

from minivllm.layers import get_rope

from spec84_helpers import rope_parameters


@pytest.fixture(scope="module")
def upstream_env():
    """上游 `CustomOp` 构造要求 current_vllm_config 上下文。"""
    from vllm.config import VllmConfig
    from vllm.config.vllm import set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        yield


def _build_pair(*, proportional: bool, dtype: torch.dtype,
                head_size: int = 16, rotary_dim: int = 8,
                max_position: int = 64, base: float = 10000.0,
                is_neox_style: bool = True):
    from vllm.model_executor.layers.rotary_embedding.base import (
        RotaryEmbedding as UpRotary,
    )
    from vllm.model_executor.layers.rotary_embedding.gemma4_rope import (
        Gemma4RotaryEmbedding as UpGemma4,
    )

    if proportional:
        upstream = UpGemma4(
            head_size, rotary_dim, max_position, base, is_neox_style, dtype)
        ours = get_rope(
            head_size, max_position, is_neox_style,
            rope_parameters(proportional=True, base=base,
                            rotary_dim=rotary_dim),
            dtype=dtype)
    else:
        upstream = UpRotary(
            head_size, rotary_dim, max_position, base, is_neox_style, dtype)
        ours = get_rope(
            head_size, max_position, is_neox_style,
            rope_parameters(base=base, rotary_dim=rotary_dim),
            dtype=dtype)
    return upstream, ours


@pytest.mark.parametrize("proportional", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compact_cache_matches_upstream_at_boundary_positions(
        upstream_env, proportional, dtype):
    head_size = 16
    rotary_dim = 4 if proportional else 8
    max_position = 64
    upstream, ours = _build_pair(
        proportional=proportional, dtype=dtype, head_size=head_size,
        rotary_dim=rotary_dim, max_position=max_position)

    assert upstream.cos_sin_cache.shape == ours.cos_sin_cache.shape
    assert ours.cos_sin_cache.shape == (
        max_position, head_size if proportional else rotary_dim)
    # 0、中间、大位置、最后一个合法位置。
    positions = [0, max_position // 2, max_position - 2, max_position - 1]
    assert torch.equal(
        upstream.cos_sin_cache[positions],
        ours.cos_sin_cache[positions])
    # 整张表也应逐位相同；位置抽查只是把边界写进测试名。
    assert torch.equal(upstream.cos_sin_cache, ours.cos_sin_cache)


@pytest.mark.parametrize("proportional", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("is_neox_style", [True, False])
def test_native_rotation_matches_upstream_and_preserves_pass_through(
        upstream_env, proportional, dtype, is_neox_style):
    head_size = 16
    rotary_dim = 4 if proportional else 8
    upstream, ours = _build_pair(
        proportional=proportional, dtype=dtype, head_size=head_size,
        rotary_dim=rotary_dim, is_neox_style=is_neox_style)
    generator = torch.Generator().manual_seed(84)
    q = torch.randn(5, 2 * head_size, generator=generator).to(dtype)
    k = torch.randn(5, 2 * head_size, generator=generator).to(dtype)
    q_before, k_before = q.clone(), k.clone()
    positions = torch.tensor([0, 1, 7, 31, 63])

    up_q, up_k = upstream.forward_native(positions, q, k)
    our_q, our_k = ours.forward_native(positions, q, k)
    assert torch.equal(up_q, our_q)
    assert torch.equal(up_k, our_k)
    # forward_native 不原地改输入。
    assert torch.equal(q, q_before)
    assert torch.equal(k, k_before)

    # 未旋转维原样保留（普通 partial 的 [rotary:]；proportional 的 nope 对）。
    if proportional and is_neox_style:
        # nope 是每个半区的后半段，不是最后一整段；直接比“整份输入”不成立。
        # 用 neox 的恒等性质：cos=1/sin=0 时 y0=x0、y1=x1。
        head = our_q.view(5, 2, head_size)
        source = q_before.view(5, 2, head_size)
        half = head_size // 2
        assert torch.equal(
            head[..., rotary_dim // 2: half],
            source[..., rotary_dim // 2: half])
        assert torch.equal(
            head[..., half + rotary_dim // 2:],
            source[..., half + rotary_dim // 2:])
    elif not proportional:
        # 普通 partial：旋转只发生在前 rotary_dim 维，与 neox/gptj 无关。
        assert torch.equal(
            our_q.view(5, 2, head_size)[..., rotary_dim:],
            q_before.view(5, 2, head_size)[..., rotary_dim:])
        assert torch.equal(
            our_k.view(5, 2, head_size)[..., rotary_dim:],
            k_before.view(5, 2, head_size)[..., rotary_dim:])


def test_position_out_of_range_errors_at_cache_boundary(upstream_env):
    ours = get_rope(
        16, 64, True, rope_parameters(rotary_dim=8), dtype=torch.float32)
    q = torch.randn(2, 32)
    for bad in (64, 65, -1):
        with pytest.raises((IndexError, RuntimeError)):
            ours.forward_native(torch.tensor([0, bad]), q)


def test_factory_rejects_unimplemented_scaling(upstream_env):
    with pytest.raises(ValueError, match="未适配"):
        get_rope(16, 64, True, {"rope_type": "yarn", "rope_theta": 10000.0})
