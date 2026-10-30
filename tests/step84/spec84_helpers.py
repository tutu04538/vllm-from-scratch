"""需求 84 的测试辅助：只放纯函数，不依赖 benchmark 脚本。"""

from __future__ import annotations

import copy

import torch


def compact_cache(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    *,
    base: float = 10000.0,
    proportional: bool = False,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """独立重建紧凑表（前半 cos、后半 sin），用于验证生产实现没有多拼一份。"""
    if proportional:
        rope_angles = rotary_dim // 2
        nope_angles = head_size // 2 - rope_angles
        exponents = (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32)
                     / head_size)
        inv_freq = 1.0 / (base ** exponents)
        if nope_angles:
            inv_freq = torch.cat(
                (inv_freq, torch.zeros(nope_angles, dtype=torch.float32)))
        width = head_size
    else:
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32)
                     / rotary_dim))
        width = rotary_dim
    positions = torch.arange(max_position, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", positions, inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    assert cache.shape == (max_position, width)
    return cache.to(dtype=dtype, device=device)


def rope_parameters(*, base: float = 10000.0, proportional: bool = False,
                    rotary_dim: int | None = None,
                    partial: float | None = None,
                    rope_type: str | None = None) -> dict:
    params = {"rope_theta": base}
    if proportional:
        params["rope_type"] = "proportional"
    elif rope_type is not None:
        params["rope_type"] = rope_type
    if rotary_dim is not None:
        params["rope_dim"] = rotary_dim
    if partial is not None:
        params["partial_rotary_factor"] = partial
    return params


def packed_qkv(
    num_tokens: int,
    q_heads: int,
    kv_heads: int,
    head_size: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """连续 QKV buffer + 三个切片；切片 token stride 是整条 buffer 宽度。"""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    width = (q_heads + 2 * kv_heads) * head_size
    qkv = torch.randn(
        num_tokens, width, generator=generator, dtype=torch.float32
    ).to(dtype=dtype, device=device)
    q = qkv[:, : q_heads * head_size]
    k = qkv[:, q_heads * head_size : (q_heads + kv_heads) * head_size]
    v = qkv[:, (q_heads + kv_heads) * head_size :]
    return qkv, q, k, v


def cloned(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone()


def deep_copy_mapping(value):
    return copy.deepcopy(value)
