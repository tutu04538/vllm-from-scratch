"""旋转位置编码（对应 vLLM `layers/rotary_embedding/` 的子集）。

本关把两件事合在一起：

1. **表按完整配置共享**：`get_rope()` 用 head/rotary dimension、位置上限、base、
   rope type/partial factor、旋转风格、dtype、device 组成不可变 key。同一 key 的
   多个模型层拿到同一份 `cos_sin_cache` storage；key 不同（哪怕 shape 相同）不共享。
2. **紧凑布局**：表宽 = `rotary_dim`（不是 `2 * rotary_dim`），前半是 cos、后半是
   sin；普通 partial RoPE 的未旋转维不写进表。Gemma4 proportional 的未旋转维仍要
   占表宽，用 `cos=1/sin=0` 表示恒等。

生命周期用 **weak-value 表缓存**：缓存不持有 module，两个活模型真正共享同一块
storage；最后一个使用者释放后条目自动消失，不会把旧引擎/旧 CUDA context 保活。
设备转换由 `_apply` 重新取 canonical 表，避免每层 `.to()` 把共享 buffer 复制成
各自独立的 storage。

数学约定（neox 风格，与 HF/vLLM 一致）：

    emb = [freqs, freqs]             # 旧布局；本实现不再物化
    q_rot = q * cos + rotate_half(q) * sin

`positions` 是请求内的**绝对位置**。CUDA 前向的融合实现在
`layers/ops/rotary_embedding.py`；CPU 路径保留显式 Torch 参考。
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .ops.rotary_embedding import (apply_rotary_embedding,
                                   apply_rotary_embedding_native)

_SUPPORTED_ROPE_TYPES = ("default", "proportional")
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


class _RopeTable:
    """一份共享的紧凑 cos/sin 表；由使用它的 RotaryEmbedding 持有强引用。"""

    __slots__ = ("key", "cos_sin_cache", "__weakref__")

    def __init__(self, key: tuple[Any, ...], cos_sin_cache: torch.Tensor) -> None:
        self.key = key
        self.cos_sin_cache = cos_sin_cache

    @property
    def nbytes(self) -> int:
        return self.cos_sin_cache.numel() * self.cos_sin_cache.element_size()


# key 的最后两项是 dtype/device；前面的部分由 get_rope 统一生成。
_ROPE_TABLE_CACHE: "weakref.WeakValueDictionary[tuple[Any, ...], _RopeTable]" = (
    weakref.WeakValueDictionary())
_ROPE_CACHE_LOCK = threading.Lock()


def _freeze(value: Any) -> Any:
    """把配置递归变成可哈希、顺序稳定的不可变 key；不修改输入。"""
    if isinstance(value, Mapping):
        items = sorted(((str(k), _freeze(v)) for k, v in value.items()),
                       key=lambda item: item[0])
        return tuple(items)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        # 10000 与 10000.0 在数学上是同一个 base，不应因此建两张表。
        return float(value)
    if isinstance(value, (str, torch.dtype, torch.device)) or value is None:
        return value
    raise TypeError(
        f"RoPE 配置里出现了不可哈希的值 {value!r} ({type(value).__name__})；"
        f"请只传标量/列表/字典")


def _canonical_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        resolved = torch.device("cpu")
    else:
        resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        # `cuda` 与 `cuda:0` 在 torch.device 比较里不相等；统一成带 index 的形式，
        # 避免同一条 GPU 因写法不同拆成两个 key。
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        resolved = torch.device("cuda", index)
    return resolved


def _resolve_rotary_dim(head_size: int, rope_parameters: Mapping[str, Any]) -> int:
    rope_dim = rope_parameters.get("rope_dim")
    if rope_dim:
        if int(rope_dim) != rope_dim:
            raise ValueError(f"rope_dim 必须是整数，收到 {rope_dim!r}")
        rotary_dim = int(rope_dim)
    else:
        partial = float(rope_parameters.get("partial_rotary_factor", 1.0))
        if partial <= 0.0 or partial > 1.0:
            raise ValueError(
                f"partial_rotary_factor 必须在 (0, 1]，收到 {partial}")
        rotary_dim = int(head_size * partial)
    if rotary_dim <= 0:
        raise ValueError(f"rotary_dim 必须为正，收到 {rotary_dim}")
    if rotary_dim > head_size:
        raise ValueError(
            f"rotary_dim={rotary_dim} 不能大于 head_size={head_size}")
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim 必须是偶数，收到 {rotary_dim}")
    return rotary_dim


def _make_key_parts(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    is_neox_style: bool,
    rope_type: str,
    rope_parameters: Mapping[str, Any],
) -> tuple[Any, ...]:
    return (
        int(head_size),
        int(rotary_dim),
        int(max_position),
        float(base),
        bool(is_neox_style),
        rope_type,
        _freeze(rope_parameters),
    )


def _compute_compact_cache(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_type: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """按上游 `_compute_cos_sin_cache` 的紧凑布局建表。

    先在 CPU FP32 上算（与上游构造时的 CPU 表逐值一致），最后一次性转 dtype/device。
    """
    if base <= 0:
        raise ValueError(f"RoPE base 必须为正，收到 {base}")
    if rope_type == "proportional":
        rope_angles = rotary_dim // 2
        nope_angles = head_size // 2 - rope_angles
        if nope_angles < 0:
            raise ValueError(
                f"proportional RoPE 要求 rotary_dim <= head_size，"
                f"收到 rotary_dim={rotary_dim}, head_size={head_size}")
        exponents = (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32)
                     / head_size)
        inv_freq = 1.0 / (base ** exponents)
        if nope_angles:
            inv_freq = torch.cat(
                (inv_freq, torch.zeros(nope_angles, dtype=torch.float32)))
        apply_dim = head_size
    else:
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32)
                     / rotary_dim))
        apply_dim = rotary_dim

    positions = torch.arange(max_position, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", positions, inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    if cache.shape != (max_position, apply_dim):
        raise AssertionError(
            f"内部错误：cos/sin 表形状 {tuple(cache.shape)} != "
            f"{(max_position, apply_dim)}")
    return cache.to(dtype=dtype, device=device)


def _get_table(
    key_parts: tuple[Any, ...],
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> _RopeTable:
    if dtype not in _FLOAT_DTYPES:
        raise ValueError(
            f"RoPE 表只支持浮点 dtype（fp16/bf16/fp32/fp64），收到 {dtype}")
    device = _canonical_device(device)
    if device.type == "meta":
        raise ValueError("RoPE 表不能建在 meta device 上；先建表再 .to(device)")
    key = key_parts + (dtype, device)
    with _ROPE_CACHE_LOCK:
        table = _ROPE_TABLE_CACHE.get(key)
        if table is None:
            head_size, rotary_dim, max_position, base, _, rope_type, _ = key_parts
            cache = _compute_compact_cache(
                head_size, rotary_dim, max_position, base, rope_type,
                dtype, device)
            table = _RopeTable(key, cache)
            _ROPE_TABLE_CACHE[key] = table
        return table


def get_rope(
    head_size: int,
    max_position: int,
    is_neox_style: bool = True,
    rope_parameters: Mapping[str, Any] | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> "RotaryEmbedding":
    """统一工厂（对齐上游 `get_rope` 的核心参数）。

    只支持本仓库已经对齐的 `default` 与 `proportional`；其他 rope_type 显式报错，
    不静默按 default 算。`rope_parameters` 不会被修改，嵌套字段进入稳定键。
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if not isinstance(head_size, int) or head_size <= 0:
        raise ValueError(f"head_size 必须是正整数，收到 {head_size!r}")
    if not isinstance(max_position, int) or max_position <= 0:
        raise ValueError(f"max_position 必须是正整数，收到 {max_position!r}")
    if not isinstance(is_neox_style, bool):
        raise ValueError(f"is_neox_style 必须是 bool，收到 {is_neox_style!r}")

    # 只复制一层并递归冻结；不改调用方字典。
    parameters = dict(rope_parameters or {})
    rope_type = str(parameters.get("rope_type", "default"))
    if rope_type not in _SUPPORTED_ROPE_TYPES:
        raise ValueError(
            f"RoPE 类型 {rope_type!r} 未适配；本仓库只实现 "
            f"default/proportional（Gemma4 full 层用 proportional）")
    base = float(parameters.get("rope_theta", 10000.0))
    rotary_dim = _resolve_rotary_dim(head_size, parameters)
    key_parts = _make_key_parts(
        head_size, rotary_dim, max_position, base, is_neox_style,
        rope_type, parameters)
    table = _get_table(key_parts, dtype, device)

    cls = (Gemma4RotaryEmbedding if rope_type == "proportional"
           else RotaryEmbedding)
    return cls(
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        dtype,
        device=device,
        _table=table,
        _key_parts=key_parts,
    )


def _init_rope_module(
    module: nn.Module,
    *,
    head_size: int,
    declared_rotary_dim: int,
    apply_dim: int,
    max_position: int,
    base: float,
    is_neox_style: bool,
    dtype: torch.dtype,
    device: torch.device | str | None,
    rope_type: str,
    table: _RopeTable | None,
    key_parts: tuple[Any, ...] | None,
) -> None:
    nn.Module.__init__(module)
    if not isinstance(head_size, int) or head_size <= 0:
        raise ValueError(f"head_size 必须是正整数，收到 {head_size!r}")
    if not isinstance(declared_rotary_dim, int) or declared_rotary_dim <= 0:
        raise ValueError(
            f"rotary_dim 必须是正整数，收到 {declared_rotary_dim!r}")
    if declared_rotary_dim > head_size:
        raise ValueError(
            f"rotary_dim={declared_rotary_dim} 不能大于 head_size={head_size}")
    if declared_rotary_dim % 2 != 0:
        raise ValueError(
            f"rotary_dim 必须是偶数，收到 {declared_rotary_dim}")
    if not isinstance(max_position, int) or max_position <= 0:
        raise ValueError(f"max_position 必须是正整数，收到 {max_position!r}")
    if dtype not in _FLOAT_DTYPES:
        raise ValueError(
            f"RoPE 表只支持浮点 dtype（fp16/bf16/fp32/fp64），收到 {dtype}")

    module.head_size = head_size
    module.declared_rotary_dim = declared_rotary_dim
    # 和上游一样：proportional 把 apply_dim 设成 head_size，未旋转维用恒等表。
    module.rotary_dim = apply_dim
    module.max_position_embeddings = max_position
    module.base = base
    module.is_neox_style = is_neox_style
    module.dtype = dtype
    module.rope_type = rope_type

    if key_parts is None:
        parameters = {
            "rope_theta": base,
            "rope_type": rope_type,
            "rope_dim": declared_rotary_dim,
        }
        key_parts = _make_key_parts(
            head_size, declared_rotary_dim, max_position, base,
            is_neox_style, rope_type, parameters)
    if table is None:
        table = _get_table(key_parts, dtype, device)
    if table.cos_sin_cache.dtype != dtype:
        raise ValueError(
            f"内部错误：表 dtype={table.cos_sin_cache.dtype} 与请求 {dtype} 不一致")
    module._rope_key_parts = key_parts
    module._rope_table = table
    module.register_buffer(
        "cos_sin_cache", table.cos_sin_cache, persistent=False)


class RotaryEmbedding(nn.Module):
    """普通（default）RoPE：只旋转前 `rotary_dim` 维。

    直接构造会按 `(head_size, rotary_dim, max_position, base, is_neox_style,
    dtype, device)` 建/取共享表；模型路径统一走 `get_rope`。
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float = 10000.0,
        is_neox_style: bool = True,
        dtype: torch.dtype | None = None,
        *,
        device: torch.device | str | None = None,
        _table: _RopeTable | None = None,
        _key_parts: tuple[Any, ...] | None = None,
    ) -> None:
        if dtype is None:
            dtype = torch.get_default_dtype()
        _init_rope_module(
            self,
            head_size=head_size,
            declared_rotary_dim=rotary_dim,
            apply_dim=rotary_dim,
            max_position=max_position_embeddings,
            base=float(base),
            is_neox_style=is_neox_style,
            dtype=dtype,
            device=device,
            rope_type="default",
            table=_table,
            key_parts=_key_parts,
        )

    def _apply(self, fn, recurse: bool = True):
        """dtype/device 转换后重新取 canonical 表，而不是每层复制一份。"""
        cache = self._buffers.pop("cos_sin_cache", None)
        try:
            result = super()._apply(fn, recurse)
        except BaseException:
            if cache is not None and "cos_sin_cache" not in self._buffers:
                self._buffers["cos_sin_cache"] = cache
            raise
        if cache is None:
            return result
        try:
            # 只探测 fn 会把表变成什么 dtype/device；不复制整张表再让每层分叉。
            probe = torch.empty(0, dtype=cache.dtype, device=cache.device)
            moved = fn(probe)
            if moved.dtype != cache.dtype or moved.device != cache.device:
                table = _get_table(
                    self._rope_key_parts, moved.dtype, moved.device)
                self._rope_table = table
                self._buffers["cos_sin_cache"] = table.cos_sin_cache
            else:
                self._buffers["cos_sin_cache"] = cache
        except BaseException:
            self._buffers["cos_sin_cache"] = cache
            raise
        return result

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """入参与返回都是扁平 `[num_tokens, num_heads * head_size]`。

        CUDA 路径原地旋转并返回同一批张量；CPU 路径返回新张量。`key=None` 是
        Gemma4 Q-only 的合法输入。
        """
        return apply_rotary_embedding(
            positions, query, key, self.cos_sin_cache, self.head_size,
            self.rotary_dim, self.is_neox_style)

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """显式 Torch 参考路径（不原地改输入）。"""
        return apply_rotary_embedding_native(
            positions, query, key, self.cos_sin_cache, self.head_size,
            self.rotary_dim, self.is_neox_style)

    def apply_to_query(self, positions: torch.Tensor,
                       query: torch.Tensor) -> torch.Tensor:
        query, _ = self.forward(positions, query, None)
        return query

    def apply_to_key(self, positions: torch.Tensor,
                     key: torch.Tensor) -> torch.Tensor:
        key, _ = self.forward(positions, key, None)
        return key

    def get_cos_sin(self, seqlen: int) -> tuple[torch.Tensor, torch.Tensor]:
        """前 `seqlen` 个位置的 `(cos, sin)`，各 `[seqlen, rotary_dim/2]`。"""
        if seqlen < 0 or seqlen > self.max_position_embeddings:
            raise ValueError(
                f"seqlen={seqlen} 超出 [0, {self.max_position_embeddings}]")
        cos_sin = self.cos_sin_cache[:seqlen]
        return cos_sin.chunk(2, dim=-1)

    def extra_repr(self) -> str:
        return (
            f"head_size={self.head_size}, rotary_dim={self.rotary_dim}, "
            f"declared_rotary_dim={self.declared_rotary_dim}, "
            f"max_position_embeddings={self.max_position_embeddings}, "
            f"base={self.base}, rope_type={self.rope_type}, "
            f"is_neox_style={self.is_neox_style}, dtype={self.dtype}")


class Gemma4RotaryEmbedding(RotaryEmbedding):
    """Gemma4 proportional RoPE（对应上游 `Gemma4RotaryEmbedding`）。

    频率指数的分母是 **head_size**，不是 `rotary_dim`；未旋转的 `nope_angles`
    对补零，因此 `cos=1/sin=0`，是恒等旋转，而不是把 cos/sin 直接写成 0。
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float = 10000.0,
        is_neox_style: bool = True,
        dtype: torch.dtype | None = None,
        *,
        device: torch.device | str | None = None,
        _table: _RopeTable | None = None,
        _key_parts: tuple[Any, ...] | None = None,
    ) -> None:
        if rotary_dim > head_size:
            raise ValueError(
                f"proportional RoPE 要求 rotary_dim <= head_size，"
                f"收到 rotary_dim={rotary_dim}, head_size={head_size}")
        if rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim 必须是偶数，收到 {rotary_dim}")
        self.rope_angles = rotary_dim // 2
        self.nope_angles = head_size // 2 - self.rope_angles
        if dtype is None:
            dtype = torch.get_default_dtype()
        _init_rope_module(
            self,
            head_size=head_size,
            declared_rotary_dim=rotary_dim,
            apply_dim=head_size,
            max_position=max_position_embeddings,
            base=float(base),
            is_neox_style=is_neox_style,
            dtype=dtype,
            device=device,
            rope_type="proportional",
            table=_table,
            key_parts=_key_parts,
        )

    def extra_repr(self) -> str:
        return (
            f"{super().extra_repr()}, rope_angles={self.rope_angles}, "
            f"nope_angles={self.nope_angles}")


def get_rope_cache_stats() -> dict[str, int]:
    """当前进程里活着的共享表数量与唯一 storage 字节数（测试/审计用）。"""
    with _ROPE_CACHE_LOCK:
        tables = list(_ROPE_TABLE_CACHE.values())
    unique: dict[int, int] = {}
    for table in tables:
        cache = table.cos_sin_cache
        unique[cache.untyped_storage().data_ptr()] = table.nbytes
    return {"entries": len(unique), "bytes": sum(unique.values())}


def _rope_cache_storage_ptrs() -> set[int]:
    """只给测试/审计用：当前弱缓存里还活着的 storage 指针。"""
    with _ROPE_CACHE_LOCK:
        tables = list(_ROPE_TABLE_CACHE.values())
    return {table.cos_sin_cache.untyped_storage().data_ptr()
            for table in tables}


def _clear_rope_cache_for_tests() -> None:
    """只给测试用：清空弱引用表缓存；不会释放仍被模型持有的表。"""
    with _ROPE_CACHE_LOCK:
        _ROPE_TABLE_CACHE.clear()
