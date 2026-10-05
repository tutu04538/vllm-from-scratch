"""Hidden States Extractor 模型（对应 vLLM `model_executor/models/extract_hidden_states.py`）。

它的全部"计算"就是**把 target 的辅助层特征按 `slot_mapping` 散射进一个分页缓存**：没有
attention、没有权重、没有 lm_head，`load_weights()` 什么都不装。所以它不是"一个能生成 token
的模型"，而是"**借 KV 缓存的寻址与生命周期来存特征**"的载体（需求 064 §1）：

    天真的做法（每步 torch.save / 存在 Python 里）没有寻址、没有块池、没有释放时机，
    prefix 命中时还会丢掉"已经算过、不重算"的那段特征；
    这里的做法是把一份 `[T, L, H]` 的特征塞进形状为 `[num_blocks, block_size, L, H]` 的
    "KV 缓存"——**L 当 head 数、H 当 head_size**，于是一个 token 恰好占一个 slot，
    和 target 的 KV 用同一份 `slot_mapping`、同一张块表、同一套块生命周期。

四个类，与上游一一对应：

    CacheOnlyAttentionBackend       后端契约：缓存形状、impl 类、metadata builder 类
    CacheOnlyAttentionMetadata      cache-only 层的元数据**只有槽位**
    CacheOnlyAttentionImpl          只做散射（`basic_cache`），不算 attention
    CacheOnlyAttentionLayer         模型里的"注意力边界"：取 metadata → 交给 impl
    ExtractHiddenStatesModel        只持有**一个** cache-only 层（层名带 target 的层数）

**与上游的实现差异**（逐条写清，另见 docs/step64_alignment.md §3）：

1. 上游的层从 `forward_context.slot_mapping`（`dict[layer_name, Tensor]`）取槽位，并用
   `unified_kv_cache_update()` 这个自定义算子把"写缓存"和后续 attention 的**顺序**钉住
   （给 torch.compile 看），本仓库的层统一从 `attn_metadata[layer_name]` 取
   （`attention/layer.py::Attention` 同款），也没有 compile（69 关才有）。
2. 上游 `@maybe_transfer_kv_layer` 装饰的 `dummy_attention()` 用来触发 KV connector 传输
   （81 关的跨进程搬运）；本仓库没有 connector，所以那一步不存在。
3. 上游 `set_default_quant_scales` / `is_quantized_kv_cache` / `CacheDType`：本仓库 KV 精度
   **就是**模型精度（没有 `cache_dtype` 这个旋钮），所以只保留"写进去的 dtype 必须与缓存一致"
   这条断言。
4. 上游 `get_kv_cache_spec()` 产出 `HiddenStateCacheSpec`（KV 规格类型树 + 单独一个 KV group）；
   本仓库只有一个 KV group、没有规格类型树，块大小/精度由提议者从配置里带进来
   （`cache_block_size` / `torch_dtype`，见 `config.extract_hidden_states_hf_config`）。
"""

import torch
from torch import nn

from ..attention.forward_context import get_forward_context
from .qwen3 import maybe_prefix
from .registry import register_model

# 上游同款：padding 行（物理存在但无效）的槽位哨兵。写缓存前重定向到 0 号块（见 basic_cache）
PADDING_SLOT_ID = -1

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def basic_cache(to_cache: torch.Tensor,     # [seq_len, num_heads, head_size]
                kv_cache: torch.Tensor,     # [num_blocks, block_size, num_heads, head_size]
                slot_mapping: torch.Tensor) -> None:
    """散射写缓存。**逐行照抄上游**（包括那条 clamp）：

    padding 槽位是 `-1`，直接拿去索引会写到最后一块的最后一个槽位（把别人的数据搞脏）；
    上游把它 `clamp_min(0)` 重定向到 **0 号块**——那个块永不分配给请求（块池从 0 号块开始留白），
    于是这次写入落在"垃圾桶"里，既不用分支、也不用同步。

    **本仓库的前提**：块池的 0 号块是**会**分配给真实请求的（`core/block_pool.py` 没有"留白
    0 号块"的约定），而 target 的 `slot_mapping` 目前从不产生 `-1`（越界会当场报错，见
    `BlockTable.compute_slot_mapping`），所以这条 clamp 现在不会碰到真实数据。**69 关**引入
    CUDA Graph 的 padding 行时，必须同时把 0 号块留白——否则 padding 行会覆盖真实请求的特征。
    """
    block_size = kv_cache.shape[1]
    slot_mapping = slot_mapping.clamp_min(0)
    kv_cache[slot_mapping // block_size, slot_mapping % block_size] = to_cache


class CacheOnlyAttentionBackend:
    """cache-only 后端的契约（上游 `CacheOnlyAttentionBackend(AttentionBackend)`）。

    本仓库的 attention 没有后端注册表（`attention/backends/torch_sdpa.py` 直接就是一个 impl 类），
    所以这里只保留**真的会被调用**的三件事：缓存形状、impl 类、metadata builder 类。
    上游还回答 `get_name()` / `supports_attn_type()` / `use_cascade_attention()` /
    `get_supported_head_sizes()`（注册表、级联注意力、多头尺寸校验），本仓库没有对应机制。
    """

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int,
                           head_size: int, cache_dtype_str: str = "auto") -> tuple[int, ...]:
        """**这一关的核心公式**。上游注释：我们把 `num_kv_heads <- num_hidden_layers`、
        `head_size <- hidden_size`，并且**没有 k/v 那一维**——所以每个 token 存的是
        "L 个辅助层的 H 维特征"，而不是 K/V 两份。
        """
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_impl_cls():
        return CacheOnlyAttentionImpl

    @staticmethod
    def get_builder_cls():
        return CacheOnlyAttentionMetadataBuilder


class CacheOnlyAttentionMetadata:
    """cache-only 层的元数据。上游只有一个字段（`slot_mapping`）——因为它不算注意力，
    不需要 query/seq/块表：**写哪几个槽位就是它的全部输入**。
    """

    def __init__(self, slot_mapping: torch.Tensor) -> None:
        self.slot_mapping = slot_mapping


class CacheOnlyAttentionMetadataBuilder:
    """从本轮的 attention 元数据建 cache-only 元数据。

    上游签名是 `build(common_prefix_len, common_attn_metadata, fast_build=False)`，并且会挡掉
    两件本仓库不存在的情况：级联注意力（`common_prefix_len > 0` → NotImplementedError）与
    非 causal 注意力。本仓库的元数据里只有 `slot_mapping` 是这一层要的，所以构建就是"取出槽位"
    （64 关需求 §2：各 cache-only 层通过层名对应 attention metadata）。
    """

    def build(self, slot_mapping: torch.Tensor) -> CacheOnlyAttentionMetadata:
        return CacheOnlyAttentionMetadata(slot_mapping=slot_mapping)


class CacheOnlyAttentionImpl:
    """只写缓存的"注意力实现"。

    上游 `forward()` 是空的（`pass`）；这里**改成报错**——"算了但没写缓存"在特征提取里是静默
    丢数据，比直接失败难查得多。真正的写缓存入口是 `do_kv_cache_update()`。
    """

    def __init__(self, num_heads: int, head_size: int) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_queries_per_kv = 1

    def do_kv_cache_update(self, layer, to_cache: torch.Tensor, kv_cache: torch.Tensor,
                           slot_mapping: torch.Tensor) -> None:
        if to_cache.dtype != kv_cache.dtype:
            raise ValueError(
                f"要缓存的张量 dtype={to_cache.dtype} 与缓存 dtype={kv_cache.dtype} 不一致："
                f"特征与 KV 缓存必须同精度（静默转换会在读出来时变成另一种数值）")
        basic_cache(to_cache, kv_cache, slot_mapping)

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "CACHE_ONLY_ATTN 不算 attention：特征由 do_kv_cache_update() 写进缓存，"
            "forward() 不该被调用")


class CacheOnlyAttentionLayer(nn.Module):
    """cache-only 层：模型里的"边界"（上游 `CacheOnlyAttentionLayer(nn.Module, AttentionLayerBase)`）。

    它是 `nn.Module`（和 `attention/layer.py::Attention` 一样）：没有参数，但要能被
    `named_modules()` 看见——提议者靠遍历模型收集层名，层名就是 forward 上下文的键。
    `kv_cache` 由提议者在分配时绑定（模型自己不知道块表与容量）。
    """

    def __init__(self, num_heads: int, head_size: int, block_size: int,
                 kv_cache_torch_dtype: torch.dtype, layer_name: str = "") -> None:
        super().__init__()
        self.layer_name = layer_name
        # 注意这两维的语义：num_heads = 辅助层数 L，head_size = hidden_size H
        self.num_heads = num_heads
        self.head_size = head_size
        self.block_size = block_size
        self.kv_cache_torch_dtype = kv_cache_torch_dtype
        self.impl = CacheOnlyAttentionBackend.get_impl_cls()(num_heads, head_size)
        # 由提议者在 KV 分配之后绑定：`[num_blocks, block_size, L, H]`
        self.kv_cache: torch.Tensor | None = None

    def forward(self, to_cache: torch.Tensor) -> torch.Tensor:
        """把 `[num_tokens, L, H]` 的特征写进缓存，返回一个空张量（上游同样返回空张量，
        调用方不看它的值）。"""
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata.get(self.layer_name)
        if attn_metadata is None:
            raise KeyError(
                f"forward 上下文里没有层 {self.layer_name!r} 的 metadata：提议者必须为每个 "
                f"cache-only 层都建一份（键是层名）")
        if self.kv_cache is None:
            raise RuntimeError(
                f"层 {self.layer_name!r} 的 kv_cache 还没绑定（提议者在 load_model 时分配）")
        self.impl.do_kv_cache_update(self, to_cache, self.kv_cache, attn_metadata.slot_mapping)
        return torch.empty(0, device=to_cache.device, dtype=to_cache.dtype)

    def get_kv_cache_shape(self, num_blocks: int) -> tuple[int, ...]:
        """本层缓存的形状。上游同名信息来自 `get_kv_cache_spec()`（`HiddenStateCacheSpec`），
        本仓库没有 KV 规格类型树，所以由层自己给出形状公式（提议者分配时调它）。
        """
        return CacheOnlyAttentionBackend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_heads, self.head_size)


@register_model("ExtractHiddenStatesModel")
class ExtractHiddenStatesModel(nn.Module):
    """只持有一个 cache-only 层的"模型"（上游 `ExtractHiddenStatesModel`）。

    层名是 `cache_only_layers.{target 的层数}`：上游这么起名是因为它要给"每个 target 层"
    都留一个位置、而实际只用最后那个（层数本身只是让名字与 target 对得上）。
    """

    def __init__(self, config: dict, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.hidden_size = int(config["hidden_size"])
        self.target_num_hidden_layers = int(config["num_hidden_layers"])
        self.num_hidden_states = len(config.get("eagle_aux_hidden_state_layer_ids") or [])
        if self.num_hidden_states <= 0:
            raise ValueError(
                "cache-only 模型需要 eagle_aux_hidden_state_layer_ids（要存哪几层特征）；"
                "空列表会让缓存的 head 维变成 0——那不是'存不下'，而是'没说要存什么'")
        cache_dtype = _DTYPES.get(str(config.get("torch_dtype")))
        if cache_dtype is None:
            raise ValueError(f"不支持的 torch_dtype={config.get('torch_dtype')!r}；"
                             f"支持 {sorted(_DTYPES)}")
        self.cache_only_layers = nn.ModuleDict({
            str(self.target_num_hidden_layers): CacheOnlyAttentionLayer(
                # num_heads <- 辅助层数 L，head_size <- hidden_size H（见 basic_cache 的说明）
                num_heads=self.num_hidden_states,
                head_size=self.hidden_size,
                block_size=int(config["cache_block_size"]),
                kv_cache_torch_dtype=cache_dtype,
                layer_name=maybe_prefix(
                    prefix, f"cache_only_layers.{self.target_num_hidden_layers}"),
            )
        })

    def forward(self, hidden_states: torch.Tensor) -> None:
        """`hidden_states: [num_tokens, L, H]`（本轮 target 每个 query 行的辅助层特征）。

        返回值不用（上游同样把层输出丢掉）：唯一的效果是**写缓存**这个副作用。
        """
        self.cache_only_layers[str(self.target_num_hidden_layers)](hidden_states)

    def load_weights(self, weights) -> set[str]:
        """**没有参数要装**（上游同样 `return set()`）。

        这里**故意不迭代** `weights`：本仓库的权重迭代器是惰性的（`iter_weights()` 是生成器，
        不取就不会去读 safetensors），派生配置里的模型目录是 target 的目录，迭代它等于白读几 GB
        权重。上游也是把这份迭代器原样丢掉。
        """
        return set()
