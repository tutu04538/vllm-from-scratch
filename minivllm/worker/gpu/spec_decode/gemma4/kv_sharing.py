"""Gemma4 assistant 的 Q-only KV 共享映射（80D）。

对应冻结的
`v1/worker/gpu/spec_decode/gemma4/speculator.py::Gemma4Speculator._setup_gemma4_kv_sharing`。

Q-only assistant 的每一层只算 Q，K/V 直接读 target 的 cache。**共享错层通常不会报 shape
错误**——多数层的 K/V 形状相同，读错上下文只会让草稿质量变差，最终文本仍可能被 target
的拒绝采样纠正。所以这里把"源层怎么选"做成纯逻辑，并由执行侧再挡一次未接线的层。

冻结规则：

    num_non_shared = len(target_layer_types) - target_num_kv_shared_layers
    只在 target 的前 num_non_shared 层里按 layer type 建候选
    draft 每层选**同类型最后一个**非共享源层（candidates[-1]）
    draft 类型缺失时默认 full_attention
    目标层名保留 target 的真实前缀，例如 language_model.model.layers.2.self_attn.attn

`num_kv_shared_layers` 数的是 target **尾部复用别人 KV 的层**（消费者），它们不拥有
cache，不能当源。80D 只消费这个字段做选源；target 内部的跨层共享执行仍属后续阶段。
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_TARGET_LAYER_PREFIX = "model.layers"
DEFAULT_DRAFT_LAYER_TYPE = "full_attention"


@dataclass(frozen=True)
class Gemma4KVSharing:
    """一个 draft 层的选源结果（`target_idx=None` 表示没找到同类型源层）。"""

    draft_idx: int
    draft_layer_type: str
    target_idx: int | None
    target_layer_name: str | None
    reason: str | None = None


def _get(config: Any, key: str, default: Any = None) -> Any:
    """同时支持 dict 与配置对象（上游 hf_config 是对象，本仓库常传 dict）。"""
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _layer_types(config: Any) -> list[str]:
    value = _get(config, "layer_types", [])
    if value is None:
        return []
    return [str(item) for item in value]


def target_layer_prefix(target_attn_layer_names: Collection[str]) -> str:
    """从 target 的真实注意力层名推出 `...layers` 前缀。

    冻结实现遍历 `target_attn_layer_names`，找第一个含 `.layers.` 的名字并截断。
    这里排序后遍历，保证同一组输入得到确定结果；没有可用名字时退回 `model.layers`
    （冻结实现的默认值）。写死 `model.layers` 会在 `language_model.model.layers...`
    这类前缀上绑错层。
    """
    for name in sorted(target_attn_layer_names):
        if ".layers." in name:
            return name.split(".layers.")[0] + ".layers"
    return DEFAULT_TARGET_LAYER_PREFIX


def select_gemma4_kv_sharing_targets(
    target_layer_types: Sequence[str],
    target_num_kv_shared_layers: int,
    draft_layer_types: Sequence[str],
    num_draft_layers: int,
    *,
    target_prefix: str = DEFAULT_TARGET_LAYER_PREFIX,
) -> list[Gemma4KVSharing]:
    """纯映射：返回每个 draft 层的源层决策，不碰模型、不碰 cache。

    非法配置（共享层数为负 / 超过 target 层数）在这里明确报错，而不是像切片那样
    静默退化成"少看几层"。真实合法配置不会触发这两条。
    """
    target_types = [str(item) for item in target_layer_types]
    draft_types = [str(item) for item in draft_layer_types]
    if num_draft_layers < 0:
        raise ValueError(f"num_draft_layers 不能为负：{num_draft_layers}")

    try:
        num_shared = int(target_num_kv_shared_layers)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "target_num_kv_shared_layers 必须是整数，"
            f"收到 {target_num_kv_shared_layers!r}"
        ) from exc
    if num_shared < 0:
        raise ValueError(f"num_kv_shared_layers 不能为负：{num_shared}")
    if num_shared > len(target_types):
        raise ValueError(
            f"num_kv_shared_layers={num_shared} 超过 target 层数 {len(target_types)}"
        )

    num_non_shared = len(target_types) - num_shared
    type_to_target_indices: dict[str, list[int]] = defaultdict(list)
    for idx, layer_type in enumerate(target_types[:num_non_shared]):
        type_to_target_indices[layer_type].append(idx)

    decisions: list[Gemma4KVSharing] = []
    for draft_idx in range(num_draft_layers):
        draft_type = (
            draft_types[draft_idx]
            if draft_idx < len(draft_types)
            else DEFAULT_DRAFT_LAYER_TYPE
        )
        candidates = type_to_target_indices.get(draft_type, [])
        if not candidates:
            decisions.append(Gemma4KVSharing(
                draft_idx=draft_idx,
                draft_layer_type=draft_type,
                target_idx=None,
                target_layer_name=None,
                reason=(
                    f"target 的前 {num_non_shared} 个非共享层里没有类型 "
                    f"{draft_type!r}"
                ),
            ))
            continue
        target_idx = candidates[-1]
        decisions.append(Gemma4KVSharing(
            draft_idx=draft_idx,
            draft_layer_type=draft_type,
            target_idx=target_idx,
            target_layer_name=(
                f"{target_prefix}.{target_idx}.self_attn.attn"
            ),
        ))
    return decisions


def setup_gemma4_kv_sharing(
    draft_model: nn.Module,
    target_attn_layer_names: Collection[str],
    draft_text_config: Any,
    target_text_config: Any,
) -> list[Gemma4KVSharing]:
    """在 draft 模型上建立 Q-only 共享关系（对应冻结 `_setup_gemma4_kv_sharing`）。

    只写两个属性：

        attn.q_only = True
        attn.kv_sharing_target_layer_name = 源层名 / None

    物理 cache 的绑定由 Runner 在分配阶段做（`Attention.bind_shared_kv_cache`），
    这里不分配、不复制。找不到同类型源层时**保留未接线状态**并告警；Runner 会拒绝
    这种层，而不是让它退回"自己分配一份 KV"。
    """
    if not (hasattr(draft_model, "model")
            and hasattr(draft_model.model, "layers")):
        return []

    draft_layers = list(draft_model.model.layers)
    decisions = select_gemma4_kv_sharing_targets(
        target_layer_types=_layer_types(target_text_config),
        target_num_kv_shared_layers=_get(
            target_text_config, "num_kv_shared_layers", 0) or 0,
        draft_layer_types=_layer_types(draft_text_config),
        num_draft_layers=len(draft_layers),
        target_prefix=target_layer_prefix(target_attn_layer_names),
    )

    mapped: list[Gemma4KVSharing] = []
    for draft_idx, layer in enumerate(draft_layers):
        self_attn = getattr(layer, "self_attn", None)
        attn = getattr(self_attn, "attn", None)
        if attn is None:
            continue
        decision = decisions[draft_idx]
        attn.q_only = True
        attn.kv_sharing_target_layer_name = decision.target_layer_name
        mapped.append(decision)
        if decision.target_layer_name is None:
            logger.warning(
                "Gemma4 MTP: draft layer %d (%s) 没有可用的 target 源层（%s）；"
                "保持未接线，执行侧会拒绝",
                draft_idx,
                decision.draft_layer_type,
                decision.reason,
            )
    return mapped
