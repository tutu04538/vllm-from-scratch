"""测试用的注意力替身（**只给测试**，生产路径不 import）。

74 关的验收要求"支持 / 不支持 `update_draft_decode_metadata` 的两个后端都跑一遍，断言实际
选中 fused / fallback"。本仓库的生产路径只有一个 Torch 后端（它声明支持原地更新），所以
"不支持"的那一档必须由替身来演——**但演的是 capability 声明，不是算法**：

    `NoDraftMetadataUpdateBuilder`  继承生产的 Torch 构造器，把
        `supports_draft_decode_metadata_update` 改成 False，并把
        `update_draft_decode_metadata()` **退回基类那一份**（被调用就报错）——声明"不支持"
        之后还被调用，就说明有人在伪造 capability，必须当场响。

于是被验证的仍然是生产那两处：`AutoRegressiveSpeculator._configure_fused_multi_step_decode()`
的门，与 `_multi_step_decode()` 的回退路径。测试通过替换
`AutoRegressiveSpeculator._init_attn_groups` 把这份声明接进去（与上游"换后端"同义）。
"""

from ..attention.backends.torch_sdpa import (
    TorchAttentionBackend,
    TorchAttentionMetadataBuilder,
)
from ..attention.metadata import AttentionMetadataBuilder
from ..worker.utils import AttentionGroup


class NoDraftMetadataUpdateBuilder(TorchAttentionMetadataBuilder):
    """声明"不能在草稿步之间原地更新元数据"的后端（上游 FA3 在 DCP>1 时就是这个状态）。

    它的 `update_draft_decode_metadata()` 用的是**基类**那一份——被调用就报错。这正是"不伪造
    capability"的另一半：声明 False 之后，融合路径根本不该调到它，调到了就是有人在造假。
    （不能就这么继承 Torch 后端的空操作：那样"支持"与"不支持"在行为上没有差别，
    测试也就钉不住任何东西。）
    """

    supports_draft_decode_metadata_update = False

    def update_draft_decode_metadata(self, metadata) -> None:
        AttentionMetadataBuilder.update_draft_decode_metadata(self, metadata)


class UnsupportedDraftMetadataBackend(TorchAttentionBackend):
    """与替身构造器配套的后端名（`_configure_fused_multi_step_decode()` 的日志要用它）。"""

    @staticmethod
    def get_name() -> str:
        return "TORCH_SDPA_NO_DRAFT_UPDATE"

    @staticmethod
    def get_builder_cls():
        return NoDraftMetadataUpdateBuilder


def draft_attn_groups(layer_names, block_size: int, *,
                      supports_update: bool) -> list[list[AttentionGroup]]:
    """按 `supports_update` 造一组草稿注意力组（测试的 `_init_attn_groups` 替身用）。"""
    if supports_update:
        backend = TorchAttentionBackend
        builder_cls = TorchAttentionBackend.get_builder_cls()
    else:
        backend = UnsupportedDraftMetadataBackend
        builder_cls = UnsupportedDraftMetadataBackend.get_builder_cls()
    return [[
        AttentionGroup(
            backend=backend,
            layer_names=list(layer_names),
            metadata_builders=[builder_cls(block_size)],
        )
    ]]
