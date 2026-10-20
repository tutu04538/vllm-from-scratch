"""V2 注意力组的公共小工具（对应上游 `v1/worker/gpu/attn_utils.py` 的两个入口）。

79 关只需要其中两件：

    AttentionCGSupportInfo                    多组后端能力取最保守一档
    get_query_lens_mismatch_unsupported_backend
                                              哪些后端"规划时依赖 CPU query 长度"

**为什么自适应验证需要第二个检查**：GPU 上按置信度裁剪后，某些请求只验证 `C_i < K_i`
个草稿，device 上的 `query_start_loc` 与 CPU 侧按"预算均分"算出的上界不再一致。如果某个
注意力后端在 CPU 侧按 query 长度做了规划（例如 SSM 的 recurrent-state 分块），它就会按错
的边界跑。上游的判据是 `AttentionBackend.supports_device_cpu_query_lens_mismatch()`；
本仓库只有 Torch SDPA 后端，它在 `_forward_generic()` 里逐请求读 **device** 的
`query_start_loc`/`seq_lens`，所以明确返回 True。

**本仓库与上游的一处能力差异**（详见 `docs/step79_alignment.md` §5）：上游要求
`AttentionCGSupport.ALWAYS`，因为它的 target 会捕获 **varlen decode CUDA graph**。本仓库
V2 的 target 图尚未实现（74 关边界），自适应验证走 eager 的逐请求路径；因此 runner 只在
"target 真的会捕获 varlen 图"时才要求 ALWAYS。本文件保留上游默认（`require_varlen_cudagraph=True`），
所以直接调用 `maybe_create_adaptive_verification_manager()` 的语义与上游一致。
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...attention.backend import AttentionCGSupport

if TYPE_CHECKING:
    from ..utils import AttentionGroup


@dataclass(frozen=True)
class AttentionCGSupportInfo:
    """所有注意力组里最保守的图能力（上游同名 dataclass）。"""

    min_cg_support: AttentionCGSupport = AttentionCGSupport.ALWAYS
    min_cg_attn_backend: str | None = None

    def narrow(self, support: AttentionCGSupport, backend: str | None) -> "AttentionCGSupportInfo":
        """如果 `support` 更保守，就收紧返回值（上游同名方法）。"""
        if support.value < self.min_cg_support.value:
            return AttentionCGSupportInfo(support, backend)
        return self


def get_query_lens_mismatch_unsupported_backend(
    attn_groups: list[list["AttentionGroup"]],
) -> str | None:
    """返回第一个"需要 CPU query 长度精确"的后端名；没有则 None。

    与上游逐行同义：后端在 CPU 侧按 query 长度做规划时，device 裁剪后的长度会让它切错块。
    """
    for groups in attn_groups:
        for group in groups:
            backend = group.backend
            supports = getattr(backend, "supports_device_cpu_query_lens_mismatch", None)
            if supports is None or not supports():
                return getattr(backend, "__name__", type(backend).__name__)
    return None
