"""注意力后端的能力声明（对应 vLLM `v1/attention/backend.py` 里的 `AttentionCGSupport`）。

**为什么需要它**：CUDA Graph 能不能用，不取决于引擎想不想用，而取决于**注意力后端**：
注意力是唯一"形状由每请求 query 长度决定"的算子。图分派器需要知道这个后端支持到哪一档，
才能决定最终模式（上游在 `CudagraphDispatcher.initialize_cudagraph_keys()` 之前调用
`CompilationConfig.resolve_cudagraph_mode_and_sizes(min_cg_support, ...)`，多个 KV cache
group 时取**最保守**的那个）。

四档（逐字对齐上游 `v1/attention/backend.py:647-661`）：

    ALWAYS                     任意批都能进图（含混合 prefill/decode）
    UNIFORM_BATCH              只有"所有请求 query 长度相同"的批能进图（投机即 1+K）
    UNIFORM_SINGLE_TOKEN_DECODE 只有"query 长度恰好 1"的纯 decode 批能进图
    NEVER                      完全不能进图

**本仓库的 Torch SDPA 后端是 `UNIFORM_BATCH`**，理由是可验证的、不是保守估计：
`TorchAttentionImpl._forward_padded()` 是为"每请求恰好 `1 + K` 行"写的固定形状路径
（形状全来自 Python int，只有张量内容每轮变）；混合批只能走 `_forward_generic()`，
那里有 `int(张量)` 这样的 CPU 同步与逐请求分支，录进图之后重放会读到**录制时**的数字
（`docs/step69_alignment.md` §2/§6 记的就是这条）。所以：

    FULL（混合批也建全图）        → 必须 ALWAYS → 本仓库不支持，会被降级到 FULL_AND_PIECEWISE
    FULL_DECODE_ONLY / PIECEWISE  → UNIFORM_BATCH 就够（混合批走 eager / 走分段图）
"""

import enum


class AttentionCGSupport(enum.Enum):
    """注意力后端的 CUDA Graph 支持档位（上游同名枚举，值必须可比大小）。"""

    ALWAYS = 3
    """任意批都能进图（支持混合 prefill/decode）。"""

    UNIFORM_BATCH = 2
    """只有"所有请求 query 长度相同"的批能进图（投机解码即每请求 `1 + K` 行）。"""

    UNIFORM_SINGLE_TOKEN_DECODE = 1
    """只有"query 长度恰好 1"的纯 decode 批能进图。"""

    NEVER = 0
    """完全不能进图。"""


def min_cudagraph_support(supports) -> AttentionCGSupport:
    """多个 KV cache group 时取**最保守**的一档（上游 `_check_and_update_cudagraph_mode` 同款）。

    一个模型可以有多组 KV cache（例如滑动窗口 + 全注意力，或 Mamba + 注意力混合），
    每组由不同后端服务。只要有一组不能进图，整批就不能进图——图是整段前向录下来的，
    不能"这半张图能用、那半张不能用"。
    """
    supports = list(supports)
    if not supports:
        raise ValueError("没有注意力后端能力可决议：至少要有一个 KV cache group")
    return min(supports, key=lambda support: support.value)
