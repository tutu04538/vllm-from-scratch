"""平台能力探测（对应 vLLM `vllm/utils/platform_utils.py` 与 `platforms/*` 的最小子集）。

73 关首次需要"**这台机器能不能用 pinned memory / UVA**"这个答案：
V2 Runner 的常驻状态（`RequestState.all_token_ids` 等）走 **UVA** 而不是全搬 GPU
（`worker/gpu/buffer_utils.py`），而 UVA 的前提是 pinned memory 可用。

上游的判定链（逐条照抄，含 WSL 那一段）：
    is_uva_available()        = is_pin_memory_available() or 平台是 CPU
    CudaPlatform.is_pin_memory_available():
        WSL 下：内核版本 < 4.19.121 → False（WSL2 早期内核没有 pinned 支持）
                否则读环境变量 `VLLM_WSL2_ENABLE_PIN_MEMORY`（**默认关**）
        非 WSL：True

本机（WSL2 + RTX 5090 Laptop）默认走 `False` 分支，所以跑 V2/CUDA 的路径要显式
`VLLM_WSL2_ENABLE_PIN_MEMORY=1`——这与上游引擎在本机的行为完全一致（不是本仓库的额外要求）。
"""

import functools
import platform
import warnings

import torch


def in_wsl() -> bool:
    """是否在 WSL 里（上游 `platforms/interface.py:62` 同款判据）。"""
    return "microsoft" in " ".join(platform.uname()).lower()


def _get_wsl_kernel_version() -> tuple[int, ...] | None:
    """WSL2 内核版本（`platform.uname().release` 形如 `5.15.167.4-microsoft-standard-WSL2`）。"""
    try:
        release = platform.uname().release
        parts = release.split("-")[0].split(".")
        return tuple(int(x) for x in parts[:3])
    except Exception:
        return None


def is_pin_memory_available() -> bool:
    """能否用 pinned memory（上游 `CudaPlatform.is_pin_memory_available` 的等价物）。

    没有 CUDA（纯 CPU 环境）时上游走 CpuPlatform 的 `True` 分支。
    """
    if not torch.cuda.is_available():
        return True
    if in_wsl():
        version = _get_wsl_kernel_version()
        if version is None or version < (4, 19, 121):
            warnings.warn(
                "检测到 WSL 且 WSL2 内核版本低于 4.19.121：pinned memory 不可用"
                "（与上游 vLLM 的判定一致）。"
            )
            return False
        import os

        return bool(os.environ.get("VLLM_WSL2_ENABLE_PIN_MEMORY", ""))
    return True


@functools.cache
def is_uva_available() -> bool:
    """UVA 是否可用（上游 `utils/platform_utils.py:51`：UVA 需要 pinned memory）。"""
    return is_pin_memory_available() or not torch.cuda.is_available()
