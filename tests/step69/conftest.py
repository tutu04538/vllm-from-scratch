"""step69 的会话级夹具：tiny 模型 + 设备门槛。

69 关的两块交付各有自己的设备要求：

    padding 语义（行/槽位/seq_len/采样行）—— CPU 上就能测，纯张量
    CUDA Graph（捕获/重放/地址稳定/污染 padding）—— 只有 CUDA 能做（`requires_cuda`）

和 58/59 关同一条约定：**投机验证走 Triton 内核**，所以"投机 + 图"的端到端用例也走 CUDA。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from minivllm.testing.tiny_models import (  # noqa: E402
    tiny_eagle3_dir,
    tiny_qwen3_config,
    tiny_qwen3_dir,
)


@pytest.fixture(scope="module")
def tiny_dir():
    return tiny_qwen3_dir("tiny_gqa")


@pytest.fixture(scope="module")
def hf_config():
    return tiny_qwen3_config("tiny_gqa")


@pytest.fixture(scope="module")
def eagle3_dir():
    return tiny_eagle3_dir("tiny_gqa")
