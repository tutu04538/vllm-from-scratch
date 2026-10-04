"""step61 的会话级夹具：tiny 模型现场生成 + CUDA 门槛（真引擎用例走 CUDA 优先）。"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minivllm.testing.tiny_models import tiny_qwen3_config, tiny_qwen3_dir  # noqa: E402


@pytest.fixture(scope="session")
def tiny_dir():
    """tiny Qwen3 模型目录（固定 seed，进程内只生成一次）。"""
    return tiny_qwen3_dir("tiny_gqa")


@pytest.fixture(scope="session")
def hf_config():
    return tiny_qwen3_config("tiny_gqa")


@pytest.fixture(scope="session")
def device():
    """真引擎默认设备：有 CUDA 用 CUDA（与 tests/step58|59|60 同款）；没有就 CPU。"""
    return "cuda" if torch.cuda.is_available() else "cpu"
