"""pytest 的会话级夹具：tiny 模型现场生成（仓库不提交权重）。"""

import sys
from pathlib import Path

import pytest

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
