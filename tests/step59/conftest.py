"""step59 的会话级夹具：tiny 模型现场生成 + CUDA 门槛（拒绝采样是 Triton 内核）。"""

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
def cuda_device():
    """验证内核只在 CUDA 上跑（上游同样只有 GPU 路径）。

    没有 CUDA 的机器上**跳过**这些用例（它们在本机一定实跑，见 docs/results.json → step59.results）；
    算法语义在 CPU 上由 `minivllm/testing/torch_rejection_sampler.py` 覆盖。
    """
    if not torch.cuda.is_available():
        pytest.skip("拒绝采样内核需要 CUDA（Torch 参考实现在 CPU 上覆盖语义）")
    return "cuda"
