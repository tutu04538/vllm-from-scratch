"""step60 的会话级夹具：tiny 模型现场生成 + CUDA 门槛（投机验证走 Triton 内核）。"""

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


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """ngram_gpu 的实现只用 torch 张量运算，所以 CPU 与 CUDA 两套都要跑。

    （上游那版要 CUDA + torch.compile；本机没有编译依赖，语义两条路径相同。）
    """
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("本机没有 CUDA：跳过 CUDA 参数（CPU 参数仍会实跑）")
    return request.param
