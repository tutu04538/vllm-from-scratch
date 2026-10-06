"""step68 的会话级夹具：tiny 模型（普通 / 带 JSON 词表）+ 设备门槛。

和 59 关同一条约定：**投机验证只在 CUDA 上可用**（拒绝采样是 Triton 内核），
所以"投机 + 语法/ logprobs"的端到端用例走 `cuda_device` 门槛；纯 CPU 的语义与逐值差分
用 CPU 张量直接调类/函数（上游类在 CPU 上同样能跑）。
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minivllm.testing.tiny_models import (  # noqa: E402
    STRUCTURED_TOKENS,
    tiny_hetero_pair,
    tiny_qwen3_config,
    tiny_qwen3_dir,
    tiny_structured_dir,
)


@pytest.fixture(scope="session")
def tiny_dir():
    return tiny_qwen3_dir("tiny_gqa")


@pytest.fixture(scope="session")
def hf_config():
    return tiny_qwen3_config("tiny_gqa")


@pytest.fixture(scope="session")
def structured_dir():
    """tiny 模型 + JSON 记号的 tokenizer（68 关结构性用例的基底）。"""
    return tiny_structured_dir("tiny_mqa")


@pytest.fixture(scope="session")
def structured_config():
    from minivllm.config import ModelConfig

    path = tiny_structured_dir("tiny_mqa")
    import json

    hf_config = json.loads(Path(path, "config.json").read_text())
    return ModelConfig(model=path, dtype="float32", max_model_len=64, hf_config=hf_config)


@pytest.fixture(scope="session")
def structured_hf_config(structured_dir):
    """tiny 结构化模型的 HF config（`eos_token_id` 已被改成 tokenizer 的 `<eos>`=1）。"""
    import json

    return json.loads(Path(structured_dir, "config.json").read_text())


@pytest.fixture(scope="session")
def structured_tokenizer(structured_dir):
    from minivllm.tokenizer_utils import load_tokenizer

    return load_tokenizer(structured_dir)


@pytest.fixture(scope="session")
def hetero_tokenizers():
    """67 关的异构词表对：用来验证"文法掩码只换 id、不改行"在 TLI 下也成立。"""
    target_dir, draft_dir, info = tiny_hetero_pair()
    from minivllm.tokenizer_utils import load_tokenizer

    return load_tokenizer(target_dir), load_tokenizer(draft_dir), info


@pytest.fixture(scope="session")
def structured_tokens():
    return dict(STRUCTURED_TOKENS)


@pytest.fixture(scope="session")
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("投机验证内核需要 CUDA（纯 CPU 语义由 CPU 用例覆盖）")
    return "cuda"
