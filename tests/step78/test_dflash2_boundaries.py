"""step78：DFlash2 的配置/分派边界（不静默降级）。

每条边界都对应一个"能跑但悄悄错"的风险：

- V1 的 `DFlashProposer` 不调用候选选择器 → 同一个 checkpoint 会静默退化成 DFlash1
- 缺 `selector_*` / `conv_*` 字段 → 模型构造时才 `KeyError`，且不知道是哪份 checkpoint
- 缩减词表（`d2t`）→ 候选 id 与 q 的词表宽度不一致（上游这条路径没做散射）
- `conv_group_size` 不整除 hidden → `unflatten` 运行期崩
- 非正 `output_multiplier` → `get_top_k_tokens` 的 top-k 归约不成立
- 方法仍是 `dflash`：分派依据是 **architecture**，不是目录名/方法名
"""

import copy
import json
import os
from pathlib import Path

import pytest
import torch

from dflash2_helpers import DEVICE, make_spec_config, tiny_pair, v2_env
from minivllm import (CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig,
                      SpeculativeConfig, VllmConfig)
from minivllm.testing.tiny_models import tiny_dflash_dir, tiny_dspark_dir


def _vllm_config(spec: SpeculativeConfig) -> VllmConfig:
    target_dir, target_hf, _, _ = tiny_pair()
    return VllmConfig(
        model_config=ModelConfig(model=target_dir, dtype="float32", max_model_len=64,
                                 hf_config=target_hf),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=64),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=64),
        device_config=DeviceConfig(device=DEVICE), speculative_config=spec)


def _draft_hf(**kwargs) -> dict:
    _, _, _, draft_hf = tiny_pair(**kwargs)
    return copy.deepcopy(draft_hf)


def _model_config(hf: dict, directory: str) -> ModelConfig:
    return ModelConfig(model=directory, dtype="float32", max_model_len=64, hf_config=hf)


def test_config_accepts_dflash2_and_writes_k_into_the_draft_config():
    spec = make_spec_config(k=2)
    assert spec.method == "dflash"
    assert spec.uses_dflash2() is True
    assert spec.parallel_drafting is True
    # 模型构造只拿得到 hf_config；`block_size = 1 + K` 靠配置期写回
    assert spec.draft_model_config.hf_config["num_speculative_tokens"] == 2


def test_v1_is_rejected_loudly():
    """上游为 DFlash2 强制 V2；本仓库默认 V1，所以必须在配置期报错。"""
    spec = make_spec_config(k=2)
    previous = os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
    try:
        with pytest.raises(NotImplementedError, match="DFlash2 只有 V2"):
            _vllm_config(spec)
    finally:
        if previous is not None:
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = previous


@pytest.mark.parametrize("missing", ["selector_top_k", "selector_rank",
                                    "conv_kernel_size", "conv_group_size"])
def test_missing_structural_field_is_rejected(missing):
    draft_hf = _draft_hf()
    del draft_hf["dflash_config"][missing]
    _, _, draft_dir, _ = tiny_pair()
    with pytest.raises(ValueError, match=missing):
        SpeculativeConfig(method="dflash", num_speculative_tokens=3,
                          draft_model_config=_model_config(draft_hf, draft_dir))


def test_reduced_vocab_is_rejected():
    """DFlash2 + `draft_vocab_size < vocab_size`：候选 id 与 q 宽度不一致，明确拒绝。"""
    draft_hf = _draft_hf()
    draft_hf["draft_vocab_size"] = draft_hf["vocab_size"] - 2
    _, _, draft_dir, _ = tiny_pair()
    with pytest.raises(NotImplementedError, match="不支持缩减词表"):
        SpeculativeConfig(method="dflash", num_speculative_tokens=3,
                          draft_model_config=_model_config(draft_hf, draft_dir))


def test_conv_group_size_must_divide_hidden_size():
    draft_hf = _draft_hf()
    draft_hf["dflash_config"]["conv_group_size"] = 7  # 32 % 7 != 0
    _, _, draft_dir, _ = tiny_pair()
    with pytest.raises(ValueError, match="必须整除 hidden_size"):
        SpeculativeConfig(method="dflash", num_speculative_tokens=3,
                          draft_model_config=_model_config(draft_hf, draft_dir))


def test_selector_top_k_must_fit_the_vocab():
    draft_hf = _draft_hf()
    draft_hf["dflash_config"]["selector_top_k"] = draft_hf["vocab_size"] + 1
    _, _, draft_dir, _ = tiny_pair()
    with pytest.raises(ValueError, match="selector_top_k"):
        SpeculativeConfig(method="dflash", num_speculative_tokens=3,
                          draft_model_config=_model_config(draft_hf, draft_dir))


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_non_positive_output_multiplier_is_rejected(value):
    draft_hf = _draft_hf()
    draft_hf["dflash_config"]["output_multiplier"] = value
    _, _, draft_dir, _ = tiny_pair()
    with pytest.raises(ValueError, match="output_multiplier"):
        SpeculativeConfig(method="dflash", num_speculative_tokens=3,
                          draft_model_config=_model_config(draft_hf, draft_dir))


def test_init_speculator_dispatches_on_architecture_not_method():
    """同一 `method="dflash"`：DFlash2 架构 → DFlash2Speculator；DFlash 架构 → DFlashSpeculator。"""
    from minivllm.worker.gpu.spec_decode import init_speculator
    from minivllm.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from minivllm.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

    dflash2_spec = make_spec_config(k=2)
    with v2_env():
        speculator = init_speculator(_vllm_config(dflash2_spec), torch.device(DEVICE))
    assert isinstance(speculator, DFlash2Speculator)

    # 普通 DFlash：同一方法名、不同 architecture
    draft_dir = tiny_dflash_dir("tiny_gqa")
    draft_hf = json.loads((Path(draft_dir) / "config.json").read_text())
    plain = SpeculativeConfig(
        method="dflash", num_speculative_tokens=2,
        draft_model_config=ModelConfig(model=draft_dir, dtype="float32",
                                       max_model_len=64, hf_config=draft_hf))
    assert plain.uses_dflash2() is False
    with v2_env():
        speculator = init_speculator(_vllm_config(plain), torch.device(DEVICE))
    assert isinstance(speculator, DFlashSpeculator)
    assert not isinstance(speculator, DFlash2Speculator)


def test_dspark_is_not_mistaken_for_dflash2():
    dspark_dir = tiny_dspark_dir("tiny_gqa")
    dspark_hf = json.loads((Path(dspark_dir) / "config.json").read_text())
    spec = SpeculativeConfig(
        method="dspark", num_speculative_tokens=2,
        draft_model_config=ModelConfig(model=dspark_dir, dtype="float32",
                                       max_model_len=64, hf_config=dspark_hf))
    assert spec.uses_dflash2() is False


def test_model_requires_k_for_the_conv_block_size():
    """模型构造只拿 hf_config：没有 K 就建不出 `block_size=1+K`，必须当场报错。"""
    from minivllm.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM

    draft_hf = _draft_hf()
    draft_hf.pop("num_speculative_tokens", None)
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        DFlash2Qwen3ForCausalLM(draft_hf)


def test_unknown_dflash_architecture_fails_at_load():
    """注册表里没有的 architecture 仍然明确报错（不会回退到 DFlash1 的类）。"""
    from minivllm.model_loader import get_model

    draft_hf = _draft_hf()
    draft_hf["architectures"] = ["DFlash3DraftModel"]
    _, _, draft_dir, _ = tiny_pair()
    with pytest.raises(ValueError, match="没有注册的模型结构"):
        get_model(_model_config(draft_hf, draft_dir), "cpu")
