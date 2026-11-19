"""115 CPU 配置解析与未开放执行边界。"""

import pytest
import torch

from minivllm.config import ModelConfig
from minivllm.layers.quantization.config import (
    AutoAWQConfig, AutoGPTQConfig, Fp8Config, resolve_quantization_config,
)
from minivllm.model_loader.base_loader import BaseModelLoader


AWQ = {"quant_method": "awq", "w_bit": 4, "q_group_size": 128, "zero_point": True}
GPTQ = {"quant_method": "gptq", "bits": 4, "group_size": 128, "desc_act": False, "sym": True}
FP8 = {"quant_method": "fp8", "activation_scheme": "dynamic"}


@pytest.mark.parametrize("alias", ["awq", "awq_marlin", "auto_awq"])
def test_awq_alias(alias):
    config = resolve_quantization_config(alias, AWQ)
    assert isinstance(config, AutoAWQConfig)
    assert (config.weight_bits, config.group_size, config.zero_point) == (4, 128, True)
    assert config.get_supported_act_dtypes() == [torch.float16, torch.bfloat16]
    assert config.get_min_capability() == 75


@pytest.mark.parametrize("alias", ["gptq", "gptq_marlin", "auto_gptq"])
def test_gptq_alias(alias):
    config = resolve_quantization_config(alias, GPTQ)
    assert isinstance(config, AutoGPTQConfig)
    assert (config.weight_bits, config.group_size, config.desc_act, config.is_sym) == (4, 128, False, True)
    assert config.get_min_capability() == 60


def test_fp8_and_external_origin():
    config = resolve_quantization_config(None, None, FP8)
    assert isinstance(config, Fp8Config)
    assert config.is_checkpoint_fp8_serialized and config.activation_scheme == "dynamic"
    assert config.source == "external"


def test_conflict_and_missing():
    with pytest.raises(ValueError, match="冲突"):
        resolve_quantization_config("gptq", AWQ)
    with pytest.raises(ValueError, match="冲突"):
        resolve_quantization_config(None, AWQ, GPTQ)
    with pytest.raises(ValueError, match="必填"):
        resolve_quantization_config(None, {"quant_method": "awq", "w_bit": 4})
    with pytest.raises(NotImplementedError, match="int8"):
        resolve_quantization_config(None, {"quant_method": "int8"})


def test_unquantized_and_loader_rejects_quantized():
    assert resolve_quantization_config(None, None) is None
    config = ModelConfig(model="unused", hf_config={"quantization_config": AWQ})
    with pytest.raises(NotImplementedError, match="auto_awq"):
        BaseModelLoader()._initialize_model(config, "cpu")


@pytest.mark.parametrize("config", [
    {**AWQ, "w_bit": 8},
    {**AWQ, "q_group_size": 64},
    {**AWQ, "lm_head": True},
    {**GPTQ, "desc_act": True},
    {**GPTQ, "sym": False},
    {**GPTQ, "dynamic": {"x": {"bits": 8}}},
    {**GPTQ, "checkpoint_format": "unknown"},
    {**FP8, "activation_scheme": "static"},
    {**FP8, "weight_block_size": [128, 128]},
])
def test_out_of_scope_config_rejected(config):
    with pytest.raises(NotImplementedError):
        resolve_quantization_config(None, config)
