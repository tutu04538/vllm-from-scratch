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


def test_external_file_method_is_inferred_only_from_explicit_user_method(tmp_path):
    import json
    config = {key: value for key, value in AWQ.items() if key != 'quant_method'}
    (tmp_path / 'quant_config.json').write_text(json.dumps(config))
    model_config = ModelConfig(model=str(tmp_path), hf_config={}, quantization='awq_marlin')
    with pytest.raises(NotImplementedError, match='auto_awq.*external'):
        BaseModelLoader()._initialize_model(model_config, 'cpu')
    with pytest.raises(ValueError, match='必填'):
        resolve_quantization_config(None, None, config)


def test_checkpoint_source_takes_precedence_over_external_file(tmp_path):
    import json
    (tmp_path / 'quant_config.json').write_text(json.dumps(GPTQ))
    model_config = ModelConfig(model=str(tmp_path), hf_config={'quantization_config': AWQ},
                               quantization='awq')
    with pytest.raises(NotImplementedError, match='auto_awq.*checkpoint'):
        BaseModelLoader()._initialize_model(model_config, 'cpu')


def test_multiple_external_files_and_user_checkpoint_conflict(tmp_path):
    import json
    for name in ('quant_config.json', 'quantize_config.json'):
        (tmp_path / name).write_text(json.dumps(AWQ))
    with pytest.raises(ValueError, match='多个'):
        BaseModelLoader()._initialize_model(
            ModelConfig(model=str(tmp_path), hf_config={}, quantization='awq'), 'cpu')
    with pytest.raises(ValueError, match='冲突'):
        BaseModelLoader()._initialize_model(
            ModelConfig(model=str(tmp_path), hf_config={'quantization_config': AWQ},
                        quantization='gptq'), 'cpu')


def test_fp8_supported_dtype_order_matches_frozen_contract():
    assert Fp8Config.get_supported_act_dtypes() == [torch.bfloat16, torch.float16]
