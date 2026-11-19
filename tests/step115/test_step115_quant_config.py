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


def test_qwen_quantized_checkpoint_rejects_execution_before_loading():
    from minivllm.testing.tiny_models import tiny_qwen3_config
    hf = {**tiny_qwen3_config('tiny_gqa'), 'quantization_config': AWQ}
    with pytest.raises(NotImplementedError, match=r'量化执行尚未开放: auto_awq'):
        BaseModelLoader()._initialize_model(ModelConfig(model='unused', hf_config=hf, dtype='bfloat16'), 'cpu')


def test_excluded_linear_matches_unquantized_output():
    from minivllm.layers.linear import RowParallelLinear
    config = resolve_quantization_config(None, {**AWQ, 'modules_to_not_convert': ['o_proj']})
    layer = RowParallelLinear(4, 3, prefix='model.layers.0.self_attn.o_proj', quant_config=config)
    x = torch.arange(8, dtype=torch.float32).view(2, 4)
    with torch.no_grad():
        layer.weight.copy_(torch.arange(12, dtype=torch.float32).view(3, 4))
    output, _ = layer(x)
    torch.testing.assert_close(output, torch.nn.functional.linear(x, layer.weight))


@pytest.mark.parametrize('ignored,expected', [
    ([], False), (['q_proj', 'k_proj', 'v_proj'], True),
    (['model.layers.0.self_attn.qkv_proj'], True),
])
def test_packed_awq_skip(ignored, expected):
    from minivllm.layers.quantization.utils import is_layer_skipped
    mapping = {'qkv_proj': ['q_proj', 'k_proj', 'v_proj']}
    assert is_layer_skipped('model.layers.0.self_attn.qkv_proj', ignored, mapping,
                            match_mode='substring') is expected


def test_packed_partial_skip_rejected_before_weights():
    from minivllm.layers.linear import QKVParallelLinear
    config = resolve_quantization_config(None, {**AWQ, 'modules_to_not_convert': ['q_proj']})
    config.packed_modules_mapping.update({'qkv_proj': ['q_proj', 'k_proj', 'v_proj']})
    with pytest.raises(ValueError, match='some but not all'):
        QKVParallelLinear(8, 2, 4, prefix='model.layers.0.self_attn.qkv_proj',
                          quant_config=config)


def test_packed_skip_matches_frozen_vllm_in_separate_process():
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    from minivllm.layers.quantization.utils import is_layer_skipped
    script = Path(__file__).with_name('step115_frozen_skip_probe.py')
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                            timeout=120, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    assert result.returncode == 0, result.stderr
    oracle = json.loads(result.stdout.splitlines()[-1])
    mapping = {'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
               'gate_up_proj': ['gate_proj', 'up_proj']}
    for (prefix, ignored, mode), expected in zip(oracle['cases'], oracle['results']):
        if expected['status'] == 'error':
            with pytest.raises(ValueError, match='some but not all'):
                is_layer_skipped(prefix, ignored, mapping, match_mode=mode)
        else:
            assert is_layer_skipped(prefix, ignored, mapping, match_mode=mode) is expected['value']



def test_qwen_quantized_activation_dtype_and_sm_are_checked(monkeypatch):
    from minivllm.testing.tiny_models import tiny_qwen3_config
    hf = {**tiny_qwen3_config('tiny_gqa'), 'quantization_config': AWQ}
    with pytest.raises(ValueError, match='dtype'):
        BaseModelLoader()._initialize_model(ModelConfig(model='unused', hf_config=hf), 'cpu')
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (7, 0))
    with pytest.raises(NotImplementedError, match='SM75.*SM70'):
        BaseModelLoader()._initialize_model(
            ModelConfig(model='unused', hf_config=hf, dtype='bfloat16'), 'cuda:0')


def test_non_linear_modules_do_not_get_linear_quant_methods():
    from minivllm.layers.linear import ParallelLMHead, VocabParallelEmbedding
    config = resolve_quantization_config(None, AWQ)
    assert config.get_quant_method(VocabParallelEmbedding(16, 8), 'model.embed_tokens') is None
    assert config.get_quant_method(ParallelLMHead(16, 8), 'lm_head') is None
    assert config.get_quant_method(torch.nn.LayerNorm(8), 'model.norm') is None


def test_gate_up_mixed_exclusion_rejected():
    from minivllm.layers.linear import MergedColumnParallelLinear
    config = resolve_quantization_config(None, {**AWQ, 'modules_to_not_convert': ['gate_proj']})
    config.packed_modules_mapping.update({'gate_up_proj': ['gate_proj', 'up_proj']})
    with pytest.raises(ValueError, match='some but not all'):
        MergedColumnParallelLinear(8, [8, 8], prefix='model.layers.0.mlp.gate_up_proj',
                                   quant_config=config)


def test_qwen_model_alone_registers_packed_mapping():
    from minivllm.models.qwen3 import Qwen3Model
    from minivllm.testing.tiny_models import tiny_qwen3_config
    config = resolve_quantization_config(None, {**AWQ, 'modules_to_not_convert': ['q_proj']})
    with torch.device('meta'), pytest.raises(ValueError, match='some but not all'):
        Qwen3Model(tiny_qwen3_config('tiny_gqa'), prefix='model', quant_config=config)


def test_compression_checkpoint_is_not_mistaken_for_unquantized_model():
    hf = {'compression_config': {'quant_method': 'compressed-tensors'}}
    with pytest.raises(NotImplementedError, match='compressed-tensors'):
        BaseModelLoader()._initialize_model(ModelConfig(model='unused', hf_config=hf), 'cpu')


def test_real_shape_dispatch_diagnostic_uses_meta_types_without_allocating_weights():
    import importlib.util
    from pathlib import Path
    from minivllm.testing.tiny_models import tiny_qwen3_config
    path = Path(__file__).resolve().parents[2] / 'benchmarks/benchmark_step115_quant_dispatch.py'
    spec = importlib.util.spec_from_file_location('step115_dispatch_diagnostic', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hf = {**tiny_qwen3_config('tiny_gqa'), 'quantization_config': AWQ}
    result = module.methods_for_qwen(hf)
    assert len(result['linear_layers']) == 4 * hf['num_hidden_layers']
    assert result['counts'] == {'AutoAWQMarlinLinearMethod': 4 * hf['num_hidden_layers']}
    assert result['quantized_execution'] == 'not_implemented'
