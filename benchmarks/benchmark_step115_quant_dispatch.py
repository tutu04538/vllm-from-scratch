"""CPU 模型配置分派清单；只检查方法选择，不执行量化或性能测试。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minivllm.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from minivllm.layers.quantization import resolve_quantization_config
from minivllm.models.qwen3 import QWEN3_PACKED_MODULES_MAPPING


def methods_for_qwen(hf_config, user_method=None):
    if hf_config.get('architectures') != ['Qwen3ForCausalLM']:
        raise ValueError('only dense Qwen3ForCausalLM is supported')
    count = hf_config.get('num_hidden_layers')
    if type(count) is not int or count <= 0:
        raise ValueError('num_hidden_layers must be positive')
    config = resolve_quantization_config(user_method, hf_config.get('quantization_config'))
    if config is None:
        raise ValueError('checkpoint does not declare quantization')
    config.packed_modules_mapping.update(QWEN3_PACKED_MODULES_MAPPING)
    layer_types = [('self_attn.qkv_proj', QKVParallelLinear), ('self_attn.o_proj', RowParallelLinear),
                   ('mlp.gate_up_proj', MergedColumnParallelLinear), ('mlp.down_proj', RowParallelLinear)]
    rows = []
    for i in range(count):
        for name, layer_type in layer_types:
            # 只用于分派诊断；不构造权重，也不声称 shape/kernel 已验证。
            layer = layer_type.__new__(layer_type)
            torch.nn.Module.__init__(layer)
            prefix = f'model.layers.{i}.{name}'
            method = config.get_quant_method(layer, prefix)
            rows.append({'prefix': prefix, 'layer': layer_type.__name__, 'method': type(method).__name__})
    return {'scope': 'cpu_dispatch_only_no_weight_allocation', 'config_name': config.get_name(),
            'config_source': config.source, 'linear_layers': rows,
            'counts': dict(Counter(row['method'] for row in rows)),
            'unquantized_non_linear': ['model.embed_tokens', 'model.norm', 'lm_head'],
            'quantized_execution': 'not_implemented'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--quantization')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--case', default='all', choices=['all'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not any((args.check, args.compare, args.trace)):
        parser.error('select --check, --compare, or --trace')
    if args.compare or args.trace:
        result = {'status': 'not_run', 'reason': 'GPU/model comparison and performance trace deferred'}
        code = 2
    else:
        config = json.loads((args.model / 'config.json').read_text())
        result = methods_for_qwen(config, args.quantization)
        result['status'] = 'passed'
        result['model'] = str(args.model.resolve())
        code = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output)}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
