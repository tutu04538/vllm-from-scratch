"""单进程 CPU 完整 AWQ 模型表示审计，不执行 forward。"""
import argparse
import json
import resource
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from minivllm.config import ModelConfig
from minivllm.layers.quantization.awq import AutoAWQMarlinLinearMethod
from minivllm.model_loader.default_loader import DefaultModelLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    hf = json.loads((args.model / 'config.json').read_text())
    config = ModelConfig(model=str(args.model), dtype='bfloat16', hf_config=hf)
    model = DefaultModelLoader(str(args.model)).load_model_for_representation(config, 'cpu')
    layers = [(name, m) for name, m in model.named_modules()
              if isinstance(getattr(m, 'quant_method', None), AutoAWQMarlinLinearMethod)]
    if len(layers) != hf['num_hidden_layers'] * 4:
        raise ValueError(f'量化层数不符: {len(layers)}')
    for name, layer in layers:
        for param_name in ('qweight', 'qzeros', 'scales'):
            p = getattr(layer, param_name)
            if p.layout_state != 'standard':
                raise ValueError(f'{name}.{param_name}: 转换不完整')
            p.validate_loaded()
    if model._quant_postload_state != 'representation_ready':
        raise ValueError('模型不是表示审计就绪状态')
    try:
        layers[0][1](torch.zeros((1, hf['hidden_size']), dtype=torch.bfloat16))
    except NotImplementedError:
        pass
    else:
        raise ValueError('意外开放了 AWQ forward')
    result = {'status': 'passed', 'scope': 'cpu_full_model_representation_only',
              'model': str(args.model), 'quantized_physical_linear_layers': len(layers),
              'converted_params': len(layers) * 3,
              'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
              'cpu_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'embed_dtype': str(model.model.embed_tokens.weight.dtype),
              'lm_head_dtype': str(model.lm_head.weight.dtype),
              'forward': 'explicitly_rejected', 'gpu': 'not_run'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
