"""真实 AWQ 模型 CUDA 加载与 Marlin 表示审计；不执行模型 forward。"""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from minivllm.config import ModelConfig
from minivllm.layers.quantization.awq import AutoAWQMarlinLinearMethod
from minivllm.model_loader.default_loader import DefaultModelLoader

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if not torch.cuda.is_available():
    raise RuntimeError('需要真实 CUDA')
torch.cuda.reset_peak_memory_stats()
hf = json.loads((args.model/'config.json').read_text())
config = ModelConfig(model=str(args.model), dtype='bfloat16', hf_config=hf)
with torch.inference_mode():
    model = DefaultModelLoader(str(args.model)).load_model_for_representation(config, 'cuda:0')
    layers = [(name, m) for name,m in model.named_modules()
              if isinstance(getattr(m, 'quant_method', None), AutoAWQMarlinLinearMethod)]
    assert len(layers) == hf['num_hidden_layers'] * 4
    workspaces = set()
    for name, layer in layers:
        for key in ('qweight','qzeros','scales'):
            p = getattr(layer,key)
            if p.layout_state != 'kernel' or p.device.type != 'cuda':
                raise ValueError(f'{name}.{key} 不在 CUDA kernel 布局')
            p.validate_loaded()
        ws = layer.quant_method.workspace
        if ws.data_ptr() in workspaces:
            raise ValueError(f'{name}: workspace 跨层共享')
        workspaces.add(ws.data_ptr())
        if layer.g_idx.device.type != 'cuda' or layer.g_idx.numel() != 0:
            raise ValueError(f'{name}: g_idx 状态错误')
    assert model._quant_postload_state == 'representation_ready'
    torch.cuda.synchronize()
    result = {'status':'passed', 'scope':'real_awq_cuda_representation_no_model_forward',
              'model':str(args.model.resolve()), 'layers':len(layers), 'parameters':len(layers)*3,
              'distinct_workspaces':len(workspaces), 'peak_cuda_allocated':torch.cuda.max_memory_allocated(),
              'peak_cuda_reserved':torch.cuda.max_memory_reserved(),
              'current_cuda_allocated':torch.cuda.memory_allocated(), 'forward':'blocked'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
