"""117 CPU 加载形状检查；真实性能与 GPU 验证尚未进行。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from minivllm.layers.linear import ColumnParallelLinear, RowParallelLinear
from minivllm.layers.parameter import PackedvLLMParameter, GroupQuantScaleParameter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--case', choices=['all'], default='all')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not any((args.check, args.compare, args.trace)):
        parser.error('select --check, --compare or --trace')
    if args.compare or args.trace:
        result = {'status': 'not_run', 'reason': 'GPU comparison and profiling deferred'}
        code = 2
    else:
        weight = torch.arange(32 * 256, dtype=torch.int32).reshape(32, 256)
        scale = torch.arange(2 * 256, dtype=torch.float32).reshape(2, 256)
        layers = []
        for cls in (ColumnParallelLinear, RowParallelLinear):
            for rank in (0, 1):
                layer = cls(256, 256)
                layer.tp_rank, layer.tp_size = rank, 2
                ws = (32, 128) if cls is ColumnParallelLinear else (16, 256)
                ss = (2, 128) if cls is ColumnParallelLinear else (1, 256)
                q = PackedvLLMParameter(torch.empty(ws, dtype=torch.int32), layer.weight_loader,
                    input_dim=0, output_dim=1, packed_dim=0, packed_factor=8)
                s = GroupQuantScaleParameter(torch.empty(ss), layer.weight_loader, input_dim=0, output_dim=1)
                layer.weight_loader(q, weight); layer.weight_loader(s, scale)
                q.validate_loaded(); s.validate_loaded()
                layers.append({'layer': cls.__name__, 'rank': rank, 'qweight_shape': ws,
                               'scale_shape': ss, 'local_bytes': q.numel()*q.element_size()+s.numel()*s.element_size()})
        result = {'status': 'passed', 'scope': 'cpu_fixture_only', 'layers': layers}
        code = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output)}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
