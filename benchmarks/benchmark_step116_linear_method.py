"""116 线性方法检查入口；GPU 对照和性能测量后续进行。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from minivllm.layers.linear import (ReplicatedLinear, ColumnParallelLinear, RowParallelLinear,
                                    QKVParallelLinear, MergedColumnParallelLinear)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true')
    p.add_argument('--compare', action='store_true')
    p.add_argument('--trace', action='store_true')
    p.add_argument('--case', default='all', choices=['all'])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not any((args.check, args.compare, args.trace)):
        p.error('select --check, --compare or --trace')
    if args.compare or args.trace:
        result = {'status': 'not_run', 'reason': 'GPU correctness comparison and performance deferred'}
        code = 2
    else:
        layers = [ReplicatedLinear(8, 12), ColumnParallelLinear(8, 12),
                  RowParallelLinear(8, 12), QKVParallelLinear(8, 2, 4, 2),
                  MergedColumnParallelLinear(8, [4, 8])]
        result = {'status': 'passed', 'scope': 'cpu_construction_only',
                  'layers': [{'type': type(x).__name__, 'method': type(x.quant_method).__name__,
                              'weight_shape': list(x.weight.shape)} for x in layers]}
        code = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output)}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
