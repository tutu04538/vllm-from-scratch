"""Compare layer hooks for one fixed request/position from diagnostic runs."""
import argparse
import json
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--mini', type=Path, required=True)
    parser.add_argument('--position', type=int, required=True)
    parser.add_argument('--tp', type=int, choices=(1, 2, 4), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = []
    for rank in range(args.tp):
        a = torch.load(args.reference / f'rank{rank}.pt', weights_only=True)
        b = torch.load(args.mini / f'rank{rank}.pt', weights_only=True)
        if a.keys() != b.keys():
            raise ValueError('different observed modules')
        for name in a:
            if args.position not in a[name] or args.position not in b[name]:
                raise ValueError(f'missing {name}, position {args.position}')
            x, y = a[name][args.position], b[name][args.position]
            if len(x) != len(y): raise ValueError(f'different outputs: {name}')
            for i, (ref, other) in enumerate(zip(x, y)):
                if ref is None and other is None: continue
                if ref is None or other is None or ref.shape != other.shape:
                    raise ValueError(f'different output shape: {name}/{i}')
                ref, other = ref.float(), other.float()
                delta = (ref - other).abs()
                report.append({'rank': rank, 'module': name, 'output': i,
                               'shape': list(ref.shape), 'max_abs': float(delta.max()),
                               'relative_l2': float(torch.linalg.vector_norm(delta) /
                                                    torch.linalg.vector_norm(ref).clamp_min(1e-12))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'position':args.position, 'values':report}, indent=2)+'\n')
    print(json.dumps({'position':args.position, 'values':len(report),
                      'first_difference':next((x for x in report if x['max_abs'] > 0), None)}))


if __name__ == '__main__': main()
