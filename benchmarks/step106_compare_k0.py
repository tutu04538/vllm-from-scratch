"""严格比较 8B K0 正常生成及 prefill/前三轮 decode 的 target trace。"""
import argparse
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--other', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--register', action='store_true')
    args = p.parse_args()
    a = json.loads((args.reference/'summary.json').read_text())
    b = json.loads((args.other/'summary.json').read_text())
    other_index = 1 if args.register else 0
    tokens_a = a['runs'][0]['outputs'][0]['tokens']
    tokens_b = b['runs'][other_index]['outputs'][0]['tokens']
    rows_a = torch.load(args.reference/'run0_rank0.pt', map_location='cpu', weights_only=True)
    rows_b = torch.load(args.other/f'run{other_index}_rank0.pt', map_location='cpu', weights_only=True)
    checks = []
    for x, y in zip(rows_a, rows_b):
        for field in ('hidden', 'logits', 'kv'):
            checks.append({'position': x['position'], 'field': field,
                           'equal': x['position'] == y['position'] and torch.equal(x[field], y[field])})
    passed = (tokens_a == tokens_b and len(rows_a) == len(rows_b) == 4
              and len(checks) == 12 and all(c['equal'] for c in checks))
    result = {'status': 'passed' if passed else 'failed', 'token_equal': tokens_a == tokens_b,
              'checks': checks, 'scope': 'target 32 tokens and first-layer KV at four positions'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'checks': len(checks)}))
    return int(not passed)


if __name__ == '__main__':
    raise SystemExit(main())
