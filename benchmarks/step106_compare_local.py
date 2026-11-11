"""严格比较 14B draft 局部参考重复或项目输出。"""
import argparse
import hashlib
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
    paths = [args.reference/'output.pt', args.other/'output.pt']
    a, b = (torch.load(path, map_location='cpu', weights_only=True) for path in paths)
    report = {}
    for key in ('aux', 'combined', 'mapped_logits'):
        x, y = a[key], b[key]
        exact = x.shape == y.shape and x.dtype == y.dtype and torch.equal(x, y)
        report[key] = {'exact': exact, 'shape': list(x.shape)}
    payload = {'status': 'passed' if all(v['exact'] for v in report.values()) else 'failed',
               'scope': 'reference repeat' if args.register else '14B draft FC and mapped logits',
               'sha256': [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
               'objects': report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload))
    return int(payload['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
