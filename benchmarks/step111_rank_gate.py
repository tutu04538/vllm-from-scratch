"""Verify identical discrete EAGLE3 state across all TP ranks."""
import argparse
import json
from pathlib import Path
import torch

FIELDS = ('req_ids', 'num_sampled', 'num_rejected', 'sampled_ids',
          'input_ids', 'positions', 'query_start_loc', 'draft_ids')


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def check(path, tp):
    data = [torch.load(path / f'state_rank{rank}.pt', map_location='cpu', weights_only=True) for rank in range(tp)]
    errors = []
    for rank, rows in enumerate(data[1:], 1):
        if len(rows) != len(data[0]):
            errors.append(f'rank {rank}: event count differs'); continue
        for i, (base, current) in enumerate(zip(data[0], rows)):
            if base.get('kind') != current.get('kind'):
                errors.append(f'rank {rank} event {i}: kind differs'); continue
            for field in FIELDS:
                if (field in base) != (field in current) or (field in base and not equal(base[field], current[field])):
                    errors.append(f'rank {rank} event {i}: {field} differs')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--other', type=Path, required=True)
    parser.add_argument('--tp', type=int, choices=(1, 2, 4), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    errors = {'reference':check(args.reference, args.tp), 'other':check(args.other, args.tp)}
    result = {'passed':not any(errors.values()), 'errors':errors}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    return int(not result['passed'])

if __name__ == '__main__': raise SystemExit(main())
