"""Compare every TP rank against the separately registered frozen reference."""
import argparse
import json
from pathlib import Path
import sys

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--mini', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    contract = json.loads(args.contract.read_text())
    assert contract['acceptance']['kind'] == 'exact'
    cases = []
    for rank in range(contract['tp']):
        ref = torch.load(args.reference / 'run0' / f'rank{rank}.pt', weights_only=True)
        mini = torch.load(args.mini / f'rank{rank}.pt', weights_only=True)
        for event, field, index in (('combine_hidden_states', 'combined', None),
                                    ('draft_qkv', 'qkv', None),
                                    ('draft_attention', 'attention_output', None),
                                    ('draft_layer', 'attention_kv', 'written_kv'),
                                    ('draft_layer', 'layer_output', 0),
                                    ('draft_layer', 'layer_residual', 1),
                                    ('compute_logits', 'layer_logits', None),
                                    ('compute_logits', 'mapped_logits', None)):
            record = next(e for e in ref if e['event'] == event)
            ref_tensor = record[index] if isinstance(index, str) else record['output']
            if isinstance(index, int):
                ref_tensor = ref_tensor[index]
            got = mini[field]
            mask = torch.isfinite(ref_tensor) & torch.isfinite(got)
            delta = (ref_tensor[mask].float() - got[mask].float()).abs()
            denom = ref_tensor[mask].float().norm()
            entry = {'rank': rank, 'field': field,
                     'exact': ref_tensor.dtype == got.dtype and torch.equal(ref_tensor, got),
                     'dtype_reference': str(ref_tensor.dtype), 'dtype_mini': str(got.dtype),
                     'finite_mask_equal': torch.equal(torch.isfinite(ref_tensor), torch.isfinite(got)),
                     'max_abs': float(delta.max()) if delta.numel() else 0.,
                     'relative_l2': float(delta.norm()/denom) if denom else 0.}
            if delta.numel():
                full_delta = torch.zeros_like(ref_tensor, dtype=torch.float32)
                full_delta[mask] = delta
                flat = int(full_delta.argmax())
                location = list(torch.unravel_index(torch.tensor(flat), full_delta.shape))
                location = [int(v) for v in location]
                entry.update(max_location=location,
                             reference_value=float(ref_tensor[tuple(location)]),
                             mini_value=float(got[tuple(location)]))
            if field in ('mapped_logits', 'layer_logits'):
                vals, ids = ref_tensor.float().topk(2, dim=-1)
                entry['reference_top1'] = ids[:, 0].tolist()
                entry['reference_top2'] = ids[:, 1].tolist()
                entry['reference_margin'] = (vals[:, 0]-vals[:, 1]).tolist()
                got_vals, got_ids = got.float().topk(2, dim=-1)
                entry['mini_top1'] = got_ids[:, 0].tolist()
                entry['mini_top2'] = got_ids[:, 1].tolist()
                entry['mini_margin'] = (got_vals[:, 0]-got_vals[:, 1]).tolist()
            cases.append(entry)
    success = all(case['exact'] and case['finite_mask_equal'] for case in cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'status': 'passed' if success else 'failed',
        'contract': str(args.contract), 'cases': cases}, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status': 'passed' if success else 'failed', 'comparisons': len(cases)}))
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
