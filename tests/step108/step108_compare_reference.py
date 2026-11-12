"""Register frozen repeats or compare all-rank fixed-history tensor traces."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


def tensors(value, path=''):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensors(item, f'{path}/{key}')
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from tensors(item, f'{path}/{i}')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_event(event):
    event = dict(event)
    if event['event'] == 'propose':
        mappings = event.get('slot_mappings') or {}
        values = list(mappings.values())
        if values:
            rows = [value.reshape(-1) for value in values if isinstance(value, torch.Tensor)]
            if rows:
                assert all(torch.equal(rows[0], row) for row in rows[1:])
                event['slot_mappings'] = {'canonical': rows[0]}
    if event['event'] == 'draft_layer' and event.get('inputs'):
        args = event['inputs']
        event['kwargs'] = {'positions': args[0], 'embeds': args[1], 'hidden_states': args[2],
                           'residual': args[3]}
        event['inputs'] = []
    return event


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--other', type=Path, required=True)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--register', action='store_true')
    a = p.parse_args()
    rs = json.loads((a.reference/'summary.json').read_text())
    os = json.loads((a.other/'summary.json').read_text())
    assert rs['tp'] == os['tp'] and rs['k'] == os['k'] and rs['tokens'] == os['tokens']
    limits = {} if a.register else json.loads(a.contract.read_text())['limits']
    errors, metrics = [], {}
    for rank in range(rs['tp']):
        left = torch.load(a.reference/f'rank{rank}.pt', weights_only=True)
        right = torch.load(a.other/f'rank{rank}.pt', weights_only=True)
        if len(left) != len(right):
            errors.append(f'rank {rank}: event count {len(left)} != {len(right)}'); continue
        for index, (x, y) in enumerate(zip(left, right)):
            x, y = canonical_event(x), canonical_event(y)
            if x['event'] != y['event']:
                errors.append(f'rank {rank}, event {index}: event name differs'); continue
            if x.get('valid_context_length') != y.get('valid_context_length'):
                errors.append(f'rank {rank}, event {index}: valid length differs')
            lhs, rhs = dict(tensors(x)), dict(tensors(y))
            if lhs.keys() != rhs.keys():
                errors.append(f'rank {rank}, event {index}: tensor names differ'); continue
            for name, ref in lhs.items():
                got = rhs[name]; key = f"{x['event']}{name}"
                if ref.shape != got.shape or ref.dtype != got.dtype:
                    errors.append(f'rank {rank}, event {index} {name}: shape/dtype mismatch'); continue
                if not ref.is_floating_point():
                    if '/slot_mappings/' in name or name.endswith('/draft_slot_mapping'):
                        valid = ref >= 0
                        equal = torch.equal(valid, got >= 0) and torch.equal(ref[valid] % 16, got[valid] % 16)
                    else:
                        equal = torch.equal(ref, got)
                    if not equal:
                        errors.append(f'rank {rank}, event {index} {name}: discrete mismatch')
                    continue
                mask = torch.isfinite(ref)
                if (torch.isnan(ref).any() or torch.isnan(got).any()
                        or not torch.equal(mask, torch.isfinite(got))
                        or not torch.equal(torch.isneginf(ref), torch.isneginf(got))
                        or not torch.equal(torch.isposinf(ref), torch.isposinf(got))):
                    errors.append(f'rank {rank}, event {index} {name}: finite mask mismatch'); continue
                if not mask.any():
                    continue
                expected, actual = ref[mask].float(), got[mask].float()
                delta = (expected-actual).abs()
                maximum = float(delta.max()); relative = float(delta.norm()/expected.norm().clamp_min(1e-30))
                m = metrics.setdefault(key, {'count': 0, 'max_abs': 0., 'relative_l2': 0.})
                m['count'] += 1; m['max_abs'] = max(m['max_abs'], maximum); m['relative_l2'] = max(m['relative_l2'], relative)
                if not a.register:
                    limit = limits.get(key)
                    if limit is None or maximum > limit['max_abs'] or relative > limit['relative_l2']:
                        errors.append(f'rank {rank}, event {index} {name}: abs={maximum} rel={relative}, limit={limit}')
    if a.register:
        limits = {key: {'max_abs': max(1e-5, 2*v['max_abs']),
                        'relative_l2': max(1e-5, 2*v['relative_l2'])} for key, v in metrics.items()}
        if not errors:
            a.contract.parent.mkdir(parents=True, exist_ok=True)
            a.contract.write_text(json.dumps({'limits': limits, 'reference_sha256': [
                sha(a.reference/f'rank{r}.pt') for r in range(rs['tp'])], 'scope': 'frozen repeat only'}, indent=2)+'\n')
    result = {'status': 'failed' if errors else 'passed', 'tp': rs['tp'], 'k': rs['k'],
              'metrics': metrics, 'errors': errors[:50], 'error_count': len(errors),
              'scope': 'reference repeat' if a.register else 'all-rank natural fixed-history trace'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'errors': len(errors), 'categories': len(metrics)}))
    return int(bool(errors))


if __name__ == '__main__':
    sys.exit(main())
