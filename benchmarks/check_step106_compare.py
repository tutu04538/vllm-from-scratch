"""比较 106 单卡 trace 的输出；固定物理块/KV 验收由独立用例处理。"""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--mini', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-kv', action='store_true')
    args = parser.parse_args()
    ref_summary = json.loads((args.reference / 'summary.json').read_text())
    mini_summary = json.loads((args.mini / 'summary.json').read_text())
    ref = torch.load(ref_summary['runs'][0]['trace'], map_location='cpu', weights_only=True)
    mini = torch.load(mini_summary['trace'], map_location='cpu', weights_only=True)
    failures = []
    if ref_summary.get('k') != mini_summary.get('k'):
        failures.append('K mismatch')
    if len(ref_summary['runs'][0]['tokens']) != 32 or len(mini_summary['tokens']) != 32:
        failures.append('32 output tokens required')
    if ref_summary['prompt_token_ids'] != mini_summary['prompt_token_ids']:
        failures.append('prompt mismatch')
    if ref_summary['runs'][0]['tokens'] != mini_summary['tokens']:
        failures.append('output tokens mismatch')
    if len(ref) != len(mini):
        failures.append('event count mismatch')
    stats = {}
    state_checks = 0
    forced_history = any('forced_output' in event for event in ref)
    for index, (left, right) in enumerate(zip(ref, mini)):
        name = left['event']
        if name != right['event']:
            failures.append(f'{index}: event {name} != {right["event"]}')
            continue
        if 'forced_output' in left or 'forced_output' in right:
            x = left.get('forced_output'); y = right.get('forced_output')
            state_checks += 1
            if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor) or not torch.equal(x, y):
                failures.append(f'{index}: fixed-history output differs or missing')
        if name == 'propose':
            for key in ('num_tokens', 'num_tokens_after_padding', 'num_reqs'):
                state_checks += 1
                if left['batch'].get(key) != right['batch'].get(key):
                    failures.append(f'{index}: batch {key} differs')
            for key in ('num_sampled', 'num_rejected'):
                state_checks += 1
                if not torch.equal(left[key], right[key]):
                    failures.append(f'{index}: {key} differs')
            for key in ('input_ids', 'positions', 'query_start_loc', 'seq_lens', 'idx_mapping'):
                x = left['batch'].get(key); y = right['batch'].get(key)
                state_checks += 1
                if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor) or not torch.equal(x, y):
                    failures.append(f'{index}: batch {key} differs or missing')
            if args.require_kv:
                state_checks += 1
                if left.get('valid_context_length') != right.get('valid_context_length'):
                    failures.append(f'{index}: valid context length differs')
                for key in ('target_valid_kv', 'draft_valid_kv'):
                    x = left.get(key); y = right.get(key)
                    state_checks += 1
                    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor) or not torch.equal(x, y):
                        failures.append(f'{index}: {key} differs or missing')
        if name == 'draft_layer' and args.require_kv:
            for key in ('draft_input_ids', 'draft_positions', 'written_kv'):
                x = left.get(key); y = right.get(key)
                state_checks += 1
                if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor) or not torch.equal(x, y):
                    failures.append(f'{index}: draft {key} differs or missing')
            x = left.get('draft_slot_mapping'); y = right.get('draft_slot_mapping')
            state_checks += 1
            if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
                failures.append(f'{index}: draft slot mapping missing')
            elif (not torch.equal(x < 0, y < 0) or not torch.equal(x[x >= 0] % 16, y[y >= 0] % 16)):
                failures.append(f'{index}: draft slot validity or in-block offset differs')
        a = left['output']; b = right['output']
        a = a if isinstance(a, list) else [a]
        b = b if isinstance(b, list) else [b]
        if len(a) != len(b):
            failures.append(f'{index}: output arity mismatch')
            continue
        for column, (x, y) in enumerate(zip(a, b)):
            if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
                failures.append(f'{index}: non-tensor output')
                continue
            if x.shape != y.shape or x.dtype != y.dtype:
                failures.append(f'{index}: output shape/dtype mismatch')
                continue
            item = stats.setdefault(name, {'tensors': 0, 'max_abs': 0.0, 'relative_l2': 0.0,
                                           'max_error_flat_index': None, 'actual_at_max': None,
                                           'reference_at_max': None, 'top2_first_row': None})
            item['tensors'] += 1
            if not torch.equal(x, y):
                failures.append(f'{index}: {name} output {column} differs')
            finite = torch.isfinite(x) & torch.isfinite(y)
            if not torch.equal(torch.isfinite(x), torch.isfinite(y)):
                failures.append(f'{index}: {name} finite mask differs')
            if finite.any():
                xf = x[finite].float(); delta = xf - y[finite].float()
                absolute = delta.abs(); maximum = float(absolute.max())
                if item['max_error_flat_index'] is None or maximum > item['max_abs']:
                    local_index = int(absolute.argmax())
                    item['max_error_flat_index'] = int(finite.flatten().nonzero()[local_index])
                    item['reference_at_max'] = float(xf[local_index])
                    item['actual_at_max'] = float(y[finite][local_index])
                    item['max_abs'] = maximum
                item['relative_l2'] = max(item['relative_l2'],
                                           float(delta.norm() / xf.norm().clamp_min(1e-30)))
            if name == 'compute_logits' and item['top2_first_row'] is None:
                values, ids = x[0].float().topk(2)
                actual_values, actual_ids = y[0].float().topk(2)
                item['top2_first_row'] = {'reference_ids': ids.tolist(), 'reference_values': values.tolist(),
                                          'actual_ids': actual_ids.tolist(), 'actual_values': actual_values.tolist(),
                                          'reference_margin': float(values[0] - values[1]),
                                          'actual_margin': float(actual_values[0] - actual_values[1])}
    if not all(name in stats for name in ('propose', 'combine_hidden_states', 'draft_layer', 'compute_logits')):
        failures.append('required draft events missing')
    mode = 'fixed-history replay' if forced_history else 'normal generation'
    payload = {'status': 'passed' if not failures else 'failed', 'k': mini_summary['k'],
               'events': len(ref), 'statistics': stats, 'state_checks': state_checks, 'failures': failures,
               'scope': (mode + ' and valid first target layer/draft KV; forced rejection separate'
                         if args.require_kv else mode + ' event outputs; no forced reject or KV content validation')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(payload, ensure_ascii=False))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
