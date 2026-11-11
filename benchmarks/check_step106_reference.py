"""仅用冻结 vLLM 重复运行结果，登记 106 的逐对象参考稳定性契约。"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def tensors(value, prefix=''):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensors(item, f'{prefix}/{key}')
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from tensors(item, f'{prefix}/{index}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.reference / 'summary.json'
    summary = json.loads(summary_path.read_text())
    paths = [Path(run['trace']) for run in summary['runs']]
    if len(paths) < 2:
        raise ValueError('至少两次独立参考运行')
    runs = [torch.load(path, map_location='cpu', weights_only=True) for path in paths]
    reference = runs[0]
    if not all(len(run) == len(reference) for run in runs):
        raise ValueError('参考 trace 长度不一致')
    if not all(run['tokens'] == summary['runs'][0]['tokens'] for run in summary['runs']):
        raise ValueError('参考输出 token 不一致')
    measurements = {}
    physical_slot_differences = 0
    for position, baseline in enumerate(reference):
        event = baseline['event']
        for run in runs[1:]:
            candidate = run[position]
            if candidate['event'] != event:
                raise ValueError(f'事件顺序不一致：{position}')
            if baseline.get('valid_context_length') != candidate.get('valid_context_length'):
                raise ValueError(f'有效上下文长度不一致：{position}')
            if event == 'propose':
                for field in ('num_tokens', 'num_tokens_after_padding', 'num_reqs'):
                    if baseline['batch'][field] != candidate['batch'][field]:
                        raise ValueError(f'batch 字段不一致：{position} {field}')
            left = dict(tensors(baseline)); right = dict(tensors(candidate))
            if left.keys() != right.keys():
                raise ValueError(f'字段不一致：{position}')
            for name, a in left.items():
                b = right[name]
                if a.shape != b.shape or a.dtype != b.dtype:
                    raise ValueError(f'shape/dtype 不一致：{position} {name}')
                category = f'{event}{name}'
                stats = measurements.setdefault(category, {'count': 0, 'max_abs': 0.0, 'relative_l2': 0.0})
                stats['count'] += 1
                if not a.is_floating_point():
                    if '/slot_mappings/' in name or name.endswith('/draft_slot_mapping'):
                        # 同一进程重复请求可取得不同的物理块；逐位验证 padding 与块内偏移，
                        # 并记录物理地址变化，不能把它当固定块表的状态对照通过。
                        if not torch.equal(a < 0, b < 0):
                            raise ValueError(f'slot 有效性不一致：{position} {name}')
                        valid = a >= 0
                        if not torch.equal(a[valid] % 16, b[valid] % 16):
                            raise ValueError(f'slot 块内位置不一致：{position} {name}')
                        physical_slot_differences += int(not torch.equal(a, b))
                        continue
                    if not torch.equal(a, b):
                        raise ValueError(f'离散值不一致：{position} {name}')
                    continue
                finite = torch.isfinite(a)
                if (not torch.equal(finite, torch.isfinite(b))
                        or not torch.equal(torch.isneginf(a), torch.isneginf(b))
                        or not torch.equal(torch.isposinf(a), torch.isposinf(b))
                        or torch.isnan(a).any() or torch.isnan(b).any()):
                    raise ValueError(f'有效列不一致：{position} {name}')
                a = a[finite].float(); b = b[finite].float()
                if a.numel():
                    delta = a - b
                    stats['max_abs'] = max(stats['max_abs'], float(delta.abs().max()))
                    denominator = float(torch.linalg.vector_norm(a))
                    stats['relative_l2'] = max(stats['relative_l2'],
                                               float(torch.linalg.vector_norm(delta)) / max(denominator, 1e-30))
    for stats in measurements.values():
        stats['absolute_limit'] = max(1e-5, 2 * stats['max_abs'])
        stats['relative_l2_limit'] = max(1e-5, 2 * stats['relative_l2'])
    contract = {'scope': '8B TP1 eager fixed-history reference repetition only; not FP32 or TP comparison',
                'k': summary['k'], 'oracle': summary['oracle'], 'summary_sha256': digest(summary_path),
                'trace_sha256': [digest(path) for path in paths], 'repeats': len(paths),
                'events_per_run': len(reference), 'physical_slot_differences': physical_slot_differences,
                'measurements': measurements}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'reference_repeat_contract_written', 'k': summary['k'],
                      'events': len(reference), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
