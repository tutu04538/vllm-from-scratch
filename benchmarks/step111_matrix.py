"""Immutable workload matrix and selected-GPU occupancy checks."""
import os
import subprocess


def build_cases():
    """同一配置只测一次；A/B/C/D 重叠点合并并保留来源。"""
    cases = {}
    def add(model, tp, batch, lengths, k, mode, group):
        key = (model, tp, batch, tuple(lengths), k, mode)
        if key not in cases:
            cases[key] = {'model': model, 'tp': tp, 'batch': batch,
                          'lengths': list(lengths), 'k': k, 'mode': mode,
                          'output_tokens': 128, 'groups': []}
        cases[key]['groups'].append(group)
    for model, tps in [('8b', (1, 2, 4)), ('14b', (2, 4))]:
        for tp in tps:
            for length in (128, 1024): add(model, tp, 1, [length], 0, 'eager', 'A')
            add(model, tp, 16, [128] * 8 + [1024] * 8, 0, 'eager', 'B')
            for batch in (1, 4):
                for mode in ('eager', 'graph'):
                    add(model, tp, batch, [128] * batch, 0, mode, 'C')
                    for k in (0, 2): add(model, tp, batch, [128] * batch, k, mode, 'D')
    return list(cases.values())


def compute_pids(tp):
    """只读取所选 GPU 的进程，不终止其他项目的进程。"""
    devices = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
                             check=True, text=True, capture_output=True, timeout=15)
    by_index = dict(line.split(', ', 1) for line in devices.stdout.splitlines() if ', ' in line)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    selected = visible.split(',')[:tp] if visible else list(by_index)[:tp]
    if len(selected) != tp:
        raise ValueError(f'only {len(selected)} GPUs visible for TP={tp}')
    uuids = set()
    for item in selected:
        item = item.strip()
        matches = [by_index[item]] if item in by_index else [uuid for uuid in by_index.values() if uuid.startswith(item)]
        if len(matches) != 1:
            raise ValueError(f'cannot uniquely resolve selected GPU {item!r}')
        uuids.add(matches[0])
    if len(uuids) != tp:
        raise ValueError('duplicate GPU selection')
    completed = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'],
                               check=True, text=True, capture_output=True, timeout=15)
    result = set()
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) == 2 and parts[0].isdigit() and parts[1] in uuids:
            result.add(int(parts[0]))
    return result


