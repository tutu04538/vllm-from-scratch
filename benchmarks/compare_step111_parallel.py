"""需求 111：隔离进程的 Qwen3 TP 公平性能矩阵。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.benchmark_runner import _atomic_write_json
from benchmarks.step88_common import package_hash
from benchmarks.step111_matrix import build_cases, compute_pids


def summarize(samples):
    names = ('wall_time_s', 'output_tokens_per_s', 'requests_per_s', 'ttft_s', 'tpot_s', 'inter_burst_latency_s', 'queue_seconds')
    result = {}
    for name in names:
        values = [sample[name] for sample in samples if sample.get(name) is not None]
        result[name] = {'values': values, 'median': statistics.median(values) if values else None,
                        'min': min(values) if values else None, 'max': max(values) if values else None}
    return result


def upstream_log_times(log, tp=None):
    """Extract upper's own logged rank times; capture is rounded to whole seconds."""
    ranks = {}
    for line in log.splitlines():
        for pattern, key, convert in ((r'Model loading took .* and ([\d.]+) seconds', 'model_loading_seconds', float),
                                      (r'Graph capturing finished in (\d+) secs', 'graph_capture_seconds_rounded', int)):
            match = re.search(pattern, line)
            if not match: continue
            prefixes = list(re.finditer(r'Worker_TP(\d+)', line[:match.start()]))
            if prefixes:
                ranks.setdefault(int(prefixes[-1].group(1)), {})[key] = convert(match.group(1))
    if tp == 1 and not ranks:
        load = re.search(r'Model loading took .* and ([\d.]+) seconds', log)
        capture = re.search(r'Graph capturing finished in (\d+) secs', log)
        if load or capture:
            ranks[0] = {}
            if load: ranks[0]['model_loading_seconds'] = float(load.group(1))
            if capture: ranks[0]['graph_capture_seconds_rounded'] = int(capture.group(1))
    return ranks


def token_difference(left, right):
    if left.keys() != right.keys():
        return {'error':'request IDs differ', 'left':sorted(left), 'right':sorted(right)}
    for req in sorted(left):
        first = next((index for index, (a, b) in enumerate(zip(left[req], right[req])) if a != b), None)
        if first is not None:
            return {'request':req, 'index':first, 'left':left[req][first], 'right':right[req][first]}
        if len(left[req]) != len(right[req]):
            return {'request':req, 'error':'length differs', 'left':len(left[req]), 'right':len(right[req])}
    return None


def compare_schedule(reference, other):
    """Compare active protocol rounds and a consistent bijection of physical blocks.

    Eager mini and frozen vLLM intentionally reserve different null blocks. The
    raw IDs are retained; equality here means identical logical allocation under
    one stable, one-to-one mapping. Empty cleanup rounds are reported separately.
    """
    left = [row for row in reference if row['scheduled_tokens']]
    right = [row for row in other if row['scheduled_tokens']]
    if len(left) != len(right):
        return {'passed':False, 'error':'active round count differs'}
    forward, reverse = {}, {}
    def blocks(a, b):
        if a is None or b is None: return a is None and b is None
        if len(a) != len(b): return False
        for group, (first, second) in enumerate(zip(a,b)):
            if len(first) != len(second): return False
            for x,y in zip(first,second):
                x, y = (group,x), (group,y)
                if (x in forward and forward[x] != y) or (y in reverse and reverse[y] != x): return False
                forward[x] = y; reverse[y] = x
        return True
    for index, (a,b) in enumerate(zip(left,right)):
        for field in ('scheduled_tokens','spec_tokens'):
            if a[field] != b[field]: return {'passed':False,'error':f'round {index}: {field} differs'}
        if len(a['new']) != len(b['new']): return {'passed':False,'error':f'round {index}: new request count differs'}
        for x,y in zip(a['new'],b['new']):
            if any(x[key] != y[key] for key in ('request_id','prompt_token_ids','num_computed_tokens')):
                return {'passed':False,'error':f'round {index}: new request state differs'}
            if not blocks(x['block_ids'],y['block_ids']):
                return {'passed':False,'error':f'round {index}: new block mapping differs'}
        x,y = a['cached'],b['cached']
        for field in ('req_ids','num_computed_tokens','num_output_tokens','resumed_req_ids'):
            if x[field] != y[field]: return {'passed':False,'error':f'round {index}: cached {field} differs'}
        if len(x['new_block_ids']) != len(y['new_block_ids']) or any(
                not blocks(u,v) for u,v in zip(x['new_block_ids'],y['new_block_ids'])):
            return {'passed':False,'error':f'round {index}: cached block mapping differs'}
    return {'passed':True,'active_rounds':len(left),'physical_block_mapping':
            [{'group':x[0],'reference':x[1],'other':y[1]} for x,y in sorted(forward.items())],
            'reference_empty_rounds':len(reference)-len(left),'other_empty_rounds':len(other)-len(right),
            'raw_finished_notifications_equal': [row['finished_req_ids'] for row in reference] ==
                                                [row['finished_req_ids'] for row in other]}


def rank_devices(result, tp):
    mapping = {}
    for item in result['rank_before']:
        rank, uuid = item['rank'], item['device_uuid']
        if rank in mapping or not uuid:
            raise ValueError('duplicate rank or missing device UUID')
        mapping[rank] = uuid
    if set(mapping) != set(range(tp)) or len(set(mapping.values())) != tp:
        raise ValueError('TP ranks do not own distinct expected GPUs')
    after = {item['rank']:item['device_uuid'] for item in result['rank_after']}
    if len(result['rank_after']) != tp or mapping != after:
        raise ValueError('rank device changed during the case')
    return mapping


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes(entry, case):
    paths = []
    gate = entry['gate']
    for label in ('frozen_ref','frozen_repeat','mini'):
        item = gate[label]; output = Path(item['command'][-1]); directory = output.with_suffix('')
        paths.extend((output,Path(item['log'])))
        prefix = 'state_rank' if case['k'] else 'run0_rank'
        paths.extend(directory/f'{prefix}{rank}.pt' for rank in range(case['tp']))
    for label in ('registration','comparison'):
        paths.extend((Path(gate[label]['report']),Path(gate[label]['contract'])))
        if gate[label].get('relative_contract'): paths.append(Path(gate[label]['relative_contract']))
        rank_command = gate[label].get('rank_command')
        if rank_command: paths.append(Path(rank_command[-1]))
    for item in entry['engines'].values():
        paths.extend((Path(item['command'][-1]),Path(item['log'])))
    for proof in (gate.get('near_tie'),entry.get('performance_near_tie')):
        if proof and proof.get('passed'):
            paths.append(Path(proof['report']))
            for item in proof['runs'].values(): paths.extend((Path(item['path']),Path(item['log'])))
    return {str(path):file_sha256(path) for path in dict.fromkeys(paths)}


def validate_measurement_metadata(result, case, *, gate, implementation):
    rank_devices(result, case['tp'])
    for item in result['rank_after']:
        for key in ('parameters_bytes', 'kv_unique_bytes', 'peak_allocated_bytes'):
            if not isinstance(item.get(key), int) or item[key] <= 0:
                raise ValueError(f'rank {item["rank"]} has missing {key}')
        if not isinstance(item.get('known_workspace_bytes'), int) or item['known_workspace_bytes'] < 0:
            raise ValueError(f'rank {item["rank"]} has missing workspace bytes')
        if case['mode'] == 'graph':
            if item.get('target_graph_count', 0) <= 0:
                raise ValueError(f'rank {item["rank"]} has no captured target graph')
            if gate and (item.get('gate_graph_replays') or 0) <= 0:
                raise ValueError(f'rank {item["rank"]} did not replay target graph')
    if not gate and implementation == 'frozen':
        log_times = result.get('upstream_log_times_by_rank', {})
        for rank in range(case['tp']):
            item = log_times.get(rank, log_times.get(str(rank), {}))
            if 'model_loading_seconds' not in item:
                raise ValueError(f'rank {rank} has no logged model loading time')
            if case['mode'] == 'graph' and 'graph_capture_seconds_rounded' not in item:
                raise ValueError(f'rank {rank} has no logged graph capture time')
    if not gate and implementation == 'mini':
        if result.get('executor_init_including_load_seconds') is None:
            raise ValueError('mini executor loading interval is missing')
        if case['mode'] == 'graph':
            capture = result.get('capture_seconds_by_rank', [])
            if len(capture) != case['tp'] or any(value is None for value in capture):
                raise ValueError('mini graph capture time is missing')


def reusable_case(entry, case, repeats):
    """Reuse only complete, verifiable artifacts from the identical harness."""
    if entry.get('case') != case or entry.get('status') not in ('passed_performance','observed_near_tie'):
        return False
    try:
        hashes=entry['artifact_hashes']
        if not hashes or any(not Path(path).is_file() or file_sha256(path)!=expected for path,expected in hashes.items()):
            return False
        gate = entry['gate']
        if entry.get('state_gate') not in ('passed','passed_near_tie') or not gate['schedule_comparison']['passed']:
            return False
        for label, impl in (('frozen_ref','frozen'),('frozen_repeat','frozen'),('mini','mini')):
            item = gate[label]
            if item['exit_code'] or not Path(item['command'][-1]).is_file() or not Path(item['log']).is_file(): return False
            saved = dict(item['result']); saved.pop('upstream_log_times_by_rank',None)
            if json.loads(Path(item['command'][-1]).read_text()) != saved: return False
            validate_measurement_metadata(item['result'], case, gate=True, implementation=impl)
            directory = Path(item['command'][-1]).with_suffix('')
            prefix = 'state_rank' if case['k'] else 'run0_rank'
            if any(not (directory/f'{prefix}{rank}.pt').is_file() for rank in range(case['tp'])): return False
        comparison = Path(gate['comparison']['report'])
        if not comparison.is_file(): return False
        report = json.loads(comparison.read_text())
        if entry['state_gate'] == 'passed' and not report.get('passed'): return False
        for path, expected in gate.get('reference_hashes', {}).items():
            if not Path(path).is_file() or hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected: return False
        for impl in ('mini','frozen'):
            item = entry['engines'][impl]
            if item['exit_code'] or not Path(item['command'][-1]).is_file() or not Path(item['log']).is_file(): return False
            saved = dict(item['result']); saved.pop('upstream_log_times_by_rank',None)
            if json.loads(Path(item['command'][-1]).read_text()) != saved: return False
            validate_measurement_metadata(item['result'], case, gate=False, implementation=impl)
            samples = item['result']['samples']
            if len(samples) < repeats or any(len(tokens) != 128 for sample in samples for tokens in sample['tokens'].values()): return False
        if entry['status'] == 'observed_near_tie':
            from benchmarks.step111_neartie import verify
            proofs = [('performance',entry.get('performance_near_tie')),('gate',gate.get('near_tie'))]
            valid = False
            for stage, proof in proofs:
                if proof and proof.get('passed') and Path(proof['report']).is_file():
                    reference=json.loads(Path(proof['runs']['frozen']['path']).read_text())
                    mini=json.loads(Path(proof['runs']['mini']['path']).read_text())
                    valid = bool(verify(entry,reference,mini,stage)['passed']); break
            if not valid: return False
        return True
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('8b', '14b'))
    parser.add_argument('--tp', type=int, choices=(1, 2, 4))
    parser.add_argument('--group', choices=('A', 'B', 'C', 'D'))
    parser.add_argument('--mode', choices=('eager', 'graph'))
    parser.add_argument('--k', type=int, choices=(0, 2))
    parser.add_argument('--batch', type=int, choices=(1, 4, 16))
    parser.add_argument('--input-length', type=int, choices=(128, 1024),
                        help='select uniform-length workloads; mixed B trace is excluded')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--calibration', type=Path,
                        help='pre-registered cross-TP frozen reference for long and mixed inputs')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true', help='resume an existing matching matrix without overwriting earlier attempts')
    args = parser.parse_args()
    if args.repeats < 3 or args.timeout <= 0: parser.error('至少重复三次，超时必须为正')
    cases = [c for c in build_cases() if (args.model is None or c['model'] == args.model)
             and (args.tp is None or c['tp'] == args.tp) and (args.group is None or args.group in c['groups'])
             and (args.mode is None or c['mode'] == args.mode) and (args.k is None or c['k'] == args.k)
             and (args.batch is None or c['batch'] == args.batch)
             and (args.input_length is None or all(n == args.input_length for n in c['lengths']))]
    if not cases: parser.error('没有匹配的 case')
    long_groups = {f"{c['model']}_b{c['batch']}" for c in cases if 1024 in c['lengths']}
    if long_groups and not args.dry_run:
        if args.calibration is None or not args.calibration.is_file():
            parser.error('1024 token 或混合输入需要预先登记的跨 TP 冻结参考 --calibration')
        calibration = json.loads(args.calibration.read_text())
        if calibration.get('status') != 'registered': parser.error('跨 TP 参考合同未完成登记')
        required_sources = {'register_step111_reference.py','step111_matrix.py','step111_parallel_worker.py','step103_hooks.py',
                            'step111_compare_gate.py','step109_compare.py'}
        if set(calibration.get('source_sha256', {})) != required_sources:
            parser.error('跨 TP 参考合同缺少源码摘要')
        for name, expected in calibration['source_sha256'].items():
            path = ROOT/'benchmarks'/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                parser.error(f'参考合同源码已变化: {name}')
        for name in long_groups:
            item = calibration.get('results', {}).get(name, {})
            model = name.split('_',1)[0]
            if item.get('status') != 'registered' or item.get('tps') != ([1,2,4] if model == '8b' else [2,4]):
                parser.error(f'跨 TP 参考覆盖不全: {name}')
            references = item.get('reference_sha256', {})
            if len(references) != (20 if model == '8b' else 16):
                parser.error(f'跨 TP 参考文件覆盖不全: {name}')
            for path, expected in references.items():
                reference = Path(path)
                if not reference.is_file() or hashlib.sha256(reference.read_bytes()).hexdigest() != expected:
                    parser.error(f'已登记参考发生变化或缺失: {path}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = ROOT/'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/源码清单.json'
    if not manifest.is_file():
        parser.error(f'冻结源码清单缺失: {manifest}')
    sources = ('compare_step111_parallel.py', 'step111_parallel_worker.py', 'step111_compare_gate.py',
               'step111_rank_gate.py', 'step111_neartie.py', 'step111_firstdiff_probe.py',
               'step111_matrix.py', 'step103_hooks.py', 'step103_memory_hooks.py', 'step106_reference_probe.py',
               'step109_real_probe.py', 'step109_compare.py', 'benchmark_runner.py', 'spec_decode_metrics.py')
    payload = {'status': 'planned' if args.dry_run else 'running', 'cases': cases,
               'results': [{'case':case, 'status':'unrun'} for case in cases],
               'repeats': args.repeats, 'timeout': args.timeout,
               'package': package_hash(),
               'source_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
               'harness_sha256': {name:hashlib.sha256((ROOT/'benchmarks'/name).read_bytes()).hexdigest()
                                  for name in sources},
               'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    if args.calibration is not None:
        payload['calibration'] = {'path':str(args.calibration), 'sha256':hashlib.sha256(args.calibration.read_bytes()).hexdigest()}
    if args.resume:
        if args.dry_run: parser.error('--resume and --dry-run cannot be combined')
        if not args.output.is_file(): parser.error('resume output does not exist')
        prior = json.loads(args.output.read_text())
        for key in ('cases','repeats','package','source_manifest_sha256','harness_sha256','cuda_visible_devices','calibration'):
            if prior.get(key) != payload.get(key): parser.error(f'resume configuration or source differs: {key}')
        if len(prior.get('results', [])) != len(cases): parser.error('resume case list is incomplete')
        payload['results'] = prior['results']
    def save(): _atomic_write_json(args.output, payload)
    save()
    if args.dry_run:
        print(f'{len(cases)} cases; {args.output}')
        return 0
    # The per-engine runner and correctness gate are intentionally required: an
    # absent result is a failure, never a zero-throughput success.
    worker = ROOT / 'benchmarks' / 'step111_parallel_worker.py'
    for index, case in enumerate(cases):
        case_id = f"{case['model']}_tp{case['tp']}_b{case['batch']}_n{case['lengths'][-1]}_k{case['k']}_{case['mode']}"
        previous = payload['results'][index]
        if args.resume and reusable_case(previous,case,args.repeats):
            print(json.dumps({'case':case_id,'status':'reused_completed'}),flush=True); continue
        history = list(previous.get('previous_attempts', []))
        if args.resume and previous.get('status') != 'unrun':
            history.append({key:value for key,value in previous.items() if key != 'previous_attempts'})
        case_tag = case_id if not history else f'{case_id}_attempt{len(history)+1}'
        entry = {'case': case, 'case_id': case_id, 'gate': {}, 'engines': {}, 'status': 'running',
                 'previous_attempts':history}
        payload['results'][index] = entry; save()
        print(json.dumps({'case':case_id, 'index':index+1, 'total':len(cases), 'status':'running'}), flush=True)
        order = ('mini', 'frozen') if index % 2 == 0 else ('frozen', 'mini')
        folder = args.output.parent / (args.output.stem + '_runs'); folder.mkdir(exist_ok=True)
        def run(impl, phase, suffix=''):
            result_path = folder / f'{case_tag}_{impl}_{phase}{suffix}.json'
            log_path = folder / f'{case_tag}_{impl}_{phase}{suffix}.log'
            cmd = [sys.executable, str(worker), '--case', json.dumps(case), '--implementation', impl,
                   '--phase', phase, '--repeats', str(args.repeats), '--output', str(result_path)]
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', VLLM_USE_V2_MODEL_RUNNER='1',
                       VLLM_ENABLE_V1_MULTIPROCESSING='0')
            try:
                before_pids = compute_pids(case['tp'])
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                return {'exit_code': 127, 'seconds': 0, 'command': cmd, 'log': None,
                        'result': None, 'error': f'GPU process query failed: {exc}'}
            if before_pids:
                return {'exit_code': 126, 'seconds': 0, 'command': cmd, 'log': None,
                        'result': None, 'baseline_gpu_pids': sorted(before_pids),
                        'error': 'selected GPUs already have compute processes'}
            result_path.unlink(missing_ok=True)
            started = time.monotonic()
            with log_path.open('w') as out:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
                try: code = proc.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try: proc.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                    code = 124
            try:
                result = json.loads(result_path.read_text()) if code == 0 and result_path.is_file() else None
            except (OSError, json.JSONDecodeError):
                result = None
            if impl == 'frozen' and result is not None:
                result['upstream_log_times_by_rank'] = upstream_log_times(log_path.read_text(errors='replace'), case['tp'])
            try:
                leaked_pids = sorted(compute_pids(case['tp']) - before_pids)
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                return {'exit_code': 127, 'seconds': time.monotonic()-started,
                        'command': cmd, 'log': str(log_path), 'result': result,
                        'error': f'GPU cleanup query failed: {exc}'}
            if leaked_pids:
                code = 125
            return {'exit_code': code, 'seconds': time.monotonic()-started,
                    'command': cmd, 'log': str(log_path), 'result': result,
                    'baseline_gpu_pids': sorted(before_pids), 'new_gpu_pids_after_exit': leaked_pids}
        def compare(other, register):
            reference = folder / f'{case_tag}_frozen_gate_ref'
            comparison = folder / f'{case_tag}_{other}_comparison.json'
            contract = folder / f'{case_tag}_contract.json'
            script = 'step109_compare.py' if case['k'] else 'step111_compare_gate.py'
            cmd = [sys.executable, str(ROOT/'benchmarks'/script), '--reference', str(reference),
                   '--other', str(folder/f'{case_tag}_{other}'), '--contract', str(contract), '--output', str(comparison)]
            if not case['k'] and 1024 in case['lengths']:
                cmd.extend(['--calibration',str(args.calibration)])
            if register: cmd.append('--register')
            comparison.unlink(missing_ok=True)
            started = time.monotonic()
            try:
                completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=args.timeout)
                code, output = completed.returncode, completed.stdout + completed.stderr
            except subprocess.TimeoutExpired:
                code, output = 124, 'comparison timed out'
            relative_contract = folder/f'{case_tag}_relative_contract.json'
            if code == 0 and case['k']:
                values = json.loads(comparison.read_text())['values']
                if register:
                    thresholds = {name:max(1e-5, 2*value['relative_l2']) for name,value in values.items()}
                    _atomic_write_json(relative_contract, {'thresholds':thresholds})
                else:
                    thresholds = json.loads(relative_contract.read_text())['thresholds']
                    failures = [name for name,value in values.items()
                                if name not in thresholds or value['relative_l2'] > thresholds[name]]
                    if failures or values.keys() != thresholds.keys():
                        code = 1
                        output += '\nrelative L2 contract failed: ' + repr(failures[:10])
            rank_command = None
            if code == 0 and case['k']:
                rank_command = [sys.executable, str(ROOT/'benchmarks/step111_rank_gate.py'),
                                '--reference', str(reference), '--other', str(folder/f'{case_tag}_{other}'),
                                '--tp', str(case['tp']), '--output', str(folder/f'{case_tag}_{other}_ranks.json')]
                try:
                    rank_run = subprocess.run(rank_command, cwd=ROOT, text=True, capture_output=True,
                                              timeout=args.timeout)
                    code, output = rank_run.returncode, output + rank_run.stdout + rank_run.stderr
                except subprocess.TimeoutExpired:
                    code, output = 124, output + '\nrank comparison timed out'
            return {'exit_code':code, 'command':cmd, 'seconds':time.monotonic()-started,
                    'output':output[-8000:], 'report':str(comparison), 'contract':str(contract),
                    'relative_contract':str(relative_contract) if case['k'] else None,
                    'rank_command':rank_command}
        def diagnose_neartie(stage):
            """Probe without timing hooks and verify every first divergence."""
            if case['k']:
                return {'passed':False, 'error':'EAGLE3 near-tie diagnostics not implemented'}
            outputs = {}
            for impl in ('frozen','mini'):
                result_path = folder/f'{case_tag}_{stage}_{impl}_neartie.json'
                log_path = folder/f'{case_tag}_{stage}_{impl}_neartie.log'
                command = [sys.executable,str(ROOT/'benchmarks/step111_firstdiff_probe.py'),
                           '--implementation',impl,'--gate-report',str(args.output),'--case-id',case_id,
                           '--output-tokens','128' if stage=='performance' else '32','--output',str(result_path)]
                env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', VLLM_USE_V2_MODEL_RUNNER='1',
                           VLLM_ENABLE_V1_MULTIPROCESSING='0')
                try: before = compute_pids(case['tp'])
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    return {'passed':False,'error':f'GPU process query failed: {exc}'}
                if before: return {'passed':False,'error':f'selected GPUs occupied: {sorted(before)}'}
                result_path.unlink(missing_ok=True)
                with log_path.open('w') as stream:
                    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                    try: code = proc.wait(timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try: proc.wait(timeout=10)
                        except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL);proc.wait()
                        code = 124
                try: leaked = sorted(compute_pids(case['tp']) - before)
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    return {'passed':False,'error':f'GPU cleanup query failed: {exc}'}
                outputs[impl] = {'exit_code':code,'command':command,'log':str(log_path),
                                 'path':str(result_path),'new_gpu_pids_after_exit':leaked}
                if code or leaked or not result_path.is_file():
                    return {'passed':False,'error':f'{impl} diagnostic failed','runs':outputs}
            report = folder/f'{case_tag}_{stage}_neartie_verified.json'
            report.unlink(missing_ok=True)
            command = [sys.executable,str(ROOT/'benchmarks/step111_neartie.py'),
                       '--gate-report',str(args.output),'--case-id',case_id,'--stage',stage,
                       '--reference',outputs['frozen']['path'],'--mini',outputs['mini']['path'],'--output',str(report)]
            try:
                done = subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=args.timeout)
                code, output = done.returncode, done.stdout + done.stderr
            except subprocess.TimeoutExpired:
                code, output = 124, 'verification timed out'
            return {'passed':code==0 and report.is_file(), 'exit_code':code,'command':command,
                    'output':output[-8000:],'report':str(report),'runs':outputs}
        for suffix in ('_ref', '_repeat'):
            entry['gate']['frozen'+suffix] = run('frozen', 'gate', suffix); save()
        refs_ok = all(v['exit_code'] == 0 and v['result'] is not None for v in entry['gate'].values())
        if refs_ok:
            try:
                for label in ('frozen_ref', 'frozen_repeat'):
                    validate_measurement_metadata(entry['gate'][label]['result'], case, gate=True, implementation='frozen')
                left = rank_devices(entry['gate']['frozen_ref']['result'], case['tp'])
                right = rank_devices(entry['gate']['frozen_repeat']['result'], case['tp'])
                refs_ok = left == right
            except (KeyError, ValueError) as exc:
                refs_ok = False; entry['gate']['device_error'] = str(exc)
        if refs_ok:
            refs_ok = (entry['gate']['frozen_ref']['result']['schedule_gate'] ==
                       entry['gate']['frozen_repeat']['result']['schedule_gate'])
            entry['gate']['reference_schedule_equal'] = refs_ok
        if refs_ok:
            entry['gate']['registration'] = compare('frozen_gate_repeat', True); save()
            refs_ok = entry['gate']['registration']['exit_code'] == 0
        if refs_ok and case['k']:
            reference = folder/f'{case_tag}_frozen_gate_ref'
            files = [reference/'summary.json', folder/f'{case_tag}_contract.json',
                     folder/f'{case_tag}_relative_contract.json',
                     *(reference/f'state_rank{rank}.pt' for rank in range(case['tp']))]
            entry['gate']['reference_hashes'] = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
            save()
        if not refs_ok:
            entry['status'] = 'failed_reference_gate'; save()
            print(json.dumps({'case':case_id, 'status':entry['status']}), flush=True); continue
        entry['gate']['mini'] = run('mini', 'gate'); save()
        gate_ok = entry['gate']['mini']['exit_code'] == 0 and entry['gate']['mini']['result'] is not None
        if gate_ok:
            try:
                validate_measurement_metadata(entry['gate']['mini']['result'], case, gate=True, implementation='mini')
                gate_ok = (rank_devices(entry['gate']['frozen_ref']['result'], case['tp']) ==
                           rank_devices(entry['gate']['mini']['result'], case['tp']))
            except (KeyError, ValueError) as exc:
                gate_ok = False; entry['gate']['device_error'] = str(exc)
        if gate_ok:
            schedule = compare_schedule(entry['gate']['frozen_ref']['result']['schedule_gate'],
                                        entry['gate']['mini']['result']['schedule_gate'])
            entry['gate']['schedule_comparison'] = schedule
            gate_ok = schedule['passed']
        if gate_ok and case['k']:
            gate_ok = all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
                          for path, expected in entry['gate']['reference_hashes'].items())
        if gate_ok:
            entry['gate']['comparison'] = compare('mini_gate', False); save()
            gate_ok = entry['gate']['comparison']['exit_code'] == 0
            if not gate_ok and not case['k']:
                path = Path(entry['gate']['comparison']['report'])
                report = json.loads(path.read_text()) if path.is_file() else {}
                if report.get('first_diff') == 'tokens':
                    entry['gate']['near_tie'] = diagnose_neartie('gate'); save()
                    gate_ok = entry['gate']['near_tie']['passed']
        entry['state_gate'] = ('passed_near_tie' if gate_ok and entry['gate'].get('near_tie')
                               else 'passed' if gate_ok else 'failed')
        if not gate_ok:
            entry['status'] = 'failed_gate'; save()
            print(json.dumps({'case':case_id, 'status':entry['status']}), flush=True); continue
        for impl in order:
            entry['engines'][impl] = run(impl, 'performance'); save()
        ok = all(v['exit_code'] == 0 and v['result'] is not None for v in entry['engines'].values())
        if ok:
            try:
                expected_devices = rank_devices(entry['gate']['frozen_ref']['result'], case['tp'])
                for impl, value in entry['engines'].items():
                    validate_measurement_metadata(value['result'], case, gate=False, implementation=impl)
                ok = all(rank_devices(value['result'], case['tp']) == expected_devices
                         for value in entry['engines'].values())
            except (KeyError, ValueError) as exc:
                ok = False; entry['device_error'] = str(exc)
        if ok:
            for value in entry['engines'].values():
                value['summary'] = summarize(value['result']['samples'])
                values = [sample['output_tokens_per_s'] / case['tp']
                          for sample in value['result']['samples']]
                value['summary']['output_tokens_per_s_per_gpu'] = {
                    'values': values, 'median': statistics.median(values),
                    'min': min(values), 'max': max(values)}
            repeat_divergences = []
            for impl, value in entry['engines'].items():
                baseline = value['result']['samples'][0]['tokens']
                for repeat, sample in enumerate(value['result']['samples']):
                    difference = token_difference(baseline, sample['tokens'])
                    if difference is not None:
                        repeat_divergences.append({'implementation':impl, 'repeat':repeat, **difference})
            entry['repeat_divergences'] = repeat_divergences
            if repeat_divergences:
                entry['status'] = 'failed_repeat_nondeterminism'
                save(); print(json.dumps({'case':case_id, 'status':entry['status']}), flush=True); continue
            frozen = entry['engines']['frozen']['result']['samples'][0]['tokens']
            mini = entry['engines']['mini']['result']['samples'][0]['tokens']
            difference = token_difference(frozen, mini)
            entry['first_output_divergence'] = difference
            entry['strict_equivalent'] = difference is None and not entry['gate'].get('near_tie')
            if difference is not None:
                save()
                entry['performance_near_tie'] = diagnose_neartie('performance'); save()
                if not entry['performance_near_tie']['passed']:
                    entry['status'] = 'failed_output_divergence'
                    save(); print(json.dumps({'case':case_id, 'status':entry['status']}), flush=True); continue
        entry['status'] = ('passed_performance' if entry.get('strict_equivalent') else 'observed_near_tie') if ok else 'failed_performance'
        if ok:
            try: entry['artifact_hashes'] = artifact_hashes(entry,case)
            except OSError as exc:
                entry['status'] = 'failed_artifact_audit'; entry['artifact_error'] = str(exc)
        save()
        print(json.dumps({'case':case_id, 'status':entry['status']}), flush=True)
    ok = len(payload['results']) == len(cases) and all(
        item['status'] in ('passed_performance', 'observed_near_tie') for item in payload['results'])
    payload['status'] = 'passed_selected_performance_cases' if ok else 'failed_selected_performance_cases'
    payload['selected_case_count'] = len(cases)
    payload['full_matrix_case_count'] = len(build_cases())
    payload['case_counts'] = {name:sum(item['status']==name for item in payload['results'])
                              for name in ('passed_performance','observed_near_tie','failed_reference_gate',
                                           'failed_gate','failed_performance','failed_repeat_nondeterminism',
                                           'failed_output_divergence','failed_artifact_audit','unrun')}
    save()
    return int(not ok)

if __name__ == '__main__':
    raise SystemExit(main())
