"""使用 Nsight Systems 采集独立于正式计时的 TP 诊断 trace。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.step111_matrix import compute_pids
from benchmarks.step88_common import package_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=('8b', '14b'), required=True)
    p.add_argument('--tp', type=int, choices=(1, 2, 4), required=True)
    p.add_argument('--k', type=int, choices=(0, 2), default=0)
    p.add_argument('--mode', choices=('eager', 'graph'), default='eager')
    p.add_argument('--implementation', choices=('mini', 'frozen'), default='mini')
    p.add_argument('--gate-report', type=Path, required=True,
                   help='performance matrix JSON with a passing matching correctness gate')
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.timeout <= 0 or (a.model == '14b' and a.tp == 1): p.error('invalid timeout or model/TP combination')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    case = {'model':a.model, 'tp':a.tp, 'batch':1, 'lengths':[128], 'k':a.k,
            'mode':a.mode, 'output_tokens':128, 'groups':['trace']}
    gate = json.loads(a.gate_report.read_text())
    def selected(value):
        return value.split(',')[:a.tp] if value else None
    gate_visible = selected(gate.get('cuda_visible_devices'))
    if gate_visible != selected(os.environ.get('CUDA_VISIBLE_DEVICES')):
        p.error('gate report and trace must use the same selected GPU set')
    def valid(row):
        if row.get('status') == 'passed_performance' and row.get('strict_equivalent') is True:
            return True
        if row.get('status') == 'observed_near_tie' and row.get('strict_equivalent') is False:
            return bool(row.get('performance_near_tie', {}).get('passed') or
                        row.get('gate', {}).get('near_tie', {}).get('passed'))
        return False
    matching = [row for row in gate.get('results', []) if
                all(row['case'].get(key) == case[key] for key in ('model','tp','batch','lengths','k','mode'))
                and valid(row)]
    if len(matching) != 1:
        p.error('matching case did not pass correctness and performance gate')
    if gate.get('package', {}).get('sha256') != package_hash()['sha256']:
        p.error('gate report package differs from current production package')
    stem = a.output.with_suffix('')
    child = stem.with_name(stem.name + '_worker.json')
    cmd = ['nsys', 'profile', '--trace=cuda,cublas,nvtx', '--cuda-graph-trace=node',
           '--sample=none', '--cpuctxsw=none', '--force-overwrite=true', '--output', str(stem),
           sys.executable, str(ROOT/'benchmarks/step111_parallel_worker.py'), '--case', json.dumps(case),
           '--implementation', a.implementation, '--phase', 'performance', '--repeats', '1', '--output', str(child)]
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', VLLM_USE_V2_MODEL_RUNNER='1', STEP111_TRACE='1',
               VLLM_ENABLE_V1_MULTIPROCESSING='0')
    before_pids = compute_pids(a.tp)
    if before_pids:
        result = {'status':'failed', 'error':'selected GPUs already have compute processes',
                  'baseline_gpu_pids':sorted(before_pids)}
        a.output.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result)); return 1
    log = stem.with_suffix('.log'); start = time.monotonic()
    with log.open('w') as f:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try: code = proc.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            code = 124
    report = stem.with_suffix('.nsys-rep')
    leaked_pids = sorted(compute_pids(a.tp) - before_pids)
    sqlite = stem.with_suffix('.sqlite')
    analysis = stem.with_name(stem.name + '_analysis.json')
    export_code = analysis_code = None
    if code == 0 and report.is_file() and not leaked_pids:
        export = ['nsys', 'export', '--type=sqlite', '--force-overwrite=true', '--output', str(sqlite), str(report)]
        try:
            export_result = subprocess.run(export, cwd=ROOT, capture_output=True, text=True, timeout=a.timeout)
            export_code = export_result.returncode
            with log.open('a') as f: f.write(export_result.stdout + export_result.stderr)
        except subprocess.TimeoutExpired:
            export_code = 124
        if export_code == 0:
            analysis.unlink(missing_ok=True)
            analyze = [sys.executable, str(ROOT/'benchmarks/step111_analyze_trace.py'),
                       '--sqlite',str(sqlite),'--tp',str(a.tp),'--worker-json',str(child),'--output',str(analysis)]
            try:
                analyzed = subprocess.run(analyze, cwd=ROOT, capture_output=True, text=True, timeout=a.timeout)
                analysis_code = analyzed.returncode
                with log.open('a') as f: f.write(analyzed.stdout + analyzed.stderr)
            except subprocess.TimeoutExpired:
                analysis_code = 124
    output = {'status': 'captured' if code == 0 and export_code == 0 and analysis_code == 0 and not leaked_pids else 'failed', 'exit_code':code,
              'command':cmd, 'elapsed_seconds':time.monotonic()-start, 'trace':str(report), 'log':str(log),
              'export_exit_code':export_code, 'analysis_exit_code':analysis_code,
              'sqlite':str(sqlite), 'analysis':str(analysis),
              'gate_report':str(a.gate_report), 'gate_report_sha256':hashlib.sha256(a.gate_report.read_bytes()).hexdigest(),
              'new_gpu_pids_after_exit':leaked_pids,
              'harness_sha256':{name:hashlib.sha256((ROOT/'benchmarks'/name).read_bytes()).hexdigest()
                                for name in ('trace_step111_parallel.py','step111_parallel_worker.py','step111_analyze_trace.py')},
              'worker_result':str(child), 'warning':'trace includes initialization and warmup; use rank NVTX ranges for steady work and do not sum asynchronous inclusive times as wall time'}
    a.output.write_text(json.dumps(output, indent=2)+'\n'); print(json.dumps(output))
    return int(output['status'] != 'captured')

if __name__ == '__main__': raise SystemExit(main())
