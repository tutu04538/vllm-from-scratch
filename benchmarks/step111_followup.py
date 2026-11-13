"""串行采集 111 的差异诊断、代表点 trace 与 NUMA 对照。"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'benchmarks/results/step111/followup_v3'
REPORT = OUT / 'results.json'
PYTHON = sys.executable
MATRIX = ROOT / 'benchmarks/results/step111/full_matrix.json'
results = []
env_base = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                VLLM_USE_V2_MODEL_RUNNER='1', VLLM_ENABLE_V1_MULTIPROCESSING='0')
current_process = None


def terminate_tree(process):
    try: children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess: children = []
    for child in reversed(children):
        try: child.terminate()
        except psutil.NoSuchProcess: pass
    try: process.terminate()
    except ProcessLookupError: pass
    _, alive = psutil.wait_procs(children, timeout=10)
    for child in alive:
        try: child.kill()
        except psutil.NoSuchProcess: pass
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired: process.kill(); process.wait()


def stop(signum, _frame):
    if current_process is not None: terminate_tree(current_process)
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)


def run(name, command, devices='2,3,0,1', timeout=1800):
    global current_process
    log = OUT / f'{name}.log'
    env = dict(env_base, CUDA_VISIBLE_DEVICES=devices)
    print(json.dumps({'job':name,'status':'running'}), flush=True)
    start = time.monotonic()
    with log.open('w') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        current_process = process
        try: code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_tree(process)
            code = 124
        finally:
            current_process = None
    item = {'name':name,'command':command,'cuda_visible_devices':devices,'exit_code':code,
            'elapsed_seconds':time.monotonic()-start,'log':str(log)}
    results.append(item)
    REPORT.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps({'job':name,'exit_code':code}), flush=True)
    return code


def script(name): return str(ROOT/'benchmarks'/name)


def main():
    global MATRIX, OUT, REPORT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',type=Path,default=MATRIX)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--trace-only',action='store_true')
    args=parser.parse_args()
    MATRIX,OUT=args.matrix,args.output_dir
    OUT.mkdir(parents=True,exist_ok=True)
    REPORT=OUT/'results.json'
    if REPORT.exists(): parser.error('output directory already contains results; choose a new directory')
    if not args.trace_only:
        rows = json.loads(MATRIX.read_text())['results']
        for case_id, request, position in [('8b_tp4_b16_n1024_k0_eager','12',1023),
                                          ('14b_tp2_b16_n1024_k0_eager','15',1023)]:
            case = next(row['case'] for row in rows if row['case_id'] == case_id)
            for implementation in ('frozen','mini'):
                output = OUT/f'{case_id}_{implementation}.json'
                run(f'{case_id}_{implementation}', [PYTHON,script('step111_mixed_diag_probe.py'),
                    '--case',json.dumps(case),'--implementation',implementation,'--phase','gate','--repeats','1',
                    '--diag-request',request,'--diag-positions',str(position),'--output',str(output)])
            a = str((OUT/f'{case_id}_frozen').with_name(f'{case_id}_frozen_layers'))
            b = str((OUT/f'{case_id}_mini').with_name(f'{case_id}_mini_layers'))
            run(f'{case_id}_compare',[PYTHON,script('step111_mixed_diag_compare.py'),
                '--reference',a,'--mini',b,'--position',str(position),'--tp',str(case['tp']),
                '--output',str(OUT/f'{case_id}_layers_comparison.json')])
        case_id = '14b_tp4_b16_n1024_k0_eager'
        base = MATRIX.with_name(MATRIX.stem + '_runs')
        a=json.loads((base/f'{case_id}_performance_frozen_neartie.json').read_text())
        b=json.loads((base/f'{case_id}_performance_mini_neartie.json').read_text())
        targets={}
        for request in a['tokens']:
            for i,(left,right) in enumerate(zip(a['tokens'][request],b['tokens'][request])):
                if left != right:
                    targets[request]=i
                    break
        for implementation in ('frozen','mini'):
            run(f'{case_id}_full_logits_{implementation}',[PYTHON,script('step111_full_logits_probe.py'),
                '--implementation',implementation,'--gate-report',str(MATRIX),'--case-id',case_id,
                '--output-tokens','128','--targets',json.dumps(targets),
                '--output',str(OUT/f'{case_id}_full_logits_{implementation}.json')])
    for model,tp,k,mode in [('8b',1,0,'eager'),('8b',2,0,'eager'),('8b',4,0,'eager'),
                            ('14b',2,0,'eager'),('14b',4,0,'eager'),('14b',2,2,'graph')]:
        name=f'trace_{model}_tp{tp}_k{k}_{mode}'
        run(name,[PYTHON,script('trace_step111_parallel.py'),'--model',model,'--tp',str(tp),
                  '--k',str(k),'--mode',mode,'--gate-report',str(MATRIX),'--output',str(OUT/f'{name}.json')])
    for devices,suffix in [('0,1','same_numa'),('0,2','cross_numa')]:
        gate=OUT/f'numa_{suffix}_gate.json'
        code=run(f'numa_{suffix}_gate',[PYTHON,script('compare_step111_parallel.py'),
            '--model','14b','--tp','2','--batch','1','--group','A','--mode','eager','--k','0',
            '--input-length','128','--repeats','3','--output',str(gate)],devices=devices)
        if code == 0:
            run(f'numa_{suffix}_trace',[PYTHON,script('trace_step111_parallel.py'),
                '--model','14b','--tp','2','--k','0','--mode','eager','--gate-report',str(gate),
                '--output',str(OUT/f'numa_{suffix}_trace.json')],devices=devices)
    return int(any(item['exit_code'] for item in results))


if __name__ == '__main__': raise SystemExit(main())
