"""Run fixed-128-token real EAGLE3 TP cases and frozen-repeat comparisons."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT/'benchmarks/step109_real_probe.py'
COMPARE = ROOT/'benchmarks/step109_compare.py'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['all', '8b', '14b'], default='all')
    p.add_argument('--tp', type=int, choices=[1, 2, 4])
    p.add_argument('--batch', type=int, choices=[1, 4])
    p.add_argument('--mode', choices=['eager', 'graph'])
    p.add_argument('--k', type=int, choices=[1, 2, 4])
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--gpu-memory-utilization', type=float, default=.60)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.timeout <= 0: p.error('timeout must be positive')
    if not 0 < a.gpu_memory_utilization < 1: p.error('gpu-memory-utilization must be in (0, 1)')
    cases = []
    for model in ('8b', '14b'):
        for tp in ((1, 2, 4) if model == '8b' else (2, 4)):
            for batch in (1, 4):
                for mode in ('eager', 'graph'):
                    cases.append((model, tp, batch, 2, mode))
        for k in (1, 4): cases.append((model, 2, 1, k, 'eager'))
    cases = [case for case in cases if (a.model == 'all' or case[0] == a.model)
             and (a.tp is None or case[1] == a.tp) and (a.batch is None or case[2] == a.batch)
             and (a.k is None or case[3] == a.k) and (a.mode is None or case[4] == a.mode)]
    if not cases: p.error('selection contains no required case')
    output = a.output.resolve(); folder = output.parent/(output.stem + '_runs');folder.mkdir(parents=True,exist_ok=True)
    executions = []
    def run(name, cmd, frozen=False):
        log = folder/f'{name}.log'; env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        if frozen: env['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
        start = time.monotonic()
        with log.open('w') as stream:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try: code = proc.wait(timeout=a.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL);proc.wait()
                code = 124
        record = {'name':name, 'command':cmd, 'frozen_engine_core_in_process':frozen,
                  'exit_code':code, 'elapsed_seconds':time.monotonic()-start, 'log':str(log)}
        executions.append(record)
        output.write_text(json.dumps({'status':'running', 'cases':cases, 'executions':executions},indent=2)+'\n')
        print(json.dumps(record),flush=True)
        if code: raise RuntimeError(f'{name} failed ({code}); {log}')
    try:
        for model,tp,batch,k,mode in cases:
            name=f'{model}_tp{tp}_b{batch}_k{k}_{mode}'
            ref,repeat,mini = (folder/f'{name}_{suffix}' for suffix in ('ref','repeat','mini'))
            common=['--model',model,'--tp',str(tp),'--batch',str(batch),'--k',str(k),'--mode',mode,
                    '--gpu-memory-utilization',str(a.gpu_memory_utilization)]
            for label,directory in (('ref',ref),('repeat',repeat)):
                run(name+'_'+label,[sys.executable,str(PROBE),'--implementation','frozen',*common,'--output',str(directory)],True)
            contract=folder/f'{name}_contract.json'
            run(name+'_register',[sys.executable,str(COMPARE),'--reference',str(ref),'--other',str(repeat),
                                 '--contract',str(contract),'--output',str(folder/f'{name}_repeat.json'),'--register'])
            run(name+'_mini',[sys.executable,str(PROBE),'--implementation','mini',*common,'--output',str(mini)])
            run(name+'_compare',[sys.executable,str(COMPARE),'--reference',str(ref),'--other',str(mini),
                                '--contract',str(contract),'--output',str(folder/f'{name}_comparison.json')])
        status='passed'
    except RuntimeError as exc:
        print(str(exc),file=sys.stderr);status='failed'
    output.write_text(json.dumps({'status':status,'cases':cases,'executions':executions},indent=2)+'\n')
    print(json.dumps({'status':status,'cases':len(cases),'executions':len(executions)}),flush=True)
    return int(status != 'passed')

if __name__ == '__main__': raise SystemExit(main())
