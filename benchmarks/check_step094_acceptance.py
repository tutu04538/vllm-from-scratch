#!/usr/bin/env python3
"""094 共享内存广播、响应顺序和多进程 RPC 验收（非模型性能基准）。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SCENARIOS=ROOT/'tests/step094/step094_rpc_scenarios.py'
QUEUE_TEST=ROOT/'tests/step094/test_step094_queue.py'


def execute(case, output, timeout):
    output.parent.mkdir(parents=True,exist_ok=True)
    cmd=([sys.executable,'-m','pytest',str(QUEUE_TEST),'-q','-ra'] if case=='queue' else
         [sys.executable,str(SCENARIOS),'--case',case,'--output',str(output)])
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    log=output.with_suffix('.log'); start=time.monotonic()
    with log.open('w') as f:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try: code=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            code=124
    if case=='queue':
        result={'status':'passed' if code==0 else 'failed','pytest_output':log.read_text()[-5000:]}
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    elif output.exists():result=json.loads(output.read_text())
    else:result={'status':'failed','error':'子进程未写结果'}
    return {'case':case,'command':cmd,'exit_code':code,'passed':code==0 and result['status']=='passed',
            'elapsed_seconds':time.monotonic()-start,'result_path':str(output),'log_path':str(log),'result':result}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['all','rpc','failure','engine','queue'],default='all')
    p.add_argument('--timeout',type=int,default=180);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.timeout<=0:p.error('--timeout must be positive')
    cases=['rpc','failure','engine','queue'] if args.case=='all' else [args.case]
    root=args.output.parent/(args.output.stem+'_cases');root.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    result={'status':'failed','cases':[], 'environment':{'python':sys.version.split()[0],
           'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}}
    for case in cases:
        result['cases'].append(execute(case,root/f'{case}.json',args.timeout))
    manifest=ROOT/'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/源码清单.json'
    result['source_manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
    result['elapsed_seconds']=time.monotonic()-start
    result['status']='passed' if all(c['passed'] for c in result['cases']) else 'failed'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'cases':len(result['cases']),'output':str(args.output)},ensure_ascii=False))
    return 0 if result['status']=='passed' else 1

if __name__=='__main__':sys.exit(main())
