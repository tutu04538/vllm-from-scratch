"""串行执行 111 自测及旧单卡/已完成并行关卡回归。"""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/results/step111/regressions'
OUT.mkdir(parents=True,exist_ok=True)
groups=[('step111',['tests/step111'],'2,3,0,1'),
        ('legacy',[f'tests/step{x}' for x in range(58,89)],'0'),
        ('parallel',[f'tests/step{x:03d}' for x in range(89,111) if (ROOT/f'tests/step{x:03d}').is_dir()],'2,3,0,1')]
results=[]
for name,paths,devices in groups:
    cmd=[sys.executable,'-m','pytest',*paths,'-q','-ra']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    log=OUT/f'{name}.log';start=time.monotonic()
    print(json.dumps({'group':name,'status':'running'}),flush=True)
    with log.open('w') as stream:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=7200)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            code=124
    summaries=[line for line in log.read_text(errors='replace').splitlines() if re.search(r'\b\d+ passed\b',line)]
    item={'group':name,'command':cmd,'cuda_visible_devices':devices,'exit_code':code,
          'elapsed_seconds':time.monotonic()-start,'log':str(log),'summary':summaries[-1] if summaries else None}
    results.append(item);(OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(item),flush=True)
raise SystemExit(int(any(r['exit_code'] or not r['summary'] for r in results)))
